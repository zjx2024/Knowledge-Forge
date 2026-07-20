"""Fail-closed HugeGraph mutation ownership and isolated live-test guards."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from typing import Any, Callable
from uuid import UUID


class GraphSafetyError(RuntimeError):
    """A graph mutation or destructive test could not prove its ownership boundary."""


def validate_namespace_id(value: str) -> str:
    try:
        return str(UUID(str(value)))
    except (TypeError, ValueError, AttributeError) as exc:
        raise GraphSafetyError("graph namespace must be a non-empty UUID") from exc


@dataclass(frozen=True)
class GraphEndpointIdentity:
    scheme: str
    host: str
    port: int
    graph: str

    def canonical(self) -> str:
        if self.scheme not in {"http", "https"} or not self.host or not self.graph:
            raise GraphSafetyError("HugeGraph physical identity is incomplete")
        return f"{self.scheme}://{self.host}:{int(self.port)}/{self.graph}"


@dataclass(frozen=True)
class GraphMutationContext:
    namespace_id: str
    project_id: int
    generation: int
    policy_hash: str
    lease_token: str

    def validate_for(self, project: dict[str, Any]) -> None:
        namespace = validate_namespace_id(self.namespace_id)
        if namespace != validate_namespace_id(str(project.get("graph_namespace_id") or "")):
            raise GraphSafetyError("graph mutation namespace does not own this project")
        if int(self.project_id) != int(project.get("id") or 0):
            raise GraphSafetyError("graph mutation project does not match")
        if int(self.generation) != int(project.get("graph_generation") or 0):
            raise GraphSafetyError("graph mutation generation is stale")
        if not self.policy_hash or not self.lease_token:
            raise GraphSafetyError("graph mutation requires policy and PostgreSQL lease token")


@dataclass(frozen=True)
class HugeGraphTestGuard:
    identity: GraphEndpointIdentity
    namespace_id: str
    run_id: str
    destructive: bool
    config_hash: str
    expires_at: str

    @classmethod
    def preflight(
        cls, *, identity: GraphEndpointIdentity,
        production_identity: GraphEndpointIdentity | None, namespace_id: str,
        run_id: str, integration_enabled: bool, destructive_enabled: bool,
        endpoint_payload: dict[str, Any], destructive: bool = True,
        mutation_transport: Callable[..., Any] | None = None,
    ) -> "HugeGraphTestGuard":
        """Perform read-only validation. ``mutation_transport`` is never called here."""

        del mutation_transport
        namespace = validate_namespace_id(namespace_id)
        physical = identity.canonical()
        if not integration_enabled:
            raise GraphSafetyError("RUN_HUGEGRAPH_INTEGRATION=1 is required")
        if destructive and not destructive_enabled:
            raise GraphSafetyError("RUN_HUGEGRAPH_DESTRUCTIVE_TESTS=1 is required")
        if destructive:
            if identity.graph == "knowledge_forge":
                raise GraphSafetyError("the default production graph is forbidden for destructive tests")
            if production_identity is None:
                raise GraphSafetyError("production physical graph identity must be configured")
            if physical == production_identity.canonical():
                raise GraphSafetyError("destructive tests require a different physical graph")
        if not run_id or not endpoint_payload.get("version"):
            raise GraphSafetyError("HugeGraph identity/version endpoint was not verified")
        if str(endpoint_payload.get("graph") or "") != identity.graph:
            raise GraphSafetyError("HugeGraph endpoint graph identity mismatch")
        config = {
            "physical": physical, "namespace": namespace, "run_id": run_id,
            "destructive": destructive,
        }
        return cls(
            identity=identity, namespace_id=namespace, run_id=run_id,
            destructive=destructive,
            config_hash=hashlib.sha256(
                json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            expires_at=(datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat(),
        )

    def sentinel(self) -> dict[str, Any]:
        return {
            "purpose": "test", "run_id": self.run_id,
            "namespace_id": self.namespace_id,
            "physical_identity": self.identity.canonical(),
            "config_hash": self.config_hash, "expires_at": self.expires_at,
        }

    def verify_sentinel(self, value: dict[str, Any]) -> bool:
        try:
            expiry = datetime.fromisoformat(str(value["expires_at"]))
        except (KeyError, TypeError, ValueError):
            return False
        if expiry.tzinfo is None:
            return False
        return bool(
            value.get("purpose") == "test"
            and value.get("run_id") == self.run_id
            and value.get("namespace_id") == self.namespace_id
            and value.get("physical_identity") == self.identity.canonical()
            and value.get("config_hash") == self.config_hash
            and expiry > datetime.now(timezone.utc)
        )

    def arm(
        self, *, read_counts: Callable[[], tuple[int, int]],
        write_sentinel: Callable[[dict[str, Any]], None],
        read_sentinel: Callable[[], dict[str, Any] | None],
    ) -> str:
        """Arm an empty disposable graph and return a short-lived bound token."""

        if not self.destructive:
            raise GraphSafetyError("only a destructive guard can be armed")
        vertices, edges = read_counts()
        if int(vertices) != 0 or int(edges) != 0:
            raise GraphSafetyError("destructive graph must be completely empty before arm")
        sentinel = self.sentinel()
        write_sentinel(sentinel)
        if not self.verify_sentinel(read_sentinel() or {}):
            raise GraphSafetyError("HugeGraph sentinel could not be verified after arm")
        return hashlib.sha256(
            f"{self.config_hash}:{self.run_id}:{self.expires_at}".encode("utf-8")
        ).hexdigest()

    def authorize(
        self, token: str, *, read_sentinel: Callable[[], dict[str, Any] | None],
    ) -> None:
        """Re-read ownership evidence before every execute or cleanup phase."""

        expected = hashlib.sha256(
            f"{self.config_hash}:{self.run_id}:{self.expires_at}".encode("utf-8")
        ).hexdigest()
        if token != expected or not self.verify_sentinel(read_sentinel() or {}):
            raise GraphSafetyError("HugeGraph guard token or sentinel is stale")

    def cleanup(
        self, token: str, *, read_sentinel: Callable[[], dict[str, Any] | None],
        delete_namespace: Callable[[str], None],
    ) -> None:
        """Delete only the namespace proven by the still-valid one-time sentinel."""

        self.authorize(token, read_sentinel=read_sentinel)
        delete_namespace(self.namespace_id)
