"""Fail-closed runtime validation for network exposure and credentials."""

from __future__ import annotations

import ipaddress
from typing import Any


class RuntimeSafetyError(RuntimeError):
    """The service configuration would expose an unsafe runtime."""


_WEAK_SECRETS = {"", "admin", "password", "postgres", "change-me", "changeme", "secret"}


def _is_loopback(host: str) -> bool:
    normalized = str(host or "").strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _validate_secret(name: str, value: Any, *, optional: bool = False) -> None:
    normalized = str(value or "").strip()
    if optional and not normalized:
        return
    if normalized.lower() in _WEAK_SECRETS or len(normalized) < 12:
        raise RuntimeSafetyError(f"{name} 缺失或仍为弱默认值")


def validate_runtime_config(config: Any) -> list[str]:
    """Validate one runtime config and return explicit development warnings."""

    warnings: list[str] = []
    postgres = getattr(config, "postgres", {}) or {}
    _validate_secret("POSTGRES_PASSWORD", postgres.get("password"))

    graph_user = str(getattr(config, "hugegraph_user", "") or "").strip()
    graph_password = str(getattr(config, "hugegraph_password", "") or "").strip()
    graph_auth_enabled = bool(getattr(config, "hugegraph_auth_enabled", False))
    if graph_auth_enabled or graph_user or graph_password:
        _validate_secret("HUGEGRAPH_PASSWORD", graph_password)

    host = str(getattr(config, "upload_host", "127.0.0.1") or "").strip()
    if _is_loopback(host):
        return warnings

    mode = str(getattr(config, "runtime_mode", "development") or "development").lower()
    auth_enabled = bool(getattr(config, "authentication_enabled", False))
    approved = bool(getattr(config, "public_bind_approved", False))
    unsafe_dev = bool(getattr(config, "allow_unsafe_public_bind", False))
    if mode in {"production", "shared", "uat"}:
        if not auth_enabled:
            raise RuntimeSafetyError("共享或生产网络监听必须启用有效认证")
        if not approved:
            raise RuntimeSafetyError("共享或生产网络监听缺少 PUBLIC_BIND_APPROVED")
        return warnings
    if not unsafe_dev:
        raise RuntimeSafetyError(
            "非回环监听需要显式设置 ALLOW_UNSAFE_PUBLIC_BIND=true"
        )
    warnings.append("开发服务正在监听非回环地址；禁止连接不可信网络或互联网")
    return warnings
