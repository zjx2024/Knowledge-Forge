#!/usr/bin/env python3
"""Prove ignored canaries are absent from Docker context and final image."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import urllib.request
import uuid


ROOT = Path(__file__).resolve().parent.parent
CANARIES = (
    Path(".git/release-canary"), Path(".venv/release-canary"),
    Path(".pytest_cache/release-canary"), Path("KnowledgeFiles/release-canary"),
    Path("logs/release-canary"), Path("dumps/release-canary.dump"),
    Path("Knowledge-Forge-Codex-release-canary.md"), Path(".env.release-canary"),
)


def _run(*command: str) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def _output(*command: str) -> str:
    return subprocess.run(
        command, cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE,
    ).stdout.strip()


def _verify_health() -> None:
    suffix = uuid.uuid4().hex[:10]
    network = f"kf-ci-{suffix}"
    postgres = f"kf-ci-postgres-{suffix}"
    application = f"kf-ci-app-{suffix}"
    try:
        _run("docker", "network", "create", network)
        _run(
            "docker", "run", "-d", "--name", postgres, "--network", network,
            "-e", "POSTGRES_USER=postgres", "-e", "POSTGRES_PASSWORD=ci-postgres-secret",
            "-e", "POSTGRES_DB=knowledge_forge", "pgvector/pgvector:0.8.1-pg16-bookworm",
        )
        for _ in range(60):
            status = subprocess.run(
                ["docker", "exec", postgres, "pg_isready", "-U", "postgres", "-d", "knowledge_forge"],
                cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            if status.returncode == 0:
                break
            time.sleep(1)
        else:
            raise RuntimeError("Docker health PostgreSQL did not become ready")
        _run(
            "docker", "run", "-d", "--name", application, "--network", network,
            "-p", "127.0.0.1::8011",
            "-e", "RUNTIME_MODE=development", "-e", "UPLOAD_HOST=0.0.0.0",
            "-e", "ALLOW_UNSAFE_PUBLIC_BIND=true", "-e", "SERVICE_HOST=127.0.0.1",
            "-e", f"POSTGRES_HOST={postgres}", "-e", "POSTGRES_PORT=5432",
            "-e", "POSTGRES_USER=postgres", "-e", "POSTGRES_PASSWORD=ci-postgres-secret",
            "-e", "POSTGRES_DB=knowledge_forge", "-e", "POSTGRES_MAINTENANCE_DB=postgres",
            "-e", "HUGEGRAPH_HOST=127.0.0.1", "-e", "OLLAMA_BASE_URL=http://127.0.0.1:9",
            "knowledge-forge:ci",
        )
        mapping = _output("docker", "port", application, "8011/tcp")
        port = int(mapping.rsplit(":", 1)[1])
        for _ in range(120):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as response:
                    if response.status == 200 and b"Knowledge Forge" in response.read():
                        return
            except Exception:
                pass
            running = _output("docker", "inspect", "-f", "{{.State.Running}}", application)
            if running != "true":
                logs = _output("docker", "logs", application)
                raise RuntimeError(f"Knowledge Forge image exited during health check:\n{logs}")
            time.sleep(1)
        logs = _output("docker", "logs", application)
        raise RuntimeError(f"Knowledge Forge image did not become healthy:\n{logs}")
    finally:
        subprocess.run(
            ["docker", "rm", "-f", application, postgres], cwd=ROOT,
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["docker", "network", "rm", network], cwd=ROOT,
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--health", action="store_true")
    args = parser.parse_args(argv)
    if args.health and not args.build:
        parser.error("--health requires --build")
    created: list[Path] = []
    marker = "KF_CANARY_" + uuid.uuid4().hex
    context_dockerfile: Path | None = None
    try:
        for relative in CANARIES:
            path = ROOT / relative
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(marker, encoding="utf-8")
            created.append(path)
        with tempfile.NamedTemporaryFile(
            "w", prefix=".docker-context-check-", suffix=".Dockerfile",
            dir=ROOT, delete=False,
        ) as handle:
            handle.write("FROM python:3.10.16-slim-bookworm\nCOPY . /context\n")
            context_dockerfile = Path(handle.name)
        _run(
            "docker", "build", "--network=host", "-f", str(context_dockerfile),
            "-t", "knowledge-forge-context-check", ".",
        )
        _run("docker", "run", "--rm", "knowledge-forge-context-check", "sh", "-c",
             f"! grep -R -F {marker} /context")
        if args.build:
            _run(
                "docker", "build", "--network=host", "-t", "knowledge-forge:ci", ".",
            )
            _run("docker", "run", "--rm", "knowledge-forge:ci", "sh", "-c",
                 f"! grep -R -F {marker} /app /home/knowledgeforge 2>/dev/null")
        if args.health:
            _verify_health()
    finally:
        for path in reversed(created):
            path.unlink(missing_ok=True)
            parent = path.parent
            while parent != ROOT and parent.exists():
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent
        if context_dockerfile is not None:
            context_dockerfile.unlink(missing_ok=True)
        if shutil.which("docker"):
            subprocess.run(
                ["docker", "image", "rm", "knowledge-forge-context-check"],
                cwd=ROOT, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
