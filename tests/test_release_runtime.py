from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from knowledge_forge.audit import redact_audit_payload
from knowledge_forge.runtime_safety import RuntimeSafetyError, validate_runtime_config


ROOT = Path(__file__).resolve().parent.parent


def _runtime(**overrides):
    values = {
        "runtime_mode": "development",
        "upload_host": "127.0.0.1",
        "allow_unsafe_public_bind": False,
        "public_bind_approved": False,
        "authentication_enabled": False,
        "postgres": {"password": "strong-postgres-secret"},
        "hugegraph_user": "graph-user",
        "hugegraph_password": "strong-graph-secret",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_runtime_defaults_are_loopback_and_missing_or_weak_secrets_fail_closed():
    validate_runtime_config(_runtime())
    with pytest.raises(RuntimeSafetyError, match="POSTGRES_PASSWORD"):
        validate_runtime_config(_runtime(postgres={"password": "change-me"}))
    with pytest.raises(RuntimeSafetyError, match="HUGEGRAPH_PASSWORD"):
        validate_runtime_config(_runtime(hugegraph_password="admin"))


def test_public_bind_requires_explicit_dev_switch_or_production_auth_and_approval():
    with pytest.raises(RuntimeSafetyError, match="ALLOW_UNSAFE_PUBLIC_BIND"):
        validate_runtime_config(_runtime(upload_host="0.0.0.0"))
    warnings = validate_runtime_config(
        _runtime(upload_host="0.0.0.0", allow_unsafe_public_bind=True),
    )
    assert any("不可信网络" in warning for warning in warnings)
    with pytest.raises(RuntimeSafetyError, match="认证"):
        validate_runtime_config(_runtime(
            runtime_mode="production", upload_host="0.0.0.0",
            allow_unsafe_public_bind=True, public_bind_approved=True,
        ))
    validate_runtime_config(_runtime(
        runtime_mode="production", upload_host="0.0.0.0",
        public_bind_approved=True, authentication_enabled=True,
    ))


def test_runtime_safety_is_validated_before_service_or_database_initialization():
    source = (ROOT / "knowledge_forge/web.py").read_text(encoding="utf-8")
    assert "SERVICE: KnowledgeForgeService | None = None" in source
    serve = source[source.index("def serve()") :]
    assert serve.index("validate_runtime_config(Config)") < serve.index("KnowledgeForgeService()")


def test_audit_redaction_removes_prompts_content_and_secrets_recursively():
    redacted = redact_audit_payload({
        "prompt": "private prompt",
        "content": "private document",
        "nested": {"api_key": "secret", "count": 3},
        "items": [{"password": "secret", "status": "ok"}],
    })
    assert redacted == {
        "prompt": "[REDACTED]",
        "content": "[REDACTED]",
        "nested": {"api_key": "[REDACTED]", "count": 3},
        "items": [{"password": "[REDACTED]", "status": "ok"}],
    }


def test_docker_context_compose_image_and_ci_are_release_safe():
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    for required in (
        ".git", ".venv", "__pycache__", ".pytest_cache", ".env", ".env.*",
        "!.env.example", "KnowledgeFiles", "artifacts", "benchmarks/results",
        "logs", "dumps", "*.dump", "*.backup", "keys", "secrets", "*.pem",
        "*.key", "*.p12", "*.crt", ".coverage", "htmlcov",
        "Knowledge-Forge-Codex-*.md",
    ):
        assert required in dockerignore

    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert '"127.0.0.1:5433:5432"' in compose
    assert '"127.0.0.1:8011:8011"' in compose
    assert '"127.0.0.1:8010:8010"' in compose
    assert "CONSTRAINTS_FILE: constraints-gpu.txt" in compose
    assert "DOCLING_REQUIREMENTS: requirements-docling-gpu.txt" in compose

    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert (
        "FROM python:3.12.13-slim-bookworm@sha256:"
        "d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b"
    ) in dockerfile
    assert "ARG CONSTRAINTS_FILE=constraints.txt" in dockerfile
    assert "ARG DOCLING_REQUIREMENTS=requirements-docling.txt" in dockerfile
    assert "Acquire::ForceIPv4=true" in dockerfile
    assert "USER knowledgeforge" in dockerfile

    cpu_requirements = (ROOT / "requirements-docling.txt").read_text(encoding="utf-8")
    gpu_requirements = (ROOT / "requirements-docling-gpu.txt").read_text(encoding="utf-8")
    assert "torch-2.7.1%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl" in cpu_requirements
    assert "torch==2.7.1+cu128" in gpu_requirements

    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    for required in (
        "permissions:", "contents: read", "persist-credentials: false",
        "timeout-minutes:", "concurrency:", "postgres-integration:",
        "docker-build:", "pytest", "py_compile", "pip check",
    ):
        assert required in workflow
    assert "KNOWLEDGE_FORGE_TEST_POSTGRES_DSN" not in workflow
    assert "POSTGRES_MAINTENANCE_DB: postgres" in workflow
    assert "--strict-markers" in workflow
    assert "--health" in workflow
    assert "Require at least one passed integration test" in workflow
    assert "scripts/run_disposable_postgres_tests.py" in workflow
    assert "cp .env.docker.example .env.docker" in workflow
    assert "python -m pytest --strict-markers -q -m integration" not in workflow
    assert "actions/checkout@v4" not in workflow
    assert "actions/setup-python@v5" not in workflow
    assert "actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5" in workflow
    assert "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065" in workflow
    assert "python -m pip install pip==25.3" in workflow
    assert "--upgrade pip" not in workflow


def test_python312_local_entrypoint_matches_ci_and_docker_dependencies():
    assert (ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.12"

    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for required in (
        ".venv/",
        ".venv-*-backup-*/",
        ".venv-*-failed-*/",
        ".graphrag-smoke/",
    ):
        assert required in gitignore

    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert ".venv-*" in dockerignore

    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    full_install = (
        "python -m pip install -c constraints.txt -r requirements.txt "
        "-r requirements-docling.txt -r requirements-entity-resolution.txt"
    )
    assert workflow.count(full_install) == 2

    for path in (
        ROOT / "README.md",
        ROOT / "docs/pre-live-readiness.md",
        ROOT / "docs/migrations-and-release-safety.md",
    ):
        content = path.read_text(encoding="utf-8")
        assert "conda run -n knowledge-forge-py312" not in content, path
        assert ".venv/bin/python" in content, path

    generator = (ROOT / "scripts/generate_python312_constraints.py").read_text(
        encoding="utf-8"
    )
    assert "conda run -n" not in generator
    assert ".venv/" not in generator
    assert "python3.12" in generator

    for path in (ROOT / "constraints.txt", ROOT / "constraints-gpu.txt"):
        content = path.read_text(encoding="utf-8")
        assert "conda run -n" not in content, path
        assert ".venv/" not in content, path
        assert "/home/" not in content, path
        assert "python3.12" in content, path

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "python3.12 -m venv .venv" in readme
    assert ".venv/bin/python -m pip install pip==25.3" in readme
    assert ".venv/bin/python" in readme
    assert "VS Code" in readme


def test_backup_restore_and_migration_runbook_are_executable_not_just_documented():
    verifier = (ROOT / "scripts/verify_postgres_backup_restore.py").read_text(encoding="utf-8")
    for required in (
        "pg_dump", "pg_restore", "knowledge_forge_backup_", "kf_backup_sentinel",
        "MigrationRunner", "manifest", "DROP DATABASE",
    ):
        assert required in verifier

    runbook = (ROOT / "docs/migrations-and-release-safety.md").read_text(encoding="utf-8")
    for required in (
        "manage_migrations.py plan", "manage_migrations.py status",
        "manage_migrations.py apply", "verify_postgres_backup_restore.py",
        "cleanup-legacy-columns", "forward-fix", "127.0.0.1",
    ):
        assert required in runbook
