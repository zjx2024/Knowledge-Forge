"""Provision and clean a sentinel-bound, least-privilege PostgreSQL test schema."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import re
import secrets
import uuid

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from .e2e_safety import E2ESafetyError, PostgresCleanupPlan


class PostgresIsolationError(RuntimeError):
    """PostgreSQL isolation or sentinel ownership could not be proven."""


def canonical_postgres_identity(
    *, host: str, port: int, database: str, user: str, schema: str,
) -> str:
    if not all((host, database, user, schema)):
        raise PostgresIsolationError("PostgreSQL identity is incomplete")
    return f"postgresql://{user}@{host}:{int(port)}/{database}/{schema}"


def validate_isolated_identity(test_identity: str, production_identity: str, *, run_id: str) -> None:
    if not production_identity:
        raise PostgresIsolationError("production PostgreSQL identity is unknown")
    if not test_identity:
        raise PostgresIsolationError("test PostgreSQL identity is unknown")
    if test_identity == production_identity:
        raise PostgresIsolationError("test and production PostgreSQL identities are equal")
    if not re.fullmatch(r"[a-f0-9]{16,64}", run_id):
        raise PostgresIsolationError("test run_id is not a random bounded identifier")
    if not test_identity.endswith(f"/kf_e2e_{run_id}"):
        raise PostgresIsolationError("test PostgreSQL schema is outside the random namespace")


def inspect_postgres_identity(dsn: str, *, schema: str) -> tuple[str, str, str]:
    values = conninfo_to_dict(dsn)
    with psycopg.connect(dsn) as connection:
        connection.execute("SET TRANSACTION READ ONLY")
        row = connection.execute(
            "SELECT current_user, current_database(), COALESCE(inet_server_addr()::text, %s), "
            "COALESCE(inet_server_port(), %s), version()",
            (values.get("host") or "localhost", int(values.get("port") or 5432)),
        ).fetchone()
        vector = connection.execute(
            "SELECT extversion FROM pg_extension WHERE extname='vector'"
        ).fetchone()
    identity = canonical_postgres_identity(
        host=str(row[2]).split("/", 1)[0], port=row[3], database=row[1],
        user=row[0], schema=schema,
    )
    return identity, str(row[4]), str(vector[0]) if vector else ""


@dataclass(frozen=True)
class DisposablePostgresContext:
    maintenance_dsn: str
    target_admin_dsn: str
    test_dsn: str
    run_id: str
    database: str
    schema: str
    role: str
    identity: str
    production_identity: str
    config_hash: str
    expires_at: str

    def safe_report(self) -> dict[str, str]:
        return {
            "run_id": self.run_id, "database": self.database,
            "schema": self.schema, "role": self.role,
            "identity": self.identity, "production_identity": self.production_identity,
            "config_hash": self.config_hash, "expires_at": self.expires_at,
        }


def _sentinel_payload(context: DisposablePostgresContext) -> tuple[str, ...]:
    return (
        context.run_id, context.identity, context.schema, context.role,
        context.config_hash, context.expires_at,
    )


def provision_disposable_schema(
    admin_dsn: str, *, production_identity: str, ttl_seconds: int = 1800,
) -> DisposablePostgresContext:
    run_id = uuid.uuid4().hex[:20]
    database = f"kf_e2e_db_{run_id}"
    schema = f"kf_e2e_{run_id}"
    role = f"kf_e2e_role_{run_id}"
    password = secrets.token_urlsafe(24)
    values = conninfo_to_dict(admin_dsn)
    maintenance_values = {**values, "dbname": values.get("dbname") or "postgres"}
    maintenance_dsn = make_conninfo(**maintenance_values)
    target_admin_dsn = make_conninfo(**{**values, "dbname": database})

    with psycopg.connect(maintenance_dsn, autocommit=True) as connection:
        if connection.execute(
            "SELECT 1 FROM pg_database WHERE datname=%s", (database,),
        ).fetchone():
            raise PostgresIsolationError("random disposable database already exists")
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))

    try:
        with psycopg.connect(target_admin_dsn, autocommit=True) as connection:
            connection.execute("CREATE EXTENSION vector")
            connection.execute("CREATE EXTENSION pg_trgm")
        actual_admin, _server_version, pgvector_version = inspect_postgres_identity(
            target_admin_dsn, schema="public",
        )
        if not pgvector_version:
            raise PostgresIsolationError("pgvector extension is not installed in the target database")
    except Exception:
        with psycopg.connect(maintenance_dsn, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(database)))
        raise

    _authority, location = actual_admin.split("@", 1)
    actual_host = location.rsplit("/", 2)[0].rsplit(":", 1)[0]
    actual_port = int(location.rsplit("/", 2)[0].rsplit(":", 1)[1])
    identity = canonical_postgres_identity(
        host=actual_host, port=actual_port, database=database,
        user=role, schema=schema,
    )
    validate_isolated_identity(identity, production_identity, run_id=run_id)
    expires_at = (datetime.now(timezone.utc) + timedelta(seconds=max(60, ttl_seconds))).isoformat()
    config = {
        "run_id": run_id, "database": database, "schema": schema, "role": role,
        "identity": identity, "production_identity": production_identity,
        "expires_at": expires_at,
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    test_values = {
        **values, "dbname": database, "user": role, "password": password,
        # Product objects are confined to the owned schema.  ``public`` is
        # present only so extension-owned types/functions (notably pgvector
        # and pg_trgm) remain resolvable; the role has no CREATE privilege
        # there.
        "options": f"-c search_path={schema},public -c statement_timeout=120000",
    }
    test_dsn = make_conninfo(**test_values)
    context = DisposablePostgresContext(
        maintenance_dsn=maintenance_dsn, target_admin_dsn=target_admin_dsn,
        test_dsn=test_dsn, run_id=run_id, database=database,
        schema=schema, role=role, identity=identity,
        production_identity=production_identity, config_hash=config_hash,
        expires_at=expires_at,
    )

    try:
        with psycopg.connect(target_admin_dsn, autocommit=True) as connection:
            connection.execute(sql.SQL(
                "CREATE ROLE {} LOGIN PASSWORD {} NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT"
            ).format(sql.Identifier(role), sql.Literal(password)))
            connection.execute(
                sql.SQL("CREATE SCHEMA {} AUTHORIZATION {}").format(
                    sql.Identifier(schema), sql.Identifier(role),
                )
            )
            connection.execute(sql.SQL(
                "CREATE TABLE {}.kf_e2e_sentinel ("
                "run_id text PRIMARY KEY, identity text NOT NULL, schema_name text NOT NULL, "
                "role_name text NOT NULL, config_hash text NOT NULL, expires_at timestamptz NOT NULL)"
            ).format(sql.Identifier(schema)))
            connection.execute(sql.SQL(
                "INSERT INTO {}.kf_e2e_sentinel "
                "(run_id,identity,schema_name,role_name,config_hash,expires_at) "
                "VALUES (%s,%s,%s,%s,%s,%s)"
            ).format(sql.Identifier(schema)), _sentinel_payload(context))
            connection.execute(sql.SQL(
                "GRANT SELECT ON {}.kf_e2e_sentinel TO {}"
            ).format(sql.Identifier(schema), sql.Identifier(role)))
            permissions = connection.execute(
                "SELECT has_database_privilege(%s, current_database(), 'CREATE'), "
                "has_schema_privilege(%s, 'public', 'CREATE')",
                (role, role),
            ).fetchone()
            if any(permissions):
                raise PostgresIsolationError(
                    "test role has CREATE outside its owned random schema"
                )
    except Exception:
        with psycopg.connect(target_admin_dsn, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))
            connection.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role)))
        with psycopg.connect(maintenance_dsn, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(database)))
        raise
    verify_postgres_sentinel(context)
    return context


def verify_postgres_sentinel(context: DisposablePostgresContext) -> None:
    with psycopg.connect(context.target_admin_dsn) as connection:
        row = connection.execute(sql.SQL(
            "SELECT run_id,identity,schema_name,role_name,config_hash,expires_at "
            "FROM {}.kf_e2e_sentinel"
        ).format(sql.Identifier(context.schema))).fetchone()
    if row is None:
        raise PostgresIsolationError("PostgreSQL test sentinel is missing")
    expected = _sentinel_payload(context)
    if tuple(str(value) for value in row[:5]) != tuple(str(value) for value in expected[:5]):
        raise PostgresIsolationError("PostgreSQL test sentinel does not match its context")
    expiry = row[5]
    if expiry.astimezone(timezone.utc) <= datetime.now(timezone.utc):
        raise PostgresIsolationError("PostgreSQL test sentinel expired")


def cleanup_disposable_schema(context: DisposablePostgresContext) -> str:
    with psycopg.connect(context.maintenance_dsn) as connection:
        database_exists = bool(connection.execute(
            "SELECT 1 FROM pg_database WHERE datname=%s", (context.database,),
        ).fetchone())
        role_exists = bool(connection.execute(
            "SELECT 1 FROM pg_roles WHERE rolname=%s", (context.role,),
        ).fetchone())
    if not database_exists and not role_exists:
        return "already_clean"
    if not database_exists:
        raise PostgresIsolationError(
            "disposable database is missing but its role remains; refusing unverified cleanup"
        )
    try:
        verify_postgres_sentinel(context)
    except (psycopg.Error, PostgresIsolationError):
        with psycopg.connect(context.target_admin_dsn) as connection:
            existing = connection.execute(
                "SELECT EXISTS(SELECT 1 FROM pg_namespace WHERE nspname=%s), "
                "EXISTS(SELECT 1 FROM pg_roles WHERE rolname=%s)",
                (context.schema, context.role),
            ).fetchone()
        if existing == (False, False):
            return "already_clean"
        raise
    try:
        operations = PostgresCleanupPlan(
            schema=context.schema, role=context.role, run_id=context.run_id,
            sentinel_verified=True,
        ).operations()
    except E2ESafetyError as exc:
        raise PostgresIsolationError(str(exc)) from exc
    with psycopg.connect(context.target_admin_dsn, autocommit=True) as connection:
        connection.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            "WHERE usename=%s AND pid<>pg_backend_pid()", (context.role,),
        )
        for operation, identifier in operations:
            if operation == "drop_schema":
                connection.execute(
                    sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(identifier))
                )
            elif operation == "drop_role":
                connection.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(identifier)))
            else:
                raise PostgresIsolationError(f"unsupported cleanup operation: {operation}")
    if context.database != f"kf_e2e_db_{context.run_id}":
        raise PostgresIsolationError("cleanup database is outside the owned random namespace")
    with psycopg.connect(context.maintenance_dsn, autocommit=True) as connection:
        connection.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            "WHERE datname=%s AND pid<>pg_backend_pid()", (context.database,),
        )
        connection.execute(
            sql.SQL("DROP DATABASE {}").format(sql.Identifier(context.database))
        )
    return "cleaned"
