"""Capability-session helpers for deployments without a user identity layer."""

from __future__ import annotations

import hashlib
import re
import secrets


SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{32,160}$")


def new_session_key() -> str:
    return secrets.token_urlsafe(36)


def validate_session_key(value: str) -> str:
    value = str(value or "").strip()
    if not SESSION_RE.fullmatch(value):
        raise PermissionError("无效的问答会话凭据")
    return value


def session_key_hash(value: str) -> str:
    return hashlib.sha256(validate_session_key(value).encode("utf-8")).hexdigest()
