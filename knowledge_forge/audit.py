"""Audit payload redaction and bounded-retention helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_SENSITIVE_KEYS = {
    "api_key", "apikey", "authorization", "content", "document", "input",
    "password", "prompt", "raw_response", "response", "secret", "source_text",
    "text", "token",
}


def redact_audit_payload(value: Any) -> Any:
    """Recursively redact likely source, prompt, response, and credential fields."""

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if normalized in _SENSITIVE_KEYS or any(
                token in normalized for token in ("password", "secret", "api_key", "prompt")
            ):
                result[str(key)] = "[REDACTED]"
            else:
                result[str(key)] = redact_audit_payload(item)
        return result
    if isinstance(value, (list, tuple)):
        return [redact_audit_payload(item) for item in value]
    return value
