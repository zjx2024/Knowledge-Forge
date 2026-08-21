"""Unicode sanitation at document and persistence boundaries.

PDF text maps and OCR/model responses can contain NUL bytes or isolated
UTF-16 surrogate code units.  Neither value is valid application text: NUL is
rejected by PostgreSQL and isolated surrogates cannot be encoded as UTF-8.
Keep the repair deterministic and deliberately narrow so ordinary Unicode,
including valid supplementary-plane mathematical symbols, is unchanged.
"""

from __future__ import annotations

from typing import Any


def unsafe_unicode_code_unit_count(value: Any) -> int:
    """Count NUL and surrogate code units recursively in JSON-like data."""

    if isinstance(value, str):
        return sum(
            character == "\x00" or 0xD800 <= ord(character) <= 0xDFFF
            for character in value
        )
    if isinstance(value, dict):
        return sum(
            unsafe_unicode_code_unit_count(key)
            + unsafe_unicode_code_unit_count(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return sum(unsafe_unicode_code_unit_count(item) for item in value)
    return 0


def sanitize_unicode_text(value: str) -> str:
    """Return UTF-8/PostgreSQL-safe text without changing valid Unicode."""

    if "\x00" not in value and not any(
        0xD800 <= ord(character) <= 0xDFFF for character in value
    ):
        return value
    # A NUL usually represents an unmapped PDF glyph.  A space preserves the
    # surrounding token boundary and character count without inventing a
    # mathematical operator or other source content.
    repaired = value.replace("\x00", " ")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in repaired):
        # surrogatepass followed by the strict replacement decoder combines a
        # valid pair into its real supplementary character and turns only an
        # isolated half into U+FFFD.
        repaired = repaired.encode(
            "utf-16-le", errors="surrogatepass",
        ).decode("utf-16-le", errors="replace")
    return repaired


def sanitize_unicode_data(value: Any) -> Any:
    """Recursively sanitize JSON-like values while preserving containers."""

    if isinstance(value, str):
        return sanitize_unicode_text(value)
    if isinstance(value, dict):
        return {
            sanitize_unicode_text(key) if isinstance(key, str) else key:
            sanitize_unicode_data(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_unicode_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_unicode_data(item) for item in value)
    if isinstance(value, set):
        return {sanitize_unicode_data(item) for item in value}
    if isinstance(value, frozenset):
        return frozenset(sanitize_unicode_data(item) for item in value)
    return value
