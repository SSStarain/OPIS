"""Secret and signed-URL redaction for workflow logs and artifacts."""

from __future__ import annotations

from collections.abc import Iterable
import re


_BEARER_RE = re.compile(r"(?i)(Bearer\s+)[^\s,;]+")
_SIGNED_URL_RE = re.compile(
    r"(https?://[^\s?]+)\?[^\s]*(?:X-Amz-|Signature=|token=)[^\s]*",
    re.IGNORECASE,
)


def sanitize_text(text: str, secrets: Iterable[str] = ()) -> str:
    """Remove credentials and signed query strings from diagnostic text."""

    sanitized = _BEARER_RE.sub(r"\1<redacted>", str(text))
    sanitized = _SIGNED_URL_RE.sub(r"\1?<redacted>", sanitized)
    for secret in sorted((item for item in secrets if item), key=len, reverse=True):
        sanitized = sanitized.replace(secret, "<redacted>")
    return sanitized
