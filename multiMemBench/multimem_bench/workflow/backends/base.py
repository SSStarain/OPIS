"""Shared video backend protocol."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from ..schema import GenerationRequest, GenerationResult


class SubmissionUnknownError(RuntimeError):
    """A paid provider POST may have succeeded without returning a task ID."""


class VideoBackend(Protocol):
    def generate(
        self,
        request: GenerationRequest,
        *,
        resume_state: dict[str, Any] | None = None,
        on_state: Callable[[dict[str, Any]], None] | None = None,
    ) -> GenerationResult: ...
