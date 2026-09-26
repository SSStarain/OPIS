"""Open-category labeling hooks for segmented crops."""

from __future__ import annotations

import json
from pathlib import Path
import shlex
import subprocess
from typing import Any, Protocol

from multimem_bench.vision.config import VisionConfig
from multimem_bench.vision.types import SegmentInstance


class SegmentLabeler(Protocol):
    backend_name: str

    def label(
        self,
        segment: SegmentInstance,
        *,
        image_path: Path,
        crop_path: Path | None,
        mask_path: Path | None,
    ) -> tuple[str, dict[str, Any]]:
        ...


class NoopLabeler:
    backend_name = "none"

    def __init__(self, unknown_category_name: str = "object") -> None:
        self.unknown_category_name = unknown_category_name

    def label(
        self,
        segment: SegmentInstance,
        *,
        image_path: Path,
        crop_path: Path | None,
        mask_path: Path | None,
    ) -> tuple[str, dict[str, Any]]:
        del image_path, crop_path, mask_path
        category = segment.category or self.unknown_category_name
        return category, {}


class ExternalCommandLabeler:
    """Run a user-provided command that emits a JSON label record.

    The command template may use:
    `{image}`, `{crop}`, `{mask}`, `{bbox}`, and `{segment_id}`.
    It should print JSON such as:
    `{"category": "green tomato", "attributes": {"color": "green"}}`.
    """

    backend_name = "external"

    def __init__(self, command_template: str, timeout_sec: float) -> None:
        self.command_template = command_template
        self.timeout_sec = timeout_sec

    def label(
        self,
        segment: SegmentInstance,
        *,
        image_path: Path,
        crop_path: Path | None,
        mask_path: Path | None,
    ) -> tuple[str, dict[str, Any]]:
        values = {
            "image": shlex.quote(str(image_path)),
            "crop": shlex.quote(str(crop_path or "")),
            "mask": shlex.quote(str(mask_path or "")),
            "bbox": shlex.quote(json.dumps(list(segment.bbox))),
            "segment_id": shlex.quote(segment.segment_id),
        }
        command = self.command_template.format(**values)
        completed = subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            capture_output=True,
            timeout=self.timeout_sec,
        )
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        category = str(payload.get("category") or payload.get("label") or segment.category)
        attributes = dict(payload.get("attributes", {}))
        return category, attributes


def build_labeler(config: VisionConfig) -> SegmentLabeler:
    backend = config.labeler_backend.lower()
    if backend == "none":
        return NoopLabeler(config.unknown_category_name)
    if backend == "external":
        if not config.labeler_command:
            raise ValueError("labeler_backend=external requires labeler_command")
        return ExternalCommandLabeler(config.labeler_command, config.labeler_timeout_sec)
    raise ValueError(f"unsupported labeler_backend: {config.labeler_backend}")
