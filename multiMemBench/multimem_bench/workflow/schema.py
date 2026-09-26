"""Durable workflow state and hashing helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


STAGE_STATUSES = {
    "pending",
    "running",
    "completed",
    "failed",
    "submission_unknown",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: str | Path, value: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True, ensure_ascii=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass
class StageRecord:
    name: str
    status: str = "pending"
    input_hash: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    outputs: list[str] = field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in STAGE_STATUSES:
            raise ValueError(f"invalid stage status: {self.status!r}")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StageRecord":
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationRequest:
    backend: str
    model: str
    image_path: str
    image_sha256: str
    reference_path: str
    reference_sha256: str
    prompt_sha256: str
    duration: float
    size: str
    seed: int
    output_path: str
    prompt: str | None = None
    image_url: str | None = None
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GenerationRequest":
        return cls(**value)

    def to_dict(self, *, include_transient: bool = False) -> dict[str, Any]:
        value = asdict(self)
        if not include_transient:
            value.pop("image_url", None)
        return value


@dataclass
class GenerationResult:
    video_path: str
    model_id: str
    provider_job_id: str | None = None
    effective_parameters: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    cost: float | None = None
    transport: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GenerationResult":
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunManifest:
    schema_version: str
    run_id: str
    scene_id: str
    status: str
    created_at: str
    updated_at: str
    run_config: dict[str, Any]
    stages: dict[str, StageRecord] = field(default_factory=dict)
    generation_request: GenerationRequest | None = None
    generation_result: GenerationResult | None = None
    video_metadata: dict[str, Any] | None = None
    result_summary: dict[str, Any] | None = None

    @classmethod
    def new(
        cls,
        *,
        run_id: str,
        scene_id: str,
        run_config: dict[str, Any],
    ) -> "RunManifest":
        now = utc_now()
        return cls(
            schema_version="1.0",
            run_id=run_id,
            scene_id=scene_id,
            status="pending",
            created_at=now,
            updated_at=now,
            run_config=run_config,
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RunManifest":
        raw = dict(value)
        raw["stages"] = {
            name: StageRecord.from_dict(stage)
            for name, stage in raw.get("stages", {}).items()
        }
        if raw.get("generation_request") is not None:
            raw["generation_request"] = GenerationRequest.from_dict(
                raw["generation_request"]
            )
        if raw.get("generation_result") is not None:
            raw["generation_result"] = GenerationResult.from_dict(
                raw["generation_result"]
            )
        return cls(**raw)

    @classmethod
    def load(cls, path: str | Path) -> "RunManifest":
        with Path(path).open("r", encoding="utf-8") as source:
            value = json.load(source)
        if not isinstance(value, dict):
            raise ValueError("run manifest must be a JSON object")
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.generation_request is not None:
            value["generation_request"] = self.generation_request.to_dict()
        return value

    def save(self, path: str | Path) -> None:
        self.updated_at = utc_now()
        atomic_write_json(path, self.to_dict())
