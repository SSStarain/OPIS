"""One-command generation, observation, geometry, and V2 evaluation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable
from urllib.parse import urlparse

from .backends.ark import ArkBackend, ArkVideoClient, DEFAULT_ARK_BASE_URL
from .backends.base import SubmissionUnknownError, VideoBackend
from .backends.openrouter import (
    OpenRouterBackend,
    OpenRouterVideoClient,
    DEFAULT_API_BASE_URL,
)
from .backends.wan22 import Wan22Backend, local_model_profile
from .backends.local_i2v import LocalI2VBackend
from .backends.local_world_model import LocalWorldModelBackend
from .backends.fal import FalBackend, resolve_preferences
from .schema import (
    GenerationRequest,
    GenerationResult,
    RunManifest,
    StageRecord,
    atomic_write_json,
    canonical_hash,
    sha256_file,
    utc_now,
)
from .security import sanitize_text
from .storage import (
    PublishedImage,
    S3ImagePublisher,
    S3PublishConfig,
    validate_direct_image_url,
)
from .video import VideoMetadata, probe_video, validate_aspect_ratio, validate_duration
from .video_models import get_video_observation_overrides, resolve_local_defaults


STAGE_ORDER = (
    "resolve_inputs",
    "publish_image",
    "generate_video",
    "validate_video",
    "observe_video",
    "prepare_geometry",
    "evaluate_v2",
    "summarize",
)
REMOTE_BACKENDS = frozenset({"openrouter", "ark", "fal"})
LOCAL_BACKENDS = frozenset({"wan22", "local_i2v", "local-wm"})
VIDEO_PROVIDER_KEY_ENVS = {"fal": "FAL_KEY", "ark": "ARK_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
DEFAULT_ARK_MODEL = "doubao-seedance-2-5-260628"
OBSERVATION_PIPELINE_VERSION = "static-3d-topology-coverage50-v1"
GEOMETRY_PIPELINE_VERSION = "static-3d-topology-coverage50-v1"
EVALUATION_PIPELINE_VERSION = "static-3d-occlusion-aware-v2"
V3_WORKFLOW_CONTRACT_VERSION = "input-grounded-workflow-v3-ungated-full-igm"


class WorkflowError(RuntimeError):
    """A benchmark workflow cannot proceed or a required stage failed."""


@dataclass(frozen=True)
class RunConfig:
    scene_dir: Path | None
    image: Path
    reference: Path
    source_video: Path | None
    source_video_sha256: str | None
    prompt: str | None
    scene_id: str
    backend: str | None
    model: str
    variant: str
    duration: float | None
    size: str | None
    seed: int | None
    output_root: Path
    output_dir: Path
    run_id: str
    generation_fingerprint: str
    image_sha256: str
    reference_sha256: str
    prompt_sha256: str | None
    image_url: str | None
    s3_config: S3PublishConfig | None
    cleanup_upload: bool
    resume: bool
    dry_run: bool
    vision_config: Path | None
    v2_config: Path | None
    geometry_backend: str | None
    geometry_checkpoint: Path | None
    track: str
    strict_models: bool
    strict_geometry: bool
    vision_conda_env: str | None
    local_conda_env: str
    cuda_visible_devices: str | None
    frame_num: int | None
    wan_repo: Path | None
    wan_checkpoint: Path | None
    world_model_root: Path | None
    world_model_final_root: Path | None
    world_model_task: Path | None
    world_model_case_id: str | None
    wm_intrinsics_gpu: str | None
    request_timeout: float
    poll_interval: float
    generation_timeout: float
    local_timeout: float
    ffprobe: str
    api_base_url: str | None = None
    video_model_profile: dict[str, Any] | None = None

    @property
    def generation_mode(self) -> str | None:
        if self.source_video:
            return None
        return "local" if self.backend in LOCAL_BACKENDS else "api"

    def manifest_config(self) -> dict[str, Any]:
        storage: dict[str, Any] | None = None
        if self.s3_config is not None:
            storage = {
                "bucket": self.s3_config.bucket,
                "endpoint_host": urlparse(
                    self.s3_config.endpoint_url or ""
                ).hostname,
                "prefix": self.s3_config.prefix,
                "url_ttl_seconds": self.s3_config.url_ttl_seconds,
                "uses_public_base_url": bool(self.s3_config.public_base_url),
            }
        return {
            "scene_dir": str(self.scene_dir) if self.scene_dir else None,
            "image": str(self.image),
            "reference": str(self.reference),
            "video_source": "existing" if self.source_video else "generated",
            "source_video": str(self.source_video) if self.source_video else None,
            "source_video_sha256": self.source_video_sha256,
            "scene_id": self.scene_id,
            "backend": self.backend,
            "generation_mode": self.generation_mode,
            "api_base_url": self.api_base_url,
            "video_model_profile": self.video_model_profile,
            "model": self.model,
            "variant": self.variant or None,
            "duration": self.duration,
            "size": self.size,
            "seed": self.seed,
            "run_id": self.run_id,
            "generation_fingerprint": self.generation_fingerprint,
            "image_sha256": self.image_sha256,
            "reference_sha256": self.reference_sha256,
            "prompt_sha256": self.prompt_sha256,
            "image_transport": (
                None
                if self.source_video
                else "local"
                if self.backend in LOCAL_BACKENDS
                else "explicit_url"
                if self.image_url
                else "s3"
            ),
            "storage": storage,
            "cleanup_upload": self.cleanup_upload,
            "vision_config": str(self.vision_config) if self.vision_config else None,
            "v2_config": str(self.v2_config) if self.v2_config else None,
            "geometry_backend": self.geometry_backend,
            "geometry_checkpoint": str(self.geometry_checkpoint)
            if self.geometry_checkpoint
            else None,
            "track": self.track,
            "strict_models": self.strict_models,
            "strict_geometry": self.strict_geometry,
            "vision_conda_env": self.vision_conda_env,
            "local_conda_env": self.local_conda_env,
            "wan_repo": str(self.wan_repo) if self.wan_repo else None,
            "wan_checkpoint": str(self.wan_checkpoint) if self.wan_checkpoint else None,
            "world_model_root": str(self.world_model_root) if self.world_model_root else None,
            "world_model_final_root": str(self.world_model_final_root) if self.world_model_final_root else None,
            "world_model_task": str(self.world_model_task) if self.world_model_task else None,
            "world_model_case_id": self.world_model_case_id,
            "wm_intrinsics_gpu": self.wm_intrinsics_gpu,
            "cuda_visible_devices": self.cuda_visible_devices,
            "frame_num": self.frame_num,
            "evaluation_pipeline_version": EVALUATION_PIPELINE_VERSION,
        }


def add_run_benchmark_arguments(parser: argparse.ArgumentParser, *, v3: bool = False) -> None:
    parser.epilog = (
        "Video generation exports: VIDEO_BACKEND=local/api/local-wm, VIDEO_MODEL. "
        "API mode: VIDEO_API_PROVIDER=fal/ark/openrouter, VIDEO_API_KEY, optional VIDEO_BASE_URL. "
        "Local mode: select a configured I2V model with VIDEO_MODEL. "
        "World-model mode: VIDEO_BACKEND=local-wm with sana-wm or lingbot-world-2.0. "
        "API inputs require VIDEO_IMAGE_URL/--image-url or S3/R2 configuration."
    )
    parser.add_argument("--scene-dir", type=Path)
    parser.add_argument("--manifest", type=Path, help="JSON manifest containing case paths for batch runs")
    parser.add_argument("--parallelism", type=int, default=1, help="Concurrent manifest cases")
    parser.add_argument("--cuda-devices", help="Comma-separated GPUs for local-model batch runs")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--video", type=Path)
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file", type=Path)
    parser.add_argument("--scene-id")
    parser.add_argument(
        "--backend", choices=("local", "local-wm", "api", "openrouter", "ark", "wan22", "fal"),
        metavar="{local,local-wm,api}",
        help="Generation mode (or export VIDEO_BACKEND); legacy provider names remain accepted",
    )
    parser.add_argument("--model", help="Video model ID (or export VIDEO_MODEL); separate from VLM_MODEL")
    parser.add_argument("--variant", choices=("i2v-a14b", "ti2v-5b"))
    parser.add_argument("--duration", type=float)
    parser.add_argument("--size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-num", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--image-url", help="Public input image URL (or export VIDEO_IMAGE_URL)")
    parser.add_argument("--s3-bucket")
    parser.add_argument("--s3-endpoint-url")
    parser.add_argument("--s3-region")
    parser.add_argument("--s3-prefix")
    parser.add_argument("--s3-url-ttl", type=int)
    parser.add_argument("--s3-public-base-url")
    parser.add_argument("--cleanup-upload", action="store_true")
    parser.add_argument("--vision-config", type=Path)
    if not v3:
        parser.add_argument("--v2-config", type=Path)
        parser.add_argument("--geometry-backend", choices=("vggt", "vggt_omega"))
        parser.add_argument("--checkpoint", type=Path)
        parser.add_argument("--track", choices=("static_3d",), default="static_3d")
    parser.add_argument(
        "--strict-models", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--strict-geometry", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--vision-conda-env", help="Optional Conda environment; defaults to the active Python environment")
    parser.add_argument("--local-conda-env", help="Legacy Wan2.2 conda environment override")
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--wan-repo", type=Path)
    parser.add_argument("--wan-checkpoint", type=Path)
    parser.add_argument("--world-model-root", type=Path)
    parser.add_argument("--world-model-final-root", type=Path)
    parser.add_argument("--wm-intrinsics-gpu")
    parser.add_argument("--request-timeout", type=float, default=60)
    parser.add_argument("--poll-interval", type=float, default=30)
    parser.add_argument("--generation-timeout", type=float, default=3600)
    parser.add_argument("--local-timeout", type=float, default=14400)
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def _generation_backend(args: argparse.Namespace) -> str | None:
    explicit = getattr(args, "backend", None)
    if getattr(args, "video", None):
        if explicit:
            raise WorkflowError("--video and --backend are mutually exclusive")
        return None
    mode = explicit or os.environ.get("VIDEO_BACKEND", "").strip()
    if mode == "local":
        model = getattr(args, "model", None) or os.environ.get("VIDEO_MODEL", "").strip()
        if not model:
            return "wan22"
        try:
            profile = local_model_profile(str(model))
        except ValueError as exc:
            raise WorkflowError(str(exc)) from exc
        backend = str(profile.get("backend", ""))
        if backend not in LOCAL_BACKENDS:
            raise WorkflowError(f"local video model has an invalid backend: {model}")
        return backend
    if mode == "api":
        provider = os.environ.get("VIDEO_API_PROVIDER", "").strip().lower()
        if provider not in REMOTE_BACKENDS:
            raise WorkflowError("api mode requires VIDEO_API_PROVIDER=fal, ark, or openrouter")
        return provider
    if mode in REMOTE_BACKENDS or mode in LOCAL_BACKENDS:
        return mode
    if mode:
        raise WorkflowError("VIDEO_BACKEND must be local, local-wm, or api")
    return None


def _local_wm_uses_gpu_group(args: argparse.Namespace) -> bool:
    """Return whether one local-wm case consumes the full GPU list."""
    if _generation_backend(args) != "local-wm":
        return False
    model = getattr(args, "model", None) or os.environ.get("VIDEO_MODEL", "").strip()
    return model == "lingbot-world-2.0"


def _video_api_key(backend: str | None) -> str | None:
    return os.environ.get("VIDEO_API_KEY") or os.environ.get(VIDEO_PROVIDER_KEY_ENVS.get(backend, ""))


def _generation_secrets() -> list[str]:
    return [os.environ.get(name, "") for name in ("VIDEO_API_KEY", *VIDEO_PROVIDER_KEY_ENVS.values())]


def _video_base_url(backend: str | None) -> str | None:
    if backend not in REMOTE_BACKENDS:
        return None
    configured = os.environ.get("VIDEO_BASE_URL", "").strip().rstrip("/")
    if backend == "fal":
        if configured and configured != "https://fal.run":
            raise WorkflowError("fal uses its SDK endpoint; unset VIDEO_BASE_URL or use https://fal.run")
        return None
    default = DEFAULT_ARK_BASE_URL if backend == "ark" else DEFAULT_API_BASE_URL
    base_url = configured or (os.environ.get("ARK_BASE_URL") if backend == "ark" else None) or default
    parsed = urlparse(base_url)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment):
        raise WorkflowError("VIDEO_BASE_URL must be an HTTPS base URL without credentials, query, or fragment")
    return base_url.rstrip("/")


def resolve_run_config(args: argparse.Namespace) -> RunConfig:
    scene_dir = _resolve_optional_path(getattr(args, "scene_dir", None))
    if scene_dir is not None and not scene_dir.is_dir():
        raise WorkflowError(f"scene directory does not exist: {scene_dir}")

    source_video = _resolve_optional_path(getattr(args, "video", None))
    backend = _generation_backend(args)
    api_base_url = _video_base_url(backend)
    if source_video is None and backend is None:
        raise WorkflowError("one of --backend or --video is required (or export VIDEO_BACKEND=local/api)")
    if source_video is not None and not source_video.is_file():
        raise WorkflowError(f"existing video does not exist: {source_video}")
    model_value = getattr(args, "model", None) or os.environ.get("VIDEO_MODEL", "").strip()
    if source_video is not None and not model_value:
        raise WorkflowError("--model is required with --video (or export VIDEO_MODEL)")
    if source_video is not None:
        invalid_existing_video_options = [
            option
            for option, value in (
                ("--image-url", getattr(args, "image_url", None)),
                ("--s3-bucket", getattr(args, "s3_bucket", None)),
                ("--s3-endpoint-url", getattr(args, "s3_endpoint_url", None)),
                ("--s3-region", getattr(args, "s3_region", None)),
                ("--s3-prefix", getattr(args, "s3_prefix", None)),
                ("--s3-url-ttl", getattr(args, "s3_url_ttl", None)),
                ("--s3-public-base-url", getattr(args, "s3_public_base_url", None)),
                ("--cleanup-upload", getattr(args, "cleanup_upload", False)),
                ("--variant", getattr(args, "variant", None)),
                ("--frame-num", getattr(args, "frame_num", None)),
                ("--wan-repo", getattr(args, "wan_repo", None)),
                ("--wan-checkpoint", getattr(args, "wan_checkpoint", None)),
            )
            if value is not None and value is not False
        ]
        if invalid_existing_video_options:
            raise WorkflowError(
                f"{', '.join(invalid_existing_video_options)} not valid with --video"
            )

    image_raw = getattr(args, "image", None)
    image = _resolve_optional_path(image_raw) or (
        scene_dir / "raw.jpg" if scene_dir is not None else None
    )
    if image is None or not image.is_file():
        raise WorkflowError(f"initial image does not exist: {image}")

    reference_raw = getattr(args, "reference", None)
    reference = _resolve_optional_path(reference_raw)
    if reference is None and scene_dir is not None:
        # Final dataset stores the annotation at case root; retain legacy support.
        candidates = (
            scene_dir / "reference_scene.json",
            scene_dir / "artifacts" / "reference_scene.json",
        )
        reference = next((path for path in candidates if path.is_file()), candidates[0])
    if reference is None or not reference.is_file():
        raise WorkflowError(f"reference annotation does not exist: {reference}")
    reference_payload = _load_json_object(reference, "reference annotation")
    objects = reference_payload.get("objects")
    if not isinstance(objects, list):
        raise WorkflowError("reference annotation has no objects list")
    if len(objects) > 256:
        raise WorkflowError(
            f"reference annotation contains {len(objects)} objects; maximum is 256"
        )

    conditions_path = scene_dir / "conditions.json" if scene_dir else None
    conditions = (
        _load_json_object(conditions_path, "conditions")
        if conditions_path is not None and conditions_path.is_file()
        else {}
    )
    if source_video is not None:
        prompt = None
    else:
        prompt_text = getattr(args, "prompt", None)
        prompt_file = _resolve_optional_path(getattr(args, "prompt_file", None))
        if prompt_text is not None:
            prompt = str(prompt_text).strip()
        elif prompt_file is not None:
            if not prompt_file.is_file():
                raise WorkflowError(f"prompt file does not exist: {prompt_file}")
            prompt = prompt_file.read_text(encoding="utf-8").strip()
        else:
            default_prompt_file = scene_dir / "text_prompt.txt" if scene_dir else None
            if default_prompt_file and default_prompt_file.is_file():
                prompt = default_prompt_file.read_text(encoding="utf-8").strip()
            else:
                i2v = conditions.get("i2v")
                prompt = (
                    str(i2v.get("motion_prompt", "")).strip()
                    if isinstance(i2v, dict)
                    else ""
                )
        if not prompt:
            raise WorkflowError("video generation prompt is empty")
        if backend == "local-wm":
            canonical_prompt_path = scene_dir / "text_prompt.txt" if scene_dir else None
            if canonical_prompt_path is None or not canonical_prompt_path.is_file():
                raise WorkflowError("local-wm requires text_prompt.txt in --scene-dir")
            canonical_prompt = canonical_prompt_path.read_text(encoding="utf-8").strip()
            if prompt != canonical_prompt:
                raise WorkflowError(
                    "local-wm currently requires the case text_prompt.txt without overrides"
                )

    scene_id = str(
        getattr(args, "scene_id", None)
        or conditions.get("scene_id")
        or reference_payload.get("scene_id")
        or (scene_dir.name if scene_dir else image.stem)
    )
    raw_variant = getattr(args, "variant", None)
    if backend in REMOTE_BACKENDS and any(
        value is not None
        for value in (
            raw_variant,
            getattr(args, "frame_num", None),
            getattr(args, "wan_repo", None),
            getattr(args, "wan_checkpoint", None),
        )
    ):
        raise WorkflowError(
            "Wan2.2 options --variant, --frame-num, --wan-repo, and "
            f"--wan-checkpoint are not valid for {backend}"
        )
    local_profile: dict[str, Any] | None = None
    if backend in LOCAL_BACKENDS:
        local_model = str(model_value or (
            f"wan2.2/{raw_variant or 'i2v-a14b'}" if backend == "wan22" else ""
        ))
        if not local_model:
            raise WorkflowError("--model or VIDEO_MODEL is required for local generation")
        try:
            local_profile = local_model_profile(local_model)
        except ValueError as exc:
            raise WorkflowError(str(exc)) from exc
        if local_profile.get("backend") != backend:
            raise WorkflowError(
                f"local video model profile backend mismatch: {local_model}"
            )
        variant = str(local_profile.get("variant", ""))
        if backend == "wan22" and not variant:
            raise WorkflowError(f"local video model profile has no variant: {local_model}")
        if raw_variant and raw_variant != variant:
            raise WorkflowError("--variant conflicts with the selected local video model")
        model_value = local_model
    else:
        variant = ""
    model = str(
        model_value
        or (
            f"wan2.2/{variant}"
            if backend == "wan22"
            else DEFAULT_ARK_MODEL
            if backend == "ark"
            else "minimax/h3-max/image-to-video"
            if backend == "fal"
            else ""
        )
    )
    if not model:
        raise WorkflowError(f"--model or VIDEO_MODEL is required for the {backend} adapter")
    wan_repo = _resolve_optional_path(getattr(args, "wan_repo", None) or (
        os.environ.get("WAN22_REPO") if backend == "wan22" else None))
    checkpoint_env = "WAN22_TI2V_5B_DIR" if variant == "ti2v-5b" else "WAN22_I2V_A14B_DIR"
    wan_checkpoint = _resolve_optional_path(getattr(args, "wan_checkpoint", None) or (
        os.environ.get(checkpoint_env) if backend == "wan22" else None))
    world_model_root: Path | None = None
    world_model_final_root: Path | None = None
    world_model_task: Path | None = None
    world_model_case_id: str | None = None
    wm_intrinsics_gpu: str | None = None
    if backend == "local-wm":
        if scene_dir is None:
            raise WorkflowError("--backend local-wm requires --scene-dir")
        repository_root = Path(__file__).resolve().parents[3]
        world_model_final_root = _resolve_optional_path(
            getattr(args, "world_model_final_root", None)
            or repository_root / "dataset/final"
        )
        try:
            world_model_case_id = scene_dir.resolve().relative_to(
                world_model_final_root
            ).as_posix()
        except ValueError as exc:
            raise WorkflowError(
                f"local-wm scene must be below {world_model_final_root}: {scene_dir}"
            ) from exc
        world_model_root = _resolve_optional_path(
            getattr(args, "world_model_root", None)
            or os.environ.get("WORLD_MODEL_DATA_ROOT")
            or repository_root / "dataset/world_model_10s_v1"
        )
        report_path = world_model_root / "report.json"
        report = _load_json_object(report_path, "world-model report")
        matches = [
            item for item in report.get("cases", [])
            if isinstance(item, dict) and item.get("case_id") == world_model_case_id
        ]
        if len(matches) != 1:
            raise WorkflowError(
                f"world-model case must match exactly one report entry: {world_model_case_id}"
            )
        world_model_task = world_model_root / str(matches[0]["output_dir"]) / "task.json"
        if not world_model_task.is_file():
            raise WorkflowError(f"world-model task does not exist: {world_model_task}")
        wm_intrinsics_gpu = getattr(args, "wm_intrinsics_gpu", None)
        if model_value == "lingbot-world-2.0":
            gpu_ids = [
                item.strip() for item in
                (getattr(args, "cuda_visible_devices", None) or "3,4,5,6").split(",")
                if item.strip()
            ]
            if (
                len(gpu_ids) < 2
                or len(set(gpu_ids)) != len(gpu_ids)
                or any(not item.isdigit() for item in gpu_ids)
                or 40 % len(gpu_ids) != 0
            ):
                raise WorkflowError(
                    "lingbot-world-2.0 requires at least two distinct GPU indices, "
                    "and the GPU count must divide its 40 attention heads; for "
                    "example --cuda-visible-devices 4,5"
                )
            if wm_intrinsics_gpu is None:
                wm_intrinsics_gpu = gpu_ids[0]
    local_conda_env = getattr(args, "local_conda_env", None)
    if local_conda_env is None:
        local_conda_env = os.environ.get("VIDEO_LOCAL_CONDA_ENV", "wan22")
    if source_video is not None:
        duration = None
        size = None
        seed = None
    else:
        duration_arg = getattr(args, "duration", None)
        shared = conditions.get("shared_memory")
        shared_duration = (
            shared.get("duration_seconds") if isinstance(shared, dict) else None
        )
        duration = float(duration_arg if duration_arg is not None else shared_duration or 10)
        if duration <= 0:
            raise WorkflowError("duration must be positive")

        if local_profile is not None:
            try:
                supported_durations = [float(value) for value in local_profile["durations"]]
                default_duration, default_size = resolve_local_defaults(local_profile)
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise WorkflowError(f"invalid duration profile for {model_value}") from exc
            if duration not in supported_durations:
                if duration_arg is None and default_duration in supported_durations:
                    duration = default_duration
                else:
                    raise WorkflowError(
                        f"duration {duration:g}s is not supported by {model_value}; "
                        f"choose from {supported_durations}"
                    )

        requested_size = getattr(args, "size", None)
        if requested_size:
            size = requested_size
        elif local_profile is not None:
            size = str(default_size)
            if not size:
                raise WorkflowError(f"local video model profile has no default_resolution: {model}")
        else:
            size = _image_size_label(image)
        if local_profile is not None:
            supported_sizes = [str(value) for value in local_profile.get("resolutions", [])]
            if size not in supported_sizes:
                raise WorkflowError(
                    f"resolution {size!r} is not supported by {model}; choose from {supported_sizes}"
                )
        seed = int(getattr(args, "seed", 0))
        if backend == "fal":
            duration, size = resolve_preferences(model, duration_arg, requested_size)
    image_url = getattr(args, "image_url", None) or (
        os.environ.get("VIDEO_IMAGE_URL") if backend in REMOTE_BACKENDS else None)
    if image_url:
        parsed = urlparse(str(image_url))
        if parsed.scheme != "https" or not parsed.netloc:
            raise WorkflowError("--image-url must be a public HTTPS URL")
        image_url = str(image_url)

    for name in (
        "request_timeout",
        "generation_timeout",
        "local_timeout",
    ):
        if float(getattr(args, name)) <= 0:
            raise WorkflowError(f"--{name.replace('_', '-')} must be positive")
    if float(getattr(args, "poll_interval")) < 0:
        raise WorkflowError("--poll-interval cannot be negative")

    image_digest = sha256_file(image)
    reference_digest = sha256_file(reference)
    prompt_digest = canonical_hash(prompt) if prompt is not None else None
    source_video_digest = sha256_file(source_video) if source_video else None
    if source_video is not None:
        fingerprint_value: dict[str, Any] = {
            "video_source": "existing",
            "video_sha256": source_video_digest,
            "model": model,
            "image_sha256": image_digest,
            "reference_sha256": reference_digest,
        }
    else:
        fingerprint_value = {
            "image_sha256": image_digest,
            "reference_sha256": reference_digest,
            "prompt_sha256": prompt_digest,
            "backend": backend,
            "model": model,
            "duration": duration,
            "size": size,
            "seed": seed,
        }
        default_base = DEFAULT_ARK_BASE_URL if backend == "ark" else DEFAULT_API_BASE_URL
        if api_base_url and api_base_url != default_base:
            fingerprint_value["api_base_url"] = api_base_url
        if backend in LOCAL_BACKENDS:
            fingerprint_value["local_video"] = {
                "variant": variant,
                "frame_num": getattr(args, "frame_num", None),
                "wan_repo": str(wan_repo or ""),
                "wan_checkpoint": str(wan_checkpoint or ""),
                "model_profile": local_profile,
            }
            if backend == "local-wm":
                fingerprint_value["local_video"].update({
                    "case_id": world_model_case_id,
                    "task_sha256": sha256_file(world_model_task),
                    "world_model_root": str(world_model_root),
                })
    generation_fingerprint = canonical_hash(fingerprint_value)
    run_id = _safe_label(
        getattr(args, "run_id", None) or f"run-{generation_fingerprint[:12]}"
    )
    if not run_id:
        raise WorkflowError("run id has no safe characters")
    output_root = _resolve_optional_path(getattr(args, "output_root", None)) or (
        Path(__file__).resolve().parents[3] / "eval_result"
    )
    output_raw = getattr(args, "output_dir", None)
    # Keep model first so a model's runs can be inspected together.  The case
    # id is the second component and is stable across dataset directory moves.
    output_dir = _resolve_optional_path(output_raw) or (
        output_root / _safe_label(model) / _safe_label(scene_id) / run_id
    )

    vision_config = _resolve_optional_path(getattr(args, "vision_config", None))
    if vision_config is None and scene_dir is not None:
        candidate = scene_dir / "artifacts" / "vision_config.json"
        vision_config = candidate if candidate.is_file() else None
    v2_config = _resolve_optional_path(getattr(args, "v2_config", None))
    geometry_checkpoint = _resolve_optional_path(getattr(args, "checkpoint", None))
    for label, path in (
        ("vision config", vision_config),
        ("V2 config", v2_config),
        ("geometry checkpoint", geometry_checkpoint),
    ):
        if path is not None and not path.exists():
            raise WorkflowError(f"{label} does not exist: {path}")

    s3_config: S3PublishConfig | None = None
    if backend in REMOTE_BACKENDS and not image_url:
        try:
            s3_config = S3PublishConfig.from_env_and_overrides(
                bucket=getattr(args, "s3_bucket", None),
                endpoint_url=getattr(args, "s3_endpoint_url", None),
                region=getattr(args, "s3_region", None),
                prefix=getattr(args, "s3_prefix", None),
                url_ttl_seconds=getattr(args, "s3_url_ttl", None),
                public_base_url=getattr(args, "s3_public_base_url", None),
                request_timeout=float(getattr(args, "request_timeout")),
            )
        except ValueError as exc:
            raise WorkflowError(str(exc)) from exc

    return RunConfig(
        scene_dir=scene_dir,
        image=image,
        reference=reference,
        source_video=source_video,
        source_video_sha256=source_video_digest,
        prompt=prompt,
        scene_id=scene_id,
        backend=backend,
        model=model,
        variant=variant,
        duration=duration,
        size=str(size) if size is not None else None,
        seed=seed,
        output_root=output_root,
        output_dir=output_dir,
        run_id=run_id,
        generation_fingerprint=generation_fingerprint,
        image_sha256=image_digest,
        reference_sha256=reference_digest,
        prompt_sha256=prompt_digest,
        image_url=image_url,
        s3_config=s3_config,
        cleanup_upload=bool(getattr(args, "cleanup_upload", False)),
        resume=bool(getattr(args, "resume", False)),
        dry_run=bool(getattr(args, "dry_run", False)),
        vision_config=vision_config,
        v2_config=v2_config,
        geometry_backend=getattr(args, "geometry_backend", None),
        geometry_checkpoint=geometry_checkpoint,
        track=str(getattr(args, "track", None) or "static_3d"),
        strict_models=bool(getattr(args, "strict_models", True)),
        strict_geometry=bool(getattr(args, "strict_geometry", True)),
        vision_conda_env=getattr(args, "vision_conda_env", None),
        local_conda_env=str(local_conda_env),
        cuda_visible_devices=getattr(args, "cuda_visible_devices", None),
        frame_num=getattr(args, "frame_num", None),
        wan_repo=wan_repo,
        wan_checkpoint=wan_checkpoint,
        world_model_root=world_model_root,
        world_model_final_root=world_model_final_root,
        world_model_task=world_model_task,
        world_model_case_id=world_model_case_id,
        wm_intrinsics_gpu=wm_intrinsics_gpu,
        request_timeout=float(getattr(args, "request_timeout")),
        poll_interval=float(getattr(args, "poll_interval")),
        generation_timeout=float(getattr(args, "generation_timeout")),
        local_timeout=float(getattr(args, "local_timeout")),
        ffprobe=str(getattr(args, "ffprobe")),
        api_base_url=api_base_url,
        video_model_profile=local_profile,
    )


class BenchmarkOrchestrator:
    def __init__(
        self,
        *,
        backend_factory: Callable[[RunConfig], VideoBackend] | None = None,
        publisher_factory: Callable[[S3PublishConfig], S3ImagePublisher] | None = None,
        probe_func: Callable[..., VideoMetadata] = probe_video,
        stage_runner: Callable[[list[str], Path, dict[str, str]], None] | None = None,
        url_validator: Callable[[str, float], None] = validate_direct_image_url,
    ) -> None:
        self.backend_factory = backend_factory or self._default_backend
        self.publisher_factory = publisher_factory or S3ImagePublisher
        self.probe_func = probe_func
        self.stage_runner = stage_runner or _default_stage_runner
        self.url_validator = url_validator

    def run(
        self,
        config: RunConfig,
        *,
        stop_after_observation: bool = False,
        stop_after_generation: bool = False,
    ) -> Path:
        if config.dry_run:
            plan = config.manifest_config()
            if stop_after_generation:
                plan["stop_after_generation"] = True
            if stop_after_observation:
                plan["evaluation_pipeline_version"] = V3_WORKFLOW_CONTRACT_VERSION
            if config.backend == "openrouter":
                if _video_api_key(config.backend):
                    backend = self.backend_factory(config)
                    validator = getattr(backend, "validate_request", None)
                    if not callable(validator):
                        raise WorkflowError(
                            "OpenRouter backend does not expose capability validation"
                        )
                    request = self._generation_request(
                        config, config.output_dir / "generation" / "generated.mp4"
                    )
                    request.image_url = config.image_url or "https://publication-deferred.invalid/input"
                    validator(request)
                    plan["openrouter_capability_check"] = "passed"
                else:
                    plan["openrouter_capability_check"] = "skipped_no_api_key"
            elif config.backend == "ark":
                backend = self.backend_factory(config)
                validator = getattr(backend, "validate_request", None)
                if not callable(validator):
                    raise WorkflowError(
                        "Ark backend does not expose request validation"
                    )
                request = self._generation_request(
                    config, config.output_dir / "generation" / "generated.mp4"
                )
                request.image_url = (
                    config.image_url
                    or "https://publication-deferred.invalid/input"
                )
                validator(request)
                plan["ark_request_validation"] = "passed"
                plan["ark_api_key_configured"] = bool(
                    _video_api_key(config.backend)
                )
            print(json.dumps(plan, indent=2, sort_keys=True))
            return config.output_dir / "run_manifest.json"

        manifest_path = config.output_dir / "run_manifest.json"
        manifest = self._load_manifest(config, manifest_path)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        directories = ("generation", "observation", "logs")
        if not stop_after_observation:
            directories += ("geometry", "eval_v2")
        for directory in directories:
            (config.output_dir / directory).mkdir(parents=True, exist_ok=True)
        resolved_outputs = [config.image, config.reference]
        if config.prompt is not None:
            prompt_path = config.output_dir / "prompt.txt"
            prompt_path.write_text(config.prompt.rstrip() + "\n", encoding="utf-8")
            resolved_outputs.append(prompt_path)
        self._complete_inline_stage(
            manifest,
            manifest_path,
            "resolve_inputs",
            canonical_hash(config.manifest_config()),
            resolved_outputs,
        )

        video_path = config.output_dir / "generation" / "generated.mp4"
        generation_hash = config.generation_fingerprint
        published: PublishedImage | None = None
        if config.source_video is not None:
            self._complete_inline_stage(
                manifest,
                manifest_path,
                "publish_image",
                canonical_hash({"video_source": "existing"}),
                [],
                metadata={"skipped": True, "reason": "existing video"},
            )
            import_reusable = self._stage_reusable(
                manifest, "generate_video", generation_hash, [video_path]
            ) and sha256_file(video_path) == config.source_video_sha256
            if not import_reusable:
                self._start_stage(
                    manifest, manifest_path, "generate_video", generation_hash
                )
                try:
                    _atomic_copy_file(config.source_video, video_path)
                    result = GenerationResult(
                        video_path=str(video_path),
                        model_id=config.model,
                        effective_parameters={
                            "source": "existing",
                            "source_video_sha256": config.source_video_sha256,
                        },
                    )
                    manifest.generation_request = None
                    manifest.generation_result = result
                    self._finish_stage(
                        manifest,
                        manifest_path,
                        "generate_video",
                        [video_path],
                        metadata={
                            "source": "existing",
                            "model_id": config.model,
                            "source_video_sha256": config.source_video_sha256,
                            "provider_job_id": None,
                            "cost": None,
                        },
                    )
                except Exception as exc:
                    self._fail_stage(
                        manifest, manifest_path, "generate_video", exc
                    )
                    raise WorkflowError(str(exc)) from exc
            elif manifest.generation_result is None:
                raise WorkflowError(
                    "reusable generation stage has no generation result"
                )
        else:
            request = self._generation_request(config, video_path)
            atomic_write_json(
                config.output_dir / "generation" / "request.json",
                request.to_dict(),
            )
            manifest.generation_request = request
            manifest.save(manifest_path)

            if not self._stage_reusable(
                manifest, "generate_video", generation_hash, [video_path]
            ):
                try:
                    backend = self.backend_factory(config)
                    image_url, published = self._resolve_image_url(
                        config, manifest, manifest_path
                    )
                    request.image_url = image_url
                    provider_path = (
                        config.output_dir / "generation" / "provider_state.json"
                    )
                    resume_state = None
                    if config.resume and provider_path.is_file():
                        resume_state = _load_json_object(
                            provider_path, "provider state"
                        )
                    self._start_stage(
                        manifest,
                        manifest_path,
                        "generate_video",
                        generation_hash,
                    )

                    def persist_provider_state(state: dict[str, Any]) -> None:
                        atomic_write_json(provider_path, state)
                        manifest.stages["generate_video"].metadata[
                            "provider_job_id"
                        ] = state.get("id")
                        manifest.save(manifest_path)

                    result = backend.generate(
                        request,
                        resume_state=resume_state,
                        on_state=persist_provider_state,
                    )
                    if published is not None:
                        result.transport = published.to_manifest_dict()
                    manifest.generation_result = result
                    self._finish_stage(
                        manifest,
                        manifest_path,
                        "generate_video",
                        [video_path],
                        metadata={
                            "model_id": result.model_id,
                            "provider_job_id": result.provider_job_id,
                            "cost": result.cost,
                        },
                    )
                except SubmissionUnknownError as exc:
                    failed_stage = _active_generation_stage(manifest)
                    self._fail_stage(
                        manifest,
                        manifest_path,
                        failed_stage,
                        exc,
                        status="submission_unknown"
                        if failed_stage == "generate_video"
                        else "failed",
                        secrets=[request.image_url or ""],
                    )
                    raise WorkflowError(str(exc)) from exc
                except Exception as exc:
                    failed_stage = _active_generation_stage(manifest)
                    self._fail_stage(
                        manifest,
                        manifest_path,
                        failed_stage,
                        exc,
                        secrets=[request.image_url or ""],
                    )
                    raise WorkflowError(str(exc)) from exc
            elif manifest.generation_result is None:
                raise WorkflowError(
                    "reusable generation stage has no generation result"
                )

        validation_hash = canonical_hash(
            {"video_sha256": sha256_file(video_path), "ffprobe": config.ffprobe,
             "aspect_policy": "fal-source-ratio-v1" if config.backend == "fal" else None,
             "reference_image_sha256": config.image_sha256}
        )
        if not self._stage_reusable(
            manifest, "validate_video", validation_hash, [video_path]
        ):
            self._start_stage(manifest, manifest_path, "validate_video", validation_hash)
            try:
                metadata = self.probe_func(video_path, ffprobe=config.ffprobe)
                if config.backend == "fal" or config.backend in LOCAL_BACKENDS:
                    reference_size = tuple(int(v) for v in _image_size_label(config.image).split("x"))
                    validate_aspect_ratio(reference_size, (metadata.width, metadata.height))
                if config.backend in {"local_i2v", "local-wm"} and config.duration is not None:
                    validate_duration(config.duration, metadata.duration_seconds)
            except Exception as exc:
                self._fail_stage(manifest, manifest_path, "validate_video", exc)
                raise WorkflowError(str(exc)) from exc
            manifest.video_metadata = metadata.to_dict()
            self._finish_stage(
                manifest,
                manifest_path,
                "validate_video",
                [video_path],
                metadata=metadata.to_dict(),
            )

        if stop_after_generation:
            manifest.status = "running"
            manifest.save(manifest_path)
            return manifest_path

        if config.cleanup_upload and published is not None:
            try:
                if config.s3_config is None:
                    raise WorkflowError("published S3 image has no storage configuration")
                cleaned = self.publisher_factory(config.s3_config).cleanup(published)
                manifest.stages["publish_image"].metadata["cleaned_up"] = cleaned
                manifest.save(manifest_path)
            except Exception as exc:
                manifest.stages["publish_image"].metadata["cleanup_error"] = sanitize_text(str(exc))
                manifest.save(manifest_path)

        observation = config.output_dir / "observation" / "video_observation.json"
        observation_vision_config = _effective_observation_vision_config(config)
        observation_command = self._observe_command(
            config, video_path, observation, vision_config=observation_vision_config
        )
        observation_hash = canonical_hash(
            {
                "pipeline_version": OBSERVATION_PIPELINE_VERSION,
                "video": sha256_file(video_path),
                "reference": config.reference_sha256,
                "vision_config": _optional_file_hash(observation_vision_config),
                "strict_models": config.strict_models,
            }
        )
        self._run_command_stage(
            manifest,
            manifest_path,
            config,
            "observe_video",
            observation_hash,
            observation_command,
            config.output_dir / "logs" / "observe_video.log",
            [observation],
        )

        if stop_after_observation:
            manifest.status = "running"
            manifest.run_config["evaluation_pipeline_version"] = V3_WORKFLOW_CONTRACT_VERSION
            manifest.save(manifest_path)
            return manifest_path

        geometry_manifest = config.output_dir / "geometry" / "geometry_manifest.json"
        geometry_command = self._geometry_command(
            config, video_path, observation, geometry_manifest.parent
        )
        geometry_hash = canonical_hash(
            {
                "pipeline_version": GEOMETRY_PIPELINE_VERSION,
                "observation": sha256_file(observation),
                "reference": config.reference_sha256,
                "image": config.image_sha256,
                "v2_config": _optional_file_hash(config.v2_config),
                "backend": config.geometry_backend,
                "checkpoint": _optional_file_hash(config.geometry_checkpoint),
                "strict_geometry": config.strict_geometry,
            }
        )
        self._run_command_stage(
            manifest,
            manifest_path,
            config,
            "prepare_geometry",
            geometry_hash,
            geometry_command,
            config.output_dir / "logs" / "prepare_geometry.log",
            [geometry_manifest],
        )

        summary = config.output_dir / "eval_v2" / "summary_v2.json"
        eval_command = self._eval_command(
            config, observation, geometry_manifest, summary.parent
        )
        eval_hash = canonical_hash(
            {
                "pipeline_version": EVALUATION_PIPELINE_VERSION,
                "observation": sha256_file(observation),
                "geometry": sha256_file(geometry_manifest),
                "reference": config.reference_sha256,
                "v2_config": _optional_file_hash(config.v2_config),
                "track": config.track,
            }
        )
        self._run_command_stage(
            manifest,
            manifest_path,
            config,
            "evaluate_v2",
            eval_hash,
            eval_command,
            config.output_dir / "logs" / "eval_v2.log",
            [summary],
        )
        manifest.result_summary = _load_json_object(summary, "V2 summary")
        self._complete_inline_stage(
            manifest,
            manifest_path,
            "summarize",
            canonical_hash(
                {
                    "summary": sha256_file(summary),
                    "manifest_schema": manifest.schema_version,
                }
            ),
            [summary],
        )
        manifest.status = "completed"
        manifest.save(manifest_path)
        return manifest_path

    def _generation_request(
        self, config: RunConfig, video_path: Path
    ) -> GenerationRequest:
        return GenerationRequest(
            backend=config.backend,
            model=config.model,
            image_path=str(config.image),
            image_sha256=config.image_sha256,
            reference_path=str(config.reference),
            reference_sha256=config.reference_sha256,
            prompt_sha256=config.prompt_sha256,
            prompt=config.prompt,
            duration=config.duration,
            size=config.size,
            seed=config.seed,
            output_path=str(video_path),
            options=self._generation_options(config),
        )

    def _load_manifest(self, config: RunConfig, path: Path) -> RunManifest:
        if path.exists():
            if not config.resume:
                raise WorkflowError(
                    f"run already exists at {config.output_dir}; pass --resume to reuse it"
                )
            manifest = RunManifest.load(path)
            old_fingerprint = manifest.run_config.get("generation_fingerprint")
            if old_fingerprint != config.generation_fingerprint:
                raise WorkflowError(
                    "existing run generation fingerprint differs; choose another "
                    "--output-dir or --run-id"
                )
            manifest.run_config = config.manifest_config()
            return manifest
        if config.output_dir.exists() and any(config.output_dir.iterdir()):
            raise WorkflowError(
                f"output directory is not empty and has no manifest: {config.output_dir}"
            )
        return RunManifest.new(
            run_id=config.run_id,
            scene_id=config.scene_id,
            run_config=config.manifest_config(),
        )

    def _resolve_image_url(
        self,
        config: RunConfig,
        manifest: RunManifest,
        manifest_path: Path,
    ) -> tuple[str | None, PublishedImage | None]:
        if config.backend not in REMOTE_BACKENDS:
            self._complete_inline_stage(
                manifest,
                manifest_path,
                "publish_image",
                canonical_hash({"backend": config.backend}),
                [],
                metadata={"skipped": True, "reason": "local backend"},
            )
            return None, None
        publish_hash = canonical_hash(
            {
                "image": config.image_sha256,
                "explicit": bool(config.image_url),
                "storage": config.manifest_config().get("storage"),
            }
        )
        self._start_stage(manifest, manifest_path, "publish_image", publish_hash)
        if config.image_url:
            self.url_validator(config.image_url, config.request_timeout)
            metadata = {
                "url_kind": "explicit",
                "endpoint_host": urlparse(config.image_url).hostname,
            }
            self._finish_stage(
                manifest, manifest_path, "publish_image", [], metadata=metadata
            )
            return config.image_url, None
        if config.s3_config is None:
            raise WorkflowError(
                f"S3 configuration is required for {config.backend}"
            )
        publisher = self.publisher_factory(config.s3_config)
        published = publisher.publish(config.image)
        self._finish_stage(
            manifest,
            manifest_path,
            "publish_image",
            [],
            metadata=published.to_manifest_dict(),
        )
        return published.url, published

    def _generation_options(self, config: RunConfig) -> dict[str, Any]:
        if config.backend not in LOCAL_BACKENDS:
            return {}
        options = {
            "variant": config.variant,
            "frame_num": config.frame_num,
            "local_conda_env": config.local_conda_env,
            "cuda_visible_devices": config.cuda_visible_devices,
            "wan_repo": str(config.wan_repo) if config.wan_repo else None,
            "ckpt_dir": str(config.wan_checkpoint) if config.wan_checkpoint else None,
            "timeout": config.local_timeout,
            "log_path": str(config.output_dir / "logs" / "generate_video.log"),
        }
        if config.backend == "local-wm":
            workspace = config.output_dir / "generation" / "world_model"
            options.update({
                "case_id": config.world_model_case_id,
                "task_sha256": sha256_file(config.world_model_task),
                "world_model_root": str(config.world_model_root),
                "final_root": str(config.world_model_final_root),
                "prepared_root": str(workspace / "prepared"),
                "result_root": str(workspace / "results"),
                "intrinsics_gpu": config.wm_intrinsics_gpu,
            })
        return options

    def _command_prefix(self, config: RunConfig) -> list[str]:
        if config.vision_conda_env:
            return [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                config.vision_conda_env,
                "python",
                "-m",
                "multimem_bench.cli",
            ]
        return [sys.executable, "-m", "multimem_bench.cli"]

    def _observe_command(
        self,
        config: RunConfig,
        video: Path,
        output: Path,
        *,
        vision_config: Path | None = None,
    ) -> list[str]:
        command = self._command_prefix(config) + [
            "observe-video",
            "--video",
            str(video),
            "--reference",
            str(config.reference),
            "--output",
            str(output),
            "--video-id",
            f"{config.model}:{config.run_id}",
            "--assets-dir",
            str(output.parent / "assets"),
        ]
        selected_vision_config = vision_config or config.vision_config
        if selected_vision_config:
            command.extend(["--vision-config", str(selected_vision_config)])
        if config.strict_models:
            command.append("--strict-models")
        return command

    def _geometry_command(
        self,
        config: RunConfig,
        video: Path,
        observation: Path,
        output: Path,
    ) -> list[str]:
        command = self._command_prefix(config) + [
            "prepare-geometry",
            "--reference-image",
            str(config.image),
            "--reference",
            str(config.reference),
            "--observations",
            str(observation),
            "--video",
            str(video),
            "--output",
            str(output),
        ]
        if config.v2_config:
            command.extend(["--config", str(config.v2_config)])
        if config.geometry_backend:
            command.extend(["--backend", config.geometry_backend])
        if config.geometry_checkpoint:
            command.extend(["--checkpoint", str(config.geometry_checkpoint)])
        if config.strict_geometry:
            command.append("--strict-geometry")
        return command

    def _eval_command(
        self,
        config: RunConfig,
        observation: Path,
        geometry: Path,
        output: Path,
    ) -> list[str]:
        command = self._command_prefix(config) + [
            "eval-v2",
            "--reference",
            str(config.reference),
            "--observations",
            str(observation),
            "--geometry",
            str(geometry),
            "--output",
            str(output),
        ]
        if config.v2_config:
            command.extend(["--config", str(config.v2_config)])
        if config.track:
            command.extend(["--track", config.track])
        return command

    def _run_command_stage(
        self,
        manifest: RunManifest,
        manifest_path: Path,
        config: RunConfig,
        name: str,
        input_hash: str,
        command: list[str],
        log_path: Path,
        outputs: list[Path],
    ) -> None:
        if self._stage_reusable(manifest, name, input_hash, outputs):
            return
        self._start_stage(manifest, manifest_path, name, input_hash)
        try:
            stage_environment = _evaluation_environment(
                config.cuda_visible_devices
            )
            self.stage_runner(
                command,
                log_path,
                stage_environment,
            )
            _require_outputs(outputs, name)
        except Exception as exc:
            self._fail_stage(manifest, manifest_path, name, exc)
            raise WorkflowError(f"{name} failed: {exc}") from exc
        self._finish_stage(
            manifest,
            manifest_path,
            name,
            outputs,
            metadata={
                "cuda_visible_devices": stage_environment.get(
                    "CUDA_VISIBLE_DEVICES"
                )
            },
        )

    @staticmethod
    def _stage_reusable(
        manifest: RunManifest, name: str, input_hash: str, outputs: list[Path]
    ) -> bool:
        stage = manifest.stages.get(name)
        return bool(
            stage
            and stage.status == "completed"
            and stage.input_hash == input_hash
            and _outputs_valid(outputs)
        )

    @staticmethod
    def _start_stage(
        manifest: RunManifest, path: Path, name: str, input_hash: str
    ) -> None:
        manifest.status = "running"
        manifest.stages[name] = StageRecord(
            name=name,
            status="running",
            input_hash=input_hash,
            started_at=utc_now(),
        )
        manifest.save(path)
        print(f"[STAGE START] {name}", flush=True)

    @staticmethod
    def _finish_stage(
        manifest: RunManifest,
        path: Path,
        name: str,
        outputs: list[Path],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        stage = manifest.stages[name]
        stage.status = "completed"
        stage.finished_at = utc_now()
        stage.outputs = [str(item) for item in outputs]
        if metadata:
            stage.metadata.update(metadata)
        stage.error = None
        manifest.save(path)
        print(f"[STAGE DONE] {name}: {', '.join(str(item) for item in outputs) or 'no outputs'}", flush=True)

    def _complete_inline_stage(
        self,
        manifest: RunManifest,
        path: Path,
        name: str,
        input_hash: str,
        outputs: list[Path],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self._stage_reusable(manifest, name, input_hash, outputs):
            print(f"[STAGE REUSE] {name}", flush=True)
            return
        self._start_stage(manifest, path, name, input_hash)
        _require_outputs(outputs, name)
        self._finish_stage(manifest, path, name, outputs, metadata=metadata)

    @staticmethod
    def _fail_stage(
        manifest: RunManifest,
        path: Path,
        name: str,
        error: Exception,
        *,
        status: str = "failed",
        secrets: list[str] | None = None,
    ) -> None:
        stage = manifest.stages.get(name) or StageRecord(name=name)
        stage.status = status
        stage.finished_at = utc_now()
        stage.error = sanitize_text(str(error), [*_generation_secrets(), *(secrets or [])])
        manifest.stages[name] = stage
        manifest.status = status
        manifest.save(path)
        print(f"[STAGE {status.upper()}] {name}: {stage.error}", file=sys.stderr, flush=True)

    @staticmethod
    def _default_backend(config: RunConfig) -> VideoBackend:
        if config.backend == "openrouter":
            api_key = _video_api_key(config.backend)
            if not api_key:
                raise WorkflowError("VIDEO_API_KEY (or OPENROUTER_API_KEY) is not set")
            return OpenRouterBackend(
                OpenRouterVideoClient(
                    api_key=api_key,
                    base_url=config.api_base_url or DEFAULT_API_BASE_URL,
                    request_timeout=config.request_timeout,
                    poll_interval=config.poll_interval,
                    generation_timeout=config.generation_timeout,
                )
            )
        if config.backend == "ark":
            api_key = _video_api_key(config.backend)
            if not api_key and not config.dry_run:
                raise WorkflowError("VIDEO_API_KEY (or ARK_API_KEY) is not set")
            return ArkBackend(
                ArkVideoClient(
                    api_key=api_key,
                    base_url=config.api_base_url or DEFAULT_ARK_BASE_URL,
                    request_timeout=config.request_timeout,
                    poll_interval=config.poll_interval,
                    generation_timeout=config.generation_timeout,
                )
            )
        if config.backend == "wan22":
            return Wan22Backend()
        if config.backend == "local_i2v":
            return LocalI2VBackend()
        if config.backend == "local-wm":
            return LocalWorldModelBackend()
        if config.backend == "fal":
            key = _video_api_key(config.backend)
            if not key and not config.dry_run:
                raise WorkflowError("VIDEO_API_KEY (or FAL_KEY) is not set")
            return FalBackend(key)
        raise WorkflowError(f"unsupported backend: {config.backend}")


def run_benchmark_command(args: argparse.Namespace, *, v3: bool = False) -> int:
    try:
        manifest = getattr(args, "manifest", None)
        if manifest:
            if v3 and getattr(args, "output_dir", None):
                raise WorkflowError("batch V3 runs require --output-root, not a shared --output-dir")
            payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
            cases = payload.get("cases") if isinstance(payload, dict) else payload
            if not isinstance(cases, list) or not cases:
                raise WorkflowError("--manifest must contain a non-empty cases list")
            failures = 0
            if args.parallelism < 1:
                raise WorkflowError("--parallelism must be positive")
            devices = [x.strip() for x in (args.cuda_devices or "").split(",") if x.strip()]
            batch_backend = _generation_backend(args)
            grouped_world_model = _local_wm_uses_gpu_group(args)
            if grouped_world_model and args.parallelism != 1:
                raise WorkflowError(
                    "lingbot-world-2.0 batch runs require --parallelism 1"
                )
            if batch_backend in LOCAL_BACKENDS and args.parallelism > 1 and not devices:
                raise WorkflowError("local batch parallelism requires --cuda-devices, e.g. 0,1")
            two_phase_local_wm = v3 and batch_backend == "local-wm"

            def run_case(item_index_item, *, generation_only: bool = False):
                index, item = item_index_item
                if not isinstance(item, dict) or not item.get("path"):
                    raise WorkflowError("manifest case must contain path")
                case_args = argparse.Namespace(**vars(args))
                case_args.manifest = None
                case_args.scene_dir = Path("dataset/final") / str(item["path"])
                # Manifest case ids are the canonical output key.  Dataset
                # metadata may contain a different scene_id, and older cases
                # without conditions would otherwise fall back to ``raw``.
                if item.get("case_id"):
                    case_args.scene_id = str(item["case_id"])
                if devices and grouped_world_model:
                    case_args.cuda_visible_devices = ",".join(devices)
                elif devices and batch_backend in LOCAL_BACKENDS:
                    case_args.cuda_visible_devices = devices[index % len(devices)]
                case_args._generation_only = generation_only
                if two_phase_local_wm and not generation_only:
                    case_args.resume = True
                phase = "GENERATE" if generation_only else "EVAL" if two_phase_local_wm else "CASE"
                print(f"[BATCH {phase}] {item.get('case_id', case_args.scene_dir.name)}", flush=True)
                if v3:
                    if run_benchmark_v3_command(case_args) != 0:
                        raise WorkflowError(f"V3 case failed: {case_args.scene_dir}")
                    print(f"[BATCH {phase} DONE] {case_args.scene_dir}", flush=True)
                else:
                    config = resolve_run_config(case_args)
                    path = BenchmarkOrchestrator().run(config)
                    print(f"[BATCH DONE] {config.scene_id}: {path}", flush=True)
                return item_index_item

            def run_jobs(selected_jobs, *, generation_only: bool = False):
                completed = []
                phase_failures = 0
                with ThreadPoolExecutor(max_workers=args.parallelism) as pool:
                    futures = [
                        pool.submit(run_case, job, generation_only=generation_only)
                        for job in selected_jobs
                    ]
                    for future in as_completed(futures):
                        try:
                            completed.append(future.result())
                        except (OSError, ValueError, WorkflowError) as exc:
                            phase_failures += 1
                            print(
                                f"[BATCH FAIL] {sanitize_text(str(exc), _generation_secrets())}",
                                file=sys.stderr,
                                flush=True,
                            )
                return completed, phase_failures

            jobs = list(enumerate(cases))
            if two_phase_local_wm:
                print("[BATCH PHASE] generate all videos", flush=True)
                generated_jobs, generation_failures = run_jobs(jobs, generation_only=True)
                failures += generation_failures
                print(
                    f"[BATCH PHASE DONE] generated {len(generated_jobs)}/{len(jobs)} videos; "
                    "starting unified V3 evaluation",
                    flush=True,
                )
                _, evaluation_failures = run_jobs(generated_jobs)
                failures += evaluation_failures
            else:
                _, failures = run_jobs(jobs)
            return 1 if failures else 0
        config = resolve_run_config(args)
        manifest_path = BenchmarkOrchestrator().run(config)
        if config.dry_run:
            print(f"dry_run_manifest: {manifest_path}")
        else:
            print(f"run_manifest: {manifest_path}")
        return 0
    except (OSError, ValueError, WorkflowError) as exc:
        print(f"Error: {sanitize_text(str(exc), _generation_secrets())}", file=sys.stderr)
        return 2


def run_benchmark_v3_command(args: argparse.Namespace) -> int:
    """Run generation/observation directly into V3, or evaluate an existing run."""
    if getattr(args, "manifest", None):
        return run_benchmark_command(args, v3=True)
    try:
        scene_dir = _resolve_optional_path(getattr(args, "scene_dir", None))
        if scene_dir is not None and getattr(args, "video", None) is None:
            scene_cases = _discover_scene_cases(scene_dir)
            if len(scene_cases) > 1:
                return _run_scene_directory_batch(args, scene_cases)
            if len(scene_cases) == 1 and scene_cases[0] != scene_dir:
                return _run_scene_directory_batch(args, scene_cases)
        source_inputs = getattr(args, "scene_dir", None) or getattr(args, "image", None)
        if (getattr(args, "backend", None) or getattr(args, "video", None)
                or (source_inputs and os.environ.get("VIDEO_BACKEND"))):
            from multimem_bench.cli import _load_v3_config
            from multimem_bench.io import load_reference_scene

            # Validate evaluation inputs before submitting a paid generation job.
            config = resolve_run_config(args)
            next_args = argparse.Namespace(**{**vars(args), "output_dir": config.output_dir})
            _load_v3_config(_v3_config_path(next_args)).validate()
            structure_config, secret_env = _resolve_v3_structure_config(next_args)
            needs_vlm = structure_config is not None and any(
                obj.mobility.evaluation_track == "dynamic_identity" and
                (structure_config.pipeline != "hybrid" or obj.mobility.kinematic_class != "articulated"
                 or structure_config.articulated_fallback == "vlm")
                for obj in load_reference_scene(config.reference).objects.values())
            saved_manifest = config.output_dir / "run_manifest.json"
            reusable_video = config.resume and saved_manifest.is_file() and BenchmarkOrchestrator._stage_reusable(
                RunManifest.load(saved_manifest), "generate_video", config.generation_fingerprint,
                [config.output_dir / "generation" / "generated.mp4"])
            if needs_vlm and not config.dry_run and not reusable_video:
                if not structure_config.endpoint or not structure_config.model:
                    raise WorkflowError("structure evaluation requires a VLM endpoint and model")
                if not (secret_env.get(structure_config.api_key_env) or os.environ.get(structure_config.api_key_env)):
                    raise WorkflowError(f"set {structure_config.api_key_env} or pass --vlm-credentials")
            generation_only = bool(getattr(args, "_generation_only", False))
            if generation_only:
                manifest_path = BenchmarkOrchestrator().run(config, stop_after_generation=True)
            else:
                manifest_path = BenchmarkOrchestrator().run(config, stop_after_observation=True)
            if config.dry_run:
                if generation_only:
                    print(json.dumps({"v3_stages": ["generate_video", "validate_video"]}))
                    return 0
                print(json.dumps({"v3_stages": ["prepare-reference-v3", "prepare-geometry-v3"] +
                                 (["prepare-structure-v3"] if structure_config else []) + ["eval-v3"]}))
                return 0
            if generation_only:
                print(f"generation_manifest: {manifest_path}")
                return 0
            return _evaluate_v3_stage(next_args)
        if not getattr(args, "output_dir", None):
            raise WorkflowError("provide --scene-dir with VIDEO_BACKEND/--backend/--video, or --output-dir for an existing run")
        return _evaluate_v3_stage(args)
    except (OSError, ValueError, WorkflowError, subprocess.SubprocessError) as exc:
        print(f"Error: {sanitize_text(str(exc), _generation_secrets())}", file=sys.stderr)
        return 2


def _is_scene_case(path: Path) -> bool:
    """Return whether a directory has the inputs required for one benchmark case."""
    if not path.is_dir() or not (path / "raw.jpg").is_file():
        return False
    return (path / "reference_scene.json").is_file() or (
        path / "artifacts" / "reference_scene.json"
    ).is_file()


def _discover_scene_cases(scene_dir: Path) -> list[Path]:
    """Discover case directories below a dataset root, style root, or case root."""
    root = scene_dir.expanduser().resolve()
    if not root.is_dir():
        raise WorkflowError(f"scene directory does not exist: {root}")
    if _is_scene_case(root):
        return [root]
    cases = sorted(
        {path.parent.resolve() for path in root.rglob("raw.jpg") if _is_scene_case(path.parent)},
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not cases:
        raise WorkflowError(
            f"no benchmark cases found below {root}; each case needs raw.jpg and reference_scene.json"
        )
    return cases


def _run_scene_directory_batch(args: argparse.Namespace, cases: list[Path]) -> int:
    """Run every discovered case with independent output/configuration."""
    if getattr(args, "output_dir", None):
        raise WorkflowError("directory batch runs require --output-root, not a shared --output-dir")
    if getattr(args, "scene_id", None):
        raise WorkflowError("--scene-id is not valid when --scene-dir contains multiple cases")
    if getattr(args, "parallelism", 1) < 1:
        raise WorkflowError("--parallelism must be positive")
    devices = [item.strip() for item in (getattr(args, "cuda_devices", None) or "").split(",") if item.strip()]
    batch_backend = _generation_backend(args)
    grouped_world_model = _local_wm_uses_gpu_group(args)
    if grouped_world_model and args.parallelism != 1:
        raise WorkflowError("lingbot-world-2.0 batch runs require --parallelism 1")
    if batch_backend in LOCAL_BACKENDS and args.parallelism > 1 and not devices:
        raise WorkflowError("local batch parallelism requires --cuda-devices, e.g. 0,1")

    def run_case(index_and_path: tuple[int, Path]) -> None:
        index, case = index_and_path
        case_args = argparse.Namespace(**vars(args))
        case_args.manifest = None
        case_args.scene_dir = case
        case_args.output_dir = None
        if devices and grouped_world_model:
            case_args.cuda_visible_devices = ",".join(devices)
        elif devices and batch_backend in LOCAL_BACKENDS:
            case_args.cuda_visible_devices = devices[index % len(devices)]
        print(f"[BATCH CASE] {case}", flush=True)
        result = run_benchmark_v3_command(case_args)
        if result != 0:
            raise WorkflowError(f"V3 case failed: {case}")
        print(f"[BATCH DONE] {case}", flush=True)

    failures = 0
    with ThreadPoolExecutor(max_workers=args.parallelism) as pool:
        futures = [pool.submit(run_case, job) for job in enumerate(cases)]
        for future in as_completed(futures):
            try:
                future.result()
            except (OSError, ValueError, WorkflowError) as exc:
                failures += 1
                print(
                    f"[BATCH FAIL] {sanitize_text(str(exc), _generation_secrets())}",
                    file=sys.stderr,
                    flush=True,
                )
    return 1 if failures else 0


def _evaluate_v3_stage(args: argparse.Namespace) -> int:
    root = Path(args.output_dir).expanduser().resolve()
    path = root / "run_manifest.json"
    payload = _load_json_object(path, "run manifest") if path.is_file() else {}
    if getattr(args, "dry_run", False) or payload.get("schema_version") != "1.0":
        return _run_existing_benchmark_v3(args)
    if not args.resume and any((root / item).is_file() for item in (
        "eval_v3/summary_v3.json", "geometry_v3/geometry_v3.json")):
        return _run_existing_benchmark_v3(args)
    manifest = RunManifest.from_dict(payload)
    stage = StageRecord("evaluate_v3", status="running", started_at=utc_now())
    manifest.stages["evaluate_v3"] = stage
    manifest.status = "running"
    manifest.result_summary = None
    manifest.save(path)
    completed = False
    try:
        result = _run_existing_benchmark_v3(args)
        if result == 0:
            summary = root / "eval_v3" / "summary_v3.json"
            manifest.result_summary = _load_json_object(summary, "V3 summary")
            stage.outputs = [str(summary)]
            stage.input_hash = sha256_file(summary)
            manifest.run_config["evaluation_pipeline_version"] = V3_WORKFLOW_CONTRACT_VERSION
            completed = True
        return result
    finally:
        stage.finished_at = utc_now()
        stage.status = "completed" if completed else "failed"
        manifest.status = stage.status
        if not completed:
            stage.error = "V3 evaluation failed; see V3 stage logs"
            manifest.result_summary = None
        manifest.save(path)


def _v3_config_path(args: argparse.Namespace) -> Path | None:
    if getattr(args, "v3_config", None):
        return Path(args.v3_config).expanduser().resolve()
    if getattr(args, "resume", False) and getattr(args, "output_dir", None):
        cached = Path(args.output_dir).expanduser().resolve() / "v3_effective_config.json"
        if cached.is_file():
            return cached
    return None


def _resolve_v3_structure_config(args: argparse.Namespace) -> tuple[Any, dict[str, str]]:
    from multimem_bench.v3.structure import StructureConfig

    path = getattr(args, "structure_config", None)
    explicit_config = path is not None
    credentials = getattr(args, "vlm_credentials", None)
    if getattr(args, "no_structure", False):
        if path or credentials:
            raise WorkflowError("--no-structure conflicts with --structure-config/--vlm-credentials")
        return None, {}
    root = getattr(args, "output_dir", None)
    if path is None and getattr(args, "resume", False) and root:
        root = Path(root).expanduser().resolve()
        cached = root / "v3_effective_structure_config.json"
        manifest = root / "v3_manifest.json"
        enabled = (not manifest.is_file() or bool(_load_json_object(manifest, "V3 manifest").get("structure")))
        if enabled and cached.is_file():
            path = cached
    exported = {"endpoint": os.environ.get("VLM_BASE_URL", "").strip(),
                "model": os.environ.get("VLM_MODEL", "").strip()}
    if path is None and credentials is None and not any(exported.values()):
        return None, {}
    data = (_load_json_object(Path(path).expanduser().resolve(), "structure config") if path else
            {"pipeline": "hybrid", "articulated_fallback": "vlm"})
    if isinstance(data.get("articulated"), dict):
        from multimem_bench.paths import resolve_config_path

        articulated = dict(data["articulated"])
        for field in ("human_model", "animal_model"):
            if isinstance(articulated.get(field), str):
                articulated[field] = resolve_config_path(articulated[field])
        custom = articulated.get("custom_profiles")
        if isinstance(custom, dict):
            articulated["custom_profiles"] = {
                name: {
                    key: resolve_config_path(value) if isinstance(value, str) else value
                    for key, value in profile.items()
                }
                for name, profile in custom.items()
            }
        data["articulated"] = articulated
    provider = {name: value for name, value in exported.items() if value}
    key = None
    if credentials:
        values = _load_json_object(Path(credentials).expanduser().resolve(), "VLM credentials")
        for field in ("endpoint", "model", "api_key"):
            if not isinstance(values.get(field), str) or not values[field].strip():
                raise WorkflowError(f"VLM credentials require a non-empty {field}")
        provider.update(endpoint=values["endpoint"].strip(), model=values["model"].strip())
        key = values["api_key"]
    if provider.get("endpoint"):
        endpoint = provider["endpoint"].rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions" if endpoint.endswith("/v1") else "/v1/chat/completions"
        provider["endpoint"] = endpoint
    for field, value in provider.items():
        if not explicit_config or not data.get(field):
            data[field] = value
    if any(exported.values()) and (not data.get("endpoint") or not data.get("model")):
        raise WorkflowError("set both VLM_BASE_URL and VLM_MODEL, or provide endpoint/model in --structure-config")
    config = StructureConfig.from_dict(data)
    return config, {config.api_key_env: key} if key else {}


def _run_existing_benchmark_v3(args: argparse.Namespace) -> int:
    """Evaluate an existing completed run with the independent-frame V3 protocol."""
    try:
        root = Path(args.output_dir).expanduser().resolve()
        manifest_path = root / "run_manifest.json"
        if not manifest_path.is_file():
            raise WorkflowError(f"existing run_manifest.json not found: {manifest_path}")
        structure_config, secret_env = _resolve_v3_structure_config(args)
        structure_enabled = structure_config is not None
        structure_path = root / "structure_v3" / "structure_v3.json"
        existing_v3_output = (
            (root / "eval_v3" / "summary_v3.json").is_file()
            or (root / "geometry_v3" / "geometry_v3.json").is_file()
            or bool(structure_enabled and structure_path.is_file())
        )
        if not args.resume and existing_v3_output:
            raise WorkflowError("V3 outputs already exist; pass --resume to reuse or replace them")
        run_manifest = _load_json_object(manifest_path, "run manifest")
        request = run_manifest.get("generation_request") or {}
        run_config = run_manifest.get("run_config") or {}
        video = root / "generation" / "generated.mp4"
        observation = root / "observation" / "video_observation.json"
        if not video.is_file() or not observation.is_file():
            raise WorkflowError("V3 resume requires generation/generated.mp4 and observation/video_observation.json")
        reference = Path(str(request.get("reference_path") or run_config.get("reference", ""))).expanduser()
        image = Path(str(request.get("image_path") or run_config.get("image", ""))).expanduser()
        if not reference.is_file() or not image.is_file():
            raise WorkflowError("run manifest does not point to readable reference/image artifacts")
        observation_data = _load_json_object(observation, "video observation")
        indices = list(dict.fromkeys(
            int(item["frame_index"])
            for item in observation_data.get("windows", [])
            if item.get("frame_index") is not None
        ))
        if not indices:
            raise WorkflowError("video observation contains no sampled frame indices")

        from multimem_bench.cli import _annotation_space_image, _load_v3_config
        from multimem_bench.io import load_reference_scene

        source_config_path = _v3_config_path(args)
        effective_config = _load_v3_config(str(source_config_path) if source_config_path else None)
        if args.matcher:
            effective_config.matcher = str(args.matcher)
        repo_root = _v3_repo_root()
        local_monocular = repo_root / "models" / "moge-2-vitl" / "model.pt"
        if not source_config_path and local_monocular.is_file():
            effective_config.monocular_model_id = str(local_monocular)
        local_matcher = repo_root / "models" / "mast3r" / "checkpoints" / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
        if args.matcher_checkpoint:
            effective_config.matcher_model_id = str(Path(args.matcher_checkpoint).expanduser().resolve())
        elif local_matcher.is_file() and not source_config_path:
            effective_config.matcher_model_id = str(local_matcher)
        effective_config.validate()
        config_payload = effective_config.to_dict()
        if getattr(args, "dry_run", False):
            print(json.dumps({"output_dir": str(root), "video": str(video),
                              "structure_enabled": structure_enabled, "config": config_payload}))
            return 0
        config_path = root / "v3_effective_config.json"
        atomic_write_json(config_path, config_payload)
        effective_structure_config_path = None
        if structure_enabled:
            from multimem_bench.v3.structure import FALLBACK_PROTOCOL, PROMPT_VERSION
            from multimem_bench.v3.articulated import POSE_PROTOCOL, PoseConfig, pose_runtime_identity

            structure_config.validate()
            pose_runtime = (pose_runtime_identity(PoseConfig.from_dict(structure_config.articulated))
                            if structure_config.pipeline == "hybrid" else None)
            effective_structure_config_path = root / "v3_effective_structure_config.json"
            atomic_write_json(effective_structure_config_path, structure_config.to_dict())
        monocular_model = str(effective_config.monocular_model_id)
        monocular_fingerprint = sha256_file(monocular_model) if Path(monocular_model).is_file() else canonical_hash(monocular_model)
        reference_scene = load_reference_scene(reference)
        annotation_image = _annotation_space_image(image, reference_scene)
        reference_cache_key = canonical_hash({
            "image_sha256": sha256_file(annotation_image),
            "reference_sha256": sha256_file(reference),
            "reference_assets_sha256": _v3_reference_assets_fingerprint(reference_scene),
            "reference_config": {
                key: config_payload[key] for key in (
                    "geometry_mode", "monocular_model_id", "monocular_device",
                    "monocular_resolution_level", "reference_query_count", "reference_mask_margin_px",
                )
            },
            "monocular_checkpoint": monocular_fingerprint,
            "monocular_source": _v3_source_fingerprint(repo_root / "models" / "moge"),
            "contract_version": V3_WORKFLOW_CONTRACT_VERSION,
        })
        reference_geometry_path = reference.resolve().parent / "reference_geometry.npz"
        reference_geometry_path.parent.mkdir(parents=True, exist_ok=True)
        reference_structure_cache = None
        if structure_enabled:
            claims_identity = canonical_hash({
                "reference_sha256": sha256_file(reference),
                "reference_image_sha256": sha256_file(annotation_image),
                "structure_config": structure_config.to_dict(),
                "prompt_version": PROMPT_VERSION,
                **({"evaluation_config": config_payload, "pose_protocol": POSE_PROTOCOL,
                    "fallback_protocol": FALLBACK_PROTOCOL if structure_config.articulated_fallback == "vlm" else None,
                    "pose_runtime": pose_runtime,
                    "reference_assets_sha256": _v3_reference_assets_fingerprint(reference_scene),
                    "monocular_checkpoint": monocular_fingerprint}
                   if structure_config.pipeline == "hybrid" else {}),
            })
            reference_structure_cache = (
                reference.resolve().parent / f"reference_structure_{claims_identity}.json"
            )

        geometry_path = root / "geometry_v3" / "geometry_v3.json"
        summary_path = root / "eval_v3" / "summary_v3.json"
        checkpoint = Path(effective_config.matcher_model_id) if Path(effective_config.matcher_model_id).is_file() else None
        dependencies = {
            "reference_identity": reference_cache_key,
            "reference_fingerprint": reference_cache_key,
            "video_sha256": sha256_file(video),
            "observation_sha256": sha256_file(observation),
            "config_sha256": sha256_file(config_path),
            "matcher": effective_config.matcher,
            "matcher_model_id": effective_config.matcher_model_id,
            "matcher_checkpoint_sha256": sha256_file(checkpoint) if checkpoint else canonical_hash(effective_config.matcher_model_id),
            "matcher_source_sha256": _v3_source_fingerprint(repo_root / "models" / "mast3r"),
            "monocular_checkpoint_sha256": monocular_fingerprint,
            "sampled_frame_indices": indices,
            "contract_version": V3_WORKFLOW_CONTRACT_VERSION,
        }
        cache_fingerprint = _v3_cache_fingerprint(dependencies)
        previous_manifest_path = root / "v3_manifest.json"
        previous = _load_json_object(previous_manifest_path, "V3 manifest") if previous_manifest_path.is_file() else {}
        structure_mode_changed = bool(previous.get("structure")) != structure_enabled
        from multimem_bench.v3.evaluator import METRIC_CONTRACT_VERSION

        evaluation_version = METRIC_CONTRACT_VERSION
        evaluation_changed = previous.get("evaluation_version") != evaluation_version
        cache_valid = previous.get("cache_fingerprint") == cache_fingerprint
        structure_dependencies = None
        structure_cache_fingerprint = None
        if structure_enabled:
            structure_dependencies = {
                "geometry_cache_fingerprint": cache_fingerprint,
                "structure_config_sha256": sha256_file(effective_structure_config_path),
                "reference_structure_cache_identity": claims_identity,
                "reference_structure_cache_sha256": (
                    sha256_file(reference_structure_cache)
                    if reference_structure_cache.is_file()
                    else None
                ),
                **({"pose_runtime": pose_runtime,
                    "observation_masks_sha256": _v3_observation_masks_fingerprint(observation)}
                   if structure_config.pipeline == "hybrid" else {}),
            }
            structure_cache_fingerprint = _v3_cache_fingerprint(structure_dependencies)
        geometry_needs_run = bool(args.force) or not args.resume or not cache_valid or not geometry_path.is_file()
        # Keep large frame exports on the run's data volume, not the system /tmp.
        with tempfile.TemporaryDirectory(prefix=".multimem_v3_frames_", dir=root) as temp:
            from multimem_bench.vision.frame_source import extract_video_frames_with_ffmpeg, list_frame_paths

            extracted = extract_video_frames_with_ffmpeg(video, temp, frame_indices=indices)
            extracted_paths = list_frame_paths(extracted)
            if len(extracted_paths) != len(indices):
                raise WorkflowError(f"ffmpeg extracted {len(extracted_paths)} frames for {len(indices)} sampled indices")
            frame_dir = Path(temp) / "indexed"
            frame_dir.mkdir()
            for frame_index, source in zip(indices, extracted_paths):
                shutil.copy2(source, frame_dir / f"{frame_index:06d}.png")
            config_arg = ["--config", str(config_path)]
            reference_command = [
                _v3_python(args.vision_conda_env), "-m", "multimem_bench.cli",
                "prepare-reference-v3", "--reference", str(reference),
                "--reference-image", str(image), "--output", str(reference_geometry_path),
                "--rebuild-invalid",
                *config_arg,
            ]
            _run_v3_subprocess(reference_command, root / "logs" / "prepare_reference_v3.log", args.cuda_visible_devices)
            from multimem_bench.v3.reference import ReferenceGeometry

            reference_artifact = ReferenceGeometry.load(reference_geometry_path)
            dependencies["reference_identity"] = reference_artifact.metadata.get("reference_identity")
            dependencies["reference_fingerprint"] = reference_artifact.fingerprint
            cache_fingerprint = _v3_cache_fingerprint(dependencies)
            cache_valid = previous.get("cache_fingerprint") == cache_fingerprint
            if structure_dependencies is not None:
                structure_dependencies["geometry_cache_fingerprint"] = cache_fingerprint
                structure_cache_fingerprint = _v3_cache_fingerprint(structure_dependencies)
            structure_cache_valid = (
                not structure_enabled
                or previous.get("structure_cache_fingerprint") == structure_cache_fingerprint
            )
            output_hashes = previous.get("output_sha256", {})
            geometry_intact = geometry_path.is_file() and output_hashes.get("geometry") == sha256_file(geometry_path)
            summary_intact = summary_path.is_file() and output_hashes.get("summary") == sha256_file(summary_path)
            structure_intact = (
                not structure_enabled
                or structure_path.is_file() and output_hashes.get("structure") == sha256_file(structure_path)
            )
            geometry_needs_run = (
                bool(args.force) or not args.resume or not cache_valid or not geometry_intact
            )
            if geometry_needs_run:
                geometry_path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_arg = ["--matcher-checkpoint", str(checkpoint)] if checkpoint else []
                geometry_command = [
                    _v3_python(args.vision_conda_env), "-m", "multimem_bench.cli",
                    "prepare-geometry-v3", "--reference", str(reference),
                    "--reference-image", str(image), "--reference-geometry", str(reference_geometry_path),
                    "--frames", str(frame_dir), "--output", str(geometry_path),
                    "--matcher", effective_config.matcher, *checkpoint_arg, *config_arg,
                ]
                _run_v3_subprocess(geometry_command, root / "logs" / "prepare_geometry_v3.log", args.cuda_visible_devices)
            # A scoring-contract bump invalidates only evaluation. If the
            # evidence artifact and geometry are intact, reuse them even when
            # an older run's auxiliary structure fingerprint cannot be
            # reconstructed (for example, a provider cache was ephemeral).
            # Changes to structure inputs/configuration still invalidate the
            # structure stage through ``structure_cache_valid``.
            previous_structure_dependencies = previous.get("structure_cache_dependencies") or {}
            structure_input_keys = (
                "geometry_cache_fingerprint", "structure_config_sha256",
                "reference_structure_cache_identity", "reference_structure_cache_sha256",
                "pose_runtime", "observation_masks_sha256",
            )
            structure_inputs_changed = any(
                previous_structure_dependencies.get(key) != structure_dependencies.get(key)
                for key in structure_input_keys
            ) if structure_dependencies is not None else False
            structure_cache_stale = not structure_cache_valid and (
                not evaluation_changed or structure_inputs_changed
            )
            structure_needs_run = structure_enabled and (
                bool(args.force) or not args.resume or structure_cache_stale
                or bool(getattr(args, "force_structure", False))
                or geometry_needs_run or not structure_intact
            )
            if structure_needs_run:
                structure_path.parent.mkdir(parents=True, exist_ok=True)
                structure_command = [
                    _v3_python(args.vision_conda_env), "-m", "multimem_bench.cli",
                    "prepare-structure-v3", "--reference", str(reference),
                    "--reference-image", str(image), "--observations", str(observation),
                    "--geometry", str(geometry_path), "--frames", str(frame_dir),
                    "--output", str(structure_path), "--structure-config", str(effective_structure_config_path),
                    "--reference-structure-cache", str(reference_structure_cache),
                    *config_arg,
                ]
                _run_v3_subprocess(
                    structure_command,
                    root / "logs" / "prepare_structure_v3.log",
                    args.cuda_visible_devices,
                    pass_env_keys=(structure_config.api_key_env,),
                    **({"secret_env": secret_env} if secret_env else {}),
                )
                structure_dependencies["reference_structure_cache_sha256"] = (
                    sha256_file(reference_structure_cache)
                    if reference_structure_cache.is_file()
                    else None
                )
                structure_cache_fingerprint = _v3_cache_fingerprint(structure_dependencies)
            if (geometry_needs_run or structure_needs_run or structure_mode_changed or evaluation_changed
                    or not summary_intact or not args.resume):
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                structure_arg = ["--structure", str(structure_path)] if structure_enabled else []
                eval_command = [
                    _v3_python(args.vision_conda_env), "-m", "multimem_bench.cli",
                    "eval-v3", "--reference", str(reference), "--observations", str(observation), "--geometry", str(geometry_path), *structure_arg, "--output", str(summary_path.parent), *config_arg,
                ]
                _run_v3_subprocess(eval_command, root / "logs" / "eval_v3.log", args.cuda_visible_devices)
        v3_manifest = root / "v3_manifest.json"
        atomic_write_json(v3_manifest, {
            "schema_version": 1,
            "protocol_version": "3.0",
            "source_run_manifest": str(manifest_path),
            "video": str(video),
            "reference": str(reference),
            "observation": str(observation),
            "geometry": str(geometry_path),
            "structure": str(structure_path) if structure_enabled else None,
            "structure_config": str(effective_structure_config_path) if structure_enabled else None,
            "reference_structure_cache": str(reference_structure_cache) if structure_enabled else None,
            "evaluation_version": evaluation_version,
            "summary": str(summary_path),
            "matcher": effective_config.matcher,
            "matcher_checkpoint": str(checkpoint) if checkpoint else None,
            "reference_geometry": str(reference_geometry_path),
            "cache_fingerprint": cache_fingerprint,
            "cache_dependencies": dependencies,
            "structure_cache_fingerprint": structure_cache_fingerprint,
            "structure_cache_dependencies": structure_dependencies,
            "output_sha256": {
                "geometry": sha256_file(geometry_path),
                "summary": sha256_file(summary_path),
                **({"structure": sha256_file(structure_path)} if structure_enabled else {}),
            },
            "vision_conda_env": args.vision_conda_env,
            "cuda_visible_devices": args.cuda_visible_devices,
            **_v3_protocol_flags(effective_config.geometry_mode),
        })
        print(f"v3_summary: {summary_path}")
        print(f"v3_manifest: {v3_manifest}")
        return 0
    except (OSError, ValueError, WorkflowError) as exc:
        print(f"Error: {sanitize_text(str(exc))}", file=sys.stderr)
        return 2


def _v3_cache_fingerprint(dependencies: dict[str, Any]) -> str:
    """Hash every artifact and setting that can affect a resumed V3 result."""
    return canonical_hash(dependencies)


def _v3_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _v3_source_fingerprint(path: Path) -> str:
    if not path.is_dir():
        return canonical_hash(str(path))
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    revision = completed.stdout.strip()
    return revision if completed.returncode == 0 and revision else canonical_hash(str(path.resolve()))


def _v3_protocol_flags(geometry_mode: str) -> dict[str, bool]:
    return {
        "input_grounded": geometry_mode == "input_grounded",
        "independent_pairwise": geometry_mode == "pairwise_diagnostic",
    }


def _v3_reference_assets_fingerprint(scene: Any) -> str:
    assets: dict[str, str] = {}
    root = Path(scene.artifact_dir or ".")
    for object_id, obj in sorted(scene.objects.items()):
        if obj.mask_path:
            path = Path(obj.mask_path)
            if not path.is_absolute():
                path = root / path
            assets[object_id] = sha256_file(path)
    return canonical_hash(assets)


def _v3_observation_masks_fingerprint(path: Path) -> str:
    from multimem_bench.io import load_video_observation

    observation = load_video_observation(path)
    assets = {}
    for window in observation.windows:
        for obj in window.objects:
            if obj.mask_path:
                mask = Path(obj.mask_path)
                if not mask.is_absolute():
                    mask = path.parent / mask
                assets[f"{window.window_id}:{obj.observed_id}"] = sha256_file(mask) if mask.is_file() else "missing"
    return canonical_hash(assets)


def _run_v3_subprocess(
    command: list[str],
    log_path: Path,
    cuda_visible_devices: str | None,
    *,
    pass_env_keys: tuple[str, ...] = (),
    secret_env: dict[str, str] | None = None,
) -> None:
    env = _evaluation_environment(cuda_visible_devices)
    secret_values: list[str] = []
    for key in pass_env_keys:
        value = os.environ.get(key)
        if value is not None:
            env[key] = value
            secret_values.append(value)
    if secret_env:
        env.update(secret_env)
        secret_values.extend(secret_env.values())
    package_root = _v3_repo_root()
    mast3r_root = package_root / "models" / "mast3r"
    if mast3r_root.is_dir():
        paths = [str(mast3r_root), str(mast3r_root / "dust3r")]
        env["PYTHONPATH"] = os.pathsep.join(paths + [env.get("PYTHONPATH", "")])
    completed = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(sanitize_text(completed.stdout or "", secret_values), encoding="utf-8")
    if completed.returncode != 0:
        raise WorkflowError(f"V3 command exited with {completed.returncode}; see {log_path}")


def _v3_python(environment: str) -> str:
    candidates = [Path(sys.executable)]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if environment and conda_prefix:
        candidates.insert(0, Path(conda_prefix).parent / environment / "bin" / "python")
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def _default_stage_runner(
    command: list[str], log_path: Path, env: dict[str, str]
) -> None:
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    secret_values = [
        os.environ.get("OPENROUTER_API_KEY", ""),
        os.environ.get("ARK_API_KEY", ""),
        os.environ.get("VIDEO_API_KEY", ""),
        os.environ.get("FAL_KEY", ""),
        os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        os.environ.get("AWS_SESSION_TOKEN", ""),
    ]
    output = sanitize_text(completed.stdout or "", secret_values)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output, encoding="utf-8")
    if completed.returncode != 0:
        raise WorkflowError(
            f"command exited with {completed.returncode}; see {log_path}"
        )


def _evaluation_environment(cuda_visible_devices: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "OPENROUTER_API_KEY",
        "ARK_API_KEY",
        "VIDEO_API_KEY",
        "FAL_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "VLM_API_KEY",
    ):
        env.pop(key, None)
    package_dir = str(Path(__file__).resolve().parents[2])
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{package_dir}{os.pathsep}{old_pythonpath}" if old_pythonpath else package_dir
    )
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    elif not env.get("CUDA_VISIBLE_DEVICES"):
        selected = _select_cuda_device()
        if selected is not None:
            env["CUDA_VISIBLE_DEVICES"] = selected
    return env


def _select_cuda_device() -> str | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    candidates: list[tuple[int, int]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            candidates.append((int(fields[1]), int(fields[0])))
        except ValueError:
            continue
    if not candidates:
        return None
    _free_memory, device_index = max(
        candidates,
        key=lambda item: (item[0], -item[1]),
    )
    return str(device_index)


def _outputs_valid(paths: list[Path]) -> bool:
    for path in paths:
        if path.is_file():
            if path.stat().st_size == 0:
                return False
            if path.suffix == ".json":
                try:
                    _load_json_object(path, path.name)
                except WorkflowError:
                    return False
        elif path.is_dir():
            if not any(path.iterdir()):
                return False
        else:
            return False
    return True


def _active_generation_stage(manifest: RunManifest) -> str:
    generation = manifest.stages.get("generate_video")
    if generation is not None and generation.status == "running":
        return "generate_video"
    publication = manifest.stages.get("publish_image")
    if publication is not None and publication.status == "running":
        return "publish_image"
    return "generate_video"


def _require_outputs(paths: list[Path], stage: str) -> None:
    if not _outputs_valid(paths):
        raise WorkflowError(f"{stage} did not produce valid declared outputs")


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise WorkflowError(f"{label} must be a JSON object: {path}")
    return value


def _optional_file_hash(path: Path | None) -> str | None:
    return sha256_file(path) if path is not None and path.is_file() else None


def _effective_observation_vision_config(config: RunConfig) -> Path | None:
    overrides = get_video_observation_overrides(config.model)
    if not overrides:
        return config.vision_config
    values = (
        _load_json_object(config.vision_config, "vision config")
        if config.vision_config is not None
        else {}
    )
    values.update(overrides)
    path = config.output_dir / "observation" / "effective_vision_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, values)
    return path


def _atomic_copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_file:
            shutil.copyfileobj(input_file, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _resolve_optional_path(value: Any) -> Path | None:
    if value is None or str(value) == "":
        return None
    return Path(value).expanduser().resolve()


def _safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")


def _image_size_label(path: Path) -> str:
    try:
        from PIL import Image
    except ImportError as exc:
        raise WorkflowError(
            "Pillow is required; install multiMemBench[orchestration]"
        ) from exc
    try:
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
    except Exception as exc:
        raise WorkflowError(f"initial image is not a readable raster: {path}") from exc
    return f"{width}x{height}"
