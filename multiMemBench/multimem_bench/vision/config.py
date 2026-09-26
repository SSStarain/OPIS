"""Configuration for reference/video visual preprocessing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class VisionConfig:
    """Knobs for SAM3-based preprocessing.

    Heavy models are intentionally behind optional imports. In production runs
    set `strict_models=true` so missing SAM3/DINOv3 fails loudly.
    """

    # General model/runtime settings.
    device: str = "cuda"
    strict_models: bool = False

    # SAM3 segmentation settings.
    sam3_backend: str = "official"
    sam3_checkpoint_path: str | None = None
    sam3_bpe_path: str | None = None
    sam3_model_id: str | None = None
    sam3_confidence_threshold: float = 0.35
    sam3_resolution: int = 1008
    sam3_mask_threshold: float = 0.0

    # Reference-side foreground discovery.
    reference_candidate_mode: str = "sam3_text"
    reference_foreground_prompts: list[str] = field(default_factory=lambda: ["foreground object"])
    reference_noun_phrases: list[str] = field(default_factory=list)
    reference_refine_with_labels: bool = True
    reference_image_width: int = 1280
    reference_image_height: int = 720
    reference_image_resize_mode: str = "letterbox"
    # Reference annotations and video observations share the 256-instance
    # capacity; observations retain all slots for hallucination scoring.
    max_instances: int | None = None
    max_reference_objects: int = 256

    # Mask filtering/deduplication.
    min_mask_area_px: float = 64.0
    min_mask_area_ratio: float = 0.00005
    max_mask_area_ratio: float = 0.65
    mask_nms_iou_threshold: float = 0.82
    mask_containment_threshold: float = 0.92
    bbox_nms_iou_threshold: float = 0.88
    crop_padding_px: int = 4
    save_masks: bool = True
    save_crops: bool = True

    # Open-category labeling hook.
    labeler_backend: str = "none"
    labeler_command: str | None = None
    labeler_timeout_sec: float = 30.0
    unknown_category_name: str = "object"

    # Appearance embeddings. `auto` tries DINOv3, then falls back to color hist.
    embedding_backend: str = "auto"
    embedding_device: str | None = None
    dinov3_repo_or_dir: str | None = None
    dinov3_model_name: str = "dinov3_vits16"
    dinov3_weights_path: str | None = None
    embedding_image_size: int = 224
    embedding_batch_size: int = 32
    color_hist_bins: int = 8

    # Video sampling and SAM3 video inference.
    video_sample_stride: int = 8
    video_sample_fps: float | None = None
    video_skip_initial_seconds: float = 1.0
    video_max_frames: int | None = 64
    video_prompt_frame_index: int = 0
    video_segmentation_mode: str = "framewise"
    video_use_reference_categories: bool = True
    video_prompts: list[str] = field(default_factory=list)
    video_max_prompts: int = 64
    video_max_objects: int = 256
    video_preload_frames: bool = True
    video_empty_frame_image_pcs_fallback: bool = False

    def __post_init__(self) -> None:
        self.video_segmentation_mode = str(self.video_segmentation_mode).lower()
        if self.video_segmentation_mode not in {"framewise", "tracking"}:
            raise ValueError(
                "video_segmentation_mode must be 'framewise' or 'tracking'"
            )
        if self.max_instances is not None:
            if self.max_reference_objects != 256 or self.video_max_objects != 256:
                raise ValueError(
                    "legacy max_instances cannot be mixed with canonical instance limits"
                )
            self.max_reference_objects = self.max_instances
            self.video_max_objects = self.max_instances
            self.max_instances = None
        limits = [self.max_reference_objects, self.video_max_objects]
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in limits
        ):
            raise ValueError("instance limits must be positive integers")
        self.max_reference_objects = int(self.max_reference_objects)
        self.video_max_objects = int(self.video_max_objects)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "VisionConfig":
        if not data:
            return cls()
        raw = dict(data)
        if "max_instances" in raw:
            if "max_reference_objects" in raw or "video_max_objects" in raw:
                raise ValueError(
                    "legacy max_instances cannot be mixed with canonical instance limits"
                )
            legacy = raw.pop("max_instances")
            raw["max_reference_objects"] = legacy
            raw["video_max_objects"] = legacy
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("max_instances", None)
        return result
