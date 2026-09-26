"""Anchor-derived camera-view helpers.

This module does not perform feature-based camera alignment. It estimates a
per-window reference-image crop from the selected anchor:

1. observed-anchor mask area / reference-anchor mask area -> actual zoom
2. anchor-mask centroid to frame-boundary distances / actual zoom -> reference
   distances
3. reference anchor centroid plus those distances -> reference view box

For legacy simulated artifacts without masks, bbox geometry is retained as an
explicit fallback. Zoom compliance is reported separately from object-memory
error statistics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any

from multimem_bench.config import EvaluationConfig
from multimem_bench.schema import (
    ObjectSignature,
    ObservedObject,
    ObservedWindow,
    ReferenceScene,
    VideoObservation,
)
from multimem_bench.similarity import clamp01
from multimem_bench.vision.mask_utils import mask_area, mask_centroid, mask_to_bbox


@dataclass(frozen=True)
class AnchorCameraEstimate:
    valid: bool
    mode: str
    actual_zoom: float | None = None
    zoom_x: float | None = None
    zoom_y: float | None = None
    zoom_anisotropy: float | None = None
    requested_zoom: float | None = None
    zoom_relative_error: float | None = None
    zoom_pass: bool | None = None
    zoom_compliance_score: float | None = None
    reference_view_box: tuple[float, float, float, float] | None = None
    reference_view_box_clipped: tuple[float, float, float, float] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def requested_zoom_from_metadata(
    scene: ReferenceScene,
    observation: VideoObservation | None = None,
) -> float | None:
    candidates = [
        scene.metadata.get("requested_zoom"),
        scene.metadata.get("prompt_zoom"),
        scene.metadata.get("zoom"),
    ]
    if observation is not None:
        candidates.extend([
            observation.metadata.get("requested_zoom"),
            observation.metadata.get("prompt_zoom"),
            observation.metadata.get("zoom"),
        ])
    for value in candidates:
        parsed = parse_zoom_value(value)
        if parsed is not None:
            return parsed
    return None


def parse_zoom_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        zoom = float(value)
        return zoom if zoom > 0 else None
    text = str(value)
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    if not match:
        return None
    zoom = float(match.group(1))
    return zoom if zoom > 0 else None


def estimate_anchor_camera(
    scene: ReferenceScene,
    window: ObservedWindow,
    observed_anchor: ObservedObject,
    reference_anchor: ObjectSignature,
    config: EvaluationConfig,
    *,
    requested_zoom: float | None = None,
) -> AnchorCameraEstimate:
    mask_geometry = _mask_anchor_geometry(observed_anchor, reference_anchor)
    if mask_geometry is not None:
        actual_zoom, zoom_x, zoom_y = mask_geometry
        mode = "anchor_mask_area"
    elif config.require_masks:
        return AnchorCameraEstimate(
            valid=False,
            mode="mask_required",
            requested_zoom=requested_zoom,
            error="anchor masks are required but unavailable",
        )
    else:
        bbox_geometry = _bbox_anchor_geometry(observed_anchor, reference_anchor)
        if bbox_geometry is None:
            return AnchorCameraEstimate(
                valid=False,
                mode="relative_fallback",
                requested_zoom=requested_zoom,
                error="anchor mask and bbox geometry are unavailable",
            )
        actual_zoom, zoom_x, zoom_y = bbox_geometry
        mode = "anchor_bbox_fallback"

    reference_center = _reference_anchor_center(reference_anchor)
    observed_center = _observed_anchor_center(observed_anchor)
    if reference_center is None or observed_center is None:
        return AnchorCameraEstimate(
            valid=False,
            mode=mode,
            requested_zoom=requested_zoom,
            error="anchor center is unavailable",
        )

    zoom_anisotropy = abs(math.log(max(zoom_x, 1e-6) / max(zoom_y, 1e-6)))
    rel_error = None
    zoom_pass = None
    zoom_score = None
    if requested_zoom is not None and requested_zoom > 0:
        rel_error = abs(actual_zoom - requested_zoom) / requested_zoom
        zoom_pass = rel_error <= config.zoom_relative_tolerance
        if zoom_pass:
            zoom_score = 1.0
        else:
            excess = rel_error - config.zoom_relative_tolerance
            zoom_score = clamp01(1.0 - excess / max(config.zoom_dropoff_relative, 1e-6))

    view_box = _reference_view_box_from_anchor(
        window,
        observed_center,
        reference_center,
        actual_zoom,
    )
    clipped = _clip_box_to_image(view_box, scene.metadata.get("image_size"))
    return AnchorCameraEstimate(
        valid=True,
        mode=mode,
        actual_zoom=actual_zoom,
        zoom_x=zoom_x,
        zoom_y=zoom_y,
        zoom_anisotropy=zoom_anisotropy,
        requested_zoom=requested_zoom,
        zoom_relative_error=rel_error,
        zoom_pass=zoom_pass,
        zoom_compliance_score=zoom_score,
        reference_view_box=view_box,
        reference_view_box_clipped=clipped,
    )


def _mask_anchor_geometry(
    observed_anchor: ObservedObject,
    reference_anchor: ObjectSignature,
) -> tuple[float, float, float] | None:
    if observed_anchor.mask is None or reference_anchor.mask is None:
        return None
    try:
        observed_area = mask_area(observed_anchor.mask)
        reference_area = mask_area(reference_anchor.mask)
        observed_support = mask_to_bbox(observed_anchor.mask)
        reference_support = mask_to_bbox(reference_anchor.mask)
    except (TypeError, ValueError):
        return None
    if (
        observed_area <= 0
        or reference_area <= 0
        or observed_support is None
        or reference_support is None
    ):
        return None

    ref_w = reference_support[2] - reference_support[0]
    ref_h = reference_support[3] - reference_support[1]
    obs_w = observed_support[2] - observed_support[0]
    obs_h = observed_support[3] - observed_support[1]
    if min(ref_w, ref_h, obs_w, obs_h) <= 0:
        return None
    return (
        math.sqrt(observed_area / reference_area),
        obs_w / ref_w,
        obs_h / ref_h,
    )


def _bbox_anchor_geometry(
    observed_anchor: ObservedObject,
    reference_anchor: ObjectSignature,
) -> tuple[float, float, float] | None:
    ref_w = reference_anchor.width
    ref_h = reference_anchor.height
    if ref_w is None or ref_h is None or ref_w <= 0 or ref_h <= 0:
        return None
    zoom_x = observed_anchor.width / ref_w
    zoom_y = observed_anchor.height / ref_h
    if zoom_x <= 0 or zoom_y <= 0:
        return None
    return math.sqrt(zoom_x * zoom_y), zoom_x, zoom_y


def _reference_anchor_center(
    anchor: ObjectSignature,
) -> tuple[float, float] | None:
    if anchor.mask is not None:
        try:
            center = mask_centroid(anchor.mask)
        except (TypeError, ValueError):
            center = None
        if center is not None:
            return center
    return anchor.center


def _observed_anchor_center(
    anchor: ObservedObject,
) -> tuple[float, float] | None:
    if anchor.mask is not None:
        try:
            center = mask_centroid(anchor.mask)
        except (TypeError, ValueError):
            center = None
        if center is not None:
            return center
    return anchor.center


def _reference_view_box_from_anchor(
    window: ObservedWindow,
    observed_center: tuple[float, float],
    reference_center: tuple[float, float],
    actual_zoom: float,
) -> tuple[float, float, float, float]:
    frame_w, frame_h = window.frame_size
    obs_cx, obs_cy = observed_center
    ref_cx, ref_cy = reference_center
    left = obs_cx / actual_zoom
    right = max(0.0, frame_w - obs_cx) / actual_zoom
    up = obs_cy / actual_zoom
    down = max(0.0, frame_h - obs_cy) / actual_zoom
    return (ref_cx - left, ref_cy - up, ref_cx + right, ref_cy + down)


def _clip_box_to_image(
    box: tuple[float, float, float, float],
    image_size: Any,
) -> tuple[float, float, float, float]:
    if not isinstance(image_size, list) or len(image_size) != 2:
        return box
    width = float(image_size[0])
    height = float(image_size[1])
    return (
        max(0.0, min(width, box[0])),
        max(0.0, min(height, box[1])),
        max(0.0, min(width, box[2])),
        max(0.0, min(height, box[3])),
    )
