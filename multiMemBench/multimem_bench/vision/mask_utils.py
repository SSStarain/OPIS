"""Mask, bbox, crop, and deduplication helpers."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


BBox = tuple[float, float, float, float]


def bbox_center(bbox: BBox) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def bbox_area(bbox: BBox) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def bbox_iou(a: BBox, b: BBox) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = bbox_area((x0, y0, x1, y1))
    denom = bbox_area(a) + bbox_area(b) - inter
    return 0.0 if denom <= 0 else inter / denom


def bbox_intersection_over_min(a: BBox, b: BBox) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = bbox_area((x0, y0, x1, y1))
    denom = min(bbox_area(a), bbox_area(b))
    return 0.0 if denom <= 0 else inter / denom


def coerce_bbox(
    box: Iterable[float],
    image_size: tuple[int, int] | None = None,
    *,
    box_format: str = "xyxy",
) -> BBox:
    vals = [float(v) for v in box]
    if len(vals) != 4:
        raise ValueError("bbox must contain exactly four numbers")
    if box_format == "xywh":
        x, y, w, h = vals
        vals = [x, y, x + w, y + h]
    elif box_format != "xyxy":
        raise ValueError(f"unsupported box_format: {box_format}")

    if image_size is not None and max(abs(v) for v in vals) <= 1.5:
        width, height = image_size
        vals = [vals[0] * width, vals[1] * height, vals[2] * width, vals[3] * height]
    x0, x1 = sorted([vals[0], vals[2]])
    y0, y1 = sorted([vals[1], vals[3]])
    if image_size is not None:
        width, height = image_size
        x0 = max(0.0, min(float(width), x0))
        x1 = max(0.0, min(float(width), x1))
        y0 = max(0.0, min(float(height), y0))
        y1 = max(0.0, min(float(height), y1))
    return (x0, y0, x1, y1)


def mask_to_bool_array(mask: Any) -> np.ndarray:
    if isinstance(mask, Image.Image):
        arr = np.asarray(mask)
    else:
        arr = np.asarray(mask)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(f"mask must be 2D or have one singleton channel: {arr.shape}")
    if arr.ndim != 2:
        raise ValueError(f"mask must be 2D after channel normalization: {arr.shape}")
    return arr.astype(float) > 0


def mask_to_bbox(mask: Any) -> BBox | None:
    arr = mask_to_bool_array(mask)
    ys, xs = np.where(arr)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def mask_area(mask: Any) -> float:
    return float(mask_to_bool_array(mask).sum())


def mask_centroid(mask: Any) -> tuple[float, float] | None:
    arr = mask_to_bool_array(mask)
    ys, xs = np.where(arr)
    if len(xs) == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))


def mask_from_bbox(bbox: BBox, image_size: tuple[int, int]) -> np.ndarray:
    """Compatibility helper for legacy simulated artifacts without masks."""
    width, height = image_size
    out = np.zeros((height, width), dtype=bool)
    x0 = max(0, min(width, int(math.floor(bbox[0]))))
    y0 = max(0, min(height, int(math.floor(bbox[1]))))
    x1 = max(0, min(width, int(math.ceil(bbox[2]))))
    y1 = max(0, min(height, int(math.ceil(bbox[3]))))
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = True
    return out


def mask_iou(a: Any, b: Any) -> float:
    aa = mask_to_bool_array(a)
    bb = mask_to_bool_array(b)
    if aa.shape != bb.shape:
        return 0.0
    inter = float(np.logical_and(aa, bb).sum())
    union = float(np.logical_or(aa, bb).sum())
    return 0.0 if union <= 0 else inter / union


def mask_intersection_over_min(a: Any, b: Any) -> float:
    aa = mask_to_bool_array(a)
    bb = mask_to_bool_array(b)
    if aa.shape != bb.shape:
        return 0.0
    inter = float(np.logical_and(aa, bb).sum())
    denom = min(float(aa.sum()), float(bb.sum()))
    return 0.0 if denom <= 0 else inter / denom


def expanded_int_bbox(
    bbox: BBox,
    image_size: tuple[int, int],
    padding: int = 0,
) -> tuple[int, int, int, int]:
    width, height = image_size
    return (
        max(0, int(math.floor(bbox[0])) - padding),
        max(0, int(math.floor(bbox[1])) - padding),
        min(width, int(math.ceil(bbox[2])) + padding),
        min(height, int(math.ceil(bbox[3])) + padding),
    )


def crop_object(
    image: Image.Image,
    bbox: BBox,
    mask: Any | None = None,
    *,
    padding: int = 0,
    background: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    image_rgb = image.convert("RGB")
    crop_box = expanded_int_bbox(bbox, image_rgb.size, padding)
    crop = image_rgb.crop(crop_box)
    if mask is None:
        return crop

    arr = np.asarray(crop).copy()
    full_mask = mask_to_bool_array(mask)
    x0, y0, x1, y1 = crop_box
    if full_mask.shape[0] < y1 or full_mask.shape[1] < x1:
        return crop
    crop_mask = full_mask[y0:y1, x0:x1]
    if crop_mask.shape[:2] != arr.shape[:2]:
        return crop
    bg = np.zeros_like(arr)
    bg[:, :] = np.asarray(background, dtype=np.uint8)
    arr = np.where(crop_mask[..., None], arr, bg)
    return Image.fromarray(arr)


def save_mask_png(mask: Any, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = (mask_to_bool_array(mask).astype(np.uint8) * 255)
    Image.fromarray(arr, mode="L").save(out)
    return out


def stable_segment_id(prefix: str, bbox: BBox, category: str, frame_index: int | None = None) -> str:
    raw = f"{category}:{frame_index}:{bbox[0]:.1f}:{bbox[1]:.1f}:{bbox[2]:.1f}:{bbox[3]:.1f}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{digest}"
