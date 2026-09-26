"""Deterministic reference-image normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from multimem_bench.vision.mask_utils import mask_to_bool_array


@dataclass(frozen=True)
class PreparedReferenceImage:
    """Normalized image and geometry needed to transform source segments."""

    path: Path
    image: Image.Image
    original_size: tuple[int, int]
    processed_size: tuple[int, int]
    scale: float
    padding: tuple[int, int]


def prepare_reference_image(
    source_path: str | Path,
    assets_dir: str | Path,
    *,
    target_size: tuple[int, int] = (1280, 720),
    mode: str = "letterbox",
) -> PreparedReferenceImage:
    """Create a fixed-size RGB reference image without crop or distortion."""

    source_path = Path(source_path)
    assets_dir = Path(assets_dir)
    target_width, target_height = (int(target_size[0]), int(target_size[1]))
    if target_width <= 0 or target_height <= 0:
        raise ValueError("target_size must contain positive width and height")
    if mode.lower() != "letterbox":
        raise ValueError(f"unsupported reference image resize mode: {mode}")

    source = Image.open(source_path).convert("RGB")
    source_width, source_height = source.size
    scale = min(target_width / source_width, target_height / source_height)
    resized_size = (
        max(1, int(round(source_width * scale))),
        max(1, int(round(source_height * scale))),
    )
    resized = source.resize(resized_size, Image.Resampling.LANCZOS)
    padding = (
        (target_width - resized_size[0]) // 2,
        (target_height - resized_size[1]) // 2,
    )
    canvas = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    canvas.paste(resized, padding)

    output_path = (
        assets_dir
        / "preprocessed"
        / f"{source_path.stem}_{target_width}x{target_height}.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return PreparedReferenceImage(
        path=output_path,
        image=canvas,
        original_size=(source_width, source_height),
        processed_size=(target_width, target_height),
        scale=float(scale),
        padding=padding,
    )


def transform_bbox(
    bbox: tuple[float, float, float, float],
    prepared: PreparedReferenceImage,
) -> tuple[float, float, float, float]:
    """Map a source-image bbox into the normalized reference image."""

    pad_x, pad_y = prepared.padding
    x0, y0, x1, y1 = bbox
    return (
        float(x0 * prepared.scale + pad_x),
        float(y0 * prepared.scale + pad_y),
        float(x1 * prepared.scale + pad_x),
        float(y1 * prepared.scale + pad_y),
    )


def transform_mask(mask: object, prepared: PreparedReferenceImage) -> np.ndarray:
    """Map a source mask into the normalized reference image with nearest resize."""

    source_mask = mask_to_bool_array(mask)
    source_image = Image.fromarray(source_mask.astype(np.uint8) * 255, mode="L")
    resized_width = max(1, int(round(source_image.width * prepared.scale)))
    resized_height = max(1, int(round(source_image.height * prepared.scale)))
    resized = source_image.resize((resized_width, resized_height), Image.Resampling.NEAREST)
    canvas = Image.new("L", prepared.processed_size, 0)
    canvas.paste(resized, prepared.padding)
    return np.asarray(canvas, dtype=np.uint8) > 0
