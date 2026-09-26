"""Build `reference_scene.json` from a reference image."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from multimem_bench.io import write_json
from multimem_bench.vision.config import VisionConfig
from multimem_bench.vision.embedding import build_embedding_extractor
from multimem_bench.vision.image_preprocess import (
    PreparedReferenceImage,
    prepare_reference_image,
    transform_bbox,
    transform_mask,
)
from multimem_bench.vision.labeler import build_labeler
from multimem_bench.vision.mask_utils import (
    bbox_area,
    bbox_intersection_over_min,
    bbox_iou,
    crop_object,
    mask_area,
    mask_intersection_over_min,
    mask_iou,
    save_mask_png,
)
from multimem_bench.vision.sam3_adapter import Sam3ImageSegmenter
from multimem_bench.vision.types import SegmentInstance, load_segment_instances


@dataclass
class ReferenceBuildResult:
    reference_path: Path
    assets_dir: Path
    scene: dict[str, Any]
    num_raw_segments: int
    num_final_objects: int


def build_reference_scene_artifact(
    *,
    image_path: str | Path,
    output_path: str | Path,
    config: VisionConfig,
    scene_id: str | None = None,
    prompt_zoom: float | None = None,
    prompts: Iterable[str] | None = None,
    precomputed_segments_path: str | Path | None = None,
    assets_dir: str | Path | None = None,
) -> ReferenceBuildResult:
    image_path = Path(image_path)
    output_path = Path(output_path)
    assets = Path(assets_dir) if assets_dir else output_path.parent / f"{output_path.stem}_assets"
    masks_dir = assets / "masks"
    crops_dir = assets / "crops"
    prepared = prepare_reference_image(
        image_path,
        assets,
        target_size=(config.reference_image_width, config.reference_image_height),
        mode=config.reference_image_resize_mode,
    )
    working_image_path = prepared.path
    image = prepared.image
    input_prompts = _effective_reference_prompt_list(
        prompts,
        config,
        precomputed_segments_path=precomputed_segments_path,
    )

    raw_segments = _initial_reference_segments(
        image_path=working_image_path,
        config=config,
        prompts=prompts,
        precomputed_segments_path=precomputed_segments_path,
    )
    _load_external_masks(
        raw_segments,
        base_dir=Path(precomputed_segments_path).parent
        if precomputed_segments_path
        else image_path.parent,
    )
    if precomputed_segments_path:
        _transform_precomputed_segments(raw_segments, prepared)
    filtered = _filter_segments(raw_segments, image.size, config)
    filtered = _deduplicate_segments(filtered, config)

    if _should_refine_with_labels(config, precomputed_segments_path) and filtered:
        refined = _refine_segments_with_labels(
            image_path=working_image_path,
            image=image,
            initial_segments=filtered,
            assets_dir=assets / "label_probe",
            config=config,
        )
        _load_external_masks(refined, base_dir=working_image_path.parent)
        filtered = _deduplicate_segments(
            _filter_segments(filtered + refined, image.size, config),
            config,
        )

    if len(filtered) > config.max_reference_objects:
        raise ValueError(
            f"detected {len(filtered)} reference objects, exceeding "
            f"max_reference_objects={config.max_reference_objects}; refine the prompts or use an "
            "explicit stress-profile override"
        )

    labeler = build_labeler(config)
    embedder = build_embedding_extractor(config)
    objects: list[dict[str, Any]] = []
    for idx, segment in enumerate(_sort_segments(filtered)):
        object_id = f"ref_{idx:04d}"
        crop = crop_object(
            image,
            segment.bbox,
            segment.mask,
            padding=config.crop_padding_px,
        )
        mask_path = None
        crop_path = None
        if config.save_masks and segment.mask is not None:
            mask_path = masks_dir / f"{object_id}.png"
            save_mask_png(segment.mask, mask_path)
            segment.mask_path = _relative_path(mask_path, output_path.parent)
        if config.save_crops:
            crop_path = crops_dir / f"{object_id}.png"
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(crop_path)
            segment.crop_path = _relative_path(crop_path, output_path.parent)

        category, label_attrs = labeler.label(
            segment,
            image_path=working_image_path,
            crop_path=crop_path,
            mask_path=mask_path,
        )
        segment.category = category or config.unknown_category_name
        segment.attributes.update(_visual_attributes(crop, segment))
        segment.attributes.update(label_attrs)
        if segment.embedding is None:
            segment.embedding = embedder.embed(crop)
        segment.metadata.update({
            "confidence": segment.confidence,
            "embedding_backend": getattr(embedder, "backend_name", config.embedding_backend),
            "labeler_backend": getattr(labeler, "backend_name", config.labeler_backend),
            "mask_area": _segment_area(segment),
        })
        objects.append(segment.to_reference_object(object_id))

    metadata: dict[str, Any] = {
        "image_path": str(image_path),
        "processed_image_path": _relative_path(working_image_path, output_path.parent),
        "source_image_size": list(prepared.original_size),
        "processed_image_size": list(prepared.processed_size),
        "resize_scale": prepared.scale,
        "resize_padding": list(prepared.padding),
        "image_size": [image.size[0], image.size[1]],
        "preprocessing": {
            "segmenter": "precomputed" if precomputed_segments_path else "sam3",
            "candidate_mode": config.reference_candidate_mode,
            "num_raw_segments": len(raw_segments),
            "num_filtered_segments": len(filtered),
            "embedding_backend": getattr(embedder, "backend_name", config.embedding_backend),
            "labeler_backend": config.labeler_backend,
            "input_prompts": input_prompts,
            "config": config.to_dict(),
        },
    }
    if prompt_zoom is not None:
        metadata["prompt_zoom"] = float(prompt_zoom)
        metadata["requested_zoom"] = float(prompt_zoom)

    scene = {
        "schema_version": 2,
        "scene_id": scene_id or image_path.stem,
        "objects": objects,
        "metadata": metadata,
    }
    write_json(output_path, scene)
    return ReferenceBuildResult(
        reference_path=output_path,
        assets_dir=assets,
        scene=scene,
        num_raw_segments=len(raw_segments),
        num_final_objects=len(scene["objects"]),
    )


def _initial_reference_segments(
    *,
    image_path: Path,
    config: VisionConfig,
    prompts: Iterable[str] | None,
    precomputed_segments_path: str | Path | None,
) -> list[SegmentInstance]:
    if precomputed_segments_path is not None:
        return load_segment_instances(precomputed_segments_path)
    segmenter = Sam3ImageSegmenter(config)
    mode = config.reference_candidate_mode.lower()
    if mode == "sam3_auto":
        return segmenter.segment_automatic(image_path)
    if mode in {"sam3_phrase_list", "sam3_pcs"}:
        phrase_list = _prompt_list(prompts, config.reference_noun_phrases)
        if not phrase_list:
            raise ValueError(
                "reference_candidate_mode=sam3_phrase_list requires noun phrases "
                "from --prompts, --prompt-file, or vision_config.reference_noun_phrases"
            )
        return segmenter.segment_by_prompts(image_path, phrase_list)
    if mode != "sam3_text":
        raise ValueError(f"unsupported reference_candidate_mode: {config.reference_candidate_mode}")
    prompt_list = _prompt_list(prompts, config.reference_foreground_prompts)
    if not prompt_list:
        raise ValueError("reference SAM3 text mode requires at least one prompt")
    return segmenter.segment_by_prompts(image_path, prompt_list)


def _effective_reference_prompt_list(
    cli_prompts: Iterable[str] | None,
    config: VisionConfig,
    *,
    precomputed_segments_path: str | Path | None,
) -> list[str]:
    if precomputed_segments_path is not None:
        return []
    mode = config.reference_candidate_mode.lower()
    if mode in {"sam3_phrase_list", "sam3_pcs"}:
        return _prompt_list(cli_prompts, config.reference_noun_phrases)
    if mode == "sam3_text":
        return _prompt_list(cli_prompts, config.reference_foreground_prompts)
    return []


def _prompt_list(
    cli_prompts: Iterable[str] | None,
    config_prompts: Iterable[str],
) -> list[str]:
    prompts = cli_prompts if cli_prompts is not None else config_prompts
    seen: set[str] = set()
    values: list[str] = []
    for prompt in prompts:
        text = str(prompt).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        values.append(text)
    return values


def _should_refine_with_labels(
    config: VisionConfig,
    precomputed_segments_path: str | Path | None,
) -> bool:
    if precomputed_segments_path is not None:
        return False
    if not config.reference_refine_with_labels:
        return False
    if config.labeler_backend.lower() == "none":
        return False
    return config.reference_candidate_mode.lower() not in {"sam3_phrase_list", "sam3_pcs"}


def _refine_segments_with_labels(
    *,
    image_path: Path,
    image: Image.Image,
    initial_segments: list[SegmentInstance],
    assets_dir: Path,
    config: VisionConfig,
) -> list[SegmentInstance]:
    labeler = build_labeler(config)
    labels: set[str] = set()
    for idx, segment in enumerate(initial_segments):
        crop = crop_object(image, segment.bbox, segment.mask, padding=config.crop_padding_px)
        crop_path = assets_dir / "crops" / f"probe_{idx:04d}.png"
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(crop_path)
        mask_path = None
        if segment.mask is not None:
            mask_path = assets_dir / "masks" / f"probe_{idx:04d}.png"
            save_mask_png(segment.mask, mask_path)
        category, _attrs = labeler.label(
            segment,
            image_path=image_path,
            crop_path=crop_path,
            mask_path=mask_path,
        )
        label = category.strip().lower()
        if label and label not in {"object", "foreground object", "thing"}:
            labels.add(label)
    if not labels:
        return []
    segmenter = Sam3ImageSegmenter(config)
    return segmenter.segment_by_prompts(image_path, sorted(labels))


def _transform_precomputed_segments(
    segments: list[SegmentInstance],
    prepared: PreparedReferenceImage,
) -> None:
    for segment in segments:
        segment.bbox = transform_bbox(segment.bbox, prepared)
        if segment.mask is not None:
            segment.mask = transform_mask(segment.mask, prepared)


def _load_external_masks(segments: list[SegmentInstance], *, base_dir: Path) -> None:
    for segment in segments:
        if segment.mask is not None or not segment.mask_path:
            continue
        path = Path(segment.mask_path)
        if not path.is_absolute():
            path = base_dir / path
        if path.exists():
            segment.mask = Image.open(path).convert("L")


def _filter_segments(
    segments: list[SegmentInstance],
    image_size: tuple[int, int],
    config: VisionConfig,
) -> list[SegmentInstance]:
    image_area = max(1.0, float(image_size[0] * image_size[1]))
    out: list[SegmentInstance] = []
    for segment in segments:
        area = _segment_area(segment)
        if area < config.min_mask_area_px:
            continue
        ratio = area / image_area
        if ratio < config.min_mask_area_ratio or ratio > config.max_mask_area_ratio:
            continue
        if segment.confidence < config.sam3_confidence_threshold:
            continue
        out.append(segment)
    return out


def _deduplicate_segments(
    segments: list[SegmentInstance],
    config: VisionConfig,
) -> list[SegmentInstance]:
    kept: list[SegmentInstance] = []
    for segment in sorted(segments, key=lambda s: (s.confidence, _segment_area(s)), reverse=True):
        duplicate = False
        for existing in kept:
            if segment.mask is not None and existing.mask is not None:
                iou = mask_iou(segment.mask, existing.mask)
                iom = mask_intersection_over_min(segment.mask, existing.mask)
                iou_threshold = config.mask_nms_iou_threshold
            else:
                iou = bbox_iou(segment.bbox, existing.bbox)
                iom = bbox_intersection_over_min(segment.bbox, existing.bbox)
                iou_threshold = config.bbox_nms_iou_threshold
            if (
                iou >= iou_threshold
                or iom >= config.mask_containment_threshold
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(segment)
    return kept


def _sort_segments(segments: list[SegmentInstance]) -> list[SegmentInstance]:
    return sorted(segments, key=lambda s: (s.center[0], s.center[1], s.category, s.segment_id))


def _segment_area(segment: SegmentInstance) -> float:
    if segment.mask is not None:
        try:
            return mask_area(segment.mask)
        except Exception:
            pass
    return bbox_area(segment.bbox)


def _visual_attributes(crop: Image.Image, segment: SegmentInstance) -> dict[str, Any]:
    arr = np.asarray(crop.convert("RGB")).astype(np.float32)
    mean_rgb = arr.reshape(-1, 3).mean(axis=0)
    aspect = segment.width / max(segment.height, 1e-6)
    if aspect > 1.35:
        shape = "wide"
    elif aspect < 0.74:
        shape = "tall"
    else:
        shape = "compact"
    return {
        "color": _nearest_color_name(mean_rgb),
        "mean_rgb": [float(v) for v in mean_rgb],
        "shape": shape,
        "aspect_ratio": float(aspect),
    }


def _nearest_color_name(rgb: np.ndarray) -> str:
    palette = {
        "black": (20, 20, 20),
        "white": (235, 235, 235),
        "gray": (128, 128, 128),
        "red": (210, 45, 45),
        "green": (55, 160, 70),
        "blue": (55, 100, 210),
        "yellow": (220, 200, 40),
        "orange": (225, 125, 35),
        "brown": (125, 80, 45),
        "purple": (145, 80, 175),
    }
    best = min(
        palette.items(),
        key=lambda item: float(np.linalg.norm(rgb - np.asarray(item[1], dtype=np.float32))),
    )
    return best[0]


def _relative_path(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)
