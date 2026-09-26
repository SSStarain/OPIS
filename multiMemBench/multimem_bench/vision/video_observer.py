"""Build `video_observation.json` from a generated video or frame folder."""

from __future__ import annotations

import gc
from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from multimem_bench.io import load_reference_scene, write_json
from multimem_bench.vision.config import VisionConfig
from multimem_bench.vision.embedding import build_embedding_extractor
from multimem_bench.vision.frame_source import (
    VIDEO_EXTENSIONS,
    FrameSourceError,
    VideoTimeline,
    ensure_frame_map,
    probe_video_timeline,
    sample_indices,
    timestamp_sample_indices,
)
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
from multimem_bench.vision.reference_builder import _visual_attributes
from multimem_bench.vision.sam3_adapter import (
    Sam3ImageSegmenter,
    Sam3UnavailableError,
    Sam3VideoSegmenter,
)
from multimem_bench.vision.types import SegmentInstance, load_segment_instances


@dataclass
class VideoObservationBuildResult:
    observation_path: Path
    assets_dir: Path
    observation: dict[str, Any]
    sampled_frame_indices: list[int]
    num_segments: int


def build_video_observation_artifact(
    *,
    video_path: str | Path,
    reference_path: str | Path,
    output_path: str | Path,
    config: VisionConfig,
    video_id: str | None = None,
    prompts: Iterable[str] | None = None,
    precomputed_segments_path: str | Path | None = None,
    frames_path: str | Path | None = None,
    assets_dir: str | Path | None = None,
    video_segmenter: Sam3VideoSegmenter | None = None,
    image_segmenter: Sam3ImageSegmenter | None = None,
    embedder: Any | None = None,
) -> VideoObservationBuildResult:
    video_path = Path(video_path)
    reference_path = Path(reference_path)
    output_path = Path(output_path)
    assets = Path(assets_dir) if assets_dir else output_path.parent / f"{output_path.stem}_assets"
    masks_dir = assets / "masks"
    crops_dir = assets / "crops"
    scene = load_reference_scene(reference_path)
    config = resolve_reference_embedding_config(config, scene)
    expected_embedding_backend = _reference_embedding_backend(scene)
    expected_embedding_dimension = _reference_embedding_dimension(scene)
    if embedder is not None:
        _validate_embedding_backend(embedder, expected_embedding_backend)

    frame_map: dict[int, Image.Image] = {}
    timeline: VideoTimeline | None = None
    frame_error = None
    try:
        if video_path.suffix.lower() in VIDEO_EXTENSIONS:
            try:
                timeline = probe_video_timeline(video_path)
            except FrameSourceError:
                if config.video_sample_fps is not None:
                    raise
        frame_map = ensure_frame_map(
            frames_path or video_path,
            config,
            timeline=timeline,
        )
    except FrameSourceError as exc:
        frame_error = str(exc)

    if frame_error and config.video_sample_fps is not None:
        raise FrameSourceError(frame_error)

    sampled = (
        sample_indices(frame_map, config, timeline=timeline)
        if frame_map
        else []
    )
    frame_size = _first_frame_size(frame_map)
    prompt_list = _video_prompts(scene, prompts, config)
    fallback_info: dict[str, Any] = {
        "enabled": bool(config.video_empty_frame_image_pcs_fallback),
        "attempted_frame_indices": [],
        "recovered_frame_indices": [],
        "unresolved_frame_indices": [],
    }

    segmentation_mode = (
        "precomputed"
        if precomputed_segments_path is not None
        else config.video_segmentation_mode
    )
    if precomputed_segments_path is not None:
        segments = load_segment_instances(precomputed_segments_path)
        _load_external_masks(
            segments,
            base_dir=Path(precomputed_segments_path).parent,
        )
        if not sampled:
            sampled = _sample_from_segment_indices(
                segments,
                config,
                timeline=timeline,
            )
        sample_set = set(sampled)
        segments = [s for s in segments if s.frame_index in sample_set]
    elif segmentation_mode == "framewise":
        if frame_error:
            raise FrameSourceError(frame_error)
        segments = _segment_sampled_frames(
            sampled=sampled,
            frame_map=frame_map,
            prompts=prompt_list,
            config=config,
            image_segmenter=image_segmenter,
        )
    else:
        if frame_error and config.strict_models:
            raise FrameSourceError(frame_error)
        owns_segmenter = video_segmenter is None
        segmenter = video_segmenter if video_segmenter is not None else Sam3VideoSegmenter(config)
        try:
            sample_set = set(sampled) if sampled else None
            segments = segmenter.segment_by_prompts(
                video_path,
                prompt_list,
                frame_indices=sample_set,
                frame_size=frame_size,
            )
        finally:
            if owns_segmenter:
                del segmenter
                gc.collect()
                _empty_cuda_cache()
        if not sampled:
            sampled = _sample_from_segment_indices(
                segments,
                config,
                timeline=timeline,
            )
            sample_set = set(sampled)
            segments = [s for s in segments if s.frame_index in sample_set]

    segments = _filter_segments(segments, frame_size, config)
    segments = _deduplicate_by_frame(segments, config)
    if (
        config.video_empty_frame_image_pcs_fallback
        and segmentation_mode == "tracking"
    ):
        segments, fallback_info = _recover_empty_sampled_frames(
            segments,
            sampled=sampled,
            frame_map=frame_map,
            prompts=prompt_list,
            config=config,
            image_segmenter=image_segmenter,
        )

    embedder = embedder or build_embedding_extractor(config)
    _validate_embedding_backend(embedder, expected_embedding_backend)
    grouped: dict[int, list[SegmentInstance]] = {}
    pending_embeddings: list[tuple[SegmentInstance, Image.Image]] = []
    for segment in segments:
        if segment.frame_index is None:
            continue
        if segment.timestamp is None:
            segment.timestamp = _frame_timestamp(timeline, segment.frame_index)
        frame = frame_map.get(segment.frame_index)
        crop_path = None
        mask_path = None
        crop = None
        if frame is not None:
            crop = crop_object(
                frame,
                segment.bbox,
                segment.mask,
                padding=config.crop_padding_px,
            )
            if config.save_masks and segment.mask is not None:
                mask_path = masks_dir / f"frame_{segment.frame_index:06d}_{segment.segment_id}.png"
                save_mask_png(segment.mask, mask_path)
                segment.mask_path = _relative_path(mask_path, output_path.parent)
            if config.save_crops:
                crop_path = crops_dir / f"frame_{segment.frame_index:06d}_{segment.segment_id}.png"
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                crop.save(crop_path)
            segment.crop_path = _relative_path(crop_path, output_path.parent)
            segment.attributes.update(_visual_attributes(crop, segment))
            if segment.embedding is None:
                pending_embeddings.append((segment, crop))
        segment.metadata.update({
            "frame_index": segment.frame_index,
            "timestamp": segment.timestamp,
            "embedding_backend": getattr(embedder, "backend_name", config.embedding_backend),
        })
        grouped.setdefault(segment.frame_index, []).append(segment)

    if pending_embeddings:
        crops = [crop for _, crop in pending_embeddings]
        embeddings = _embed_many(embedder, crops)
        if len(embeddings) != len(pending_embeddings):
            raise RuntimeError(
                f"embedding extractor returned {len(embeddings)} vectors for "
                f"{len(pending_embeddings)} crops"
            )
        for (segment, _), embedding in zip(pending_embeddings, embeddings):
            segment.embedding = embedding
    _validate_embedding_dimensions(segments, expected_embedding_dimension)

    windows: list[dict[str, Any]] = []
    for frame_index in sampled:
        frame = frame_map.get(frame_index)
        timestamp = _frame_timestamp(timeline, frame_index)
        size = frame.size if frame is not None else _infer_frame_size(grouped.get(frame_index, []), frame_size)
        objects = []
        for idx, segment in enumerate(grouped.get(frame_index, [])):
            observed_id = f"f{frame_index:06d}_obj_{idx:04d}"
            objects.append(segment.to_observed_object(observed_id))
        windows.append({
            "window_id": f"frame_{frame_index:06d}",
            "frame_index": frame_index,
            "timestamp": timestamp,
            "frame_size": [int(size[0]), int(size[1])],
            "objects": objects,
            "metadata": {"num_observed_objects": len(objects)},
        })

    metadata: dict[str, Any] = {
        "video_path": str(video_path),
        "reference_path": str(reference_path),
        "sampled_frame_indices": sampled,
        "preprocessing": {
            "segmenter": "precomputed" if precomputed_segments_path else "sam3",
            "segmentation_mode": segmentation_mode,
            "prompts": prompt_list,
            "embedding_backend": getattr(embedder, "backend_name", config.embedding_backend),
            "config": config.to_dict(),
            "sampling": {
                "mode": "timestamp" if config.video_sample_fps is not None else "stride",
                "source_fps": timeline.average_fps if timeline is not None else None,
                "target_fps": config.video_sample_fps,
                "stride": None if config.video_sample_fps is not None else config.video_sample_stride,
                "skip_initial_seconds": config.video_skip_initial_seconds,
                "sampled_timestamps": [
                    _frame_timestamp(timeline, frame_index)
                    for frame_index in sampled
                ],
            },
            "empty_frame_image_pcs_fallback": fallback_info,
        },
    }
    requested_zoom = scene.metadata.get("requested_zoom") or scene.metadata.get("prompt_zoom")
    if requested_zoom is not None:
        metadata["requested_zoom"] = requested_zoom
        metadata["prompt_zoom"] = requested_zoom
    if frame_error:
        metadata["frame_loading_warning"] = frame_error

    observation = {
        "schema_version": 2,
        "video_id": video_id or video_path.stem,
        "windows": windows,
        "metadata": metadata,
    }
    write_json(output_path, observation)
    return VideoObservationBuildResult(
        observation_path=output_path,
        assets_dir=assets,
        observation=observation,
        sampled_frame_indices=sampled,
        num_segments=sum(len(items) for items in grouped.values()),
    )


_EMBEDDING_CONTRACT_FIELDS = (
    "dinov3_repo_or_dir",
    "dinov3_model_name",
    "dinov3_weights_path",
    "embedding_image_size",
    "color_hist_bins",
)
_REFERENCE_RUNTIME_DEFAULT_FIELDS = (
    "sam3_checkpoint_path",
    "sam3_bpe_path",
    "sam3_model_id",
)


def resolve_reference_embedding_config(
    config: VisionConfig,
    scene: Any,
) -> VisionConfig:
    expected = _reference_embedding_backend(scene)
    requested = config.embedding_backend.lower()
    if expected is not None and requested not in {"auto", expected}:
        raise ValueError(
            "embedding backend mismatch: reference uses "
            f"{expected}, observation requested {requested}"
        )

    preprocessing = scene.metadata.get("preprocessing", {})
    reference_config = (
        preprocessing.get("config", {})
        if isinstance(preprocessing, dict)
        else {}
    )
    updates: dict[str, Any] = {}
    if isinstance(reference_config, dict):
        for field_name in _REFERENCE_RUNTIME_DEFAULT_FIELDS:
            if getattr(config, field_name) is None and reference_config.get(field_name):
                updates[field_name] = reference_config[field_name]
        if expected is not None and requested == "auto":
            updates["embedding_backend"] = expected
            for field_name in _EMBEDDING_CONTRACT_FIELDS:
                if field_name in reference_config:
                    updates[field_name] = reference_config[field_name]
    from multimem_bench.paths import resolve_config_path

    for field_name in (*_REFERENCE_RUNTIME_DEFAULT_FIELDS, *_EMBEDDING_CONTRACT_FIELDS):
        value = updates.get(field_name)
        if isinstance(value, str):
            updates[field_name] = resolve_config_path(value)
    return replace(config, **updates) if updates else config


def _reference_embedding_backend(scene: Any) -> str | None:
    values: set[str] = set()
    preprocessing = scene.metadata.get("preprocessing", {})
    if isinstance(preprocessing, dict) and preprocessing.get("embedding_backend"):
        values.add(str(preprocessing["embedding_backend"]).lower())
    for obj in scene.objects.values():
        value = obj.metadata.get("embedding_backend")
        if value:
            values.add(str(value).lower())
    if len(values) > 1:
        raise ValueError(
            "reference annotation contains inconsistent embedding backends: "
            + ", ".join(sorted(values))
        )
    return next(iter(values), None)


def _reference_embedding_dimension(scene: Any) -> int | None:
    dimensions = {
        len(obj.embedding)
        for obj in scene.objects.values()
        if obj.embedding is not None
    }
    if len(dimensions) > 1:
        raise ValueError(
            "reference annotation contains inconsistent embedding dimensions: "
            + ", ".join(str(value) for value in sorted(dimensions))
        )
    return next(iter(dimensions), None)


def _validate_embedding_backend(
    embedder: Any,
    expected_backend: str | None,
) -> None:
    actual = str(getattr(embedder, "backend_name", "unknown")).lower()
    if expected_backend is not None and actual != expected_backend:
        raise ValueError(
            "embedding backend mismatch: reference uses "
            f"{expected_backend}, observation produced {actual}"
        )


def _validate_embedding_dimensions(
    segments: list[SegmentInstance],
    expected_dimension: int | None,
) -> None:
    if expected_dimension is None:
        return
    observed_dimensions = {
        len(segment.embedding)
        for segment in segments
        if segment.embedding is not None
    }
    if observed_dimensions and observed_dimensions != {expected_dimension}:
        actual = ", ".join(str(value) for value in sorted(observed_dimensions))
        raise ValueError(
            "embedding dimension mismatch: reference uses "
            f"{expected_dimension}, observation produced {actual}"
        )


def _video_prompts(
    scene: Any,
    prompts: Iterable[str] | None,
    config: VisionConfig,
) -> list[str]:
    if prompts:
        values = [str(p).strip() for p in prompts if str(p).strip()]
    elif config.video_prompts:
        values = [p.strip() for p in config.video_prompts if p.strip()]
    elif config.video_use_reference_categories:
        values = sorted({
            obj.category.strip()
            for obj in scene.objects.values()
            if obj.category and obj.category.strip()
        })
    else:
        values = []
    if not values:
        values = list(config.reference_foreground_prompts)
    return values[: max(1, int(config.video_max_prompts))]


def _frame_timestamp(timeline: VideoTimeline | None, frame_index: int) -> float | None:
    if timeline is None or not (0 <= frame_index < len(timeline.frame_timestamps)):
        return None
    return float(timeline.frame_timestamps[frame_index])


def _segment_sampled_frames(
    *,
    sampled: list[int],
    frame_map: dict[int, Image.Image],
    prompts: list[str],
    config: VisionConfig,
    image_segmenter: Sam3ImageSegmenter | None = None,
) -> list[SegmentInstance]:
    owns_segmenter = image_segmenter is None
    segmenter = (
        image_segmenter
        if image_segmenter is not None
        else Sam3ImageSegmenter(config)
    )
    segments: list[SegmentInstance] = []
    try:
        for frame_index in sampled:
            frame = frame_map[frame_index]
            found = segmenter.segment_by_prompts(
                frame,
                prompts,
                id_prefix=f"framewise_f{frame_index:06d}",
            )
            for segment in found:
                segment.frame_index = frame_index
                segment.track_id = None
                segment.metadata["segment_source"] = "sam3_image_framewise"
            found = _filter_segments(found, frame.size, config)
            segments.extend(_deduplicate_by_frame(found, config))
    finally:
        if owns_segmenter:
            del segmenter
            gc.collect()
            _empty_cuda_cache()
    return segments


def _recover_empty_sampled_frames(
    segments: list[SegmentInstance],
    *,
    sampled: list[int],
    frame_map: dict[int, Image.Image],
    prompts: list[str],
    config: VisionConfig,
    image_segmenter: Sam3ImageSegmenter | None = None,
) -> tuple[list[SegmentInstance], dict[str, Any]]:
    present = {
        int(segment.frame_index)
        for segment in segments
        if segment.frame_index is not None
    }
    attempted = [frame_index for frame_index in sampled if frame_index not in present]
    recovered: list[int] = []
    fallback_segments: list[SegmentInstance] = []
    warning = None

    if attempted:
        owns_segmenter = image_segmenter is None
        segmenter = None
        try:
            segmenter = image_segmenter if image_segmenter is not None else Sam3ImageSegmenter(config)
            for frame_index in attempted:
                frame = frame_map[frame_index]
                found = segmenter.segment_by_prompts(
                    frame,
                    prompts,
                    id_prefix=f"fallback_f{frame_index:06d}",
                )
                for segment in found:
                    segment.frame_index = frame_index
                    segment.track_id = None
                    segment.metadata["segment_source"] = "sam3_image_pcs_fallback"
                found = _filter_segments(found, frame.size, config)
                found = _deduplicate_by_frame(found, config)
                if found:
                    recovered.append(frame_index)
                    fallback_segments.extend(found)
        except Sam3UnavailableError as exc:
            if config.strict_models:
                raise
            warning = str(exc)
        finally:
            if owns_segmenter and segmenter is not None:
                del segmenter
                gc.collect()
                _empty_cuda_cache()

    recovered_set = set(recovered)
    info: dict[str, Any] = {
        "enabled": True,
        "attempted_frame_indices": attempted,
        "recovered_frame_indices": recovered,
        "unresolved_frame_indices": [
            frame_index for frame_index in attempted if frame_index not in recovered_set
        ],
    }
    if warning:
        info["warning"] = warning
    return segments + fallback_segments, info


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        return


def _embed_many(embedder: Any, images: list[Image.Image]) -> list[list[float]]:
    method = getattr(embedder, "embed_many", None)
    if callable(method):
        return method(images)
    return [embedder.embed(image) for image in images]


def _filter_segments(
    segments: list[SegmentInstance],
    frame_size: tuple[int, int] | None,
    config: VisionConfig,
) -> list[SegmentInstance]:
    frame_area = float(frame_size[0] * frame_size[1]) if frame_size else None
    out: list[SegmentInstance] = []
    for segment in segments:
        area = _segment_area(segment)
        if area < config.min_mask_area_px:
            continue
        if frame_area:
            ratio = area / max(1.0, frame_area)
            if ratio < config.min_mask_area_ratio or ratio > config.max_mask_area_ratio:
                continue
        if segment.confidence < config.sam3_confidence_threshold:
            continue
        out.append(segment)
    return out


def _deduplicate_by_frame(
    segments: list[SegmentInstance],
    config: VisionConfig,
) -> list[SegmentInstance]:
    grouped: dict[int | None, list[SegmentInstance]] = {}
    for segment in segments:
        grouped.setdefault(segment.frame_index, []).append(segment)

    kept_all: list[SegmentInstance] = []
    for group in grouped.values():
        kept: list[SegmentInstance] = []
        for segment in sorted(group, key=lambda s: (s.confidence, _segment_area(s)), reverse=True):
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
        kept_all.extend(kept)
    return kept_all


def _load_external_masks(segments: list[SegmentInstance], *, base_dir: Path) -> None:
    for segment in segments:
        if segment.mask is not None or not segment.mask_path:
            continue
        path = Path(segment.mask_path)
        if not path.is_absolute():
            path = base_dir / path
        if path.exists():
            segment.mask = Image.open(path).convert("L")


def _segment_area(segment: SegmentInstance) -> float:
    if segment.mask is not None:
        try:
            return mask_area(segment.mask)
        except Exception:
            pass
    return bbox_area(segment.bbox)


def _first_frame_size(frame_map: dict[int, Image.Image]) -> tuple[int, int] | None:
    if not frame_map:
        return None
    first_key = sorted(frame_map)[0]
    return frame_map[first_key].size


def _infer_frame_size(
    segments: list[SegmentInstance],
    default: tuple[int, int] | None,
) -> tuple[int, int]:
    if default:
        return default
    width = 1
    height = 1
    for segment in segments:
        width = max(width, int(segment.bbox[2]))
        height = max(height, int(segment.bbox[3]))
    return (width, height)


def _sample_from_segment_indices(
    segments: list[SegmentInstance],
    config: VisionConfig,
    *,
    timeline: VideoTimeline | None = None,
) -> list[int]:
    timestamp_by_index: dict[int, float] = {}
    indices_set: set[int] = set()
    for segment in segments:
        if segment.frame_index is None:
            continue
        index = int(segment.frame_index)
        indices_set.add(index)
        if segment.timestamp is not None:
            timestamp_by_index.setdefault(index, float(segment.timestamp))
    indices = sorted(indices_set)
    if not indices:
        return []
    skip_seconds = float(config.video_skip_initial_seconds)
    if skip_seconds < 0:
        raise ValueError("video_skip_initial_seconds must be non-negative")
    if skip_seconds > 0:
        if timeline is not None:
            indices = [
                index for index in indices
                if index < len(timeline.frame_timestamps)
                and timeline.frame_timestamps[index] >= skip_seconds - 1e-9
            ]
        elif timestamp_by_index:
            indices = [
                index for index in indices
                if timestamp_by_index.get(index, -1.0) >= skip_seconds - 1e-9
            ]
        else:
            raise FrameSourceError(
                "video_skip_initial_seconds requires timestamps in precomputed "
                "segments or an encoded video timeline"
            )
    if not indices:
        return []
    if config.video_sample_fps is not None:
        if timeline is not None:
            available_timestamps = {
                index: timeline.frame_timestamps[index]
                for index in indices
                if 0 <= index < len(timeline.frame_timestamps)
            }
        else:
            available_timestamps = {
                index: timestamp_by_index[index]
                for index in indices
                if index in timestamp_by_index
            }
        if len(available_timestamps) != len(indices):
            raise FrameSourceError(
                "video_sample_fps requires timestamps for every precomputed "
                "segment frame"
            )
        available_indices = sorted(available_timestamps)
        available_times = tuple(
            float(available_timestamps[index]) for index in available_indices
        )
        synthetic_timeline = VideoTimeline(
            frame_timestamps=available_times,
            average_fps=None,
            duration=(
                available_times[-1] + 1.0 / float(config.video_sample_fps)
                if available_times
                else None
            ),
        )
        positions = timestamp_sample_indices(
            synthetic_timeline,
            config.video_sample_fps,
            max_frames=config.video_max_frames,
            start_time=skip_seconds,
        )
        return [available_indices[position] for position in positions]
    stride = max(1, int(config.video_sample_stride))
    sampled = indices[::stride]
    if config.video_max_frames is not None:
        sampled = sampled[: max(0, int(config.video_max_frames))]
    return sampled


def _relative_path(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)
