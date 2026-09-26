"""Reference-first geometry inference, query sampling, and content-addressed cache."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image

from multimem_bench.schema import ObjectSignature, ReferenceScene, VideoObservation
from multimem_bench.vision.mask_utils import mask_to_bool_array
from multimem_bench.v2.config import V2EvaluationConfig
from multimem_bench.v2.geometry.artifact import GeometryBundle
from multimem_bench.v2.geometry.base import (
    GeometryBackend,
    GeometryBackendUnavailable,
    GeometryPrediction,
)


_GEOMETRY_CACHE_CONFIG_FIELDS = (
    "protocol_version",
    "queries_per_instance",
    "background_queries",
    "ransac_iterations",
    "max_reprojection_error_ratio",
    "geometry_backend",
    "geometry_model_id",
    "geometry_device",
    "geometry_dtype",
    "geometry_image_size",
    "geometry_checkpoint",
)


@dataclass(frozen=True)
class GeometryPreparationResult:
    manifest_path: Path
    status: str
    cache_hit: bool
    cache_key: str


def prepare_geometry_artifact(
    *,
    reference_image: str | Path | Image.Image,
    scene: ReferenceScene,
    observation: VideoObservation,
    frame_images: Mapping[int, Image.Image],
    output_dir: str | Path,
    config: V2EvaluationConfig | None = None,
    backend: GeometryBackend | None = None,
    force: bool = False,
) -> GeometryPreparationResult:
    cfg = config or V2EvaluationConfig()
    cfg.validate()
    if len(scene.objects) > cfg.max_instances:
        raise ValueError(
            f"reference contains {len(scene.objects)} instances, exceeding "
            f"max_instances={cfg.max_instances}"
        )
    reference = _reference_in_annotation_space(reference_image, scene)
    frame_indices = [
        int(window.frame_index)
        for window in observation.windows
        if window.frame_index is not None
    ]
    missing = [index for index in frame_indices if index not in frame_images]
    if missing:
        raise ValueError(f"missing sampled frames for geometry: {missing}")
    source_images = [reference] + [frame_images[index].convert("RGB") for index in frame_indices]
    cache_key = _cache_key(source_images, scene, frame_indices, cfg)
    root = Path(output_dir)
    manifest_path = root / "geometry_manifest.json"
    if not force and manifest_path.exists():
        cached = GeometryBundle.load(manifest_path)
        if cached.status == "available" and cached.metadata.get("cache_key") == cache_key:
            return GeometryPreparationResult(
                manifest_path=manifest_path,
                status=cached.status,
                cache_hit=True,
                cache_key=cache_key,
            )

    prepared, transforms = _letterbox_images(source_images, cfg.geometry_image_size)
    query_points, query_ids = _reference_queries(
        scene,
        reference.size,
        transforms[0],
        cfg,
    )
    selected_backend = backend
    common_metadata = {
        "cache_key": cache_key,
        "reference_view_index": 0,
        "frame_order": [-1, *frame_indices],
        "preprocessing": {
            "mode": "letterbox",
            "model_size": [cfg.geometry_image_size, cfg.geometry_image_size],
        },
        "checkpoint": cfg.geometry_checkpoint,
        "model_id": cfg.geometry_model_id,
    }
    try:
        if selected_backend is None:
            selected_backend = build_geometry_backend(cfg)
        prediction = selected_backend.predict(prepared, query_points)
        prediction.validate(frame_count=len(prepared), query_count=len(query_points))
        prediction = _normalize_prediction_confidence(prediction)
        camera_metadata = _refine_cameras_with_background(
            prediction,
            query_points,
            query_ids,
            cfg,
        )
        bundle = GeometryBundle(
            status="available",
            backend=selected_backend.name,
            frame_indices=np.asarray([-1, *frame_indices], dtype=np.int64),
            extrinsics=np.asarray(prediction.extrinsics, dtype=np.float32),
            intrinsics=np.asarray(prediction.intrinsics, dtype=np.float32),
            world_points=np.asarray(prediction.world_points, dtype=np.float32),
            point_confidence=np.asarray(prediction.point_confidence, dtype=np.float32),
            depth=np.asarray(prediction.depth, dtype=np.float32),
            depth_confidence=np.asarray(prediction.depth_confidence, dtype=np.float32),
            source_sizes=np.asarray([image.size for image in source_images], dtype=np.int64),
            model_size=(cfg.geometry_image_size, cfg.geometry_image_size),
            image_transforms=np.asarray(transforms, dtype=np.float32),
            query_points=np.asarray(query_points, dtype=np.float32),
            query_reference_ids=tuple(query_ids),
            tracks=np.asarray(prediction.tracks, dtype=np.float32),
            track_visibility=np.asarray(prediction.track_visibility, dtype=np.float32),
            track_confidence=np.asarray(prediction.track_confidence, dtype=np.float32),
            camera_confidence=np.asarray(prediction.camera_confidence, dtype=np.float32),
            metadata={
                **common_metadata,
                **prediction.metadata,
                "camera_refinement": camera_metadata,
            },
        )
    except (GeometryBackendUnavailable, ImportError, OSError, RuntimeError) as exc:
        if cfg.geometry_strict:
            if isinstance(exc, GeometryBackendUnavailable):
                raise
            raise GeometryBackendUnavailable(str(exc)) from exc
        bundle = GeometryBundle.unavailable(
            backend=getattr(selected_backend, "name", cfg.geometry_backend),
            error=str(exc),
            metadata=common_metadata,
        )
    paths = bundle.save(root)
    return GeometryPreparationResult(
        manifest_path=paths["manifest"],
        status=bundle.status,
        cache_hit=False,
        cache_key=cache_key,
    )


def build_geometry_backend(config: V2EvaluationConfig) -> GeometryBackend:
    from multimem_bench.v2.geometry.vggt import VGGTBackend, VGGTOmegaBackend

    backend = config.geometry_backend.lower()
    if backend == "vggt":
        return VGGTBackend(config)
    if backend in {"vggt_omega", "omega"}:
        return VGGTOmegaBackend(config)
    raise GeometryBackendUnavailable(f"unsupported geometry backend: {config.geometry_backend}")


def _load_image(value: str | Path | Image.Image) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    return Image.open(value).convert("RGB")


def _reference_in_annotation_space(
    value: str | Path | Image.Image,
    scene: ReferenceScene,
) -> Image.Image:
    reference = _load_image(value)
    raw_size = scene.metadata.get("image_size") or scene.metadata.get("processed_image_size")
    if not isinstance(raw_size, (list, tuple)) or len(raw_size) != 2:
        return reference
    expected = (int(raw_size[0]), int(raw_size[1]))
    if reference.size == expected:
        return reference
    processed_path = scene.metadata.get("processed_image_path")
    if processed_path:
        candidate = Path(str(processed_path))
        if not candidate.is_absolute() and scene.artifact_dir:
            candidate = Path(scene.artifact_dir) / candidate
        if candidate.exists():
            processed = Image.open(candidate).convert("RGB")
            if processed.size == expected:
                return processed
    raise ValueError(
        f"reference image size {reference.size} does not match annotation coordinate "
        f"space {expected}; provide metadata.processed_image_path or the processed image"
    )


def _letterbox_images(
    images: Sequence[Image.Image],
    target_size: int,
) -> tuple[list[Image.Image], list[tuple[float, float, float, float]]]:
    prepared: list[Image.Image] = []
    transforms: list[tuple[float, float, float, float]] = []
    for image in images:
        rgb = image.convert("RGB")
        width, height = rgb.size
        scale = min(target_size / width, target_size / height)
        new_width = max(1, min(target_size, int(round(width * scale))))
        new_height = max(1, min(target_size, int(round(height * scale))))
        resized = rgb.resize((new_width, new_height), Image.Resampling.BICUBIC)
        offset_x = (target_size - new_width) // 2
        offset_y = (target_size - new_height) // 2
        canvas = Image.new("RGB", (target_size, target_size), "white")
        canvas.paste(resized, (offset_x, offset_y))
        prepared.append(canvas)
        transforms.append(
            (
                new_width / width,
                new_height / height,
                float(offset_x),
                float(offset_y),
            )
        )
    return prepared, transforms


def _reference_queries(
    scene: ReferenceScene,
    image_size: tuple[int, int],
    transform: tuple[float, float, float, float],
    config: V2EvaluationConfig,
) -> tuple[np.ndarray, list[str]]:
    width, height = image_size
    foreground = np.zeros((height, width), dtype=bool)
    source_points: list[tuple[float, float]] = []
    query_ids: list[str] = []
    for reference_id, instance in scene.objects.items():
        mask = _instance_mask(instance, image_size)
        foreground |= mask
        points = _sample_mask_points(mask, config.queries_per_instance)
        source_points.extend(points)
        query_ids.extend([reference_id] * len(points))
    background_points = _sample_mask_points(
        _valid_reference_canvas(scene, image_size) & ~foreground,
        config.background_queries,
    )
    source_points.extend(background_points)
    query_ids.extend(["__background__"] * len(background_points))
    if not source_points:
        raise ValueError("reference image produced no geometry query points")
    scale_x, scale_y, offset_x, offset_y = transform
    model_points = np.asarray(
        [
            [x * scale_x + offset_x, y * scale_y + offset_y]
            for x, y in source_points
        ],
        dtype=np.float32,
    )
    return model_points, query_ids


def _instance_mask(
    instance: ObjectSignature,
    image_size: tuple[int, int],
) -> np.ndarray:
    width, height = image_size
    if instance.mask is not None:
        mask = mask_to_bool_array(instance.mask)
        if mask.shape != (height, width):
            image = Image.fromarray(mask.astype(np.uint8) * 255)
            image = image.resize((width, height), Image.Resampling.NEAREST)
            mask = np.asarray(image) > 0
        return mask
    mask = np.zeros((height, width), dtype=bool)
    if instance.bbox is None:
        return mask
    x0, y0, x1, y1 = instance.bbox
    left = max(0, min(width, int(math_floor(x0))))
    top = max(0, min(height, int(math_floor(y0))))
    right = max(left, min(width, int(math_ceil(x1))))
    bottom = max(top, min(height, int(math_ceil(y1))))
    mask[top:bottom, left:right] = True
    return mask


def _valid_reference_canvas(
    scene: ReferenceScene,
    image_size: tuple[int, int],
) -> np.ndarray:
    width, height = image_size
    valid = np.ones((height, width), dtype=bool)
    source_size = scene.metadata.get("source_image_size")
    padding = scene.metadata.get("resize_padding")
    scale = scene.metadata.get("resize_scale")
    if (
        not isinstance(source_size, (list, tuple))
        or len(source_size) != 2
        or not isinstance(padding, (list, tuple))
        or len(padding) < 2
        or scale is None
    ):
        return valid
    left = max(0, int(round(float(padding[0]))))
    top = max(0, int(round(float(padding[1]))))
    content_width = min(width - left, int(round(float(source_size[0]) * float(scale))))
    content_height = min(height - top, int(round(float(source_size[1]) * float(scale))))
    if content_width <= 0 or content_height <= 0:
        return valid
    valid[:] = False
    valid[top : top + content_height, left : left + content_width] = True
    return valid


def _sample_mask_points(mask: np.ndarray, maximum: int) -> list[tuple[float, float]]:
    if maximum <= 0 or not mask.any():
        return []
    height, width = mask.shape
    cells = max(1, int(np.ceil(np.sqrt(maximum))))
    points: list[tuple[float, float]] = []
    for row in range(cells):
        y0 = int(round(row * height / cells))
        y1 = int(round((row + 1) * height / cells))
        for col in range(cells):
            x0 = int(round(col * width / cells))
            x1 = int(round((col + 1) * width / cells))
            ys, xs = np.nonzero(mask[y0:y1, x0:x1])
            if len(xs) == 0:
                continue
            center_x = (x1 - x0 - 1) / 2.0
            center_y = (y1 - y0 - 1) / 2.0
            index = int(np.argmin((xs - center_x) ** 2 + (ys - center_y) ** 2))
            points.append((float(x0 + xs[index]), float(y0 + ys[index])))
            if len(points) >= maximum:
                return points
    if len(points) < maximum:
        ys, xs = np.nonzero(mask)
        indices = np.linspace(0, len(xs) - 1, min(maximum, len(xs)), dtype=int)
        seen = set(points)
        for index in indices:
            point = (float(xs[index]), float(ys[index]))
            if point not in seen:
                points.append(point)
                seen.add(point)
            if len(points) >= maximum:
                break
    return points


def _cache_key(
    images: Sequence[Image.Image],
    scene: ReferenceScene,
    frame_indices: list[int],
    config: V2EvaluationConfig,
) -> str:
    digest = hashlib.sha256()
    digest.update(scene.scene_id.encode("utf-8"))
    digest.update(json.dumps(frame_indices).encode("utf-8"))
    digest.update(
        json.dumps(
            {
                name: getattr(config, name)
                for name in _GEOMETRY_CACHE_CONFIG_FIELDS
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    )
    for image in images:
        rgb = image.convert("RGB")
        digest.update(str(rgb.size).encode("ascii"))
        digest.update(rgb.tobytes())
    for reference_id, instance in scene.objects.items():
        digest.update(reference_id.encode("utf-8"))
        digest.update(str(instance.bbox).encode("utf-8"))
        if instance.mask is not None:
            digest.update(mask_to_bool_array(instance.mask).tobytes())
    return digest.hexdigest()


def _normalize_prediction_confidence(prediction: GeometryPrediction) -> GeometryPrediction:
    prediction.point_confidence = _normalize_confidence(prediction.point_confidence)
    prediction.depth_confidence = _normalize_confidence(prediction.depth_confidence)
    prediction.track_visibility = _normalize_confidence(prediction.track_visibility)
    prediction.track_confidence = _normalize_confidence(prediction.track_confidence)
    prediction.camera_confidence = _normalize_confidence(prediction.camera_confidence)
    return prediction


def _normalize_confidence(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values)
    valid = values[finite]
    if float(valid.min()) >= 0.0 and float(valid.max()) <= 1.0:
        return np.clip(values, 0.0, 1.0)
    low, high = np.percentile(valid, [10, 90])
    if float(high - low) <= 1e-6:
        result = np.zeros_like(values)
        result[finite] = 1.0 if float(high) > 0.0 else 0.0
        return result
    normalized = (values - low) / max(float(high - low), 1e-6)
    normalized[~finite] = 0.0
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def _refine_cameras_with_background(
    prediction: GeometryPrediction,
    query_points: np.ndarray,
    query_ids: list[str],
    config: V2EvaluationConfig,
) -> list[dict[str, object]]:
    background = np.flatnonzero(np.asarray(query_ids) == "__background__")
    reports: list[dict[str, object]] = [
        {"view_index": 0, "status": "reference"}
    ]
    if len(background) < 6:
        return reports + [
            {"view_index": index, "status": "insufficient_background_points"}
            for index in range(1, len(prediction.extrinsics))
        ]
    try:
        import cv2
    except ImportError:
        return reports + [
            {"view_index": index, "status": "opencv_unavailable"}
            for index in range(1, len(prediction.extrinsics))
        ]
    reference_points = _sample_array(prediction.world_points[0], query_points[background])
    for view_index in range(1, len(prediction.extrinsics)):
        image_points = prediction.tracks[view_index, background]
        valid = (
            np.isfinite(reference_points).all(axis=1)
            & np.isfinite(image_points).all(axis=1)
            & (prediction.track_visibility[view_index, background] >= 0.5)
        )
        if int(valid.sum()) < 6:
            reports.append({"view_index": view_index, "status": "insufficient_valid_background"})
            continue
        success, rotation_vector, translation, inliers = cv2.solvePnPRansac(
            reference_points[valid].astype(np.float64),
            image_points[valid].astype(np.float64),
            prediction.intrinsics[view_index].astype(np.float64),
            None,
            iterationsCount=max(32, config.ransac_iterations),
            reprojectionError=max(1.0, config.max_reprojection_error_ratio * prediction.world_points.shape[2]),
            confidence=0.99,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not success or inliers is None or len(inliers) < 6:
            reports.append({"view_index": view_index, "status": "pnp_failed"})
            continue
        rotation, _ = cv2.Rodrigues(rotation_vector)
        prediction.extrinsics[view_index, :3, :3] = rotation
        prediction.extrinsics[view_index, :3, 3] = translation.reshape(3)
        inlier_ratio = float(len(inliers) / valid.sum())
        prediction.camera_confidence[view_index] = inlier_ratio
        reports.append(
            {
                "view_index": view_index,
                "status": "refined",
                "inlier_ratio": inlier_ratio,
                "num_inliers": int(len(inliers)),
            }
        )
    return reports


def _sample_array(array: np.ndarray, points: np.ndarray) -> np.ndarray:
    height, width = array.shape[:2]
    xy = np.rint(points).astype(np.int64)
    xy[:, 0] = np.clip(xy[:, 0], 0, width - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, height - 1)
    return np.asarray(array[xy[:, 1], xy[:, 0]])


def math_floor(value: float) -> int:
    return int(np.floor(float(value)))


def math_ceil(value: float) -> int:
    return int(np.ceil(float(value)))
