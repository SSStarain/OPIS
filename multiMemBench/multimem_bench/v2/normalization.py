"""Instance motion normalization without per-instance scale fitting."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from multimem_bench.v2.config import V2EvaluationConfig
from multimem_bench.v2.math_utils import clamp01


@dataclass(frozen=True)
class RigidFit:
    valid: bool
    rotation: np.ndarray
    translation: np.ndarray
    scale: float
    scale_ratio: float
    aligned_points: np.ndarray
    inlier_mask: np.ndarray
    inlier_ratio: float
    median_residual: float
    median_residual_ratio: float
    error: str | None = None


@dataclass(frozen=True)
class CanonicalizationResult:
    valid: bool
    mode: str
    geometry_score: float | None
    scale_score: float | None
    topology_score: float | None
    scale_ratio: float
    inlier_ratio: float
    residual_ratio: float | None
    transform: np.ndarray | None = None
    error: str | None = None


@dataclass(frozen=True)
class _LocalStructure:
    score: float
    scale_ratio: float
    scale_score: float
    inlier_ratio: float
    median_log_error: float


def fit_rigid_transform(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | Sequence[float] | None,
    config: V2EvaluationConfig | None = None,
) -> RigidFit:
    cfg = config or V2EvaluationConfig()
    source = _points(source)
    target = _points(target)
    if source.shape != target.shape or source.shape[0] < 3:
        return _invalid_fit(source, "rigid fitting requires at least three paired 3D points")
    weight_array = np.ones(source.shape[0], dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if weight_array.shape != (source.shape[0],):
        return _invalid_fit(source, "weights must match the point count")
    finite = (
        np.isfinite(source).all(axis=1)
        & np.isfinite(target).all(axis=1)
        & np.isfinite(weight_array)
        & (weight_array > 0)
    )
    if int(finite.sum()) < 3:
        return _invalid_fit(source, "fewer than three finite weighted correspondences")
    source_valid = source[finite]
    target_valid = target[finite]
    weights_valid = weight_array[finite]
    diameter = max(_diameter(target_valid), 1e-9)
    threshold = cfg.rigid_inlier_threshold_ratio * diameter
    rng = np.random.default_rng(cfg.random_seed)
    best_inliers = np.ones(source_valid.shape[0], dtype=bool)
    best_count = 0
    best_median = math.inf

    iterations = max(1, cfg.ransac_iterations)
    for _ in range(iterations):
        sample = (
            np.arange(source_valid.shape[0])
            if source_valid.shape[0] == 3
            else rng.choice(source_valid.shape[0], size=3, replace=False)
        )
        rotation, translation = _weighted_kabsch(
            source_valid[sample], target_valid[sample], weights_valid[sample]
        )
        aligned = source_valid @ rotation.T + translation
        residuals = np.linalg.norm(aligned - target_valid, axis=1)
        inliers = residuals <= threshold
        count = int(inliers.sum())
        median = float(np.median(residuals[inliers])) if count else math.inf
        if count > best_count or (count == best_count and median < best_median):
            best_count = count
            best_median = median
            best_inliers = inliers

    if best_count < 3:
        best_inliers = np.ones(source_valid.shape[0], dtype=bool)
    rotation, translation = _weighted_kabsch(
        source_valid[best_inliers],
        target_valid[best_inliers],
        weights_valid[best_inliers],
    )
    aligned_valid = source_valid @ rotation.T + translation
    residuals = np.linalg.norm(aligned_valid - target_valid, axis=1)
    final_inliers = residuals <= threshold
    if int(final_inliers.sum()) >= 3:
        rotation, translation = _weighted_kabsch(
            source_valid[final_inliers],
            target_valid[final_inliers],
            weights_valid[final_inliers],
        )
        aligned_valid = source_valid @ rotation.T + translation
        residuals = np.linalg.norm(aligned_valid - target_valid, axis=1)
        final_inliers = residuals <= threshold

    aligned = np.full_like(source, np.nan, dtype=np.float64)
    aligned[finite] = aligned_valid
    full_inliers = np.zeros(source.shape[0], dtype=bool)
    full_inliers[finite] = final_inliers
    median_residual = float(np.median(residuals[final_inliers])) if final_inliers.any() else float(np.median(residuals))
    return RigidFit(
        valid=True,
        rotation=rotation,
        translation=translation,
        scale=1.0,
        scale_ratio=_scale_ratio(source_valid, target_valid),
        aligned_points=aligned,
        inlier_mask=full_inliers,
        inlier_ratio=float(full_inliers.sum() / max(int(finite.sum()), 1)),
        median_residual=median_residual,
        median_residual_ratio=median_residual / diameter,
    )


def canonicalize_instance(
    reference_points: np.ndarray,
    current_points: np.ndarray,
    *,
    kind: str,
    config: V2EvaluationConfig | None = None,
    weights: np.ndarray | Sequence[float] | None = None,
) -> CanonicalizationResult:
    cfg = config or V2EvaluationConfig()
    reference = _points(reference_points)
    current = _points(current_points)
    if reference.shape != current.shape or reference.shape[0] < 3:
        return CanonicalizationResult(
            valid=False,
            mode="unavailable",
            geometry_score=None,
            scale_score=None,
            topology_score=None,
            scale_ratio=1.0,
            inlier_ratio=0.0,
            residual_ratio=None,
            error="canonicalization requires at least three paired points",
        )
    local = _local_structure_stats(reference, current, cfg)
    topology = local.score
    scale_ratio = _scale_ratio(current, reference)
    scale_score = math.exp(-abs(math.log(max(scale_ratio, 1e-12))))

    if kind == "deformable":
        return CanonicalizationResult(
            valid=True,
            mode="local_topology",
            geometry_score=None,
            scale_score=local.scale_score,
            topology_score=topology,
            scale_ratio=local.scale_ratio,
            inlier_ratio=local.inlier_ratio,
            residual_ratio=local.median_log_error,
        )
    if kind == "articulated":
        return CanonicalizationResult(
            valid=True,
            mode="articulated_local",
            geometry_score=topology,
            scale_score=local.scale_score,
            topology_score=topology,
            scale_ratio=local.scale_ratio,
            inlier_ratio=local.inlier_ratio,
            residual_ratio=local.median_log_error,
        )
    if kind == "unknown":
        return CanonicalizationResult(
            valid=False,
            mode="unknown",
            geometry_score=None,
            scale_score=None,
            topology_score=topology,
            scale_ratio=scale_ratio,
            inlier_ratio=0.0,
            residual_ratio=None,
            error="instance kind is unknown",
        )
    if kind == "static":
        residuals = np.linalg.norm(current - reference, axis=1)
        ratio = float(np.median(residuals)) / max(_diameter(reference), 1e-9)
        return CanonicalizationResult(
            valid=True,
            mode="camera_only",
            geometry_score=clamp01(math.exp(-ratio / max(cfg.geometry_error_scale, 1e-9))),
            scale_score=clamp01(scale_score),
            topology_score=topology,
            scale_ratio=scale_ratio,
            inlier_ratio=1.0,
            residual_ratio=ratio,
        )

    fit = fit_rigid_transform(current, reference, weights, cfg)
    if not fit.valid:
        return CanonicalizationResult(
            valid=False,
            mode="rigid",
            geometry_score=None,
            scale_score=None,
            topology_score=topology,
            scale_ratio=scale_ratio,
            inlier_ratio=0.0,
            residual_ratio=None,
            error=fit.error,
        )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = fit.rotation
    transform[:3, 3] = fit.translation
    geometry_score = math.exp(
        -fit.median_residual_ratio / max(cfg.geometry_error_scale, 1e-9)
    )
    return CanonicalizationResult(
        valid=True,
        mode="rigid",
        geometry_score=clamp01(geometry_score),
        scale_score=clamp01(math.exp(-abs(math.log(max(fit.scale_ratio, 1e-12))))),
        topology_score=topology,
        scale_ratio=fit.scale_ratio,
        inlier_ratio=fit.inlier_ratio,
        residual_ratio=fit.median_residual_ratio,
        transform=transform,
    )


def score_articulated_keypoints(
    reference: Mapping[str, Sequence[float]],
    current: Mapping[str, Sequence[float]],
    *,
    bones: Sequence[tuple[str, str]],
) -> float | None:
    errors: list[float] = []
    for first, second in bones:
        if first not in reference or second not in reference or first not in current or second not in current:
            continue
        ref_length = math.dist(reference[first], reference[second])
        cur_length = math.dist(current[first], current[second])
        if ref_length <= 1e-9 or cur_length <= 1e-9:
            continue
        errors.append(abs(math.log(cur_length / ref_length)))
    if not errors:
        return None
    return clamp01(math.exp(-sum(errors) / len(errors)))


def _weighted_kabsch(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    normalized = weights / max(float(weights.sum()), 1e-12)
    source_center = np.sum(source * normalized[:, None], axis=0)
    target_center = np.sum(target * normalized[:, None], axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    covariance = (source_zero * normalized[:, None]).T @ target_zero
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_t[-1] *= -1
        rotation = right_t.T @ left.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def _points(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    return array


def _diameter(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    return float(np.max(distances))


def _scale_ratio(source: np.ndarray, target: np.ndarray) -> float:
    source_radius = math.sqrt(float(np.mean(np.sum((source - source.mean(axis=0)) ** 2, axis=1))))
    target_radius = math.sqrt(float(np.mean(np.sum((target - target.mean(axis=0)) ** 2, axis=1))))
    return source_radius / max(target_radius, 1e-12)


def _local_structure_stats(
    reference: np.ndarray,
    current: np.ndarray,
    config: V2EvaluationConfig,
) -> _LocalStructure:
    count = len(reference)
    if count < 2:
        return _LocalStructure(0.0, 1.0, 0.0, 0.0, math.inf)
    distances = np.linalg.norm(reference[:, None, :] - reference[None, :, :], axis=-1)
    np.fill_diagonal(distances, math.inf)
    min_edge_length = config.topology_min_edge_ratio * _diameter(reference)
    candidate_distances = np.where(distances >= min_edge_length, distances, math.inf)
    neighbor_count = min(2, count - 1)
    edges: set[tuple[int, int]] = set()
    for index, row in enumerate(candidate_distances):
        candidates = np.flatnonzero(np.isfinite(row))
        if len(candidates) == 0:
            continue
        nearest = candidates[np.argsort(row[candidates])[:neighbor_count]]
        edges.update(
            (min(index, int(neighbor)), max(index, int(neighbor)))
            for neighbor in nearest
        )
    if not edges:
        return _LocalStructure(0.0, 1.0, 0.0, 0.0, math.inf)
    sorted_edges = sorted(edges)
    first = np.asarray([edge[0] for edge in sorted_edges], dtype=np.int64)
    second = np.asarray([edge[1] for edge in sorted_edges], dtype=np.int64)
    reference_lengths = np.linalg.norm(reference[first] - reference[second], axis=1)
    current_lengths = np.linalg.norm(current[first] - current[second], axis=1)
    valid = (
        np.isfinite(reference_lengths)
        & np.isfinite(current_lengths)
        & (reference_lengths > 1e-9)
    )
    if not valid.any():
        return _LocalStructure(0.0, 1.0, 0.0, 0.0, math.inf)
    log_ratios = np.log(
        np.maximum(current_lengths[valid], 1e-12) / reference_lengths[valid]
    )
    median_log_scale = float(np.median(log_ratios))
    shape_errors = np.abs(log_ratios - median_log_scale)
    median_error = float(np.median(shape_errors))
    scale_ratio = math.exp(median_log_scale)
    scale_score = clamp01(math.exp(-abs(median_log_scale)))
    inlier_threshold = max(config.rigid_inlier_threshold_ratio, 1e-9)
    inlier_ratio = float(np.mean(shape_errors <= inlier_threshold))
    calibrated_error = max(0.0, median_error - config.topology_noise_floor)
    score = clamp01(
        math.exp(-calibrated_error / config.topology_error_scale)
    )
    return _LocalStructure(
        score=score,
        scale_ratio=scale_ratio,
        scale_score=scale_score,
        inlier_ratio=inlier_ratio,
        median_log_error=median_error,
    )


def _invalid_fit(source: np.ndarray, error: str) -> RigidFit:
    count = len(source) if source.ndim == 2 else 0
    return RigidFit(
        valid=False,
        rotation=np.eye(3),
        translation=np.zeros(3),
        scale=1.0,
        scale_ratio=1.0,
        aligned_points=np.asarray(source, dtype=np.float64),
        inlier_mask=np.zeros(count, dtype=bool),
        inlier_ratio=0.0,
        median_residual=math.inf,
        median_residual_ratio=math.inf,
        error=error,
    )
