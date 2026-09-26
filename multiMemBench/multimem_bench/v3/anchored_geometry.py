"""Camera and visibility evidence anchored to immutable single-image geometry."""

from __future__ import annotations

from typing import Any
import cv2
import numpy as np

from .config import V3EvaluationConfig
from .geometry import V3FrameGeometry


def anchor_frame_geometry(reference: Any, current: Any, matches: V3FrameGeometry,
                          config: V3EvaluationConfig) -> V3FrameGeometry:
    cfg = config
    result = V3FrameGeometry(matches.frame_index, "evaluator_failure", matcher=matches.matcher,
        reference_pixels=matches.reference_pixels.copy(), current_pixels=matches.current_pixels.copy(),
        error_space="source_pixels", diagnostics={
            "geometry_level": "input_grounded_monocular_pnp", "fixed_reference_geometry": True,
            "reference_fingerprint": reference.fingerprint, "intrinsics_source": "current_single_image_moge2",
            "camera_support": "non_target_background_only", "confidence_is_calibrated": False,
            "reference_is_ground_truth_3d": False,
        })

    def fail(reason: str) -> V3FrameGeometry:
        result.diagnostics["reason"] = reason
        return result

    points, valid = _sample(reference.prediction.points, matches.reference_pixels, reference.prediction.valid)
    _, current_valid = _sample(current.depth, matches.current_pixels, current.valid)
    background, bg_valid = _sample(reference.background_mask, matches.reference_pixels, reference.prediction.valid)
    bg = np.flatnonzero(valid & current_valid & bg_valid & background.astype(bool))
    result.reference_points_3d = points
    result.diagnostics["background_matches"] = int(len(bg))
    if len(bg) < max(cfg.min_background_matches, 2 * cfg.min_depth_alignment_samples):
        return fail("insufficient_background_matches")
    ref_size = reference.prediction.depth.shape[::-1]
    cur_size = current.depth.shape[::-1]
    support = min(_coverage(matches.reference_pixels[bg], ref_size), _coverage(matches.current_pixels[bg], cur_size))
    result.diagnostics["background_spatial_coverage"] = support
    if support < cfg.min_background_coverage:
        return fail("insufficient_background_spatial_support")
    if np.linalg.matrix_rank(points[bg] - points[bg].mean(axis=0), tol=1e-6) < 2:
        return fail("degenerate_background_geometry")
    # Canonical spatial ordering makes the held-out split independent of match order.
    xy = matches.reference_pixels[bg]
    bg = bg[np.lexsort((xy[:,0], xy[:,1]))]
    train, holdout = bg[::2], bg[1::2]
    try:
        # Single-image focal estimates can drift across frames. Select between
        # two independent priors using training background only, never holdout.
        reference_k = reference.prediction.intrinsics.astype(np.float64).copy()
        for axis in (0, 1):
            ratio = cur_size[axis] / ref_size[axis]
            reference_k[axis, axis] *= ratio
            reference_k[axis, 2] = (reference_k[axis, 2] + .5) * ratio - .5
        candidates = []
        for source, matrix in (("current_single_image_moge2", current.intrinsics),
                               ("reference_single_image_resized", reference_k)):
            matrix = matrix.astype(np.float64)
            cv2.setRNGSeed(0)
            ok, rv, tv, indices = cv2.solvePnPRansac(
                points[train].astype(np.float64), matches.current_pixels[train].astype(np.float64),
                matrix, None, iterationsCount=cfg.ransac_iterations,
                reprojectionError=cfg.max_reprojection_error_px, confidence=cfg.confidence,
                flags=cv2.SOLVEPNP_EPNP)
            if not ok or indices is None or len(indices) < 6:
                continue
            support_indices = train[indices.reshape(-1)]
            refined_rv, refined_tv = cv2.solvePnPRefineLM(
                points[support_indices].astype(np.float64),
                matches.current_pixels[support_indices].astype(np.float64), matrix, None,
                rv.copy(), tv.copy())
            for method, r, t in (("ransac", rv, tv), ("ransac_lm", refined_rv, refined_tv)):
                rotation_candidate, _ = cv2.Rodrigues(r)
                transformed = points[train] @ rotation_candidate.T + t.reshape(3)
                residual = np.linalg.norm(_project(transformed, matrix) - matches.current_pixels[train], axis=1)
                residual[transformed[:, 2] <= 0] = np.inf
                loss = float(np.minimum(residual, cfg.max_reprojection_error_px).mean())
                candidates.append((loss, source, method, matrix, r, t))
        if not candidates:
            return fail("background_pnp_failed")
        loss, source, method, camera_matrix, rotation_vec, translation_vec = min(candidates, key=lambda c: c[0])
        result.diagnostics.update(intrinsics_source=source, camera_fit_method=method,
            camera_selection="training_background_clipped_reprojection_loss",
            camera_train_loss_px=loss, camera_matrix_source_px=camera_matrix.tolist(),
            camera_candidates=[{"source": c[1], "method": c[2], "train_loss_px": c[0]} for c in candidates])
        rotation, _ = cv2.Rodrigues(rotation_vec)
        camera_points = np.where(valid[:,None],points,0.) @ rotation.T + translation_vec.reshape(3)
        projected = _project(camera_points, camera_matrix)
        errors = np.linalg.norm(projected - matches.current_pixels, axis=1)
        errors[~valid] = np.inf
        inliers = valid & current_valid & np.isfinite(errors) & (camera_points[:,2] > 0) & (errors <= cfg.max_reprojection_error_px)
        train_ratio, heldout_ratio = float(inliers[train].mean()), float(inliers[holdout].mean())
        reliability = min(train_ratio, heldout_ratio)
        result.diagnostics.update(background_train_inlier_ratio=train_ratio,
            background_heldout_inlier_ratio=heldout_ratio, camera_confidence=reliability)
        if reliability < max(cfg.min_inlier_ratio, cfg.min_visibility_camera_confidence):
            return fail("unreliable_background_camera_pose")
        inlier_support = min(_coverage(pixels[index[inliers[index]]],size)
            for index in (train,holdout)
            for pixels,size in ((matches.reference_pixels,ref_size),(matches.current_pixels,cur_size)))
        result.diagnostics["background_inlier_spatial_coverage"] = inlier_support
        if inlier_support < cfg.min_background_coverage:
            return fail("insufficient_background_inlier_support")
        current_z, depth_valid = _sample(current.depth, matches.current_pixels, current.valid)
        train_depth = train[inliers[train] & depth_valid[train]]
        heldout_depth = holdout[inliers[holdout] & depth_valid[holdout]]
        if min(len(train_depth), len(heldout_depth)) < cfg.min_depth_alignment_samples:
            return fail("insufficient_depth_alignment_samples")
        scale = float(np.median(camera_points[train_depth,2] / current_z[train_depth]))
        if not np.isfinite(scale) or not 1/cfg.max_depth_scale_ratio <= scale <= cfg.max_depth_scale_ratio:
            return fail("unreliable_depth_scale")
        log_errors = np.abs(np.log(current_z[heldout_depth]*scale / camera_points[heldout_depth,2]))
        alignment_error = float(np.quantile(log_errors,.9))
        result.diagnostics.update(current_depth_scale=scale, depth_alignment_heldout_log_error_p90=alignment_error,
            depth_alignment_train_samples=len(train_depth), depth_alignment_heldout_samples=len(heldout_depth))
        if alignment_error > cfg.max_depth_alignment_error:
            return fail("unreliable_depth_alignment")
        pose = np.eye(4, dtype=np.float32)
        pose[:3,:3], pose[:3,3] = rotation, translation_vec.reshape(3)
        result.relative_pose, result.reprojection_errors, result.inlier_mask = pose, errors, inliers
        result.diagnostics.update(inlier_ratio=float(inliers.mean()), matches=len(points),
            reason="fixed_reference_background_pose", current_size=list(cur_size))
        # Compare fitted projections to a resized initial view. Targets never
        # contribute to the motion evidence or the fitted camera.
        initial_pixels = (matches.reference_pixels[heldout_depth] + .5) * np.array(cur_size) / np.array(ref_size) - .5
        displacement = np.linalg.norm(projected[heldout_depth] - initial_pixels, axis=1)
        result.diagnostics["camera_motion"] = {
            "displacement_fraction": float(np.median(displacement) / np.hypot(*cur_size)),
            "rotation_degrees": float(np.degrees(np.linalg.norm(rotation_vec))),
            "translation_over_reference_depth": float(np.linalg.norm(translation_vec) / np.median(points[train_depth, 2])),
            "support": "heldout_background_inliers", "n_points": len(heldout_depth),
        }
        for object_id, query in reference.instance_queries.items():
            query_points, query_valid = _sample(reference.prediction.points, query, reference.prediction.valid)
            query_points = query_points[query_valid]
            if len(query_points) < 4:
                continue
            transformed = query_points @ rotation.T + translation_vec.reshape(3)
            pixels = _project(transformed, camera_matrix)
            sampled_depth, sampled_valid = _sample(current.depth, pixels, current.valid)
            result.instance_visibility[object_id] = {
                "reference_source": "input_only", "reference_fingerprint": result.diagnostics["reference_fingerprint"],
                "camera_confidence": reliability, "projected_pixels": pixels.tolist(),
                "projected_depth": transformed[:,2].tolist(),
                "current_depth": np.where(sampled_valid, sampled_depth*scale, 0.).tolist(),
                "depth_confidence": sampled_valid.astype(float).tolist(),
                "depth_confidence_kind": "binary_validity_not_calibrated_probability",
                "depth_alignment_error": alignment_error,
            }
        result.status = "scored"
        return result
    except cv2.error as exc:
        result.diagnostics["error"] = str(exc)
        return fail("background_geometry_backend_failure")


def _sample(array: np.ndarray, pixels: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.asarray(pixels, dtype=float).reshape(-1,2)
    height, width = array.shape[:2]
    finite = np.isfinite(pixels).all(axis=1)
    x = np.rint(np.where(finite,pixels[:,0],0)).astype(int)
    y = np.rint(np.where(finite,pixels[:,1],0)).astype(int)
    inside = finite & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y = np.clip(x,0,width-1), np.clip(y,0,height-1)
    values = array[y,x]
    value_valid = np.isfinite(values).all(axis=-1) if values.ndim > 1 else np.isfinite(values)
    valid = inside & valid_mask[y,x] & value_valid
    return values, valid


def _project(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    homogeneous = points @ intrinsics.T
    depth = homogeneous[:,2]
    return homogeneous[:,:2] / np.where(np.abs(depth)>1e-10,depth,1e-10)[:,None]


def _coverage(pixels: np.ndarray, size: tuple[int,int]) -> float:
    if len(pixels) < 3:
        return 0.
    return float(cv2.contourArea(cv2.convexHull(pixels.astype(np.float32)))) / (size[0]*size[1])
