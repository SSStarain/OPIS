"""Independent reference-to-frame geometry for V3.

The current frame is processed together with the immutable reference image only.
No state from another generated frame is accepted by this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import math
import tempfile
import threading

import numpy as np
from PIL import Image

from .config import V3EvaluationConfig


_MAST3R_MODEL: Any | None = None
_MAST3R_MODEL_KEY: tuple[str, str] | None = None
_MAST3R_LAST_ERROR: str | None = None
_MAST3R_MODEL_LOCK = threading.Lock()


@dataclass
class V3FrameGeometry:
    frame_index: int
    status: str
    reference_pixels: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float32))
    current_pixels: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float32))
    inlier_mask: np.ndarray = field(default_factory=lambda: np.empty((0,), bool))
    homography: np.ndarray | None = None
    essential: np.ndarray | None = None
    relative_pose: np.ndarray | None = None
    reference_points_3d: np.ndarray | None = None
    reprojection_errors: np.ndarray | None = None
    matcher: str = "none"
    diagnostics: dict[str, Any] = field(default_factory=dict)
    instance_visibility: dict[str, dict[str, Any]] = field(default_factory=dict)
    artifact_version: int = 2
    error_space: str = "unknown"


@dataclass
class V3GeometryArtifact:
    status: str
    backend: str
    reference_size: tuple[int, int]
    frames: dict[int, V3FrameGeometry]
    metadata: dict[str, Any] = field(default_factory=dict)
    artifact_version: int = 2

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 3,
            "artifact_version": self.artifact_version,
            "status": self.status,
            "backend": self.backend,
            "reference_size": list(self.reference_size),
            "metadata": self.metadata,
            "frames": {
                str(index): _frame_to_json(frame) for index, frame in self.frames.items()
            },
        }
        target.write_text(json.dumps(_json_values(payload), indent=2, allow_nan=False), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "V3GeometryArtifact":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        frames = {int(k): _frame_from_json(v) for k, v in data.get("frames", {}).items()}
        artifact = cls(
            status=str(data.get("status", "unavailable")),
            backend=str(data.get("backend", "unknown")),
            reference_size=tuple(int(x) for x in data.get("reference_size", [0, 0])),
            frames=frames,
            metadata=dict(data.get("metadata", {})),
            artifact_version=int(data.get("artifact_version", 1)),
        )
        return artifact


def prepare_v3_geometry(
    *,
    reference_image: str | Path | Image.Image,
    frame_images: dict[int, str | Path | Image.Image],
    output_path: str | Path | None = None,
    config: V3EvaluationConfig | None = None,
    reference_geometry: Any | None = None,
    predictor: Any | None = None,
) -> V3GeometryArtifact:
    cfg = config or V3EvaluationConfig()
    cfg.validate()
    reference = _load_image(reference_image)
    if cfg.geometry_mode == "input_grounded":
        from .reference import MoGePredictor, reference_image_sha256
        if reference_geometry is None:
            raise ValueError("input_grounded mode requires reference_geometry")
        if reference_geometry.metadata.get("image_sha256") != reference_image_sha256(reference):
            raise ValueError("reference_geometry image hash does not match reference image")
        predictor = predictor or MoGePredictor(cfg)
    frames: dict[int, V3FrameGeometry] = {}
    matchers: set[str] = set()
    for index, value in sorted(frame_images.items()):
        current = _load_image(value)
        result = _match_pair(reference, current, int(index), cfg)
        if cfg.geometry_mode == "input_grounded":
            from .anchored_geometry import anchor_frame_geometry
            try:
                current_prediction = predictor.predict(current)
                result = anchor_frame_geometry(reference_geometry, current_prediction, result, cfg)
            except (ImportError, RuntimeError, OSError, ValueError) as exc:
                result = V3FrameGeometry(int(index), "evaluator_failure", matcher=result.matcher,
                    diagnostics={"reason": "current_monocular_prediction_failed", "error": str(exc),
                                 "fixed_reference_geometry": True})
        frames[int(index)] = result
        matchers.add(result.matcher)
    if matchers == {"mast3r"}:
        backend = "mast3r"
    elif matchers == {"opencv"}:
        backend = "opencv"
    elif matchers:
        backend = "mixed"
    else:
        backend = "unknown"
    artifact = V3GeometryArtifact(
        status="available" if frames else "unavailable",
        backend=backend,
        reference_size=reference.size,
        frames=frames,
        metadata={
            "independent_pairwise": True,
            "reference_frame_index": -1,
            "config": cfg.to_dict(),
            "geometry_mode": cfg.geometry_mode,
            "fixed_reference_geometry": cfg.geometry_mode == "input_grounded",
            "reference_fingerprint": reference_geometry.fingerprint if reference_geometry is not None else None,
            "reference_identity": reference_geometry.metadata.get("reference_identity") if reference_geometry is not None else None,
            "matcher_policy": "mast3r_primary" if not cfg.allow_opencv_fallback else "mast3r_primary_with_opencv_fallback",
            "motion_diagnostics": summarize_camera_motion(frames, cfg),
        },
    )
    if output_path:
        artifact.save(output_path)
    return artifact


def summarize_camera_motion(frames: dict[int, V3FrameGeometry], cfg: V3EvaluationConfig) -> dict[str, Any]:
    """Provisional anti-static gate, not textual camera-path compliance.

    Independent reference-to-frame evidence is aggregated without fitting any
    cross-frame geometry. Unreliable cameras remain missing evidence.
    """
    values = {}
    for index, frame in sorted(frames.items()):
        value = frame.diagnostics.get("camera_motion", {}).get("displacement_fraction")
        if frame.status == "scored" and isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
            values[str(index)] = float(value)
    coverage = len(values) / len(frames) if frames else 0.
    moving = sum(value >= cfg.min_motion_displacement_fraction for value in values.values())
    enough = len(values) >= cfg.min_geometry_frames and (
        cfg.min_geometry_coverage is None or coverage >= cfg.min_geometry_coverage)
    # Missing frames do not provide evidence of camera movement.
    moving_fraction = moving / len(frames) if frames else 0.
    valid = bool(moving_fraction >= cfg.min_motion_frame_fraction) if enough else None
    return {"motion_valid": valid, "scope": "reference_relative_camera_change_not_prompt_compliance",
            "provisional": True, "reason": "insufficient_camera_evidence" if not enough else
            "camera_change_detected" if valid else "insufficient_camera_change",
            "n_frames": len(frames), "n_valid_frames": len(values), "coverage": coverage,
            "moving_frame_fraction": moving_fraction, "frame_displacement_fraction": values,
            "min_displacement_fraction": cfg.min_motion_displacement_fraction,
            "min_moving_frame_fraction": cfg.min_motion_frame_fraction}


def _match_pair(reference: Image.Image, current: Image.Image, frame_index: int, cfg: V3EvaluationConfig) -> V3FrameGeometry:
    if cfg.matcher in {"mast3r", "auto"}:
        if not _cuda_available():
            if cfg.matcher == "mast3r":
                return V3FrameGeometry(frame_index, "evaluator_failure", matcher="mast3r", diagnostics={"error": "MASt3R requires a CUDA-capable device for the configured production path"})
        else:
            mast3r_result = _match_pair_mast3r(reference, current, frame_index, cfg)
            if mast3r_result is not None:
                return mast3r_result
        if cfg.matcher == "mast3r" or not cfg.allow_opencv_fallback:
            return V3FrameGeometry(frame_index, "evaluator_failure", matcher="mast3r", diagnostics={"error": _MAST3R_LAST_ERROR or "MASt3R is unavailable or failed to load"})
    # OpenCV is an explicitly selected diagnostic backend or an opt-in fallback.
    try:
        import cv2
    except ImportError:
        return V3FrameGeometry(frame_index, "evaluator_failure", matcher="opencv", diagnostics={"error": "opencv unavailable"})
    ref = cv2.cvtColor(np.asarray(reference.convert("RGB")), cv2.COLOR_RGB2GRAY)
    cur = cv2.cvtColor(np.asarray(current.convert("RGB")), cv2.COLOR_RGB2GRAY)
    detector = cv2.SIFT_create(nfeatures=4000) if hasattr(cv2, "SIFT_create") else cv2.ORB_create(nfeatures=4000)
    kp_ref, des_ref = detector.detectAndCompute(ref, None)
    kp_cur, des_cur = detector.detectAndCompute(cur, None)
    if des_ref is None or des_cur is None:
        return V3FrameGeometry(frame_index, "evaluator_failure", matcher="opencv", diagnostics={"matches": 0, "geometry_level": "projective_2d", "failure_reason": "insufficient_correspondences"})
    norm = cv2.NORM_L2 if des_ref.dtype == np.float32 else cv2.NORM_HAMMING
    pairs = cv2.BFMatcher(norm).knnMatch(des_ref, des_cur, k=2)
    good = [m for m, n in pairs if m.distance < cfg.ratio_test * n.distance]
    ref_pts = np.float32([kp_ref[m.queryIdx].pt for m in good])
    cur_pts = np.float32([kp_cur[m.trainIdx].pt for m in good])
    if len(good) < cfg.min_matches:
        return V3FrameGeometry(frame_index, "evaluator_failure", ref_pts, cur_pts, matcher="opencv", diagnostics={"matches": len(good), "geometry_level": "projective_2d", "failure_reason": "insufficient_correspondences"})
    homography, mask = cv2.findHomography(ref_pts, cur_pts, cv2.RANSAC, cfg.max_reprojection_error_px, maxIters=cfg.ransac_iterations, confidence=cfg.confidence)
    inliers = mask.reshape(-1).astype(bool) if mask is not None else np.zeros(len(good), bool)
    ratio = float(inliers.mean()) if len(inliers) else 0.0
    status = "scored" if ratio >= cfg.min_inlier_ratio else "evaluator_failure"
    return V3FrameGeometry(
        frame_index=frame_index,
        status=status,
        reference_pixels=ref_pts,
        current_pixels=cur_pts,
        inlier_mask=inliers,
        homography=homography,
        matcher="opencv",
        error_space="source_pixels",
        diagnostics={"matches": len(good), "inliers": int(inliers.sum()), "inlier_ratio": ratio, "geometry_level": "projective_2d"},
    )


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _match_pair_mast3r(reference: Image.Image, current: Image.Image, frame_index: int, cfg: V3EvaluationConfig) -> V3FrameGeometry | None:
    """Run official MASt3R inference for exactly one reference/current pair.

    The imports are delayed because MASt3R is an optional, heavyweight dependency.
    The adapter follows the official ``dust3r.inference`` and
    ``mast3r.fast_reciprocal_NNs`` API and returns ``None`` when that optional
    stack is absent. Classical fallback requires an explicit configuration opt-in.
    The reference-side pointmap is conditioned on both images, not immutable GT.
    """
    global _MAST3R_LAST_ERROR
    _MAST3R_LAST_ERROR = None
    try:
        import torch
        from dust3r.inference import inference
        from dust3r.utils.image import load_images
        from mast3r.model import AsymmetricMASt3R
        from mast3r.fast_nn import fast_reciprocal_NNs
        import mast3r.utils.path_to_dust3r  # noqa: F401
    except ImportError as exc:
        _MAST3R_LAST_ERROR = f"MASt3R import failed: {exc}"
        return None
    device = cfg.matcher_device if torch.cuda.is_available() and cfg.matcher_device.startswith("cuda") else "cpu"
    try:
        with tempfile.TemporaryDirectory(prefix="multimem_mast3r_") as tmp:
            ref_path = Path(tmp) / "reference.png"
            cur_path = Path(tmp) / "current.png"
            reference.save(ref_path)
            current.save(cur_path)
            images = load_images([str(ref_path), str(cur_path)], size=cfg.matcher_image_size)
            global _MAST3R_MODEL, _MAST3R_MODEL_KEY
            model_key = (cfg.matcher_model_id, device)
            # Geometry frames are processed sequentially today, but the lock
            # keeps lazy initialization single-shot if callers become
            # concurrent. The model remains resident for the whole subprocess.
            with _MAST3R_MODEL_LOCK:
                if _MAST3R_MODEL is None or _MAST3R_MODEL_KEY != model_key:
                    _MAST3R_MODEL = AsymmetricMASt3R.from_pretrained(cfg.matcher_model_id).to(device).eval()
                    _MAST3R_MODEL_KEY = model_key
                model = _MAST3R_MODEL
            output = inference([tuple(images)], model, device=device, batch_size=1, verbose=False)
            pred1 = output["pred1"]
            pred2 = output["pred2"]
            desc1 = pred1.get("desc")
            desc2 = pred2.get("desc")
            if desc1 is None or desc2 is None:
                return None
            desc1 = desc1.squeeze(0).detach()
            desc2 = desc2.squeeze(0).detach()
            xy1, xy2 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8, device=device, dist="dot", block_size=2**13)
            xy1_model = np.asarray(xy1, dtype=np.float32)
            xy2_model = np.asarray(xy2, dtype=np.float32)
            ref_model_size = (int(desc1.shape[1]), int(desc1.shape[0]))
            cur_model_size = (int(desc2.shape[1]), int(desc2.shape[0]))
            ref_pts = _mast3r_to_source(xy1_model, reference.size, cfg.matcher_image_size, model_size=ref_model_size)
            cur_pts = _mast3r_to_source(xy2_model, current.size, cfg.matcher_image_size, model_size=cur_model_size)
            if len(ref_pts) < cfg.min_matches:
                return V3FrameGeometry(frame_index, "evaluator_failure", ref_pts, cur_pts, matcher="mast3r", diagnostics={"matches": int(len(ref_pts)), "geometry_level": "pairwise_pointmap_pnp", "failure_reason": "insufficient_correspondences"})
            if cfg.geometry_mode == "input_grounded":
                return V3FrameGeometry(frame_index, "correspondences", ref_pts, cur_pts, matcher="mast3r",
                    diagnostics={"matches": len(ref_pts), "geometry_level": "correspondences_only",
                                 "pairwise_pointmap_used": False})
            import cv2
            homography, mask = cv2.findHomography(ref_pts, cur_pts, cv2.RANSAC, cfg.max_reprojection_error_px, maxIters=cfg.ransac_iterations, confidence=cfg.confidence)
            pts3d_map = pred1["pts3d"].squeeze(0).detach().float().cpu().numpy()
            query_x = np.clip(np.rint(xy1_model[:, 0]).astype(int), 0, pts3d_map.shape[1] - 1)
            query_y = np.clip(np.rint(xy1_model[:, 1]).astype(int), 0, pts3d_map.shape[0] - 1)
            reference_points_3d = pts3d_map[query_y, query_x]
            finite = np.isfinite(reference_points_3d).all(axis=1)
            # Approximate current-view intrinsics; transform them to source pixels
            # so RANSAC thresholds and returned residuals share a single unit.
            focal = float(max(cur_model_size))
            origin, axis_x, axis_y, principal = _mast3r_to_source(
                np.asarray([[0, 0], [focal, 0], [0, focal], [(cur_model_size[0]-1) / 2, (cur_model_size[1]-1) / 2]]),
                current.size, cfg.matcher_image_size, model_size=cur_model_size,
            )
            camera_matrix = np.asarray([[axis_x[0] - origin[0], 0.0, principal[0]], [0.0, axis_y[1] - origin[1], principal[1]], [0.0, 0.0, 1.0]], dtype=np.float64)
            pose_inliers = np.zeros(len(ref_pts), dtype=bool)
            relative_pose = None
            reprojection_errors = np.full(len(ref_pts), np.inf, dtype=np.float32)
            if int(finite.sum()) >= max(6, cfg.min_matches):
                ok, rotation_vec, translation_vec, pnp_indices = cv2.solvePnPRansac(
                    reference_points_3d[finite].astype(np.float64),
                    cur_pts[finite].astype(np.float64),
                    camera_matrix,
                    None,
                    iterationsCount=cfg.ransac_iterations,
                    reprojectionError=cfg.max_reprojection_error_px,
                    confidence=cfg.confidence,
                    flags=cv2.SOLVEPNP_EPNP,
                )
                if ok and pnp_indices is not None:
                    finite_indices = np.flatnonzero(finite)
                    pose_inliers[finite_indices[pnp_indices.reshape(-1)]] = True
                    rotation, _ = cv2.Rodrigues(rotation_vec)
                    relative_pose = np.eye(4, dtype=np.float32)
                    relative_pose[:3, :3] = rotation
                    relative_pose[:3, 3] = translation_vec.reshape(3)
                    projected, _ = cv2.projectPoints(reference_points_3d[finite].astype(np.float64), rotation_vec, translation_vec, camera_matrix, None)
                    reprojection_errors[finite] = np.linalg.norm(projected.reshape(-1, 2) - cur_pts[finite], axis=1)
            inliers = pose_inliers
            ratio = float(inliers.mean()) if len(inliers) else 0.0
            return V3FrameGeometry(
                frame_index, "scored" if ratio >= cfg.min_inlier_ratio else "evaluator_failure",
                ref_pts, cur_pts, inliers, homography=homography, relative_pose=relative_pose,
                reference_points_3d=reference_points_3d, reprojection_errors=reprojection_errors,
                matcher="mast3r", error_space="source_pixels",
                diagnostics={
                    "matches": len(ref_pts), "inliers": int(inliers.sum()), "inlier_ratio": ratio,
                    "geometry_level": "pairwise_pointmap_pnp", "pointmap_conditioning": "reference_and_current",
                    "fixed_reference_geometry": False, "intrinsics_source": "approximate",
                    "reprojection_error_space": "source_pixels", "model_id": cfg.matcher_model_id,
                    "pose_method": "pairwise_pointmap_pnp_ransac", "pnp_focal_model_px": focal,
                    "current_camera_matrix_source_px": camera_matrix.tolist(),
                    "reference_model_size": list(ref_model_size), "current_model_size": list(cur_model_size),
                    "current_size": list(current.size),
                },
            )
    except (OSError, RuntimeError, ValueError, KeyError, IndexError) as exc:
        _MAST3R_LAST_ERROR = f"{type(exc).__name__}: {exc}"
        return None


def _mast3r_to_source(points: np.ndarray, source_size: tuple[int, int], long_edge: int, patch_size: int = 16, *, model_size: tuple[int, int] | None = None) -> np.ndarray:
    """Invert DUSt3R/MASt3R's long-edge resize and centered patch crop."""
    width, height = source_size
    resize_edge = round(long_edge * max(width / height, height / width)) if long_edge == 224 else long_edge
    scale = float(resize_edge) / max(width, height)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    center_x, center_y = resized_width // 2, resized_height // 2
    crop_width = (2 * center_x // patch_size) * patch_size
    crop_height = (2 * center_y // patch_size) * patch_size
    if long_edge == 224:
        crop_width = crop_height = 2 * min(center_x, center_y)
    elif resized_width == resized_height:
        crop_height = int(3 * crop_width / 4)
    if model_size is not None:
        crop_width, crop_height = model_size
    offset_x = center_x - crop_width / 2.0
    offset_y = center_y - crop_height / 2.0
    mapped = np.asarray(points, dtype=np.float32).copy()
    mapped[:, 0] = (mapped[:, 0] + offset_x + .5) * (width / resized_width) - .5
    mapped[:, 1] = (mapped[:, 1] + offset_y + .5) * (height / resized_height) - .5
    return mapped


def _load_image(value: str | Path | Image.Image) -> Image.Image:
    return value.convert("RGB") if isinstance(value, Image.Image) else Image.open(value).convert("RGB")


def _frame_to_json(frame: V3FrameGeometry) -> dict[str, Any]:
    return {"frame_index": frame.frame_index, "status": frame.status, "reference_pixels": frame.reference_pixels.tolist(), "current_pixels": frame.current_pixels.tolist(), "inlier_mask": frame.inlier_mask.astype(int).tolist(), "homography": frame.homography.tolist() if frame.homography is not None else None, "essential": frame.essential.tolist() if frame.essential is not None else None, "relative_pose": frame.relative_pose.tolist() if frame.relative_pose is not None else None, "reference_points_3d": frame.reference_points_3d.tolist() if frame.reference_points_3d is not None else None, "reprojection_errors": frame.reprojection_errors.tolist() if frame.reprojection_errors is not None else None, "matcher": frame.matcher, "diagnostics": frame.diagnostics, "instance_visibility": frame.instance_visibility, "artifact_version": frame.artifact_version, "error_space": frame.error_space}


def _frame_from_json(data: dict[str, Any]) -> V3FrameGeometry:
    return V3FrameGeometry(int(data["frame_index"]), str(data["status"]), np.asarray(data.get("reference_pixels", []), np.float32).reshape(-1, 2), np.asarray(data.get("current_pixels", []), np.float32).reshape(-1, 2), np.asarray(data.get("inlier_mask", []), bool), np.asarray(data["homography"], np.float32) if data.get("homography") is not None else None, np.asarray(data["essential"], np.float32) if data.get("essential") is not None else None, np.asarray(data["relative_pose"], np.float32) if data.get("relative_pose") is not None else None, np.asarray(data["reference_points_3d"], np.float32) if data.get("reference_points_3d") is not None else None, np.asarray(data["reprojection_errors"], np.float32) if data.get("reprojection_errors") is not None else None, str(data.get("matcher", "none")), dict(data.get("diagnostics", {})), instance_visibility=dict(data.get("instance_visibility", {})), artifact_version=int(data.get("artifact_version", 1)), error_space=str(data.get("error_space", "unknown")))


def _json_values(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_values(value.tolist())
    if isinstance(value, np.generic):
        return _json_values(value.item())
    if isinstance(value, dict):
        return {key: _json_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_values(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
