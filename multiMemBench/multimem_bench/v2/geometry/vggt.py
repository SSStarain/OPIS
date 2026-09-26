"""Optional adapters for official VGGT and VGGT-Omega packages."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from multimem_bench.v2.config import V2EvaluationConfig
from multimem_bench.v2.geometry.base import GeometryBackendUnavailable, GeometryPrediction


class VGGTBackend:
    name = "vggt"

    def __init__(self, config: V2EvaluationConfig) -> None:
        self.config = config
        self._model: Any | None = None

    def predict(
        self,
        images: Sequence[Image.Image],
        query_points: np.ndarray,
    ) -> GeometryPrediction:
        torch, model, pose_converter = self._load()
        tensor = _images_to_tensor(images, torch).to(self.config.geometry_device)
        queries = torch.from_numpy(np.asarray(query_points, dtype=np.float32)).to(
            self.config.geometry_device
        )
        dtype = _torch_dtype(torch, self.config.geometry_dtype)
        device_type = "cuda" if str(self.config.geometry_device).startswith("cuda") else "cpu"
        with torch.inference_mode():
            with torch.autocast(
                device_type=device_type,
                dtype=dtype,
                enabled=device_type == "cuda",
            ):
                output = model(tensor, query_points=queries)
        extrinsics, intrinsics = pose_converter(output["pose_enc"], tensor.shape[-2:])
        return GeometryPrediction(
            extrinsics=_numpy(extrinsics),
            intrinsics=_numpy(intrinsics),
            world_points=_numpy(output["world_points"]),
            point_confidence=_numpy(output["world_points_conf"]),
            depth=_squeeze_last(_numpy(output["depth"])),
            depth_confidence=_numpy(output["depth_conf"]),
            tracks=_numpy(output["track"]),
            track_visibility=_numpy(output["vis"]),
            track_confidence=_numpy(output["conf"]),
            camera_confidence=np.ones(len(images), dtype=np.float32),
            metadata={
                "track_source": "vggt_track_head",
                "dynamic_instance_tracking": True,
            },
        )

    def _load(self):
        try:
            import torch
            from vggt.models.vggt import VGGT
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        except ImportError as exc:
            raise GeometryBackendUnavailable(
                "VGGT is not installed; install the official vggt package"
            ) from exc
        if self._model is None:
            try:
                if self.config.geometry_checkpoint:
                    model = VGGT()
                    state = torch.load(
                        self.config.geometry_checkpoint,
                        map_location="cpu",
                        weights_only=True,
                    )
                    model.load_state_dict(_state_dict(state))
                else:
                    model = VGGT.from_pretrained(self.config.geometry_model_id)
                self._model = model.to(self.config.geometry_device).eval()
            except (OSError, RuntimeError, ValueError) as exc:
                raise GeometryBackendUnavailable(f"failed to load VGGT: {exc}") from exc
        return torch, self._model, pose_encoding_to_extri_intri


class VGGTOmegaBackend:
    name = "vggt_omega"

    def __init__(self, config: V2EvaluationConfig) -> None:
        self.config = config
        self._model: Any | None = None

    def predict(
        self,
        images: Sequence[Image.Image],
        query_points: np.ndarray,
    ) -> GeometryPrediction:
        torch, model, camera_converter = self._load()
        tensor = _images_to_tensor(images, torch).to(self.config.geometry_device)
        with torch.inference_mode():
            output = model(tensor)
        extrinsics_t, intrinsics_t = camera_converter(output["pose_enc"], tensor.shape[-2:])
        extrinsics = _numpy(extrinsics_t)
        intrinsics = _numpy(intrinsics_t)
        depth = _squeeze_last(_numpy(output["depth"]))
        depth_confidence = _numpy(output["depth_conf"])
        world_points = _unproject_depth(depth, extrinsics, intrinsics)
        tracks, visibility, confidence = _static_projection_tracks(
            world_points,
            depth_confidence,
            extrinsics,
            intrinsics,
            query_points,
        )
        return GeometryPrediction(
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            world_points=world_points,
            point_confidence=depth_confidence,
            depth=depth,
            depth_confidence=depth_confidence,
            tracks=tracks,
            track_visibility=visibility,
            track_confidence=confidence,
            camera_confidence=np.ones(len(images), dtype=np.float32),
            metadata={
                "track_source": "static_camera_projection",
                "dynamic_instance_tracking": False,
            },
        )

    def _load(self):
        if not self.config.geometry_checkpoint:
            raise GeometryBackendUnavailable(
                "VGGT-Omega requires geometry_checkpoint with approved weights"
            )
        try:
            import torch
            from vggt_omega.models import VGGTOmega
            from vggt_omega.utils.pose_enc import encoding_to_camera
        except ImportError as exc:
            raise GeometryBackendUnavailable(
                "VGGT-Omega is not installed; install the official vggt-omega package"
            ) from exc
        if self._model is None:
            try:
                model = VGGTOmega().to(self.config.geometry_device).eval()
                state = torch.load(
                    Path(self.config.geometry_checkpoint),
                    map_location="cpu",
                    weights_only=True,
                )
                model.load_state_dict(_state_dict(state))
                self._model = model
            except (OSError, RuntimeError, ValueError) as exc:
                raise GeometryBackendUnavailable(f"failed to load VGGT-Omega: {exc}") from exc
        return torch, self._model, encoding_to_camera


def _images_to_tensor(images: Sequence[Image.Image], torch):
    arrays = [
        np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
        for image in images
    ]
    return torch.from_numpy(np.stack(arrays))


def _torch_dtype(torch, name: str):
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    return torch.bfloat16


def _numpy(value) -> np.ndarray:
    array = value.detach().float().cpu().numpy()
    return array[0] if array.ndim > 0 and array.shape[0] == 1 else array


def _squeeze_last(array: np.ndarray) -> np.ndarray:
    return array[..., 0] if array.ndim >= 4 and array.shape[-1] == 1 else array


def _state_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        for key in ("state_dict", "model", "module"):
            nested = value.get(key)
            if isinstance(nested, dict):
                return nested
        return value
    raise ValueError("checkpoint does not contain a state dict")


def _unproject_depth(
    depth: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    frame_count, height, width = depth.shape
    y, x = np.indices((height, width), dtype=np.float64)
    pixels = np.stack([x, y, np.ones_like(x)], axis=-1).reshape(-1, 3)
    output = np.empty((frame_count, height, width, 3), dtype=np.float32)
    for index in range(frame_count):
        rays = pixels @ np.linalg.inv(intrinsics[index]).T
        camera = rays * depth[index].reshape(-1, 1)
        rotation = extrinsics[index, :3, :3]
        translation = extrinsics[index, :3, 3]
        world = (camera - translation) @ rotation
        output[index] = world.reshape(height, width, 3)
    return output


def _static_projection_tracks(
    world_points: np.ndarray,
    depth_confidence: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    query_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_count, height, width, _ = world_points.shape
    xy = np.rint(query_points).astype(np.int64)
    xy[:, 0] = np.clip(xy[:, 0], 0, width - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, height - 1)
    reference = world_points[0, xy[:, 1], xy[:, 0]]
    tracks = np.zeros((frame_count, len(query_points), 2), dtype=np.float32)
    visibility = np.zeros((frame_count, len(query_points)), dtype=np.float32)
    confidence = np.zeros_like(visibility)
    for index in range(frame_count):
        camera = reference @ extrinsics[index, :3, :3].T + extrinsics[index, :3, 3]
        projected = camera @ intrinsics[index].T
        positive = projected[:, 2] > 1e-9
        pixels = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-9)
        inside = (
            positive
            & (pixels[:, 0] >= 0)
            & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < height)
        )
        tracks[index] = pixels
        visibility[index, inside] = 1.0
        sample = np.rint(pixels).astype(np.int64)
        sample[:, 0] = np.clip(sample[:, 0], 0, width - 1)
        sample[:, 1] = np.clip(sample[:, 1], 0, height - 1)
        confidence[index] = depth_confidence[index, sample[:, 1], sample[:, 0]]
    return tracks, visibility, confidence
