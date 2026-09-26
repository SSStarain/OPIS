"""Immutable single-image geometry for the input-grounded V3 protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any
import json
import subprocess
import os
import tempfile
from zipfile import BadZipFile

import numpy as np
from PIL import Image

from multimem_bench.schema import ReferenceScene
from multimem_bench.vision.mask_utils import mask_from_bbox, mask_to_bool_array

from .config import V3EvaluationConfig


_MOGE_MODELS: dict[tuple[str, str], tuple[Any, dict[str, str]]] = {}
_SCHEMA_VERSION = 1


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical(list(array.shape)))
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class SingleImageGeometry:
    points: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray
    valid: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        points = np.asarray(self.points)
        depth = np.asarray(self.depth)
        intrinsics = np.asarray(self.intrinsics)
        valid = np.asarray(self.valid)
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("points must have shape HxWx3")
        if depth.shape != points.shape[:2] or valid.shape != points.shape[:2]:
            raise ValueError("depth and valid must match the point map dimensions")
        if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
            raise ValueError("intrinsics must be a finite 3x3 matrix")
        valid_bool = valid.astype(bool, copy=False)
        if not np.isfinite(points[valid_bool]).all():
            raise ValueError("valid points must be finite")
        if not np.isfinite(depth[valid_bool]).all() or np.any(depth[valid_bool] <= 0):
            raise ValueError("valid depth must be finite and positive")
        object.__setattr__(self, "points", points.astype(np.float32, copy=True))
        object.__setattr__(self, "depth", depth.astype(np.float32, copy=True))
        object.__setattr__(self, "intrinsics", intrinsics.astype(np.float32, copy=True))
        object.__setattr__(self, "valid", valid_bool.copy())
        for value in (self.points, self.depth, self.intrinsics, self.valid):
            value.setflags(write=False)


@dataclass(frozen=True)
class ReferenceGeometry:
    prediction: SingleImageGeometry
    instance_queries: dict[str, np.ndarray]
    background_mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        shape = self.prediction.depth.shape
        background = np.array(self.background_mask, dtype=bool, copy=True)
        if background.shape != shape:
            raise ValueError("background_mask must match prediction dimensions")
        queries: dict[str, np.ndarray] = {}
        for object_id, raw in self.instance_queries.items():
            value = np.array(raw, dtype=np.float32, copy=True)
            if value.ndim != 2 or value.shape[1:] != (2,) or not np.isfinite(value).all():
                raise ValueError(f"queries for {object_id} must have shape Nx2")
            queries[str(object_id)] = value
        object.__setattr__(self, "background_mask", background)
        object.__setattr__(self, "instance_queries", queries)
        self.background_mask.setflags(write=False)
        for value in self.instance_queries.values():
            value.setflags(write=False)

    @property
    def fingerprint(self) -> str:
        arrays = self._arrays()
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "metadata": self.metadata,
            "prediction_metadata": self.prediction.metadata,
            "query_ids": sorted(self.instance_queries),
            "array_hashes": {key: _array_hash(value) for key, value in sorted(arrays.items())},
        }
        return sha256(_canonical(payload)).hexdigest()

    def _arrays(self) -> dict[str, np.ndarray]:
        arrays = {
            "points": self.prediction.points,
            "depth": self.prediction.depth,
            "intrinsics": self.prediction.intrinsics,
            "valid": self.prediction.valid,
            "background_mask": self.background_mask,
        }
        for index, object_id in enumerate(sorted(self.instance_queries)):
            arrays[f"query_{index}"] = self.instance_queries[object_id]
        return arrays

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        arrays = self._arrays()
        manifest = {
            "schema_version": _SCHEMA_VERSION,
            "metadata": self.metadata,
            "prediction_metadata": self.prediction.metadata,
            "query_ids": sorted(self.instance_queries),
            "array_hashes": {key: _array_hash(value) for key, value in sorted(arrays.items())},
        }
        manifest["fingerprint"] = sha256(_canonical({
            "schema_version": _SCHEMA_VERSION,
            "metadata": self.metadata,
            "prediction_metadata": self.prediction.metadata,
            "query_ids": sorted(self.instance_queries),
            "array_hashes": manifest["array_hashes"],
        })).hexdigest()
        descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                np.savez_compressed(stream, **arrays, manifest=np.frombuffer(_canonical(manifest), dtype=np.uint8))
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "ReferenceGeometry":
        try:
            with np.load(Path(path), allow_pickle=False) as data:
                manifest = json.loads(np.asarray(data["manifest"], dtype=np.uint8).tobytes())
                if manifest.get("schema_version") != _SCHEMA_VERSION:
                    raise ValueError("unsupported reference geometry schema")
                arrays = {key: np.array(data[key], copy=True) for key in manifest["array_hashes"]}
        except (KeyError, json.JSONDecodeError, OSError, BadZipFile, EOFError) as exc:
            raise ValueError("invalid reference geometry cache") from exc
        actual = {key: _array_hash(value) for key, value in sorted(arrays.items())}
        if actual != manifest["array_hashes"]:
            raise ValueError("reference geometry array hash mismatch")
        queries = {object_id: arrays[f"query_{index}"] for index, object_id in enumerate(manifest["query_ids"])}
        prediction = SingleImageGeometry(
            arrays["points"], arrays["depth"], arrays["intrinsics"], arrays["valid"],
            dict(manifest["prediction_metadata"]),
        )
        artifact = cls(prediction, queries, arrays["background_mask"], dict(manifest["metadata"]))
        if artifact.fingerprint != manifest["fingerprint"]:
            raise ValueError("reference geometry manifest hash mismatch")
        return artifact


class MoGePredictor:
    def __init__(self, config: V3EvaluationConfig) -> None:
        config.validate()
        self.config = config

    def _load_model(self) -> tuple[Any, dict[str, str]]:
        key = (self.config.monocular_model_id, self.config.monocular_device)
        if key in _MOGE_MODELS:
            return _MOGE_MODELS[key]
        try:
            from moge.model.v2 import MoGeModel
        except ImportError as exc:
            raise RuntimeError("MoGe-2 is not installed") from exc
        model = MoGeModel.from_pretrained(self.config.monocular_model_id).to(self.config.monocular_device).eval()
        provenance = {
            "checkpoint_sha256": _checkpoint_sha(self.config.monocular_model_id),
            "source_revision": _module_revision(MoGeModel),
        }
        _MOGE_MODELS[key] = model, provenance
        return model, provenance

    def predict(self, image: Image.Image) -> SingleImageGeometry:
        image = image.convert("RGB")
        model, provenance = self._load_model()
        try:
            import torch
            tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div_(255)
            tensor = tensor.to(self.config.monocular_device)
            with torch.inference_mode():
                output = model.infer(tensor, resolution_level=self.config.monocular_resolution_level, apply_mask=True)
        except ImportError:
            output = model.infer(np.asarray(image), resolution_level=self.config.monocular_resolution_level, apply_mask=True)
        values = {key: _to_numpy(value) for key, value in output.items()}
        width, height = image.size
        intrinsics = np.asarray(values["intrinsics"], dtype=np.float32).copy()
        intrinsics[0, :] *= width
        intrinsics[1, :] *= height
        intrinsics[0, 2] -= .5
        intrinsics[1, 2] -= .5
        metadata = {
            "backend": "moge2", "model_id": self.config.monocular_model_id,
            "resolution_level": self.config.monocular_resolution_level,
            "intrinsics_convention": "opencv_pixel_centers", "normalized_pixel_center_offset": -0.5,
            "validity_semantics": "binary_not_calibrated_confidence", **provenance,
        }
        return SingleImageGeometry(values["points"], values["depth"], intrinsics, values["mask"], metadata)


def build_reference_geometry(
    reference_image: str | Path | Image.Image,
    scene: ReferenceScene,
    config: V3EvaluationConfig,
    predictor: Any | None = None,
) -> ReferenceGeometry:
    config.validate()
    image = _load_image(reference_image)
    width, height = image.size
    _validate_scene_size(scene, image.size)
    prediction = (predictor or MoGePredictor(config)).predict(image)
    if prediction.depth.shape != (height, width):
        raise ValueError("monocular prediction dimensions do not match the reference image")
    supports: dict[str, np.ndarray] = {}
    queries: dict[str, np.ndarray] = {}
    exclusions: dict[str, str] = {}
    excluded = np.zeros((height, width), bool)
    for object_id, obj in sorted(scene.objects.items()):
        support = _object_mask(scene, obj, image.size)
        supports[object_id] = support
        excluded |= _binary_expand(support, config.reference_mask_margin_px)
        interior = _binary_erode(support, config.reference_mask_margin_px) & prediction.valid
        coords = np.argwhere(interior)
        if not len(coords):
            queries[object_id] = np.empty((0, 2), dtype=np.float32)
            exclusions[object_id] = "no_valid_eroded_support"
        else:
            chosen = _evenly_spaced(coords, config.reference_query_count)
            queries[object_id] = chosen[:, ::-1].astype(np.float32)
    identity = reference_identity(image, scene, config, _provenance=prediction.metadata)
    metadata = {
        "reference_identity": identity,
        "image_sha256": reference_image_sha256(image),
        "image_size": [width, height],
        "validity_semantics": "binary_not_calibrated_confidence",
        "config": _identity_config(config),
        "instance_exclusions": exclusions,
    }
    return ReferenceGeometry(prediction, queries, prediction.valid & ~excluded, metadata)


def reference_identity(
    reference_image: str | Path | Image.Image,
    scene: ReferenceScene,
    config: V3EvaluationConfig,
    *,
    _provenance: dict[str, Any] | None = None,
) -> str:
    image = _load_image(reference_image)
    _validate_scene_size(scene, image.size)
    objects = []
    for object_id, obj in sorted(scene.objects.items()):
        mask = _object_mask(scene, obj, image.size)
        objects.append({"object_id": object_id, "category": obj.category, "bbox": obj.bbox, "mask_sha256": _array_hash(mask)})
    provenance = _provenance or _installed_moge_provenance(config)
    payload = {
        "reference_builder_version": "2-stratified-queries",
        "image_sha256": reference_image_sha256(image), "scene_id": scene.scene_id,
        "objects": objects, "config": _identity_config(config),
        "checkpoint_sha256": provenance.get("checkpoint_sha256"),
        "source_revision": provenance.get("source_revision"),
    }
    return sha256(_canonical(payload)).hexdigest()


def _identity_config(config: V3EvaluationConfig) -> dict[str, Any]:
    return {key: config.to_dict()[key] for key in (
        "geometry_mode", "monocular_model_id", "monocular_device", "monocular_resolution_level",
        "reference_query_count", "reference_mask_margin_px",
    )}


def _load_image(value: str | Path | Image.Image) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    with Image.open(value) as image:
        return image.convert("RGB")


def reference_image_sha256(reference_image: str | Path | Image.Image) -> str:
    image = _load_image(reference_image)
    value = image.convert("RGB")
    digest = sha256(_canonical({"mode": value.mode, "size": list(value.size)}))
    digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


# Compatibility for the geometry integration written against the initial draft.
_image_hash = reference_image_sha256


def _installed_moge_provenance(config: V3EvaluationConfig) -> dict[str, str]:
    try:
        from moge.model.v2 import MoGeModel
    except ImportError as exc:
        raise RuntimeError("MoGe-2 is required to resolve reference provenance") from exc
    return {
        "checkpoint_sha256": _checkpoint_sha(config.monocular_model_id),
        "source_revision": _module_revision(MoGeModel),
    }


def _validate_scene_size(scene: ReferenceScene, image_size: tuple[int, int]) -> None:
    annotated = scene.metadata.get("image_size") or scene.metadata.get("processed_image_size")
    if not isinstance(annotated, (list, tuple)) or len(annotated) != 2:
        raise ValueError("reference scene metadata must declare image_size")
    if tuple(int(value) for value in annotated) != image_size:
        raise ValueError("reference scene image_size does not match reference image")


def _object_mask(scene: ReferenceScene, obj: Any, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    if obj.bbox is not None:
        x0, y0, x1, y1 = (float(value) for value in obj.bbox)
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f"object {obj.object_id} bbox is outside annotation coordinates")
    if obj.mask is not None:
        mask = mask_to_bool_array(obj.mask)
    elif obj.mask_path:
        root = Path(scene.artifact_dir or ".")
        with Image.open(root / obj.mask_path) as image:
            mask = mask_to_bool_array(image)
    elif obj.bbox is not None:
        mask = mask_from_bbox(obj.bbox, image_size)
    else:
        raise ValueError(f"object {obj.object_id} has neither mask nor bbox")
    if mask.shape != (height, width):
        raise ValueError(f"object {obj.object_id} mask does not match annotation image_size")
    return mask


def _binary_expand(mask: np.ndarray, radius: int) -> np.ndarray:
    result = mask.copy()
    for _ in range(radius):
        padded = np.pad(result, 1)
        result = np.logical_or.reduce([padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]] for dy in range(3) for dx in range(3)])
    return result


def _binary_erode(mask: np.ndarray, radius: int) -> np.ndarray:
    result = mask.copy()
    for _ in range(radius):
        padded = np.pad(result, 1, constant_values=False)
        result = np.logical_and.reduce([padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]] for dy in range(3) for dx in range(3)])
    return result


def _evenly_spaced(coords: np.ndarray, count: int) -> np.ndarray:
    if len(coords) <= count:
        return coords
    # Randomize within fixed strata to avoid a regular mask aliasing into one column.
    edges = np.linspace(0,len(coords),count+1,dtype=int)
    indices = np.random.default_rng(0).integers(edges[:-1],edges[1:])
    return coords[indices]


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _checkpoint_sha(model_id: str) -> str:
    direct = Path(model_id)
    candidates = [direct] if direct.is_file() else ([direct / "model.pt"] if direct.is_dir() else [])
    try:
        from huggingface_hub import hf_hub_download
        for filename in ("model.pt", "model.safetensors", "pytorch_model.bin"):
            try:
                candidates.append(Path(hf_hub_download(model_id, filename, local_files_only=True)))
            except Exception:
                continue
    except ImportError:
        pass
    for path in candidates:
        if path.is_file():
            digest = sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()
    raise RuntimeError("could not locate the local MoGe checkpoint for provenance hashing")


def _module_revision(model_class: Any) -> str:
    module_path = Path(__import__(model_class.__module__, fromlist=["x"]).__file__).resolve()
    try:
        return subprocess.check_output(
            ["git", "-C", str(module_path.parent), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL, timeout=2,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        try:
            from importlib.metadata import version
            return f"package:{version('moge')}"
        except Exception:
            return "unknown"
