"""Input-relative articulated landmark measurements; no language-model judge.

MoGe samples surface locations at pose landmarks, not anatomical joint centers.
Consequently this is a provisional visible-landmark proportion metric, not a
test of canonical anatomy, topology, or the number of limbs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
import math
import re

import numpy as np
from PIL import Image

from .config import V3EvaluationConfig
from .reference import MoGePredictor, reference_image_sha256
from multimem_bench.vision.mask_utils import mask_to_bool_array


POSE_PROTOCOL = "v3_visible_landmark_proportions_v1"
POSE_ROUTING_VERSION = "v2_semantic_aliases"
HUMAN_CATEGORIES = frozenset({
    "person", "man", "woman", "boy", "girl", "child", "student", "pedestrian",
    "customer", "soldier", "musician", "baseball player", "motorcyclist",
    "astronaut", "spectator", "tennis player", "skier", "cyclist", "police officer",
    "vendor", "standing man", "sitting woman", "male character", "female character",
})
ANIMAL_CATEGORIES = frozenset({
    "dog", "cat", "cow", "horse", "sheep", "goat", "elephant", "hippo",
    "hippopotamus", "bear", "beaver", "wildebeest", "golden retriever", "skunk",
    "deer", "zebra", "giraffe", "lion", "tiger", "pig", "rabbit", "monkey",
    "fox", "wolf", "rhinoceros", "buffalo", "camel", "panda",
})
HUMAN_ALIASES = frozenset({
    "woman vendor", "older man with glasses", "man in green shirt", "woman with gray hair",
    "woman in shawl", "cowboy", "rider", "umpire", "catcher", "baby",
})
ANIMAL_ALIASES = frozenset({"bull", "black bear", "white cat"})
HUMAN_EDGES = ((5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15),
               (12, 14), (14, 16), (5, 6), (11, 12))
AP10K_NAMES = ("L_Eye", "R_Eye", "Nose", "Neck", "Root of tail", "L_Shoulder", "L_Elbow",
               "L_F_Paw", "R_Shoulder", "R_Elbow", "R_F_Paw", "L_Hip", "L_Knee", "L_B_Paw",
               "R_Hip", "R_Knee", "R_B_Paw")
# Restrict to individual limb segments; the neck-to-tail span can bend normally.
AP10K_EDGES = ((5, 6), (6, 7), (8, 9), (9, 10), (11, 12), (12, 13), (14, 15), (15, 16))


@dataclass(frozen=True)
class PoseConfig:
    human_model: str = "usyd-community/vitpose-plus-small"
    human_revision: str = "0c30b6534bb621af0162b481176742577264e36e"
    animal_model: str = ("https://huggingface.co/JunkyByte/easy_ViTPose/resolve/"
                         "e83805274e89428969355ec4afffcbc413e79188/onnx/ap10k/vitpose-s-ap10k.onnx")
    animal_device: str = "cpu"
    device: str = "cuda"
    min_keypoint_confidence: float = .5
    min_edges: int = 3
    min_coverage: float = .5
    log_length_tolerance: float = .2
    max_relative_depth_spread: float = .1
    category_profiles: dict[str, str] = field(default_factory=dict)
    custom_profiles: dict[str, dict[str, str]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PoseConfig":
        if data is not None and not isinstance(data, dict):
            raise ValueError("articulated config must be an object")
        unknown = set(data or {}) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown articulated config fields: {sorted(unknown)}")
        result = cls(**(data or {}))
        result.validate()
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        for name in ("human_model", "human_revision", "animal_model", "animal_device", "device"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty string")
        for name in ("min_keypoint_confidence", "min_coverage", "log_length_tolerance", "max_relative_depth_spread"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be finite and in (0, 1]")
        if type(self.min_edges) is not int or self.min_edges < 3:
            raise ValueError("min_edges must be an integer >= 3")
        if not isinstance(self.custom_profiles, dict) or not isinstance(self.category_profiles, dict):
            raise ValueError("pose profile settings must be objects")
        for name, profile in self.custom_profiles.items():
            if (not isinstance(name, str) or not name or name in {"human", "animal", "unsupported"}
                    or not isinstance(profile, dict) or set(profile) != {"model", "weights"}
                    or any(not isinstance(v, str) or not v for v in profile.values())):
                raise ValueError("custom profiles require a name and MMPose model/weights strings")
        for category, profile in self.category_profiles.items():
            if (not isinstance(category, str) or not category or category != normalize_category(category)
                    or not isinstance(profile, str)
                    or profile not in {"human", "animal", "unsupported", *self.custom_profiles}):
                raise ValueError("category_profiles requires normalized categories and known profiles")


def normalize_category(category: str) -> str:
    return re.sub(r"\s+", " ", category.lower().replace("_", " ").replace("-", " ")).strip()


def pose_profile(category: str, config: PoseConfig, routing_version: str = POSE_ROUTING_VERSION) -> str:
    key = normalize_category(category)
    if key in config.category_profiles:
        return config.category_profiles[key]
    if key in HUMAN_CATEGORIES:
        return "human"
    if key in ANIMAL_CATEGORIES:
        return "animal"
    if routing_version == POSE_ROUTING_VERSION:
        if key in HUMAN_ALIASES:
            return "human"
        if key in ANIMAL_ALIASES:
            return "animal"
    elif routing_version != "v1":
        raise ValueError("unknown pose routing version")
    return "unsupported"


def pose_runtime_identity(config: PoseConfig) -> dict[str, Any]:
    """Bind local checkpoints and installed adapter versions to reference caches."""
    models = [config.human_model, config.animal_model]
    models.extend(value for profile in config.custom_profiles.values() for value in profile.values())
    identities = {}
    for model in models:
        path = Path(model)
        candidates = ([path] if path.is_file() else
                      sorted(p for p in path.rglob("*") if p.suffix in {".json", ".safetensors", ".bin", ".pt", ".pth"})
                      if path.is_dir() else [])
        identities[model] = {str(p): _file_sha256(p) for p in candidates if p.is_file()}
    packages = {}
    for package in ("torch", "transformers", "rtmlib", "onnxruntime", "mmpose"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {"protocol": POSE_PROTOCOL, "routing_version": POSE_ROUTING_VERSION,
            "local_model_files": identities, "packages": packages}


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _onnx_model_path(model: str) -> str:
    # RTMLib otherwise caches downloads by basename, losing the HF revision.
    parsed = urlparse(model)
    parts = parsed.path.strip("/").split("/")
    if parsed.netloc == "huggingface.co" and len(parts) >= 5 and parts[2] == "resolve":
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id="/".join(parts[:2]), revision=parts[3], filename="/".join(parts[4:]))
    if parsed.scheme:
        raise ValueError("animal_model must be a local ONNX file or a Hugging Face resolve URL")
    if not Path(model).is_file():
        raise FileNotFoundError(f"animal ONNX model not found: {model}")
    return model


def unavailable_pose(profile: str, reason: str) -> dict[str, Any]:
    return {"profile": profile, "points": [], "edges": [], "reason": reason, "provenance": {}}


def validate_pose_record(record: Any) -> None:
    if not isinstance(record, dict) or set(record) != {"profile", "points", "edges", "reason", "provenance"}:
        raise ValueError("malformed articulated pose record")
    if any(not isinstance(record[k], str) or not record[k] for k in ("profile", "reason")):
        raise ValueError("pose profile/reason must be nonempty")
    if not isinstance(record["provenance"], dict) or not isinstance(record["points"], list) or not isinstance(record["edges"], list):
        raise ValueError("malformed pose evidence containers")
    names = set()
    for point in record["points"]:
        if not isinstance(point, dict) or set(point) != {"name", "xy", "xyz", "confidence"}:
            raise ValueError("malformed pose landmark")
        if not isinstance(point["name"], str) or not point["name"] or point["name"] in names:
            raise ValueError("duplicate or invalid landmark name")
        names.add(point["name"])
        for key, count in (("xy", 2), ("xyz", 3)):
            value = point[key]
            if key == "xyz" and value is None:
                continue
            if (not isinstance(value, list) or len(value) != count or
                    any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
                raise ValueError("pose coordinates must be finite")
        conf = point["confidence"]
        if type(conf) not in (int, float) or not math.isfinite(conf) or not 0 <= conf <= 1:
            raise ValueError("pose confidence must be in [0, 1]")
    seen = set()
    for edge in record["edges"]:
        if (not isinstance(edge, list) or len(edge) != 2 or
                any(not isinstance(k, str) or k not in names for k in edge) or edge[0] == edge[1]):
            raise ValueError("pose edge requires two known distinct landmarks")
        key = tuple(sorted(edge))
        if key in seen:
            raise ValueError("duplicate pose edge")
        seen.add(key)


def _edge_lengths(record: dict[str, Any], config: PoseConfig) -> dict[tuple[str, str], float]:
    points = {p["name"]: p for p in record["points"]}
    lengths = {}
    for a, b in record["edges"]:
        first, second = points[a], points[b]
        if any(p["xyz"] is None or p["confidence"] < config.min_keypoint_confidence for p in (first, second)):
            continue
        length = float(np.linalg.norm(np.asarray(first["xyz"]) - second["xyz"]))
        if math.isfinite(length) and length > 1e-8:
            lengths[tuple(sorted((a, b)))] = length
    return lengths


def score_pose_pair(reference: dict[str, Any], current: dict[str, Any], config: PoseConfig) -> dict[str, Any]:
    config.validate()
    validate_pose_record(reference)
    validate_pose_record(current)
    if reference["profile"] != current["profile"]:
        raise ValueError("pose profiles must match")
    if current["points"] and current["edges"] != reference["edges"]:
        raise ValueError("pose skeletons must match")
    ref, cur = _edge_lengths(reference, config), _edge_lengths(current, config)
    shared = sorted(set(ref) & set(cur))
    coverage = len(shared) / len(ref) if ref else 0.
    result = {"score": None, "coverage": coverage, "family_scores": {},
              "reason": "insufficient_landmark_evidence", "n_reference_eligible": len(ref),
              "n_evaluable": len(shared), "n_template_edges": len(reference["edges"]),
              "reference_coverage": len(ref) / len(reference["edges"]) if reference["edges"] else 0.,
              "edge_log_errors": {}, "evidence_type": "pose_moge_landmark_proportions",
              "profile": reference["profile"]}
    if len(shared) < config.min_edges or coverage < config.min_coverage:
        if not reference["points"]:
            result["reason"] = reference["reason"]
        elif not current["points"]:
            result["reason"] = current["reason"]
        return result
    # One nuisance scale per object; joint angles and global rigid pose never
    # enter the metric. Uniform size changes cannot be identified this way.
    log_ratios = np.log([cur[e] / ref[e] for e in shared])
    log_scale = float(np.median(log_ratios))
    residuals = np.abs(log_ratios - log_scale)
    score = float(np.exp(-residuals / config.log_length_tolerance).mean())
    result.update(score=score, reason="scored", family_scores={"landmark_proportions": score},
                  edge_log_errors={":".join(e): float(v) for e, v in zip(shared, residuals)},
                  log_scale=log_scale)
    return result


class PoseModelAdapter:
    """Lazy top-down inference using existing object boxes, without a detector."""

    def __init__(self, config: PoseConfig):
        self.config = config
        self.models: dict[str, Any] = {}

    def predict(self, image: Image.Image, bbox: tuple, profile: str) -> dict[str, Any]:
        if profile == "human":
            return self._human(image, bbox)
        if profile == "animal":
            return self._animal(image, bbox)
        return self._mmpose(image, bbox, profile)

    def _animal(self, image: Image.Image, bbox: tuple) -> dict[str, Any]:
        from rtmlib import ViTPose
        if "animal" not in self.models:
            model_path = _onnx_model_path(self.config.animal_model)
            self.models["animal"] = ViTPose(model_path, model_input_size=(192, 256),
                to_openpose=False, backend="onnxruntime", device=self.config.animal_device)
        # The exported easy_ViTPose checkpoint is trained with RGB normalization.
        xy, scores = self.models["animal"](np.asarray(image.convert("RGB")), bboxes=[list(bbox)])
        if np.asarray(xy).shape != (1, 17, 2) or np.asarray(scores).shape != (1, 17):
            raise ValueError("animal ViTPose profile requires the AP-10K 17-landmark head")
        return {"xy": xy[0], "scores": scores[0], "names": list(AP10K_NAMES),
                "edges": [[AP10K_NAMES[a], AP10K_NAMES[b]] for a, b in AP10K_EDGES],
                "provenance": {"pose_backend": "rtmlib_vitpose_onnx", "pose_model": self.config.animal_model,
                               "device": self.config.animal_device}}

    def _human(self, image: Image.Image, bbox: tuple) -> dict[str, Any]:
        import torch
        from transformers import AutoProcessor, VitPoseForPoseEstimation
        cfg = self.config
        if "human" not in self.models:
            processor = AutoProcessor.from_pretrained(cfg.human_model, revision=cfg.human_revision)
            model = VitPoseForPoseEstimation.from_pretrained(cfg.human_model, revision=cfg.human_revision).to(cfg.device).eval()
            self.models["human"] = processor, model
        processor, model = self.models["human"]
        x1, y1, x2, y2 = bbox
        boxes = [[[x1, y1, x2 - x1, y2 - y1]]]
        inputs = processor(images=image, boxes=boxes, return_tensors="pt").to(cfg.device)
        kwargs = {}
        if getattr(model.config.backbone_config, "num_experts", 1) > 1:
            kwargs["dataset_index"] = torch.zeros(1, dtype=torch.long, device=cfg.device)
        with torch.inference_mode():
            output = model(**inputs, **kwargs)
        prediction = processor.post_process_pose_estimation(output, boxes=boxes)[0][0]
        xy = prediction["keypoints"].detach().cpu().numpy()
        scores = prediction["scores"].detach().cpu().numpy()
        labels = model.config.id2label
        names = [labels.get(i, labels.get(str(i), str(i))) for i in range(len(xy))]
        if len(names) != 17:
            raise ValueError("human ViTPose profile requires the 17-landmark COCO head")
        return {"xy": xy, "scores": scores, "names": names,
                "edges": [[names[a], names[b]] for a, b in HUMAN_EDGES],
                "provenance": {"pose_backend": "transformers_vitpose", "pose_model": cfg.human_model,
                               "revision": getattr(model.config, "_commit_hash", None) or cfg.human_revision}}

    def _mmpose(self, image: Image.Image, bbox: tuple, profile: str) -> dict[str, Any]:
        from mmpose.apis import inference_topdown, init_model
        import mmpose
        if profile not in self.models:
            custom = self.config.custom_profiles[profile]
            self.models[profile] = init_model(custom["model"], custom["weights"], device=self.config.device)
        model = self.models[profile]
        predictions = inference_topdown(model, np.asarray(image.convert("RGB"))[:, :, ::-1].copy(),
                                        bboxes=np.asarray([bbox], dtype=np.float32), bbox_format="xyxy")
        if len(predictions) != 1:
            raise ValueError("top-down pose requires exactly one target")
        pred = predictions[0].pred_instances.cpu().numpy()
        xy, scores = pred.keypoints[0], pred.keypoint_scores[0]
        meta = model.dataset_meta
        names = [meta["keypoint_id2name"][i] for i in range(len(xy))]
        return {"xy": xy, "scores": scores, "names": names,
                "edges": [[names[a], names[b]] for a, b in meta["skeleton_links"]],
                "provenance": {"pose_backend": "mmpose", "version": mmpose.__version__,
                    "pose_model": self.config.custom_profiles[profile]}}


class ArticulatedStructureProvider:
    def __init__(self, config: PoseConfig, evaluation_config: V3EvaluationConfig,
                 pose_model: Any = None, geometry_predictor: Any = None):
        config.validate()
        self.config = config
        self.pose_model = pose_model or PoseModelAdapter(config)
        self.geometry_predictor = geometry_predictor or MoGePredictor(evaluation_config)
        self.geometry_model_id = evaluation_config.monocular_model_id
        self._reference_prediction: tuple[str, Any] | None = None
        self._current_prediction: tuple[str, Any] | None = None

    def reference(self, image: Image.Image, reference: Any) -> dict[str, Any]:
        profile = pose_profile(reference.category, self.config)
        return self._measure(image, reference.bbox, profile, initial=True, mask=reference.mask)

    def check(self, image: Image.Image, observed: Any, reference_record: dict[str, Any]) -> dict[str, Any]:
        return self._measure(image, observed.bbox, reference_record["profile"], initial=False, mask=observed.mask)

    def _measure(self, image: Image.Image, bbox: Any, profile: str, initial: bool, mask: Any = None) -> dict[str, Any]:
        if profile == "unsupported":
            return unavailable_pose(profile, "unsupported_articulated_category")
        if bbox is None:
            return unavailable_pose(profile, "missing_target_bbox")
        x1, y1, x2, y2 = (float(v) for v in bbox)
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
            raise ValueError("invalid pose target bbox")
        foreground = mask_to_bool_array(mask) if mask is not None else None
        if foreground is not None and foreground.shape != (image.height, image.width):
            raise ValueError("pose mask must use source image coordinates")
        raw = self.pose_model.predict(image, (x1, y1, x2, y2), profile)
        image_hash = reference_image_sha256(image)
        slot = "_reference_prediction" if initial else "_current_prediction"
        cached = getattr(self, slot)
        if cached is None or cached[0] != image_hash:
            cached = (image_hash, self.geometry_predictor.predict(image))
            setattr(self, slot, cached)
        prediction = cached[1]
        if prediction.points.shape[:2] != (image.height, image.width):
            raise ValueError("pose geometry must use source image coordinates")
        xy, scores, names = np.asarray(raw["xy"]), np.asarray(raw["scores"]), raw["names"]
        if xy.shape != (len(names), 2) or scores.shape != (len(names),):
            raise ValueError("malformed pose model output dimensions")
        points = []
        for name, position, confidence in zip(names, xy, scores):
            if not np.isfinite(position).all() or not math.isfinite(float(confidence)):
                raise ValueError("nonfinite pose output")
            x, y = (float(v) for v in position)
            confidence = float(np.clip(confidence, 0., 1.))
            xyz = None
            if (confidence >= self.config.min_keypoint_confidence and
                    max(0., x1) <= x < min(image.width, x2) and max(0., y1) <= y < min(image.height, y2)):
                ix, iy = int(x), int(y)
                ys, xs = slice(max(0, iy - 1), min(image.height, iy + 2)), slice(max(0, ix - 1), min(image.width, ix + 2))
                valid = prediction.valid[ys, xs]
                if foreground is not None:
                    valid = valid & foreground[ys, xs]
                depths = prediction.depth[ys, xs][valid]
                if (prediction.valid[iy, ix] and (foreground is None or foreground[iy, ix])
                        and valid.mean() >= .75 and len(depths) and
                        (np.max(depths) - np.min(depths)) / np.median(depths) <= self.config.max_relative_depth_spread):
                    xyz = prediction.points[iy, ix].astype(float).tolist()
            points.append({"name": name, "xy": [x, y], "xyz": xyz, "confidence": confidence})
        result = {"profile": profile, "points": points, "edges": raw["edges"], "reason": "available",
                  "provenance": {**raw["provenance"], "geometry_model": self.geometry_model_id,
                                 "geometry": prediction.metadata, "protocol": POSE_PROTOCOL,
                                 "foreground_gate": "instance_mask" if foreground is not None else "bbox_only"}}
        validate_pose_record(result)
        return result
