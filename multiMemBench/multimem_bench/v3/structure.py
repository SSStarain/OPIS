"""Input-only learned structure evidence for dynamic V3 objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import base64
import copy
import io
import json
import math
import os
import tempfile

from PIL import Image

from multimem_bench.schema import ObjectSignature, ObservedObject, ReferenceScene, VideoObservation
from .config import V3EvaluationConfig
from .reference import reference_image_sha256
from .articulated import (ArticulatedStructureProvider, POSE_PROTOCOL, POSE_ROUTING_VERSION, PoseConfig,
                          pose_profile, pose_runtime_identity, score_pose_pair, unavailable_pose, validate_pose_record)
from multimem_bench.vision.mask_utils import mask_to_bool_array


FAMILIES = frozenset({"parts", "connectivity", "local_shape", "material_continuity"})
STATUSES = frozenset({"supported", "contradicted", "unknown"})
PROMPT_VERSION = "v3_structure_input_only_v1"
FALLBACK_PROTOCOL = "v3_pose_then_vlm_input_only_v1"


@dataclass(frozen=True)
class StructureConfig:
    pipeline: str = "vlm"
    articulated_fallback: str = "none"
    articulated: dict[str, Any] = field(default_factory=dict)
    endpoint: str = ""
    model: str = ""
    api_key_env: str = "VLM_API_KEY"
    timeout: float = 60.0
    max_tokens: int = 2048
    min_confidence: float = 0.7
    max_claims_per_object: int = 12

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "StructureConfig":
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(data or {}) - allowed)
        if unknown:
            raise ValueError(f"unknown structure config fields: {', '.join(unknown)}")
        result = cls(**dict(data or {}))
        result.validate()
        return result

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.pipeline == "hybrid":
            result["articulated"] = PoseConfig.from_dict(self.articulated).to_dict()
        return result

    def validate(self) -> None:
        if not isinstance(self.pipeline, str) or self.pipeline not in {"vlm", "hybrid"}:
            raise ValueError("structure pipeline must be vlm or hybrid")
        if not isinstance(self.articulated_fallback, str) or self.articulated_fallback not in {"none", "vlm"}:
            raise ValueError("articulated_fallback must be none or vlm")
        if self.articulated_fallback == "vlm" and self.pipeline != "hybrid":
            raise ValueError("articulated fallback requires hybrid pipeline")
        PoseConfig.from_dict(self.articulated)
        if not isinstance(self.endpoint, str) or not isinstance(self.model, str):
            raise ValueError("endpoint and model must be strings")
        numeric = (self.timeout, self.min_confidence)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in numeric):
            raise ValueError("structure numeric settings must be finite")
        if self.timeout <= 0 or not 0 < self.min_confidence <= 1:
            raise ValueError("timeout must be positive and min_confidence must be in (0, 1]")
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int) or self.max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if (isinstance(self.max_claims_per_object, bool) or
                not isinstance(self.max_claims_per_object, int) or self.max_claims_per_object < 1):
            raise ValueError("max_claims_per_object must be a positive integer")
        if not isinstance(self.api_key_env, str) or not self.api_key_env:
            raise ValueError("api_key_env must be a non-empty string")
        if self.endpoint:
            parsed = urlparse(self.endpoint)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc or not parsed.path.endswith("/chat/completions"):
                raise ValueError("endpoint must be a full HTTP(S) chat/completions URL")


class StructureProvider(Protocol):
    def reference(self, image: Image.Image, reference: ObjectSignature) -> list[dict[str, Any]]: ...
    def check(self, reference_image: Image.Image, current_image: Image.Image,
              reference: ObjectSignature, observed: ObservedObject,
              claims: list[dict[str, Any]]) -> list[dict[str, Any]]: ...


class OpenAICompatibleStructureProvider:
    """Minimal OpenAI-compatible JSON VLM client."""

    def __init__(self, config: StructureConfig):
        config.validate()
        if not config.endpoint or not config.model:
            raise ValueError("endpoint and model are required for VLM provider construction")
        if not os.environ.get(config.api_key_env):
            raise ValueError(f"API key environment variable {config.api_key_env} is not set")
        self.config = config

    def reference(self, image: Image.Image, reference: ObjectSignature) -> list[dict[str, Any]]:
        policy = _kinematic_policy(reference.mobility.kinematic_class)
        target = _normalized_bbox(reference.bbox, image.size)
        prompt = (
            f"Image size is {image.width}x{image.height}. Target {reference.category} bbox is {target} in normalized "
            "full-image coordinates. Extract facts only from that target region. "
            f"Allowed families/policy: {policy}. Do not infer canonical anatomy, color, appearance, pose, or hidden parts. "
            f"Return at most {self.config.max_claims_per_object} claims as JSON object {{\"claims\":[{{\"fact_id\":\"...\","
            "\"family\":\"parts|connectivity|local_shape|material_continuity\",\"text\":\"...\","
            "\"reference_region\":[x1,y1,x2,y2],\"confidence\":0.0}}]}. Coordinates are normalized full-image coordinates."
        )
        images = [image]
        if self.config.articulated_fallback == "vlm":
            prompt += _instance_scope_policy()
            prompt += " Image 1 is the full input. Image 2 is its supplementary target crop."
            images.append(_target_crop(image, reference))
        return self._request(prompt, images, "claims")

    def check(self, reference_image: Image.Image, current_image: Image.Image,
              reference: ObjectSignature, observed: ObservedObject,
              claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
        policy = _kinematic_policy(reference.mobility.kinematic_class)
        prompt = (
            "Compare image 1 (fixed reference) only with image 2 (current frame). Judge only the supplied facts. "
            f"Image 1 size is {reference_image.width}x{reference_image.height}, target bbox "
            f"{_normalized_bbox(reference.bbox, reference_image.size)}. Image 2 size is {current_image.width}x{current_image.height}, "
            f"target bbox {_normalized_bbox(observed.bbox, current_image.size)}. Object policy: {policy}. "
            "A missing detection, occlusion, unobservable region, or uncertainty is unknown. When the target is clearly visible, "
            "a clearly absent required part, broken connection, discontinuity, or extra structural part is contradicted. "
            "Do not use color/appearance or "
            "category anatomy priors. Return JSON object {\"judgments\":[{\"fact_id\":\"...\","
            "\"status\":\"supported|contradicted|unknown\",\"confidence\":0.0,\"visible\":true,"
            "\"evidence\":\"...\",\"current_region\":[x1,y1,x2,y2]}]}. Facts: " +
            json.dumps(claims, sort_keys=True, separators=(",", ":"))
        )
        images = [reference_image, current_image]
        if self.config.articulated_fallback == "vlm":
            prompt += _instance_scope_policy()
            prompt += " Images 3 and 4 are supplementary target crops of images 1 and 2, respectively."
            images.extend([_target_crop(reference_image, reference), _target_crop(current_image, observed)])
        return self._request(prompt, images, "judgments")

    def _request(self, prompt: str, images: list[Image.Image], result_key: str) -> list[dict[str, Any]]:
        key = os.environ.get(self.config.api_key_env)
        if not key:
            raise RuntimeError(f"API key environment variable {self.config.api_key_env} is not set")
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend({"type": "image_url", "image_url": {"url": _data_url(image)}} for image in images)
        payload = {"model": self.config.model, "messages": [{"role": "user", "content": content}],
                   "max_tokens": self.config.max_tokens, "temperature": 0,
                   "response_format": {"type": "json_object"}}
        request = Request(self.config.endpoint, data=json.dumps(payload).encode("utf-8"), method="POST",
                          headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                   "User-Agent": "multimem-bench/3.8"})
        with urlopen(request, timeout=self.config.timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
        try:
            choices = raw["choices"]
            if not isinstance(choices, list) or not choices:
                raise ValueError
            content_value = choices[0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError("malformed VLM chat completion response") from exc
        if isinstance(content_value, str):
            lines = content_value.strip().splitlines()
            if len(lines) >= 3 and lines[0].strip() in {"```json", "```"} and lines[-1].strip() == "```":
                content_value = "\n".join(lines[1:-1])
        parsed = json.loads(content_value) if isinstance(content_value, str) else content_value
        result = parsed.get(result_key) if isinstance(parsed, dict) else None
        if not isinstance(result, list):
            raise ValueError(f"VLM response missing {result_key} list")
        return result


@dataclass
class V3StructureArtifact:
    metadata: dict[str, Any]
    claims: list[dict[str, Any]] = field(default_factory=list)
    judgments: list[dict[str, Any]] = field(default_factory=list)
    artifact_version: int = 1
    pose_references: dict[str, dict[str, Any]] = field(default_factory=dict)
    pose_measurements: list[dict[str, Any]] = field(default_factory=list)

    def save(self, path: str | Path) -> Path:
        self.validate()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": 1, "artifact_version": self.artifact_version,
                   "metadata": self.metadata, "claims": self.claims, "judgments": self.judgments,
                   "pose_references": self.pose_references, "pose_measurements": self.pose_measurements}
        encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent,
                                         prefix=f".{target.name}.", delete=False) as handle:
            handle.write(encoded)
            temporary = Path(handle.name)
        os.replace(temporary, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "V3StructureArtifact":
        raw = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda value: (_raise(f"non-finite {value}")))
        if not isinstance(raw, dict) or raw.get("schema_version") != 1:
            raise ValueError("unsupported structure artifact schema")
        artifact = cls(metadata=raw.get("metadata"), claims=raw.get("claims"), judgments=raw.get("judgments"),
                       artifact_version=raw.get("artifact_version", 0),
                       pose_references=raw.get("pose_references", {}), pose_measurements=raw.get("pose_measurements", []))
        artifact.validate()
        return artifact

    def validate(self) -> None:
        if self.artifact_version not in {1, 2, 3} or not isinstance(self.metadata, dict):
            raise ValueError("malformed structure artifact")
        if not isinstance(self.claims, list) or not isinstance(self.judgments, list):
            raise ValueError("structure claims and judgments must be lists")
        claim_ids: set[tuple[str, str]] = set()
        facts_by_reference: dict[str, set[str]] = {}
        for claim in self.claims:
            _validate_claim(claim)
            key = (claim["reference_id"], claim["fact_id"])
            if key in claim_ids:
                raise ValueError("duplicate structure claim id")
            claim_ids.add(key)
            facts_by_reference.setdefault(key[0], set()).add(key[1])
        judgment_ids: set[tuple[str, str, str, str]] = set()
        for judgment in self.judgments:
            _validate_judgment(judgment)
            key = (judgment["window_id"], judgment["reference_id"], judgment["observed_id"], judgment["fact_id"])
            if key in judgment_ids:
                raise ValueError("duplicate structure judgment id")
            judgment_ids.add(key)
            if judgment["fact_id"] not in facts_by_reference.get(judgment["reference_id"], set()):
                raise ValueError("structure judgment references foreign fact")
        if not isinstance(self.pose_references, dict) or not isinstance(self.pose_measurements, list):
            raise ValueError("malformed articulated evidence")
        if (self.pose_references or self.pose_measurements) and self.artifact_version not in {2, 3}:
            raise ValueError("pose evidence requires structure artifact version 2 or 3")
        config = StructureConfig.from_dict(self.metadata.get("structure_config", {}))
        allow_fallback = config.articulated_fallback == "vlm" and self.artifact_version == 3
        for ref_id, record in self.pose_references.items():
            if not isinstance(ref_id, str) or not ref_id or ref_id in facts_by_reference and not allow_fallback:
                raise ValueError("overlapping pose/VLM references require the fallback protocol")
            validate_pose_record(record)
        seen = set()
        for item in self.pose_measurements:
            if not isinstance(item, dict) or set(item) != {"window_id", "reference_id", "observed_id", "record"}:
                raise ValueError("malformed articulated measurement")
            key = tuple(item[k] for k in ("window_id", "reference_id", "observed_id"))
            if any(not isinstance(v, str) or not v for v in key) or key in seen:
                raise ValueError("duplicate or malformed articulated measurement ids")
            seen.add(key)
            reference = self.pose_references.get(item["reference_id"])
            if reference is None:
                raise ValueError("articulated measurement references foreign object")
            validate_pose_record(item["record"])
            if reference["profile"] != item["record"]["profile"]:
                raise ValueError("articulated measurement profile mismatch")
            if item["record"]["points"] and item["record"]["edges"] != reference["edges"]:
                raise ValueError("articulated measurement skeleton mismatch")

    def validate_context(self, scene: ReferenceScene, observation: VideoObservation,
                         evaluation_config: V3EvaluationConfig, geometry: Any = None) -> None:
        self.validate()
        cfg = StructureConfig.from_dict(self.metadata.get("structure_config", {}))
        evidence_type = "hybrid_pose_vlm" if cfg.pipeline == "hybrid" else "learned_vlm"
        for name, value in {"conditioning": "input_only", "evidence_type": evidence_type,
                            "prompt_version": PROMPT_VERSION}.items():
            if self.metadata.get(name) != value:
                raise ValueError(f"incompatible structure protocol: {name}")
        if self.metadata.get("independent_pair_checks") is not True:
            raise ValueError("structure protocol requires independent pair checks")
        config_data = self.metadata.get("structure_config")
        # Version-1 VLM artifacts predate pipeline/articulated settings.
        complete = cfg.to_dict()
        if isinstance(config_data, dict) and self.artifact_version == 1:
            complete = {k: v for k, v in complete.items() if k in config_data or k not in {"pipeline", "articulated"}}
        if isinstance(config_data, dict) and self.artifact_version <= 2 and "articulated_fallback" not in config_data:
            complete.pop("articulated_fallback", None)
        if not isinstance(config_data, dict) or complete != config_data:
            raise ValueError("structure artifact requires a complete structure config")
        if cfg.pipeline == "hybrid":
            expected_version = 3 if cfg.articulated_fallback == "vlm" else 2
            if self.artifact_version != expected_version or self.metadata.get("pose_protocol") != POSE_PROTOCOL:
                raise ValueError("incompatible articulated protocol")
            if cfg.articulated_fallback == "vlm" and self.metadata.get("fallback_protocol") != FALLBACK_PROTOCOL:
                raise ValueError("incompatible articulated fallback protocol")
            if self.metadata.get("pose_routing_version", "v1") not in {"v1", POSE_ROUTING_VERSION}:
                raise ValueError("incompatible pose routing version")
        elif self.pose_references or self.pose_measurements:
            raise ValueError("VLM-only protocol cannot contain pose evidence")
        image_hash = self.metadata.get("reference_image_sha256")
        if (not isinstance(image_hash, str) or len(image_hash) != 64 or
                any(character not in "0123456789abcdef" for character in image_hash)):
            raise ValueError("structure artifact requires a reference image hash")
        reference_size = self.metadata.get("reference_size")
        if (not isinstance(reference_size, list) or len(reference_size) != 2 or
                any(type(value) is not int or value <= 0 for value in reference_size)):
            raise ValueError("structure reference size must contain positive integer dimensions")
        expected_size = scene.metadata.get("image_size") or scene.metadata.get("processed_image_size")
        if expected_size is not None and tuple(expected_size) != tuple(reference_size):
            raise ValueError("structure reference size does not match annotation coordinates")
        expected = _context_bindings(scene, observation, evaluation_config)
        for name, value in expected.items():
            if self.metadata.get(name) != value:
                raise ValueError(f"structure artifact {name.replace('_', ' ')} binding mismatch")
        _validate_reference_image_binding(image_hash, geometry)
        dynamic = {key: obj for key, obj in scene.objects.items()
                   if obj.mobility.evaluation_track == "dynamic_identity"}
        windows = {window.window_id: window for window in observation.windows}
        if cfg.pipeline == "hybrid":
            expected_pose = {key for key, obj in dynamic.items() if obj.mobility.kinematic_class == "articulated"}
            if set(self.pose_references) != expected_pose:
                raise ValueError("articulated references do not cover the requested objects")
            if self.metadata.get("pose_mask_bindings") != _pose_mask_bindings(scene, observation):
                raise ValueError("articulated mask binding mismatch")
            for ref_id, record in self.pose_references.items():
                if record["profile"] != pose_profile(dynamic[ref_id].category, PoseConfig.from_dict(cfg.articulated),
                                                     self.metadata.get("pose_routing_version", "v1")):
                    raise ValueError("articulated reference profile does not match category routing")
                _validate_pose_target(record, dynamic[ref_id].bbox, tuple(reference_size))
            for item in self.pose_measurements:
                window = windows.get(item["window_id"])
                observed = next((obj for obj in window.objects if obj.observed_id == item["observed_id"]), None) if window else None
                if observed is None:
                    raise ValueError("articulated measurement context is unknown")
                _validate_pose_target(item["record"], observed.bbox, window.frame_size)
        for claim in self.claims:
            obj = dynamic.get(claim["reference_id"])
            if (cfg.pipeline == "hybrid" and cfg.articulated_fallback != "vlm" and obj is not None
                    and obj.mobility.kinematic_class == "articulated"):
                raise ValueError("hybrid articulated objects cannot use VLM claims")
            if obj is None or claim["family"] not in _allowed_families(obj.mobility.kinematic_class):
                raise ValueError("structure claim is not valid for a known dynamic reference")
            if not _overlaps(claim["reference_region"], _normalized_bbox(obj.bbox, tuple(reference_size))):
                raise ValueError("structure claim region does not overlap target bbox")
        for judgment in self.judgments:
            window = windows.get(judgment["window_id"])
            if (judgment["reference_id"] not in dynamic or window is None or
                    judgment["observed_id"] not in {obj.observed_id for obj in window.objects}):
                raise ValueError("structure judgment context is unknown")
            observed = next(obj for obj in window.objects if obj.observed_id == judgment["observed_id"])
            if judgment["current_region"] is not None and not _overlaps(
                    judgment["current_region"], _normalized_bbox(observed.bbox, window.frame_size)):
                raise ValueError("structure judgment region does not overlap target bbox")

    def score(self, window_id: str, reference_id: str, observed_id: str) -> dict[str, Any]:
        self.validate()
        config = StructureConfig.from_dict(self.metadata.get("structure_config", {}))
        if reference_id in self.pose_references:
            reference = self.pose_references[reference_id]
            current = next((item["record"] for item in self.pose_measurements
                            if (item["window_id"], item["reference_id"], item["observed_id"]) ==
                            (window_id, reference_id, observed_id)), None)
            pose = score_pose_pair(reference, current or unavailable_pose(reference["profile"], "missing_pose_measurement"),
                                   PoseConfig.from_dict(config.articulated))
            pose.update(backend="pose", fallback_used=False, fallback_attempted=False)
            if (pose["score"] is not None or config.articulated_fallback != "vlm"
                    or current is None or current["reason"] == "ambiguous_association"):
                return pose
            result = self._score_vlm(window_id, reference_id, observed_id, config)
            result.update(backend="vlm_fallback", fallback_used=result["score"] is not None,
                          fallback_attempted=True, fallback_reason=pose["reason"],
                          pose_coverage=pose["coverage"], pose_reference_coverage=pose["reference_coverage"],
                          pose_n_evaluable=pose["n_evaluable"], profile=reference["profile"])
            return result
        return self._score_vlm(window_id, reference_id, observed_id, config)

    def _score_vlm(self, window_id: str, reference_id: str, observed_id: str,
                   config: StructureConfig) -> dict[str, Any]:
        claims = [item for item in self.claims if item["reference_id"] == reference_id]
        judgments = {(item["fact_id"]): item for item in self.judgments
                     if item["window_id"] == window_id and item["reference_id"] == reference_id
                     and item["observed_id"] == observed_id}
        eligible = [claim for claim in claims if claim["confidence"] >= config.min_confidence]
        by_family: dict[str, list[float]] = {}
        for claim in eligible:
            judgment = judgments.get(claim["fact_id"])
            if not judgment or judgment["status"] == "unknown" or not judgment["visible"] or judgment["confidence"] < config.min_confidence:
                continue
            by_family.setdefault(claim["family"], []).append(1.0 if judgment["status"] == "supported" else 0.0)
        family_scores = {name: sum(values) / len(values) for name, values in sorted(by_family.items())}
        n_evaluable = sum(len(values) for values in by_family.values())
        score = sum(family_scores.values()) / len(family_scores) if family_scores else None
        coverage = n_evaluable / len(eligible) if eligible else None
        reason = "scored" if score is not None else "no_evaluable_structure_evidence"
        return {"score": score, "coverage": coverage, "reason": reason, "family_scores": family_scores,
                "backend": "vlm", "fallback_used": False, "fallback_attempted": False,
                "n_claims": len(claims), "n_reference_eligible": len(eligible), "n_evaluable": n_evaluable}


def prepare_v3_structure(
    scene: ReferenceScene,
    observation: VideoObservation,
    reference_image: Image.Image,
    frame_images: Mapping[int, Image.Image],
    geometry: Any,
    config: V3EvaluationConfig | None = None,
    structure_config: StructureConfig | None = None,
    provider: StructureProvider | None = None,
    reference_cache_path: str | Path | None = None,
    output_path: str | Path | None = None,
    articulated_provider: Any = None,
) -> V3StructureArtifact:
    cfg = structure_config or StructureConfig()
    cfg.validate()
    eval_cfg = config or V3EvaluationConfig()
    eval_cfg.validate()
    reference = _require_image(reference_image, "reference_image")
    image_hash = _image_hash(reference)
    expected_size = scene.metadata.get("image_size") or scene.metadata.get("processed_image_size")
    if expected_size is not None and tuple(expected_size) != reference.size:
        raise ValueError("reference image size does not match annotation coordinates")
    _validate_reference_image_binding(image_hash, geometry)
    for window in observation.windows:
        if window.frame_index in frame_images:
            current = _require_image(frame_images[window.frame_index], "frame image")
            if current.size != tuple(window.frame_size):
                raise ValueError("frame image size does not match observation coordinates")
    dynamic = {key: value for key, value in scene.objects.items()
               if value.mobility.evaluation_track == "dynamic_identity"}
    pose_objects = {key: value for key, value in dynamic.items()
                    if cfg.pipeline == "hybrid" and value.mobility.kinematic_class == "articulated"}
    fallback_enabled = cfg.pipeline == "hybrid" and cfg.articulated_fallback == "vlm"
    if provider is None and (set(dynamic) - set(pose_objects) or pose_objects and fallback_enabled):
        provider = OpenAICompatibleStructureProvider(cfg)
    pose_cfg = PoseConfig.from_dict(cfg.articulated)
    if pose_objects and articulated_provider is None:
        articulated_provider = ArticulatedStructureProvider(pose_cfg, eval_cfg)
    cache_binding = _cache_binding(scene, image_hash, cfg, eval_cfg)
    cached = _load_reference_cache(reference_cache_path, cache_binding) if reference_cache_path else None
    claims = cached if cached is not None else []
    pose_references = _load_pose_reference_cache(reference_cache_path) if cached is not None and cfg.pipeline == "hybrid" else {}
    reference_errors: list[dict[str, str]] = []
    if cached is None:
        for reference_id, obj in dynamic.items():
            if reference_id in pose_objects:
                try:
                    record = articulated_provider.reference(reference.copy(), obj)
                    validate_pose_record(record)
                    if record["profile"] != pose_profile(obj.category, pose_cfg):
                        raise ValueError("pose provider returned incompatible profile")
                    _validate_pose_target(record, obj.bbox, reference.size)
                    pose_references[reference_id] = copy.deepcopy(record)
                except (ImportError, OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                    reference_errors.append({"reference_id": reference_id, "type": type(exc).__name__,
                                             "backend": "pose", "message": str(exc)})
                    pose_references[reference_id] = unavailable_pose(pose_profile(obj.category, pose_cfg),
                                                                     f"reference_pose_failure:{type(exc).__name__}")
                if not fallback_enabled:
                    continue
            if obj.bbox is None:
                reference_errors.append({"reference_id": reference_id, "type": "missing_reference_bbox"})
                continue
            try:
                generated = provider.reference(reference.copy(), obj)
                claims.extend(_normalize_claims(generated, reference_id, obj, reference.size, cfg))
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                reference_errors.append({"reference_id": reference_id, "type": type(exc).__name__})
        if reference_cache_path:
            # Freeze usable fallback facts even if the specialist failed. This
            # prevents different generated videos from receiving new I0 facts.
            fallback_refs = {claim["reference_id"] for claim in claims if claim["confidence"] >= cfg.min_confidence}
            blocking_errors = [error for error in reference_errors if not (
                fallback_enabled and error.get("backend") == "pose" and error["reference_id"] in fallback_refs)]
            if not blocking_errors:
                _save_reference_cache(reference_cache_path, cache_binding, claims, pose_references)
            published = _load_reference_cache(reference_cache_path, cache_binding)
            if published is not None:
                claims = published
                if cfg.pipeline == "hybrid":
                    pose_references = _load_pose_reference_cache(reference_cache_path)
    if set(pose_references) != set(pose_objects):
        raise ValueError("cached pose references do not match articulated objects")
    for reference_id, record in pose_references.items():
        if record["profile"] != pose_profile(pose_objects[reference_id].category, pose_cfg):
            raise ValueError("cached pose profile does not match category routing")
        _validate_pose_target(record, pose_objects[reference_id].bbox, reference.size)
    # Validate and freeze every reference fact before a current image is exposed.
    reference_artifact = V3StructureArtifact({"structure_config": cfg.to_dict()}, copy.deepcopy(claims), [])
    reference_artifact.validate()
    for claim in claims:
        obj = dynamic.get(claim["reference_id"])
        if obj is None or claim["family"] not in _allowed_families(obj.mobility.kinematic_class):
            raise ValueError("cached structure claim is not valid for a known dynamic reference")
        if not _overlaps(claim["reference_region"], _normalized_bbox(
                obj.bbox, reference.size)):
            raise ValueError("cached claim reference region does not overlap target bbox")

    from .evaluator import evaluate_v3
    base = evaluate_v3(scene, observation, geometry, eval_cfg)
    windows = {window.window_id: window for window in observation.windows}
    judgments: list[dict[str, Any]] = []
    frame_hashes: dict[str, str] = {}
    errors: list[dict[str, str]] = []
    pose_measurements: list[dict[str, Any]] = []
    for row in base.frames:
        if row.state != "dynamic_identity" or row.presence != 1.0 or row.observed_id is None:
            continue
        window_id = str(row.diagnostics["window_id"])
        window = windows[window_id]
        if window.frame_index is None or window.frame_index not in frame_images:
            errors.append({"window_id": window_id, "reference_id": row.reference_id, "type": "missing_frame_image"})
            continue
        current = _require_image(frame_images[window.frame_index], f"frame_images[{window.frame_index}]")
        frame_hashes[str(window.frame_index)] = _image_hash(current)
        observed = next(item for item in window.objects if item.observed_id == row.observed_id)
        if row.reference_id in pose_references:
            pose_reference = pose_references[row.reference_id]
            if row.diagnostics.get("ambiguous_match"):
                record = unavailable_pose(pose_reference["profile"], "ambiguous_association")
            elif not pose_reference["points"]:
                record = unavailable_pose(pose_reference["profile"], pose_reference["reason"])
            else:
                try:
                    record = articulated_provider.check(current.copy(), observed, copy.deepcopy(pose_reference))
                    validate_pose_record(record)
                    _validate_pose_target(record, observed.bbox, current.size)
                    if record["profile"] != pose_reference["profile"] or (record["points"] and record["edges"] != pose_reference["edges"]):
                        raise ValueError("pose model changed skeleton between frames")
                except (ImportError, OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                    errors.append({"window_id": window_id, "reference_id": row.reference_id,
                                   "type": type(exc).__name__, "backend": "pose", "message": str(exc)})
                    record = unavailable_pose(pose_reference["profile"], f"current_pose_failure:{type(exc).__name__}")
            pose_measurements.append({"window_id": window_id, "reference_id": row.reference_id,
                                      "observed_id": row.observed_id, "record": copy.deepcopy(record)})
            pose_score = score_pose_pair(pose_reference, record, pose_cfg)
            if not fallback_enabled or pose_score["score"] is not None or row.diagnostics.get("ambiguous_match"):
                continue
        obj_claims = [copy.deepcopy(item) for item in claims if item["reference_id"] == row.reference_id and not item.get("error")]
        if not obj_claims:
            continue
        if row.diagnostics.get("ambiguous_match"):
            errors.append({"window_id": window_id, "reference_id": row.reference_id, "type": "ambiguous_association"})
            judgments.extend(_unknown_judgments(window_id, row.reference_id, row.observed_id,
                                                obj_claims, "ambiguous_association"))
            continue
        try:
            raw = provider.check(reference.copy(), current.copy(), scene.objects[row.reference_id], observed,
                                 copy.deepcopy([{k: v for k, v in item.items() if k != "reference_id"} for item in obj_claims]))
            judgments.extend(_normalize_judgments(raw, window_id, row.reference_id, row.observed_id,
                                                  obj_claims, observed, current.size))
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            errors.append({"window_id": window_id, "reference_id": row.reference_id, "type": type(exc).__name__})
            judgments.extend(_unknown_judgments(window_id, row.reference_id, row.observed_id, obj_claims, type(exc).__name__))
    metadata = {**_context_bindings(scene, observation, eval_cfg), "reference_image_sha256": image_hash,
                "reference_size": list(reference.size),
                "frame_image_sha256": frame_hashes, "structure_config": cfg.to_dict(),
                "provider": {"type": type(provider).__name__, "model": cfg.model}, "prompt_version": PROMPT_VERSION,
                "evidence_type": "hybrid_pose_vlm" if cfg.pipeline == "hybrid" else "learned_vlm",
                "pose_protocol": POSE_PROTOCOL if cfg.pipeline == "hybrid" else None,
                "conditioning": "input_only", "provisional": True,
                "independent_pair_checks": True, "errors": reference_errors + errors}
    if cfg.pipeline == "hybrid":
        metadata.update(pose_mask_bindings=_pose_mask_bindings(scene, observation),
                        pose_runtime=cache_binding["pose_runtime"],
                        pose_routing_version=POSE_ROUTING_VERSION,
                        pose_reference_reasons={key: record["reason"] for key, record in pose_references.items()},
                        fallback_protocol=FALLBACK_PROTOCOL if fallback_enabled else None,
                        routes={"articulated": "pose_then_vlm_fallback" if fallback_enabled else "pose_moge_landmark_proportions",
                                "deformable": "learned_vlm",
                                "rigid": "learned_vlm", "unknown": "learned_vlm"})
    artifact = V3StructureArtifact(metadata, copy.deepcopy(claims), judgments,
        artifact_version=3 if fallback_enabled else 2 if cfg.pipeline == "hybrid" else 1,
        pose_references=copy.deepcopy(pose_references), pose_measurements=pose_measurements)
    artifact.validate()
    if output_path is not None:
        artifact.save(output_path)
    return artifact


def _normalize_claims(raw: Any, reference_id: str, reference: ObjectSignature,
                      image_size: tuple[int, int], config: StructureConfig) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or len(raw) > config.max_claims_per_object:
        raise ValueError("provider claims must be a list within max_claims_per_object")
    allowed = _allowed_families(reference.mobility.kinematic_class)
    result = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != {"fact_id", "family", "text", "reference_region", "confidence"}:
            raise ValueError("malformed provider claim")
        claim = {**copy.deepcopy(item), "reference_id": reference_id}
        _validate_claim(claim)
        if claim["family"] not in allowed:
            raise ValueError("claim family is incompatible with kinematic policy")
        if not _overlaps(claim["reference_region"], _normalized_bbox(reference.bbox, image_size)):
            raise ValueError("claim reference region does not overlap target bbox")
        result.append(claim)
    if len({item["fact_id"] for item in result}) != len(result):
        raise ValueError("duplicate provider fact ids")
    return result


def _normalize_judgments(raw: Any, window_id: str, reference_id: str, observed_id: str,
                         claims: list[dict[str, Any]], observed: ObservedObject,
                         image_size: tuple[int, int]) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise ValueError("provider judgments must be a list")
    facts = {item["fact_id"] for item in claims}
    result = []
    for item in raw:
        required = {"fact_id", "status", "confidence", "visible", "evidence", "current_region"}
        if not isinstance(item, dict) or set(item) != required or item.get("fact_id") not in facts:
            raise ValueError("malformed or foreign provider judgment")
        judgment = {**copy.deepcopy(item), "window_id": window_id, "reference_id": reference_id,
                    "observed_id": observed_id}
        _validate_judgment(judgment)
        if judgment["current_region"] is not None and not _overlaps(
                judgment["current_region"], _normalized_bbox(observed.bbox, image_size)):
            raise ValueError("judgment current region does not overlap target bbox")
        result.append(judgment)
    if len({item["fact_id"] for item in result}) != len(result):
        raise ValueError("duplicate provider judgment ids")
    returned = {item["fact_id"] for item in result}
    result.extend(_unknown_judgments(window_id, reference_id, observed_id,
                                     [item for item in claims if item["fact_id"] not in returned], "missing_judgment"))
    return result


def _unknown_judgments(window_id: str, reference_id: str, observed_id: str,
                       claims: list[dict[str, Any]], reason: str) -> list[dict[str, Any]]:
    return [{"window_id": window_id, "reference_id": reference_id, "observed_id": observed_id,
             "fact_id": item["fact_id"], "status": "unknown", "confidence": 0.0, "visible": False,
             "evidence": reason, "current_region": None} for item in claims]


def _validate_claim(item: Any) -> None:
    required = {"reference_id", "fact_id", "family", "text", "reference_region", "confidence"}
    if not isinstance(item, dict) or not required <= set(item) or item.get("family") not in FAMILIES:
        raise ValueError("malformed structure claim")
    if any(not isinstance(item.get(name), str) or not item[name] for name in ("reference_id", "fact_id", "text")):
        raise ValueError("structure claim ids/text must be non-empty")
    _confidence(item.get("confidence"))
    _region(item.get("reference_region"))


def _validate_judgment(item: Any) -> None:
    required = {"window_id", "reference_id", "observed_id", "fact_id", "status", "confidence", "visible", "evidence", "current_region"}
    if not isinstance(item, dict) or not required <= set(item) or item.get("status") not in STATUSES:
        raise ValueError("malformed structure judgment")
    if any(not isinstance(item.get(name), str) or not item[name] for name in ("window_id", "reference_id", "observed_id", "fact_id", "evidence")):
        raise ValueError("structure judgment ids/evidence must be non-empty")
    if not isinstance(item.get("visible"), bool):
        raise ValueError("structure judgment visible must be boolean")
    _confidence(item.get("confidence"))
    if item.get("current_region") is not None:
        _region(item["current_region"])
    if item["status"] != "unknown" and (not item["visible"] or item["current_region"] is None):
        raise ValueError("known structure judgments require visible localized evidence")


def _confidence(value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("confidence must be finite and in [0, 1]")


def _region(value: Any) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != 4 or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value
    ) or not (0 <= value[0] < value[2] <= 1 and 0 <= value[1] < value[3] <= 1):
        raise ValueError("region must be a finite normalized [x1,y1,x2,y2] box")


def _context_bindings(scene: ReferenceScene, observation: VideoObservation,
                      config: V3EvaluationConfig) -> dict[str, Any]:
    scene_payload = {"scene_id": scene.scene_id, "objects": [{"id": key, "category": value.category,
        "bbox": value.bbox, "embedding": value.embedding, "attributes": value.attributes,
        "size_signature": value.size_signature, "mobility": value.mobility.to_dict()}
        for key, value in sorted(scene.objects.items())]}
    observation_payload = {"video_id": observation.video_id, "windows": [{"window_id": window.window_id,
        "frame_index": window.frame_index, "frame_size": window.frame_size, "objects": [{"observed_id": obj.observed_id,
        "bbox": obj.bbox, "category": obj.category, "track_id": obj.track_id, "confidence": obj.confidence,
        "embedding": obj.embedding, "attributes": obj.attributes} for obj in window.objects]} for window in observation.windows]}
    return {"scene_id": scene.scene_id, "video_id": observation.video_id,
            "scene_fingerprint": _fingerprint(scene_payload), "observation_fingerprint": _fingerprint(observation_payload),
            "evaluation_config_fingerprint": _fingerprint(config.to_dict())}


def _cache_binding(scene: ReferenceScene, image_hash: str, config: StructureConfig,
                   evaluation_config: V3EvaluationConfig | None = None) -> dict[str, Any]:
    scene_only = {"scene_id": scene.scene_id, "objects": [{"id": key, "category": obj.category, "bbox": obj.bbox,
        "mobility": obj.mobility.to_dict()} for key, obj in sorted(scene.objects.items())]}
    result = {"scene_fingerprint": _fingerprint(scene_only), "reference_image_sha256": image_hash,
              "prompt_version": PROMPT_VERSION, "structure_config": config.to_dict()}
    if config.pipeline == "hybrid":
        result.update(pose_protocol=POSE_PROTOCOL,
                      pose_routing_version=POSE_ROUTING_VERSION,
                      fallback_protocol=FALLBACK_PROTOCOL if config.articulated_fallback == "vlm" else None,
                      pose_runtime=pose_runtime_identity(PoseConfig.from_dict(config.articulated)),
                      pose_reference_masks=_pose_mask_bindings(scene)["reference"],
                      evaluation_config=(evaluation_config or V3EvaluationConfig()).to_dict())
    return result


def _load_reference_cache(path: str | Path, binding: dict[str, Any]) -> list[dict[str, Any]] | None:
    target = Path(path)
    if not target.exists():
        return None
    raw = json.loads(target.read_text(encoding="utf-8"), parse_constant=lambda value: (_raise(f"non-finite {value}")))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1 or raw.get("binding") != binding:
        raise ValueError("stale or malformed structure reference cache")
    artifact = V3StructureArtifact({"structure_config": binding["structure_config"]}, raw.get("claims"), [])
    artifact.validate()
    if binding["structure_config"].get("pipeline") == "hybrid":
        _validate_pose_reference_cache(raw.get("pose_references"))
    return copy.deepcopy(artifact.claims)


def _save_reference_cache(path: str | Path, binding: dict[str, Any], claims: list[dict[str, Any]],
                          pose_references: dict[str, Any] | None = None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"schema_version": 1, "binding": binding, "claims": claims,
                          "pose_references": pose_references or {}}, sort_keys=True, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}.", delete=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, target)
    except FileExistsError:
        pass
    finally:
        temporary.unlink(missing_ok=True)


def _validate_pose_reference_cache(records: Any) -> None:
    if not isinstance(records, dict):
        raise ValueError("missing articulated reference cache")
    for ref_id, record in records.items():
        if not isinstance(ref_id, str) or not ref_id:
            raise ValueError("invalid articulated reference id")
        validate_pose_record(record)


def _load_pose_reference_cache(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    records = raw.get("pose_references")
    _validate_pose_reference_cache(records)
    return records


def _validate_pose_target(record: dict[str, Any], bbox: Any, size: tuple[int, int]) -> None:
    for point in record["points"]:
        if point["xyz"] is None:
            continue
        x, y = point["xy"]
        if (bbox is None or not 0 <= x < size[0] or not 0 <= y < size[1]
                or not bbox[0] <= x < bbox[2] or not bbox[1] <= y < bbox[3]):
            raise ValueError("measured pose landmark is outside target bbox")


def _instance_scope_policy() -> str:
    return (
        " Treat the supplied instance region as the target, not its category label. "
        "Do not infer single versus multiple subjects from singular/plural wording. "
        "A group region is a composite target; a body fragment is only that visible fragment, not an incomplete canonical body. "
        "For mechanisms, inspect the actual visible components and connections without applying a human or animal skeleton. "
        "Assert local shape only within components with evidence of rigidity; the kinematic label alone is not such evidence. "
        "In supplementary crops, gray outside the instance mask is removed background, not a missing part or damage. "
        "Use full images for context and occlusion. All reported regions must use normalized FULL-IMAGE coordinates."
    )


def _target_crop(image: Image.Image, target: Any) -> Image.Image:
    if target.bbox is None:
        raise ValueError("target crop requires a bbox")
    x1, y1, x2, y2 = target.bbox
    bounds = (max(0, math.floor(x1)), max(0, math.floor(y1)),
              min(image.width, math.ceil(x2)), min(image.height, math.ceil(y2)))
    if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
        raise ValueError("target crop is outside image")
    cropped = image.convert("RGB").crop(bounds)
    if target.mask is not None:
        foreground = mask_to_bool_array(target.mask)
        if foreground.shape != (image.height, image.width):
            raise ValueError("target mask must use source image coordinates")
        mask = Image.fromarray(foreground.astype("uint8") * 255).crop(bounds)
        cropped = Image.composite(cropped, Image.new("RGB", cropped.size, (128, 128, 128)), mask)
    return cropped


def _pose_mask_bindings(scene: ReferenceScene, observation: VideoObservation | None = None) -> dict[str, Any]:
    def fingerprint(mask: Any) -> str | None:
        if mask is None:
            return None
        array = mask_to_bool_array(mask)
        return sha256(str(array.shape).encode("ascii") + array.tobytes()).hexdigest()
    result = {"reference": {key: fingerprint(obj.mask) for key, obj in sorted(scene.objects.items())
                            if obj.mobility.evaluation_track == "dynamic_identity" and obj.mobility.kinematic_class == "articulated"}}
    if observation is not None:
        result["current"] = {window.window_id: {obj.observed_id: fingerprint(obj.mask) for obj in window.objects}
                             for window in observation.windows}
    return result


def _allowed_families(kind: str) -> set[str]:
    if kind == "deformable":
        return {"parts", "connectivity", "material_continuity"}
    if kind in {"rigid", "articulated"}:
        return {"parts", "connectivity", "local_shape"}
    return {"parts", "connectivity"}


def _kinematic_policy(kind: str) -> str:
    if kind == "deformable":
        return ("parts, connectivity, material_continuity; allow rigid motion, viewpoint/perspective change, and legitimate deformation; "
                "do not assert local/global shape preservation")
    if kind == "articulated":
        return ("parts, connectivity, local_shape only within locally rigid parts; allow rigid motion, viewpoint/perspective change, and "
                "joint-angle changes; never use between-part distances or pose as local_shape")
    if kind == "rigid":
        return "parts, connectivity, local_shape; allow rigid motion, viewpoint change, and perspective change"
    return "parts, connectivity only; kinematics are unknown, so do not assume fixed shape, pose, or distances"


def _normalized_bbox(bbox: tuple[float, float, float, float] | None,
                     size: tuple[int, int]) -> list[float] | None:
    if bbox is None:
        return None
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    return [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]


def _overlaps(first: list[float] | tuple[float, ...], second: list[float] | None) -> bool:
    if second is None:
        return False
    return min(first[2], second[2]) > max(first[0], second[0]) and \
        min(first[3], second[3]) > max(first[1], second[1])


def _require_image(value: Any, name: str) -> Image.Image:
    if not isinstance(value, Image.Image):
        raise TypeError(f"{name} must be a PIL Image")
    return value.convert("RGB")


def _image_hash(image: Image.Image) -> str:
    return reference_image_sha256(image)


def _validate_reference_image_binding(image_hash: str | None, geometry: Any) -> None:
    identity = getattr(geometry, "metadata", {}).get("reference_identity")
    expected = identity.get("image_sha256") if isinstance(identity, dict) else None
    if expected is not None and image_hash != expected:
        raise ValueError("structure reference image does not match geometry reference image")


def _fingerprint(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _data_url(image: Image.Image) -> str:
    output = io.BytesIO()
    image.convert("RGB").save(output, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def _raise(message: str) -> None:
    raise ValueError(message)
