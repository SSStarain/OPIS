"""SAM3 image/video segmentation adapters.

This module is written against the public SAM3/SAM3.1 style APIs while keeping
all imports optional. If the installed package exposes a slightly different
method name, the error points at the adapter boundary instead of the evaluator.
"""

from __future__ import annotations

import importlib.util
import inspect
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from multimem_bench.vision.config import VisionConfig
from multimem_bench.vision.mask_utils import coerce_bbox, mask_to_bbox, stable_segment_id
from multimem_bench.vision.types import SegmentInstance


class Sam3UnavailableError(RuntimeError):
    pass


class Sam3ImageSegmenter:
    """SAM3 image segmenter for text-prompted foreground concepts."""

    def __init__(self, config: VisionConfig) -> None:
        if importlib.util.find_spec("sam3") is None:
            raise Sam3UnavailableError(
                "sam3 is not installed. Install facebookresearch/sam3 or provide "
                "precomputed segments."
            )
        self.config = config
        self._processor = None

    def segment_by_prompts(
        self,
        image_path: str | Path | Image.Image,
        prompts: Iterable[str],
        *,
        id_prefix: str = "ref",
    ) -> list[SegmentInstance]:
        if isinstance(image_path, Image.Image):
            image = image_path.convert("RGB")
        else:
            image = Image.open(image_path).convert("RGB")
        processor = self._load_processor()
        segments: list[SegmentInstance] = []
        for prompt_index, prompt in enumerate(prompts):
            prompt = str(prompt).strip()
            if not prompt:
                continue
            output = self._run_image_text_prompt(processor, image, prompt)
            segments.extend(_segments_from_sam3_output(
                output,
                image_size=image.size,
                category=prompt,
                prompt=prompt,
                id_prefix=f"{id_prefix}_p{prompt_index:02d}",
            ))
        return segments

    def segment_automatic(self, image_path: str | Path) -> list[SegmentInstance]:
        image = Image.open(image_path).convert("RGB")
        processor = self._load_processor()
        for method_name in (
            "generate_automatic_masks",
            "generate_masks",
            "automatic_mask_generation",
        ):
            method = getattr(processor, method_name, None)
            if method is None:
                continue
            output = method(image)
            return _segments_from_sam3_output(
                output,
                image_size=image.size,
                category=self.config.unknown_category_name,
                prompt=None,
                id_prefix="ref_auto",
            )
        raise Sam3UnavailableError(
            "The installed SAM3 image processor does not expose automatic mask "
            "generation. Use text prompts or precomputed segments."
        )

    def _load_processor(self) -> Any:
        if self._processor is not None:
            return self._processor
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        kwargs: dict[str, Any] = {"device": self.config.device}
        if self.config.sam3_checkpoint_path:
            kwargs["checkpoint_path"] = self.config.sam3_checkpoint_path
        if self.config.sam3_bpe_path:
            kwargs["bpe_path"] = self.config.sam3_bpe_path
        if self.config.sam3_model_id:
            kwargs["model_id"] = self.config.sam3_model_id
        model = build_sam3_image_model(**kwargs)
        self._processor = Sam3Processor(
            model,
            resolution=self.config.sam3_resolution,
            device=self.config.device,
            confidence_threshold=self.config.sam3_confidence_threshold,
        )
        return self._processor

    def _run_image_text_prompt(self, processor: Any, image: Image.Image, prompt: str) -> Any:
        # SAM3's fused MLP emits bfloat16 activations; the official CUDA
        # predictor supplies autocast around inference, while Sam3Processor
        # does not. Keep the same context at this adapter boundary.
        context = _sam3_autocast_context(self.config.device)
        with context:
            try:
                state = processor.set_image(image)
            except TypeError:
                state = processor.set_image(image=image)
            try:
                return processor.set_text_prompt(state=state, prompt=prompt)
            except TypeError:
                return processor.set_text_prompt(prompt=prompt, state=state)


def _sam3_autocast_context(device: str):
    if not str(device).lower().startswith("cuda"):
        return nullcontext()
    import torch

    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _infer_sam3_video_version(config: VisionConfig) -> str:
    hints = " ".join(
        value.lower()
        for value in (config.sam3_checkpoint_path, config.sam3_model_id)
        if value
    )
    if "sam3.1" in hints or "multiplex" in hints:
        return "sam3.1"
    return "sam3"


class Sam3VideoSegmenter:
    """SAM3/SAM3.1 video segmenter.

    The adapter runs one concept prompt at a time and prefixes track ids by
    prompt index, which avoids cross-prompt id collisions.
    """

    def __init__(self, config: VisionConfig) -> None:
        if importlib.util.find_spec("sam3") is None:
            raise Sam3UnavailableError(
                "sam3 is not installed. Install facebookresearch/sam3 or provide "
                "precomputed video segments."
            )
        self.config = config
        self._predictor = None

    def segment_by_prompts(
        self,
        video_path: str | Path,
        prompts: Iterable[str],
        *,
        frame_indices: set[int] | None = None,
        frame_size: tuple[int, int] | None = None,
    ) -> list[SegmentInstance]:
        predictor = self._load_predictor()
        session_id = self._start_session(predictor, video_path)
        segments: list[SegmentInstance] = []
        try:
            for prompt_index, prompt in enumerate(prompts):
                prompt = str(prompt).strip()
                if not prompt:
                    continue
                self._reset_session(predictor, session_id)
                response = self._add_text_prompt(predictor, session_id, prompt)
                for output in self._iter_video_outputs(predictor, session_id, response):
                    frame_index = _frame_index_from_output(output)
                    if frame_index is None:
                        continue
                    if frame_indices is not None and frame_index not in frame_indices:
                        continue
                    segments.extend(_segments_from_sam3_output(
                        output,
                        image_size=frame_size,
                        category=prompt,
                        prompt=prompt,
                        id_prefix=f"vid_p{prompt_index:02d}_f{frame_index:06d}",
                        frame_index=frame_index,
                        track_prefix=f"p{prompt_index:02d}",
                    ))
        finally:
            self._close_session(predictor, session_id)
        return segments

    def _load_predictor(self) -> Any:
        if self._predictor is not None:
            return self._predictor
        version = _infer_sam3_video_version(self.config)
        kwargs: dict[str, Any] = {}
        if self.config.sam3_checkpoint_path:
            kwargs["checkpoint_path"] = self.config.sam3_checkpoint_path
        if self.config.sam3_bpe_path:
            kwargs["bpe_path"] = self.config.sam3_bpe_path
        if self.config.sam3_model_id and version == "sam3":
            kwargs["model_id"] = self.config.sam3_model_id
        if version == "sam3.1":
            from sam3.model_builder import build_sam3_predictor

            if int(self.config.video_max_objects) < 1:
                raise ValueError("video_max_objects must be at least 1")
            kwargs["version"] = "sam3.1"
            kwargs["use_fa3"] = False
            kwargs["max_num_objects"] = int(self.config.video_max_objects)
            self._predictor = build_sam3_predictor(**kwargs)
        else:
            from sam3.model_builder import build_sam3_video_predictor

            if self.config.device.startswith("cuda"):
                suffix = self.config.device.split(":", 1)[1] if ":" in self.config.device else "0"
                try:
                    kwargs["gpus_to_use"] = [int(suffix)]
                except ValueError:
                    kwargs["gpus_to_use"] = [0]
            self._predictor = build_sam3_video_predictor(**kwargs)
        return self._predictor

    def _start_session(self, predictor: Any, video_path: str | Path) -> str:
        init_state = getattr(getattr(predictor, "model", None), "init_state", None)
        if init_state is not None:
            parameters = inspect.signature(init_state).parameters
            accepts_extra_kwargs = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
            if "offload_state_to_cpu" not in parameters and not accepts_extra_kwargs:
                init_kwargs: dict[str, Any] = {
                    "resource_path": str(video_path),
                    "offload_video_to_cpu": False,
                }
                for name in ("async_loading_frames", "video_loader_type"):
                    if name in parameters and hasattr(predictor, name):
                        init_kwargs[name] = getattr(predictor, name)
                inference_state = init_state(**init_kwargs)
                session_id = str(uuid.uuid4())
                now = time.time()
                predictor._all_inference_states[session_id] = {
                    "state": inference_state,
                    "session_id": session_id,
                    "start_time": now,
                    "last_use_time": now,
                }
                return session_id

        request = {
            "type": "start_session",
            "resource_path": str(video_path),
            "preload_frames": self.config.video_preload_frames,
        }
        response = predictor.handle_request(request)
        if isinstance(response, dict):
            return str(response.get("session_id") or response.get("id") or response)
        return str(response)

    def _reset_session(self, predictor: Any, session_id: str) -> None:
        if hasattr(predictor, "handle_request"):
            for req in (
                {"type": "reset_session", "session_id": session_id},
                {"type": "clear_prompts", "session_id": session_id},
            ):
                try:
                    predictor.handle_request(req)
                    return
                except Exception:
                    continue

    def _add_text_prompt(self, predictor: Any, session_id: str, prompt: str) -> Any:
        request_variants = [
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": self.config.video_prompt_frame_index,
                "text": prompt,
            },
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": self.config.video_prompt_frame_index,
                "prompt": prompt,
            },
            {
                "type": "add_text_prompt",
                "session_id": session_id,
                "frame_index": self.config.video_prompt_frame_index,
                "text": prompt,
            },
        ]
        compatibility_errors: list[str] = []
        for request in request_variants:
            try:
                return predictor.handle_request(request)
            except Exception as exc:
                if not _is_prompt_api_compatibility_error(exc):
                    fields = ", ".join(sorted(request))
                    raise Sam3UnavailableError(
                        "SAM3 text prompt execution failed "
                        f"for {request['type']} ({fields}): {exc}"
                    ) from exc
                compatibility_errors.append(
                    f"{request['type']}({', '.join(sorted(request))}): {exc}"
                )
        details = "; ".join(compatibility_errors)
        raise Sam3UnavailableError(
            f"no compatible SAM3 text prompt request variant succeeded: {details}"
        )

    def _iter_video_outputs(self, predictor: Any, session_id: str, first_response: Any) -> Iterable[Any]:
        yield first_response
        if hasattr(predictor, "handle_stream_request"):
            stream_request = {"type": "propagate_in_video", "session_id": session_id}
            try:
                for output in predictor.handle_stream_request(stream_request):
                    yield output
                return
            except Exception:
                pass
        for request in (
            {"type": "propagate_in_video", "session_id": session_id},
            {"type": "propagate", "session_id": session_id},
        ):
            try:
                output = predictor.handle_request(request)
                if isinstance(output, list):
                    yield from output
                else:
                    yield output
                return
            except Exception:
                continue

    def _close_session(self, predictor: Any, session_id: str) -> None:
        try:
            predictor.handle_request({"type": "close_session", "session_id": session_id})
        except Exception:
            return


def _is_prompt_api_compatibility_error(exc: Exception) -> bool:
    message = str(exc).lower()
    if isinstance(exc, KeyError):
        return any(field in message for field in ("text", "prompt", "type"))
    markers = (
        "invalid request type",
        "unknown request type",
        "unsupported request type",
        "unexpected keyword",
        "unexpected field",
        "missing required field",
        "missing required argument",
    )
    return any(marker in message for marker in markers)


def _segments_from_sam3_output(
    output: Any,
    *,
    image_size: tuple[int, int] | None,
    category: str,
    prompt: str | None,
    id_prefix: str,
    frame_index: int | None = None,
    track_prefix: str | None = None,
) -> list[SegmentInstance]:
    if output is None:
        return []
    outer_data = _output_dict(output)
    if outer_data.get("outputs") is not None:
        output = outer_data["outputs"]
    rows = _output_rows(output)
    if rows is not None:
        return [
            _segment_from_row(
                row,
                image_size=image_size,
                category=category,
                prompt=prompt,
                id_prefix=id_prefix,
                frame_index=frame_index,
                track_prefix=track_prefix,
            )
            for row in rows
            if row is not None
        ]

    data = _output_dict(output)
    masks = _to_numpy(_first_present(data, ("masks", "out_binary_masks", "binary_masks")))
    boxes = _to_numpy(_first_present(data, ("boxes", "out_boxes", "bboxes")))
    box_format = "xyxy"
    if boxes is None:
        boxes = _to_numpy(data.get("out_boxes_xywh"))
        box_format = "xywh"
    scores = _to_numpy(_first_present(data, ("scores", "out_scores", "out_probs", "confidences")))
    object_ids = _to_numpy(_first_present(data, ("object_ids", "out_obj_ids", "ids")))
    if masks is None and boxes is None:
        return []

    num = _infer_num_instances(masks, boxes, scores, object_ids)
    segments: list[SegmentInstance] = []
    for idx in range(num):
        mask = _slice_first_dim(masks, idx)
        bbox = None
        if mask is not None:
            bbox = mask_to_bbox(mask)
        if bbox is None and boxes is not None:
            bbox = coerce_bbox(
                _slice_first_dim(boxes, idx),
                image_size=image_size,
                box_format=box_format,
            )
        if bbox is None:
            continue
        score = float(_slice_first_dim(scores, idx)) if scores is not None else 1.0
        obj_id = _slice_first_dim(object_ids, idx) if object_ids is not None else idx
        track_id = None
        if obj_id is not None and frame_index is not None:
            track_id = f"{track_prefix or 'sam3'}:{str(obj_id)}"
        segment_id = stable_segment_id(id_prefix, bbox, category, frame_index)
        segments.append(
            SegmentInstance(
                segment_id=segment_id,
                bbox=bbox,
                category=category,
                confidence=score,
                frame_index=frame_index,
                track_id=track_id,
                source_prompt=prompt,
                mask=mask,
                metadata={"sam3_object_id": str(obj_id)},
            )
        )
    return segments


def _segment_from_row(
    row: dict[str, Any],
    *,
    image_size: tuple[int, int] | None,
    category: str,
    prompt: str | None,
    id_prefix: str,
    frame_index: int | None,
    track_prefix: str | None,
) -> SegmentInstance | None:
    mask = _first_present(row, ("mask", "segmentation"))
    bbox = None
    if mask is not None:
        bbox = mask_to_bbox(mask)
    if bbox is None:
        raw_box = _first_present(row, ("bbox", "box", "xyxy"))
        if raw_box is not None:
            bbox = coerce_bbox(raw_box, image_size=image_size)
    if bbox is None:
        return None
    score = float(row.get("score", row.get("confidence", 1.0)))
    obj_id = row.get("object_id", row.get("id"))
    track_id = None
    if obj_id is not None and frame_index is not None:
        track_id = f"{track_prefix or 'sam3'}:{str(obj_id)}"
    segment_id = stable_segment_id(id_prefix, bbox, category, frame_index)
    return SegmentInstance(
        segment_id=segment_id,
        bbox=bbox,
        category=str(row.get("category") or row.get("label") or category),
        confidence=score,
        frame_index=frame_index,
        track_id=track_id,
        source_prompt=prompt,
        mask=mask,
        metadata={"sam3_object_id": str(obj_id) if obj_id is not None else None},
    )


def _output_dict(output: Any) -> dict[str, Any]:
    if isinstance(output, dict):
        return output
    if hasattr(output, "to_dict"):
        return output.to_dict()
    if hasattr(output, "__dict__"):
        return vars(output)
    return {}


def _first_present(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _output_rows(output: Any) -> list[dict[str, Any]] | None:
    if isinstance(output, list):
        return output
    data = _output_dict(output)
    for key in ("segments", "objects", "instances", "predictions"):
        if isinstance(data.get(key), list):
            return data[key]
    return None


def _to_numpy(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        try:
            value = value.numpy()
        except TypeError:
            # PyTorch does not expose bfloat16 tensors through NumPy.
            value = value.float().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    return value


def _slice_first_dim(value: Any, idx: int) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value[idx]
    if isinstance(value, (list, tuple)):
        return value[idx]
    return value


def _infer_num_instances(*values: Any) -> int:
    for value in values:
        if value is None:
            continue
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                continue
            return int(value.shape[0])
        if isinstance(value, (list, tuple)):
            return len(value)
    return 0


def _frame_index_from_output(output: Any) -> int | None:
    data = _output_dict(output)
    for key in ("frame_index", "out_frame_idx", "frame_idx", "index"):
        if data.get(key) is not None:
            return int(data[key])
    return None
