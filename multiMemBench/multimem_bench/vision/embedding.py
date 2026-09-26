"""Object appearance embedding extractors."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from multimem_bench.vision.config import VisionConfig


class EmbeddingExtractor(Protocol):
    backend_name: str

    def embed(self, image: Image.Image) -> list[float]:
        ...

    def embed_many(self, images: list[Image.Image]) -> list[list[float]]:
        ...


class ColorHistogramEmbeddingExtractor:
    """Small deterministic fallback used when DINOv3 is unavailable."""

    backend_name = "colorhist"

    def __init__(self, bins: int = 8) -> None:
        self.bins = bins

    def embed(self, image: Image.Image) -> list[float]:
        return self.embed_many([image])[0]

    def embed_many(self, images: list[Image.Image]) -> list[list[float]]:
        return [self._embed_one(image) for image in images]

    def _embed_one(self, image: Image.Image) -> list[float]:
        arr = np.asarray(image.convert("RGB").resize((64, 64))).astype(np.float32)
        feats: list[float] = []
        for channel in range(3):
            hist, _ = np.histogram(
                arr[..., channel],
                bins=self.bins,
                range=(0.0, 255.0),
                density=False,
            )
            feats.extend(hist.astype(np.float32).tolist())
        vec = np.asarray(feats, dtype=np.float32)
        denom = float(np.linalg.norm(vec))
        if denom > 0:
            vec = vec / denom
        return [float(v) for v in vec]


class DinoV3EmbeddingExtractor:
    """DINOv3 crop embedding adapter.

    DINOv3 weights are usually supplied by a local repo/weights path in benchmark
    environments. This adapter accepts either a local `dinov3_repo_or_dir` or a
    torch.hub-compatible repo string.
    """

    backend_name = "dinov3"

    def __init__(self, config: VisionConfig) -> None:
        if importlib.util.find_spec("torch") is None:
            raise RuntimeError("torch is required for DINOv3 embeddings")
        if importlib.util.find_spec("torchvision") is None:
            raise RuntimeError("torchvision is required for DINOv3 embeddings")

        import torch
        from torchvision import transforms

        self.torch = torch
        device = config.embedding_device or config.device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.batch_size = max(1, int(config.embedding_batch_size))

        repo = config.dinov3_repo_or_dir or "facebookresearch/dinov3"
        repo_path = Path(repo)
        self._processor = None
        if _is_local_transformers_snapshot(repo_path):
            from transformers import AutoImageProcessor, AutoModel

            self._processor = AutoImageProcessor.from_pretrained(
                str(repo_path),
                local_files_only=True,
            )
            self.model = AutoModel.from_pretrained(
                str(repo_path),
                local_files_only=True,
            )
        else:
            source = "local" if repo_path.exists() else "github"
            hub_kwargs = {}
            if config.dinov3_weights_path:
                hub_kwargs["weights"] = config.dinov3_weights_path
            self.model = torch.hub.load(
                repo,
                config.dinov3_model_name,
                source=source,
                **hub_kwargs,
            )
        self.model.eval().to(self.device)
        self.transform = None
        if self._processor is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(config.embedding_image_size),
                    transforms.CenterCrop(config.embedding_image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                ]
            )

    def embed(self, image: Image.Image) -> list[float]:
        return self.embed_many([image])[0]

    def embed_many(self, images: list[Image.Image]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(images), self.batch_size):
            vectors.extend(self._embed_batch(images[start : start + self.batch_size]))
        return vectors

    def _embed_batch(self, images: list[Image.Image]) -> list[list[float]]:
        if not images:
            return []
        with self.torch.inference_mode():
            if self._processor is not None:
                inputs = self._processor(
                    images=[image.convert("RGB") for image in images],
                    return_tensors="pt",
                )
                inputs = {
                    key: value.to(self.device) if hasattr(value, "to") else value
                    for key, value in inputs.items()
                }
                out = self.model(**inputs)
            else:
                x = self.torch.stack([
                    self.transform(image.convert("RGB"))
                    for image in images
                ]).to(self.device)
                out = self.model(x)
        vec = _extract_vector(out)
        vec = vec.detach().float().cpu().numpy()
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        else:
            vec = vec.reshape(vec.shape[0], -1)
        denom = np.linalg.norm(vec, axis=1, keepdims=True)
        denom[denom <= 0] = 1.0
        vec = vec / denom
        return [[float(v) for v in row] for row in vec]


def build_embedding_extractor(config: VisionConfig) -> EmbeddingExtractor:
    backend = config.embedding_backend.lower()
    if backend == "colorhist":
        return ColorHistogramEmbeddingExtractor(config.color_hist_bins)
    if backend == "dinov3":
        return DinoV3EmbeddingExtractor(config)
    if backend != "auto":
        raise ValueError(f"unsupported embedding_backend: {config.embedding_backend}")
    if not config.dinov3_repo_or_dir and not config.dinov3_weights_path:
        return ColorHistogramEmbeddingExtractor(config.color_hist_bins)
    try:
        return DinoV3EmbeddingExtractor(config)
    except Exception:
        if config.strict_models:
            raise
        return ColorHistogramEmbeddingExtractor(config.color_hist_bins)


def _extract_vector(output: object) -> object:
    if isinstance(output, dict):
        for key in ("pooler_output", "cls_token", "x_norm_clstoken", "last_hidden_state"):
            if key in output:
                value = output[key]
                if hasattr(value, "ndim") and value.ndim == 3:
                    return value[:, 0]
                return value
    for key in ("pooler_output", "cls_token", "x_norm_clstoken", "last_hidden_state"):
        value = getattr(output, key, None)
        if value is not None:
            if hasattr(value, "ndim") and value.ndim == 3:
                return value[:, 0]
            return value
    if isinstance(output, (list, tuple)):
        return output[0]
    if hasattr(output, "ndim") and output.ndim == 3:
        return output[:, 0]
    if hasattr(output, "ndim") and output.ndim == 4:
        return output.mean(dim=(-1, -2))
    if not hasattr(output, "detach"):
        raise RuntimeError("DINOv3 model returned an unsupported output type")
    if getattr(output, "ndim", 0) == 2:
        return output
    return output.reshape(output.shape[0], math.prod(output.shape[1:]))


def _is_local_transformers_snapshot(repo_path: Path) -> bool:
    return repo_path.is_dir() and (repo_path / "config.json").is_file() and (
        repo_path / "model.safetensors"
    ).is_file()
