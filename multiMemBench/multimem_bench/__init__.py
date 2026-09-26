"""Observation-first multi-object memory benchmark prototype."""

from multimem_bench.config import EvaluationConfig
from multimem_bench.pipeline import evaluate_video_observations
from multimem_bench.schema import ReferenceScene, VideoObservation

__all__ = [
    "EvaluationConfig",
    "ReferenceScene",
    "VideoObservation",
    "evaluate_video_observations",
]

