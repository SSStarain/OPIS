"""Global one-to-one multi-instance association without spatial anchoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from multimem_bench.schema import (
    ObjectSignature,
    ObservedObject,
    ReferenceScene,
    VideoObservation,
)
from multimem_bench.similarity import attribute_similarity, category_similarity, cosine_similarity
from multimem_bench.v2.config import V2EvaluationConfig
from multimem_bench.v2.math_utils import clamp01, linear_sum_assignment
from multimem_bench.v2.schema import ReferenceAnnotation


@dataclass(frozen=True)
class AssociationMatch:
    reference_id: str
    observed_id: str
    track_id: str | None
    score: float
    margin: float
    identity_state: str
    ambiguity_group: str | None
    components: dict[str, float | None] = field(default_factory=dict)


@dataclass
class FrameAssociation:
    window_id: str
    frame_index: int | None
    matches: list[AssociationMatch]
    missing_reference_ids: list[str]
    unmatched_observed_ids: list[str]
    excluded_reference_ids: list[str] = field(default_factory=list)


@dataclass
class AssociationBundle:
    scene_id: str
    video_id: str
    frames: list[FrameAssociation]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def associate_instances(
    scene: ReferenceScene,
    observation: VideoObservation,
    config: V2EvaluationConfig | None = None,
    *,
    expected_reference_ids: dict[str, set[str]] | None = None,
) -> AssociationBundle:
    cfg = config or V2EvaluationConfig()
    cfg.validate()
    annotation = ReferenceAnnotation.from_reference_scene(scene)
    previous_reference_by_track: dict[str, str] = {}
    frames: list[FrameAssociation] = []

    for window in observation.windows:
        expected = (
            expected_reference_ids.get(window.window_id)
            if expected_reference_ids is not None
            else None
        )
        reference_ids = [
            reference_id
            for reference_id in scene.objects
            if expected is None or reference_id in expected
        ]
        excluded = [reference_id for reference_id in scene.objects if reference_id not in reference_ids]
        observed = list(window.objects)
        scores: list[list[float]] = []
        components: dict[tuple[int, int], dict[str, float | None]] = {}
        for ref_index, reference_id in enumerate(reference_ids):
            row: list[float] = []
            reference = scene.objects[reference_id]
            for obs_index, observed_object in enumerate(observed):
                score, detail = _association_score(
                    observed_object,
                    reference,
                    previous_reference_by_track,
                    cfg,
                )
                row.append(score)
                components[(ref_index, obs_index)] = detail
            scores.append(row)

        selected = _assign_with_unmatched(scores, cfg.min_identity_similarity)
        matched_observed: set[int] = set()
        matches: list[AssociationMatch] = []
        missing: list[str] = []
        for ref_index, reference_id in enumerate(reference_ids):
            obs_index = selected.get(ref_index)
            if obs_index is None:
                missing.append(reference_id)
                continue
            matched_observed.add(obs_index)
            observed_object = observed[obs_index]
            score = scores[ref_index][obs_index]
            margin = _identity_margin(scores, ref_index, obs_index)
            group = annotation.instances[reference_id].ambiguity_group
            competing_group = _competing_ambiguity_group(
                annotation,
                reference_ids,
                scores,
                ref_index,
                obs_index,
                cfg.ambiguity_epsilon,
            )
            if margin >= cfg.min_identity_margin:
                identity_state = "matched"
            elif group is not None and competing_group == group:
                identity_state = "set_matched"
            else:
                identity_state = "ambiguous"
            matches.append(
                AssociationMatch(
                    reference_id=reference_id,
                    observed_id=observed_object.observed_id,
                    track_id=observed_object.track_id,
                    score=score,
                    margin=margin,
                    identity_state=identity_state,
                    ambiguity_group=group,
                    components=components[(ref_index, obs_index)],
                )
            )

        for match in matches:
            if match.track_id:
                previous_reference_by_track[match.track_id] = match.reference_id
        frames.append(
            FrameAssociation(
                window_id=window.window_id,
                frame_index=window.frame_index,
                matches=matches,
                missing_reference_ids=missing,
                unmatched_observed_ids=[
                    item.observed_id
                    for index, item in enumerate(observed)
                    if index not in matched_observed
                ],
                excluded_reference_ids=excluded,
            )
        )
    return AssociationBundle(scene_id=scene.scene_id, video_id=observation.video_id, frames=frames)


def _association_score(
    observed: ObservedObject,
    reference: ObjectSignature,
    previous_reference_by_track: dict[str, str],
    config: V2EvaluationConfig,
) -> tuple[float, dict[str, float | None]]:
    embedding = cosine_similarity(observed.embedding, reference.embedding)
    attributes = attribute_similarity(observed.attributes, reference.attributes)
    category = category_similarity(observed, reference)
    continuity = None
    if observed.track_id and observed.track_id in previous_reference_by_track:
        continuity = 1.0 if previous_reference_by_track[observed.track_id] == reference.object_id else 0.0

    weighted = 0.0
    if embedding is not None:
        weighted += config.embedding_weight * embedding
    if attributes is not None:
        weighted += config.attribute_weight * attributes
    weighted += config.category_weight * category
    if continuity is not None:
        weighted += config.track_continuity_weight * continuity
    if embedding is not None:
        total_weight = (
            config.embedding_weight
            + (config.attribute_weight if attributes is not None else 0.0)
            + config.category_weight
            + (config.track_continuity_weight if continuity is not None else 0.0)
        )
    else:
        total_weight = (
            config.embedding_weight
            + config.attribute_weight
            + config.category_weight
            + (config.track_continuity_weight if continuity is not None else 0.0)
        )
    score = clamp01(weighted / max(total_weight, 1e-12))
    return score, {
        "embedding_similarity": embedding,
        "attribute_similarity": attributes,
        "category_similarity": category,
        "track_continuity": continuity,
    }


def _assign_with_unmatched(
    scores: list[list[float]],
    minimum_score: float,
) -> dict[int, int]:
    if not scores:
        return {}
    observed_count = len(scores[0]) if scores[0] else 0
    reference_count = len(scores)
    weights: list[list[float]] = []
    for ref_index, row in enumerate(scores):
        dummy = [-1.0] * reference_count
        dummy[ref_index] = minimum_score
        weights.append(list(row) + dummy)
    assignment = linear_sum_assignment([[-value for value in row] for row in weights])
    return {
        ref_index: col_index
        for ref_index, col_index in assignment
        if col_index < observed_count and scores[ref_index][col_index] >= minimum_score
    }


def _identity_margin(scores: list[list[float]], ref_index: int, obs_index: int) -> float:
    selected = scores[ref_index][obs_index]
    alternatives = [
        value for index, value in enumerate(scores[ref_index]) if index != obs_index
    ]
    alternatives.extend(
        row[obs_index] for index, row in enumerate(scores) if index != ref_index
    )
    runner_up = max(alternatives, default=0.0)
    return max(0.0, selected - runner_up)


def _competing_ambiguity_group(
    annotation: ReferenceAnnotation,
    reference_ids: list[str],
    scores: list[list[float]],
    ref_index: int,
    obs_index: int,
    epsilon: float,
) -> str | None:
    selected = scores[ref_index][obs_index]
    candidates = [
        index
        for index, row in enumerate(scores)
        if index != ref_index and selected - row[obs_index] <= epsilon
    ]
    if not candidates:
        return None
    best = max(candidates, key=lambda index: scores[index][obs_index])
    return annotation.instances[reference_ids[best]].ambiguity_group
