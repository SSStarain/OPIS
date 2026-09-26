"""Command line entry points."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

from multimem_bench.config import EvaluationConfig
from multimem_bench.io import load_json, load_reference_scene, load_video_observation
from multimem_bench.pipeline import evaluate_video_observations, write_run_result
from multimem_bench.vision.config import VisionConfig
from multimem_bench.v2.config import V2EvaluationConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multimem-eval",
        description="Evaluate observation-first multi-object memory artifacts.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    eval_parser = sub.add_parser("eval", help="Evaluate reference and observation JSON artifacts.")
    eval_parser.add_argument("--reference", required=True, help="Path to reference_scene.json")
    eval_parser.add_argument("--observations", required=True, help="Path to video_observation.json")
    eval_parser.add_argument("--output", required=True, help="Output run directory")
    eval_parser.add_argument("--config", help="Optional JSON config override")

    demo_parser = sub.add_parser("demo", help="Run the bundled simulated tomato example.")
    demo_parser.add_argument("--output", default="runs/demo", help="Output run directory")
    demo_parser.add_argument("--config", help="Optional JSON config override")

    ref_parser = sub.add_parser(
        "build-reference",
        help="Build reference_scene.json from a reference image.",
    )
    ref_parser.add_argument("--image", required=True, help="Reference image path")
    ref_parser.add_argument("--output", required=True, help="Output reference_scene.json path")
    ref_parser.add_argument("--scene-id", help="Scene id stored in the artifact")
    ref_parser.add_argument("--prompt-zoom", type=float, help="Requested zoom from the text prompt")
    ref_parser.add_argument("--vision-config", help="Optional JSON vision config override")
    ref_parser.add_argument("--segments", help="Optional precomputed reference segments JSON")
    ref_parser.add_argument("--prompts", help="Comma-separated SAM3 image prompts or noun phrases")
    ref_parser.add_argument("--prompt-file", help="JSON or text file with SAM3 image prompts or noun phrases")
    ref_parser.add_argument("--assets-dir", help="Directory for saved masks/crops")
    ref_parser.add_argument(
        "--strict-models",
        action="store_true",
        help="Fail if requested optional models are unavailable.",
    )

    obs_parser = sub.add_parser(
        "observe-video",
        help="Build video_observation.json from a generated video or frame folder.",
    )
    obs_parser.add_argument("--video", required=True, help="Generated video path or frame folder")
    obs_parser.add_argument("--reference", required=True, help="reference_scene.json path")
    obs_parser.add_argument("--output", required=True, help="Output video_observation.json path")
    obs_parser.add_argument("--video-id", help="Video id stored in the artifact")
    obs_parser.add_argument("--vision-config", help="Optional JSON vision config override")
    obs_parser.add_argument("--segments", help="Optional precomputed video segments JSON")
    obs_parser.add_argument(
        "--frames",
        help="Optional frame folder used for crops/embeddings when --video is an mp4.",
    )
    obs_parser.add_argument("--prompts", help="Comma-separated SAM3 video prompts")
    obs_parser.add_argument("--prompt-file", help="JSON or text file with SAM3 video prompts")
    obs_parser.add_argument("--assets-dir", help="Directory for saved masks/crops")
    obs_parser.add_argument(
        "--strict-models",
        action="store_true",
        help="Fail if requested optional models are unavailable.",
    )

    obs_batch_parser = sub.add_parser(
        "observe-video-batch",
        help="Build multiple video_observation.json artifacts while reusing heavy vision models.",
    )
    obs_batch_parser.add_argument("--manifest", required=True, help="JSON manifest with a jobs list")
    obs_batch_parser.add_argument("--reference", help="Default reference_scene.json path for jobs")
    obs_batch_parser.add_argument("--vision-config", help="Optional JSON vision config override")
    obs_batch_parser.add_argument("--prompts", help="Default comma-separated SAM3 video prompts")
    obs_batch_parser.add_argument("--prompt-file", help="Default JSON or text file with SAM3 video prompts")
    obs_batch_parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip jobs whose output video_observation.json already exists.",
    )
    obs_batch_parser.add_argument(
        "--strict-models",
        action="store_true",
        help="Fail if requested optional models are unavailable.",
    )

    geometry_parser = sub.add_parser(
        "prepare-geometry",
        help="Prepare and cache reference-first feed-forward geometry for V2.",
    )
    geometry_parser.add_argument("--reference-image", required=True, help="Initial reference image")
    geometry_parser.add_argument("--reference", required=True, help="reference_scene.json path")
    geometry_parser.add_argument("--observations", required=True, help="video_observation.json path")
    source = geometry_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="Generated encoded video")
    source.add_argument("--frames", help="Frame directory containing source or sampled frames")
    geometry_parser.add_argument("--output", required=True, help="Geometry cache directory")
    geometry_parser.add_argument("--config", help="Optional V2 JSON config")
    geometry_parser.add_argument("--backend", choices=("vggt", "vggt_omega"))
    geometry_parser.add_argument("--checkpoint", help="Local geometry checkpoint")
    geometry_parser.add_argument("--strict-geometry", action="store_true")
    geometry_parser.add_argument("--force", action="store_true", help="Ignore a matching cache entry")

    eval_v2_parser = sub.add_parser(
        "eval-v2",
        help="Run pure-initial-frame Static-3D evaluation.",
    )
    eval_v2_parser.add_argument("--reference", required=True, help="reference_scene.json path")
    eval_v2_parser.add_argument("--observations", required=True, help="video_observation.json path")
    eval_v2_parser.add_argument("--geometry", required=True, help="geometry_manifest.json path")
    eval_v2_parser.add_argument("--output", required=True, help="V2 result directory")
    eval_v2_parser.add_argument("--config", help="Optional V2 JSON config")
    eval_v2_parser.add_argument(
        "--track",
        choices=("static_3d",),
        help="Override the track selected in the V2 config",
    )

    geometry_v3_parser = sub.add_parser(
        "prepare-geometry-v3",
        help="Prepare independent reference-to-frame geometry for V3.",
    )
    geometry_v3_parser.add_argument("--reference", required=True, help="reference_scene.json path")
    geometry_v3_parser.add_argument("--reference-image", required=True)
    geometry_v3_parser.add_argument("--reference-geometry", required=True, help="Immutable reference geometry NPZ")
    geometry_v3_parser.add_argument("--frames", required=True, help="Frame directory")
    geometry_v3_parser.add_argument("--output", required=True, help="V3 geometry JSON")
    geometry_v3_parser.add_argument("--config", help="Optional V3 JSON config")
    geometry_v3_parser.add_argument("--matcher", choices=("auto", "mast3r", "opencv"), help="Override matcher in V3 config (default: mast3r)")
    geometry_v3_parser.add_argument("--matcher-checkpoint", help="Local MASt3R checkpoint or Hugging Face model id")

    structure_v3_parser = sub.add_parser(
        "prepare-structure-v3",
        help="Prepare input-grounded VLM or hybrid pose/VLM structure evidence for V3.",
    )
    structure_v3_parser.add_argument("--reference", required=True)
    structure_v3_parser.add_argument("--reference-image", required=True)
    structure_v3_parser.add_argument("--observations", required=True)
    structure_v3_parser.add_argument("--geometry", required=True)
    structure_v3_parser.add_argument("--frames", required=True)
    structure_v3_parser.add_argument("--output", required=True)
    structure_v3_parser.add_argument("--config", help="Optional V3 JSON config")
    structure_v3_parser.add_argument("--structure-config", required=True, help="Structure provider JSON config")
    structure_v3_parser.add_argument(
        "--reference-structure-cache",
        help="Optional immutable reference-claims cache shared across videos",
    )

    reference_v3_parser = sub.add_parser(
        "prepare-reference-v3",
        help="Build or validate immutable single-image reference geometry for V3.",
    )
    reference_v3_parser.add_argument("--reference", required=True, help="reference_scene.json path")
    reference_v3_parser.add_argument("--reference-image", required=True, help="Initial reference image")
    reference_v3_parser.add_argument("--output", required=True, help="Reference geometry NPZ")
    reference_v3_parser.add_argument("--config", help="Optional V3 JSON config")
    reference_v3_parser.add_argument("--rebuild-invalid", action="store_true",
                                     help="Regenerate stale or corrupt reference caches")

    eval_v3_parser = sub.add_parser(
        "eval-v3",
        help="Run independently anchored V3 evaluation.",
    )
    eval_v3_parser.add_argument("--reference", required=True)
    eval_v3_parser.add_argument("--observations", required=True)
    eval_v3_parser.add_argument("--geometry", required=True)
    eval_v3_parser.add_argument("--structure", help="Optional V3 structure evidence artifact")
    eval_v3_parser.add_argument("--output", required=True)
    eval_v3_parser.add_argument("--config")

    workflow_parser = sub.add_parser(
        "run-benchmark",
        help="Generate or import a video and run the complete MultiMemBench V2 pipeline.",
        description=(
            "Generate or import a video and run the complete MultiMemBench V2 "
            "pipeline."
        ),
    )
    from multimem_bench.workflow.orchestrator import add_run_benchmark_arguments

    add_run_benchmark_arguments(workflow_parser)

    workflow_v3_parser = sub.add_parser(
        "run-benchmark_v3",
        help="Generate or import a video and evaluate V3, or resume an existing run.",
        description=("Generate/import and evaluate V3, or resume an existing run. "
                     "Export VLM_BASE_URL, VLM_MODEL and VLM_API_KEY to enable hybrid structure evaluation."),
    )
    add_run_benchmark_arguments(workflow_v3_parser, v3=True)
    workflow_v3_parser.add_argument("--force", action="store_true", help="Recompute V3 geometry and evaluation")
    workflow_v3_parser.add_argument("--force-structure", action="store_true", help="Refresh structure evidence and scores while reusing valid geometry")
    workflow_v3_parser.add_argument("--matcher", choices=("auto", "mast3r", "opencv"), help="Override matcher in V3 config (default: mast3r)")
    workflow_v3_parser.add_argument("--matcher-checkpoint", type=Path)
    workflow_v3_parser.add_argument("--v3-config", type=Path)
    workflow_v3_parser.add_argument("--structure-config", type=Path)
    workflow_v3_parser.add_argument("--no-structure", action="store_true", help="Explicitly disable structure evaluation, including on resume")
    workflow_v3_parser.add_argument("--vlm-credentials", type=Path, help="Internal testing: optional endpoint/model/api_key JSON; public usage exports VLM_BASE_URL, VLM_MODEL and VLM_API_KEY")
    return parser


def _load_config(path: str | None) -> EvaluationConfig:
    if not path:
        return EvaluationConfig()
    return EvaluationConfig.from_dict(load_json(path))


def _load_vision_config(path: str | None, *, strict_models: bool = False) -> VisionConfig:
    data = load_json(path) if path else None
    if isinstance(data, dict):
        from multimem_bench.paths import resolve_config_path

        config_dir = Path(path).expanduser().resolve().parent
        for field in (
            "sam3_checkpoint_path",
            "sam3_bpe_path",
            "dinov3_repo_or_dir",
            "dinov3_weights_path",
        ):
            if isinstance(data.get(field), str):
                data[field] = resolve_config_path(data[field], config_dir=config_dir)
    config = VisionConfig.from_dict(data) if data else VisionConfig()
    if strict_models:
        config.strict_models = True
    return config


def _load_v2_config(path: str | None) -> V2EvaluationConfig:
    data = load_json(path) if path else None
    if isinstance(data, dict) and isinstance(data.get("geometry_checkpoint"), str):
        from multimem_bench.paths import resolve_config_path

        data["geometry_checkpoint"] = resolve_config_path(
            data["geometry_checkpoint"],
            config_dir=Path(path).expanduser().resolve().parent,
        )
    return V2EvaluationConfig.from_dict(data)


def _load_prompts(csv_text: str | None, prompt_file: str | None) -> list[str] | None:
    prompts: list[str] = []
    if csv_text:
        prompts.extend(part.strip() for part in csv_text.split(",") if part.strip())
    if prompt_file:
        path = Path(prompt_file)
        if path.suffix.lower() == ".json":
            data = load_json(path)
            if isinstance(data, dict):
                values = data.get("prompts") or data.get("categories") or []
            else:
                values = data
            prompts.extend(str(item).strip() for item in values if str(item).strip())
        else:
            prompts.extend(
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
    return prompts or None


def _run_eval(reference: str, observations: str, output: str, config: EvaluationConfig) -> int:
    scene = load_reference_scene(reference)
    video = load_video_observation(observations)
    result = evaluate_video_observations(scene, video, config)
    paths = write_run_result(result, output)
    summary = result.summary
    print(f"scene_id: {summary.get('scene_id')}")
    print(f"video_id: {summary.get('video_id')}")
    print(f"num_valid_windows: {summary.get('num_valid_windows')}/{summary.get('num_windows')}")
    key_metrics = summary.get("key_metrics", {})
    for key in (
        "precision_recall_score",
        "object_memory_error_rate",
        "missing_rate",
        "hallucination_rate",
        "position_error_rate",
        "shape_error_rate",
        "mean_primary_mask_iou",
        "mean_expected_coverage",
        "mean_observed_coverage",
        "mean_center_error_ratio",
        "track_match_stability",
        "id_switch_rate",
        "track_fragmentation_rate",
        "reappearance_identity_failure_rate",
        "unexpected_disappearance_rate",
        "temporal_appearance_drift",
        "zoom_compliance_score",
        "zoom_pass_rate",
        "mean_actual_zoom",
        "mean_zoom_relative_error",
    ):
        print(f"{key}: {key_metrics.get(key)}")
    print(f"summary: {paths['summary']}")
    return 0


def _run_build_reference(args: argparse.Namespace) -> int:
    from multimem_bench.vision.reference_builder import build_reference_scene_artifact

    config = _load_vision_config(args.vision_config, strict_models=args.strict_models)
    prompts = _load_prompts(args.prompts, args.prompt_file)
    result = build_reference_scene_artifact(
        image_path=args.image,
        output_path=args.output,
        config=config,
        scene_id=args.scene_id,
        prompt_zoom=args.prompt_zoom,
        prompts=prompts,
        precomputed_segments_path=args.segments,
        assets_dir=args.assets_dir,
    )
    print(f"reference_scene: {result.reference_path}")
    print(f"assets_dir: {result.assets_dir}")
    print(f"num_raw_segments: {result.num_raw_segments}")
    print(f"num_final_objects: {result.num_final_objects}")
    return 0


def _run_observe_video(args: argparse.Namespace) -> int:
    from multimem_bench.vision.video_observer import build_video_observation_artifact

    config = _load_vision_config(args.vision_config, strict_models=args.strict_models)
    prompts = _load_prompts(args.prompts, args.prompt_file)
    result = build_video_observation_artifact(
        video_path=args.video,
        reference_path=args.reference,
        output_path=args.output,
        config=config,
        video_id=args.video_id,
        prompts=prompts,
        precomputed_segments_path=args.segments,
        frames_path=args.frames,
        assets_dir=args.assets_dir,
    )
    print(f"video_observation: {result.observation_path}")
    print(f"assets_dir: {result.assets_dir}")
    print(f"sampled_frame_indices: {result.sampled_frame_indices}")
    print(f"num_segments: {result.num_segments}")
    return 0


def _run_prepare_geometry(args: argparse.Namespace) -> int:
    import tempfile

    from multimem_bench.v2.geometry.prepare import prepare_geometry_artifact
    from multimem_bench.vision.frame_source import extract_video_frames_with_ffmpeg

    config = _load_v2_config(args.config)
    if args.backend:
        config.geometry_backend = args.backend
    if args.checkpoint:
        config.geometry_checkpoint = args.checkpoint
    if args.strict_geometry:
        config.geometry_strict = True
    config.validate()
    scene = load_reference_scene(args.reference)
    observation = load_video_observation(args.observations)
    frame_indices = [
        int(window.frame_index)
        for window in observation.windows
        if window.frame_index is not None
    ]

    if args.frames:
        frame_images = _load_selected_frame_paths(args.frames, frame_indices)
    else:
        with tempfile.TemporaryDirectory(prefix="multimem_geometry_frames_") as tmp:
            extracted = extract_video_frames_with_ffmpeg(
                args.video,
                tmp,
                frame_indices=frame_indices,
            )
            frame_images = _load_selected_frame_paths(extracted, frame_indices)
            result = prepare_geometry_artifact(
                reference_image=args.reference_image,
                scene=scene,
                observation=observation,
                frame_images=frame_images,
                output_dir=args.output,
                config=config,
                force=args.force,
            )
            print(f"geometry_manifest: {result.manifest_path}")
            print(f"geometry_status: {result.status}")
            print(f"cache_hit: {result.cache_hit}")
            return 0

    result = prepare_geometry_artifact(
        reference_image=args.reference_image,
        scene=scene,
        observation=observation,
        frame_images=frame_images,
        output_dir=args.output,
        config=config,
        force=args.force,
    )
    print(f"geometry_manifest: {result.manifest_path}")
    print(f"geometry_status: {result.status}")
    print(f"cache_hit: {result.cache_hit}")
    return 0


def _load_selected_frame_paths(
    directory: str | Path,
    frame_indices: list[int],
) -> dict[int, Any]:
    from PIL import Image

    from multimem_bench.vision.frame_source import list_frame_paths

    paths = list_frame_paths(directory)
    if len(paths) < len(frame_indices):
        raise ValueError(
            f"frame directory contains {len(paths)} images for {len(frame_indices)} sampled frames"
        )
    numeric: dict[int, Path] = {}
    for path in paths:
        try:
            numeric[int(path.stem)] = path
        except ValueError:
            numeric = {}
            break
    if numeric and all(index in numeric for index in frame_indices):
        selected = [numeric[index] for index in frame_indices]
    elif len(paths) == len(frame_indices):
        selected = paths
    else:
        selected = [paths[index] for index in frame_indices if 0 <= index < len(paths)]
        if len(selected) != len(frame_indices):
            raise ValueError("frame directory cannot be mapped to observation frame indices")
    return {
        index: Image.open(path).convert("RGB")
        for index, path in zip(frame_indices, selected)
    }


def _run_eval_v2(args: argparse.Namespace) -> int:
    from multimem_bench.v2.evaluator import evaluate_v2, write_v2_result
    from multimem_bench.v2.geometry.artifact import GeometryBundle

    config = _load_v2_config(args.config)
    if args.track:
        config.track = args.track
    config.validate()
    scene = load_reference_scene(args.reference)
    observation = load_video_observation(args.observations)
    geometry = GeometryBundle.load(args.geometry)
    result = evaluate_v2(scene, observation, geometry, config)
    paths = write_v2_result(result, args.output)
    print(f"scene_id: {result.scene_id}")
    print(f"video_id: {result.video_id}")
    for name, report in result.reports.items():
        print(f"{name}.headline_score: {report.headline_score}")
        print(f"{name}.geometry_coverage: {report.geometry_coverage}")
        print(f"{name}.observability_coverage: {report.observability_coverage}")
        print(f"{name}.eligible: {report.eligible}")
    print(f"summary: {paths['summary']}")
    return 0


def _load_v3_config(path: str | None):
    from multimem_bench.v3.config import V3EvaluationConfig

    data = load_json(path) if path else None
    if path:
        from multimem_bench.paths import resolve_config_path

        base = Path(path).expanduser().resolve().parent
        for field in ("monocular_model_id", "matcher_model_id"):
            if not isinstance(data, dict) or field not in data:
                continue
            data[field] = resolve_config_path(str(data[field]), config_dir=base)
    return V3EvaluationConfig.from_dict(data)


def _annotation_space_image(reference_image: str | Path, scene: Any) -> Path:
    image = Path(reference_image).expanduser().resolve()
    from PIL import Image

    with Image.open(image) as source:
        actual = source.size
    expected = scene.metadata.get("image_size") or scene.metadata.get("processed_image_size")
    if isinstance(expected, (list, tuple)) and len(expected) == 2 and actual != tuple(map(int, expected)):
        processed = scene.metadata.get("processed_image_path")
        if not processed:
            raise ValueError("reference image does not match annotation coordinates and no processed_image_path is declared")
        candidate = Path(str(processed))
        if not candidate.is_absolute():
            candidate = Path(scene.artifact_dir or ".") / candidate
        image = candidate.resolve()
        if not image.is_file():
            raise ValueError(f"processed reference image does not exist: {image}")
    return image


def _run_prepare_reference_v3(args: argparse.Namespace) -> int:
    from multimem_bench.v3.reference import (
        ReferenceGeometry,
        build_reference_geometry,
        reference_identity,
    )

    config = _load_v3_config(args.config)
    scene = load_reference_scene(args.reference)
    image = _annotation_space_image(args.reference_image, scene)
    output = Path(args.output)
    if output.is_file():
        expected = reference_identity(image, scene, config)
        try:
            cached = ReferenceGeometry.load(output)
            if cached.metadata.get("reference_identity") != expected:
                raise ValueError(f"stale reference geometry cache; refusing to overwrite: {output}")
        except (ValueError, OSError) as exc:
            if not getattr(args, "rebuild_invalid", False):
                raise
            print(f"reference_cache: rebuilding ({exc})")
        else:
            print(f"reference_geometry: {output}")
            print("reference_cache: reused")
            return 0
    artifact = build_reference_geometry(image, scene, config)
    artifact.save(output)
    print(f"reference_geometry: {output}")
    print("reference_cache: created")
    return 0


def _run_prepare_geometry_v3(args: argparse.Namespace) -> int:
    from multimem_bench.v3.geometry import prepare_v3_geometry
    from multimem_bench.v3.reference import ReferenceGeometry, reference_identity

    config = _load_v3_config(args.config)
    if args.matcher:
        config.matcher = args.matcher
    if args.matcher_checkpoint:
        config.matcher_model_id = str(args.matcher_checkpoint)
    scene = load_reference_scene(args.reference)
    reference_image = _annotation_space_image(args.reference_image, scene)
    reference_geometry = ReferenceGeometry.load(args.reference_geometry)
    expected_identity = reference_identity(reference_image, scene, config)
    if reference_geometry.metadata.get("reference_identity") != expected_identity:
        raise ValueError("stale reference geometry does not match image, annotation, config, or checkpoint")
    frame_paths = _load_selected_frame_paths(args.frames, _frame_indices_from_directory(args.frames))
    result = prepare_v3_geometry(
        reference_image=reference_image,
        frame_images=frame_paths,
        output_path=args.output,
        config=config,
        reference_geometry=reference_geometry,
    )
    print(f"geometry_v3: {args.output}")
    print(f"geometry_status: {result.status}")
    print(f"geometry_backend: {result.backend}")
    return 0


def _run_prepare_structure_v3(args: argparse.Namespace) -> int:
    from PIL import Image

    from multimem_bench.v3.geometry import V3GeometryArtifact
    from multimem_bench.v3.structure import StructureConfig, prepare_v3_structure

    scene = load_reference_scene(args.reference)
    observation = load_video_observation(args.observations)
    config = _load_v3_config(args.config)
    structure_config = StructureConfig.from_dict(load_json(args.structure_config))
    structure_config.validate()
    geometry = V3GeometryArtifact.load(args.geometry)
    expected_sizes: dict[int, tuple[int, int]] = {}
    for window in observation.windows:
        if window.frame_index is None:
            continue
        index = int(window.frame_index)
        previous = expected_sizes.setdefault(index, window.frame_size)
        if previous != window.frame_size:
            raise ValueError(f"inconsistent frame_size for repeated frame_index {index}")
    frame_images = _load_exact_indexed_frames(args.frames, set(expected_sizes))
    for index, image in list(frame_images.items()):
        expected = expected_sizes[index]
        if image.size != expected:
            frame_images[index] = image.resize(expected, Image.Resampling.LANCZOS)
    reference_path = _annotation_space_image(args.reference_image, scene)
    with Image.open(reference_path) as source:
        reference_image = source.convert("RGB").copy()
    prepare_v3_structure(
        scene=scene,
        observation=observation,
        reference_image=reference_image,
        frame_images=frame_images,
        geometry=geometry,
        config=config,
        structure_config=structure_config,
        reference_cache_path=getattr(args, "reference_structure_cache", None),
        output_path=args.output,
    )
    print(f"structure_v3: {args.output}")
    return 0


def _load_exact_indexed_frames(directory: str | Path, expected: set[int]) -> dict[int, Any]:
    from PIL import Image
    from multimem_bench.vision.frame_source import list_frame_paths

    indexed: dict[int, Path] = {}
    for path in list_frame_paths(directory):
        try:
            index = int(path.stem)
        except ValueError as exc:
            raise ValueError(f"structure frame requires a numeric filename: {path.name}") from exc
        if index in indexed:
            raise ValueError(f"duplicate structure frame index {index}")
        indexed[index] = path
    missing = sorted(expected - set(indexed))
    if missing:
        raise ValueError(f"structure frames missing observation indices: {missing}")
    return {index: Image.open(indexed[index]).convert("RGB") for index in sorted(expected)}


def _frame_indices_from_directory(directory: str | Path) -> list[int]:
    from multimem_bench.vision.frame_source import list_frame_paths

    paths = list_frame_paths(directory)
    indices: list[int] = []
    for fallback, path in enumerate(paths):
        try:
            indices.append(int(path.stem))
        except ValueError:
            indices.append(fallback)
    return indices


def _run_eval_v3(args: argparse.Namespace) -> int:
    from multimem_bench.v3.evaluator import evaluate_v3, write_v3_result
    from multimem_bench.v3.geometry import V3GeometryArtifact

    config = _load_v3_config(args.config)
    geometry = V3GeometryArtifact.load(args.geometry)
    _validate_v3_eval_contract(geometry, config)
    structure = None
    if getattr(args, "structure", None):
        from multimem_bench.v3.structure import V3StructureArtifact

        structure = V3StructureArtifact.load(args.structure)
    result = evaluate_v3(
        load_reference_scene(args.reference),
        load_video_observation(args.observations),
        geometry,
        config,
        structure=structure,
    )
    paths = write_v3_result(result, args.output)
    print(f"summary: {paths['summary']}")
    for key, value in result.metrics.items():
        print(f"{key}: {value}")
    return 0


def _validate_v3_eval_contract(geometry: Any, config: Any) -> None:
    if config.geometry_mode != "input_grounded":
        return
    metadata = geometry.metadata
    if (
        metadata.get("geometry_mode") != "input_grounded"
        or metadata.get("fixed_reference_geometry") is not True
        or not metadata.get("reference_identity")
        or not metadata.get("reference_fingerprint")
    ):
        raise ValueError("input-grounded evaluation requires current fixed-reference geometry")
    if metadata.get("config") != config.to_dict():
        raise ValueError("V3 geometry config does not match evaluation config")


def _load_observe_batch_jobs(path: str | Path) -> list[dict[str, Any]]:
    data = load_json(path)
    raw_jobs = data.get("jobs") if isinstance(data, dict) else data
    if not isinstance(raw_jobs, list):
        raise ValueError("observe-video-batch manifest must be a JSON list or an object with a jobs list")
    jobs: list[dict[str, Any]] = []
    for index, item in enumerate(raw_jobs):
        if not isinstance(item, dict):
            raise ValueError(f"observe-video-batch job {index} must be an object")
        if not item.get("video"):
            raise ValueError(f"observe-video-batch job {index} is missing video")
        if not item.get("output"):
            raise ValueError(f"observe-video-batch job {index} is missing output")
        jobs.append(item)
    return jobs


def _resolve_manifest_path(value: Any, base_dir: Path) -> str | None:
    if value is None:
        return None
    path = Path(str(value))
    return str(path if path.is_absolute() else base_dir / path)


def _prompts_from_job(
    job: dict[str, Any],
    default_prompts: list[str] | None,
    manifest_dir: Path,
) -> list[str] | None:
    values: list[str] = []
    if job.get("prompts") is not None:
        raw = job["prompts"]
        if isinstance(raw, str):
            values.extend(part.strip() for part in raw.split(",") if part.strip())
        else:
            values.extend(str(item).strip() for item in raw if str(item).strip())
    if job.get("prompt_file"):
        loaded = _load_prompts(None, _resolve_manifest_path(job["prompt_file"], manifest_dir))
        if loaded:
            values.extend(loaded)
    return values or default_prompts


def _run_observe_video_batch(args: argparse.Namespace) -> int:
    from multimem_bench.vision.embedding import build_embedding_extractor
    from multimem_bench.vision.sam3_adapter import Sam3ImageSegmenter, Sam3VideoSegmenter
    from multimem_bench.vision.video_observer import (
        build_video_observation_artifact,
        resolve_reference_embedding_config,
    )

    manifest_path = Path(args.manifest)
    manifest_dir = manifest_path.parent
    jobs = _load_observe_batch_jobs(manifest_path)
    config = _load_vision_config(args.vision_config, strict_models=args.strict_models)
    default_prompts = _load_prompts(args.prompts, args.prompt_file)

    runnable_jobs = [
        job for job in jobs
        if not (args.skip_existing and Path(_resolve_manifest_path(job["output"], manifest_dir) or "").is_file())
    ]
    if runnable_jobs:
        first_reference = _resolve_manifest_path(
            runnable_jobs[0].get("reference") or args.reference,
            manifest_dir,
        )
        if first_reference is None:
            raise ValueError("observe-video-batch job 0 is missing reference")
        config = resolve_reference_embedding_config(
            config,
            load_reference_scene(first_reference),
        )
    embedder = build_embedding_extractor(config) if runnable_jobs else None
    needs_segmenter = (
        config.video_segmentation_mode == "tracking"
        and any(not job.get("segments") for job in runnable_jobs)
    )
    video_segmenter = Sam3VideoSegmenter(config) if needs_segmenter else None
    needs_image_segmenter = any(
        not job.get("segments")
        and (
            config.video_segmentation_mode == "framewise"
            or config.video_empty_frame_image_pcs_fallback
        )
        for job in runnable_jobs
    )
    image_segmenter = Sam3ImageSegmenter(config) if needs_image_segmenter else None

    for index, job in enumerate(jobs, start=1):
        output = _resolve_manifest_path(job["output"], manifest_dir)
        if output is None:
            raise ValueError(f"observe-video-batch job {index - 1} is missing output")
        if args.skip_existing and Path(output).is_file():
            print(f"[{index}/{len(jobs)}] skipped existing: {output}")
            continue

        reference = _resolve_manifest_path(job.get("reference") or args.reference, manifest_dir)
        if reference is None:
            raise ValueError(f"observe-video-batch job {index - 1} is missing reference")
        video = _resolve_manifest_path(job["video"], manifest_dir)
        frames = _resolve_manifest_path(job.get("frames"), manifest_dir)
        segments = _resolve_manifest_path(job.get("segments"), manifest_dir)
        assets_dir = _resolve_manifest_path(job.get("assets_dir"), manifest_dir)
        prompts = _prompts_from_job(job, default_prompts, manifest_dir)

        result = build_video_observation_artifact(
            video_path=video,
            reference_path=reference,
            output_path=output,
            config=config,
            video_id=job.get("video_id"),
            prompts=prompts,
            precomputed_segments_path=segments,
            frames_path=frames,
            assets_dir=assets_dir,
            video_segmenter=video_segmenter,
            image_segmenter=image_segmenter,
            embedder=embedder,
        )
        print(f"[{index}/{len(jobs)}] video_observation: {result.observation_path}")
        print(f"[{index}/{len(jobs)}] num_segments: {result.num_segments}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "eval":
        config = _load_config(args.config)
        return _run_eval(args.reference, args.observations, args.output, config)
    if args.command == "demo":
        # The bundled JSON example predates mask assets. Real benchmark evals
        # default to strict mask mode; the demo deliberately exercises the
        # documented legacy bbox fallback.
        config = (
            _load_config(args.config)
            if args.config
            else EvaluationConfig(require_masks=False)
        )
        root = Path(__file__).resolve().parents[1]
        return _run_eval(
            str(root / "examples" / "reference_scene.json"),
            str(root / "examples" / "video_observation.json"),
            args.output,
            config,
        )
    if args.command == "build-reference":
        return _run_build_reference(args)
    if args.command == "observe-video":
        return _run_observe_video(args)
    if args.command == "observe-video-batch":
        return _run_observe_video_batch(args)
    if args.command == "prepare-geometry":
        return _run_prepare_geometry(args)
    if args.command == "eval-v2":
        return _run_eval_v2(args)
    if args.command == "prepare-geometry-v3":
        return _run_prepare_geometry_v3(args)
    if args.command == "prepare-structure-v3":
        return _run_prepare_structure_v3(args)
    if args.command == "prepare-reference-v3":
        return _run_prepare_reference_v3(args)
    if args.command == "eval-v3":
        return _run_eval_v3(args)
    if args.command == "run-benchmark":
        from multimem_bench.workflow.orchestrator import run_benchmark_command

        return run_benchmark_command(args)
    if args.command == "run-benchmark_v3":
        from multimem_bench.workflow.orchestrator import run_benchmark_v3_command

        return run_benchmark_v3_command(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
