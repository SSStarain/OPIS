#!/usr/bin/env python3
"""Run one of the repository's isolated local I2V deployments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MODEL_LAYOUT = {
    "cogvideox/5b-i2v": {
        "python": ROOT / ".venvs-video/cogvideox/bin/python",
        "repo": ROOT / "models/video_generation/CogVideo",
        "checkpoint": ROOT / "models/video_generation/checkpoints/CogVideoX-5b-I2V",
    },
    "hunyuanvideo/1.5-i2v": {
        "python": ROOT / ".venvs-video/hunyuan-1.5/bin/python",
        "repo": ROOT / "models/video_generation/HunyuanVideo-1.5",
        "checkpoint": ROOT / "models/video_generation/HunyuanVideo-1.5/ckpts",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_LAYOUT))
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--size", required=True)
    parser.add_argument("--duration", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--frame-num", type=int)
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--timeout", type=int, default=14400)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def dimensions(value: str) -> tuple[int, int]:
    try:
        width, height = (int(item) for item in value.lower().split("x", 1))
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"invalid size {value!r}; expected WIDTHxHEIGHT") from exc
    if width <= 0 or height <= 0:
        raise SystemExit("video dimensions must be positive")
    return width, height


def aligned_frames(duration: float, fps: int, alignment: int) -> int:
    intervals = max(alignment, round(duration * fps / alignment) * alignment)
    return intervals + 1


def build_command(args: argparse.Namespace, work: Path) -> tuple[list[str], Path]:
    layout = MODEL_LAYOUT[args.model]
    python = str(layout["python"])
    repo = Path(layout["repo"])
    checkpoint = Path(layout["checkpoint"])
    width, height = dimensions(args.size)
    if args.model == "cogvideox/5b-i2v":
        frames = args.frame_num or aligned_frames(args.duration, 5, 8)
        if frames > 49 or (frames - 1) % 8:
            raise SystemExit(
                "CogVideoX-5B-I2V requires 8N+1 frames with a maximum of 49"
            )
        script = ROOT / "run_model/cogvideox_i2v.py"
        return ([
            python, str(script), "--model-path", str(checkpoint),
            "--image", str(args.image), "--prompt", args.prompt,
            "--output", str(args.output), "--width", str(width),
            "--height", str(height), "--num-frames", str(frames),
            "--fps", "5", "--seed", str(args.seed),
        ], args.output)
    frames = args.frame_num or aligned_frames(args.duration, 24, 4)
    return ([
        python, str(repo / "generate.py"), "--prompt", args.prompt,
        "--image_path", str(args.image), "--resolution", "480p",
        "--aspect_ratio", "16:9", "--video_length", str(frames),
        "--seed", str(args.seed), "--rewrite", "false", "--sr", "false",
        "--offloading", "true", "--overlap_group_offloading", "true",
        "--output_path", str(args.output), "--model_path", str(checkpoint),
    ], args.output)


def validate_layout(model: str) -> None:
    for label, path in MODEL_LAYOUT[model].items():
        path = Path(path)
        if label == "python" and not path.is_file():
            raise SystemExit(f"local model Python is missing: {path}")
        if label != "python" and not path.exists():
            raise SystemExit(f"local model {label} is missing: {path}")


def main() -> int:
    args = parse_args()
    args.image = args.image.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.image.is_file():
        raise SystemExit(f"input image does not exist: {args.image}")
    if not args.prompt.strip():
        raise SystemExit("prompt is empty")
    validate_layout(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="multimem-i2v-") as temporary:
        work = Path(temporary)
        command, produced = build_command(args, work)
        print(f"Local I2V command: {shlex.join(command)}", flush=True)
        if args.dry_run:
            return 0
        env = os.environ.copy()
        if args.cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        env["HF_HUB_OFFLINE"] = "1"
        completed = subprocess.run(
            command, cwd=MODEL_LAYOUT[args.model]["repo"], env=env,
            timeout=args.timeout, check=False,
        )
        if completed.returncode:
            return completed.returncode
        if produced.is_dir():
            candidates = sorted(produced.glob("*.mp4"), key=lambda path: path.stat().st_mtime)
            if not candidates:
                raise SystemExit(f"{args.model} did not create an MP4 in {produced}")
            shutil.move(str(candidates[-1]), args.output)
        if not args.output.is_file() or args.output.stat().st_size == 0:
            raise SystemExit(f"{args.model} did not create {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
