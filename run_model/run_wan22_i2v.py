#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WAN_REPO = PROJECT_ROOT / "models" / "video_generation" / "Wan2.2"
VARIANTS = {
    "i2v-a14b": {
        "task": "i2v-A14B",
        "size": "1280*720",
        "ckpt_dir": PROJECT_ROOT / "models" / "video_generation" / "checkpoints" / "Wan2.2-I2V-A14B",
        "ckpt_env": "WAN22_I2V_A14B_DIR",
    },
    "ti2v-5b": {
        "task": "ti2v-5B",
        "size": "1280*704",
        "ckpt_dir": PROJECT_ROOT / "models" / "video_generation" / "checkpoints" / "Wan2.2-TI2V-5B",
        "ckpt_env": "WAN22_TI2V_5B_DIR",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local Wan2.2 image-to-video generation from one image and one prompt."
    )
    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Input image path.",
    )
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--prompt",
        help="Text prompt.",
    )
    prompt_group.add_argument(
        "--prompt-file",
        type=Path,
        help="UTF-8 text file containing the prompt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output MP4 path. Defaults to outputs/wan22_<variant>_<timestamp>_seed<seed>.mp4.",
    )
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANTS),
        default="i2v-a14b",
        help="Wan2.2 variant. i2v-a14b is the official Image-to-Video model; ti2v-5b is a smaller text-image-to-video model that also accepts an image.",
    )
    parser.add_argument(
        "--wan-repo",
        type=Path,
        default=Path(os.environ.get("WAN22_REPO", DEFAULT_WAN_REPO)),
        help=f"Wan2.2 repository path. Default: {DEFAULT_WAN_REPO}",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        help="Checkpoint directory. Defaults to the variant's WAN22_* env var, then the local model path.",
    )
    parser.add_argument(
        "--size",
        help="Wan2.2 size string, for example 1280*720. Defaults to the variant's local deployment size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed. Use -1 to let Wan2.2 choose a random seed.",
    )
    parser.add_argument(
        "--frame-num",
        type=int,
        help="Number of frames. Wan2.2 expects 4n+1 when set.",
    )
    parser.add_argument(
        "--sample-steps",
        type=int,
        help="Override Wan2.2 sampling steps.",
    )
    parser.add_argument(
        "--sample-guide-scale",
        type=float,
        help="Override Wan2.2 classifier-free guidance scale.",
    )
    parser.add_argument(
        "--sample-shift",
        type=float,
        help="Override Wan2.2 sampling shift.",
    )
    parser.add_argument(
        "--conda-env",
        default="wan22",
        help="Conda environment used for Wan2.2 inference.",
    )
    parser.add_argument(
        "--no-conda",
        action="store_true",
        help="Run generate.py with the current Python instead of conda run.",
    )
    parser.add_argument(
        "--python-path",
        type=Path,
        help="Run generate.py with this isolated Python interpreter.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        help="Optional CUDA_VISIBLE_DEVICES value for this run, for example 0.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=14400,
        help="Timeout in seconds.",
    )
    parser.add_argument(
        "--no-offload",
        action="store_true",
        help="Do not pass --offload_model True.",
    )
    parser.add_argument(
        "--no-convert-dtype",
        action="store_true",
        help="Do not pass --convert_model_dtype.",
    )
    parser.add_argument(
        "--no-t5-cpu",
        action="store_true",
        help="Do not pass --t5_cpu.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without running inference.",
    )
    return parser.parse_args()


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return args.prompt
    try:
        return args.prompt_file.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise SystemExit(f"Failed to read prompt file: {args.prompt_file}: {exc}") from exc


def resolve_ckpt_dir(args: argparse.Namespace) -> Path:
    if args.ckpt_dir:
        return args.ckpt_dir.expanduser().resolve()
    variant = VARIANTS[args.variant]
    env_value = os.environ.get(variant["ckpt_env"])
    if env_value:
        return Path(env_value).expanduser().resolve()
    return variant["ckpt_dir"].resolve()


def resolve_output(args: argparse.Namespace) -> Path:
    if args.output:
        return args.output.expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_label = "random" if args.seed < 0 else str(args.seed)
    filename = f"wan22_{args.variant}_{timestamp}_seed{seed_label}.mp4"
    return (Path.cwd() / "outputs" / filename).resolve()


def require_path(path: Path, label: str, must_be_file: bool | None = None) -> None:
    if not path.exists():
        raise SystemExit(f"{label} does not exist: {path}")
    if must_be_file is True and not path.is_file():
        raise SystemExit(f"{label} is not a file: {path}")
    if must_be_file is False and not path.is_dir():
        raise SystemExit(f"{label} is not a directory: {path}")


def build_command(
    args: argparse.Namespace,
    prompt: str,
    ckpt_dir: Path,
    output: Path,
) -> list[str]:
    variant = VARIANTS[args.variant]
    size = args.size or variant["size"]
    generate_py = args.wan_repo / "generate.py"

    if args.python_path:
        command = [str(args.python_path.expanduser().resolve()), str(generate_py)]
    elif args.no_conda:
        command = [sys.executable, str(generate_py)]
    else:
        command = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            args.conda_env,
            "python",
            "generate.py",
        ]

    command.extend(
        [
            "--task",
            variant["task"],
            "--size",
            size,
            "--ckpt_dir",
            str(ckpt_dir),
            "--prompt",
            prompt,
            "--image",
            str(args.image.expanduser().resolve()),
            "--save_file",
            str(output),
            "--base_seed",
            str(args.seed),
        ]
    )

    if not args.no_offload:
        command.extend(["--offload_model", "True"])
    if not args.no_convert_dtype:
        command.append("--convert_model_dtype")
    if not args.no_t5_cpu:
        command.append("--t5_cpu")
    if args.frame_num is not None:
        command.extend(["--frame_num", str(args.frame_num)])
    if args.sample_steps is not None:
        command.extend(["--sample_steps", str(args.sample_steps)])
    if args.sample_guide_scale is not None:
        command.extend(["--sample_guide_scale", str(args.sample_guide_scale)])
    if args.sample_shift is not None:
        command.extend(["--sample_shift", str(args.sample_shift)])

    return command


def main() -> int:
    args = parse_args()
    args.image = args.image.expanduser().resolve()
    args.wan_repo = args.wan_repo.expanduser().resolve()
    prompt = read_prompt(args)
    if not prompt:
        raise SystemExit("Prompt is empty.")

    ckpt_dir = resolve_ckpt_dir(args)
    output = resolve_output(args)

    require_path(args.image, "Input image", must_be_file=True)
    require_path(args.wan_repo, "Wan2.2 repository", must_be_file=False)
    require_path(args.wan_repo / "generate.py", "Wan2.2 generate.py", must_be_file=True)
    require_path(ckpt_dir, "Checkpoint directory", must_be_file=False)
    if args.python_path:
        require_path(args.python_path.expanduser().resolve(), "Python interpreter", must_be_file=True)

    command = build_command(args, prompt=prompt, ckpt_dir=ckpt_dir, output=output)
    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    print("Wan2.2 command:")
    print(shlex.join(command))
    print(f"cwd: {args.wan_repo}")
    print(f"output: {output}")
    if args.dry_run:
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            command,
            cwd=args.wan_repo,
            env=env,
            text=True,
            timeout=args.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(f"Wan2.2 inference timed out after {args.timeout}s.") from exc

    if completed.returncode != 0:
        return completed.returncode
    if not output.exists():
        raise SystemExit(f"Wan2.2 exited successfully but output file was not found: {output}")
    print(f"Generated video: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
