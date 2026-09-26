"""Adapter for the repository's local Wan2.2 runner."""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from ..schema import GenerationRequest, GenerationResult
from ..security import sanitize_text
from ..video_models import get_local_video_model_profile


VARIANT_FPS = {"i2v-a14b": 16, "ti2v-5b": 24}


def local_model_profile(model: str) -> dict[str, Any]:
    try:
        return get_local_video_model_profile(model)
    except ValueError as exc:
        raise ValueError(f"local Wan2.2 {exc}") from exc


def frame_count_for_duration(duration: float, fps: int) -> int:
    if duration <= 0 or fps <= 0:
        raise ValueError("duration and frame rate must be positive")
    intervals = max(4, round(duration * fps))
    intervals = max(4, round(intervals / 4) * 4)
    return intervals + 1


class Wan22Backend:
    def __init__(
        self,
        *,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.runner = runner

    def build_command(self, request: GenerationRequest) -> list[str]:
        if not request.prompt:
            raise ValueError("Wan2.2 generation request has no prompt")
        options = request.options
        profile = local_model_profile(request.model)
        configured_variant = str(profile.get("variant", ""))
        variant = str(options.get("variant", configured_variant))
        if variant != configured_variant:
            raise ValueError(
                f"local model profile variant mismatch: {request.model} uses {configured_variant}, got {variant}"
            )
        try:
            fps = int(profile["fps"])
            resolutions = [str(value) for value in profile["resolutions"]]
            requirements = {str(value) for value in profile.get("requires", [])}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid local model profile for {request.model}") from exc
        unknown_requirements = requirements - {
            "--offload_model", "--convert_model_dtype", "--t5_cpu"
        }
        if unknown_requirements:
            raise ValueError(
                f"unsupported local model requirements for {request.model}: "
                f"{sorted(unknown_requirements)}"
            )
        if request.size not in resolutions:
            raise ValueError(
                f"unsupported local resolution {request.size!r} for {request.model}; choose from {resolutions}"
            )
        frame_num_raw = options.get("frame_num")
        frame_num = (
            int(frame_num_raw)
            if frame_num_raw is not None
            else frame_count_for_duration(request.duration, fps)
        )
        if frame_num < 1 or (frame_num - 1) % 4:
            raise ValueError("Wan2.2 frame count must satisfy 4n+1")
        runner_path = Path(
            options.get(
                "runner_path",
                Path(__file__).resolve().parents[4]
                / "run_model"
                / "run_wan22_i2v.py",
            )
        ).expanduser().resolve()
        command = [
            sys.executable,
            str(runner_path),
            "--image",
            str(Path(request.image_path).expanduser().resolve()),
            "--prompt",
            request.prompt,
            "--output",
            str(Path(request.output_path).expanduser().resolve()),
            "--variant",
            variant,
            "--size",
            request.size.replace("x", "*"),
            "--seed",
            str(request.seed),
            "--frame-num",
            str(frame_num),
            "--timeout",
            str(int(options.get("timeout", 14400))),
        ]
        if "--offload_model" not in requirements:
            command.append("--no-offload")
        if "--convert_model_dtype" not in requirements:
            command.append("--no-convert-dtype")
        if "--t5_cpu" not in requirements:
            command.append("--no-t5-cpu")
        if options.get("wan_repo"):
            command.extend(["--wan-repo", str(options["wan_repo"])])
        if options.get("ckpt_dir"):
            command.extend(["--ckpt-dir", str(options["ckpt_dir"])])
        if options.get("local_conda_env"):
            command.extend(["--conda-env", str(options["local_conda_env"])])
        local_python = Path(__file__).resolve().parents[4] / ".venvs-video" / "wan22" / "bin" / "python"
        if local_python.is_file():
            command.extend(["--python-path", str(local_python)])
        if options.get("no_conda"):
            command.append("--no-conda")
        if options.get("cuda_visible_devices"):
            command.extend(
                ["--cuda-visible-devices", str(options["cuda_visible_devices"])]
            )
        for option_name, cli_name in (
            ("sample_steps", "--sample-steps"),
            ("sample_guide_scale", "--sample-guide-scale"),
            ("sample_shift", "--sample-shift"),
        ):
            if options.get(option_name) is not None:
                command.extend([cli_name, str(options[option_name])])
        return command

    def generate(
        self,
        request: GenerationRequest,
        *,
        resume_state: dict[str, Any] | None = None,
        on_state: Callable[[dict[str, Any]], None] | None = None,
    ) -> GenerationResult:
        del resume_state
        command = self.build_command(request)
        timeout = float(request.options.get("timeout", 14400)) + 60
        env = os.environ.copy()
        secrets = []
        for key in (
            "OPENROUTER_API_KEY",
            "ARK_API_KEY",
            "FAL_KEY",
            "VIDEO_API_KEY",
            "VLM_API_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
        ):
            value = env.pop(key, None)
            if value:
                secrets.append(value)
        completed = self.runner(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            env=env,
        )
        log_path_raw = request.options.get("log_path")
        if log_path_raw:
            log_path = Path(str(log_path_raw))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(sanitize_text(completed.stdout or "", secrets), encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"Wan2.2 runner failed with exit code {completed.returncode}"
            )
        output = Path(request.output_path).expanduser().resolve()
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"Wan2.2 did not create a nonempty video: {output}")
        if on_state is not None:
            on_state({"status": "completed"})
        profile = local_model_profile(request.model)
        variant = str(request.options.get("variant", profile["variant"]))
        frame_num = request.options.get("frame_num")
        if frame_num is None:
            frame_num = frame_count_for_duration(
                request.duration, int(profile["fps"])
            )
        return GenerationResult(
            video_path=str(output),
            model_id=request.model,
            effective_parameters={
                "variant": variant,
                "duration": request.duration,
                "size": request.size,
                "seed": request.seed,
                "frame_num": int(frame_num),
                "fps": int(profile["fps"]),
            },
        )
