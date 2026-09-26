"""Adapter for locally deployed image-to-video models."""

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


class LocalI2VBackend:
    def __init__(
        self,
        *,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.runner = runner

    def build_command(self, request: GenerationRequest) -> list[str]:
        if not request.prompt:
            raise ValueError("local I2V generation request has no prompt")
        profile = get_local_video_model_profile(request.model)
        if profile.get("backend") != "local_i2v":
            raise ValueError(f"video model is not handled by local_i2v: {request.model}")
        resolutions = [str(value) for value in profile.get("resolutions", [])]
        if request.size not in resolutions:
            raise ValueError(
                f"unsupported local resolution {request.size!r} for {request.model}; "
                f"choose from {resolutions}"
            )
        root = Path(__file__).resolve().parents[4]
        command = [
            sys.executable,
            str(root / "run_model" / "run_local_i2v.py"),
            "--model",
            request.model,
            "--image",
            str(Path(request.image_path).expanduser().resolve()),
            "--prompt",
            request.prompt,
            "--output",
            str(Path(request.output_path).expanduser().resolve()),
            "--size",
            request.size,
            "--duration",
            str(request.duration),
            "--seed",
            str(request.seed),
            "--timeout",
            str(int(request.options.get("timeout", 14400))),
        ]
        if request.options.get("cuda_visible_devices"):
            command.extend(
                ["--cuda-visible-devices", str(request.options["cuda_visible_devices"])]
            )
        if request.options.get("frame_num") is not None:
            command.extend(["--frame-num", str(request.options["frame_num"])])
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
        secrets: list[str] = []
        for key in (
            "OPENROUTER_API_KEY", "ARK_API_KEY", "FAL_KEY", "VIDEO_API_KEY",
            "VLM_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN", "HF_TOKEN",
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
            log_path.write_text(
                sanitize_text(completed.stdout or "", secrets), encoding="utf-8"
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"local I2V runner failed with exit code {completed.returncode}"
            )
        output = Path(request.output_path).expanduser().resolve()
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"local I2V runner did not create a video: {output}")
        if on_state is not None:
            on_state({"status": "completed"})
        profile = get_local_video_model_profile(request.model)
        fps = int(profile["fps"])
        frame_num = request.options.get("frame_num")
        if frame_num is None:
            alignment = int(profile.get("frame_alignment", 1))
            intervals = max(
                alignment,
                round(float(request.duration) * fps / alignment) * alignment,
            )
            frame_num = intervals + 1
        effective_parameters = {
            "duration": request.duration,
            "size": request.size,
            "seed": request.seed,
            "frame_num": int(frame_num),
            "fps": fps,
        }
        for name in (
            "native_resolution", "native_fps", "spatial_adapter", "temporal_adapter"
        ):
            if name in profile:
                effective_parameters[name] = profile[name]
        return GenerationResult(
            video_path=str(output),
            model_id=request.model,
            effective_parameters=effective_parameters,
        )
