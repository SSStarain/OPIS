"""Adapter for repository-local camera-conditioned world models."""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from ..schema import GenerationRequest, GenerationResult
from ..security import sanitize_text
from ..video_models import get_local_video_model_profile


MODEL_ADAPTERS = {
    "sana-wm": "sana",
    "lingbot-world-2.0": "lingbot",
    "matrix-game-3.5": "matrix35",
    "echo-wm-flash": "echo",
}


class LocalWorldModelBackend:
    """Run a supported camera-conditioned model through its native adapter."""

    def __init__(
        self,
        *,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.runner = runner

    def build_commands(self, request: GenerationRequest) -> tuple[list[str], list[str], Path]:
        if not request.prompt:
            raise ValueError("local world-model generation request has no prompt")
        profile = get_local_video_model_profile(request.model)
        if profile.get("backend") != "local-wm":
            raise ValueError(f"video model is not handled by local-wm: {request.model}")
        adapter = MODEL_ADAPTERS.get(request.model)
        if adapter is None:
            raise ValueError(f"unsupported local world model: {request.model}")

        options = request.options
        case_id = str(options.get("case_id", "")).strip("/")
        if not case_id:
            raise ValueError("local-wm requires a dataset-relative case_id")
        root = Path(__file__).resolve().parents[4]
        prepared_root = Path(
            str(options.get("prepared_root") or root / "dataset/world_model_native")
        ).expanduser().resolve()
        result_root = Path(
            str(options.get("result_root") or root / "eval_result/world_model_10s_v1")
        ).expanduser().resolve()
        world_model_root = Path(
            str(options.get("world_model_root") or root / "dataset/world_model_10s_v1")
        ).expanduser().resolve()
        final_root = Path(
            str(options.get("final_root") or root / "dataset/final")
        ).expanduser().resolve()

        common = [
            sys.executable,
            "-m",
            "tools.dataset_clean_tools.run_world_model_50",
        ]
        prepare = common + [
            "prepare",
            "--model",
            adapter,
            "--case-id",
            case_id,
            "--final-root",
            str(final_root),
            "--world-model-root",
            str(world_model_root),
            "--prepared-root",
            str(prepared_root),
            "--result-root",
            str(result_root),
        ]
        gpus = str(options.get("cuda_visible_devices") or "3,4,5,6")
        intrinsics_gpu = str(options.get("intrinsics_gpu") or gpus.split(",")[0])
        if adapter in ("lingbot", "matrix35"):
            prepare.extend(["--intrinsics-gpu", intrinsics_gpu])
        run = common + [
            "run",
            "--model",
            adapter,
            "--case-id",
            case_id,
            "--final-root",
            str(final_root),
            "--world-model-root",
            str(world_model_root),
            "--prepared-root",
            str(prepared_root),
            "--result-root",
            str(result_root),
            "--gpus",
            gpus,
        ]
        return prepare, run, result_root / adapter / case_id / "video.mp4"

    def generate(
        self,
        request: GenerationRequest,
        *,
        resume_state: dict[str, Any] | None = None,
        on_state: Callable[[dict[str, Any]], None] | None = None,
    ) -> GenerationResult:
        del resume_state
        prepare, run, native_output = self.build_commands(request)
        timeout = float(request.options.get("timeout", 14400)) + 60
        root = Path(__file__).resolve().parents[4]
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

        outputs: list[str] = []
        for name, command in (("prepare", prepare), ("run", run)):
            completed = self.runner(
                command,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                env=env,
                cwd=root,
            )
            outputs.append(f"[{name}]\n{completed.stdout or ''}")
            if completed.returncode != 0:
                self._write_log(request, "\n".join(outputs), secrets)
                raise RuntimeError(
                    f"{request.model} {name} failed with exit code {completed.returncode}"
                )

        self._write_log(request, "\n".join(outputs), secrets)
        if not native_output.is_file() or native_output.stat().st_size == 0:
            raise RuntimeError(f"local world model did not create a video: {native_output}")
        output = Path(request.output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if native_output.resolve() != output:
            shutil.copy2(native_output, output)
        if on_state is not None:
            on_state({"status": "completed"})
        profile = get_local_video_model_profile(request.model)
        return GenerationResult(
            video_path=str(output),
            model_id=request.model,
            effective_parameters={
                "duration": request.duration,
                "size": request.size,
                "fps": int(profile["fps"]),
                "request_frames": int(profile["request_frames"]),
                "output_frames": int(profile["output_frames"]),
                "case_id": str(request.options["case_id"]),
                "task_sha256": str(request.options["task_sha256"]),
            },
        )

    @staticmethod
    def _write_log(
        request: GenerationRequest, content: str, secrets: list[str]
    ) -> None:
        raw = request.options.get("log_path")
        if not raw:
            return
        path = Path(str(raw))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(sanitize_text(content, secrets), encoding="utf-8")
