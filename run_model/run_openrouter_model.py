#!/usr/bin/env python3
"""Generate image-to-video clips through OpenRouter's asynchronous video API."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import mimetypes
import os
from pathlib import Path
import re
import sys
from urllib.parse import urlparse


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent / "multiMemBench"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from multimem_bench.workflow.backends.openrouter import (  # noqa: E402
    AuthorizationSafeRedirectHandler,
    OpenRouterError,
    OpenRouterVideoClient,
    build_payload,
    compatibility_error,
    output_geometry,
    payload_geometry,
    safe_url_label,
    same_origin,
)

__all__ = [
    "AuthorizationSafeRedirectHandler",
    "OpenRouterError",
    "OpenRouterVideoClient",
    "build_payload",
    "compatibility_error",
    "output_geometry",
    "payload_geometry",
    "safe_url_label",
    "same_origin",
]


_REQUESTED_MODELS = (
    "google/veo-3.1",
    "bytedance/seedance-2.5",
    "black-forest-labs/flux-3-video",
    "minimax/hailuo-3",
    "runway/gen-4.5",
    "x-ai/grok-imagine-video-1.5",
    "alibaba/happyhorse-1.1",
    "kwaivgi/kling-v3.0-pro",
    "kwaivgi/kling-v3.0-std",
    "alibaba/wan-2.7",
    "google/veo-3.1",
    "openai/sora-2-pro",
)
REQUESTED_MODELS = tuple(dict.fromkeys(_REQUESTED_MODELS))

DEFAULT_MODEL = "bytedance/seedance-2.5"
DEFAULT_DURATION = 5
MODEL_DURATION_OVERRIDES = {"google/veo-3.1": 4}
DEFAULT_SIZE = "1280x720"
DEFAULT_IMAGE = SCRIPT_DIR / "test2.jpg"
DEFAULT_PROMPT_FILE = SCRIPT_DIR / "input" / "text_prompt.txt"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "openrouter"


def duration_for_model(model_id: str) -> int:
    return MODEL_DURATION_OVERRIDES.get(model_id, DEFAULT_DURATION)


def image_to_data_url(path: str | Path) -> str:
    image_path = Path(path).expanduser().resolve()
    if not image_path.is_file():
        raise ValueError(f"input image is not a file: {image_path}")
    media_type = mimetypes.guess_type(image_path.name)[0]
    if media_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise ValueError(f"unsupported input image type: {media_type or 'unknown'}")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate OpenRouter videos from an image and a text prompt."
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--model", default=DEFAULT_MODEL)
    selection.add_argument("--all", dest="run_all", action="store_true")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--image-url")
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--size", default=DEFAULT_SIZE)
    parser.add_argument("--request-timeout", type=float, default=60)
    parser.add_argument("--poll-interval", type=float, default=30)
    parser.add_argument("--generation-timeout", type=float, default=3600)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def model_output_path(
    output_dir: Path,
    model_id: str,
    *,
    duration: int,
    run_id: str,
) -> Path:
    safe_model_id = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")
    return output_dir / f"{safe_model_id}_{duration}s_{run_id}.mp4"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.run_all and args.output is not None:
            raise ValueError("--output cannot be combined with --all; use --output-dir")
        if args.request_timeout <= 0 or args.generation_timeout <= 0:
            raise ValueError("timeouts must be positive")
        if args.poll_interval < 0:
            raise ValueError("--poll-interval cannot be negative")

        if args.prompt is not None:
            prompt = args.prompt.strip()
        else:
            prompt_path = args.prompt_file.expanduser().resolve()
            prompt = prompt_path.read_text(encoding="utf-8").strip()
        if not prompt:
            raise ValueError("prompt cannot be empty")

        image_override = args.image_url or os.environ.get("OPENROUTER_IMAGE_URL")
        if image_override:
            parsed = urlparse(image_override)
            if parsed.scheme != "https" or not parsed.netloc:
                raise ValueError("the image URL must be a public HTTPS URL")
            image_url = image_override
            image_label = safe_url_label(image_override)
        else:
            if not args.dry_run:
                raise ValueError(
                    "paid video runs require a public HTTPS URL; set --image-url "
                    "or OPENROUTER_IMAGE_URL"
                )
            image_url = image_to_data_url(args.image)
            image_label = str(args.image.expanduser().resolve())
            print(
                "Warning: using a local data URL for dry-run capability checks.",
                file=sys.stderr,
            )

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not args.dry_run and not api_key:
            raise OpenRouterError("OPENROUTER_API_KEY is not set")
        client = OpenRouterVideoClient(
            api_key=api_key,
            request_timeout=args.request_timeout,
            poll_interval=args.poll_interval,
            generation_timeout=args.generation_timeout,
        )
        live_models = {model.get("id"): model for model in client.list_models()}
        selected_models = REQUESTED_MODELS if args.run_all else (args.model,)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        failures = 0
        eligible_jobs = 0
        for model_id in selected_models:
            model = live_models.get(model_id)
            if not isinstance(model, dict):
                print(f"[SKIP] {model_id}: not present in OpenRouter /videos/models")
                continue
            duration = duration_for_model(model_id)
            reason = compatibility_error(model, duration=duration, size=args.size)
            if reason:
                print(f"[SKIP] {model_id}: {reason}")
                continue
            payload = build_payload(
                model=model,
                prompt=prompt,
                duration=duration,
                size=args.size,
                image_url=image_url,
            )
            output = (
                args.output.expanduser().resolve()
                if args.output is not None
                else model_output_path(
                    args.output_dir.expanduser().resolve(),
                    model_id,
                    duration=duration,
                    run_id=run_id,
                )
            )
            print(
                f"[PLAN] {model_id}: duration={duration}s, "
                f"size={payload_geometry(payload)}, image={image_label}, output={output}"
            )
            eligible_jobs += 1
            if args.dry_run:
                continue
            try:
                completed = client.generate(
                    payload,
                    output,
                    on_submitted=lambda job: print(
                        f"[SUBMITTED] {model_id}: id={job.get('id')}", flush=True
                    ),
                )
            except (OpenRouterError, OSError) as exc:
                failures += 1
                print(f"[FAIL] {model_id}: {exc}", file=sys.stderr)
                continue
            print(f"[DONE] {model_id}: output={output}, usage={completed.get('usage')}")
        return 1 if failures or eligible_jobs == 0 else 0
    except (OSError, ValueError, OpenRouterError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
