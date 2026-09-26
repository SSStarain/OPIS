"""OpenRouter asynchronous image-to-video backend."""

from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .base import SubmissionUnknownError
from ..schema import GenerationRequest, GenerationResult


DEFAULT_API_BASE_URL = "https://openrouter.ai/api/v1"
FIRST_FRAME_RATIO_FROM_IMAGE_MODELS = frozenset({"bytedance/seedance-2.5"})


class OpenRouterError(RuntimeError):
    """An OpenRouter request or video job failed."""


def same_origin(first_url: str, second_url: str) -> bool:
    def origin(url: str) -> tuple[str, str | None, int | None]:
        parsed = urlparse(url)
        default_port = 443 if parsed.scheme == "https" else 80
        return parsed.scheme.lower(), parsed.hostname, parsed.port or default_port

    return origin(first_url) == origin(second_url)


class AuthorizationSafeRedirectHandler(HTTPRedirectHandler):
    """Keep authorization only when an HTTP redirect stays on origin."""

    def redirect_request(
        self,
        request: Request,
        fp: Any,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> Request | None:
        redirected = super().redirect_request(
            request, fp, code, message, headers, new_url
        )
        if redirected is not None and not same_origin(request.full_url, new_url):
            redirected.remove_header("Authorization")
        return redirected


class OpenRouterVideoClient:
    """Small standard-library client for OpenRouter's asynchronous video API."""

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str = DEFAULT_API_BASE_URL,
        request_timeout: float = 60,
        poll_interval: float = 30,
        generation_timeout: float = 3600,
        max_get_retries: int = 3,
        retry_backoff: float = 1,
    ) -> None:
        if request_timeout <= 0 or generation_timeout <= 0:
            raise ValueError("OpenRouter timeouts must be positive")
        if poll_interval < 0 or max_get_retries < 0 or retry_backoff < 0:
            raise ValueError("OpenRouter retry settings cannot be negative")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.request_timeout = request_timeout
        self.poll_interval = poll_interval
        self.generation_timeout = generation_timeout
        self.max_get_retries = max_get_retries
        self.retry_backoff = retry_backoff
        self._opener = build_opener(AuthorizationSafeRedirectHandler())

    def list_models(self) -> list[dict[str, Any]]:
        response = self._request_json("GET", f"{self.base_url}/videos/models")
        models = response.get("data")
        if not isinstance(models, list):
            raise OpenRouterError("OpenRouter model response has no data list")
        return [model for model in models if isinstance(model, dict)]

    def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise OpenRouterError("OPENROUTER_API_KEY is not set")
        return self._request_json(
            "POST",
            f"{self.base_url}/videos",
            payload=payload,
            require_auth=True,
        )

    def wait_for_completion(self, job: dict[str, Any]) -> dict[str, Any]:
        current = job
        deadline = time.monotonic() + self.generation_timeout
        while True:
            status = current.get("status")
            if status == "completed":
                return current
            if status in {"failed", "cancelled", "expired"}:
                detail = current.get("error") or f"video generation {status}"
                raise OpenRouterError(str(detail))
            if status not in {"pending", "in_progress"}:
                raise OpenRouterError(f"unexpected video job status: {status!r}")
            if time.monotonic() >= deadline:
                raise OpenRouterError(
                    f"video generation timed out after {self.generation_timeout:g}s"
                )
            polling_url = current.get("polling_url")
            if not isinstance(polling_url, str) or not polling_url:
                raise OpenRouterError("video job has no polling_url")
            if self.poll_interval:
                time.sleep(self.poll_interval)
            resolved = urljoin(f"{self.base_url}/", polling_url)
            if not same_origin(self.base_url, resolved):
                raise OpenRouterError("video job returned a cross-origin polling_url")
            current = self._request_json("GET", resolved, require_auth=True)

    def download(self, job: dict[str, Any], output_path: str | Path) -> Path:
        job_id = job.get("id")
        if not isinstance(job_id, str) or not job_id:
            raise OpenRouterError("completed video job has no id")
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.part")
        url = f"{self.base_url}/videos/{job_id}/content?index=0"
        try:
            for attempt in range(self.max_get_retries + 1):
                request = Request(
                    url, headers=self._headers(require_auth=True), method="GET"
                )
                try:
                    with self._opener.open(
                        request, timeout=self.request_timeout
                    ) as response:
                        content_type = response.headers.get_content_type()
                        if not content_type.startswith("video/"):
                            raise OpenRouterError(
                                "video download returned unexpected content type: "
                                f"{content_type}"
                            )
                        with temporary.open("wb") as destination:
                            while chunk := response.read(1024 * 1024):
                                destination.write(chunk)
                    break
                except HTTPError as exc:
                    retryable = exc.code in {429, 500, 502, 503, 504}
                    if retryable and attempt < self.max_get_retries:
                        exc.close()
                        self._sleep_before_retry(attempt)
                        continue
                    raise self._http_error(exc) from None
                except URLError as exc:
                    if attempt < self.max_get_retries:
                        self._sleep_before_retry(attempt)
                        continue
                    raise OpenRouterError(
                        f"OpenRouter download failed: {exc.reason}"
                    ) from None
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise OpenRouterError("OpenRouter returned an empty video")
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        return output

    def generate(
        self,
        payload: dict[str, Any],
        output_path: str | Path,
        *,
        on_submitted: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        job = self.submit(payload)
        if on_submitted is not None:
            on_submitted(job)
        completed = self.wait_for_completion(job)
        self.download(completed, output_path)
        return completed

    def _request_json(
        self,
        method: str,
        url: str,
        *,
        payload: dict[str, Any] | None = None,
        require_auth: bool = False,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = self._headers(require_auth=require_auth)
        headers["Accept"] = "application/json"
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = Request(url, data=data, headers=headers, method=method)
        attempts = self.max_get_retries + 1 if method == "GET" else 1
        value: Any = None
        for attempt in range(attempts):
            try:
                with self._opener.open(
                    request, timeout=self.request_timeout
                ) as response:
                    value = json.load(response)
                break
            except HTTPError as exc:
                retryable = exc.code in {429, 500, 502, 503, 504}
                if retryable and attempt + 1 < attempts:
                    exc.close()
                    self._sleep_before_retry(attempt)
                    continue
                raise self._http_error(exc) from None
            except URLError as exc:
                if attempt + 1 < attempts:
                    self._sleep_before_retry(attempt)
                    continue
                error_type = SubmissionUnknownError if method == "POST" else OpenRouterError
                raise error_type(f"OpenRouter request failed: {exc.reason}") from None
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                error_type = SubmissionUnknownError if method == "POST" else OpenRouterError
                raise error_type("OpenRouter returned invalid JSON") from exc
        if not isinstance(value, dict):
            error_type = SubmissionUnknownError if method == "POST" else OpenRouterError
            raise error_type("OpenRouter returned a non-object JSON response")
        return value

    def _sleep_before_retry(self, attempt: int) -> None:
        delay = self.retry_backoff * (2**attempt)
        if delay:
            time.sleep(delay)

    def _headers(self, *, require_auth: bool) -> dict[str, str]:
        if require_auth and not self.api_key:
            raise OpenRouterError("OPENROUTER_API_KEY is not set")
        headers = {"User-Agent": "multi-mem-bench-openrouter-i2v/2.0"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _http_error(exc: HTTPError) -> OpenRouterError:
        try:
            detail = exc.read(4096).decode("utf-8", errors="replace").strip()
        except OSError:
            detail = ""
        suffix = f": {detail}" if detail else ""
        return OpenRouterError(f"OpenRouter HTTP {exc.code}{suffix}")


def output_geometry(size: str) -> tuple[str, str]:
    try:
        width_text, height_text = size.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"invalid output size: {size!r}; expected WIDTHxHEIGHT") from exc
    if width < 1 or height < 1:
        raise ValueError("output dimensions must be positive")
    aspect_ratio = _supported_aspect_ratio(width, height)
    if aspect_ratio is None:
        raise ValueError(f"unsupported output aspect ratio for {size}")
    resolution = f"{min(width, height)}p"
    return resolution, aspect_ratio


def _supported_aspect_ratio(width: int, height: int) -> str | None:
    from math import gcd

    ratios = {
        (16, 9): "16:9",
        (9, 16): "9:16",
        (1, 1): "1:1",
    }
    divisor = gcd(width, height)
    exact = ratios.get((width // divisor, height // divisor))
    if exact is not None:
        return exact
    ratio = width / height
    for base, label in (((16, 9), "16:9"), ((9, 16), "9:16"), ((1, 1), "1:1")):
        if abs(ratio - (base[0] / base[1])) <= 0.015:
            return label
    return None


def compatibility_error(
    model: dict[str, Any], *, duration: int | float, size: str
) -> str | None:
    durations = model.get("supported_durations")
    if not isinstance(durations, list) or duration not in durations:
        return f"duration {duration:g}s is not supported (available: {durations})"
    resolution, aspect_ratio = output_geometry(size)
    sizes = model.get("supported_sizes")
    if not (isinstance(sizes, list) and size in sizes):
        resolutions = model.get("supported_resolutions")
        if not isinstance(resolutions, list) or resolution not in resolutions:
            return (
                f"resolution {resolution} for {size} is not supported "
                f"(available: {resolutions})"
            )
        aspect_ratios = model.get("supported_aspect_ratios")
        if not isinstance(aspect_ratios, list) or aspect_ratio not in aspect_ratios:
            return (
                f"aspect ratio {aspect_ratio} for {size} is not supported "
                f"(available: {aspect_ratios})"
            )
    frame_images = model.get("supported_frame_images")
    if not isinstance(frame_images, list) or "first_frame" not in frame_images:
        return "first_frame image input is not supported"
    return None


def build_payload(
    *,
    model: dict[str, Any],
    prompt: str,
    duration: int | float,
    size: str,
    image_url: str,
    seed: int | None = None,
) -> dict[str, Any]:
    model_id = model.get("id")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model capability record has no id")
    if not prompt.strip():
        raise ValueError("prompt is empty")
    parsed = urlparse(image_url)
    if parsed.scheme not in {"https", "data"}:
        raise ValueError("image URL must use HTTPS")
    reason = compatibility_error(model, duration=duration, size=size)
    if reason:
        raise ValueError(f"{model_id}: {reason}")
    normalized_duration: int | float = duration
    if isinstance(duration, float) and duration.is_integer():
        normalized_duration = int(duration)
    payload: dict[str, Any] = {
        "model": model_id,
        "prompt": prompt.strip(),
        "duration": normalized_duration,
        "generate_audio": False,
        "frame_images": [
            {
                "type": "image_url",
                "image_url": {"url": image_url},
                "frame_type": "first_frame",
            }
        ],
    }
    if seed is not None and seed >= 0:
        payload["seed"] = seed
    supported_sizes = model.get("supported_sizes")
    if model_id in FIRST_FRAME_RATIO_FROM_IMAGE_MODELS:
        payload["resolution"], _ = output_geometry(size)
    elif isinstance(supported_sizes, list) and size in supported_sizes:
        payload["size"] = size
    else:
        payload["resolution"], payload["aspect_ratio"] = output_geometry(size)
    return payload


def payload_geometry(payload: dict[str, Any]) -> str:
    size = payload.get("size")
    if isinstance(size, str):
        return size
    resolution = payload.get("resolution")
    aspect_ratio = payload.get("aspect_ratio")
    if isinstance(resolution, str) and isinstance(aspect_ratio, str):
        return f"{resolution} {aspect_ratio}"
    if isinstance(resolution, str):
        return f"{resolution} (first-frame ratio from image)"
    raise ValueError("payload has no output geometry")


def safe_url_label(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.hostname or 'unknown-host'}/<redacted>"


def sanitized_job_state(job: dict[str, Any]) -> dict[str, Any]:
    allowed = ("id", "polling_url", "status", "generation_id", "error", "usage")
    return {key: job[key] for key in allowed if key in job}


class OpenRouterBackend:
    def __init__(self, client: OpenRouterVideoClient) -> None:
        self.client = client

    def validate_request(self, request: GenerationRequest) -> dict[str, Any]:
        models = {model.get("id"): model for model in self.client.list_models()}
        model = models.get(request.model)
        if not isinstance(model, dict):
            raise ValueError(
                f"model {request.model!r} is not present in OpenRouter /videos/models"
            )
        reason = compatibility_error(
            model, duration=request.duration, size=request.size
        )
        if reason:
            raise ValueError(f"{request.model}: {reason}")
        return model

    def generate(
        self,
        request: GenerationRequest,
        *,
        resume_state: dict[str, Any] | None = None,
        on_state: Callable[[dict[str, Any]], None] | None = None,
    ) -> GenerationResult:
        if not request.prompt:
            raise ValueError("OpenRouter generation request has no prompt")
        if not request.image_url:
            raise ValueError("OpenRouter generation request has no public image URL")
        model = self.validate_request(request)
        payload = build_payload(
            model=model,
            prompt=request.prompt,
            duration=request.duration,
            size=request.size,
            image_url=request.image_url,
            seed=request.seed,
        )
        if resume_state and resume_state.get("id"):
            job = dict(resume_state)
        else:
            job = self.client.submit(payload)
            state = sanitized_job_state(job)
            if not state.get("id"):
                raise SubmissionUnknownError(
                    "OpenRouter accepted the request without a recoverable job id"
                )
            if on_state is not None:
                on_state(state)
        completed = self.client.wait_for_completion(job)
        self.client.download(completed, request.output_path)
        usage = completed.get("usage")
        usage = dict(usage) if isinstance(usage, dict) else {}
        cost_value = usage.get("cost")
        cost = float(cost_value) if isinstance(cost_value, (int, float)) else None
        effective = {
            key: value
            for key, value in payload.items()
            if key not in {"prompt", "frame_images"}
        }
        return GenerationResult(
            video_path=str(Path(request.output_path).resolve()),
            model_id=request.model,
            provider_job_id=str(completed.get("id") or job.get("id")),
            effective_parameters=effective,
            usage=usage,
            cost=cost,
        )
