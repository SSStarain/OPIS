"""Volcengine Ark asynchronous image-to-video backend."""

from __future__ import annotations

from collections.abc import Callable
import json
from math import gcd
import os
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .base import SubmissionUnknownError
from ..schema import GenerationRequest, GenerationResult
from ..security import sanitize_text


DEFAULT_ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"


class ArkError(RuntimeError):
    """A Volcengine Ark request or video task failed."""


class _RejectRedirectHandler(HTTPRedirectHandler):
    """Ark API redirects are unexpected and must not receive credentials."""

    def redirect_request(
        self,
        request: Request,
        fp: Any,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> None:
        del request, fp, code, message, headers, new_url
        return None


def ark_output_geometry(size: str) -> tuple[str, str]:
    try:
        width_text, height_text = size.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"invalid output size: {size!r}; expected WIDTHxHEIGHT") from exc
    if width < 1 or height < 1:
        raise ValueError("output dimensions must be positive")
    ratio = _supported_aspect_ratio(width, height)
    if ratio is None:
        raise ValueError(f"unsupported Ark output aspect ratio for {size}")
    edge = min(width, height)
    if edge not in {480, 720, 1080}:
        raise ValueError(
            f"unsupported Ark resolution for {size}; shortest edge must be 480, 720, or 1080"
        )
    return f"{edge}p", ratio


def _supported_aspect_ratio(width: int, height: int) -> str | None:
    ratios = {
        (16, 9): "16:9",
        (4, 3): "4:3",
        (1, 1): "1:1",
        (3, 4): "3:4",
        (9, 16): "9:16",
        (21, 9): "21:9",
    }
    divisor = gcd(width, height)
    exact = ratios.get((width // divisor, height // divisor))
    if exact is not None:
        return exact
    ratio = width / height
    for base, label in (
        ((16, 9), "16:9"),
        ((4, 3), "4:3"),
        ((1, 1), "1:1"),
        ((3, 4), "3:4"),
        ((9, 16), "9:16"),
        ((21, 9), "21:9"),
    ):
        if abs(ratio - (base[0] / base[1])) <= 0.015:
            return label
    return None


def build_ark_payload(request: GenerationRequest) -> dict[str, Any]:
    model = request.model.strip()
    prompt = (request.prompt or "").strip()
    image_url = (request.image_url or "").strip()
    if not model:
        raise ValueError("Ark model ID is empty")
    if not prompt:
        raise ValueError("Ark generation request has no prompt")
    parsed = urlparse(image_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("Ark generation request requires a public HTTPS image URL")
    if isinstance(request.duration, bool) or not float(request.duration).is_integer():
        raise ValueError("Ark duration must be an integer number of seconds")
    duration = int(request.duration)
    if duration <= 0:
        raise ValueError("Ark duration must be positive")
    if (
        isinstance(request.seed, bool)
        or not isinstance(request.seed, int)
        or not 0 <= request.seed <= 2**32 - 1
    ):
        raise ValueError("Ark seed must be an integer between 0 and 2^32-1")
    resolution, _ = ark_output_geometry(request.size)
    return {
        "model": model,
        "content": [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": image_url},
                "role": "first_frame",
            },
        ],
        "duration": duration,
        "resolution": resolution,
        "seed": request.seed,
        "generate_audio": False,
        "watermark": False,
    }


def sanitized_ark_state(task: dict[str, Any]) -> dict[str, Any]:
    state = {
        "provider": "ark",
        "id": task.get("id"),
        "status": task.get("status") or "queued",
    }
    return {key: value for key, value in state.items() if value is not None}


class ArkVideoClient:
    """Small HTTP client for Ark content-generation tasks."""

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str = DEFAULT_ARK_BASE_URL,
        request_timeout: float = 60,
        poll_interval: float = 30,
        generation_timeout: float = 3600,
        max_get_retries: int = 3,
        retry_backoff: float = 1,
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("Ark base URL must be a valid HTTPS URL")
        if request_timeout <= 0 or generation_timeout <= 0:
            raise ValueError("Ark timeouts must be positive")
        if poll_interval < 0 or max_get_retries < 0 or retry_backoff < 0:
            raise ValueError("Ark retry settings cannot be negative")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.request_timeout = request_timeout
        self.poll_interval = poll_interval
        self.generation_timeout = generation_timeout
        self.max_get_retries = max_get_retries
        self.retry_backoff = retry_backoff
        self._api_opener = build_opener(_RejectRedirectHandler())
        self._download_opener = build_opener()

    def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request_json(
            "POST", f"{self.base_url}/contents/generations/tasks", payload=payload
        )

    def get_task(self, task_id: str) -> dict[str, Any]:
        if not task_id:
            raise ArkError("Ark task has no id")
        encoded = quote(task_id, safe="")
        return self._request_json(
            "GET", f"{self.base_url}/contents/generations/tasks/{encoded}"
        )

    def wait_for_completion(self, task: dict[str, Any]) -> dict[str, Any]:
        current = task
        deadline = time.monotonic() + self.generation_timeout
        while True:
            status = current.get("status") or "queued"
            if status == "succeeded":
                return current
            if status in {"failed", "expired", "cancelled"}:
                raise ArkError(self._task_error(current, str(status)))
            if status not in {"queued", "running"}:
                raise ArkError(f"unexpected Ark task status: {status!r}")
            if time.monotonic() >= deadline:
                raise ArkError(
                    f"Ark video generation timed out after {self.generation_timeout:g}s"
                )
            task_id = current.get("id")
            if not isinstance(task_id, str) or not task_id:
                raise ArkError("Ark task has no id")
            if self.poll_interval:
                time.sleep(self.poll_interval)
            current = self.get_task(task_id)

    def download(self, task: dict[str, Any], output_path: str | Path) -> Path:
        content = task.get("content")
        video_url = content.get("video_url") if isinstance(content, dict) else None
        if not isinstance(video_url, str) or not video_url:
            raise ArkError("succeeded Ark task has no content.video_url")
        parsed = urlparse(video_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ArkError("Ark video URL must use HTTPS")
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.part")
        try:
            for attempt in range(self.max_get_retries + 1):
                request = Request(
                    video_url,
                    headers={"User-Agent": "MultiMemBench-Ark-I2V/2.0"},
                    method="GET",
                )
                try:
                    with self._download_opener.open(
                        request, timeout=self.request_timeout
                    ) as response:
                        content_type = response.headers.get_content_type()
                        if not (
                            content_type.startswith("video/")
                            or content_type == "application/octet-stream"
                        ):
                            raise ArkError(
                                "Ark video download returned unexpected content type: "
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
                    raise ArkError(f"Ark video download failed: {exc.reason}") from None
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise ArkError("Ark returned an empty video")
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        return output

    def _request_json(
        self,
        method: str,
        url: str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise ArkError("ARK_API_KEY is not set")
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = Request(url, data=data, headers=headers, method=method)
        attempts = self.max_get_retries + 1 if method == "GET" else 1
        value: Any = None
        for attempt in range(attempts):
            try:
                with self._api_opener.open(
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
                error_type = SubmissionUnknownError if method == "POST" else ArkError
                raise error_type(f"Ark request failed: {exc.reason}") from None
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                error_type = SubmissionUnknownError if method == "POST" else ArkError
                raise error_type("Ark returned invalid JSON") from exc
        if not isinstance(value, dict):
            error_type = SubmissionUnknownError if method == "POST" else ArkError
            raise error_type("Ark returned a non-object JSON response")
        return value

    def _task_error(self, task: dict[str, Any], status: str) -> str:
        error = task.get("error")
        if isinstance(error, dict):
            code = error.get("code")
            message = error.get("message")
            detail = ": ".join(str(item) for item in (code, message) if item)
        else:
            detail = str(error or "")
        text = f"Ark video generation {status}"
        if detail:
            text = f"{text}: {detail}"
        return sanitize_text(text, [self.api_key or ""])

    def _sleep_before_retry(self, attempt: int) -> None:
        delay = self.retry_backoff * (2**attempt)
        if delay:
            time.sleep(delay)

    def _http_error(self, exc: HTTPError) -> ArkError:
        try:
            detail = exc.read(4096).decode("utf-8", errors="replace").strip()
        except OSError:
            detail = ""
        detail = sanitize_text(detail, [self.api_key or ""])
        suffix = f": {detail}" if detail else ""
        return ArkError(f"Ark HTTP {exc.code}{suffix}")


class ArkBackend:
    def __init__(self, client: ArkVideoClient) -> None:
        self.client = client

    def validate_request(self, request: GenerationRequest) -> dict[str, Any]:
        return build_ark_payload(request)

    def generate(
        self,
        request: GenerationRequest,
        *,
        resume_state: dict[str, Any] | None = None,
        on_state: Callable[[dict[str, Any]], None] | None = None,
    ) -> GenerationResult:
        payload = self.validate_request(request)
        if resume_state and resume_state.get("id"):
            provider = resume_state.get("provider")
            if provider not in {None, "ark"}:
                raise ValueError(f"provider state belongs to {provider!r}, not Ark")
            task = {
                "id": resume_state["id"],
                "status": resume_state.get("status") or "queued",
            }
        else:
            task = self.client.submit(payload)
            if not task.get("status"):
                task["status"] = "queued"
            state = sanitized_ark_state(task)
            if not state.get("id"):
                raise SubmissionUnknownError(
                    "Ark accepted the request without a recoverable task id"
                )
            if on_state is not None:
                on_state(state)
        completed = self.client.wait_for_completion(task)
        self.client.download(completed, request.output_path)
        usage_value = completed.get("usage")
        usage = dict(usage_value) if isinstance(usage_value, dict) else {}
        cost_value = completed.get("cost", usage.get("cost"))
        cost = (
            float(cost_value)
            if isinstance(cost_value, (int, float)) and not isinstance(cost_value, bool)
            else None
        )
        effective = {
            key: value
            for key, value in payload.items()
            if key not in {"content"}
        }
        return GenerationResult(
            video_path=str(Path(request.output_path).resolve()),
            model_id=str(completed.get("model") or request.model),
            provider_job_id=str(completed.get("id") or task.get("id")),
            effective_parameters=effective,
            usage=usage,
            cost=cost,
        )
