"""S3-compatible publication of first-frame images."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import mimetypes
import os
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .schema import sha256_file


class StorageError(RuntimeError):
    """S3 publication or public image validation failed."""


@dataclass(frozen=True)
class S3PublishConfig:
    bucket: str
    endpoint_url: str | None = None
    region: str | None = None
    prefix: str = "multimembench/inputs"
    url_ttl_seconds: int = 86400
    public_base_url: str | None = None
    request_timeout: float = 30.0

    def __post_init__(self) -> None:
        if not self.bucket.strip():
            raise ValueError("S3 bucket is required")
        if not 60 <= self.url_ttl_seconds <= 604800:
            raise ValueError("S3 URL lifetime must be between 60 and 604800 seconds")
        if self.request_timeout <= 0:
            raise ValueError("S3 URL validation timeout must be positive")
        if not self.prefix.strip("/"):
            raise ValueError("S3 prefix cannot be empty")
        if self.public_base_url:
            parsed = urlparse(self.public_base_url)
            if parsed.scheme != "https" or not parsed.netloc:
                raise ValueError("S3 public base URL must be HTTPS")

    @classmethod
    def from_env_and_overrides(
        cls,
        *,
        bucket: str | None = None,
        endpoint_url: str | None = None,
        region: str | None = None,
        prefix: str | None = None,
        url_ttl_seconds: int | None = None,
        public_base_url: str | None = None,
        request_timeout: float = 30.0,
    ) -> "S3PublishConfig":
        raw_ttl = os.environ.get("MULTIMEM_S3_URL_TTL_SECONDS", "86400")
        try:
            env_ttl = int(raw_ttl)
        except ValueError as exc:
            raise ValueError("MULTIMEM_S3_URL_TTL_SECONDS must be an integer") from exc
        return cls(
            bucket=bucket or os.environ.get("MULTIMEM_S3_BUCKET", ""),
            endpoint_url=endpoint_url or os.environ.get("MULTIMEM_S3_ENDPOINT_URL"),
            region=region
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION"),
            prefix=prefix or os.environ.get("MULTIMEM_S3_PREFIX", "multimembench/inputs"),
            url_ttl_seconds=url_ttl_seconds if url_ttl_seconds is not None else env_ttl,
            public_base_url=public_base_url
            or os.environ.get("MULTIMEM_S3_PUBLIC_BASE_URL"),
            request_timeout=request_timeout,
        )


@dataclass(frozen=True)
class PublishedImage:
    bucket: str
    key: str
    sha256: str
    content_type: str
    created: bool
    url_kind: str
    endpoint_host: str
    url: str
    expires_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_manifest_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("url", None)
        return value


class _RejectRedirects(HTTPRedirectHandler):
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


def validate_direct_image_url(url: str, timeout: float) -> None:
    """Require a redirect-free HTTPS response with an image content type."""

    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise StorageError("first-frame URL must be a public HTTPS URL")
    request = Request(url, method="GET", headers={"User-Agent": "MultiMemBench/2"})
    try:
        with build_opener(_RejectRedirects()).open(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            content_type = response.headers.get_content_type()
            response.read(1)
    except HTTPError as exc:
        exc.close()
        raise StorageError(f"first-frame URL returned HTTP {exc.code}") from None
    except URLError as exc:
        raise StorageError(f"first-frame URL is not reachable: {exc.reason}") from None
    if status != 200:
        raise StorageError(f"first-frame URL returned HTTP {status}")
    if not content_type.startswith("image/"):
        raise StorageError(
            f"first-frame URL returned unexpected content type: {content_type}"
        )


class S3ImagePublisher:
    def __init__(
        self,
        config: S3PublishConfig,
        *,
        client: Any | None = None,
        url_validator: Callable[[str, float], None] = validate_direct_image_url,
    ) -> None:
        self.config = config
        self.client = client if client is not None else self._build_client(config)
        self.url_validator = url_validator

    @staticmethod
    def _build_client(config: S3PublishConfig) -> Any:
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise StorageError(
                "boto3 is required for S3 publication; install multiMemBench[orchestration]"
            ) from exc
        return boto3.client(
            "s3",
            endpoint_url=config.endpoint_url,
            region_name=config.region,
            config=Config(signature_version="s3v4"),
        )

    def publish(self, path: str | Path) -> PublishedImage:
        image_path = Path(path).expanduser().resolve()
        content_type, extension = _inspect_image(image_path)
        digest = sha256_file(image_path)
        byte_size = image_path.stat().st_size
        key = f"{self.config.prefix.strip('/')}/{digest}{extension}"
        existing = self._head(key)
        created = existing is None
        if existing is not None:
            _require_matching_object(
                existing,
                byte_size=byte_size,
                content_type=content_type,
                digest=digest,
                key=key,
            )
        else:
            self.client.upload_file(
                str(image_path),
                self.config.bucket,
                key,
                ExtraArgs={
                    "ContentType": content_type,
                    "Metadata": {"sha256": digest},
                },
            )

        expires_at: str | None = None
        if self.config.public_base_url:
            url = f"{self.config.public_base_url.rstrip('/')}/{quote(key, safe='/')}"
            url_kind = "public"
        else:
            url = self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.config.bucket, "Key": key},
                ExpiresIn=self.config.url_ttl_seconds,
            )
            url_kind = "presigned"
            expires_at = (
                datetime.now(timezone.utc)
                + timedelta(seconds=self.config.url_ttl_seconds)
            ).isoformat().replace("+00:00", "Z")

        self.url_validator(url, self.config.request_timeout)
        return PublishedImage(
            bucket=self.config.bucket,
            key=key,
            sha256=digest,
            content_type=content_type,
            created=created,
            url_kind=url_kind,
            endpoint_host=urlparse(url).hostname or "unknown",
            url=url,
            expires_at=expires_at,
        )

    def cleanup(self, published: PublishedImage) -> bool:
        if not published.created:
            return False
        self.client.delete_object(Bucket=published.bucket, Key=published.key)
        return True

    def _head(self, key: str) -> dict[str, Any] | None:
        try:
            return self.client.head_object(Bucket=self.config.bucket, Key=key)
        except Exception as exc:
            response = getattr(exc, "response", {})
            code = str(response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise StorageError(f"failed to inspect S3 object {key}: {exc}") from exc


def _inspect_image(path: Path) -> tuple[str, str]:
    if not path.is_file():
        raise StorageError(f"input image is not a file: {path}")
    try:
        from PIL import Image
    except ImportError as exc:
        raise StorageError(
            "Pillow is required for image validation; install multiMemBench[orchestration]"
        ) from exc
    try:
        with Image.open(path) as image:
            image.verify()
            image_format = (image.format or "").upper()
    except Exception as exc:
        raise StorageError(f"input image is not a readable raster: {path}") from exc
    formats = {
        "JPEG": ("image/jpeg", ".jpg"),
        "PNG": ("image/png", ".png"),
        "WEBP": ("image/webp", ".webp"),
    }
    if image_format not in formats:
        guessed = mimetypes.guess_type(path.name)[0] or image_format or "unknown"
        raise StorageError(f"unsupported input image type: {guessed}")
    return formats[image_format]


def _require_matching_object(
    value: dict[str, Any],
    *,
    byte_size: int,
    content_type: str,
    digest: str,
    key: str,
) -> None:
    metadata = value.get("Metadata") or {}
    matches = (
        int(value.get("ContentLength", -1)) == byte_size
        and str(value.get("ContentType", "")).split(";", 1)[0].strip().lower()
        == content_type
        and metadata.get("sha256") == digest
    )
    if not matches:
        raise StorageError(
            f"refusing to overwrite conflicting object at content-addressed key {key}"
        )
