"""Portable repository-relative path handling."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse


PROJECT_PATH_PREFIXES = (
    ".venv",
    "artifacts/",
    "dataset/",
    "models/",
    "multiMemBench/",
    "run_model/",
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_repository_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repository_root() / path).resolve()


def resolve_config_path(value: str, *, config_dir: Path | None = None) -> str:
    """Resolve a local config path while leaving model IDs and URLs intact."""
    if not value or urlparse(value).scheme:
        return value
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    if value.startswith(PROJECT_PATH_PREFIXES):
        return str(resolve_repository_path(path))
    if config_dir is not None:
        candidate = (config_dir / path).resolve()
        if candidate.exists():
            return str(candidate)
    return value
