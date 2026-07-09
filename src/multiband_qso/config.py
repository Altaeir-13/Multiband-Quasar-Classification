from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return config


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve a project path without requiring the file to exist."""
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path
    root = Path.cwd() if base_dir is None else Path(base_dir)
    return (root / raw_path).resolve()


def ensure_parent(path: str | Path) -> Path:
    """Create the parent directory for an output path and return the path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def require_config(config: dict[str, Any], *keys: str) -> Any:
    """Read a nested key and raise a clear error if it is missing."""
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            dotted = ".".join(keys)
            raise KeyError(f"Missing required config key: {dotted}")
        value = value[key]
    return value
