"""Config loading. Single source of truth = config/config.yaml."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


class Config(dict):
    """dict with attribute access, so cfg.data.raw_path works."""

    def __getattr__(self, item: str) -> Any:
        try:
            val = self[item]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(item) from exc
        return Config(val) if isinstance(val, dict) else val


def load_config(path: str | os.PathLike | None = None) -> Config:
    path = Path(path or os.getenv("CONFIG_PATH", ROOT / "config" / "config.yaml"))
    with open(path, "r", encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh))


def resolve(rel: str) -> Path:
    """Resolve a config-relative path against the project root."""
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


CFG = load_config()
