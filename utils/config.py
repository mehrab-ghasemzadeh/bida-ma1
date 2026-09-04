"""Configuration loading: YAML -> nested dot-accessible dict, with CLI overrides."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


class Config(dict):
    """A dict that also supports attribute access and nested `get`/`set` by dotted key."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in list(self.items()):
            if isinstance(value, dict) and not isinstance(value, Config):
                self[key] = Config(value)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = Config(value) if isinstance(value, dict) else value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node: Any = self
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = Config()
            node = node[part]
        node[parts[-1]] = Config(value) if isinstance(value, dict) else value

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in self.items():
            out[key] = value.to_dict() if isinstance(value, Config) else value
        return out

    def dump(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return json.dumps(self.to_dict(), indent=2, default=str)


def _deep_update(base: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in other.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _parse_scalar(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> Config:
    """Load a YAML config.

    Supports a `_base_` key (path relative to the child config) for inheritance, and
    `key.subkey=value` CLI overrides.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    base_ref = raw.pop("_base_", None)
    if base_ref is not None:
        base_path = (path.parent / base_ref).resolve()
        merged = load_config(base_path).to_dict()
        raw = _deep_update(merged, raw)

    cfg = Config(copy.deepcopy(raw))

    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override '{override}' is not of the form key.subkey=value")
        key, _, value = override.partition("=")
        cfg.set_path(key.strip(), _parse_scalar(value.strip()))
    return cfg
