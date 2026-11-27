"""Config loading: the defaults < --config file < --set KEY=VALUE (see
the package docstring)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

DEFAULTS_DIR = Path(__file__).with_name("defaults")
# Default files, in the order of the merged config
DEFAULT_FILES = ["inputs", "common", "jet_classifier", "jet_pt", "train", "seeds_tau", "seeds_hh"]


def _read_yaml(path):
    """The keys of a YAML config file (an empty file has none)."""
    with open(path) as f:
        values = yaml.safe_load(f) or {}
    if not isinstance(values, dict):
        raise ValueError(
            f"{path}: a config file maps keys to values, got"
            f" {type(values).__name__}"
        )
    return values


def _defaults():
    """The keys of every default file; a key defined in two files is an
    error."""
    config, origin = {}, {}
    for name in DEFAULT_FILES:
        for key, value in _read_yaml(DEFAULTS_DIR / f"{name}.yaml").items():
            if key in config:
                raise KeyError(
                    f"config key {key} is defined in"
                    f" defaults/{origin[key]}.yaml and defaults/{name}.yaml"
                )
            config[key], origin[key] = value, name
    return config


def load(
    config_files: Sequence[str] = (), overrides: Sequence[str] = ()
) -> dict[str, Any]:
    """The merged config: the defaults, then the --config files, then
    the --set overrides.

    Args:
        config_files: Paths of YAML files, each setting some config
            keys.
        overrides: KEY=VALUE strings, VALUE in YAML.

    Returns:
        dict: Config key -> value.

    Raises:
        KeyError: A --config file or --set names an unknown key, or a
            default file repeats a key.
        ValueError: A --set without "=", or a config file that is not a
            mapping.
    """
    config = _defaults()
    for path in config_files:
        for key, value in _read_yaml(path).items():
            if key not in config:
                raise KeyError(
                    f"{path}: unknown config key {key}, known keys:"
                    f" {', '.join(sorted(config))}"
                )
            config[key] = value
    for item in overrides:
        key, sep, text = item.partition("=")
        key = key.strip()
        if not sep:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        if key not in config:
            raise KeyError(
                f"--set: unknown config key {key}, known keys:"
                f" {', '.join(sorted(config))}"
            )
        config[key] = yaml.safe_load(text)
    return config


def config_hash(config: dict[str, Any]) -> str:
    """Short stable hash of a config, stored with every result.

    Args:
        config: Config dict.

    Returns:
        str: The first 12 hex digits of its sha256.
    """
    text = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:12]
