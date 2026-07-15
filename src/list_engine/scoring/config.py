"""Strict, reviewable loading for segment-as-config YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError

from list_engine.scoring.models import SegmentConfig

_SEQUENCE_FIELDS = (
    "ateco_prefixes",
    "target_regions",
    "excluded_company_statuses",
    "hot_signal_types",
)
_YAML_SUFFIXES = frozenset({".yaml", ".yml"})


class SegmentConfigError(ValueError):
    """A segment configuration could not be loaded without ambiguity."""


def load_segment_config(path: str | Path) -> SegmentConfig:
    """Load one YAML file without applying business-value coercions.

    PyYAML represents a YAML sequence as a list, while ``SegmentConfig`` uses
    immutable tuples. Those four sequence fields are the only normalization
    performed here; Pydantic's strict model validation handles everything else.
    """

    config_path = _regular_file(path)
    try:
        loaded: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise SegmentConfigError(f"cannot parse segment config {config_path}: {exc}") from exc

    if not isinstance(loaded, dict):
        raise SegmentConfigError(f"segment config {config_path} must contain a YAML object")
    if any(not isinstance(key, str) for key in loaded):
        raise SegmentConfigError(f"segment config {config_path} must use string keys")

    payload = dict(cast(dict[str, object], loaded))
    for field_name in _SEQUENCE_FIELDS:
        value = payload.get(field_name)
        if isinstance(value, list):
            payload[field_name] = tuple(value)

    # ``bool`` is a subclass of ``int`` in Python, and Literal[1] would
    # otherwise accept YAML ``true``. It is not a valid schema version.
    if isinstance(payload.get("schema_version"), bool):
        raise SegmentConfigError(f"segment config {config_path} has a non-integer schema_version")

    try:
        return SegmentConfig.model_validate(payload)
    except ValidationError as exc:
        raise SegmentConfigError(f"invalid segment config {config_path}: {exc}") from exc


def load_segment_configs(directory: str | Path) -> tuple[SegmentConfig, ...]:
    """Load direct YAML children and reject duplicate segment versions.

    The directory and every selected file must be real paths, not symlinks.
    Files are never discovered recursively, keeping traversal outside the
    declared configuration root impossible.
    """

    config_directory = _real_directory(directory)
    paths: list[Path] = []
    for candidate in config_directory.iterdir():
        if candidate.suffix.lower() not in _YAML_SUFFIXES:
            continue
        if candidate.is_symlink():
            raise SegmentConfigError(f"segment config symlinks are not allowed: {candidate}")
        resolved = candidate.resolve(strict=True)
        if resolved.parent != config_directory or not resolved.is_file():
            raise SegmentConfigError(f"segment config escapes its directory: {candidate}")
        paths.append(resolved)

    if not paths:
        raise SegmentConfigError(f"no segment YAML files found in {config_directory}")

    configs = tuple(load_segment_config(path) for path in sorted(paths))
    seen_versions: set[str] = set()
    for config in configs:
        if config.scoring_version in seen_versions:
            raise SegmentConfigError(
                f"duplicate segment configuration version: {config.scoring_version}"
            )
        seen_versions.add(config.scoring_version)
    return configs


def _regular_file(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        raise SegmentConfigError(f"segment config symlinks are not allowed: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise SegmentConfigError(f"segment config does not exist: {candidate}") from exc
    if not resolved.is_file():
        raise SegmentConfigError(f"segment config is not a regular file: {candidate}")
    return resolved


def _real_directory(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        raise SegmentConfigError(f"segment config directory symlinks are not allowed: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise SegmentConfigError(f"segment config directory does not exist: {candidate}") from exc
    if not resolved.is_dir():
        raise SegmentConfigError(f"segment config path is not a directory: {candidate}")
    return resolved
