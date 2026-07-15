"""Deterministic data-quality checks and canonical raw-payload hashing."""

from __future__ import annotations

import hashlib
import json
import math
import re
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

if TYPE_CHECKING:
    from list_engine.ingestion.contracts import SchemaContract

_ISSUE_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_MAX_JSON_DEPTH = 100


def _has_surrogate(value: str) -> bool:
    return any(0xD800 <= ord(character) <= 0xDFFF for character in value)


def _validate_json_value(value: object, *, path: str, depth: int, seen: set[int]) -> None:
    if depth > _MAX_JSON_DEPTH:
        raise ValueError(f"payload exceeds the maximum JSON depth at {path}")
    if value is None or isinstance(value, bool | int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"payload contains a non-finite number at {path}")
        return
    if isinstance(value, str):
        if _has_surrogate(value):
            raise ValueError(f"payload contains an invalid Unicode surrogate at {path}")
        return
    if isinstance(value, dict):
        if type(value) is not dict:
            raise ValueError(f"payload contains a non-JSON mapping at {path}")
        identity = id(value)
        if identity in seen:
            raise ValueError(f"payload contains a circular reference at {path}")
        seen.add(identity)
        try:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError(f"payload object keys must be strings at {path}")
                if _has_surrogate(key):
                    raise ValueError(f"payload contains an invalid Unicode key at {path}")
                _validate_json_value(item, path=f"{path}.{key}", depth=depth + 1, seen=seen)
        finally:
            seen.remove(identity)
        return
    if isinstance(value, list):
        if type(value) is not list:
            raise ValueError(f"payload contains a non-JSON sequence at {path}")
        identity = id(value)
        if identity in seen:
            raise ValueError(f"payload contains a circular reference at {path}")
        seen.add(identity)
        try:
            for index, item in enumerate(value):
                _validate_json_value(item, path=f"{path}[{index}]", depth=depth + 1, seen=seen)
        finally:
            seen.remove(identity)
        return
    raise ValueError(f"payload contains a non-JSON value at {path}")


def validate_json_object(value: object) -> dict[str, object]:
    """Return ``value`` after proving it is a strict, finite JSON object."""

    if type(value) is not dict:
        raise ValueError("raw payload must be a JSON object")
    _validate_json_value(value, path="$", depth=0, seen=set())
    return value


def canonical_payload_hash(payload: object) -> str:
    """Return a deterministic SHA-256 for an exact JSON object.

    Sorting keys removes input-order differences. ``ensure_ascii`` makes the byte
    representation independent of terminal/source encoding without changing the raw
    payload retained for provenance.
    """

    validated = validate_json_object(payload)
    canonical = json.dumps(
        validated,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


class IssueSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class QualityIssue(BaseModel):
    """One deterministic quality finding suitable for quarantine or logging."""

    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        extra="forbid",
        strict=True,
    )

    severity: IssueSeverity
    code: str
    message: str = Field(min_length=1, max_length=2_000)
    field: str | None = Field(default=None, min_length=1, max_length=128)
    record_index: int | None = Field(default=None, ge=0)

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        if _ISSUE_CODE.fullmatch(value) is None:
            raise ValueError("quality issue code must be a lowercase ASCII identifier")
        return value


class QualityReport(BaseModel):
    """Immutable quality-gate result with explicit quarantine semantics."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    issues: tuple[QualityIssue, ...] = ()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_errors(self) -> bool:
        return any(issue.severity is IssueSeverity.ERROR for issue in self.issues)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_warnings(self) -> bool:
        return any(issue.severity is IssueSeverity.WARNING for issue in self.issues)

    @property
    def should_quarantine(self) -> bool:
        return self.has_errors


def evaluate_schema(
    payload: object,
    contract: SchemaContract,
    *,
    record_index: int | None = None,
) -> QualityReport:
    """Compare one raw payload against a versioned schema contract."""

    validated = validate_json_object(payload)
    actual_fields = set(validated)
    missing = sorted(contract.required_fields - actual_fields)
    unexpected = sorted(actual_fields - contract.known_fields)
    issues: list[QualityIssue] = []

    for field in missing:
        issues.append(
            QualityIssue(
                severity=IssueSeverity.ERROR,
                code="missing_required_field",
                message=f"Required source field is missing: {field}",
                field=field,
                record_index=record_index,
            )
        )
    extra_severity = (
        IssueSeverity.WARNING if contract.allow_additional_fields else IssueSeverity.ERROR
    )
    for field in unexpected:
        issues.append(
            QualityIssue(
                severity=extra_severity,
                code="unexpected_field",
                message=f"Source schema contains an unexpected field: {field}",
                field=field,
                record_index=record_index,
            )
        )
    return QualityReport(issues=tuple(issues))
