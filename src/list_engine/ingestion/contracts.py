"""Strict, provider-neutral contracts for cold-source ingestion."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from list_engine.core.models import Company
from list_engine.ingestion.quality import (
    QualityReport,
    canonical_payload_hash,
    validate_json_object,
)

_SOURCE_ID = re.compile(r"[a-z][a-z0-9_.-]{0,63}")
_PRINTABLE_ASCII = re.compile(r"[\x20-\x7e]+")


def _validate_source(value: str) -> str:
    if _SOURCE_ID.fullmatch(value) is None:
        raise ValueError(
            "source must be a lowercase ASCII identifier using letters, digits, '.', '_' or '-'"
        )
    return value


def _validate_printable_ascii(value: str, *, label: str) -> str:
    if value != value.strip() or _PRINTABLE_ASCII.fullmatch(value) is None:
        raise ValueError(f"{label} must be non-blank printable ASCII without outer whitespace")
    return value


class SourceReadStatus(StrEnum):
    """Terminal outcome of one source read.

    An explicit empty result is deliberately not represented as a successful read
    with zero records: scraper drift and blocked pages must remain failures.
    """

    SUCCESS = "success"
    EMPTY_VERIFIED = "empty_verified"
    FAILED = "failed"


class SchemaContract(BaseModel):
    """Versioned field contract used to detect source schema drift."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    source: str
    version: int = Field(default=1, ge=1)
    required_fields: frozenset[str] = Field(min_length=1)
    optional_fields: frozenset[str] = frozenset()
    allow_additional_fields: bool = False

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        return _validate_source(value)

    @field_validator("required_fields", "optional_fields")
    @classmethod
    def validate_field_names(cls, values: frozenset[str]) -> frozenset[str]:
        for value in values:
            if len(value) > 128:
                raise ValueError("schema field names may not exceed 128 characters")
            _validate_printable_ascii(value, label="schema field name")
        return values

    @model_validator(mode="after")
    def validate_disjoint_fields(self) -> Self:
        overlap = self.required_fields & self.optional_fields
        if overlap:
            joined = ", ".join(sorted(overlap))
            raise ValueError(f"schema fields cannot be both required and optional: {joined}")
        return self

    @property
    def known_fields(self) -> frozenset[str]:
        return self.required_fields | self.optional_fields


class RawRecordEnvelope(BaseModel):
    """Lossless JSON payload plus source metadata, before golden-record mapping."""

    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        extra="forbid",
        strict=True,
    )

    source: str
    source_record_id: str | None = Field(default=None, min_length=1, max_length=255)
    payload: dict[str, object]
    observed_at: AwareDatetime | None = None

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        return _validate_source(value)

    @field_validator("source_record_id")
    @classmethod
    def validate_source_record_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_printable_ascii(value, label="source_record_id")

    @field_validator("payload", mode="before")
    @classmethod
    def validate_payload(cls, value: object) -> dict[str, object]:
        return validate_json_object(value)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def payload_hash(self) -> str:
        """Canonical SHA-256 used by the append-only source replay guard."""

        return canonical_payload_hash(self.payload)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def record_key(self) -> str:
        return self.source_record_id or self.payload_hash


class RawBatchEnvelope(BaseModel):
    """One terminal source-read result with state invariants enforced."""

    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        extra="forbid",
        strict=True,
    )

    source: str
    status: SourceReadStatus
    records: tuple[RawRecordEnvelope, ...] = ()
    status_detail: str | None = Field(default=None, min_length=1, max_length=2_000)

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        return _validate_source(value)

    @model_validator(mode="after")
    def validate_terminal_state(self) -> Self:
        wrong_sources = sorted(
            {record.source for record in self.records if record.source != self.source}
        )
        if wrong_sources:
            raise ValueError("every raw record must belong to the batch source")
        if self.status is SourceReadStatus.SUCCESS and not self.records:
            raise ValueError("successful source reads must contain at least one record")
        if self.status is SourceReadStatus.EMPTY_VERIFIED:
            if self.records:
                raise ValueError("verified-empty source reads cannot contain records")
            if self.status_detail is None:
                raise ValueError(
                    "verified-empty source reads require source evidence in status_detail"
                )
        if self.status is SourceReadStatus.FAILED and self.status_detail is None:
            raise ValueError("failed source reads require a failure detail")
        return self


class MappedCompanyRecord(BaseModel):
    """One raw observation and its narrow canonical projection, if safe."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    raw: RawRecordEnvelope
    company: Company | None = None
    quality: QualityReport = QualityReport()

    @model_validator(mode="after")
    def validate_mapping_boundary(self) -> Self:
        if self.company is None and not self.quality.has_errors:
            raise ValueError("an unmapped record requires a quarantine error")
        if self.company is not None:
            if self.company.source != self.raw.source:
                raise ValueError("mapped company and raw record must have the same source")
            if self.raw.observed_at != self.company.source_observed_at:
                raise ValueError("mapped company and raw record must share observed_at")
        return self


class SourceSchemaObservation(BaseModel):
    """One exact source schema seen at an adapter boundary.

    ``actual_fields`` is canonicalised only as a set representation (sorted and
    unique). Field spelling is otherwise retained exactly so whitespace, blank,
    or unexpected provider headers remain observable instead of being repaired.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    source: str
    contract_version: int = Field(ge=1)
    actual_fields: tuple[str, ...]
    quality: QualityReport = QualityReport()

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        return _validate_source(value)

    @field_validator("actual_fields")
    @classmethod
    def validate_actual_fields(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if values != tuple(sorted(set(values))):
            raise ValueError("actual_fields must be sorted and unique")
        for value in values:
            if len(value) > 2_000:
                raise ValueError("observed source field names may not exceed 2,000 characters")
            if _has_unicode_surrogate(value):
                raise ValueError("observed source field names cannot contain Unicode surrogates")
        return values


class MappedSourceBatch(BaseModel):
    """Provider-neutral batch passed to the deterministic ingestion service."""

    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        extra="forbid",
        strict=True,
    )

    source: str
    status: SourceReadStatus
    records: tuple[MappedCompanyRecord, ...] = ()
    status_detail: str | None = Field(default=None, min_length=1, max_length=2_000)
    schema_observation: SourceSchemaObservation | None = None

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        return _validate_source(value)

    @model_validator(mode="after")
    def validate_terminal_state(self) -> Self:
        if any(record.raw.source != self.source for record in self.records):
            raise ValueError("every mapped record must belong to the batch source")
        if self.schema_observation is not None and self.schema_observation.source != self.source:
            raise ValueError("schema observation and mapped batch must have the same source")
        if self.status is SourceReadStatus.SUCCESS and not self.records:
            raise ValueError("successful mapped source reads must contain records")
        if self.status is SourceReadStatus.EMPTY_VERIFIED:
            if self.records:
                raise ValueError("verified-empty mapped reads cannot contain records")
            if self.status_detail is None:
                raise ValueError("verified-empty mapped reads require source evidence")
        if self.status is SourceReadStatus.FAILED:
            if self.records:
                raise ValueError("failed mapped reads cannot contain records")
            if self.status_detail is None:
                raise ValueError("failed mapped reads require a failure detail")
        if self.schema_observation is not None:
            schema_has_errors = self.schema_observation.quality.has_errors
            if self.status is SourceReadStatus.FAILED and not schema_has_errors:
                raise ValueError("a failed schema observation must contain a quality error")
            if self.status is not SourceReadStatus.FAILED and schema_has_errors:
                raise ValueError("a breaking schema observation requires a failed mapped read")
        return self


def _has_unicode_surrogate(value: str) -> bool:
    return any(0xD800 <= ord(character) <= 0xDFFF for character in value)
