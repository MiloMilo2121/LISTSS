"""Deterministic ingestion workflow shared by batch and event-driven sources."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from list_engine.core.repository import CompanyRepository, InMemoryCompanyRepository
from list_engine.ingestion.contracts import (
    MappedCompanyRecord,
    MappedSourceBatch,
    RawRecordEnvelope,
    SchemaContract,
    SourceReadStatus,
)
from list_engine.ingestion.quality import (
    IssueSeverity,
    QualityIssue,
    QualityReport,
    canonical_payload_hash,
    evaluate_schema,
)


class ApplyAction(StrEnum):
    CREATED = "created"
    UPDATED = "updated"
    UNCHANGED = "unchanged"
    QUARANTINED = "quarantined"
    REPLAYED = "replayed"


class IngestionSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    source: str
    status: SourceReadStatus
    schema_hash: str
    records_seen: int = Field(ge=0)
    records_created: int = Field(ge=0)
    records_updated: int = Field(ge=0)
    records_unchanged: int = Field(ge=0)
    records_quarantined: int = Field(ge=0)
    records_replayed: int = Field(ge=0)
    warning_count: int = Field(ge=0)
    status_detail: str | None = None


class ProcessingContext(BaseModel):
    """Idempotency identity for one raw-to-domain processing attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    contract_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    processing_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class StoredRawRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    raw: RawRecordEnvelope
    piva: str | None


class QuarantinedRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    raw: RawRecordEnvelope
    issues: tuple[QualityIssue, ...]
    status: Literal["pending", "resolved"] = "pending"


class SchemaObservation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    source: str
    schema_hash: str
    field_names: tuple[str, ...]
    drift_status: str


class IngestionStore(Protocol):
    """Atomic persistence boundary for one source observation."""

    def apply_record(
        self,
        record: MappedCompanyRecord,
        quality: QualityReport,
        context: ProcessingContext,
    ) -> ApplyAction: ...

    def observe_schema(
        self,
        source: str,
        contract: SchemaContract,
        field_names: tuple[str, ...],
        report: QualityReport,
    ) -> None: ...

    def save_summary(self, summary: IngestionSummary) -> None: ...


class InMemoryIngestionStore:
    """Production-shaped DEMO store with atomic single-process semantics."""

    def __init__(self, companies: CompanyRepository | None = None) -> None:
        self.companies = companies or InMemoryCompanyRepository()
        self.raw_records: dict[tuple[str, str], StoredRawRecord] = {}
        self.quality_issues: dict[tuple[str, str, str, str, str | None], QualityIssue] = {}
        self.quarantine: dict[tuple[str, str], QuarantinedRecord] = {}
        self.processing: dict[tuple[str, str, str], ApplyAction] = {}
        self.schema_observations: dict[tuple[str, str], SchemaObservation] = {}
        self.runs: list[IngestionSummary] = []

    def apply_record(
        self,
        record: MappedCompanyRecord,
        quality: QualityReport,
        context: ProcessingContext,
    ) -> ApplyAction:
        key = (record.raw.source, record.raw.payload_hash)
        processing_key = (*key, context.processing_hash)
        if processing_key in self.processing:
            return ApplyAction.REPLAYED

        self.raw_records.setdefault(
            key,
            StoredRawRecord(
                raw=record.raw,
                piva=record.company.piva if record.company else None,
            ),
        )
        self._record_issues(record.raw, quality)
        if quality.has_errors or record.company is None:
            self.quarantine[key] = QuarantinedRecord(raw=record.raw, issues=quality.issues)
            self.processing[processing_key] = ApplyAction.QUARANTINED
            return ApplyAction.QUARANTINED

        try:
            result = self.companies.upsert(record.company)
        except ValueError as error:
            issue = QualityIssue(
                severity=IssueSeverity.ERROR,
                code="golden_record_conflict",
                message=str(error),
            )
            conflict_report = QualityReport(issues=(*quality.issues, issue))
            self._record_issues(record.raw, conflict_report)
            self.quarantine[key] = QuarantinedRecord(raw=record.raw, issues=conflict_report.issues)
            self.processing[processing_key] = ApplyAction.QUARANTINED
            return ApplyAction.QUARANTINED

        action = ApplyAction(result.action.value)
        existing_quarantine = self.quarantine.get(key)
        if existing_quarantine is not None:
            self.quarantine[key] = existing_quarantine.model_copy(update={"status": "resolved"})
        self.processing[processing_key] = action
        return action

    def observe_schema(
        self,
        source: str,
        contract: SchemaContract,
        field_names: tuple[str, ...],
        report: QualityReport,
    ) -> None:
        schema_hash = _schema_hash(field_names)
        status = _drift_status(report)
        self.schema_observations[(source, schema_hash)] = SchemaObservation(
            source=source,
            schema_hash=schema_hash,
            field_names=field_names,
            drift_status=status,
        )

    def save_summary(self, summary: IngestionSummary) -> None:
        self.runs.append(summary)

    def _record_issues(self, raw: RawRecordEnvelope, report: QualityReport) -> None:
        for issue in report.issues:
            key = (raw.source, raw.record_key, raw.payload_hash, issue.code, issue.field)
            self.quality_issues.setdefault(key, issue)


class IngestionService:
    """Apply quality gates and persist each observation through one atomic store call."""

    def __init__(self, store: IngestionStore) -> None:
        self._store = store

    def ingest(self, batch: MappedSourceBatch, contract: SchemaContract) -> IngestionSummary:
        if batch.source != contract.source:
            raise ValueError("batch source and schema contract source must match")

        counts = {action: 0 for action in ApplyAction}
        warning_count = 0
        observed_schemas: set[tuple[str, ...]] = set()
        contract_hash = _contract_hash(contract)

        if batch.schema_observation is not None:
            observation = batch.schema_observation
            if observation.contract_version != contract.version:
                raise ValueError("batch schema observation and contract versions must match")
            self._store.observe_schema(
                batch.source,
                contract,
                observation.actual_fields,
                observation.quality,
            )
            observed_schemas.add(observation.actual_fields)
            warning_count += sum(
                issue.severity is IssueSeverity.WARNING for issue in observation.quality.issues
            )

        for record in batch.records:
            schema_report = evaluate_schema(record.raw.payload, contract)
            quality = QualityReport(issues=(*record.quality.issues, *schema_report.issues))
            warning_count += sum(
                issue.severity is IssueSeverity.WARNING for issue in quality.issues
            )
            fields = tuple(sorted(record.raw.payload))
            if fields not in observed_schemas:
                self._store.observe_schema(batch.source, contract, fields, schema_report)
                observed_schemas.add(fields)
            context = processing_context_for(record, quality, contract_hash)
            action = self._store.apply_record(record, quality, context)
            counts[action] += 1

        summary = IngestionSummary(
            source=batch.source,
            status=batch.status,
            schema_hash=contract_hash,
            records_seen=len(batch.records),
            records_created=counts[ApplyAction.CREATED],
            records_updated=counts[ApplyAction.UPDATED],
            records_unchanged=counts[ApplyAction.UNCHANGED],
            records_quarantined=counts[ApplyAction.QUARANTINED],
            records_replayed=counts[ApplyAction.REPLAYED],
            warning_count=warning_count,
            status_detail=batch.status_detail,
        )
        self._store.save_summary(summary)
        return summary


def _schema_hash(field_names: tuple[str, ...]) -> str:
    return canonical_payload_hash({"fields": list(field_names)})


def _contract_hash(contract: SchemaContract) -> str:
    return canonical_payload_hash(
        {
            "source": contract.source,
            "version": contract.version,
            "required_fields": sorted(contract.required_fields),
            "optional_fields": sorted(contract.optional_fields),
            "allow_additional_fields": contract.allow_additional_fields,
        }
    )


def processing_context_for(
    record: MappedCompanyRecord,
    quality: QualityReport,
    contract_hash: str,
) -> ProcessingContext:
    """Fingerprint contract, mapper projection, and gate findings for safe reprocessing."""

    company_projection = (
        record.company.model_dump(mode="json") if record.company is not None else None
    )
    quality_projection = [issue.model_dump(mode="json") for issue in quality.issues]
    processing_hash = canonical_payload_hash(
        {
            "raw_payload_hash": record.raw.payload_hash,
            "contract_hash": contract_hash,
            "company_projection": company_projection,
            "quality_issues": quality_projection,
        }
    )
    return ProcessingContext(
        contract_hash=contract_hash,
        processing_hash=processing_hash,
    )


def _drift_status(report: QualityReport) -> str:
    if report.has_errors:
        return "breaking"
    if report.has_warnings:
        return "warning"
    return "expected"


__all__ = [
    "ApplyAction",
    "InMemoryIngestionStore",
    "IngestionService",
    "IngestionStore",
    "IngestionSummary",
    "ProcessingContext",
    "QuarantinedRecord",
    "SchemaObservation",
    "StoredRawRecord",
    "processing_context_for",
]
