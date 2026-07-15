"""Atomic PostgreSQL persistence for deterministic ingestion workflows."""

from __future__ import annotations

from typing import Any, Self
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from list_engine.adapters.postgres import PostgresCompanyRepository
from list_engine.ingestion.contracts import (
    MappedCompanyRecord,
    RawRecordEnvelope,
    SchemaContract,
    SourceReadStatus,
)
from list_engine.ingestion.quality import (
    IssueSeverity,
    QualityIssue,
    QualityReport,
    canonical_payload_hash,
)
from list_engine.ingestion.service import ApplyAction, IngestionSummary, ProcessingContext


class PostgresIngestionStore:
    """Persist one raw observation and its golden-record effect atomically.

    ``source_records`` is the content-addressed append-only evidence boundary. A
    separate processing ledger fingerprints the contract, mapper projection, and
    quality result: an exact processing replay is skipped, while a corrected mapper
    may safely revisit the same raw evidence and resolve its quarantine.
    """

    def __init__(self, connection: psycopg.Connection[dict[str, Any]]) -> None:
        self._connection = connection
        self._companies = PostgresCompanyRepository(connection)

    @classmethod
    def connect(cls, dsn: str) -> Self:
        connection = psycopg.connect(
            dsn,
            autocommit=True,
            prepare_threshold=None,
            row_factory=dict_row,
        )
        return cls(connection)

    def close(self) -> None:
        self._connection.close()

    def apply_record(
        self,
        record: MappedCompanyRecord,
        quality: QualityReport,
        context: ProcessingContext,
    ) -> ApplyAction:
        """Apply raw, quality, quarantine, and golden writes in one transaction."""

        with self._connection.transaction():
            source_record_id = self._lock_raw_record(record)
            if self._was_processed(source_record_id, context.processing_hash):
                return ApplyAction.REPLAYED

            self._record_issues(source_record_id, record.raw, quality)
            if quality.has_errors or record.company is None:
                self._quarantine(record.raw, quality)
                action = ApplyAction.QUARANTINED
                final_quality = quality
            else:
                try:
                    result = self._companies.upsert(record.company)
                except ValueError as error:
                    conflict = QualityIssue(
                        severity=IssueSeverity.ERROR,
                        code="golden_record_conflict",
                        message=str(error),
                    )
                    final_quality = QualityReport(issues=(*quality.issues, conflict))
                    self._record_issues(source_record_id, record.raw, final_quality)
                    self._quarantine(record.raw, final_quality)
                    action = ApplyAction.QUARANTINED
                else:
                    action = ApplyAction(result.action.value)
                    final_quality = quality
                    self._resolve_quarantine(record.raw)

            self._record_processing(
                source_record_id,
                record,
                final_quality,
                context,
                action,
            )
            return action

    def observe_schema(
        self,
        source: str,
        contract: SchemaContract,
        field_names: tuple[str, ...],
        report: QualityReport,
    ) -> None:
        if source != contract.source:
            raise ValueError("schema observation source and contract source must match")

        schema_hash = canonical_payload_hash({"fields": list(field_names)})
        if report.has_errors:
            drift_status = "breaking"
        elif report.has_warnings:
            drift_status = "warning"
        else:
            drift_status = "expected"

        with self._connection.transaction():
            self._connection.execute(
                """
                INSERT INTO list_engine.source_schema_snapshots (
                    source, schema_hash, field_names, required_fields, drift_status
                )
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (source, schema_hash) DO UPDATE
                SET field_names = EXCLUDED.field_names,
                    required_fields = EXCLUDED.required_fields,
                    drift_status = EXCLUDED.drift_status,
                    last_seen_at = now()
                """,
                (
                    source,
                    schema_hash,
                    Jsonb(list(field_names)),
                    Jsonb(sorted(contract.required_fields)),
                    drift_status,
                ),
            )

    def save_summary(self, summary: IngestionSummary) -> None:
        status = {
            SourceReadStatus.SUCCESS: "succeeded",
            SourceReadStatus.EMPTY_VERIFIED: "empty_verified",
            SourceReadStatus.FAILED: "failed",
        }[summary.status]

        if summary.status is SourceReadStatus.EMPTY_VERIFIED:
            if summary.status_detail is None:
                raise ValueError("verified-empty ingestion summaries require evidence")
            empty_proof: Jsonb | None = Jsonb({"status_detail": summary.status_detail})
        else:
            empty_proof = None

        if summary.status is SourceReadStatus.FAILED and summary.status_detail is None:
            raise ValueError("failed ingestion summaries require an error detail")

        with self._connection.transaction():
            self._connection.execute(
                """
                INSERT INTO list_engine.ingestion_runs (
                    source, mode, status, schema_hash, records_seen,
                    records_created, records_updated, records_unchanged,
                    records_quarantined, records_replayed, warning_count, empty_proof,
                    error_summary, completed_at
                )
                VALUES (
                    %s, 'batch', %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, now()
                )
                """,
                (
                    summary.source,
                    status,
                    summary.schema_hash,
                    summary.records_seen,
                    summary.records_created,
                    summary.records_updated,
                    summary.records_unchanged,
                    summary.records_quarantined,
                    summary.records_replayed,
                    summary.warning_count,
                    empty_proof,
                    summary.status_detail if summary.status is SourceReadStatus.FAILED else None,
                ),
            )

    def _lock_raw_record(self, record: MappedCompanyRecord) -> UUID:
        row = self._connection.execute(
            """
            INSERT INTO list_engine.source_records (
                source, source_record_id, piva, payload, payload_hash, observed_at
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (source, payload_hash) DO NOTHING
            RETURNING id
            """,
            (
                record.raw.source,
                record.raw.source_record_id,
                record.company.piva if record.company is not None else None,
                Jsonb(record.raw.payload),
                record.raw.payload_hash,
                record.raw.observed_at,
            ),
        ).fetchone()
        if row is not None:
            inserted_id = row["id"]
            if not isinstance(inserted_id, UUID):  # pragma: no cover - adapter contract
                raise RuntimeError("PostgreSQL returned a non-UUID source record ID")
            return inserted_id

        existing = self._connection.execute(
            """
            SELECT id
            FROM list_engine.source_records
            WHERE source = %s AND payload_hash = %s
            FOR UPDATE
            """,
            (record.raw.source, record.raw.payload_hash),
        ).fetchone()
        if existing is None:  # pragma: no cover - protected by the unique index
            raise RuntimeError("Raw replay guard conflicted without an existing record")
        existing_id = existing["id"]
        if not isinstance(existing_id, UUID):  # pragma: no cover - adapter contract
            raise RuntimeError("PostgreSQL returned a non-UUID source record ID")
        return existing_id

    def _was_processed(self, source_record_id: UUID, processing_hash: str) -> bool:
        row = self._connection.execute(
            """
            SELECT 1
            FROM list_engine.source_record_processing
            WHERE source_record_id = %s AND processing_hash = %s
            """,
            (source_record_id, processing_hash),
        ).fetchone()
        return row is not None

    def _record_issues(
        self,
        source_record_id: UUID,
        raw: RawRecordEnvelope,
        report: QualityReport,
    ) -> None:
        for issue in report.issues:
            details = {"record_index": issue.record_index} if issue.record_index is not None else {}
            self._connection.execute(
                """
                INSERT INTO list_engine.data_quality_issues (
                    source_record_id, source, record_key, severity, code,
                    field_name, message, details
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (
                    source_record_id,
                    raw.source,
                    raw.record_key,
                    issue.severity.value,
                    issue.code,
                    issue.field,
                    issue.message,
                    Jsonb(details),
                ),
            )

    def _quarantine(self, raw: RawRecordEnvelope, report: QualityReport) -> None:
        severity = "error" if report.has_errors else "warning"
        reasons = [issue.model_dump(mode="json") for issue in report.issues]
        self._connection.execute(
            """
            INSERT INTO list_engine.quarantine (
                source, payload, reasons, severity, record_key, payload_hash
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            (
                raw.source,
                Jsonb(raw.payload),
                Jsonb(reasons),
                severity,
                raw.record_key,
                raw.payload_hash,
            ),
        )
        self._connection.execute(
            """
            UPDATE list_engine.quarantine
            SET reasons = %s,
                severity = %s,
                status = 'pending',
                resolved_at = NULL
            WHERE source = %s AND payload_hash = %s
            """,
            (Jsonb(reasons), severity, raw.source, raw.payload_hash),
        )

    def _resolve_quarantine(self, raw: RawRecordEnvelope) -> None:
        self._connection.execute(
            """
            UPDATE list_engine.quarantine
            SET status = 'resolved', resolved_at = now()
            WHERE source = %s AND payload_hash = %s AND status = 'pending'
            """,
            (raw.source, raw.payload_hash),
        )

    def _record_processing(
        self,
        source_record_id: UUID,
        record: MappedCompanyRecord,
        quality: QualityReport,
        context: ProcessingContext,
        action: ApplyAction,
    ) -> None:
        if action is ApplyAction.REPLAYED:  # pragma: no cover - callers return earlier
            raise ValueError("Replay is not a new processing outcome")
        quality_projection = [issue.model_dump(mode="json") for issue in quality.issues]
        company_projection = (
            record.company.model_dump(mode="json") if record.company is not None else None
        )
        self._connection.execute(
            """
            INSERT INTO list_engine.source_record_processing (
                source_record_id, contract_hash, processing_hash, outcome,
                quality_report, company_projection
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                source_record_id,
                context.contract_hash,
                context.processing_hash,
                action.value,
                Jsonb(quality_projection),
                Jsonb(company_projection) if company_projection is not None else None,
            ),
        )


__all__ = ["PostgresIngestionStore"]
