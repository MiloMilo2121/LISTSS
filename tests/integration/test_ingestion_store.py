from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg
import pytest
from psycopg.rows import dict_row

from list_engine.core.models import Company
from list_engine.ingestion.contracts import (
    MappedCompanyRecord,
    RawRecordEnvelope,
    SchemaContract,
    SourceReadStatus,
)
from list_engine.ingestion.postgres_store import PostgresIngestionStore
from list_engine.ingestion.quality import IssueSeverity, QualityIssue, QualityReport
from list_engine.ingestion.service import ApplyAction, IngestionSummary, ProcessingContext

ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.getenv("TEST_DATABASE_URL")
BASE_TIME = datetime(2026, 7, 15, 8, tzinfo=UTC)
PROCESSING = ProcessingContext(contract_hash="a" * 64, processing_hash="b" * 64)

pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="TEST_DATABASE_URL is required for disposable PostgreSQL integration tests",
)


@pytest.fixture()
def database() -> Iterator[psycopg.Connection[dict[str, Any]]]:
    assert DATABASE_URL is not None
    connection = psycopg.connect(DATABASE_URL, autocommit=True, row_factory=dict_row)
    if "test" not in connection.info.dbname.lower():
        connection.close()
        pytest.fail("Refusing to reset a database whose name does not contain 'test'")

    connection.execute("DROP SCHEMA IF EXISTS list_engine CASCADE")
    for migration_name in (
        "202607150001_initial.sql",
        "202607150002_ingestion.sql",
    ):
        migration = (ROOT / "supabase" / "migrations" / migration_name).read_text()
        connection.execute(migration, prepare=False)
        connection.execute(migration, prepare=False)

    yield connection
    connection.rollback()
    connection.execute("DROP SCHEMA IF EXISTS list_engine CASCADE")
    connection.close()


def mapped_company_record(
    *,
    source: str = "fixture",
    source_record_id: str = "row-1",
    legal_name: str = "Atomic Fixture S.r.l.",
    observed_at: datetime | None = None,
) -> MappedCompanyRecord:
    timestamp = observed_at or BASE_TIME
    payload: dict[str, object] = {
        "piva": "99000000002",
        "legal_name": legal_name,
    }
    return MappedCompanyRecord(
        raw=RawRecordEnvelope(
            source=source,
            source_record_id=source_record_id,
            payload=payload,
            observed_at=timestamp,
        ),
        company=Company(
            piva="99000000002",
            legal_name=legal_name,
            source=source,
            source_observed_at=timestamp,
        ),
    )


def test_apply_record_creates_replays_and_updates_atomically(
    database: psycopg.Connection[dict[str, Any]],
) -> None:
    store = PostgresIngestionStore(database)
    created = mapped_company_record()

    assert store.apply_record(created, created.quality, PROCESSING) is ApplyAction.CREATED
    assert store.apply_record(created, created.quality, PROCESSING) is ApplyAction.REPLAYED

    updated = mapped_company_record(
        legal_name="Atomic Fixture Italia S.r.l.",
        observed_at=BASE_TIME + timedelta(hours=1),
    )
    assert store.apply_record(updated, updated.quality, PROCESSING) is ApplyAction.UPDATED

    assert database.execute(
        "SELECT legal_name FROM list_engine.companies WHERE piva = '99000000002'"
    ).fetchone() == {"legal_name": "Atomic Fixture Italia S.r.l."}
    assert database.execute(
        "SELECT count(*) FROM list_engine.source_records WHERE source = 'fixture'"
    ).fetchone() == {"count": 2}


def test_quality_error_keeps_raw_evidence_and_quarantines_once(
    database: psycopg.Connection[dict[str, Any]],
) -> None:
    store = PostgresIngestionStore(database)
    issue = QualityIssue(
        severity=IssueSeverity.ERROR,
        code="invalid_piva",
        message="P.IVA checksum is invalid",
        field="piva",
        record_index=3,
    )
    record = MappedCompanyRecord(
        raw=RawRecordEnvelope(
            source="fixture",
            source_record_id="bad-row-1",
            payload={"piva": "invalid", "legal_name": "Rejected Fixture"},
            observed_at=datetime(2026, 7, 15, 9, tzinfo=UTC),
        ),
        quality=QualityReport(issues=(issue,)),
    )

    assert store.apply_record(record, record.quality, PROCESSING) is ApplyAction.QUARANTINED
    assert store.apply_record(record, record.quality, PROCESSING) is ApplyAction.REPLAYED

    assert database.execute(
        "SELECT count(*) FROM list_engine.source_records WHERE source_record_id = 'bad-row-1'"
    ).fetchone() == {"count": 1}
    assert database.execute(
        "SELECT count(*) FROM list_engine.quarantine WHERE record_key = 'bad-row-1'"
    ).fetchone() == {"count": 1}
    assert database.execute(
        """
        SELECT code, severity, details
        FROM list_engine.data_quality_issues
        WHERE record_key = 'bad-row-1'
        """
    ).fetchone() == {
        "code": "invalid_piva",
        "severity": "error",
        "details": {"record_index": 3},
    }


def test_fixed_mapping_reprocesses_same_raw_and_resolves_quarantine(
    database: psycopg.Connection[dict[str, Any]],
) -> None:
    store = PostgresIngestionStore(database)
    raw = RawRecordEnvelope(
        source="fixture",
        source_record_id="repairable-row",
        payload={"piva": "99000000002", "legal_name": "Repairable Fixture S.r.l."},
        observed_at=BASE_TIME,
    )
    mapping_issue = QualityReport(
        issues=(
            QualityIssue(
                severity=IssueSeverity.ERROR,
                code="mapper_contract_error",
                message="old mapper could not promote this record",
            ),
        )
    )
    rejected = MappedCompanyRecord(raw=raw, quality=mapping_issue)
    first_context = ProcessingContext(
        contract_hash="a" * 64,
        processing_hash="c" * 64,
    )

    assert store.apply_record(rejected, rejected.quality, first_context) is ApplyAction.QUARANTINED

    repaired = MappedCompanyRecord(
        raw=raw,
        company=Company(
            piva="99000000002",
            legal_name="Repairable Fixture S.r.l.",
            source="fixture",
            source_observed_at=BASE_TIME,
        ),
    )
    repaired_context = ProcessingContext(
        contract_hash="a" * 64,
        processing_hash="d" * 64,
    )
    assert store.apply_record(repaired, repaired.quality, repaired_context) is ApplyAction.CREATED

    assert database.execute(
        "SELECT count(*) FROM list_engine.source_records WHERE source = 'fixture'"
    ).fetchone() == {"count": 1}
    assert database.execute(
        "SELECT count(*) FROM list_engine.source_record_processing"
    ).fetchone() == {"count": 2}
    assert database.execute(
        "SELECT status FROM list_engine.quarantine WHERE record_key = 'repairable-row'"
    ).fetchone() == {"status": "resolved"}
    assert database.execute(
        "SELECT count(*) FROM list_engine.companies WHERE piva = '99000000002'"
    ).fetchone() == {"count": 1}


def test_database_failure_rolls_back_raw_and_golden_writes(
    database: psycopg.Connection[dict[str, Any]],
) -> None:
    database.execute(
        """
        CREATE OR REPLACE FUNCTION list_engine.reject_forced_company()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION 'forced company failure';
        END;
        $$
        """
    )
    database.execute(
        """
        CREATE TRIGGER reject_forced_company
        BEFORE INSERT ON list_engine.companies
        FOR EACH ROW WHEN (NEW.source = 'force_failure')
        EXECUTE FUNCTION list_engine.reject_forced_company()
        """
    )
    store = PostgresIngestionStore(database)
    record = mapped_company_record(source="force_failure", source_record_id="forced-1")

    with pytest.raises(psycopg.errors.RaiseException, match="forced company failure"):
        store.apply_record(record, record.quality, PROCESSING)

    assert database.execute(
        "SELECT count(*) FROM list_engine.source_records WHERE source = 'force_failure'"
    ).fetchone() == {"count": 0}
    assert database.execute(
        "SELECT count(*) FROM list_engine.companies WHERE piva = '99000000002'"
    ).fetchone() == {"count": 0}


def test_schema_observation_is_upserted_with_explicit_drift_status(
    database: psycopg.Connection[dict[str, Any]],
) -> None:
    store = PostgresIngestionStore(database)
    contract = SchemaContract(
        source="fixture",
        required_fields=frozenset({"piva", "legal_name"}),
        optional_fields=frozenset({"employees"}),
    )
    fields = ("legal_name", "piva")

    store.observe_schema("fixture", contract, fields, QualityReport())
    store.observe_schema("fixture", contract, fields, QualityReport())

    assert database.execute(
        """
        SELECT field_names, required_fields, drift_status
        FROM list_engine.source_schema_snapshots
        WHERE source = 'fixture'
        """
    ).fetchone() == {
        "field_names": ["legal_name", "piva"],
        "required_fields": ["legal_name", "piva"],
        "drift_status": "expected",
    }
    assert database.execute(
        "SELECT count(*) FROM list_engine.source_schema_snapshots WHERE source = 'fixture'"
    ).fetchone() == {"count": 1}


def test_run_summaries_preserve_success_and_verified_empty_semantics(
    database: psycopg.Connection[dict[str, Any]],
) -> None:
    store = PostgresIngestionStore(database)
    store.save_summary(
        IngestionSummary(
            source="fixture",
            status=SourceReadStatus.SUCCESS,
            schema_hash="a" * 64,
            records_seen=1,
            records_created=1,
            records_updated=0,
            records_unchanged=0,
            records_quarantined=0,
            records_replayed=0,
            warning_count=0,
        )
    )
    store.save_summary(
        IngestionSummary(
            source="fixture",
            status=SourceReadStatus.EMPTY_VERIFIED,
            schema_hash="a" * 64,
            records_seen=0,
            records_created=0,
            records_updated=0,
            records_unchanged=0,
            records_quarantined=0,
            records_replayed=0,
            warning_count=0,
            status_detail="HTTP 200 with a valid empty result envelope",
        )
    )

    assert database.execute(
        """
        SELECT status, warning_count, empty_proof, error_summary
        FROM list_engine.ingestion_runs
        WHERE source = 'fixture'
        ORDER BY started_at, id
        """
    ).fetchall() == [
        {
            "status": "succeeded",
            "warning_count": 0,
            "empty_proof": None,
            "error_summary": None,
        },
        {
            "status": "empty_verified",
            "warning_count": 0,
            "empty_proof": {"status_detail": "HTTP 200 with a valid empty result envelope"},
            "error_summary": None,
        },
    ]
