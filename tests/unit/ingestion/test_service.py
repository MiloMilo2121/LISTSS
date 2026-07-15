from __future__ import annotations

from datetime import UTC, datetime, timedelta

from list_engine.core.models import Company
from list_engine.ingestion.contracts import (
    MappedCompanyRecord,
    MappedSourceBatch,
    RawRecordEnvelope,
    SchemaContract,
    SourceReadStatus,
    SourceSchemaObservation,
)
from list_engine.ingestion.quality import IssueSeverity, QualityIssue, QualityReport
from list_engine.ingestion.service import IngestionService, InMemoryIngestionStore

SOURCE = "demo_csv"
PIVA = "99000000002"
OBSERVED_AT = datetime(2026, 7, 15, 8, 30, tzinfo=UTC)
OBSERVED_LATER = OBSERVED_AT + timedelta(days=1)


def schema_contract(*, allow_additional_fields: bool = False) -> SchemaContract:
    return SchemaContract(
        source=SOURCE,
        required_fields=frozenset({"piva", "legal_name"}),
        optional_fields=frozenset({"employees"}),
        allow_additional_fields=allow_additional_fields,
    )


def mapped_record(
    *,
    source_record_id: str = "row-001",
    observed_at: datetime = OBSERVED_AT,
    employees: int | None = 28,
    raw_payload: dict[str, object] | None = None,
    company: Company | None = None,
    quality: QualityReport | None = None,
) -> MappedCompanyRecord:
    payload = raw_payload or {
        "piva": PIVA,
        "legal_name": "Aurora Logistica Demo S.r.l.",
        "employees": employees,
    }
    raw = RawRecordEnvelope(
        source=SOURCE,
        source_record_id=source_record_id,
        payload=payload,
        observed_at=observed_at,
    )
    if company is None and quality is None:
        company = Company(
            piva=PIVA,
            legal_name="Aurora Logistica Demo S.r.l.",
            employees=employees,
            source=SOURCE,
            is_demo=True,
            source_observed_at=observed_at,
        )
    return MappedCompanyRecord(
        raw=raw,
        company=company,
        quality=quality or QualityReport(),
    )


def success_batch(*records: MappedCompanyRecord) -> MappedSourceBatch:
    return MappedSourceBatch(
        source=SOURCE,
        status=SourceReadStatus.SUCCESS,
        records=records,
    )


def test_valid_record_creates_golden_record_and_auditable_raw_observation() -> None:
    store = InMemoryIngestionStore()
    summary = IngestionService(store).ingest(success_batch(mapped_record()), schema_contract())

    assert summary.records_seen == 1
    assert summary.records_created == 1
    assert summary.records_updated == 0
    assert summary.records_quarantined == 0
    assert summary.warning_count == 0
    assert store.companies.get(PIVA) is not None
    assert len(store.raw_records) == 1
    assert len(store.schema_observations) == 1
    assert next(iter(store.schema_observations.values())).drift_status == "expected"
    assert store.runs == [summary]


def test_exact_replay_is_counted_without_duplicate_raw_or_golden_records() -> None:
    store = InMemoryIngestionStore()
    service = IngestionService(store)
    batch = success_batch(mapped_record())

    first = service.ingest(batch, schema_contract())
    replay = service.ingest(batch, schema_contract())

    assert first.records_created == 1
    assert replay.records_replayed == 1
    assert replay.records_created == 0
    assert replay.records_unchanged == 0
    assert len(store.raw_records) == 1
    assert len(store.companies.list_all()) == 1
    assert store.runs == [first, replay]


def test_newer_changed_observation_updates_golden_record_and_keeps_both_raws() -> None:
    store = InMemoryIngestionStore()
    service = IngestionService(store)
    service.ingest(success_batch(mapped_record(employees=28)), schema_contract())

    summary = service.ingest(
        success_batch(mapped_record(observed_at=OBSERVED_LATER, employees=35)),
        schema_contract(),
    )

    assert summary.records_updated == 1
    assert summary.records_created == 0
    assert summary.records_quarantined == 0
    assert len(store.raw_records) == 2
    golden = store.companies.get(PIVA)
    assert golden is not None
    assert golden.employees == 35
    assert golden.source_observed_at == OBSERVED_LATER


def test_schema_and_mapper_errors_quarantine_raws_without_writing_golden_records() -> None:
    schema_invalid = mapped_record(
        source_record_id="row-schema-invalid",
        raw_payload={"piva": PIVA},
    )
    mapper_invalid = mapped_record(
        source_record_id="row-mapper-invalid",
        raw_payload={
            "piva": "99000000003",
            "legal_name": "Checksum errato",
        },
        company=None,
        quality=QualityReport(
            issues=(
                QualityIssue(
                    severity=IssueSeverity.ERROR,
                    code="invalid_company",
                    message="P.IVA checksum is invalid",
                    field="piva",
                ),
            )
        ),
    )
    store = InMemoryIngestionStore()

    summary = IngestionService(store).ingest(
        success_batch(schema_invalid, mapper_invalid), schema_contract()
    )

    assert summary.records_seen == 2
    assert summary.records_quarantined == 2
    assert summary.records_created == 0
    assert len(store.raw_records) == 2
    assert len(store.quarantine) == 2
    assert store.companies.list_all() == ()
    assert {issue.code for issue in store.quality_issues.values()} == {
        "invalid_company",
        "missing_required_field",
    }


def test_unexpected_field_warning_is_audited_but_does_not_block_upsert() -> None:
    store = InMemoryIngestionStore()
    record = mapped_record(
        raw_payload={
            "piva": PIVA,
            "legal_name": "Aurora Logistica Demo S.r.l.",
            "employees": 28,
            "new_provider_field": "preserved raw evidence",
        }
    )

    summary = IngestionService(store).ingest(
        success_batch(record), schema_contract(allow_additional_fields=True)
    )

    assert summary.records_created == 1
    assert summary.records_quarantined == 0
    assert summary.warning_count == 1
    assert [issue.code for issue in store.quality_issues.values()] == ["unexpected_field"]
    assert next(iter(store.schema_observations.values())).drift_status == "warning"
    assert store.companies.get(PIVA) is not None


def test_stale_observation_is_kept_as_raw_without_overwriting_the_golden_record() -> None:
    store = InMemoryIngestionStore()
    service = IngestionService(store)
    service.ingest(
        success_batch(mapped_record(observed_at=OBSERVED_LATER, employees=35)),
        schema_contract(),
    )

    summary = service.ingest(
        success_batch(mapped_record(observed_at=OBSERVED_AT, employees=12)),
        schema_contract(),
    )

    assert summary.records_unchanged == 1
    assert summary.records_quarantined == 0
    assert len(store.raw_records) == 2
    assert not any(
        issue.code == "golden_record_conflict" for issue in store.quality_issues.values()
    )
    golden = store.companies.get(PIVA)
    assert golden is not None
    assert golden.employees == 35


def test_equal_time_conflicting_observation_is_quarantined() -> None:
    store = InMemoryIngestionStore()
    service = IngestionService(store)
    service.ingest(success_batch(mapped_record(employees=28)), schema_contract())

    summary = service.ingest(success_batch(mapped_record(employees=12)), schema_contract())

    assert summary.records_quarantined == 1
    assert summary.records_unchanged == 0
    assert any(issue.code == "golden_record_conflict" for issue in store.quality_issues.values())
    golden = store.companies.get(PIVA)
    assert golden is not None
    assert golden.employees == 28


def test_quarantined_raw_can_be_reprocessed_after_mapper_fix() -> None:
    store = InMemoryIngestionStore()
    service = IngestionService(store)
    payload: dict[str, object] = {
        "piva": PIVA,
        "legal_name": "Aurora Logistica Demo S.r.l.",
        "employees": 28,
    }
    mapper_error = QualityReport(
        issues=(
            QualityIssue(
                severity=IssueSeverity.ERROR,
                code="mapper_contract_error",
                message="old mapper rejected a valid projection",
            ),
        )
    )
    rejected = mapped_record(raw_payload=payload, company=None, quality=mapper_error)

    first = service.ingest(success_batch(rejected), schema_contract())

    assert first.records_quarantined == 1
    assert store.companies.get(PIVA) is None

    repaired = mapped_record(raw_payload=payload)
    second = service.ingest(success_batch(repaired), schema_contract())

    assert second.records_created == 1
    assert second.records_replayed == 0
    assert len(store.raw_records) == 1
    assert len(store.processing) == 2
    assert next(iter(store.quarantine.values())).status == "resolved"
    assert store.companies.get(PIVA) is not None


def test_verified_empty_and_failed_reads_have_distinct_persisted_summaries() -> None:
    store = InMemoryIngestionStore()
    service = IngestionService(store)
    contract = schema_contract()

    empty = service.ingest(
        MappedSourceBatch(
            source=SOURCE,
            status=SourceReadStatus.EMPTY_VERIFIED,
            status_detail="provider explicitly returned zero records",
        ),
        contract,
    )
    failed = service.ingest(
        MappedSourceBatch(
            source=SOURCE,
            status=SourceReadStatus.FAILED,
            status_detail="provider schema could not be verified",
        ),
        contract,
    )

    assert empty.status is SourceReadStatus.EMPTY_VERIFIED
    assert failed.status is SourceReadStatus.FAILED
    assert empty.status_detail != failed.status_detail
    assert empty.records_seen == failed.records_seen == 0
    assert store.runs == [empty, failed]
    assert store.schema_observations == {}


def test_failed_schema_drift_is_persisted_even_without_records() -> None:
    store = InMemoryIngestionStore()
    service = IngestionService(store)
    drift = QualityReport(
        issues=(
            QualityIssue(
                severity=IssueSeverity.ERROR,
                code="missing_required_field",
                message="required header legal_name disappeared",
                field="legal_name",
            ),
        )
    )
    batch = MappedSourceBatch(
        source=SOURCE,
        status=SourceReadStatus.FAILED,
        status_detail="source header drifted",
        schema_observation=SourceSchemaObservation(
            source=SOURCE,
            contract_version=1,
            actual_fields=("piva",),
            quality=drift,
        ),
    )

    summary = service.ingest(batch, schema_contract())

    assert summary.status is SourceReadStatus.FAILED
    assert len(store.schema_observations) == 1
    observation = next(iter(store.schema_observations.values()))
    assert observation.field_names == ("piva",)
    assert observation.drift_status == "breaking"
