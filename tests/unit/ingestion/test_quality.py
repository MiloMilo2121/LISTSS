from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from list_engine.ingestion import (
    IssueSeverity,
    RawBatchEnvelope,
    RawRecordEnvelope,
    SchemaContract,
    SourceReadStatus,
    canonical_payload_hash,
    evaluate_schema,
)


def raw_record(**overrides: object) -> RawRecordEnvelope:
    data: dict[str, object] = {
        "source": "demo_csv",
        "source_record_id": "row-001",
        "payload": {
            "piva": "99000000002",
            "legal_name": "Aurora Logistica Demo S.r.l.",
        },
        "observed_at": datetime(2026, 7, 1, tzinfo=UTC),
    }
    data.update(overrides)
    return RawRecordEnvelope.model_validate(data)


def test_canonical_payload_hash_is_order_independent_and_ascii_safe() -> None:
    first = {"name": "Caffè Demo", "nested": {"b": 2, "a": True}}
    reordered = {"nested": {"a": True, "b": 2}, "name": "Caffè Demo"}

    assert canonical_payload_hash(first) == canonical_payload_hash(reordered)
    assert canonical_payload_hash(first) == (
        "d12513c194ef67b7aaacd2cda96942e977cff6f36ebc41a7bb49b7e5da9c1a60"
    )
    assert canonical_payload_hash({"name": "Caffe Demo"}) != canonical_payload_hash(first)


@pytest.mark.parametrize(
    "payload",
    [
        ["not", "an", "object"],
        {"value": float("nan")},
        {"value": datetime(2026, 7, 1, tzinfo=UTC)},
        {1: "non-string key"},
    ],
)
def test_canonical_payload_hash_rejects_non_json_or_ambiguous_values(payload: object) -> None:
    with pytest.raises(ValueError):
        canonical_payload_hash(payload)


def test_raw_envelope_computes_replay_hash_without_mutating_payload() -> None:
    record = raw_record()

    assert record.payload_hash == canonical_payload_hash(record.payload)
    assert record.model_dump()["payload_hash"] == record.payload_hash
    assert record.payload["piva"] == "99000000002"


@pytest.mark.parametrize("source", ["CSV Import", "../csv", "café", "CSV"])
def test_source_identifier_rejects_unsafe_or_non_ascii_values(source: str) -> None:
    with pytest.raises(ValidationError):
        raw_record(source=source)


def test_schema_gate_reports_errors_and_warnings_in_deterministic_order() -> None:
    contract = SchemaContract(
        source="demo_csv",
        required_fields=frozenset({"piva", "legal_name"}),
        optional_fields=frozenset({"website"}),
        allow_additional_fields=True,
    )

    report = evaluate_schema(
        {"piva": "99000000002", "zzz_new": 1, "aaa_new": 2},
        contract,
        record_index=7,
    )

    assert [(issue.severity, issue.code, issue.field) for issue in report.issues] == [
        (IssueSeverity.ERROR, "missing_required_field", "legal_name"),
        (IssueSeverity.WARNING, "unexpected_field", "aaa_new"),
        (IssueSeverity.WARNING, "unexpected_field", "zzz_new"),
    ]
    assert report.has_errors
    assert report.has_warnings
    assert report.should_quarantine
    assert {issue.record_index for issue in report.issues} == {7}


def test_closed_schema_treats_unexpected_field_as_quarantine_error() -> None:
    contract = SchemaContract(
        source="demo_csv",
        required_fields=frozenset({"piva"}),
    )

    report = evaluate_schema({"piva": "99000000002", "new_column": "value"}, contract)

    assert [issue.severity for issue in report.issues] == [IssueSeverity.ERROR]
    assert report.should_quarantine


def test_batch_status_never_confuses_empty_with_failure() -> None:
    empty = RawBatchEnvelope(
        source="demo_csv",
        status=SourceReadStatus.EMPTY_VERIFIED,
        status_detail="source returned an explicit zero-row result",
    )
    failed = RawBatchEnvelope(
        source="demo_csv",
        status=SourceReadStatus.FAILED,
        status_detail="source schema could not be verified",
    )

    assert empty.status is SourceReadStatus.EMPTY_VERIFIED
    assert failed.status is SourceReadStatus.FAILED


def test_batch_terminal_state_invariants_fail_closed() -> None:
    with pytest.raises(ValidationError, match="successful source reads"):
        RawBatchEnvelope(source="demo_csv", status=SourceReadStatus.SUCCESS)
    with pytest.raises(ValidationError, match="require source evidence"):
        RawBatchEnvelope(source="demo_csv", status=SourceReadStatus.EMPTY_VERIFIED)
    with pytest.raises(ValidationError, match="require a failure detail"):
        RawBatchEnvelope(source="demo_csv", status=SourceReadStatus.FAILED)
    with pytest.raises(ValidationError, match="batch source"):
        RawBatchEnvelope(
            source="demo_csv",
            status=SourceReadStatus.SUCCESS,
            records=(raw_record(source="openapi_it"),),
        )


def test_contracts_are_strict_and_forbid_coercion_or_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        RawBatchEnvelope.model_validate(
            {
                "source": "demo_csv",
                "status": "success",
                "records": [raw_record()],
            }
        )
    with pytest.raises(ValidationError):
        raw_record(unexpected="schema drift")
