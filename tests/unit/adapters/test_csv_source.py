from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from list_engine.adapters.csv_source import (
    CSVContract,
    CSVHeaderDriftError,
    CSVSourceAdapter,
    parse_aware_datetime,
    parse_italian_decimal,
)
from list_engine.ingestion import IssueSeverity, SourceReadStatus


@pytest.fixture()
def contract() -> CSVContract:
    return CSVContract(
        source="partner_csv",
        header_mapping={
            "Partita IVA": "piva",
            "Ragione sociale": "legal_name",
            "Fatturato": "revenue_eur",
            "Addetti": "employees",
            "Osservato il": "source_observed_at",
            "Provincia": "province",
        },
    )


def test_parse_normalises_values_and_preserves_raw_row(contract: CSVContract) -> None:
    payload = (
        "Partita IVA;Ragione sociale;Fatturato;Addetti;Osservato il;Provincia\n"
        'IT 990.000.000-02;"  Aurora Demo S.r.l.  ";1.234.567,89;1.234;'
        "2026-07-15T10:30:00+02:00;mi\n"
    )

    result = CSVSourceAdapter(contract).parse_text(payload)

    assert result.issues == ()
    row = result.rows[0]
    assert row.values == {
        "source": "partner_csv",
        "piva": "99000000002",
        "legal_name": "Aurora Demo S.r.l.",
        "revenue_eur": Decimal("1234567.89"),
        "employees": 1234,
        "source_observed_at": datetime(2026, 7, 15, 8, 30, tzinfo=UTC),
        "province": "mi",
    }
    assert row.raw.values[1] == "  Aurora Demo S.r.l.  "
    assert row.raw.as_mapping()["Fatturato"] == "1.234.567,89"


def test_header_order_may_change_without_drift(contract: CSVContract) -> None:
    payload = (
        "Provincia;Osservato il;Addetti;Fatturato;Ragione sociale;Partita IVA\n"
        "MI;2026-07-15T08:30:00Z;28;6.500.000,00;Aurora Demo S.r.l.;99000000002\n"
    )
    result = CSVSourceAdapter(contract).parse_text(payload)
    assert result.issues == ()
    assert result.rows[0].values["employees"] == 28


@pytest.mark.parametrize(
    ("header", "missing", "unexpected", "duplicated"),
    [
        (
            "Partita IVA;Ragione sociale;Fatturato;Addetti;Osservato il\n",
            ("Provincia",),
            (),
            (),
        ),
        (
            "Partita IVA;Ragione sociale;Fatturato;Addetti;Osservato il;Provincia;PEC\n",
            (),
            ("PEC",),
            (),
        ),
        (
            "Partita IVA;Ragione sociale;Fatturato;Addetti;Osservato il;Provincia;Provincia\n",
            (),
            (),
            ("Provincia",),
        ),
    ],
)
def test_header_drift_is_exact(
    contract: CSVContract,
    header: str,
    missing: tuple[str, ...],
    unexpected: tuple[str, ...],
    duplicated: tuple[str, ...],
) -> None:
    with pytest.raises(CSVHeaderDriftError) as caught:
        CSVSourceAdapter(contract).parse_text(header)
    assert caught.value.missing == missing
    assert caught.value.unexpected == unexpected
    assert caught.value.duplicated == duplicated


def test_invalid_rows_are_quarantinable_with_raw_values(contract: CSVContract) -> None:
    payload = (
        "Partita IVA;Ragione sociale;Fatturato;Addetti;Osservato il;Provincia\n"
        "99000000003;Checksum errato;1.000,00;12;2026-07-15T10:00:00+02:00;MI\n"
        "99000000002;Timestamp naive;1.000,00;12;2026-07-15T10:00:00;MI\n"
        "99000000010;Valido;2.500,00;18;2026-07-15T10:00:00+02:00;BG\n"
    )
    result = CSVSourceAdapter(contract).parse_text(payload)
    assert len(result.rows) == 1
    assert [issue.field_name for issue in result.issues] == ["piva", "source_observed_at"]
    assert result.issues[0].raw.as_mapping()["Partita IVA"] == "99000000003"
    assert result.issues[1].raw.values[1] == "Timestamp naive"


def test_wrong_cell_count_preserves_all_cells(contract: CSVContract) -> None:
    payload = (
        "Partita IVA;Ragione sociale;Fatturato;Addetti;Osservato il;Provincia\n"
        "99000000002;Aurora Demo S.r.l.;1.000,00\n"
    )
    result = CSVSourceAdapter(contract).parse_text(payload)
    assert result.rows == ()
    assert result.issues[0].code == "row_shape"
    assert result.issues[0].raw.values == (
        "99000000002",
        "Aurora Demo S.r.l.",
        "1.000,00",
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", None),
        ("1234", Decimal("1234")),
        ("1.234", Decimal("1234")),
        ("1.234,56", Decimal("1234.56")),
        ("-12,50", Decimal("-12.50")),
    ],
)
def test_italian_decimal_parser(raw: str, expected: Decimal | None) -> None:
    assert parse_italian_decimal(raw) == expected


@pytest.mark.parametrize("raw", ["1,234.56", "12.34", "1.23,00", "EUR 1.000,00"])
def test_italian_decimal_parser_rejects_ambiguous_formats(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_italian_decimal(raw)


def test_datetime_requires_timezone_and_normalises_to_utc() -> None:
    assert parse_aware_datetime("2026-07-15T10:30:00+02:00") == datetime(
        2026, 7, 15, 8, 30, tzinfo=UTC
    )
    with pytest.raises(ValueError, match="timezone"):
        parse_aware_datetime("2026-07-15T10:30:00")


def test_contract_rejects_duplicate_and_trusted_targets() -> None:
    with pytest.raises(ValueError, match="mapped more than once"):
        CSVContract(
            source="csv",
            header_mapping={
                "PIVA": "piva",
                "VAT": "piva",
                "Name": "legal_name",
                "Observed": "source_observed_at",
            },
        )
    with pytest.raises(ValueError, match="trusted fields"):
        CSVContract(
            source="csv",
            header_mapping={
                "PIVA": "piva",
                "Name": "legal_name",
                "Observed": "source_observed_at",
                "Demo": "is_demo",
            },
        )


def test_read_text_maps_valid_row_to_shared_contract(contract: CSVContract) -> None:
    payload = (
        "Partita IVA;Ragione sociale;Fatturato;Addetti;Osservato il;Provincia\n"
        "99000000002;Aurora Demo S.r.l.;6.500.000,00;28;"
        "2026-07-15T10:30:00+02:00;mi\n"
    )

    batch = CSVSourceAdapter(contract).read_text(payload, is_demo=True)

    assert batch.status is SourceReadStatus.SUCCESS
    mapped = batch.records[0]
    assert mapped.company is not None
    assert mapped.company.is_demo is True
    assert mapped.company.province == "MI"
    assert mapped.raw.observed_at == mapped.company.source_observed_at
    assert mapped.raw.payload["Fatturato"] == "6.500.000,00"
    assert mapped.quality.issues == ()


def test_read_text_header_drift_is_failed_not_empty(contract: CSVContract) -> None:
    batch = CSVSourceAdapter(contract).read_text(
        "Partita IVA;Ragione sociale;Osservato il\n", is_demo=True
    )

    assert batch.status is SourceReadStatus.FAILED
    assert batch.records == ()
    assert batch.status_detail is not None
    assert "missing=" in batch.status_detail
    assert batch.schema_observation is not None
    assert batch.schema_observation.actual_fields == (
        "Osservato il",
        "Partita IVA",
        "Ragione sociale",
    )
    assert batch.schema_observation.quality.has_errors
    assert {issue.code for issue in batch.schema_observation.quality.issues} == {
        "missing_required_field"
    }
    assert all(
        issue.severity is IssueSeverity.ERROR for issue in batch.schema_observation.quality.issues
    )


def test_read_text_header_drift_reports_missing_unexpected_and_duplicate(
    contract: CSVContract,
) -> None:
    batch = CSVSourceAdapter(contract).read_text(
        "Partita IVA;Ragione sociale;Addetti;Osservato il;Provincia;PEC;Provincia\n",
        is_demo=True,
    )

    assert batch.status is SourceReadStatus.FAILED
    assert batch.schema_observation is not None
    assert batch.schema_observation.actual_fields == (
        "Addetti",
        "Osservato il",
        "PEC",
        "Partita IVA",
        "Provincia",
        "Ragione sociale",
    )
    assert {issue.code for issue in batch.schema_observation.quality.issues} == {
        "duplicate_header_field",
        "missing_required_field",
        "unexpected_field",
    }
    assert all(
        issue.severity is IssueSeverity.ERROR for issue in batch.schema_observation.quality.issues
    )


def test_read_text_without_header_is_failed_with_empty_schema_observation(
    contract: CSVContract,
) -> None:
    batch = CSVSourceAdapter(contract).read_text("", is_demo=True)

    assert batch.status is SourceReadStatus.FAILED
    assert batch.records == ()
    assert batch.schema_observation is not None
    assert batch.schema_observation.actual_fields == ()
    assert batch.schema_observation.quality.has_errors
    assert {issue.field for issue in batch.schema_observation.quality.issues} == set(
        contract.header_mapping
    )


def test_read_text_header_only_is_verified_empty_with_evidence(contract: CSVContract) -> None:
    batch = CSVSourceAdapter(contract).read_text(
        "Partita IVA;Ragione sociale;Fatturato;Addetti;Osservato il;Provincia\n",
        is_demo=True,
    )

    assert batch.status is SourceReadStatus.EMPTY_VERIFIED
    assert batch.records == ()
    assert batch.status_detail is not None
    assert "matched all 6 contracted fields" in batch.status_detail
    assert batch.schema_observation is not None
    assert batch.schema_observation.quality.issues == ()
    assert batch.schema_observation.actual_fields == tuple(sorted(contract.header_mapping))


def test_read_text_keeps_invalid_rows_as_quarantine_records(contract: CSVContract) -> None:
    payload = (
        "Partita IVA;Ragione sociale;Fatturato;Addetti;Osservato il;Provincia\n"
        "99000000003;Checksum errato;1.000,00;12;2026-07-15T10:00:00+02:00;MI\n"
        "99000000002;Ricavi negativi;-1,00;12;2026-07-15T10:00:00+02:00;MI\n"
    )

    batch = CSVSourceAdapter(contract).read_text(payload, is_demo=True)

    assert batch.status is SourceReadStatus.SUCCESS
    assert len(batch.records) == 2
    assert all(record.company is None for record in batch.records)
    assert all(record.quality.should_quarantine for record in batch.records)
    assert [record.quality.issues[0].code for record in batch.records] == [
        "invalid_field",
        "invalid_company",
    ]
    assert all(record.quality.issues[0].severity is IssueSeverity.ERROR for record in batch.records)
    assert batch.records[0].raw.payload["Partita IVA"] == "99000000003"
    assert batch.records[1].raw.payload["Fatturato"] == "-1,00"
