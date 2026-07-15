from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from list_engine.core.models import Company
from list_engine.core.repository import (
    InMemoryCompanyRepository,
    UpsertAction,
    deduplicate_companies,
    merge_companies,
)

OBSERVED_AT = datetime(2026, 7, 1, tzinfo=UTC)
OBSERVED_LATER = OBSERVED_AT + timedelta(days=1)


def company(**overrides: object) -> Company:
    data: dict[str, object] = {
        "piva": "99000000002",
        "legal_name": "Aurora Logistica Demo S.r.l.",
        "source": "demo_seed",
        "source_observed_at": OBSERVED_AT,
        "is_demo": True,
    }
    data.update(overrides)
    return Company.model_validate(data)


def test_company_model_normalises_identity_and_province() -> None:
    record = company(piva="IT 990.000.000-02", province="mi")

    assert record.piva == "99000000002"
    assert record.province == "MI"


def test_repository_upsert_is_idempotent() -> None:
    repository = InMemoryCompanyRepository()
    record = company()

    assert repository.upsert(record).action is UpsertAction.CREATED
    assert repository.upsert(record).action is UpsertAction.UNCHANGED
    assert repository.get("IT 990.000.000-02") == record
    assert repository.list_all() == (record,)


def test_repository_upsert_updates_without_erasing_known_fields() -> None:
    repository = InMemoryCompanyRepository(
        [company(website="https://aurora-logistica.example.invalid", employees=20)]
    )

    result = repository.upsert(
        company(
            source="openapi_it",
            source_observed_at=OBSERVED_LATER,
            employees=28,
            revenue_eur=Decimal("6500000"),
        )
    )

    assert result.action is UpsertAction.UPDATED
    assert result.company.website == "https://aurora-logistica.example.invalid"
    assert result.company.employees == 28
    assert result.company.revenue_eur == Decimal("6500000")
    assert result.company.source == "openapi_it"


@pytest.mark.parametrize(("existing_demo", "incoming_demo"), [(True, False), (False, True)])
def test_merge_rejects_cross_mode_identity_collision(
    existing_demo: bool, incoming_demo: bool
) -> None:
    with pytest.raises(ValueError, match="DEMO and production"):
        merge_companies(
            company(is_demo=existing_demo),
            company(is_demo=incoming_demo, source="openapi_it"),
        )


def test_stale_replay_cannot_overwrite_newer_golden_record() -> None:
    current = company(source="openapi_it", source_observed_at=OBSERVED_LATER, employees=28)
    stale = company(source="csv", source_observed_at=OBSERVED_AT, employees=10)

    assert merge_companies(current, stale) == current


def test_equal_time_conflicting_claims_fail_closed() -> None:
    with pytest.raises(ValueError, match="employees"):
        merge_companies(company(employees=28), company(source="csv", employees=10))


def test_equal_time_firmographic_fill_fails_closed_for_scd_consistency() -> None:
    with pytest.raises(ValueError, match="revenue_eur"):
        merge_companies(
            company(revenue_eur=None),
            company(source="csv", revenue_eur=Decimal("6500000")),
        )


def test_equal_time_complementary_claims_merge_deterministically() -> None:
    first = company(source="openapi_it", city="Milano")
    second = company(source="csv", website="https://aurora-logistica.example.invalid")
    forward = merge_companies(first, second)
    reverse = merge_companies(second, first)

    assert forward == reverse
    assert forward.city == "Milano"
    assert forward.website == "https://aurora-logistica.example.invalid"
    assert forward.source == "csv"


def test_company_requires_aware_source_timestamp() -> None:
    with pytest.raises(ValidationError):
        company(source_observed_at=datetime(2026, 7, 1))


def test_company_forbids_unknown_source_fields() -> None:
    with pytest.raises(ValidationError):
        company(unexpected="schema drift")


def test_merge_rejects_different_canonical_identities() -> None:
    other = company(piva="99000000010")

    try:
        merge_companies(company(), other)
    except ValueError as error:
        assert "different P.IVA" in str(error)
    else:
        raise AssertionError("Expected merge to reject different identities")


def test_deduplicate_preserves_first_seen_order_and_merges_later_evidence() -> None:
    second = company(piva="99000000010", legal_name="Boreale Demo S.r.l.")
    merged = deduplicate_companies(
        [
            company(employees=20),
            second,
            company(employees=28, source_observed_at=OBSERVED_LATER),
        ]
    )

    assert [item.piva for item in merged] == ["99000000002", "99000000010"]
    assert merged[0].employees == 28
