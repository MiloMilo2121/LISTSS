"""Repository ports plus a production-shaped in-memory DEMO implementation."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from list_engine.core.models import Company
from list_engine.core.piva import normalize_piva


class UpsertAction(StrEnum):
    CREATED = "created"
    UPDATED = "updated"
    UNCHANGED = "unchanged"


class UpsertResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    company: Company
    action: UpsertAction


class CompanyRepository(Protocol):
    """Port used by workflows; concrete storage is selected from environment config."""

    def get(self, piva: str) -> Company | None: ...

    def upsert(self, company: Company) -> UpsertResult: ...

    def list_all(self) -> tuple[Company, ...]: ...


def merge_companies(existing: Company, incoming: Company) -> Company:
    """Pure, monotonic golden-record merge.

    A provider may omit fields it does not know; those ``None`` values never erase an
    existing verified value. A stale replay cannot overwrite a newer snapshot. Equal-
    time conflicting claims are rejected for quarantine rather than resolved by ingest
    order. Field-level provenance is persisted separately in ``enrichment_log`` by
    infrastructure adapters.
    """

    if existing.piva != incoming.piva:
        raise ValueError("Cannot merge companies with different P.IVA values")
    if existing.is_demo != incoming.is_demo:
        raise ValueError("Cannot merge records across DEMO and production boundaries")
    if incoming.source_observed_at < existing.source_observed_at:
        return existing

    incoming_data = incoming.model_dump()
    merged = existing.model_dump()
    mergeable_fields = set(Company.model_fields) - {
        "piva",
        "source",
        "is_demo",
        "source_observed_at",
    }

    if incoming.source_observed_at == existing.source_observed_at:
        conflicts = {
            field
            for field in mergeable_fields
            if incoming_data[field] is not None
            and merged[field] is not None
            and incoming_data[field] != merged[field]
        }
        # SCD2 state cannot have two different current values at the same valid_from.
        # Treat an equal-time firmographic correction as ambiguous even when it only
        # fills a previously-null value; it must be re-issued with a later observation
        # timestamp or resolved explicitly in the review queue.
        conflicts.update(
            field
            for field in {"revenue_eur", "employees", "company_status"}
            if incoming_data[field] is not None and incoming_data[field] != merged[field]
        )
        if conflicts:
            conflicting_fields = ", ".join(sorted(conflicts))
            raise ValueError(f"Equal-time source conflict for fields: {conflicting_fields}")

    for field in mergeable_fields:
        value = incoming_data[field]
        if value is not None:
            merged[field] = value
    if incoming.source_observed_at > existing.source_observed_at:
        merged["source"] = incoming.source
        merged["source_observed_at"] = incoming.source_observed_at
    else:
        merged["source"] = min(existing.source, incoming.source)
    return Company.model_validate(merged)


def deduplicate_companies(companies: Iterable[Company]) -> tuple[Company, ...]:
    """Deduplicate by P.IVA with deterministic, non-destructive last-observation merge."""

    by_piva: dict[str, Company] = {}
    order: list[str] = []
    for company in companies:
        if company.piva not in by_piva:
            by_piva[company.piva] = company
            order.append(company.piva)
        else:
            by_piva[company.piva] = merge_companies(by_piva[company.piva], company)
    return tuple(by_piva[piva] for piva in order)


class InMemoryCompanyRepository:
    """Deterministic DEMO adapter with the same upsert contract as Postgres."""

    def __init__(self, companies: Iterable[Company] = ()) -> None:
        self._companies = {company.piva: company for company in deduplicate_companies(companies)}

    def get(self, piva: str) -> Company | None:
        return self._companies.get(normalize_piva(piva))

    def upsert(self, company: Company) -> UpsertResult:
        existing = self._companies.get(company.piva)
        if existing is None:
            self._companies[company.piva] = company
            return UpsertResult(company=company, action=UpsertAction.CREATED)

        merged = merge_companies(existing, company)
        action = UpsertAction.UNCHANGED if merged == existing else UpsertAction.UPDATED
        self._companies[company.piva] = merged
        return UpsertResult(company=merged, action=action)

    def list_all(self) -> tuple[Company, ...]:
        return tuple(self._companies[piva] for piva in sorted(self._companies))
