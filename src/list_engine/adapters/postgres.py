"""Atomic PostgreSQL implementation of the company repository port."""

from __future__ import annotations

from typing import Any, Self

import psycopg
from psycopg.rows import dict_row

from list_engine.core.models import Company
from list_engine.core.piva import normalize_piva
from list_engine.core.repository import UpsertAction, UpsertResult, merge_companies

_COLUMNS = (
    "piva, legal_name, website, ateco_code, city, province, region, revenue_eur, "
    "employees, company_status, source, is_demo, source_observed_at"
)

_SELECT_ONE = f"SELECT {_COLUMNS} FROM list_engine.companies WHERE piva = %s"
_SELECT_FOR_UPDATE = f"{_SELECT_ONE} FOR UPDATE"
_SELECT_ALL = f"SELECT {_COLUMNS} FROM list_engine.companies ORDER BY piva"

_INSERT = f"""
    INSERT INTO list_engine.companies ({_COLUMNS})
    VALUES (
        %(piva)s, %(legal_name)s, %(website)s, %(ateco_code)s, %(city)s,
        %(province)s, %(region)s, %(revenue_eur)s, %(employees)s,
        %(company_status)s, %(source)s, %(is_demo)s, %(source_observed_at)s
    )
    ON CONFLICT (piva) DO NOTHING
    RETURNING {_COLUMNS}
"""

_UPDATE = f"""
    UPDATE list_engine.companies
    SET legal_name = %(legal_name)s,
        website = %(website)s,
        ateco_code = %(ateco_code)s,
        city = %(city)s,
        province = %(province)s,
        region = %(region)s,
        revenue_eur = %(revenue_eur)s,
        employees = %(employees)s,
        company_status = %(company_status)s,
        source = %(source)s,
        is_demo = %(is_demo)s,
        source_observed_at = %(source_observed_at)s
    WHERE piva = %(piva)s
    RETURNING {_COLUMNS}
"""


def _params(company: Company) -> dict[str, object]:
    return company.model_dump()


def _company(row: dict[str, Any]) -> Company:
    return Company.model_validate(row)


class PostgresCompanyRepository:
    """Golden-record repository using row locks for concurrency-safe idempotence.

    Application queries may use Supavisor transaction mode. Prepared statements are
    explicitly disabled because transaction pooling cannot guarantee connection
    affinity. DBOS durable state uses its own direct/session DSN in M5.
    """

    def __init__(self, connection: psycopg.Connection[dict[str, Any]]) -> None:
        self._connection = connection

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

    def get(self, piva: str) -> Company | None:
        row = self._connection.execute(_SELECT_ONE, (normalize_piva(piva),)).fetchone()
        return _company(row) if row is not None else None

    def upsert(self, company: Company) -> UpsertResult:
        with self._connection.transaction():
            for _attempt in range(2):
                row = self._connection.execute(_SELECT_FOR_UPDATE, (company.piva,)).fetchone()
                if row is not None:
                    existing = _company(row)
                    merged = merge_companies(existing, company)
                    if merged == existing:
                        return UpsertResult(company=existing, action=UpsertAction.UNCHANGED)

                    updated_row = self._connection.execute(_UPDATE, _params(merged)).fetchone()
                    if updated_row is None:  # pragma: no cover - protected by row lock
                        raise RuntimeError("Locked company disappeared during upsert")
                    updated = _company(updated_row)
                    self._record_firmographic_history(existing, updated)
                    return UpsertResult(company=updated, action=UpsertAction.UPDATED)

                inserted_row = self._connection.execute(_INSERT, _params(company)).fetchone()
                if inserted_row is not None:
                    inserted = _company(inserted_row)
                    self._record_firmographic_history(None, inserted)
                    return UpsertResult(company=inserted, action=UpsertAction.CREATED)
                # A concurrent transaction inserted the row after our SELECT. The next
                # pass locks it and applies the same deterministic merge contract.

        raise RuntimeError("Company upsert could not converge after a concurrent insert")

    def list_all(self) -> tuple[Company, ...]:
        rows = self._connection.execute(_SELECT_ALL).fetchall()
        return tuple(_company(row) for row in rows)

    def _record_firmographic_history(self, previous: Company | None, current: Company) -> None:
        previous_values = (
            (
                previous.revenue_eur,
                previous.employees,
                previous.company_status,
            )
            if previous
            else None
        )
        current_values = (current.revenue_eur, current.employees, current.company_status)
        if previous_values == current_values:
            return

        self._connection.execute(
            """
            UPDATE list_engine.company_firmographic_history
            SET valid_to = %(valid_from)s
            WHERE piva = %(piva)s
              AND valid_to IS NULL
              AND valid_from < %(valid_from)s
            """,
            {"piva": current.piva, "valid_from": current.source_observed_at},
        )
        self._connection.execute(
            """
            INSERT INTO list_engine.company_firmographic_history (
                piva, revenue_eur, employees, company_status, source, valid_from
            )
            VALUES (
                %(piva)s, %(revenue_eur)s, %(employees)s, %(company_status)s,
                %(source)s, %(valid_from)s
            )
            ON CONFLICT (piva, source, valid_from) DO NOTHING
            """,
            {
                "piva": current.piva,
                "revenue_eur": current.revenue_eur,
                "employees": current.employees,
                "company_status": current.company_status,
                "source": current.source,
                "valid_from": current.source_observed_at,
            },
        )
