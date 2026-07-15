from __future__ import annotations

import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import psycopg
import pytest
from psycopg.rows import dict_row

from list_engine.adapters.postgres import PostgresCompanyRepository
from list_engine.core.models import Company
from list_engine.core.piva import calculate_check_digit
from list_engine.core.repository import UpsertAction

ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.getenv("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="TEST_DATABASE_URL is required for disposable PostgreSQL integration tests",
)


@pytest.fixture()
def database() -> Iterator[psycopg.Connection[dict[str, object]]]:
    assert DATABASE_URL is not None
    connection = psycopg.connect(DATABASE_URL, autocommit=True, row_factory=dict_row)
    if "test" not in connection.info.dbname.lower():
        connection.close()
        pytest.fail("Refusing to reset a database whose name does not contain 'test'")

    migration = (ROOT / "supabase" / "migrations" / "202607150001_initial.sql").read_text()
    connection.execute("DROP SCHEMA IF EXISTS list_engine CASCADE")
    connection.execute(migration, prepare=False)
    connection.execute(migration, prepare=False)
    yield connection
    connection.rollback()
    connection.execute("DROP SCHEMA IF EXISTS list_engine CASCADE")
    connection.close()


def apply_demo_seed(connection: psycopg.Connection[dict[str, object]]) -> None:
    seed = (ROOT / "seeds" / "demo.sql").read_text()
    connection.execute("SET list_engine.app_mode = 'demo'")
    connection.execute(seed, prepare=False)


def test_schema_is_private_by_default(
    database: psycopg.Connection[dict[str, object]],
) -> None:
    assert database.execute("SELECT to_regclass('public.companies')").fetchone() == {
        "to_regclass": None
    }
    assert database.execute(
        """
        SELECT count(*)
        FROM pg_namespace namespace
        CROSS JOIN LATERAL aclexplode(namespace.nspacl) acl
        WHERE namespace.nspname = 'list_engine'
          AND acl.grantee = 0
        """
    ).fetchone() == {"count": 0}
    assert database.execute(
        """
        SELECT count(*)
        FROM pg_proc procedure
        JOIN pg_namespace namespace ON namespace.oid = procedure.pronamespace
        CROSS JOIN LATERAL aclexplode(procedure.proacl) acl
        WHERE namespace.nspname = 'list_engine'
          AND acl.grantee = 0
          AND acl.privilege_type = 'EXECUTE'
        """
    ).fetchone() == {"count": 0}


def test_demo_seed_fails_closed_and_replays_idempotently(
    database: psycopg.Connection[dict[str, object]],
) -> None:
    seed = (ROOT / "seeds" / "demo.sql").read_text()
    with pytest.raises(psycopg.errors.RaiseException, match="Demo seed refused"):
        database.execute(seed, prepare=False)
    database.rollback()

    apply_demo_seed(database)
    apply_demo_seed(database)

    assert database.execute(
        "SELECT count(*) FROM list_engine.companies WHERE is_demo"
    ).fetchone() == {"count": 20}
    assert database.execute(
        "SELECT count(*) FROM list_engine.company_firmographic_history"
    ).fetchone() == {"count": 20}


def test_append_only_and_compliance_ownership_constraints_are_enforced(
    database: psycopg.Connection[dict[str, object]],
) -> None:
    apply_demo_seed(database)
    source_record_id = database.execute(
        """
        INSERT INTO list_engine.source_records (source, piva, payload, payload_hash)
        VALUES ('fixture', '99000000002', '{}'::jsonb, %s)
        RETURNING id
        """,
        ("0" * 64,),
    ).fetchone()
    assert source_record_id is not None
    with pytest.raises(psycopg.errors.RaiseException, match="append-only"):
        database.execute(
            "UPDATE list_engine.source_records SET source = 'changed' WHERE id = %s",
            (source_record_id["id"],),
        )

    contact = database.execute(
        """
        INSERT INTO list_engine.contacts (piva, phone, phone_type, source)
        VALUES ('99000000002', '+390212345678', 'switchboard', 'fixture')
        RETURNING id
        """
    ).fetchone()
    assert contact is not None
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        database.execute(
            """
            INSERT INTO list_engine.compliance_checks (
                piva, contact_id, check_type, result, provider, evidence, checked_at
            )
            VALUES (
                '99000000010', %s, 'rpo', 'clear', 'fixture', '{}'::jsonb, now()
            )
            """,
            (contact["id"],),
        )


def test_postgres_upsert_is_concurrency_safe_and_idempotent(
    database: psycopg.Connection[dict[str, object]],
) -> None:
    assert DATABASE_URL is not None
    stem = "9800000000"
    piva = f"{stem}{calculate_check_digit(stem)}"
    record = Company(
        piva=piva,
        legal_name="Concurrency Fixture S.r.l.",
        source="integration_test",
        source_observed_at=datetime(2026, 7, 1, tzinfo=UTC),
    )

    def upsert_once() -> UpsertAction:
        repository = PostgresCompanyRepository.connect(DATABASE_URL)
        try:
            return repository.upsert(record).action
        finally:
            repository.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        actions = list(pool.map(lambda _index: upsert_once(), range(2)))

    assert sorted(actions) == sorted([UpsertAction.CREATED, UpsertAction.UNCHANGED])
    assert database.execute(
        "SELECT count(*) FROM list_engine.companies WHERE piva = %s", (piva,)
    ).fetchone() == {"count": 1}
    assert database.execute(
        "SELECT count(*) FROM list_engine.company_firmographic_history WHERE piva = %s",
        (piva,),
    ).fetchone() == {"count": 1}
