from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

import psycopg
import pytest
from psycopg.rows import dict_row

from list_engine.core.repository import InMemoryCompanyRepository
from list_engine.hot.demo import demo_clay_signal, demo_company
from list_engine.hot.models import HotOutcome, HotSignal, OutboxStatus
from list_engine.hot.postgres import PostgresHotStore
from list_engine.hot.worker import (
    DirectResearchRunner,
    HotEventWorker,
    HotWorkflow,
    SignalEvidenceBuilder,
)
from list_engine.research.agent import DemoResearchAgent
from list_engine.research.gate import CitationGate

ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.getenv("TEST_DATABASE_URL")
BASE = datetime(2026, 7, 15, 8, 0, 0, tzinfo=UTC)
DIGEST = "a" * 64

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
    for migration in sorted((ROOT / "supabase" / "migrations").glob("*.sql")):
        ddl = migration.read_text()
        connection.execute(ddl, prepare=False)
        connection.execute(ddl, prepare=False)  # applied twice: replay-safety
    connection.execute("SET list_engine.app_mode = 'demo'")
    connection.execute((ROOT / "seeds" / "demo.sql").read_text(), prepare=False)

    yield connection
    connection.rollback()
    connection.execute("DROP SCHEMA IF EXISTS list_engine CASCADE")
    connection.close()


def _count(connection: psycopg.Connection[dict[str, Any]], table: str) -> int:
    row = connection.execute(f"SELECT count(*) AS n FROM list_engine.{table}").fetchone()
    assert row is not None
    return int(row["n"])


def _signal() -> HotSignal:
    return demo_clay_signal(received_at=BASE)


def test_append_is_idempotent_across_raw_and_outbox(
    database: psycopg.Connection[dict[str, Any]],
) -> None:
    store = PostgresHotStore(database)
    signal = _signal()

    first = store.append(signal, now=BASE)
    second = store.append(signal, now=BASE)

    assert first.created is True
    assert first.stored_raw is True
    assert second.created is False
    assert second.stored_raw is False
    assert second.event_id == first.event_id
    assert _count(database, "hot_signals") == 1
    assert _count(database, "signal_outbox") == 1


def test_differing_content_lands_but_does_not_duplicate_event(
    database: psycopg.Connection[dict[str, Any]],
) -> None:
    store = PostgresHotStore(database)
    store.append(demo_clay_signal(received_at=BASE), now=BASE)
    # same identity, different confidence -> different content_hash, same natural key
    variant = demo_clay_signal(received_at=BASE).model_copy(update={"confidence": 0.5})
    result = store.append(variant, now=BASE)

    assert result.created is False
    assert result.stored_raw is True
    assert _count(database, "hot_signals") == 2
    assert _count(database, "signal_outbox") == 1


def test_claim_settle_done_and_no_reclaim(
    database: psycopg.Connection[dict[str, Any]],
) -> None:
    store = PostgresHotStore(database)
    store.append(_signal(), now=BASE)

    lease = store.claim_next(owner_token=uuid4(), lease_ttl=timedelta(minutes=5), now=BASE)
    assert lease is not None
    assert store.claim_next(owner_token=uuid4(), lease_ttl=timedelta(minutes=5), now=BASE) is None

    assert store.settle_done(lease, now=BASE) is True
    row = database.execute("SELECT status FROM list_engine.signal_outbox").fetchone()
    assert row is not None
    assert row["status"] == OutboxStatus.done.value
    assert store.claim_next(owner_token=uuid4(), lease_ttl=timedelta(minutes=5), now=BASE) is None


def test_expired_lease_takeover_refuses_stale_owner(
    database: psycopg.Connection[dict[str, Any]],
) -> None:
    store = PostgresHotStore(database)
    store.append(_signal(), now=BASE)
    first = store.claim_next(owner_token=uuid4(), lease_ttl=timedelta(minutes=5), now=BASE)
    assert first is not None

    # deterministically expire the lease, then a second worker takes over
    database.execute(
        "UPDATE list_engine.signal_outbox "
        "SET lease_expires_at = clock_timestamp() - interval '1 second'"
    )
    second = store.claim_next(owner_token=uuid4(), lease_ttl=timedelta(minutes=5), now=BASE)
    assert second is not None
    assert second.owner_token != first.owner_token
    assert second.event.attempts == 2

    assert store.settle_done(first, now=BASE) is False  # lost the lease
    assert store.settle_done(second, now=BASE) is True


def test_create_task_is_idempotent_and_records_latency(
    database: psycopg.Connection[dict[str, Any]],
) -> None:
    store = PostgresHotStore(database)
    signal = _signal()

    task1, created1 = store.create_task(
        piva="99000000002",
        signal=signal,
        agent_session_id=None,
        dossier_digest=DIGEST,
        now=BASE + timedelta(seconds=4),
    )
    task2, created2 = store.create_task(
        piva="99000000002",
        signal=signal,
        agent_session_id=None,
        dossier_digest="b" * 64,
        now=BASE + timedelta(seconds=9),
    )

    assert created1 is True
    assert created2 is False
    assert task1.id == task2.id
    assert task2.dossier_digest == DIGEST  # the first write wins
    assert _count(database, "hot_tasks") == 1

    row = database.execute("SELECT signal_to_task_seconds FROM list_engine.hot_tasks").fetchone()
    assert row is not None
    assert row["signal_to_task_seconds"] == Decimal("4")


def test_worker_end_to_end_creates_task_with_latency(
    database: psycopg.Connection[dict[str, Any]],
) -> None:
    store = PostgresHotStore(database)
    signal = _signal()
    store.append(signal, now=BASE)

    task_time = BASE + timedelta(seconds=4)
    workflow = HotWorkflow(
        companies=InMemoryCompanyRepository([demo_company(observed_at=BASE)]),
        research=DirectResearchRunner(DemoResearchAgent(), CitationGate()),
        tasks=store,
        evidence=SignalEvidenceBuilder(),
    )
    worker = HotEventWorker(outbox=store, runner=workflow, clock=lambda: task_time)

    result = asyncio.run(worker.run_once())
    assert result is not None
    assert result.outcome is HotOutcome.task_created
    assert result.task is not None
    assert result.task.signal_to_task_seconds == Decimal("4")

    status_row = database.execute(
        "SELECT status FROM list_engine.signal_outbox WHERE dedupe_key = %s",
        (signal.natural_key_hash,),
    ).fetchone()
    assert status_row is not None
    assert status_row["status"] == OutboxStatus.done.value

    # redelivery creates no second event and no second task
    assert store.append(signal, now=BASE).created is False
    assert asyncio.run(worker.run_once()) is None
    assert _count(database, "hot_tasks") == 1
    assert _count(database, "signal_outbox") == 1


def test_latency_view_reports_percentiles(
    database: psycopg.Connection[dict[str, Any]],
) -> None:
    store = PostgresHotStore(database)
    store.create_task(
        piva="99000000002",
        signal=_signal(),
        agent_session_id=None,
        dossier_digest=DIGEST,
        now=BASE + timedelta(seconds=5),
    )
    row = database.execute(
        "SELECT tasks, p50_seconds FROM list_engine.hot_signal_latency "
        "WHERE source = 'clay' AND signal_type = 'administrative_hiring'"
    ).fetchone()
    assert row is not None
    assert row["tasks"] == 1
    assert row["p50_seconds"] == Decimal("5")
