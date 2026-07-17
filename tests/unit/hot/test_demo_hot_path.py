"""End-to-end DEMO hot path: signal -> event -> dossier -> task, fully offline."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal

from list_engine.hot.adapters import normalize_clay
from list_engine.hot.demo import build_demo_pipeline, demo_clay_signal
from list_engine.hot.models import HotOutcome, OutboxStatus

RECEIVED = datetime(2026, 7, 17, 8, 0, 0, tzinfo=UTC)
TASK_TIME = datetime(2026, 7, 17, 8, 0, 3, tzinfo=UTC)


class _Clock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


def test_hot_path_creates_task_and_measures_latency() -> None:
    pipeline = build_demo_pipeline(clock=_Clock(TASK_TIME), seed_observed_at=RECEIVED)
    signal = demo_clay_signal(received_at=RECEIVED)

    append = pipeline.ingest.accept(signal, now=RECEIVED)
    assert append.created is True
    assert pipeline.outbox.pending_count() == 1

    result = asyncio.run(pipeline.worker.run_once())

    assert result is not None
    assert result.outcome is HotOutcome.task_created
    assert result.task is not None
    assert result.task.signal_to_task_seconds == Decimal("3")  # 08:00:00 -> 08:00:03
    assert result.latency_seconds == Decimal("3")
    assert len(pipeline.tasks.all_tasks()) == 1
    assert pipeline.outbox.status_of(signal.natural_key_hash) is OutboxStatus.done


def test_hot_path_is_idempotent_on_redelivery() -> None:
    pipeline = build_demo_pipeline(clock=_Clock(TASK_TIME), seed_observed_at=RECEIVED)
    signal = demo_clay_signal(received_at=RECEIVED)

    pipeline.ingest.accept(signal, now=RECEIVED)
    first = asyncio.run(pipeline.worker.run_once())
    assert first is not None
    assert first.outcome is HotOutcome.task_created

    redelivery = pipeline.ingest.accept(signal, now=RECEIVED)
    assert redelivery.created is False  # no second event
    assert redelivery.stored_raw is False  # no second raw row

    assert asyncio.run(pipeline.worker.run_once()) is None  # settled event is not re-claimed
    assert len(pipeline.tasks.all_tasks()) == 1
    assert pipeline.outbox.raw_count() == 1


def test_hot_path_unknown_company_is_dead_lettered() -> None:
    pipeline = build_demo_pipeline(clock=_Clock(TASK_TIME), seed_observed_at=RECEIVED)
    # a valid but unseeded P.IVA
    signal = normalize_clay(
        {
            "piva": "99000000028",
            "signal_type": "administrative_hiring",
            "observed_at": RECEIVED.isoformat(),
            "source_url": "https://jobs.demo.invalid/x",
        },
        received_at=RECEIVED,
    )
    pipeline.ingest.accept(signal, now=RECEIVED)

    result = asyncio.run(pipeline.worker.run_once())

    assert result is not None
    assert result.outcome is HotOutcome.unknown_company
    assert result.task is None
    assert pipeline.tasks.all_tasks() == ()
    assert pipeline.outbox.status_of(signal.natural_key_hash) is OutboxStatus.dead


def test_hot_path_expired_signal_is_skipped() -> None:
    long_after = datetime(2026, 12, 1, tzinfo=UTC)
    pipeline = build_demo_pipeline(clock=_Clock(long_after), seed_observed_at=RECEIVED)
    signal = demo_clay_signal(received_at=RECEIVED)  # valid_until ~ 2026-10-15
    pipeline.ingest.accept(signal, now=RECEIVED)

    result = asyncio.run(pipeline.worker.run_once())

    assert result is not None
    assert result.outcome is HotOutcome.expired
    assert result.task is None
    assert pipeline.outbox.status_of(signal.natural_key_hash) is OutboxStatus.done
