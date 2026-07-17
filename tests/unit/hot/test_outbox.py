"""Unit tests for the in-memory transactional outbox and its fenced lease."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from list_engine.hot.models import HotSignal, HotSignalType, OutboxStatus, SignalSource
from list_engine.hot.outbox import InMemoryOutbox, SignalIngestService

T0 = datetime(2026, 7, 17, 8, 0, 0, tzinfo=UTC)
TTL = timedelta(minutes=5)


def signal(**overrides: Any) -> HotSignal:
    params: dict[str, Any] = {
        "source": SignalSource.clay,
        "piva": "99000000002",
        "signal_type": HotSignalType.administrative_hiring,
        "observed_at": T0,
        "received_at": T0,
        "valid_until": datetime(2026, 8, 16, tzinfo=UTC),
        "payload": {"n": 1},
    }
    params.update(overrides)
    return HotSignal(**params)


def test_append_is_idempotent_on_natural_key() -> None:
    outbox = InMemoryOutbox()
    first = outbox.append(signal(), now=T0)
    second = outbox.append(signal(payload={"n": 2}), now=T0)  # same identity, new content

    assert first.created is True
    assert first.stored_raw is True
    assert second.created is False  # no duplicate event
    assert second.stored_raw is True  # but the differing raw bytes still land
    assert second.event_id == first.event_id
    assert outbox.pending_count() == 1
    assert outbox.raw_count() == 2


def test_claim_then_empty() -> None:
    outbox = InMemoryOutbox()
    outbox.append(signal(), now=T0)

    lease = outbox.claim_next(owner_token=uuid4(), lease_ttl=TTL, now=T0)
    assert lease is not None
    assert lease.event.attempts == 1
    assert outbox.claim_next(owner_token=uuid4(), lease_ttl=TTL, now=T0) is None


def test_expired_lease_is_taken_over() -> None:
    outbox = InMemoryOutbox()
    outbox.append(signal(), now=T0)
    first = outbox.claim_next(owner_token=uuid4(), lease_ttl=TTL, now=T0)
    assert first is not None

    later = T0 + TTL + timedelta(seconds=1)
    second = outbox.claim_next(owner_token=uuid4(), lease_ttl=TTL, now=later)
    assert second is not None
    assert second.owner_token != first.owner_token
    assert second.event.attempts == 2


def test_settle_done_prevents_reclaim() -> None:
    outbox = InMemoryOutbox()
    outbox.append(signal(), now=T0)
    lease = outbox.claim_next(owner_token=uuid4(), lease_ttl=TTL, now=T0)
    assert lease is not None

    assert outbox.settle_done(lease, now=T0) is True
    assert outbox.status_of(signal().natural_key_hash) is OutboxStatus.done
    assert outbox.claim_next(owner_token=uuid4(), lease_ttl=TTL, now=T0 + TTL * 10) is None


def test_settle_retry_reschedules_with_backoff() -> None:
    outbox = InMemoryOutbox()
    outbox.append(signal(), now=T0)
    lease = outbox.claim_next(owner_token=uuid4(), lease_ttl=TTL, now=T0)
    assert lease is not None

    assert outbox.settle_retry(lease, backoff=timedelta(minutes=1), now=T0, error_type="x") is True
    assert outbox.status_of(signal().natural_key_hash) is OutboxStatus.pending
    # not available before the backoff elapses, available after
    assert (
        outbox.claim_next(owner_token=uuid4(), lease_ttl=TTL, now=T0 + timedelta(seconds=30))
        is None
    )
    assert (
        outbox.claim_next(owner_token=uuid4(), lease_ttl=TTL, now=T0 + timedelta(minutes=1))
        is not None
    )


def test_stale_owner_settle_is_refused() -> None:
    outbox = InMemoryOutbox()
    outbox.append(signal(), now=T0)
    first = outbox.claim_next(owner_token=uuid4(), lease_ttl=TTL, now=T0)
    assert first is not None
    # a second worker takes over the expired lease
    later = T0 + TTL + timedelta(seconds=1)
    second = outbox.claim_next(owner_token=uuid4(), lease_ttl=TTL, now=later)
    assert second is not None

    # the original owner's settle must be refused (it lost the lease)
    assert outbox.settle_done(first, now=later) is False
    assert outbox.status_of(signal().natural_key_hash) is OutboxStatus.leased
    # the current owner can still settle
    assert outbox.settle_done(second, now=later) is True


def test_ingest_service_delegates_to_outbox() -> None:
    outbox = InMemoryOutbox()
    service = SignalIngestService(outbox)
    result = service.accept(signal(), now=T0)
    assert result.created is True
    assert outbox.pending_count() == 1
