"""Transactional-outbox port plus a deterministic in-memory DEMO adapter.

The outbox is the ingress -> worker boundary: one ``append`` atomically lands the
raw signal and enqueues one event per distinct real-world signal, and ``claim``
hands a fenced lease (owner token + expiry) to exactly one worker — the same
fenced-lease shape used by ``research/postgres.py``. A durable-execution engine
(DBOS/Hatchet) can replace this behind the identical port at M8.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol
from uuid import UUID, uuid4

from list_engine.hot.models import HotSignal, OutboxEvent, OutboxStatus


@dataclass(frozen=True, slots=True)
class OutboxAppendResult:
    event_id: UUID
    created: bool  # False when a same-identity event was already enqueued
    stored_raw: bool  # False when identical raw bytes had already landed


@dataclass(frozen=True, slots=True)
class LeasedEvent:
    event: OutboxEvent
    owner_token: UUID
    lease_expires_at: datetime


class Outbox(Protocol):
    """Append/claim/settle port; every settle is fenced by the lease owner token."""

    def append(self, signal: HotSignal, *, now: datetime) -> OutboxAppendResult: ...

    def claim_next(
        self, *, owner_token: UUID, lease_ttl: timedelta, now: datetime
    ) -> LeasedEvent | None: ...

    def settle_done(self, lease: LeasedEvent, *, now: datetime) -> bool: ...

    def settle_retry(
        self, lease: LeasedEvent, *, backoff: timedelta, now: datetime, error_type: str
    ) -> bool: ...

    def settle_dead(
        self, lease: LeasedEvent, *, now: datetime, error_type: str, retryable: bool
    ) -> bool: ...


@dataclass
class _OutboxRow:
    id: UUID
    signal: HotSignal
    status: OutboxStatus
    attempts: int
    available_at: datetime
    created_at: datetime
    owner_token: UUID | None
    lease_expires_at: datetime | None
    last_error: str | None


class InMemoryOutbox:
    """Production-shaped DEMO outbox: same semantics, no database, injected clock."""

    def __init__(self) -> None:
        self._events: dict[str, _OutboxRow] = {}  # keyed by signal.natural_key_hash
        self._raw: set[str] = set()  # content_hash landing guard

    def append(self, signal: HotSignal, *, now: datetime) -> OutboxAppendResult:
        stored_raw = signal.content_hash not in self._raw
        self._raw.add(signal.content_hash)

        key = signal.natural_key_hash
        existing = self._events.get(key)
        if existing is not None:
            return OutboxAppendResult(existing.id, created=False, stored_raw=stored_raw)

        row = _OutboxRow(
            id=uuid4(),
            signal=signal,
            status=OutboxStatus.pending,
            attempts=0,
            available_at=now,
            created_at=now,
            owner_token=None,
            lease_expires_at=None,
            last_error=None,
        )
        self._events[key] = row
        return OutboxAppendResult(row.id, created=True, stored_raw=stored_raw)

    def claim_next(
        self, *, owner_token: UUID, lease_ttl: timedelta, now: datetime
    ) -> LeasedEvent | None:
        claimable = [
            row
            for row in self._events.values()
            if row.available_at <= now and self._is_claimable(row, now=now)
        ]
        if not claimable:
            return None
        row = min(claimable, key=lambda candidate: (candidate.available_at, candidate.created_at))
        row.status = OutboxStatus.leased
        row.owner_token = owner_token
        row.lease_expires_at = now + lease_ttl
        row.attempts += 1
        return LeasedEvent(
            event=self._to_event(row),
            owner_token=owner_token,
            lease_expires_at=row.lease_expires_at,
        )

    def settle_done(self, lease: LeasedEvent, *, now: datetime) -> bool:
        return self._settle(lease, status=OutboxStatus.done, error_type=None)

    def settle_retry(
        self, lease: LeasedEvent, *, backoff: timedelta, now: datetime, error_type: str
    ) -> bool:
        row = self._fenced_row(lease)
        if row is None:
            return False
        row.status = OutboxStatus.pending
        row.available_at = now + backoff
        row.owner_token = None
        row.lease_expires_at = None
        row.last_error = error_type
        return True

    def settle_dead(
        self, lease: LeasedEvent, *, now: datetime, error_type: str, retryable: bool
    ) -> bool:
        return self._settle(lease, status=OutboxStatus.dead, error_type=error_type)

    # --- inspection helpers used by tests ---------------------------------

    def pending_count(self) -> int:
        return sum(row.status is OutboxStatus.pending for row in self._events.values())

    def status_of(self, natural_key: str) -> OutboxStatus | None:
        row = self._events.get(natural_key)
        return row.status if row is not None else None

    def raw_count(self) -> int:
        return len(self._raw)

    # --- internals --------------------------------------------------------

    @staticmethod
    def _is_claimable(row: _OutboxRow, *, now: datetime) -> bool:
        if row.status is OutboxStatus.pending:
            return True
        return (
            row.status is OutboxStatus.leased
            and row.lease_expires_at is not None
            and row.lease_expires_at <= now
        )

    def _fenced_row(self, lease: LeasedEvent) -> _OutboxRow | None:
        row = self._events.get(lease.event.signal.natural_key_hash)
        if row is None or row.owner_token != lease.owner_token:
            return None  # taken over by another worker: refuse the stale settle
        return row

    def _settle(self, lease: LeasedEvent, *, status: OutboxStatus, error_type: str | None) -> bool:
        row = self._fenced_row(lease)
        if row is None:
            return False
        row.status = status
        row.owner_token = None
        row.lease_expires_at = None
        row.last_error = error_type
        return True

    @staticmethod
    def _to_event(row: _OutboxRow) -> OutboxEvent:
        return OutboxEvent(
            id=row.id,
            signal=row.signal,
            status=row.status,
            attempts=row.attempts,
            available_at=row.available_at,
            created_at=row.created_at,
        )


class SignalIngestService:
    """Ingress-facing seam: accept a normalized signal into the outbox atomically."""

    def __init__(self, outbox: Outbox) -> None:
        self._outbox = outbox

    def accept(self, signal: HotSignal, *, now: datetime) -> OutboxAppendResult:
        return self._outbox.append(signal, now=now)


__all__ = [
    "InMemoryOutbox",
    "LeasedEvent",
    "Outbox",
    "OutboxAppendResult",
    "SignalIngestService",
]
