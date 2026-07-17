"""PostgreSQL adapter for the hot pipeline: raw landing, fenced outbox, tasks.

Implements the ``Outbox`` and ``HotTaskStore`` ports on one connection, reusing
the fenced-lease shape from ``research/postgres.py``: server ``clock_timestamp()``
authority, ``FOR UPDATE SKIP LOCKED`` claims, and settles fenced by the lease
owner token so a taken-over lease cannot double-write.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from threading import Lock
from typing import Any, Self
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from list_engine.hot.models import (
    HotSignal,
    HotSignalType,
    HotTask,
    HotTaskStatus,
    OutboxEvent,
    OutboxStatus,
    SignalSource,
)
from list_engine.hot.outbox import LeasedEvent, OutboxAppendResult


def _signal_payload(signal: HotSignal) -> dict[str, object]:
    # Computed hashes are stored in their own columns; excluding them keeps the
    # jsonb round-trippable through the extra="forbid" model.
    return signal.model_dump(mode="json", exclude={"content_hash", "natural_key_hash"})


def _load_signal(raw: object) -> HotSignal:
    # JSONB decodes to Python dicts/strings; strict models need JSON-mode validation.
    return HotSignal.model_validate_json(json.dumps(raw, ensure_ascii=False))


class PostgresHotStore:
    """Durable outbox + task store; one connection per worker execution context."""

    def __init__(self, connection: psycopg.Connection[dict[str, Any]]) -> None:
        self._connection = connection
        self._lock = Lock()

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
        with self._lock:
            self._connection.close()

    # --- Outbox port ------------------------------------------------------

    def append(self, signal: HotSignal, *, now: datetime) -> OutboxAppendResult:
        with self._lock, self._connection.transaction():
            raw = self._connection.execute(
                """
                INSERT INTO list_engine.hot_signals (
                    source, piva, signal_type, observed_at, received_at, valid_until,
                    confidence, source_url, payload, content_hash, natural_key_hash
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (source, content_hash) DO NOTHING
                RETURNING id
                """,
                (
                    signal.source.value,
                    signal.piva,
                    signal.signal_type.value,
                    signal.observed_at,
                    signal.received_at,
                    signal.valid_until,
                    signal.confidence,
                    signal.source_url,
                    Jsonb(signal.payload),
                    signal.content_hash,
                    signal.natural_key_hash,
                ),
            ).fetchone()
            if raw is not None:
                hot_signal_id = raw["id"]
                stored_raw = True
            else:
                existing_raw = self._connection.execute(
                    "SELECT id FROM list_engine.hot_signals "
                    "WHERE source = %s AND content_hash = %s",
                    (signal.source.value, signal.content_hash),
                ).fetchone()
                assert existing_raw is not None
                hot_signal_id = existing_raw["id"]
                stored_raw = False

            event = self._connection.execute(
                """
                INSERT INTO list_engine.signal_outbox (
                    hot_signal_id, dedupe_key, piva, source, signal_type, signal
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (dedupe_key) DO NOTHING
                RETURNING id
                """,
                (
                    hot_signal_id,
                    signal.natural_key_hash,
                    signal.piva,
                    signal.source.value,
                    signal.signal_type.value,
                    Jsonb(_signal_payload(signal)),
                ),
            ).fetchone()
            if event is not None:
                return OutboxAppendResult(event["id"], created=True, stored_raw=stored_raw)
            existing_event = self._connection.execute(
                "SELECT id FROM list_engine.signal_outbox WHERE dedupe_key = %s",
                (signal.natural_key_hash,),
            ).fetchone()
            assert existing_event is not None
            return OutboxAppendResult(existing_event["id"], created=False, stored_raw=stored_raw)

    def claim_next(
        self, *, owner_token: UUID, lease_ttl: timedelta, now: datetime
    ) -> LeasedEvent | None:
        with self._lock:
            row = self._connection.execute(
                """
                UPDATE list_engine.signal_outbox AS o
                SET status = 'leased',
                    owner_token = %s,
                    lease_expires_at = clock_timestamp() + make_interval(secs => %s),
                    leased_at = clock_timestamp(),
                    attempts = o.attempts + 1
                WHERE o.id = (
                    SELECT id FROM list_engine.signal_outbox
                    WHERE available_at <= clock_timestamp()
                      AND (
                          status = 'pending'
                          OR (status = 'leased' AND lease_expires_at <= clock_timestamp())
                      )
                    ORDER BY available_at, created_at
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                RETURNING o.id, o.signal, o.status, o.attempts,
                          o.available_at, o.created_at, o.lease_expires_at
                """,
                (owner_token, lease_ttl.total_seconds()),
            ).fetchone()
        if row is None:
            return None
        event = OutboxEvent(
            id=row["id"],
            signal=_load_signal(row["signal"]),
            status=OutboxStatus(row["status"]),
            attempts=row["attempts"],
            available_at=row["available_at"],
            created_at=row["created_at"],
        )
        return LeasedEvent(
            event=event, owner_token=owner_token, lease_expires_at=row["lease_expires_at"]
        )

    def settle_done(self, lease: LeasedEvent, *, now: datetime) -> bool:
        return self._settle(
            lease,
            "status = 'done', done_at = clock_timestamp(), "
            "owner_token = NULL, lease_expires_at = NULL",
            (),
        )

    def settle_retry(
        self, lease: LeasedEvent, *, backoff: timedelta, now: datetime, error_type: str
    ) -> bool:
        return self._settle(
            lease,
            "status = 'pending', "
            "available_at = clock_timestamp() + make_interval(secs => %s), "
            "owner_token = NULL, lease_expires_at = NULL, last_error = %s",
            (backoff.total_seconds(), error_type),
        )

    def settle_dead(
        self, lease: LeasedEvent, *, now: datetime, error_type: str, retryable: bool
    ) -> bool:
        return self._settle(
            lease,
            "status = 'dead', dead_at = clock_timestamp(), "
            "owner_token = NULL, lease_expires_at = NULL, last_error = %s",
            (error_type,),
        )

    def _settle(self, lease: LeasedEvent, assignments: str, params: tuple[object, ...]) -> bool:
        with self._lock:
            cursor = self._connection.execute(
                f"""
                UPDATE list_engine.signal_outbox
                SET {assignments}
                WHERE id = %s
                  AND owner_token = %s
                  AND status = 'leased'
                  AND lease_expires_at > clock_timestamp()
                """,
                (*params, lease.event.id, lease.owner_token),
            )
            return cursor.rowcount > 0

    # --- HotTaskStore port ------------------------------------------------

    def create_task(
        self,
        *,
        piva: str,
        signal: HotSignal,
        agent_session_id: UUID | None,
        dossier_digest: str,
        now: datetime,
    ) -> tuple[HotTask, bool]:
        with self._lock, self._connection.transaction():
            inserted = self._connection.execute(
                """
                INSERT INTO list_engine.hot_tasks (
                    piva, signal_natural_key, source, signal_type, agent_session_id,
                    dossier_digest, signal_observed_at, signal_received_at, task_created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (signal_natural_key) DO NOTHING
                RETURNING id
                """,
                (
                    piva,
                    signal.natural_key_hash,
                    signal.source.value,
                    signal.signal_type.value,
                    agent_session_id,
                    dossier_digest,
                    signal.observed_at,
                    signal.received_at,
                    now,
                ),
            ).fetchone()
            created = inserted is not None
            row = self._connection.execute(
                """
                SELECT id, piva, signal_natural_key, source, signal_type, agent_session_id,
                       dossier_digest, status, signal_observed_at, signal_received_at,
                       task_created_at
                FROM list_engine.hot_tasks
                WHERE signal_natural_key = %s
                """,
                (signal.natural_key_hash,),
            ).fetchone()
        assert row is not None
        task = HotTask(
            id=row["id"],
            piva=row["piva"],
            signal_natural_key=row["signal_natural_key"],
            source=SignalSource(row["source"]),
            signal_type=HotSignalType(row["signal_type"]),
            agent_session_id=row["agent_session_id"],
            dossier_digest=row["dossier_digest"],
            status=HotTaskStatus(row["status"]),
            signal_observed_at=row["signal_observed_at"],
            signal_received_at=row["signal_received_at"],
            task_created_at=row["task_created_at"],
        )
        return task, created


__all__ = ["PostgresHotStore"]
