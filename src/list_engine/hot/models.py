"""Provider-neutral contracts for the event-driven hot-signal pipeline.

A ``HotSignal`` is the neutral shape every source adapter produces. It carries
its own two content addresses:

* ``natural_key_hash`` — identity of the *real-world* signal
  ``(source, piva, signal_type, observed_at)``. It mirrors the golden
  ``signals`` unique key and is the outbox event dedupe key and the hot-task
  idempotency key. Two deliveries of the same signal share it.
* ``content_hash`` — the exact-content replay guard for the append-only raw
  landing. It excludes ``received_at`` (ingress metadata that differs between
  redeliveries) so identical webhook bytes collapse to one raw row.
"""

from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Self
from urllib.parse import urlsplit
from uuid import UUID

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from list_engine.core.piva import normalize_piva
from list_engine.ingestion.quality import canonical_payload_hash, validate_json_object

_HEX_64 = re.compile(r"[a-f0-9]{64}")


class SignalSource(StrEnum):
    """The fresh-signal sources named in the M5 spec (one adapter each)."""

    company_monitoring = "company_monitoring"  # OpenAPI.it Company Monitoring (visura changes)
    hiring = "hiring"  # job-board hiring signal
    clay = "clay"  # Clay webhook (contact waterfall)


class HotSignalType(StrEnum):
    """Closed set of readiness triggers, aligned to SegmentConfig.hot_signal_types."""

    new_site = "new_site"
    governance_change = "governance_change"
    administrative_hiring = "administrative_hiring"
    revenue_growth = "revenue_growth"


class OutboxStatus(StrEnum):
    pending = "pending"
    leased = "leased"
    done = "done"
    failed = "failed"
    dead = "dead"


class HotTaskStatus(StrEnum):
    ready = "ready"  # M6 delivery extends this set


class HotOutcome(StrEnum):
    """Terminal outcome of one hot-workflow run over a single event."""

    task_created = "task_created"
    task_exists = "task_exists"
    unknown_company = "unknown_company"
    expired = "expired"
    research_review = "research_review"
    research_abstained = "research_abstained"
    deferred = "deferred"
    failed = "failed"


def _validated_source_url(value: str) -> str:
    if value != value.strip() or not value or len(value) > 2_048:
        raise ValueError("source_url must be trimmed, non-blank and at most 2048 chars")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("source_url must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("source_url cannot contain credentials")
    return value


def _validated_hex_64(value: str, *, label: str) -> str:
    if _HEX_64.fullmatch(value) is None:
        raise ValueError(f"{label} must be a 64-character lowercase hex digest")
    return value


def _seconds_between(start: datetime, end: datetime) -> Decimal:
    """Exact elapsed seconds as Decimal (no float rounding), for replayable latency."""

    delta = end - start
    whole = Decimal(delta.days * 86_400 + delta.seconds)
    return whole + Decimal(delta.microseconds) / Decimal(1_000_000)


class HotSignal(BaseModel):
    """A fresh readiness signal normalized to the neutral hot-pipeline shape."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    source: SignalSource
    piva: str
    signal_type: HotSignalType
    observed_at: AwareDatetime  # when the real-world event occurred (== signals.occurred_at)
    received_at: AwareDatetime  # when the ingress accepted it
    valid_until: AwareDatetime  # freshness horizon; a stale signal is skipped
    confidence: float = Field(default=1.0, ge=0, le=1)
    source_url: str | None = None
    payload: dict[str, object]

    @field_validator("piva", mode="before")
    @classmethod
    def _normalize_piva(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("P.IVA must be text")
        return normalize_piva(value)

    @field_validator("source_url")
    @classmethod
    def _validate_source_url(cls, value: str | None) -> str | None:
        return _validated_source_url(value) if value is not None else None

    @field_validator("payload", mode="before")
    @classmethod
    def _validate_payload(cls, value: object) -> dict[str, object]:
        return validate_json_object(value)

    @model_validator(mode="after")
    def _validate_time_bounds(self) -> Self:
        if self.observed_at > self.received_at:
            raise ValueError("observed_at cannot follow received_at")
        if self.valid_until <= self.observed_at:
            raise ValueError("valid_until must be after observed_at")
        return self

    def is_expired(self, *, at: datetime) -> bool:
        return self.valid_until <= at

    @computed_field  # type: ignore[prop-decorator]
    @property
    def natural_key_hash(self) -> str:
        """Identity of the real-world signal; the outbox/task dedupe anchor."""

        return canonical_payload_hash(
            {
                "source": self.source.value,
                "piva": self.piva,
                "signal_type": self.signal_type.value,
                "observed_at": self.observed_at.isoformat(),
            }
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def content_hash(self) -> str:
        """Exact-content replay guard for the raw landing (excludes received_at)."""

        return canonical_payload_hash(
            {
                "source": self.source.value,
                "piva": self.piva,
                "signal_type": self.signal_type.value,
                "observed_at": self.observed_at.isoformat(),
                "valid_until": self.valid_until.isoformat(),
                "confidence": self.confidence,
                "source_url": self.source_url,
                "payload": self.payload,
            }
        )


class OutboxEvent(BaseModel):
    """One durable event per distinct real-world signal, carried through the worker."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    id: UUID
    signal: HotSignal
    status: OutboxStatus = OutboxStatus.pending
    attempts: int = Field(default=0, ge=0)
    available_at: AwareDatetime
    created_at: AwareDatetime

    @property
    def dedupe_key(self) -> str:
        return self.signal.natural_key_hash

    @property
    def piva(self) -> str:
        return self.signal.piva

    @property
    def source(self) -> SignalSource:
        return self.signal.source

    @property
    def signal_type(self) -> HotSignalType:
        return self.signal.signal_type


class HotTask(BaseModel):
    """Internal BDR call task carrying a gated dossier; M6 delivers it to HubSpot."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    id: UUID
    piva: str
    signal_natural_key: str
    source: SignalSource
    signal_type: HotSignalType
    agent_session_id: UUID | None = None  # None for the DEMO direct-run path
    dossier_digest: str
    status: HotTaskStatus = HotTaskStatus.ready
    signal_observed_at: AwareDatetime
    signal_received_at: AwareDatetime
    task_created_at: AwareDatetime

    @field_validator("piva", mode="before")
    @classmethod
    def _normalize_piva(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("P.IVA must be text")
        return normalize_piva(value)

    @field_validator("signal_natural_key")
    @classmethod
    def _validate_natural_key(cls, value: str) -> str:
        return _validated_hex_64(value, label="signal_natural_key")

    @field_validator("dossier_digest")
    @classmethod
    def _validate_dossier_digest(cls, value: str) -> str:
        return _validated_hex_64(value, label="dossier_digest")

    @model_validator(mode="after")
    def _validate_time_bounds(self) -> Self:
        if self.signal_observed_at > self.signal_received_at:
            raise ValueError("signal_observed_at cannot follow signal_received_at")
        if self.task_created_at < self.signal_received_at:
            raise ValueError("task_created_at cannot precede signal_received_at")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def signal_to_task_seconds(self) -> Decimal:
        return _seconds_between(self.signal_received_at, self.task_created_at)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def source_to_task_seconds(self) -> Decimal:
        return _seconds_between(self.signal_observed_at, self.task_created_at)


__all__ = [
    "HotOutcome",
    "HotSignal",
    "HotSignalType",
    "HotTask",
    "HotTaskStatus",
    "OutboxEvent",
    "OutboxStatus",
    "SignalSource",
]
