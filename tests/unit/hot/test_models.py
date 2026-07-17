"""Unit tests for the neutral hot-pipeline contracts."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

import pytest
from pydantic import ValidationError

from list_engine.hot.models import (
    HotSignal,
    HotSignalType,
    HotTask,
    OutboxEvent,
    OutboxStatus,
    SignalSource,
)

OBSERVED = datetime(2026, 7, 17, 8, 0, 0, tzinfo=UTC)
RECEIVED = datetime(2026, 7, 17, 8, 0, 3, tzinfo=UTC)
VALID_UNTIL = datetime(2026, 8, 16, 0, 0, 0, tzinfo=UTC)
_HEX_64 = re.compile(r"^[a-f0-9]{64}$")


def signal(**overrides: Any) -> HotSignal:
    params: dict[str, Any] = {
        "source": SignalSource.clay,
        "piva": "99000000002",
        "signal_type": HotSignalType.administrative_hiring,
        "observed_at": OBSERVED,
        "received_at": RECEIVED,
        "valid_until": VALID_UNTIL,
        "confidence": 0.9,
        "source_url": "https://example.invalid/news",
        "payload": {"headline": "Nuova assunzione amministrativa"},
    }
    params.update(overrides)
    return HotSignal(**params)


def test_signal_normalizes_piva() -> None:
    assert signal(piva="  IT99000000002 ").piva == "99000000002"


def test_signal_rejects_invalid_piva() -> None:
    with pytest.raises(ValidationError):
        signal(piva="99000000001")  # wrong check digit


@pytest.mark.parametrize("field", ["observed_at", "received_at", "valid_until"])
def test_signal_rejects_naive_datetime(field: str) -> None:
    with pytest.raises(ValidationError):
        signal(**{field: datetime(2026, 7, 17, 8, 0, 0)})


def test_signal_rejects_observed_after_received() -> None:
    with pytest.raises(ValidationError, match="observed_at cannot follow received_at"):
        signal(observed_at=datetime(2026, 7, 17, 8, 0, 5, tzinfo=UTC))


def test_signal_requires_valid_until_after_observed() -> None:
    with pytest.raises(ValidationError, match="valid_until must be after observed_at"):
        signal(valid_until=OBSERVED)


@pytest.mark.parametrize(
    "bad_url", ["ftp://example.invalid", "not-a-url", "https://user:pw@example.invalid"]
)
def test_signal_rejects_bad_source_url(bad_url: str) -> None:
    with pytest.raises(ValidationError):
        signal(source_url=bad_url)


def test_signal_rejects_non_json_payload() -> None:
    with pytest.raises(ValidationError):
        signal(payload={"when": OBSERVED})  # datetime is not JSON


def test_content_hash_is_64_hex() -> None:
    assert _HEX_64.fullmatch(signal().content_hash) is not None
    assert _HEX_64.fullmatch(signal().natural_key_hash) is not None


def test_content_hash_excludes_received_at() -> None:
    early = signal(received_at=datetime(2026, 7, 17, 8, 0, 1, tzinfo=UTC))
    late = signal(received_at=datetime(2026, 7, 17, 8, 0, 30, tzinfo=UTC))
    assert early.content_hash == late.content_hash  # redelivery collapses to one raw row


def test_content_hash_changes_with_payload() -> None:
    assert signal(payload={"a": 1}).content_hash != signal(payload={"a": 2}).content_hash


def test_natural_key_stable_across_non_identity_noise() -> None:
    base = signal()
    noisy = signal(
        confidence=0.1,
        source_url="https://other.invalid/x",
        valid_until=datetime(2026, 9, 1, tzinfo=UTC),
        payload={"totally": "different"},
    )
    assert base.natural_key_hash == noisy.natural_key_hash


@pytest.mark.parametrize(
    "overrides",
    [
        {"source": SignalSource.hiring},
        {"piva": "99000000028"},
        {"signal_type": HotSignalType.new_site},
        {"observed_at": datetime(2026, 7, 16, 8, 0, 0, tzinfo=UTC)},
    ],
)
def test_natural_key_changes_with_identity(overrides: dict[str, Any]) -> None:
    assert signal().natural_key_hash != signal(**overrides).natural_key_hash


def test_is_expired() -> None:
    assert signal().is_expired(at=datetime(2026, 9, 1, tzinfo=UTC)) is True
    assert signal().is_expired(at=datetime(2026, 7, 20, tzinfo=UTC)) is False


def test_outbox_event_derives_keys_from_signal() -> None:
    event = OutboxEvent(
        id=UUID("00000000-0000-0000-0000-000000000001"),
        signal=signal(),
        available_at=RECEIVED,
        created_at=RECEIVED,
    )
    assert event.status is OutboxStatus.pending
    assert event.dedupe_key == signal().natural_key_hash
    assert event.piva == "99000000002"
    assert event.source is SignalSource.clay
    assert event.signal_type is HotSignalType.administrative_hiring


def hot_task(**overrides: Any) -> HotTask:
    params: dict[str, Any] = {
        "id": UUID("00000000-0000-0000-0000-000000000002"),
        "piva": "99000000002",
        "signal_natural_key": signal().natural_key_hash,
        "source": SignalSource.clay,
        "signal_type": HotSignalType.administrative_hiring,
        "agent_session_id": None,
        "dossier_digest": "a" * 64,
        "signal_observed_at": OBSERVED,
        "signal_received_at": RECEIVED,
        "task_created_at": datetime(2026, 7, 17, 8, 0, 6, tzinfo=UTC),
    }
    params.update(overrides)
    return HotTask(**params)


def test_hot_task_latency_is_exact_decimal() -> None:
    task = hot_task()
    assert task.signal_to_task_seconds == Decimal("3")  # 08:00:03 -> 08:00:06
    assert task.source_to_task_seconds == Decimal("6")  # 08:00:00 -> 08:00:06


def test_hot_task_latency_handles_fractions() -> None:
    task = hot_task(task_created_at=datetime(2026, 7, 17, 8, 0, 6, 500_000, tzinfo=UTC))
    assert task.signal_to_task_seconds == Decimal("3.5")


def test_hot_task_rejects_task_before_received() -> None:
    with pytest.raises(ValidationError, match="task_created_at cannot precede signal_received_at"):
        hot_task(task_created_at=datetime(2026, 7, 17, 8, 0, 1, tzinfo=UTC))


def test_hot_task_rejects_non_hex_digest() -> None:
    with pytest.raises(ValidationError):
        hot_task(dossier_digest="not-hex")
