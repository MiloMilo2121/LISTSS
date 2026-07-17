"""Unit tests for source normalizers and webhook signature verifiers."""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import SecretStr

from list_engine.hot.adapters import (
    HmacSignatureVerifier,
    SharedTokenVerifier,
    SignalNormalizationError,
    SignatureError,
    normalize_clay,
    normalize_company_monitoring,
    normalize_hiring,
)
from list_engine.hot.models import HotSignalType, SignalSource

RECEIVED = datetime(2026, 7, 17, 9, 0, 0, tzinfo=UTC)


def clay_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "piva": "99000000002",
        "signal_type": "administrative_hiring",
        "observed_at": "2026-07-17T08:00:00+00:00",
        "source_url": "https://example.invalid/job",
        "role": "Responsabile Amministrativo",
    }
    base.update(overrides)
    return base


def monitoring_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "vat_code": "99000000002",
        "event_type": "governance_update",
        "event_date": "2026-07-15",
    }
    base.update(overrides)
    return base


def hiring_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "piva": "99000000002",
        "department": "administrative",
        "posted_at": "2026-07-16T00:00:00+00:00",
    }
    base.update(overrides)
    return base


def test_normalize_clay() -> None:
    result = normalize_clay(clay_payload(), received_at=RECEIVED)
    assert result.source is SignalSource.clay
    assert result.signal_type is HotSignalType.administrative_hiring
    assert result.piva == "99000000002"
    assert result.observed_at == datetime(2026, 7, 17, 8, 0, 0, tzinfo=UTC)
    assert result.received_at == RECEIVED
    assert result.valid_until == result.observed_at + timedelta(days=90)  # default horizon
    assert result.payload["role"] == "Responsabile Amministrativo"


def test_normalize_company_monitoring_date_only() -> None:
    result = normalize_company_monitoring(monitoring_payload(), received_at=RECEIVED)
    assert result.source is SignalSource.company_monitoring
    assert result.signal_type is HotSignalType.governance_change
    assert result.observed_at == datetime(2026, 7, 15, 0, 0, 0, tzinfo=UTC)  # bare date -> midnight


def test_normalize_hiring() -> None:
    result = normalize_hiring(hiring_payload(), received_at=RECEIVED)
    assert result.source is SignalSource.hiring
    assert result.signal_type is HotSignalType.administrative_hiring


def test_clay_accepts_explicit_valid_until() -> None:
    result = normalize_clay(
        clay_payload(valid_until="2026-07-20T00:00:00+00:00"), received_at=RECEIVED
    )
    assert result.valid_until == datetime(2026, 7, 20, 0, 0, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("normalizer", "payload"),
    [
        (normalize_clay, clay_payload(signal_type="unknown_trigger")),
        (normalize_company_monitoring, monitoring_payload(event_type="mystery")),
        (normalize_hiring, hiring_payload(department="sales")),
    ],
)
def test_unmapped_type_raises(normalizer: Callable[..., Any], payload: dict[str, Any]) -> None:
    with pytest.raises(SignalNormalizationError, match="not mapped"):
        normalizer(payload, received_at=RECEIVED)


@pytest.mark.parametrize(
    ("normalizer", "payload"),
    [
        (normalize_clay, clay_payload(piva=None)),
        (normalize_company_monitoring, monitoring_payload(vat_code=None, piva=None)),
        (normalize_hiring, hiring_payload(posted_at=None)),
    ],
)
def test_missing_required_field_raises(
    normalizer: Callable[..., Any], payload: dict[str, Any]
) -> None:
    payload = {key: value for key, value in payload.items() if value is not None}
    with pytest.raises(SignalNormalizationError, match="missing required field"):
        normalizer(payload, received_at=RECEIVED)


def test_invalid_piva_wrapped_as_normalization_error() -> None:
    with pytest.raises(SignalNormalizationError, match="invalid clay signal"):
        normalize_clay(clay_payload(piva="99000000001"), received_at=RECEIVED)


def test_naive_timestamp_raises() -> None:
    with pytest.raises(SignalNormalizationError, match="timezone-aware"):
        normalize_clay(clay_payload(observed_at="2026-07-17T08:00:00"), received_at=RECEIVED)


def test_small_skew_is_clamped() -> None:
    result = normalize_clay(
        clay_payload(observed_at="2026-07-17T09:02:00+00:00"), received_at=RECEIVED
    )
    assert result.observed_at == RECEIVED  # 2 min ahead -> clamped to received_at


def test_large_skew_rejected() -> None:
    with pytest.raises(SignalNormalizationError, match="implausibly ahead"):
        normalize_clay(clay_payload(observed_at="2026-07-17T09:10:00+00:00"), received_at=RECEIVED)


def test_hmac_verifier_accepts_valid_and_rejects_tampering() -> None:
    secret = SecretStr("clay-webhook-secret")
    body = b'{"piva":"99000000002"}'
    digest = hmac.new(b"clay-webhook-secret", body, hashlib.sha256).hexdigest()
    verifier = HmacSignatureVerifier(secret, header="X-Clay-Signature")

    verifier.verify(body, {"X-Clay-Signature": f"sha256={digest}"})  # no raise

    with pytest.raises(SignatureError, match="mismatch"):
        verifier.verify(body, {"X-Clay-Signature": "sha256=" + "0" * 64})
    with pytest.raises(SignatureError, match="mismatch"):
        verifier.verify(b'{"piva":"99000000028"}', {"X-Clay-Signature": f"sha256={digest}"})
    with pytest.raises(SignatureError, match="missing"):
        verifier.verify(body, {})


def test_hmac_verifier_header_lookup_is_case_insensitive() -> None:
    secret = SecretStr("s")
    body = b"payload"
    digest = hmac.new(b"s", body, hashlib.sha256).hexdigest()
    verifier = HmacSignatureVerifier(secret, header="X-Signature")
    verifier.verify(body, {"x-signature": f"sha256={digest}"})


def test_shared_token_verifier() -> None:
    verifier = SharedTokenVerifier(SecretStr("api-token"), header="authorization")
    verifier.verify(b"", {"Authorization": "Bearer api-token"})

    with pytest.raises(SignatureError, match="mismatch"):
        verifier.verify(b"", {"Authorization": "Bearer wrong"})
    with pytest.raises(SignatureError, match="scheme"):
        verifier.verify(b"", {"Authorization": "api-token"})
    with pytest.raises(SignatureError, match="missing"):
        verifier.verify(b"", {})
