"""One normalizer per source + webhook signature verifiers (pure, offline).

Every source adapter converts its provider-specific JSON into the neutral
``HotSignal``. Adapters classify failures (``SignalNormalizationError`` with a
``retryable`` flag) but never retry themselves — retry/DLQ live in the worker.
Signature verifiers fail closed and use constant-time comparison.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Protocol

from pydantic import SecretStr, ValidationError

from list_engine.hot.models import HotSignal, HotSignalType, SignalSource

_SKEW_TOLERANCE = timedelta(minutes=5)
_DEFAULT_HORIZON = timedelta(days=90)

# Clay webhooks are configured to emit canonical hot-signal type strings.
_CLAY_TYPE_MAP: dict[str, HotSignalType] = {member.value: member for member in HotSignalType}
_COMPANY_MONITORING_TYPE_MAP: dict[str, HotSignalType] = {
    "new_registered_office": HotSignalType.new_site,
    "registered_office_change": HotSignalType.new_site,
    "governance_update": HotSignalType.governance_change,
    "director_change": HotSignalType.governance_change,
    "share_capital_increase": HotSignalType.revenue_growth,
}
_HIRING_TYPE_MAP: dict[str, HotSignalType] = {
    "administrative": HotSignalType.administrative_hiring,
    "accounting": HotSignalType.administrative_hiring,
    "finance": HotSignalType.administrative_hiring,
}


class SignalNormalizationError(ValueError):
    """A provider payload could not be mapped to a neutral HotSignal."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class SignatureError(ValueError):
    """A webhook signature or token failed verification (fail-closed at ingress)."""


class SignatureVerifier(Protocol):
    def verify(self, body: bytes, headers: Mapping[str, str]) -> None: ...


def _header_value(headers: Mapping[str, str], name: str) -> str | None:
    lowered = name.casefold()
    for key, value in headers.items():
        if key.casefold() == lowered:
            return value
    return None


class HmacSignatureVerifier:
    """Verify a hex HMAC-SHA256 signature computed over the raw request body."""

    def __init__(self, secret: SecretStr, *, header: str, prefix: str = "sha256=") -> None:
        self._secret = secret
        self._header = header
        self._prefix = prefix

    def verify(self, body: bytes, headers: Mapping[str, str]) -> None:
        provided = _header_value(headers, self._header)
        if provided is None:
            raise SignatureError("missing signature header")
        candidate = provided.strip()
        if self._prefix and candidate.startswith(self._prefix):
            candidate = candidate[len(self._prefix) :]
        expected = hmac.new(
            self._secret.get_secret_value().encode(), body, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected, candidate):
            raise SignatureError("signature mismatch")


class SharedTokenVerifier:
    """Verify a shared bearer token in a header with constant-time comparison."""

    def __init__(
        self, token: SecretStr, *, header: str = "authorization", scheme: str = "Bearer"
    ) -> None:
        self._token = token
        self._header = header
        self._scheme = scheme

    def verify(self, body: bytes, headers: Mapping[str, str]) -> None:
        provided = _header_value(headers, self._header)
        if provided is None:
            raise SignatureError("missing token header")
        candidate = provided.strip()
        if self._scheme:
            prefix = f"{self._scheme} "
            if not candidate.startswith(prefix):
                raise SignatureError("token header must use the expected scheme")
            candidate = candidate[len(prefix) :]
        if not hmac.compare_digest(self._token.get_secret_value(), candidate.strip()):
            raise SignatureError("token mismatch")


def _require_str(payload: Mapping[str, object], *keys: str, label: str) -> str:
    for key in keys:
        if key in payload:
            value = payload[key]
            if not isinstance(value, str) or not value.strip():
                raise SignalNormalizationError(f"{label} must be a non-empty string")
            return value
    raise SignalNormalizationError(f"missing required field: {label}")


def _optional_str(payload: Mapping[str, object], *keys: str, label: str) -> str | None:
    for key in keys:
        if key in payload and payload[key] is not None:
            value = payload[key]
            if not isinstance(value, str) or not value.strip():
                raise SignalNormalizationError(f"{label} must be a non-empty string when present")
            return value
    return None


def _optional_float(payload: Mapping[str, object], key: str, *, default: float) -> float:
    if key not in payload or payload[key] is None:
        return default
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SignalNormalizationError(f"{key} must be a number")
    return float(value)


def _parse_aware(value: str, *, label: str) -> datetime:
    text = value.strip()
    if len(text) == 10:  # a bare calendar date → midnight UTC
        text = f"{text}T00:00:00+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise SignalNormalizationError(f"{label} is not a valid ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SignalNormalizationError(f"{label} must be timezone-aware")
    return parsed


def _resolve_observed_at(observed_at: datetime, *, received_at: datetime) -> datetime:
    if observed_at <= received_at:
        return observed_at
    if observed_at <= received_at + _SKEW_TOLERANCE:
        return received_at  # clamp small provider clock skew rather than reject
    raise SignalNormalizationError("observed_at is implausibly ahead of received_at")


def _map_type(
    raw: str, type_map: Mapping[str, HotSignalType], *, source: SignalSource
) -> HotSignalType:
    mapped = type_map.get(raw)
    if mapped is None:
        raise SignalNormalizationError(f"{source.value} signal type is not mapped: {raw!r}")
    return mapped


def _build_signal(
    *,
    source: SignalSource,
    piva: str,
    signal_type: HotSignalType,
    observed_at: datetime,
    received_at: datetime,
    valid_until: datetime | None,
    confidence: float,
    source_url: str | None,
    payload: Mapping[str, object],
) -> HotSignal:
    try:
        return HotSignal(
            source=source,
            piva=piva,
            signal_type=signal_type,
            observed_at=observed_at,
            received_at=received_at,
            valid_until=valid_until if valid_until is not None else observed_at + _DEFAULT_HORIZON,
            confidence=confidence,
            source_url=source_url,
            payload=dict(payload),
        )
    except ValidationError as error:
        raise SignalNormalizationError(f"invalid {source.value} signal: {error}") from error


def normalize_clay(payload: Mapping[str, object], *, received_at: datetime) -> HotSignal:
    """Clay webhook: canonical fields ``piva``, ``signal_type``, ``observed_at``."""

    signal_type = _map_type(
        _require_str(payload, "signal_type", label="signal_type"),
        _CLAY_TYPE_MAP,
        source=SignalSource.clay,
    )
    observed_at = _resolve_observed_at(
        _parse_aware(
            _require_str(payload, "observed_at", label="observed_at"), label="observed_at"
        ),
        received_at=received_at,
    )
    valid_until_raw = _optional_str(payload, "valid_until", label="valid_until")
    return _build_signal(
        source=SignalSource.clay,
        piva=_require_str(payload, "piva", label="piva"),
        signal_type=signal_type,
        observed_at=observed_at,
        received_at=received_at,
        valid_until=_parse_aware(valid_until_raw, label="valid_until") if valid_until_raw else None,
        confidence=_optional_float(payload, "confidence", default=1.0),
        source_url=_optional_str(payload, "source_url", label="source_url"),
        payload=payload,
    )


def normalize_company_monitoring(
    payload: Mapping[str, object], *, received_at: datetime
) -> HotSignal:
    """OpenAPI.it Company Monitoring: ``vat_code``/``piva``, ``event_type``, ``event_date``."""

    signal_type = _map_type(
        _require_str(payload, "event_type", label="event_type"),
        _COMPANY_MONITORING_TYPE_MAP,
        source=SignalSource.company_monitoring,
    )
    observed_at = _resolve_observed_at(
        _parse_aware(_require_str(payload, "event_date", label="event_date"), label="event_date"),
        received_at=received_at,
    )
    return _build_signal(
        source=SignalSource.company_monitoring,
        piva=_require_str(payload, "vat_code", "piva", label="vat_code"),
        signal_type=signal_type,
        observed_at=observed_at,
        received_at=received_at,
        valid_until=None,
        confidence=_optional_float(payload, "confidence", default=1.0),
        source_url=_optional_str(payload, "source_url", label="source_url"),
        payload=payload,
    )


def normalize_hiring(payload: Mapping[str, object], *, received_at: datetime) -> HotSignal:
    """Hiring signal (Jooble + targeted scraping): ``piva``, ``department``, ``posted_at``."""

    signal_type = _map_type(
        _require_str(payload, "department", label="department"),
        _HIRING_TYPE_MAP,
        source=SignalSource.hiring,
    )
    observed_at = _resolve_observed_at(
        _parse_aware(_require_str(payload, "posted_at", label="posted_at"), label="posted_at"),
        received_at=received_at,
    )
    return _build_signal(
        source=SignalSource.hiring,
        piva=_require_str(payload, "piva", label="piva"),
        signal_type=signal_type,
        observed_at=observed_at,
        received_at=received_at,
        valid_until=None,
        confidence=_optional_float(payload, "confidence", default=1.0),
        source_url=_optional_str(payload, "source_url", label="source_url"),
        payload=payload,
    )


NORMALIZERS = {
    SignalSource.clay: normalize_clay,
    SignalSource.company_monitoring: normalize_company_monitoring,
    SignalSource.hiring: normalize_hiring,
}


__all__ = [
    "NORMALIZERS",
    "HmacSignatureVerifier",
    "SharedTokenVerifier",
    "SignalNormalizationError",
    "SignatureError",
    "SignatureVerifier",
    "normalize_clay",
    "normalize_company_monitoring",
    "normalize_hiring",
]
