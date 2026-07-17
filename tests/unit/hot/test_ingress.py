"""Unit tests for the FastAPI hot-signal ingress (needs the ingress extra)."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

pytest.importorskip("fastapi")

from list_engine.hot.adapters import HmacSignatureVerifier
from list_engine.hot.ingress import create_ingress_app
from list_engine.hot.models import SignalSource
from list_engine.hot.outbox import InMemoryOutbox, SignalIngestService

SECRET = "clay-webhook-secret"
NOW = datetime(2026, 7, 17, 9, 0, 0, tzinfo=UTC)


def _app(outbox: InMemoryOutbox) -> Any:
    return create_ingress_app(
        ingest=SignalIngestService(outbox),
        verifiers={
            SignalSource.clay: HmacSignatureVerifier(SecretStr(SECRET), header="X-Clay-Signature")
        },
        clock=lambda: NOW,
    )


def _sign(body: bytes) -> str:
    return "sha256=" + hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()


def _clay_body(**overrides: Any) -> bytes:
    payload: dict[str, Any] = {
        "piva": "99000000002",
        "signal_type": "administrative_hiring",
        "observed_at": "2026-07-17T08:00:00+00:00",
        "source_url": "https://jobs.invalid/amministrazione",
    }
    payload.update(overrides)
    return json.dumps(payload).encode()


def _post(app: Any, path: str, body: bytes, headers: dict[str, str]) -> httpx.Response:
    async def run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(path, content=body, headers=headers)

    return asyncio.run(run())


def test_valid_webhook_accepts_and_enqueues() -> None:
    outbox = InMemoryOutbox()
    body = _clay_body()
    response = _post(_app(outbox), "/webhooks/clay", body, {"X-Clay-Signature": _sign(body)})
    assert response.status_code == 202
    assert response.json()["deduped"] is False
    assert outbox.pending_count() == 1


def test_duplicate_webhook_is_deduped() -> None:
    outbox = InMemoryOutbox()
    app = _app(outbox)
    body = _clay_body()
    headers = {"X-Clay-Signature": _sign(body)}

    first = _post(app, "/webhooks/clay", body, headers)
    second = _post(app, "/webhooks/clay", body, headers)

    assert first.status_code == 202
    assert first.json()["deduped"] is False
    assert second.status_code == 202
    assert second.json()["deduped"] is True
    assert outbox.pending_count() == 1


def test_bad_signature_is_rejected_and_nothing_enqueued() -> None:
    outbox = InMemoryOutbox()
    body = _clay_body()
    response = _post(
        _app(outbox), "/webhooks/clay", body, {"X-Clay-Signature": "sha256=" + "0" * 64}
    )
    assert response.status_code == 401
    assert outbox.pending_count() == 0


def test_missing_signature_is_rejected() -> None:
    outbox = InMemoryOutbox()
    response = _post(_app(outbox), "/webhooks/clay", _clay_body(), {})
    assert response.status_code == 401


def test_malformed_json_is_rejected() -> None:
    outbox = InMemoryOutbox()
    body = b"{not valid json"
    response = _post(_app(outbox), "/webhooks/clay", body, {"X-Clay-Signature": _sign(body)})
    assert response.status_code == 400
    assert outbox.pending_count() == 0


def test_invalid_piva_is_rejected() -> None:
    outbox = InMemoryOutbox()
    body = _clay_body(piva="99000000001")  # wrong check digit
    response = _post(_app(outbox), "/webhooks/clay", body, {"X-Clay-Signature": _sign(body)})
    assert response.status_code == 400
    assert outbox.pending_count() == 0


def test_unconfigured_source_returns_503() -> None:
    outbox = InMemoryOutbox()
    app = create_ingress_app(ingest=SignalIngestService(outbox), verifiers={}, clock=lambda: NOW)
    body = _clay_body()
    response = _post(app, "/webhooks/clay", body, {"X-Clay-Signature": _sign(body)})
    assert response.status_code == 503


def test_healthz_ok() -> None:
    async def run() -> httpx.Response:
        transport = httpx.ASGITransport(app=_app(InMemoryOutbox()))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/healthz")

    response = asyncio.run(run())
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
