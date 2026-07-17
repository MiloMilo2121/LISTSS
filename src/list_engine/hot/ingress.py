"""FastAPI webhook ingress for hot signals (needs the optional ``ingress`` extra).

Thin edge: verify the signature over the raw body (fail closed), normalize to a
neutral ``HotSignal``, and enqueue it transactionally — then return 202. All heavy
work happens in the worker. This module imports FastAPI, so it is reached only by
its full path and never from ``list_engine.hot``.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from list_engine.hot.adapters import (
    SignalNormalizationError,
    SignatureError,
    SignatureVerifier,
    normalize_clay,
    normalize_company_monitoring,
    normalize_hiring,
)
from list_engine.hot.models import HotSignal, SignalSource
from list_engine.hot.outbox import SignalIngestService

_MAX_BODY_BYTES = 256 * 1024

_NORMALIZERS: dict[SignalSource, Callable[[Mapping[str, object], datetime], HotSignal]] = {
    SignalSource.clay: lambda payload, received_at: normalize_clay(
        payload, received_at=received_at
    ),
    SignalSource.company_monitoring: lambda payload, received_at: normalize_company_monitoring(
        payload, received_at=received_at
    ),
    SignalSource.hiring: lambda payload, received_at: normalize_hiring(
        payload, received_at=received_at
    ),
}

_ROUTES: dict[str, SignalSource] = {
    "/webhooks/clay": SignalSource.clay,
    "/webhooks/company-monitoring": SignalSource.company_monitoring,
    "/webhooks/hiring": SignalSource.hiring,
}


def create_ingress_app(
    *,
    ingest: SignalIngestService,
    verifiers: Mapping[SignalSource, SignatureVerifier],
    clock: Callable[[], datetime],
) -> FastAPI:
    """Build the ingress app; one signed webhook endpoint per configured source."""

    app = FastAPI(title="List Engine hot-signal ingress")

    def _endpoint(source: SignalSource) -> Callable[[Request], Awaitable[JSONResponse]]:
        verifier = verifiers.get(source)
        normalizer = _NORMALIZERS[source]

        async def handle(request: Request) -> JSONResponse:
            if verifier is None:
                return JSONResponse(
                    status_code=503, content={"detail": f"{source.value} ingress is not configured"}
                )
            body = await request.body()
            if len(body) > _MAX_BODY_BYTES:
                return JSONResponse(status_code=413, content={"detail": "payload too large"})
            try:
                verifier.verify(body, dict(request.headers))
            except SignatureError:
                return JSONResponse(
                    status_code=401, content={"detail": "signature verification failed"}
                )
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                return JSONResponse(status_code=400, content={"detail": "body is not valid JSON"})
            if not isinstance(payload, dict):
                return JSONResponse(
                    status_code=400, content={"detail": "body must be a JSON object"}
                )
            try:
                signal = normalizer(payload, clock())
            except SignalNormalizationError as error:
                return JSONResponse(status_code=400, content={"detail": str(error)})
            result = ingest.accept(signal, now=clock())
            return JSONResponse(
                status_code=202,
                content={"event_id": str(result.event_id), "deduped": not result.created},
            )

        return handle

    for path, source in _ROUTES.items():
        app.add_api_route(path, _endpoint(source), methods=["POST"])

    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    app.add_api_route("/healthz", healthz, methods=["GET"])

    return app


__all__ = ["create_ingress_app"]
