from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from list_engine.adapters.openapi_it import (
    OpenApiItAdapter,
    OpenApiItAdapterError,
    OpenApiItConfig,
    OpenApiItIngestionEnvelope,
    OpenApiItLookup,
    OpenApiItSchemaError,
)
from list_engine.ingestion import (
    MappedCompanyRecord,
    MappedSourceBatch,
    RawRecordEnvelope,
    SourceReadStatus,
)

PIVA = "99000000002"


def provider_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "id": "provider-company-1",
        "taxCode": PIVA,
        "vatCode": PIVA,
        "companyName": "Aurora Logistica S.r.l.",
        "activityStatus": "ATTIVA",
        "address": {
            "registeredOffice": {
                "town": "MILANO",
                "province": "MI",
                "region": {"code": "03", "description": "LOMBARDIA"},
            }
        },
        "atecoClassification": {
            "ateco": {"code": "4941", "description": "Trasporto merci su strada"}
        },
        "balanceSheets": {
            "last": {"year": 2025, "turnover": 6_500_000, "employees": 28},
            "all": [],
        },
        "lastUpdateTimestamp": 1_751_328_000,
        # A documented field not promoted into Company must survive as raw evidence.
        "pec": "aurora@example.invalid",
    }
    record.update(overrides)
    return record


def response_payload(record: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "data": [record or provider_record()],
        "success": True,
        "message": "",
        "error": None,
    }


def run_fetch(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    config: OpenApiItConfig | None = None,
) -> OpenApiItIngestionEnvelope:
    async def scenario() -> OpenApiItIngestionEnvelope:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenApiItAdapter(
                client,
                config
                or OpenApiItConfig(
                    bearer_token=SecretStr("test-token"),
                    base_url="https://test.company.openapi.com",
                ),
            )
            return await adapter.fetch(OpenApiItLookup(piva=PIVA))

    return asyncio.run(scenario())


def run_fetch_mapped(
    handler: Callable[[httpx.Request], httpx.Response],
) -> MappedSourceBatch:
    async def scenario() -> MappedSourceBatch:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenApiItAdapter(
                client,
                OpenApiItConfig(
                    bearer_token=SecretStr("test-token"),
                    base_url="https://test.company.openapi.com",
                ),
            )
            return await adapter.fetch_mapped(OpenApiItLookup(piva=PIVA))

    return asyncio.run(scenario())


def test_fetch_maps_only_typed_claims_and_preserves_raw_provider_record() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == f"https://test.company.openapi.com/IT-advanced/{PIVA}"
        assert request.headers["Authorization"] == "Bearer test-token"
        assert request.headers["Accept"] == "application/json"
        return httpx.Response(200, json=response_payload(), request=request)

    envelope = run_fetch(handler)

    assert envelope.source == "openapi_it"
    assert envelope.source_record_id == "provider-company-1"
    assert envelope.piva == PIVA
    assert envelope.raw_payload["pec"] == "aurora@example.invalid"
    assert envelope.company.piva == PIVA
    assert envelope.company.legal_name == "Aurora Logistica S.r.l."
    assert envelope.company.ateco_code == "4941"
    assert envelope.company.city == "MILANO"
    assert envelope.company.province == "MI"
    assert envelope.company.region == "LOMBARDIA"
    assert envelope.company.revenue_eur == Decimal("6500000")
    assert envelope.company.employees == 28
    assert envelope.company.company_status == "active"
    assert envelope.company.source == "openapi_it"
    assert envelope.company.is_demo is False
    assert envelope.observed_at == datetime.fromtimestamp(1_751_328_000, tz=UTC)


def test_envelope_converts_to_shared_mapped_record_without_changing_raw_evidence() -> None:
    exact_provider_record = provider_record(
        nestedUnmapped={"flags": [True, False], "note": "verbatim evidence"}
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=response_payload(exact_provider_record),
            request=request,
        )

    envelope = run_fetch(handler)
    mapped = envelope.to_mapped_record()

    assert isinstance(mapped, MappedCompanyRecord)
    assert isinstance(mapped.raw, RawRecordEnvelope)
    assert mapped.raw.payload == exact_provider_record
    assert mapped.raw.payload != response_payload(exact_provider_record)
    assert mapped.raw.source_record_id == "provider-company-1"
    assert mapped.raw.observed_at == envelope.observed_at
    assert mapped.raw.observed_at is not None
    assert mapped.raw.observed_at.utcoffset() == timedelta(0)
    assert mapped.company == envelope.company


def test_fetch_mapped_returns_successful_shared_source_batch() -> None:
    exact_provider_record = provider_record()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=response_payload(exact_provider_record),
            request=request,
        )

    batch = run_fetch_mapped(handler)

    assert isinstance(batch, MappedSourceBatch)
    assert batch.source == "openapi_it"
    assert batch.status is SourceReadStatus.SUCCESS
    assert len(batch.records) == 1
    assert batch.records[0].raw.payload == exact_provider_record
    assert batch.records[0].raw.observed_at is not None
    assert batch.records[0].raw.observed_at.utcoffset() == timedelta(0)
    assert batch.records[0].company is not None
    assert batch.records[0].company.piva == PIVA


def test_base_url_and_endpoint_are_configurable() -> None:
    config = OpenApiItConfig(
        bearer_token=SecretStr("test-token"),
        base_url="https://provider.example.test/root/",
        endpoint_template="v2/company/{identifier}",
        is_demo=True,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == f"https://provider.example.test/root/v2/company/{PIVA}"
        return httpx.Response(200, json=response_payload(), request=request)

    envelope = run_fetch(handler, config=config)

    assert envelope.company.is_demo is True


def test_lookup_normalizes_identity_before_building_the_request() -> None:
    lookup = OpenApiItLookup(piva="IT 990.000.000-02")

    assert lookup.piva == PIVA


def test_success_envelope_is_strict_about_documented_control_fields() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = response_payload()
        payload["success"] = "true"
        return httpx.Response(200, json=payload, request=request)

    with pytest.raises(OpenApiItSchemaError) as caught:
        run_fetch(handler)

    assert caught.value.retryable is False
    assert caught.value.status_code == 200


@pytest.mark.parametrize(
    ("status_code", "retryable"),
    [(400, False), (402, False), (404, False), (406, False), (429, True), (500, True), (503, True)],
)
def test_http_failures_are_classified_without_retrying(
    status_code: int,
    retryable: bool,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        headers = {"Retry-After": "7"} if status_code == 429 else {}
        return httpx.Response(
            status_code,
            json={"success": False, "message": "provider failure", "error": 610, "data": None},
            headers=headers,
            request=request,
        )

    with pytest.raises(OpenApiItAdapterError) as caught:
        run_fetch(handler)

    assert calls == 1
    assert caught.value.retryable is retryable
    assert caught.value.status_code == status_code
    assert caught.value.provider_error_code == 610
    assert caught.value.retry_after_seconds == (7 if status_code == 429 else None)


def test_mismatched_provider_identity_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=response_payload(provider_record(vatCode="99000000010")),
            request=request,
        )

    with pytest.raises(OpenApiItSchemaError, match="matching VAT identity"):
        run_fetch(handler)


def test_transport_failure_is_retryable_but_not_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("offline", request=request)

    with pytest.raises(OpenApiItAdapterError) as caught:
        run_fetch(handler)

    assert calls == 1
    assert caught.value.retryable is True
    assert caught.value.status_code is None


def test_config_rejects_an_endpoint_without_identity_placeholder() -> None:
    with pytest.raises(ValidationError, match="identifier"):
        OpenApiItConfig(
            bearer_token=SecretStr("test-token"),
            endpoint_template="/IT-advanced",
        )
