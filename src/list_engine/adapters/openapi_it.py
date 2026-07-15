"""OpenAPI.it Company Advanced adapter.

The provider response is kept verbatim in an ingestion envelope. Only explicitly
typed, source-observed fields are promoted into the golden-record ``Company``;
additional provider fields remain raw evidence and cannot become trusted claims by
accident.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal, cast

import httpx
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
)

from list_engine.core.models import Company
from list_engine.core.piva import InvalidPIVA, normalize_piva
from list_engine.ingestion import (
    MappedCompanyRecord,
    MappedSourceBatch,
    RawRecordEnvelope,
    SourceReadStatus,
)

SOURCE: Literal["openapi_it"] = "openapi_it"
DEFAULT_BASE_URL = "https://company.openapi.com"
DEFAULT_ENDPOINT_TEMPLATE = "/IT-advanced/{identifier}"


class OpenApiItConfig(BaseModel):
    """Runtime configuration supplied by the composition root, never from constants."""

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    bearer_token: SecretStr
    base_url: str = DEFAULT_BASE_URL
    endpoint_template: str = DEFAULT_ENDPOINT_TEMPLATE
    is_demo: bool = False

    @field_validator("bearer_token")
    @classmethod
    def token_must_not_be_empty(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("OpenAPI.it bearer token cannot be empty")
        return value

    @field_validator("base_url")
    @classmethod
    def base_url_must_be_http(cls, value: str) -> str:
        url = httpx.URL(value)
        if url.scheme not in {"http", "https"} or not url.host:
            raise ValueError("OpenAPI.it base URL must be an absolute HTTP(S) URL")
        return value.rstrip("/")

    @field_validator("endpoint_template")
    @classmethod
    def endpoint_must_contain_identifier(cls, value: str) -> str:
        if "{identifier}" not in value:
            raise ValueError("OpenAPI.it endpoint template must contain {identifier}")
        return value if value.startswith("/") else f"/{value}"


class OpenApiItLookup(BaseModel):
    """Strict lookup request; this adapter intentionally accepts Italian VAT IDs only."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    piva: str

    @field_validator("piva", mode="before")
    @classmethod
    def canonicalize_piva(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("P.IVA must be text")
        return normalize_piva(value)


class OpenApiItIngestionEnvelope(BaseModel):
    """Raw provider evidence plus the deliberately narrow canonical projection."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    source: Literal["openapi_it"] = SOURCE
    source_record_id: str
    piva: str
    observed_at: AwareDatetime
    raw_payload: dict[str, object]
    company: Company

    def to_mapped_record(self) -> MappedCompanyRecord:
        """Convert the provider-specific envelope at the shared ingestion boundary."""

        raw = RawRecordEnvelope(
            source=self.source,
            source_record_id=self.source_record_id,
            payload=self.raw_payload,
            observed_at=self.observed_at,
        )
        return MappedCompanyRecord(raw=raw, company=self.company)


class OpenApiItAdapterError(RuntimeError):
    """Classified failure for the workflow/DLQ layer; the adapter never retries itself."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        status_code: int | None = None,
        provider_error_code: int | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code
        self.provider_error_code = provider_error_code
        self.retry_after_seconds = retry_after_seconds


class OpenApiItSchemaError(OpenApiItAdapterError):
    """The provider response cannot safely be mapped to the canonical contract."""


class _ProviderModel(BaseModel):
    # OpenAPI.it's official OAS and product examples currently disagree on several
    # non-promoted fields (for example vatGroup/gruppo_iva). Unknown fields therefore
    # remain allowed and preserved in raw_payload, while every promoted field is strict.
    model_config = ConfigDict(
        strict=True,
        extra="allow",
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class _Region(_ProviderModel):
    description: str | None = None


class _RegisteredOffice(_ProviderModel):
    town: str | None = None
    province: str | None = None
    region: _Region | None = None


class _Address(_ProviderModel):
    registered_office: _RegisteredOffice | None = Field(default=None, alias="registeredOffice")


class _AtecoEntry(_ProviderModel):
    code: str | None = None
    description: str | None = None


class _AtecoClassification(_ProviderModel):
    ateco: _AtecoEntry | None = None
    ateco_2022: _AtecoEntry | None = Field(default=None, alias="ateco2022")
    ateco_2007: _AtecoEntry | None = Field(default=None, alias="ateco2007")


class _BalanceSheet(_ProviderModel):
    year: int | None = Field(default=None, ge=0)
    turnover: int | None = Field(default=None, ge=0)
    employees: int | None = Field(default=None, ge=0)


class _BalanceSheets(_ProviderModel):
    last: _BalanceSheet | None = None


class _ProviderCompany(_ProviderModel):
    provider_id: str | None = Field(default=None, alias="id")
    tax_code: str | None = Field(default=None, alias="taxCode")
    vat_code: str | None = Field(default=None, alias="vatCode")
    company_name: str | None = Field(default=None, alias="companyName")
    address: _Address | None = None
    activity_status: str | None = Field(default=None, alias="activityStatus")
    ateco_classification: _AtecoClassification | None = Field(
        default=None,
        alias="atecoClassification",
    )
    balance_sheets: _BalanceSheets | None = Field(default=None, alias="balanceSheets")
    last_update_timestamp: int | None = Field(
        default=None,
        alias="lastUpdateTimestamp",
        ge=0,
    )


class _ProviderResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", str_strip_whitespace=True)

    data: list[_ProviderCompany]
    success: bool
    message: str = ""
    error: int | None = None


_STATUS_MAP = {
    "ATTIVA": "active",
    "REGISTRATA": "registered",
    "INATTIVA": "inactive",
    "SOSPESA": "suspended",
    "IN_ISCRIZIONE": "registering",
    "CESSATA": "ceased",
}


def _is_retryable_status(status_code: int) -> bool:
    return status_code == 429 or 500 <= status_code < 600


def _retry_after_seconds(response: httpx.Response) -> float | None:
    raw_value = response.headers.get("Retry-After")
    if raw_value is None:
        return None
    try:
        value = float(raw_value)
    except ValueError:
        return None
    return value if value >= 0 else None


def _provider_error_details(response: httpx.Response) -> tuple[str | None, int | None]:
    try:
        payload = response.json()
    except ValueError:
        return None, None
    if not isinstance(payload, dict):
        return None, None
    raw_message = payload.get("message")
    raw_code = payload.get("error")
    message = raw_message if isinstance(raw_message, str) else None
    error_code = raw_code if isinstance(raw_code, int) and not isinstance(raw_code, bool) else None
    return message, error_code


def _raw_record(raw_payload: dict[str, object], index: int) -> dict[str, object]:
    raw_data = raw_payload.get("data")
    if not isinstance(raw_data, list) or index >= len(raw_data):
        raise OpenApiItSchemaError("OpenAPI.it response data is not a record list", retryable=False)
    item = raw_data[index]
    if not isinstance(item, dict):
        raise OpenApiItSchemaError("OpenAPI.it response record is not an object", retryable=False)
    return cast(dict[str, object], item)


def _to_company(record: _ProviderCompany, *, piva: str, is_demo: bool) -> Company:
    if not record.company_name:
        raise OpenApiItSchemaError("OpenAPI.it response has no company name", retryable=False)
    if record.last_update_timestamp is None:
        raise OpenApiItSchemaError(
            "OpenAPI.it response has no source update timestamp",
            retryable=False,
        )
    try:
        observed_at = datetime.fromtimestamp(record.last_update_timestamp, tz=UTC)
    except (OverflowError, OSError, ValueError) as error:
        raise OpenApiItSchemaError(
            "OpenAPI.it response has an invalid source update timestamp",
            retryable=False,
        ) from error

    office = record.address.registered_office if record.address else None
    ateco = record.ateco_classification
    ateco_entry = (ateco.ateco or ateco.ateco_2022 or ateco.ateco_2007) if ateco else None
    latest_balance = record.balance_sheets.last if record.balance_sheets else None
    status = _STATUS_MAP.get(record.activity_status or "", record.activity_status)

    return Company(
        piva=piva,
        legal_name=record.company_name,
        ateco_code=ateco_entry.code if ateco_entry else None,
        city=office.town if office else None,
        province=office.province if office else None,
        region=office.region.description if office and office.region else None,
        revenue_eur=(
            Decimal(latest_balance.turnover)
            if latest_balance and latest_balance.turnover is not None
            else None
        ),
        employees=latest_balance.employees if latest_balance else None,
        company_status=status,
        source=SOURCE,
        is_demo=is_demo,
        source_observed_at=observed_at,
    )


class OpenApiItAdapter:
    """Fetch one Company Advanced record using an injected asynchronous HTTP client."""

    def __init__(self, client: httpx.AsyncClient, config: OpenApiItConfig) -> None:
        self._client = client
        self._config = config

    async def fetch(self, lookup: OpenApiItLookup) -> OpenApiItIngestionEnvelope:
        endpoint = self._config.endpoint_template.format(identifier=lookup.piva)
        url = f"{self._config.base_url}{endpoint}"
        try:
            response = await self._client.get(
                url,
                headers={
                    "Accept": "application/json",
                    "Authorization": (f"Bearer {self._config.bearer_token.get_secret_value()}"),
                },
            )
        except httpx.RequestError as error:
            raise OpenApiItAdapterError(
                "OpenAPI.it transport failure",
                retryable=True,
            ) from error

        if response.status_code == 204:
            raise OpenApiItAdapterError(
                "OpenAPI.it returned no company data",
                retryable=False,
                status_code=204,
            )
        if not response.is_success:
            provider_message, provider_code = _provider_error_details(response)
            message = "OpenAPI.it request failed"
            if provider_message:
                message = f"{message}: {provider_message}"
            raise OpenApiItAdapterError(
                message,
                retryable=_is_retryable_status(response.status_code),
                status_code=response.status_code,
                provider_error_code=provider_code,
                retry_after_seconds=_retry_after_seconds(response),
            )

        try:
            json_payload = response.json()
        except ValueError as error:
            raise OpenApiItSchemaError(
                "OpenAPI.it returned invalid JSON",
                retryable=False,
                status_code=response.status_code,
            ) from error
        if not isinstance(json_payload, dict):
            raise OpenApiItSchemaError(
                "OpenAPI.it response root is not an object",
                retryable=False,
                status_code=response.status_code,
            )
        raw_payload = cast(dict[str, object], json_payload)
        try:
            payload = _ProviderResponse.model_validate(raw_payload)
        except ValidationError as error:
            raise OpenApiItSchemaError(
                "OpenAPI.it response does not match the documented envelope",
                retryable=False,
                status_code=response.status_code,
            ) from error
        if not payload.success:
            raise OpenApiItAdapterError(
                f"OpenAPI.it rejected the request: {payload.message}",
                retryable=False,
                status_code=response.status_code,
                provider_error_code=payload.error,
            )

        matches: list[tuple[int, _ProviderCompany]] = []
        for index, record in enumerate(payload.data):
            if record.vat_code is None:
                continue
            try:
                returned_piva = normalize_piva(record.vat_code)
            except InvalidPIVA as error:
                raise OpenApiItSchemaError(
                    "OpenAPI.it returned an invalid VAT identity",
                    retryable=False,
                    status_code=response.status_code,
                ) from error
            if returned_piva == lookup.piva:
                matches.append((index, record))

        if len(matches) != 1:
            raise OpenApiItSchemaError(
                "OpenAPI.it response must contain exactly one matching VAT identity",
                retryable=False,
                status_code=response.status_code,
            )
        record_index, record = matches[0]
        if not record.provider_id:
            raise OpenApiItSchemaError(
                "OpenAPI.it response has no provider record ID",
                retryable=False,
                status_code=response.status_code,
            )

        company = _to_company(record, piva=lookup.piva, is_demo=self._config.is_demo)
        return OpenApiItIngestionEnvelope(
            source=SOURCE,
            source_record_id=record.provider_id,
            piva=lookup.piva,
            observed_at=company.source_observed_at,
            raw_payload=_raw_record(raw_payload, record_index),
            company=company,
        )

    async def fetch_mapped(self, lookup: OpenApiItLookup) -> MappedSourceBatch:
        """Fetch one provider record and return the provider-neutral ingestion batch."""

        envelope = await self.fetch(lookup)
        return MappedSourceBatch(
            source=SOURCE,
            status=SourceReadStatus.SUCCESS,
            records=(envelope.to_mapped_record(),),
        )


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_ENDPOINT_TEMPLATE",
    "SOURCE",
    "OpenApiItAdapter",
    "OpenApiItAdapterError",
    "OpenApiItConfig",
    "OpenApiItIngestionEnvelope",
    "OpenApiItLookup",
    "OpenApiItSchemaError",
]
