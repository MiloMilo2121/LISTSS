"""Core models kept independent from infrastructure and provider payloads."""

from __future__ import annotations

from decimal import Decimal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

from list_engine.core.piva import normalize_piva


class Company(BaseModel):
    """The current golden record for an Italian company."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True, extra="forbid", strict=True)

    piva: str
    legal_name: str = Field(min_length=1, max_length=255)
    website: str | None = None
    ateco_code: str | None = None
    city: str | None = None
    province: str | None = Field(default=None, min_length=2, max_length=2)
    region: str | None = None
    revenue_eur: Decimal | None = Field(default=None, ge=0)
    employees: int | None = Field(default=None, ge=0)
    company_status: str | None = None
    source: str = Field(min_length=1)
    is_demo: bool = False
    source_observed_at: AwareDatetime

    @field_validator("piva", mode="before")
    @classmethod
    def validate_piva(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("P.IVA must be text")
        return normalize_piva(value)

    @field_validator("province", mode="after")
    @classmethod
    def uppercase_province(cls, value: str | None) -> str | None:
        return value.upper() if value else None
