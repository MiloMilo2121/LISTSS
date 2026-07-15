"""Strict inputs and outputs for the deterministic scoring rubric."""

from __future__ import annotations

import re
from datetime import timedelta
from enum import StrEnum
from typing import Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from list_engine.core.models import Company

_IDENTIFIER = re.compile(r"[a-z][a-z0-9_-]{1,63}")


class Tier(StrEnum):
    T0 = "T0"
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"
    UNQUALIFIED = "UNQUALIFIED"


class PhoneKind(StrEnum):
    NONE = "none"
    SWITCHBOARD = "switchboard"
    MOBILE = "mobile"


class RecalculationReason(StrEnum):
    WEEKLY = "weekly"
    HOT_EVENT = "hot_event"
    MANUAL = "manual"


class NumericBand(BaseModel):
    """Inclusive outer bounds with a full-score sweet spot."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    minimum: int = Field(ge=0)
    sweet_minimum: int = Field(ge=0)
    sweet_maximum: int = Field(ge=0)
    maximum: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        if not (self.minimum <= self.sweet_minimum <= self.sweet_maximum <= self.maximum):
            raise ValueError("numeric band bounds must be monotonic")
        if self.minimum == self.maximum:
            raise ValueError("numeric band must have a non-zero range")
        return self


class FitWeights(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    ateco: int = Field(ge=0)
    revenue: int = Field(ge=0)
    structure: int = Field(ge=0)
    digital_maturity: int = Field(ge=0)
    geography: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_total(self) -> Self:
        if sum(self.model_dump().values()) != 40:
            raise ValueError("fit weights must sum to 40")
        return self


class SignalWeights(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    fresh_trigger: int = Field(ge=0)
    growth: int = Field(ge=0)
    administrative_hiring: int = Field(ge=0)
    engagement: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_total(self) -> Self:
        if sum(self.model_dump().values()) != 40:
            raise ValueError("signal weights must sum to 40")
        return self


class ReachabilityWeights(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    decision_maker: int = Field(ge=0)
    phone: int = Field(ge=0)
    email: int = Field(ge=0)
    multi_source: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_total(self) -> Self:
        if sum(self.model_dump().values()) != 20:
            raise ValueError("reachability weights must sum to 20")
        return self


class RubricWeights(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    fit: FitWeights
    signal: SignalWeights
    reachability: ReachabilityWeights


class TierCapacity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    t1_per_bdr: int = Field(ge=30, le=40)
    t2_per_bdr: int = Field(ge=60, le=80)


class SegmentConfig(BaseModel):
    """Versioned segment hypothesis loaded from a reviewable YAML file."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal[1]
    segment_id: str
    version: str = Field(min_length=1, max_length=64)
    status: Literal["hypothesis", "validated"]
    persona: str = Field(min_length=1, max_length=255)
    ateco_prefixes: tuple[str, ...] = Field(min_length=1)
    revenue_band_eur: NumericBand
    employee_band: NumericBand
    target_regions: tuple[str, ...] = Field(min_length=1)
    excluded_company_statuses: tuple[str, ...] = Field(min_length=1)
    hot_signal_types: tuple[str, ...] = Field(min_length=1)
    signal_lookback_days: int = Field(default=90, ge=1, le=365)
    growth_full_score_at: float = Field(default=0.20, gt=0, le=10)
    multi_source_full_score_at: int = Field(default=3, ge=2, le=10)
    weights: RubricWeights
    capacity: TierCapacity

    @field_validator("segment_id")
    @classmethod
    def validate_segment_id(cls, value: str) -> str:
        if _IDENTIFIER.fullmatch(value) is None:
            raise ValueError("segment_id must be a lowercase ASCII identifier")
        return value

    @field_validator(
        "ateco_prefixes",
        "target_regions",
        "excluded_company_statuses",
        "hot_signal_types",
    )
    @classmethod
    def validate_unique_values(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() for value in values):
            raise ValueError("segment lists cannot contain blank values")
        if len(set(values)) != len(values):
            raise ValueError("segment lists cannot contain duplicates")
        return values

    @property
    def scoring_version(self) -> str:
        return f"{self.segment_id}@{self.version}"

    @property
    def signal_lookback(self) -> timedelta:
        return timedelta(days=self.signal_lookback_days)


class ReadinessSignal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    signal_type: str = Field(min_length=1, max_length=128)
    occurred_at: AwareDatetime
    confidence: float = Field(ge=0, le=1)
    verified: bool


class ScoreFeatures(BaseModel):
    """Only verified or explicitly nullable facts consumed by the rubric."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    company: Company
    suppressed: bool = False
    suppression_reason: str | None = Field(default=None, min_length=1, max_length=255)
    administrative_function_verified: bool = False
    digital_maturity: float = Field(default=0, ge=0, le=1)
    signals: tuple[ReadinessSignal, ...] = ()
    revenue_growth: float | None = Field(default=None, ge=-1, le=10)
    employee_growth: float | None = Field(default=None, ge=-1, le=10)
    administrative_hiring_verified: bool = False
    engagement_level: float = Field(default=0, ge=0, le=1)
    decision_maker_verified: bool = False
    phone_kind: PhoneKind = PhoneKind.NONE
    phone_verified: bool = False
    email_verified: bool = False
    source_count: int = Field(default=1, ge=0)

    @model_validator(mode="after")
    def validate_suppression_and_phone(self) -> Self:
        if self.suppressed != (self.suppression_reason is not None):
            raise ValueError("suppression_reason is required exactly when suppressed")
        if not self.phone_verified and self.phone_kind is not PhoneKind.NONE:
            raise ValueError("an unverified phone cannot contribute a phone kind")
        if self.phone_verified and self.phone_kind is PhoneKind.NONE:
            raise ValueError("a verified phone requires its phone kind")
        return self


class ScoreBreakdown(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    fit: dict[str, int]
    signal: dict[str, int]
    reachability: dict[str, int]


class ScoreResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    piva: str
    scoring_version: str
    fit_score: int = Field(ge=0, le=40)
    signal_score: int = Field(ge=0, le=40)
    reachability_score: int = Field(ge=0, le=20)
    total_score: int = Field(ge=0, le=100)
    eligible_tier: Tier
    active_trigger: bool
    exclusion_reasons: tuple[str, ...] = ()
    breakdown: ScoreBreakdown
    scored_at: AwareDatetime
    recalculation_reason: RecalculationReason

    @model_validator(mode="after")
    def validate_total(self) -> Self:
        if self.total_score != self.fit_score + self.signal_score + self.reachability_score:
            raise ValueError("total_score must equal the three rubric blocks")
        if self.eligible_tier is Tier.T0 and not self.exclusion_reasons:
            raise ValueError("T0 scores require an exclusion reason")
        if self.eligible_tier is not Tier.T0 and self.exclusion_reasons:
            raise ValueError("only T0 scores may carry exclusion reasons")
        return self


class TierAssignment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    score: ScoreResult
    assigned_tier: Tier
    queue_rank: int | None = Field(default=None, ge=1)
    capacity_reason: str | None = Field(default=None, min_length=1, max_length=255)

    @model_validator(mode="after")
    def validate_capacity_transition(self) -> Self:
        allowed = {
            Tier.T0: {Tier.T0},
            Tier.T1: {Tier.T1, Tier.T2, Tier.T3},
            Tier.T2: {Tier.T2, Tier.T3},
            Tier.T3: {Tier.T3},
            Tier.UNQUALIFIED: {Tier.UNQUALIFIED},
        }
        if self.assigned_tier not in allowed[self.score.eligible_tier]:
            raise ValueError("capacity allocation may only preserve or lower an eligible tier")

        is_actionable = self.assigned_tier in {Tier.T1, Tier.T2, Tier.T3}
        if is_actionable != (self.queue_rank is not None):
            raise ValueError("only actionable tier assignments require a queue rank")

        was_downgraded = self.assigned_tier is not self.score.eligible_tier
        if was_downgraded != (self.capacity_reason is not None):
            raise ValueError("capacity_reason is required exactly for a tier downgrade")
        return self


__all__ = [
    "FitWeights",
    "NumericBand",
    "PhoneKind",
    "ReachabilityWeights",
    "ReadinessSignal",
    "RecalculationReason",
    "RubricWeights",
    "ScoreBreakdown",
    "ScoreFeatures",
    "ScoreResult",
    "SegmentConfig",
    "SignalWeights",
    "Tier",
    "TierAssignment",
    "TierCapacity",
]
