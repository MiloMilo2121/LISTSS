"""Deterministic scoring, tier allocation, and recalculation hooks."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal

from list_engine.scoring.models import (
    NumericBand,
    PhoneKind,
    ReadinessSignal,
    RecalculationReason,
    ScoreBreakdown,
    ScoreFeatures,
    ScoreResult,
    SegmentConfig,
    Tier,
    TierAssignment,
)

_ZERO = Decimal(0)
_ONE = Decimal(1)
_SWITCHBOARD_SHARE = Decimal(4) / Decimal(6)
_WEEKLY_CADENCE = timedelta(days=7)


def _require_aware(value: datetime, *, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")


def _clamp_ratio(value: Decimal) -> Decimal:
    return min(_ONE, max(_ZERO, value))


def _points(weight: int, ratio: Decimal | float | int | bool) -> int:
    """Round each component independently so totals are replayable across runtimes."""

    decimal_ratio = (
        ratio
        if isinstance(ratio, Decimal)
        else Decimal(str(int(ratio) if isinstance(ratio, bool) else ratio))
    )
    result = Decimal(weight) * _clamp_ratio(decimal_ratio)
    return int(result.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _band_ratio(value: Decimal | int | None, band: NumericBand) -> Decimal:
    if value is None:
        return _ZERO

    observed = Decimal(value)
    minimum = Decimal(band.minimum)
    sweet_minimum = Decimal(band.sweet_minimum)
    sweet_maximum = Decimal(band.sweet_maximum)
    maximum = Decimal(band.maximum)

    if observed < minimum or observed > maximum:
        return _ZERO
    if sweet_minimum <= observed <= sweet_maximum:
        return _ONE
    if observed < sweet_minimum:
        if sweet_minimum == minimum:
            return _ONE
        return (observed - minimum) / (sweet_minimum - minimum)
    if sweet_maximum == maximum:
        return _ONE
    return (maximum - observed) / (maximum - sweet_maximum)


def _normalized(value: str) -> str:
    return value.strip().casefold()


def _is_active_trigger(
    signal: ReadinessSignal,
    config: SegmentConfig,
    *,
    as_of: datetime,
) -> bool:
    _require_aware(as_of, field="as_of")
    if not signal.verified or signal.confidence <= 0:
        return False
    configured_types = {_normalized(value) for value in config.hot_signal_types}
    if _normalized(signal.signal_type) not in configured_types:
        return False
    age = as_of - signal.occurred_at
    return timedelta(0) <= age <= config.signal_lookback


def _exclusion_reasons(features: ScoreFeatures, config: SegmentConfig) -> tuple[str, ...]:
    reasons: list[str] = []
    if features.suppressed:
        reasons.append(f"suppression:{features.suppression_reason}")

    status = features.company.company_status
    excluded_statuses = {_normalized(value) for value in config.excluded_company_statuses}
    if status is not None and _normalized(status) in excluded_statuses:
        reasons.append(f"excluded_company_status:{status}")
    return tuple(reasons)


def _zero_breakdown() -> ScoreBreakdown:
    return ScoreBreakdown(
        fit={
            "ateco": 0,
            "revenue": 0,
            "structure": 0,
            "digital_maturity": 0,
            "geography": 0,
        },
        signal={
            "fresh_trigger": 0,
            "growth": 0,
            "administrative_hiring": 0,
            "engagement": 0,
        },
        reachability={
            "decision_maker": 0,
            "phone": 0,
            "email": 0,
            "multi_source": 0,
        },
    )


def score_company(
    features: ScoreFeatures,
    config: SegmentConfig,
    *,
    as_of: datetime,
    reason: RecalculationReason = RecalculationReason.MANUAL,
) -> ScoreResult:
    """Apply the versioned 40/40/20 rubric without I/O or implicit time."""

    _require_aware(as_of, field="as_of")
    exclusions = _exclusion_reasons(features, config)
    if exclusions:
        return ScoreResult(
            piva=features.company.piva,
            scoring_version=config.scoring_version,
            fit_score=0,
            signal_score=0,
            reachability_score=0,
            total_score=0,
            eligible_tier=Tier.T0,
            active_trigger=False,
            exclusion_reasons=exclusions,
            breakdown=_zero_breakdown(),
            scored_at=as_of,
            recalculation_reason=reason,
        )

    fit_weights = config.weights.fit
    ateco = features.company.ateco_code
    ateco_ratio = ateco is not None and any(
        ateco.startswith(prefix.strip()) for prefix in config.ateco_prefixes
    )
    employee_ratio = _band_ratio(features.company.employees, config.employee_band)
    structure_ratio = max(employee_ratio, Decimal(int(features.administrative_function_verified)))
    region = features.company.region
    geography_ratio = region is not None and _normalized(region) in {
        _normalized(value) for value in config.target_regions
    }
    fit = {
        "ateco": _points(fit_weights.ateco, ateco_ratio),
        "revenue": _points(
            fit_weights.revenue,
            _band_ratio(features.company.revenue_eur, config.revenue_band_eur),
        ),
        "structure": _points(fit_weights.structure, structure_ratio),
        "digital_maturity": _points(fit_weights.digital_maturity, features.digital_maturity),
        "geography": _points(fit_weights.geography, geography_ratio),
    }

    trigger_confidence = max(
        (
            Decimal(str(signal.confidence))
            for signal in features.signals
            if _is_active_trigger(signal, config, as_of=as_of)
        ),
        default=_ZERO,
    )
    active_trigger = trigger_confidence > 0
    growth_candidates = [
        value for value in (features.revenue_growth, features.employee_growth) if value is not None
    ]
    strongest_growth = max([0.0, *growth_candidates])
    growth_ratio = Decimal(str(strongest_growth)) / Decimal(str(config.growth_full_score_at))
    signal_weights = config.weights.signal
    signal = {
        "fresh_trigger": _points(signal_weights.fresh_trigger, trigger_confidence),
        "growth": _points(signal_weights.growth, growth_ratio),
        "administrative_hiring": _points(
            signal_weights.administrative_hiring,
            features.administrative_hiring_verified,
        ),
        "engagement": _points(signal_weights.engagement, features.engagement_level),
    }

    reachability_weights = config.weights.reachability
    phone_ratio = {
        PhoneKind.NONE: _ZERO,
        PhoneKind.SWITCHBOARD: _SWITCHBOARD_SHARE,
        PhoneKind.MOBILE: _ONE,
    }[features.phone_kind]
    reachability = {
        "decision_maker": _points(
            reachability_weights.decision_maker,
            features.decision_maker_verified,
        ),
        "phone": _points(reachability_weights.phone, phone_ratio),
        "email": _points(reachability_weights.email, features.email_verified),
        "multi_source": _points(
            reachability_weights.multi_source,
            Decimal(features.source_count) / Decimal(config.multi_source_full_score_at),
        ),
    }

    fit_score = sum(fit.values())
    signal_score = sum(signal.values())
    reachability_score = sum(reachability.values())
    total_score = fit_score + signal_score + reachability_score
    if total_score >= 70 and active_trigger:
        eligible_tier = Tier.T1
    elif total_score >= 50:
        eligible_tier = Tier.T2
    elif total_score >= 30:
        eligible_tier = Tier.T3
    else:
        eligible_tier = Tier.UNQUALIFIED

    return ScoreResult(
        piva=features.company.piva,
        scoring_version=config.scoring_version,
        fit_score=fit_score,
        signal_score=signal_score,
        reachability_score=reachability_score,
        total_score=total_score,
        eligible_tier=eligible_tier,
        active_trigger=active_trigger,
        breakdown=ScoreBreakdown(fit=fit, signal=signal, reachability=reachability),
        scored_at=as_of,
        recalculation_reason=reason,
    )


def _score_order(score: ScoreResult) -> tuple[int, str]:
    return (-score.total_score, score.piva)


def allocate_capacity(
    scores: Iterable[ScoreResult],
    config: SegmentConfig,
    *,
    bdr_count: int,
) -> tuple[TierAssignment, ...]:
    """Allocate T1 then T2 capacity, deterministically overflowing downward."""

    if not isinstance(bdr_count, int) or isinstance(bdr_count, bool) or bdr_count < 0:
        raise ValueError("bdr_count must be a non-negative integer")
    materialized = tuple(scores)
    pivas = [score.piva for score in materialized]
    if len(pivas) != len(set(pivas)):
        raise ValueError("scores must contain at most one result per P.IVA")
    if any(score.scoring_version != config.scoring_version for score in materialized):
        raise ValueError("all scores must use the supplied segment scoring version")

    by_eligible = {
        tier: sorted(
            (score for score in materialized if score.eligible_tier is tier),
            key=_score_order,
        )
        for tier in Tier
    }
    t1_capacity = config.capacity.t1_per_bdr * bdr_count
    t2_capacity = config.capacity.t2_per_bdr * bdr_count

    assigned_t1 = by_eligible[Tier.T1][:t1_capacity]
    overflow_t1 = by_eligible[Tier.T1][t1_capacity:]
    t2_candidates = sorted([*overflow_t1, *by_eligible[Tier.T2]], key=_score_order)
    assigned_t2 = t2_candidates[:t2_capacity]
    overflow_t2 = t2_candidates[t2_capacity:]
    assigned_t3 = sorted([*overflow_t2, *by_eligible[Tier.T3]], key=_score_order)

    assignments: list[TierAssignment] = []
    for rank, score in enumerate(assigned_t1, start=1):
        assignments.append(TierAssignment(score=score, assigned_tier=Tier.T1, queue_rank=rank))
    for rank, score in enumerate(assigned_t2, start=1):
        reason = None
        if score.eligible_tier is Tier.T1:
            reason = "T1 capacity exhausted; downgraded to T2"
        assignments.append(
            TierAssignment(
                score=score,
                assigned_tier=Tier.T2,
                queue_rank=rank,
                capacity_reason=reason,
            )
        )
    for rank, score in enumerate(assigned_t3, start=1):
        reason = None
        if score.eligible_tier is Tier.T1:
            reason = "T1 and T2 capacity exhausted; downgraded to T3"
        elif score.eligible_tier is Tier.T2:
            reason = "T2 capacity exhausted; downgraded to T3"
        assignments.append(
            TierAssignment(
                score=score,
                assigned_tier=Tier.T3,
                queue_rank=rank,
                capacity_reason=reason,
            )
        )
    for tier in (Tier.T0, Tier.UNQUALIFIED):
        assignments.extend(
            TierAssignment(score=score, assigned_tier=tier) for score in by_eligible[tier]
        )
    return tuple(assignments)


def weekly_due(last_scored_at: datetime | None, *, as_of: datetime) -> bool:
    """Return whether the fixed weekly batch cadence is due."""

    _require_aware(as_of, field="as_of")
    if last_scored_at is None:
        return True
    _require_aware(last_scored_at, field="last_scored_at")
    return as_of - last_scored_at >= _WEEKLY_CADENCE


def hot_event_recalculation_due(
    signal: ReadinessSignal,
    config: SegmentConfig,
    *,
    as_of: datetime,
) -> bool:
    """Event hook: only a current, verified, configured hot signal can enqueue a rescore."""

    return _is_active_trigger(signal, config, as_of=as_of)


__all__ = [
    "allocate_capacity",
    "hot_event_recalculation_due",
    "score_company",
    "weekly_due",
]
