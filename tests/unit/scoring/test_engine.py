from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from list_engine.core.models import Company
from list_engine.scoring.engine import (
    allocate_capacity,
    hot_event_recalculation_due,
    score_company,
    weekly_due,
)
from list_engine.scoring.models import (
    FitWeights,
    NumericBand,
    PhoneKind,
    ReachabilityWeights,
    ReadinessSignal,
    RecalculationReason,
    RubricWeights,
    ScoreBreakdown,
    ScoreFeatures,
    ScoreResult,
    SegmentConfig,
    SignalWeights,
    Tier,
    TierCapacity,
)

AS_OF = datetime(2026, 7, 15, 9, 0, tzinfo=UTC)


def segment_config() -> SegmentConfig:
    return SegmentConfig(
        schema_version=1,
        segment_id="logistica_nord",
        version="2026-07-v1",
        status="hypothesis",
        persona="Responsabile amministrativo o titolare",
        ateco_prefixes=("49", "52"),
        revenue_band_eur=NumericBand(
            minimum=1_000_000,
            sweet_minimum=3_000_000,
            sweet_maximum=8_000_000,
            maximum=10_000_000,
        ),
        employee_band=NumericBand(
            minimum=5,
            sweet_minimum=10,
            sweet_maximum=50,
            maximum=100,
        ),
        target_regions=("Lombardia", "Veneto"),
        excluded_company_statuses=("cessata", "procedura concorsuale"),
        hot_signal_types=("new_office", "governance_change"),
        signal_lookback_days=90,
        growth_full_score_at=0.20,
        multi_source_full_score_at=3,
        weights=RubricWeights(
            fit=FitWeights(
                ateco=10,
                revenue=10,
                structure=10,
                digital_maturity=5,
                geography=5,
            ),
            signal=SignalWeights(
                fresh_trigger=15,
                growth=10,
                administrative_hiring=10,
                engagement=5,
            ),
            reachability=ReachabilityWeights(
                decision_maker=8,
                phone=6,
                email=3,
                multi_source=3,
            ),
        ),
        capacity=TierCapacity(t1_per_bdr=30, t2_per_bdr=60),
    )


def company(**overrides: object) -> Company:
    data: dict[str, object] = {
        "piva": "99000000002",
        "legal_name": "Aurora Logistica Demo S.r.l.",
        "ateco_code": "49.41",
        "region": "Lombardia",
        "revenue_eur": Decimal("5000000"),
        "employees": 25,
        "source": "demo",
        "is_demo": True,
        "source_observed_at": AS_OF - timedelta(days=1),
    }
    data.update(overrides)
    return Company.model_validate(data)


def hot_signal(
    *,
    signal_type: str = "new_office",
    occurred_at: datetime = AS_OF - timedelta(days=1),
    verified: bool = True,
    confidence: float = 1.0,
) -> ReadinessSignal:
    return ReadinessSignal(
        signal_type=signal_type,
        occurred_at=occurred_at,
        confidence=confidence,
        verified=verified,
    )


def perfect_features(**overrides: object) -> ScoreFeatures:
    data: dict[str, object] = {
        "company": company(),
        "administrative_function_verified": True,
        "digital_maturity": 1.0,
        "signals": (hot_signal(),),
        "revenue_growth": 0.20,
        "administrative_hiring_verified": True,
        "engagement_level": 1.0,
        "decision_maker_verified": True,
        "phone_kind": PhoneKind.MOBILE,
        "phone_verified": True,
        "email_verified": True,
        "source_count": 3,
    }
    data.update(overrides)
    return ScoreFeatures.model_validate(data)


def test_full_rubric_scores_100_and_records_every_component() -> None:
    result = score_company(
        perfect_features(),
        segment_config(),
        as_of=AS_OF,
        reason=RecalculationReason.HOT_EVENT,
    )

    assert result.fit_score == 40
    assert result.signal_score == 40
    assert result.reachability_score == 20
    assert result.total_score == 100
    assert result.active_trigger
    assert result.eligible_tier is Tier.T1
    assert result.scoring_version == "logistica_nord@2026-07-v1"
    assert result.scored_at == AS_OF
    assert result.recalculation_reason is RecalculationReason.HOT_EVENT
    assert result.breakdown.fit == {
        "ateco": 10,
        "revenue": 10,
        "structure": 10,
        "digital_maturity": 5,
        "geography": 5,
    }
    assert result.breakdown.signal == {
        "fresh_trigger": 15,
        "growth": 10,
        "administrative_hiring": 10,
        "engagement": 5,
    }
    assert result.breakdown.reachability == {
        "decision_maker": 8,
        "phone": 6,
        "email": 3,
        "multi_source": 3,
    }


def test_interpolation_and_each_component_use_round_half_up() -> None:
    features = ScoreFeatures(
        company=company(
            revenue_eur=Decimal("2000000"),
            employees=5,
            ateco_code="01.11",
            region="Piemonte",
        ),
        digital_maturity=0.5,
        revenue_growth=0.05,
        engagement_level=0.5,
        phone_kind=PhoneKind.SWITCHBOARD,
        phone_verified=True,
        source_count=2,
    )

    result = score_company(features, segment_config(), as_of=AS_OF)

    assert result.breakdown.fit == {
        "ateco": 0,
        "revenue": 5,
        "structure": 0,
        "digital_maturity": 3,
        "geography": 0,
    }
    assert result.breakdown.signal == {
        "fresh_trigger": 0,
        "growth": 3,
        "administrative_hiring": 0,
        "engagement": 3,
    }
    assert result.breakdown.reachability == {
        "decision_maker": 0,
        "phone": 4,
        "email": 0,
        "multi_source": 2,
    }
    assert result.total_score == 20
    assert result.eligible_tier is Tier.UNQUALIFIED


@pytest.mark.parametrize(
    ("revenue", "expected_points"),
    [
        (Decimal("999999"), 0),
        (Decimal("1000000"), 0),
        (Decimal("3000000"), 10),
        (Decimal("8000000"), 10),
        (Decimal("9000000"), 5),
        (Decimal("10000000"), 0),
        (Decimal("10000001"), 0),
    ],
)
def test_revenue_band_has_linear_shoulders_and_full_sweet_spot(
    revenue: Decimal,
    expected_points: int,
) -> None:
    features = ScoreFeatures(
        company=company(
            ateco_code=None,
            revenue_eur=revenue,
            employees=None,
            region=None,
        ),
        source_count=0,
    )

    result = score_company(features, segment_config(), as_of=AS_OF)

    assert result.breakdown.fit["revenue"] == expected_points


def test_verified_administrative_function_fills_structure_when_employees_do_not() -> None:
    features = ScoreFeatures(
        company=company(employees=None),
        administrative_function_verified=True,
    )

    result = score_company(features, segment_config(), as_of=AS_OF)

    assert result.breakdown.fit["structure"] == 10


@pytest.mark.parametrize(
    ("signal", "expected"),
    [
        (hot_signal(occurred_at=AS_OF - timedelta(days=90)), True),
        (hot_signal(occurred_at=AS_OF - timedelta(days=90, seconds=1)), False),
        (hot_signal(occurred_at=AS_OF + timedelta(microseconds=1)), False),
        (hot_signal(verified=False), False),
        (hot_signal(confidence=0), False),
        (hot_signal(signal_type="unconfigured_event"), False),
        (hot_signal(signal_type=" NEW_OFFICE "), True),
    ],
)
def test_trigger_must_be_verified_configured_and_current(
    signal: ReadinessSignal,
    expected: bool,
) -> None:
    result = score_company(
        perfect_features(signals=(signal,)),
        segment_config(),
        as_of=AS_OF,
    )

    assert result.active_trigger is expected
    assert result.breakdown.signal["fresh_trigger"] == (15 if expected else 0)


def test_fresh_trigger_uses_the_strongest_verified_confidence() -> None:
    result = score_company(
        perfect_features(
            signals=(hot_signal(confidence=0.2), hot_signal(confidence=0.5)),
        ),
        segment_config(),
        as_of=AS_OF,
    )

    assert result.active_trigger
    assert result.breakdown.signal["fresh_trigger"] == 8


def test_tier_thresholds_include_no_trigger_high_score_fallback_to_t2() -> None:
    config = segment_config()
    unqualified = score_company(
        ScoreFeatures(
            company=company(
                ateco_code=None,
                revenue_eur=None,
                employees=None,
                region=None,
            ),
            source_count=0,
        ),
        config,
        as_of=AS_OF,
    )
    t3 = score_company(
        ScoreFeatures(company=company(), digital_maturity=0),
        config,
        as_of=AS_OF,
    )
    t2 = score_company(
        perfect_features(signals=()),
        config,
        as_of=AS_OF,
    )
    t1 = score_company(perfect_features(), config, as_of=AS_OF)

    assert (unqualified.total_score, unqualified.eligible_tier) == (0, Tier.UNQUALIFIED)
    assert (t3.total_score, t3.eligible_tier) == (36, Tier.T3)
    assert t2.total_score == 85
    assert not t2.active_trigger
    assert t2.eligible_tier is Tier.T2
    assert (t1.total_score, t1.eligible_tier) == (100, Tier.T1)


def test_suppression_and_normalized_excluded_status_are_zero_score_t0() -> None:
    result = score_company(
        perfect_features(
            company=company(company_status="CESSATA"),
            suppressed=True,
            suppression_reason="RPO hit",
        ),
        segment_config(),
        as_of=AS_OF,
    )

    assert result.eligible_tier is Tier.T0
    assert result.total_score == 0
    assert result.breakdown == ScoreBreakdown(
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
    assert result.exclusion_reasons == (
        "suppression:RPO hit",
        "excluded_company_status:CESSATA",
    )
    assert not result.active_trigger


def score_result(piva: str, total: int, tier: Tier) -> ScoreResult:
    fit = min(total, 40)
    signal = min(total - fit, 40)
    reachability = total - fit - signal
    return ScoreResult(
        piva=piva,
        scoring_version=segment_config().scoring_version,
        fit_score=fit,
        signal_score=signal,
        reachability_score=reachability,
        total_score=total,
        eligible_tier=tier,
        active_trigger=tier is Tier.T1,
        exclusion_reasons=("suppression:test",) if tier is Tier.T0 else (),
        breakdown=ScoreBreakdown(fit={}, signal={}, reachability={}),
        scored_at=AS_OF,
        recalculation_reason=RecalculationReason.WEEKLY,
    )


def test_capacity_allocator_caps_t1_then_t2_and_overflows_downward() -> None:
    t1_scores = [score_result(f"1{index:010d}", 100 - index, Tier.T1) for index in range(31)]
    t2_scores = [
        score_result(f"2{index:010d}", 69 - min(index, 19), Tier.T2) for index in range(61)
    ]

    assignments = allocate_capacity(
        reversed([*t2_scores, *t1_scores]),
        segment_config(),
        bdr_count=1,
    )

    assigned_t1 = [item for item in assignments if item.assigned_tier is Tier.T1]
    assigned_t2 = [item for item in assignments if item.assigned_tier is Tier.T2]
    assigned_t3 = [item for item in assignments if item.assigned_tier is Tier.T3]
    assert len(assigned_t1) == 30
    assert len(assigned_t2) == 60
    assert len(assigned_t3) == 2
    assert [item.queue_rank for item in assigned_t1] == list(range(1, 31))
    assert [item.queue_rank for item in assigned_t2] == list(range(1, 61))
    assert [item.queue_rank for item in assigned_t3] == [1, 2]

    t1_overflow = next(item for item in assignments if item.score.piva == "10000000030")
    assert t1_overflow.score.eligible_tier is Tier.T1
    assert t1_overflow.assigned_tier is Tier.T2
    assert t1_overflow.capacity_reason == "T1 capacity exhausted; downgraded to T2"
    assert t1_overflow.queue_rank == 1
    assert all(
        item.capacity_reason == "T2 capacity exhausted; downgraded to T3" for item in assigned_t3
    )


def test_capacity_ties_use_piva_and_non_actionable_tiers_have_no_rank() -> None:
    tied = [score_result(f"3{index:010d}", 80, Tier.T1) for index in range(30, -1, -1)]
    t0 = score_result("90000000000", 0, Tier.T0)
    unqualified = score_result("80000000000", 0, Tier.UNQUALIFIED)

    assignments = allocate_capacity(
        [unqualified, *tied, t0],
        segment_config(),
        bdr_count=1,
    )

    assigned_t1 = [item for item in assignments if item.assigned_tier is Tier.T1]
    assert [item.score.piva for item in assigned_t1] == [f"3{index:010d}" for index in range(30)]
    overflow = next(item for item in assignments if item.score.piva == "30000000030")
    assert overflow.assigned_tier is Tier.T2
    assert overflow.queue_rank == 1
    assert all(
        item.queue_rank is None
        for item in assignments
        if item.assigned_tier in {Tier.T0, Tier.UNQUALIFIED}
    )


def test_zero_bdrs_overflows_every_capacity_bound_tier_to_t3() -> None:
    assignments = allocate_capacity(
        [
            score_result("10000000001", 80, Tier.T1),
            score_result("20000000001", 60, Tier.T2),
        ],
        segment_config(),
        bdr_count=0,
    )

    assert [item.assigned_tier for item in assignments] == [Tier.T3, Tier.T3]
    assert assignments[0].capacity_reason == ("T1 and T2 capacity exhausted; downgraded to T3")
    assert assignments[1].capacity_reason == "T2 capacity exhausted; downgraded to T3"


def test_allocator_rejects_ambiguous_inputs() -> None:
    score = score_result("10000000001", 80, Tier.T1)

    with pytest.raises(ValueError, match="one result"):
        allocate_capacity([score, score], segment_config(), bdr_count=1)
    with pytest.raises(ValueError, match="non-negative"):
        allocate_capacity([score], segment_config(), bdr_count=-1)
    with pytest.raises(ValueError, match="non-negative"):
        allocate_capacity([score], segment_config(), bdr_count=True)
    with pytest.raises(ValueError, match="non-negative"):
        allocate_capacity([score], segment_config(), bdr_count=1.5)  # type: ignore[arg-type]


def test_weekly_hook_has_explicit_replayable_boundaries() -> None:
    assert weekly_due(None, as_of=AS_OF)
    assert not weekly_due(AS_OF - timedelta(days=6, hours=23), as_of=AS_OF)
    assert weekly_due(AS_OF - timedelta(days=7), as_of=AS_OF)
    assert not weekly_due(AS_OF + timedelta(seconds=1), as_of=AS_OF)

    with pytest.raises(ValueError, match="timezone-aware"):
        weekly_due(datetime(2026, 7, 1), as_of=AS_OF)
    with pytest.raises(ValueError, match="timezone-aware"):
        weekly_due(None, as_of=datetime(2026, 7, 15))


def test_hot_event_hook_reuses_the_same_freshness_gate_as_scoring() -> None:
    config = segment_config()

    assert hot_event_recalculation_due(
        hot_signal(occurred_at=AS_OF - timedelta(days=90)),
        config,
        as_of=AS_OF,
    )
    assert not hot_event_recalculation_due(
        hot_signal(occurred_at=AS_OF + timedelta(seconds=1)),
        config,
        as_of=AS_OF,
    )
    assert not hot_event_recalculation_due(
        hot_signal(verified=False),
        config,
        as_of=AS_OF,
    )


def test_score_rejects_implicit_local_time() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        score_company(
            perfect_features(),
            segment_config(),
            as_of=datetime(2026, 7, 15, 9, 0),
        )
