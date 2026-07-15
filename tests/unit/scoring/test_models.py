from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from list_engine.core.models import Company
from list_engine.scoring.models import (
    FitWeights,
    PhoneKind,
    ReachabilityWeights,
    RecalculationReason,
    RubricWeights,
    ScoreBreakdown,
    ScoreFeatures,
    ScoreResult,
    SignalWeights,
    Tier,
    TierAssignment,
)


def company() -> Company:
    return Company(
        piva="99000000002",
        legal_name="Scoring Model Demo S.r.l.",
        source="demo",
        is_demo=True,
        source_observed_at=datetime(2026, 7, 1, tzinfo=UTC),
    )


def weights() -> RubricWeights:
    return RubricWeights(
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
    )


def test_rubric_blocks_must_keep_the_40_40_20_contract() -> None:
    assert sum(weights().fit.model_dump().values()) == 40

    with pytest.raises(ValidationError, match="sum to 40"):
        FitWeights(
            ateco=9,
            revenue=10,
            structure=10,
            digital_maturity=5,
            geography=5,
        )


def test_unverified_phone_and_unexplained_suppression_fail_closed() -> None:
    with pytest.raises(ValidationError, match="unverified phone"):
        ScoreFeatures(company=company(), phone_kind=PhoneKind.MOBILE)

    with pytest.raises(ValidationError, match="suppression_reason"):
        ScoreFeatures(company=company(), suppressed=True)


def test_score_result_checks_arithmetic_and_t0_reason() -> None:
    breakdown = ScoreBreakdown(fit={}, signal={}, reachability={})

    with pytest.raises(ValidationError, match="must equal"):
        ScoreResult(
            piva=company().piva,
            scoring_version="segment@v1",
            fit_score=20,
            signal_score=20,
            reachability_score=10,
            total_score=49,
            eligible_tier=Tier.T2,
            active_trigger=False,
            breakdown=breakdown,
            scored_at=datetime(2026, 7, 15, tzinfo=UTC),
            recalculation_reason=RecalculationReason.WEEKLY,
        )

    with pytest.raises(ValidationError, match="exclusion reason"):
        ScoreResult(
            piva=company().piva,
            scoring_version="segment@v1",
            fit_score=0,
            signal_score=0,
            reachability_score=0,
            total_score=0,
            eligible_tier=Tier.T0,
            active_trigger=False,
            breakdown=breakdown,
            scored_at=datetime(2026, 7, 15, tzinfo=UTC),
            recalculation_reason=RecalculationReason.MANUAL,
        )


def test_capacity_assignment_cannot_promote_suppressed_or_unqualified_scores() -> None:
    breakdown = ScoreBreakdown(fit={}, signal={}, reachability={})
    suppressed = ScoreResult(
        piva=company().piva,
        scoring_version="segment@v1",
        fit_score=0,
        signal_score=0,
        reachability_score=0,
        total_score=0,
        eligible_tier=Tier.T0,
        active_trigger=False,
        exclusion_reasons=("suppression:opt_out",),
        breakdown=breakdown,
        scored_at=datetime(2026, 7, 15, tzinfo=UTC),
        recalculation_reason=RecalculationReason.WEEKLY,
    )

    with pytest.raises(ValidationError, match="only preserve or lower"):
        TierAssignment(score=suppressed, assigned_tier=Tier.T1, queue_rank=1)

    audit_only = TierAssignment(score=suppressed, assigned_tier=Tier.T0)
    assert audit_only.queue_rank is None


def test_actionable_assignment_requires_rank_and_downgrade_reason() -> None:
    breakdown = ScoreBreakdown(fit={}, signal={}, reachability={})
    eligible = ScoreResult(
        piva=company().piva,
        scoring_version="segment@v1",
        fit_score=35,
        signal_score=30,
        reachability_score=15,
        total_score=80,
        eligible_tier=Tier.T1,
        active_trigger=True,
        breakdown=breakdown,
        scored_at=datetime(2026, 7, 15, tzinfo=UTC),
        recalculation_reason=RecalculationReason.WEEKLY,
    )

    with pytest.raises(ValidationError, match="queue rank"):
        TierAssignment(score=eligible, assigned_tier=Tier.T1)
    with pytest.raises(ValidationError, match="capacity_reason"):
        TierAssignment(score=eligible, assigned_tier=Tier.T2, queue_rank=1)

    downgraded = TierAssignment(
        score=eligible,
        assigned_tier=Tier.T2,
        queue_rank=1,
        capacity_reason="T1 capacity exhausted",
    )
    assert downgraded.assigned_tier is Tier.T2
