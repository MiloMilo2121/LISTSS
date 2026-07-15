from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg
import pytest
from psycopg.rows import dict_row

from list_engine.scoring.models import (
    FitWeights,
    NumericBand,
    ReachabilityWeights,
    RecalculationReason,
    RubricWeights,
    ScoreBreakdown,
    ScoreResult,
    SegmentConfig,
    SignalWeights,
    Tier,
    TierAssignment,
    TierCapacity,
)
from list_engine.scoring.postgres import PostgresScoreStore

ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.getenv("TEST_DATABASE_URL")
BASE_TIME = datetime(2026, 7, 15, 8, tzinfo=UTC)

pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="TEST_DATABASE_URL is required for disposable PostgreSQL integration tests",
)


@pytest.fixture()
def database() -> Iterator[psycopg.Connection[dict[str, Any]]]:
    assert DATABASE_URL is not None
    connection = psycopg.connect(DATABASE_URL, autocommit=True, row_factory=dict_row)
    if "test" not in connection.info.dbname.lower():
        connection.close()
        pytest.fail("Refusing to reset a database whose name does not contain 'test'")

    connection.execute("DROP SCHEMA IF EXISTS list_engine CASCADE")
    for migration_name in (
        "202607150001_initial.sql",
        "202607150002_ingestion.sql",
    ):
        migration = (ROOT / "supabase" / "migrations" / migration_name).read_text()
        connection.execute(migration, prepare=False)
        connection.execute(migration, prepare=False)

    # Exercise the M3 backfill path, including a pre-M3 identifier that cannot be
    # assigned to a real segment without inventing ownership.
    connection.execute(
        """
        INSERT INTO list_engine.scoring_weight_versions (
            version, weights, rationale, is_active, valid_from
        )
        VALUES ('pre-m3-v0', '{}'::jsonb, 'pre-M3 fixture', true, %s)
        """,
        (BASE_TIME - timedelta(days=30),),
    )
    scoring_migration = (ROOT / "supabase" / "migrations" / "202607150003_scoring.sql").read_text()
    connection.execute(scoring_migration, prepare=False)
    connection.execute(scoring_migration, prepare=False)

    connection.execute("SET list_engine.app_mode = 'demo'")
    connection.execute((ROOT / "seeds" / "demo.sql").read_text(), prepare=False)

    yield connection
    connection.rollback()
    connection.execute("DROP SCHEMA IF EXISTS list_engine CASCADE")
    connection.close()


def segment_config(segment_id: str, version: str) -> SegmentConfig:
    return SegmentConfig(
        schema_version=1,
        segment_id=segment_id,
        version=version,
        status="hypothesis",
        persona="Responsabile amministrativo",
        ateco_prefixes=("49",),
        revenue_band_eur=NumericBand(
            minimum=1_000_000,
            sweet_minimum=3_000_000,
            sweet_maximum=10_000_000,
            maximum=25_000_000,
        ),
        employee_band=NumericBand(
            minimum=10,
            sweet_minimum=20,
            sweet_maximum=80,
            maximum=200,
        ),
        target_regions=("Lombardia",),
        excluded_company_statuses=("ceased",),
        hot_signal_types=("new_office",),
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
        capacity=TierCapacity(t1_per_bdr=35, t2_per_bdr=70),
    )


def assignment(
    *,
    piva: str,
    scoring_version: str,
    fit: int,
    signal: int,
    reachability: int,
    eligible_tier: Tier,
    assigned_tier: Tier,
    queue_rank: int,
    scored_at: datetime,
) -> TierAssignment:
    score = ScoreResult(
        piva=piva,
        scoring_version=scoring_version,
        fit_score=fit,
        signal_score=signal,
        reachability_score=reachability,
        total_score=fit + signal + reachability,
        eligible_tier=eligible_tier,
        active_trigger=eligible_tier is Tier.T1,
        breakdown=ScoreBreakdown(
            fit={"fixture": fit},
            signal={"fixture": signal},
            reachability={"fixture": reachability},
        ),
        scored_at=scored_at,
        recalculation_reason=RecalculationReason.WEEKLY,
    )
    return TierAssignment(
        score=score,
        assigned_tier=assigned_tier,
        queue_rank=queue_rank,
        capacity_reason=("T1 capacity exhausted" if eligible_tier is not assigned_tier else None),
    )


def test_migration_is_replay_safe_and_backfills_unknown_legacy_version(
    database: psycopg.Connection[dict[str, Any]],
) -> None:
    assert database.execute(
        """
        SELECT segment_id, is_active
        FROM list_engine.scoring_weight_versions
        WHERE version = 'pre-m3-v0'
        """
    ).fetchone() == {"segment_id": "legacy", "is_active": True}

    index = database.execute(
        """
        SELECT indexdef
        FROM pg_indexes
        WHERE schemaname = 'list_engine'
          AND indexname = 'one_active_scoring_version_per_segment'
        """
    ).fetchone()
    assert index is not None
    assert "(segment_id) WHERE is_active" in index["indexdef"]


def test_two_segments_can_have_independent_active_versions(
    database: psycopg.Connection[dict[str, Any]],
) -> None:
    store = PostgresScoreStore(database)
    logistics = segment_config("logistics", "v1")
    construction = segment_config("construction", "v1")

    first_id = store.register_version(
        logistics, "Initial logistics hypothesis", activate=True, valid_from=BASE_TIME
    )
    assert (
        store.register_version(
            logistics, "Initial logistics hypothesis", activate=True, valid_from=BASE_TIME
        )
        == first_id
    )
    store.register_version(
        construction,
        "Initial construction hypothesis",
        activate=True,
        valid_from=BASE_TIME,
    )

    assert database.execute(
        """
        SELECT segment_id, version
        FROM list_engine.scoring_weight_versions
        WHERE is_active AND segment_id IN ('logistics', 'construction')
        ORDER BY segment_id
        """
    ).fetchall() == [
        {"segment_id": "construction", "version": "construction@v1"},
        {"segment_id": "logistics", "version": "logistics@v1"},
    ]


def test_activation_closes_only_the_prior_version_in_the_same_segment(
    database: psycopg.Connection[dict[str, Any]],
) -> None:
    store = PostgresScoreStore(database)
    first = segment_config("logistics", "v1")
    second = segment_config("logistics", "v2")
    store.register_version(first, "Initial hypothesis", activate=True, valid_from=BASE_TIME)
    store.register_version(
        second,
        "Reviewed hypothesis",
        valid_from=BASE_TIME + timedelta(days=1),
    )

    activated_at = BASE_TIME + timedelta(days=2)
    store.activate_version(second.scoring_version, activated_at=activated_at)

    rows = database.execute(
        """
        SELECT version, is_active, valid_to
        FROM list_engine.scoring_weight_versions
        WHERE segment_id = 'logistics'
        ORDER BY version
        """
    ).fetchall()
    assert rows == [
        {"version": "logistics@v1", "is_active": False, "valid_to": activated_at},
        {"version": "logistics@v2", "is_active": True, "valid_to": None},
    ]


def test_score_replay_is_idempotent_immutable_and_queue_is_ordered(
    database: psycopg.Connection[dict[str, Any]],
) -> None:
    store = PostgresScoreStore(database)
    config = segment_config("logistics", "v1")
    store.register_version(config, "Queue fixture", activate=True, valid_from=BASE_TIME)
    scored_at = BASE_TIME + timedelta(hours=1)
    high_t1 = assignment(
        piva="99000000002",
        scoring_version=config.scoring_version,
        fit=38,
        signal=35,
        reachability=17,
        eligible_tier=Tier.T1,
        assigned_tier=Tier.T1,
        queue_rank=1,
        scored_at=scored_at,
    )
    lower_t1 = assignment(
        piva="99000000010",
        scoring_version=config.scoring_version,
        fit=30,
        signal=30,
        reachability=15,
        eligible_tier=Tier.T1,
        assigned_tier=Tier.T1,
        queue_rank=2,
        scored_at=scored_at,
    )
    capacity_downgrade = assignment(
        piva="99000000028",
        scoring_version=config.scoring_version,
        fit=40,
        signal=38,
        reachability=17,
        eligible_tier=Tier.T1,
        assigned_tier=Tier.T2,
        queue_rank=1,
        scored_at=scored_at,
    )

    score_id = store.persist_assignment(high_t1)
    assert store.persist_assignment(high_t1) == score_id
    store.persist_assignment(lower_t1)
    store.persist_assignment(capacity_downgrade)

    assert database.execute(
        "SELECT count(*) FROM list_engine.scores WHERE scoring_version = %s",
        (config.scoring_version,),
    ).fetchone() == {"count": 3}
    queue = store.list_ordered_queue(segment_id=config.segment_id)
    assert [item.score.piva for item in queue] == [
        "99000000002",
        "99000000010",
        "99000000028",
    ]
    assert queue[2].score.eligible_tier is Tier.T1
    assert queue[2].assigned_tier is Tier.T2
    assert queue[2].capacity_reason == "T1 capacity exhausted"

    stored_inputs = database.execute(
        "SELECT inputs FROM list_engine.scores WHERE id = %s", (score_id,)
    ).fetchone()
    assert stored_inputs is not None
    assert stored_inputs["inputs"]["active_trigger"] is True
    assert stored_inputs["inputs"]["eligible_tier"] == "T1"
    assert stored_inputs["inputs"]["recalculation_reason"] == "weekly"
    assert stored_inputs["inputs"]["breakdown"]["fit"] == {"fixture": 38}

    conflicting_replay = high_t1.model_copy(update={"assigned_tier": Tier.T2})
    with pytest.raises(ValueError, match="immutable historical observation"):
        store.persist_assignment(conflicting_replay)

    with pytest.raises(psycopg.errors.RaiseException, match="append-only"):
        database.execute("UPDATE list_engine.scores SET tier = 'T2' WHERE id = %s", (score_id,))
