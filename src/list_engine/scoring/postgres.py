"""PostgreSQL persistence for immutable scores and segment-scoped weight versions."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import psycopg
from psycopg.types.json import Jsonb

from list_engine.scoring.models import (
    RecalculationReason,
    ScoreBreakdown,
    ScoreResult,
    SegmentConfig,
    Tier,
    TierAssignment,
)


class PostgresScoreStore:
    """Keep scoring configuration versioned and score observations append-only.

    The store expects a connection using ``dict_row``. It deliberately activates
    versions under a per-segment PostgreSQL advisory lock: the partial unique index
    is the final invariant, while the lock makes competing activations deterministic
    instead of turning them into intermittent uniqueness failures.
    """

    def __init__(self, connection: psycopg.Connection[dict[str, Any]]) -> None:
        self._connection = connection

    def register_version(
        self,
        config: SegmentConfig,
        rationale: str,
        *,
        activate: bool = False,
        valid_from: datetime | None = None,
    ) -> UUID:
        """Register an immutable config, optionally activating it atomically.

        Replaying an identical registration returns the original id. Reusing a
        scoring-version identifier for different content is rejected; silently
        changing a historical rubric would make prior scores irreproducible.
        """

        rationale = rationale.strip()
        if not rationale:
            raise ValueError("rationale cannot be blank")
        effective_at = _aware_datetime(valid_from or datetime.now(UTC), "valid_from")
        configuration = config.model_dump(mode="json")

        with self._connection.transaction():
            if activate:
                self._lock_segment(config.segment_id)

            inserted = self._connection.execute(
                """
                INSERT INTO list_engine.scoring_weight_versions (
                    version, segment_id, weights, rationale, is_active, valid_from
                )
                VALUES (%s, %s, %s, %s, false, %s)
                ON CONFLICT (version) DO NOTHING
                RETURNING id
                """,
                (
                    config.scoring_version,
                    config.segment_id,
                    Jsonb(configuration),
                    rationale,
                    effective_at,
                ),
            ).fetchone()

            row = self._connection.execute(
                """
                SELECT id, segment_id, weights, rationale
                FROM list_engine.scoring_weight_versions
                WHERE version = %s
                FOR UPDATE
                """,
                (config.scoring_version,),
            ).fetchone()
            if row is None:  # pragma: no cover - INSERT/SELECT share one transaction
                raise RuntimeError("registered scoring version disappeared")
            if (
                row["segment_id"] != config.segment_id
                or row["weights"] != configuration
                or row["rationale"] != rationale
            ):
                raise ValueError(
                    f"scoring version {config.scoring_version!r} already has different content"
                )

            if activate:
                self._activate_locked(config.scoring_version, config.segment_id, effective_at)

            identifier = inserted["id"] if inserted is not None else row["id"]
            return UUID(str(identifier))

    def activate_version(
        self,
        scoring_version: str,
        *,
        activated_at: datetime | None = None,
    ) -> None:
        """Make one registered version active, closing only its segment predecessor."""

        effective_at = _aware_datetime(activated_at or datetime.now(UTC), "activated_at")
        with self._connection.transaction():
            target = self._connection.execute(
                """
                SELECT segment_id
                FROM list_engine.scoring_weight_versions
                WHERE version = %s
                """,
                (scoring_version,),
            ).fetchone()
            if target is None:
                raise KeyError(f"unknown scoring version: {scoring_version}")
            segment_id = str(target["segment_id"])
            self._lock_segment(segment_id)
            self._activate_locked(scoring_version, segment_id, effective_at)

    def persist_assignment(self, assignment: TierAssignment) -> UUID:
        """Insert one assigned score, treating an identical replay as a no-op."""

        score = assignment.score
        inputs = _assignment_inputs(assignment)
        with self._connection.transaction():
            inserted = self._connection.execute(
                """
                INSERT INTO list_engine.scores (
                    piva,
                    scoring_version,
                    fit_score,
                    signal_score,
                    reachability_score,
                    tier,
                    inputs,
                    scored_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (piva, scoring_version, scored_at) DO NOTHING
                RETURNING id
                """,
                (
                    score.piva,
                    score.scoring_version,
                    score.fit_score,
                    score.signal_score,
                    score.reachability_score,
                    assignment.assigned_tier.value,
                    Jsonb(inputs),
                    score.scored_at,
                ),
            ).fetchone()
            if inserted is not None:
                return UUID(str(inserted["id"]))

            existing = self._connection.execute(
                """
                SELECT
                    id,
                    fit_score,
                    signal_score,
                    reachability_score,
                    total_score,
                    tier,
                    inputs
                FROM list_engine.scores
                WHERE piva = %s
                  AND scoring_version = %s
                  AND scored_at = %s
                """,
                (score.piva, score.scoring_version, score.scored_at),
            ).fetchone()
            if existing is None:  # pragma: no cover - guarded by the unique constraint
                raise RuntimeError("score replay target disappeared")
            expected = {
                "fit_score": score.fit_score,
                "signal_score": score.signal_score,
                "reachability_score": score.reachability_score,
                "total_score": score.total_score,
                "tier": assignment.assigned_tier.value,
                "inputs": inputs,
            }
            actual = {key: existing[key] for key in expected}
            if actual != expected:
                raise ValueError("score replay conflicts with the immutable historical observation")
            return UUID(str(existing["id"]))

    def list_ordered_queue(
        self,
        *,
        segment_id: str | None = None,
        limit: int = 1_000,
        include_non_actionable: bool = False,
    ) -> tuple[TierAssignment, ...]:
        """Return the latest score per company for currently active rubrics.

        The default queue is operational: T1, T2 and T3 only. T0 suppressions and
        unqualified records remain queryable for audits via ``include_non_actionable``.
        A newly activated rubric therefore has an empty queue until records are
        rescored, preventing stale scores from a prior rubric from leaking to sales.
        """

        if limit < 1 or limit > 10_000:
            raise ValueError("limit must be between 1 and 10000")

        segment_filter = "" if segment_id is None else "AND versions.segment_id = %s"
        actionable_filter = (
            "" if include_non_actionable else "WHERE latest.tier IN ('T1', 'T2', 'T3')"
        )
        parameters: list[object] = []
        if segment_id is not None:
            parameters.append(segment_id)
        parameters.append(limit)

        rows = self._connection.execute(
            f"""
            WITH latest AS (
                SELECT DISTINCT ON (scores.piva, versions.segment_id)
                    scores.id,
                    scores.piva,
                    scores.scoring_version,
                    scores.fit_score,
                    scores.signal_score,
                    scores.reachability_score,
                    scores.total_score,
                    scores.tier,
                    scores.inputs,
                    scores.scored_at,
                    versions.segment_id
                FROM list_engine.scores AS scores
                JOIN list_engine.scoring_weight_versions AS versions
                  ON versions.version = scores.scoring_version
                WHERE versions.is_active
                  {segment_filter}
                ORDER BY
                    scores.piva,
                    versions.segment_id,
                    scores.scored_at DESC,
                    scores.id DESC
            )
            SELECT *
            FROM latest
            {actionable_filter}
            ORDER BY
                CASE latest.tier
                    WHEN 'T1' THEN 1
                    WHEN 'T2' THEN 2
                    WHEN 'T3' THEN 3
                    WHEN 'T0' THEN 4
                    ELSE 5
                END,
                CASE
                    WHEN latest.inputs->>'queue_rank' ~ '^[0-9]+$'
                        THEN (latest.inputs->>'queue_rank')::integer
                    ELSE NULL
                END NULLS LAST,
                latest.total_score DESC,
                latest.scored_at DESC,
                latest.piva
            LIMIT %s
            """,
            tuple(parameters),
        ).fetchall()
        return tuple(_row_to_assignment(row) for row in rows)

    def _lock_segment(self, segment_id: str) -> None:
        self._connection.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (f"list-engine:scoring:{segment_id}",),
        )

    def _activate_locked(
        self,
        scoring_version: str,
        segment_id: str,
        effective_at: datetime,
    ) -> None:
        target = self._connection.execute(
            """
            SELECT version, segment_id, is_active, valid_from, valid_to
            FROM list_engine.scoring_weight_versions
            WHERE version = %s
            FOR UPDATE
            """,
            (scoring_version,),
        ).fetchone()
        if target is None:
            raise KeyError(f"unknown scoring version: {scoring_version}")
        if target["segment_id"] != segment_id:
            raise ValueError("scoring version belongs to a different segment")
        if target["is_active"]:
            return
        if target["valid_to"] is not None:
            raise ValueError(
                "a closed scoring version cannot be reactivated; register a new version"
            )

        active = self._connection.execute(
            """
            SELECT version, valid_from
            FROM list_engine.scoring_weight_versions
            WHERE segment_id = %s AND is_active
            FOR UPDATE
            """,
            (segment_id,),
        ).fetchone()
        if active is not None:
            if effective_at <= active["valid_from"]:
                raise ValueError("activation time must follow the current version's valid_from")
            self._connection.execute(
                """
                UPDATE list_engine.scoring_weight_versions
                SET is_active = false, valid_to = %s
                WHERE version = %s
                """,
                (effective_at, active["version"]),
            )

        self._connection.execute(
            """
            UPDATE list_engine.scoring_weight_versions
            SET is_active = true, valid_from = %s, valid_to = NULL
            WHERE version = %s
            """,
            (effective_at, scoring_version),
        )


def _aware_datetime(value: datetime, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value


def _assignment_inputs(assignment: TierAssignment) -> dict[str, object]:
    score = assignment.score
    return {
        "eligible_tier": score.eligible_tier.value,
        "breakdown": score.breakdown.model_dump(mode="json"),
        "active_trigger": score.active_trigger,
        "exclusion_reasons": list(score.exclusion_reasons),
        "recalculation_reason": score.recalculation_reason.value,
        "queue_rank": assignment.queue_rank,
        "capacity_reason": assignment.capacity_reason,
    }


def _row_to_assignment(row: dict[str, Any]) -> TierAssignment:
    inputs = row["inputs"]
    score = ScoreResult(
        piva=row["piva"],
        scoring_version=row["scoring_version"],
        fit_score=row["fit_score"],
        signal_score=row["signal_score"],
        reachability_score=row["reachability_score"],
        total_score=row["total_score"],
        eligible_tier=Tier(inputs["eligible_tier"]),
        active_trigger=inputs["active_trigger"],
        exclusion_reasons=tuple(inputs["exclusion_reasons"]),
        breakdown=ScoreBreakdown.model_validate(inputs["breakdown"]),
        scored_at=row["scored_at"],
        recalculation_reason=RecalculationReason(inputs["recalculation_reason"]),
    )
    return TierAssignment(
        score=score,
        assigned_tier=Tier(row["tier"]),
        queue_rank=inputs["queue_rank"],
        capacity_reason=inputs["capacity_reason"],
    )


__all__ = ["PostgresScoreStore"]
