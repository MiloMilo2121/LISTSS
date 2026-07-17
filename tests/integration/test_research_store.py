from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import psycopg
import pytest
from psycopg.rows import dict_row

from list_engine.research.models import (
    AgentAttempt,
    AgentGeneration,
    CitedClaim,
    DossierEvaluation,
    EvaluationIssue,
    EvidenceItem,
    ResearchDossier,
    ResearchInput,
)
from list_engine.research.postgres import (
    PostgresResearchStore,
    ResearchGenerationLease,
    ResearchLeaseLostError,
    research_input_hash,
)

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
    for migration in sorted((ROOT / "supabase" / "migrations").glob("*.sql")):
        ddl = migration.read_text()
        connection.execute(ddl, prepare=False)
        connection.execute(ddl, prepare=False)
    connection.execute("SET list_engine.app_mode = 'demo'")
    connection.execute((ROOT / "seeds" / "demo.sql").read_text(), prepare=False)

    yield connection
    connection.rollback()
    connection.execute("DROP SCHEMA IF EXISTS list_engine CASCADE")
    connection.close()


def research_input() -> ResearchInput:
    return ResearchInput(
        piva="99000000002",
        as_of=BASE_TIME.date(),
        website="https://azienda.invalid",
        hub_data={"legal_name": "Azienda Demo"},
        evidence=(
            EvidenceItem(
                evidence_id="official_1",
                source_url="https://registro.invalid/99000000002",
                source_kind="official_registry",
                title="Registro sintetico",
                excerpt="Azienda attiva con sede a Milano.",
                observed_at=BASE_TIME,
                verified=True,
            ),
        ),
    )


def generation(
    *,
    session_id: str = "sdk-session-1",
    requested_model: str = "deterministic-v1",
    resolved_model: str = "deterministic-v1",
) -> AgentGeneration:
    claim = CitedClaim(
        claim="Azienda attiva con sede a Milano.",
        evidence_id="official_1",
        source_url="https://registro.invalid/99000000002",
        supporting_excerpt="Azienda attiva con sede a Milano.",
        confidence=1.0,
    )
    dossier = ResearchDossier(
        piva="99000000002",
        fatti=(claim,),
        segnali_attivi=(),
        persona_probabile=None,
        hook_apertura="Aprire dalla fonte verificata: Azienda attiva con sede a Milano.",
        opening_hook_evidence_ids=("official_1",),
        obiezione_probabile=None,
        cosa_non_dire=("Non presentare ipotesi come fatti.",),
    )
    return AgentGeneration(
        dossier=dossier,
        provider="demo",
        requested_model=requested_model,
        model=resolved_model,
        prompt_version="research-v1",
        session_id=session_id,
        cost_usd=Decimal("0.0123"),
        input_tokens=100,
        output_tokens=50,
    )


def acquire_lease(
    store: PostgresResearchStore,
    packet: ResearchInput,
    *,
    model: str = "deterministic-v1",
    lease_ttl: timedelta = timedelta(minutes=10),
) -> ResearchGenerationLease:
    lease = store.acquire_generation_lease(
        packet,
        provider="demo",
        model=model,
        prompt_version="research-v1",
        lease_ttl=lease_ttl,
    )
    assert lease is not None
    return lease


def expire_lease(
    database: psycopg.Connection[dict[str, Any]],
    lease: ResearchGenerationLease,
) -> None:
    expired = database.execute(
        """
        UPDATE list_engine.research_generation_leases
        SET acquired_at = statement_timestamp() - interval '2 minutes',
            expires_at = statement_timestamp() - interval '1 minute'
        WHERE input_hash = %s
          AND provider = %s
          AND model = %s
          AND prompt_version = %s
          AND owner_token = %s
        RETURNING owner_token
        """,
        (
            lease.input_hash,
            lease.provider,
            lease.model,
            lease.prompt_version,
            lease.owner_token,
        ),
    ).fetchone()
    assert expired == {"owner_token": lease.owner_token}


def evaluation(*, passed: bool) -> DossierEvaluation:
    issues = (
        ()
        if passed
        else (
            EvaluationIssue(
                code="unsupported_claim",
                message="Il claim non e supportato dall'excerpt.",
                field="fatti.0",
            ),
        )
    )
    score = 1.0 if passed else 0.0
    return DossierEvaluation(
        passed=passed,
        factual_support=score,
        citation_accuracy=score,
        completeness=score,
        source_quality=score,
        hallucination_free=score,
        issues=issues,
        evaluator_version="citation-gate-v1",
    )


def test_research_migration_is_replay_safe_and_cache_is_approved_only(
    database: psycopg.Connection[dict[str, Any]],
) -> None:
    cache_constraint = database.execute(
        """
        SELECT pg_get_constraintdef(oid) AS definition
        FROM pg_constraint
        WHERE conrelid = 'list_engine.research_cache'::regclass
          AND contype = 'c'
        """
    ).fetchall()
    assert any('"passed": true' in row["definition"] for row in cache_constraint)


def test_passed_generation_is_audited_cached_and_immutable(
    database: psycopg.Connection[dict[str, Any]],
) -> None:
    store = PostgresResearchStore(database)
    packet = research_input()
    lease = acquire_lease(store, packet)
    result = store.record(packet, generation(), evaluation(passed=True), lease=lease)
    assert store.release_generation_lease(lease)

    assert result.approved is not None
    assert result.review_queue_id is None
    assert (
        store.get_cached(
            packet,
            provider="demo",
            model="deterministic-v1",
            prompt_version="research-v1",
        )
        == result.approved
    )
    row = database.execute(
        """
        SELECT status, input_hash, eval_passed, cost_usd, output->>'session_id' AS sdk_session_id
        FROM list_engine.agent_sessions
        WHERE id = %s
        """,
        (result.session_id,),
    ).fetchone()
    assert row == {
        "status": "succeeded",
        "input_hash": research_input_hash(packet),
        "eval_passed": True,
        "cost_usd": Decimal("0.012300"),
        "sdk_session_id": "sdk-session-1",
    }

    with pytest.raises(psycopg.errors.RaiseException, match="immutable"):
        database.execute(
            "UPDATE list_engine.agent_sessions SET status = 'review' WHERE id = %s",
            (result.session_id,),
        )


def test_failed_generation_goes_to_review_and_never_to_cache(
    database: psycopg.Connection[dict[str, Any]],
) -> None:
    store = PostgresResearchStore(database)
    packet = research_input()
    lease = acquire_lease(store, packet)
    result = store.record(packet, generation(), evaluation(passed=False), lease=lease)
    assert store.release_generation_lease(lease)

    assert result.approved is None
    assert result.review_queue_id is not None
    assert (
        store.get_cached(
            packet,
            provider="demo",
            model="deterministic-v1",
            prompt_version="research-v1",
        )
        is None
    )
    review = database.execute(
        "SELECT reason, payload->>'agent_session_id' AS session_id FROM list_engine.review_queue"
    ).fetchone()
    assert review == {
        "reason": "research_eval:unsupported_claim",
        "session_id": str(result.session_id),
    }


def test_cache_refreshes_only_after_expiry_and_identity_is_exact(
    database: psycopg.Connection[dict[str, Any]],
) -> None:
    store = PostgresResearchStore(database)
    packet = research_input()
    first_lease = acquire_lease(store, packet)
    first = store.record(
        packet,
        generation(session_id="first"),
        evaluation(passed=True),
        lease=first_lease,
        cache_ttl=timedelta(hours=1),
    )
    assert store.release_generation_lease(first_lease)
    cached = store.get_cached(
        packet,
        provider="demo",
        model="deterministic-v1",
        prompt_version="research-v1",
    )
    assert cached is not None
    assert cached.generation.session_id == "first"

    expired = database.execute(
        """
        UPDATE list_engine.research_cache
        SET created_at = statement_timestamp() - interval '2 hours',
            expires_at = statement_timestamp() - interval '1 hour'
        WHERE agent_session_id = %s
        RETURNING id
        """,
        (first.session_id,),
    ).fetchone()
    assert expired is not None
    assert (
        store.get_cached(
            packet,
            provider="demo",
            model="deterministic-v1",
            prompt_version="research-v1",
        )
        is None
    )

    refresh_lease = acquire_lease(store, packet)
    refreshed_result = store.record(
        packet,
        generation(session_id="after-expiry"),
        evaluation(passed=True),
        lease=refresh_lease,
        cache_ttl=timedelta(hours=1),
    )
    assert store.release_generation_lease(refresh_lease)
    assert first.session_id != refreshed_result.session_id
    refreshed = store.get_cached(
        packet,
        provider="demo",
        model="deterministic-v1",
        prompt_version="research-v1",
    )
    assert refreshed is not None
    assert refreshed.generation.session_id == "after-expiry"
    assert (
        store.get_cached(
            packet,
            provider="demo",
            model="other-model",
            prompt_version="research-v1",
        )
        is None
    )


def test_cache_identity_uses_requested_model_not_resolved_model(
    database: psycopg.Connection[dict[str, Any]],
) -> None:
    store = PostgresResearchStore(database)
    packet = research_input()
    requested_model = "claude-sonnet-alias"
    resolved_model = "claude-sonnet-resolved-20260715"
    lease = acquire_lease(store, packet, model=requested_model)
    result = store.record(
        packet,
        generation(
            requested_model=requested_model,
            resolved_model=resolved_model,
        ),
        evaluation(passed=True),
        lease=lease,
    )
    assert store.release_generation_lease(lease)

    cached = store.get_cached(
        packet,
        provider="demo",
        model=requested_model,
        prompt_version="research-v1",
    )
    assert cached is not None
    assert cached.generation.requested_model == requested_model
    assert cached.generation.model == resolved_model
    assert (
        store.get_cached(
            packet,
            provider="demo",
            model=resolved_model,
            prompt_version="research-v1",
        )
        is None
    )
    identity = database.execute(
        """
        SELECT
            session.model AS audited_model,
            cache.model AS cache_model,
            cache.generation->>'requested_model' AS generation_requested_model,
            cache.generation->>'model' AS generation_resolved_model
        FROM list_engine.agent_sessions AS session
        JOIN list_engine.research_cache AS cache
          ON cache.agent_session_id = session.id
        WHERE session.id = %s
        """,
        (result.session_id,),
    ).fetchone()
    assert identity == {
        "audited_model": resolved_model,
        "cache_model": requested_model,
        "generation_requested_model": requested_model,
        "generation_resolved_model": resolved_model,
    }


def test_store_rejects_cross_company_generation_before_any_write(
    database: psycopg.Connection[dict[str, Any]],
) -> None:
    payload = generation().model_dump(mode="python")
    dossier_payload = generation().dossier.model_dump(mode="python")
    dossier_payload["piva"] = "99000000010"
    payload["dossier"] = dossier_payload
    wrong_company = AgentGeneration.model_validate(payload)
    store = PostgresResearchStore(database)
    packet = research_input()
    lease = acquire_lease(store, packet)

    with pytest.raises(ValueError, match="does not match"):
        store.record(packet, wrong_company, evaluation(passed=True), lease=lease)
    assert store.release_generation_lease(lease)
    assert database.execute("SELECT count(*) FROM list_engine.agent_sessions").fetchone() == {
        "count": 0
    }


def test_store_recomputes_the_gate_and_rejects_a_forged_passing_eval(
    database: psycopg.Connection[dict[str, Any]],
) -> None:
    original = generation()
    claim = original.dossier.facts[0].model_copy(
        update={"claim": "Azienda non attiva con sede a Milano."}
    )
    dossier = original.dossier.model_copy(update={"facts": (claim,)})
    forged_generation = original.model_copy(update={"dossier": dossier})
    forged_pass = DossierEvaluation(
        passed=True,
        factual_support=0.0,
        citation_accuracy=0.0,
        completeness=0.0,
        source_quality=0.0,
        hallucination_free=0.0,
        evaluator_version="attacker",
    )

    store = PostgresResearchStore(database)
    packet = research_input()
    lease = acquire_lease(store, packet)
    result = store.record(
        packet,
        forged_generation,
        forged_pass,
        lease=lease,
    )
    assert store.release_generation_lease(lease)

    assert result.approved is None
    assert result.review_queue_id is not None
    assert database.execute("SELECT count(*) FROM list_engine.research_cache").fetchone() == {
        "count": 0
    }
    assert database.execute(
        "SELECT reason FROM list_engine.review_queue WHERE id = %s",
        (result.review_queue_id,),
    ).fetchone() == {"reason": "research_eval:claim_not_extractive"}


def test_session_and_cache_timestamps_use_the_database_clock(
    database: psycopg.Connection[dict[str, Any]],
) -> None:
    store = PostgresResearchStore(database)
    packet = research_input()
    before = database.execute("SELECT statement_timestamp() AS now").fetchone()
    assert before is not None
    lease = acquire_lease(store, packet)
    result = store.record(
        packet,
        generation(),
        evaluation(passed=True),
        lease=lease,
        cache_ttl=timedelta(seconds=90),
    )
    after = database.execute("SELECT statement_timestamp() AS now").fetchone()
    assert after is not None
    assert store.release_generation_lease(lease)

    timestamps = database.execute(
        """
        SELECT
            session.started_at,
            session.completed_at,
            cache.created_at,
            cache.expires_at
        FROM list_engine.agent_sessions AS session
        JOIN list_engine.research_cache AS cache
          ON cache.agent_session_id = session.id
        WHERE session.id = %s
        """,
        (result.session_id,),
    ).fetchone()
    assert timestamps is not None
    for field in ("started_at", "completed_at", "created_at"):
        assert before["now"] <= timestamps[field] <= after["now"]
    assert timestamps["expires_at"] - timestamps["created_at"] == timedelta(seconds=90)


def test_generation_lease_is_atomic_expires_and_cannot_be_released_by_stale_owner(
    database: psycopg.Connection[dict[str, Any]],
) -> None:
    store = PostgresResearchStore(database)
    packet = research_input()
    first = store.acquire_generation_lease(
        packet,
        provider="demo",
        model="deterministic-v1",
        prompt_version="research-v1",
        lease_ttl=timedelta(minutes=5),
    )
    assert first is not None
    assert (
        store.acquire_generation_lease(
            packet,
            provider="demo",
            model="deterministic-v1",
            prompt_version="research-v1",
            lease_ttl=timedelta(minutes=5),
        )
        is None
    )

    expire_lease(database, first)
    replacement = store.acquire_generation_lease(
        packet,
        provider="demo",
        model="deterministic-v1",
        prompt_version="research-v1",
        lease_ttl=timedelta(minutes=5),
    )
    assert replacement is not None
    assert replacement.owner_token != first.owner_token
    assert not store.release_generation_lease(first)
    assert store.release_generation_lease(replacement)


def test_stale_owner_cannot_publish_and_fenced_transaction_rolls_back(
    database: psycopg.Connection[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = PostgresResearchStore(database)
    packet = research_input()
    stale = acquire_lease(store, packet)
    expire_lease(database, stale)
    replacement = acquire_lease(store, packet)
    assert replacement.owner_token != stale.owner_token

    # Model the takeover race occurring after the initial SELECT ... FOR UPDATE.
    # The cache INSERT repeats the owner-token fence and must roll back the audit
    # row it inserted earlier in the same transaction.
    monkeypatch.setattr(store, "_assert_active_lease", lambda _lease: None)
    with pytest.raises(ResearchLeaseLostError, match="expired before cache publish"):
        store.record(
            packet,
            generation(session_id="stale-owner"),
            evaluation(passed=True),
            lease=stale,
        )

    superseded = store.record_superseded_generation(
        packet,
        generation(session_id="stale-owner"),
        evaluation(passed=True),
        lease=stale,
    )
    published = store.record(
        packet,
        generation(session_id="replacement-owner"),
        evaluation(passed=True),
        lease=replacement,
    )
    assert store.release_generation_lease(replacement)

    assert superseded.superseded
    assert superseded.approved is None
    assert superseded.review_queue_id is None
    assert published.approved is not None
    assert database.execute(
        """
        SELECT
            (SELECT count(*) FROM list_engine.agent_sessions) AS sessions,
            (SELECT count(*) FROM list_engine.research_cache) AS cache_rows,
            (SELECT count(*) FROM list_engine.review_queue) AS review_rows
        """
    ).fetchone() == {"sessions": 2, "cache_rows": 1, "review_rows": 0}
    assert database.execute(
        """
        SELECT status, eval_passed, cost_usd, requested_model, lease_owner_token
        FROM list_engine.agent_sessions
        WHERE id = %s
        """,
        (superseded.session_id,),
    ).fetchone() == {
        "status": "superseded",
        "eval_passed": True,
        "cost_usd": Decimal("0.012300"),
        "requested_model": "deterministic-v1",
        "lease_owner_token": stale.owner_token,
    }


def test_stale_failed_call_preserves_attempt_cost_without_cache_or_review(
    database: psycopg.Connection[dict[str, Any]],
) -> None:
    store = PostgresResearchStore(database)
    packet = research_input()
    stale = acquire_lease(store, packet)
    expire_lease(database, stale)
    replacement = acquire_lease(store, packet)
    attempt = AgentAttempt(
        attempt=1,
        session_id="failed-sdk-session",
        model="deterministic-v1-resolved",
        status="validation_failed",
        structured_output_json="{}",
        validation_error="schema_validation",
        cost_usd=Decimal("0.0042"),
        input_tokens=12,
        output_tokens=4,
    )

    with pytest.raises(ResearchLeaseLostError, match="no longer active"):
        store.record_failure(
            packet,
            lease=stale,
            error_type="ClaudeSdkResearchError",
            attempts=(attempt,),
        )
    superseded = store.record_superseded_failure(
        packet,
        lease=stale,
        error_type="ClaudeSdkResearchError",
        attempts=(attempt,),
    )

    assert superseded.superseded
    assert superseded.review_queue_id is None
    assert database.execute(
        """
        SELECT status, model, requested_model, input_tokens, output_tokens, cost_usd
        FROM list_engine.agent_sessions
        WHERE id = %s
        """,
        (superseded.session_id,),
    ).fetchone() == {
        "status": "superseded",
        "model": "deterministic-v1-resolved",
        "requested_model": "deterministic-v1",
        "input_tokens": 12,
        "output_tokens": 4,
        "cost_usd": Decimal("0.004200"),
    }
    assert database.execute(
        """
        SELECT
            (SELECT count(*) FROM list_engine.research_cache) AS cache_rows,
            (SELECT count(*) FROM list_engine.review_queue) AS review_rows
        """
    ).fetchone() == {"cache_rows": 0, "review_rows": 0}
    assert store.release_generation_lease(replacement)


def test_generation_failure_is_audited_and_routed_without_raw_error_text(
    database: psycopg.Connection[dict[str, Any]],
) -> None:
    store = PostgresResearchStore(database)
    packet = research_input()
    lease = acquire_lease(store, packet)
    result = store.record_failure(
        packet,
        lease=lease,
        error_type="ResearchAbstentionError",
    )
    assert store.release_generation_lease(lease)

    session = database.execute(
        """
        SELECT status, eval_passed, cost_usd, messages->1->>'error_type' AS error_type
        FROM list_engine.agent_sessions
        WHERE id = %s
        """,
        (result.session_id,),
    ).fetchone()
    assert session == {
        "status": "failed",
        "eval_passed": False,
        "cost_usd": Decimal("0.000000"),
        "error_type": "ResearchAbstentionError",
    }
    assert database.execute(
        "SELECT reason FROM list_engine.review_queue WHERE id = %s",
        (result.review_queue_id,),
    ).fetchone() == {"reason": "research_generation_failed:ResearchAbstentionError"}
