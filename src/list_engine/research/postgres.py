"""PostgreSQL audit log and derived cache for research dossiers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from threading import Lock
from typing import Any, Self
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from list_engine.research.gate import CitationGate
from list_engine.research.models import (
    AgentAttempt,
    AgentGeneration,
    ApprovedDossier,
    DossierEvaluation,
    ResearchInput,
)


def research_input_hash(research_input: ResearchInput) -> str:
    """Return a stable content address for the complete bounded agent input."""

    canonical = json.dumps(
        research_input.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True, slots=True)
class PersistedResearchResult:
    """Identifiers and downstream-safe result created by one recorded generation."""

    session_id: UUID
    approved: ApprovedDossier | None
    review_queue_id: UUID | None
    superseded: bool = False


@dataclass(frozen=True, slots=True)
class PersistedResearchFailure:
    session_id: UUID
    review_queue_id: UUID | None
    superseded: bool = False


@dataclass(frozen=True, slots=True)
class ResearchGenerationLease:
    input_hash: str
    provider: str
    model: str
    prompt_version: str
    piva: str
    owner_token: UUID


class ResearchLeaseLostError(RuntimeError):
    """Raised when a worker tries to publish after its lease was lost or expired."""


class PostgresResearchStore:
    """Persist every generation; cache only dossiers that passed evaluation.

    ``agent_sessions`` is the durable audit source of truth. ``research_cache`` is
    deliberately disposable derived state and can be refreshed after its TTL.
    One instance serializes its short database operations because a psycopg
    connection shares transaction state; durable workers should create one store
    (and therefore one connection) per worker execution context.
    """

    def __init__(self, connection: psycopg.Connection[dict[str, Any]]) -> None:
        self._connection = connection
        self._operation_lock = Lock()

    @classmethod
    def connect(cls, dsn: str) -> Self:
        connection = psycopg.connect(
            dsn,
            autocommit=True,
            prepare_threshold=None,
            row_factory=dict_row,
        )
        return cls(connection)

    def close(self) -> None:
        with self._operation_lock:
            self._connection.close()

    def acquire_generation_lease(
        self,
        research_input: ResearchInput,
        *,
        provider: str,
        model: str,
        prompt_version: str,
        lease_ttl: timedelta = timedelta(minutes=10),
    ) -> ResearchGenerationLease | None:
        with self._operation_lock:
            return self._acquire_generation_lease(
                research_input,
                provider=provider,
                model=model,
                prompt_version=prompt_version,
                lease_ttl=lease_ttl,
            )

    def _acquire_generation_lease(
        self,
        research_input: ResearchInput,
        *,
        provider: str,
        model: str,
        prompt_version: str,
        lease_ttl: timedelta,
    ) -> ResearchGenerationLease | None:
        """Acquire or take over an expired lease without holding a transaction open."""

        if lease_ttl <= timedelta(0):
            raise ValueError("lease_ttl must be positive")
        identity = _identity(provider, model, prompt_version)
        input_hash = research_input_hash(research_input)
        owner_token = uuid4()
        row = self._connection.execute(
            """
            INSERT INTO list_engine.research_generation_leases (
                input_hash, provider, model, prompt_version, piva,
                owner_token, acquired_at, expires_at
            )
            VALUES (
                %s, %s, %s, %s, %s, %s,
                clock_timestamp(),
                clock_timestamp() + make_interval(secs => %s)
            )
            ON CONFLICT (input_hash, provider, model, prompt_version)
            DO UPDATE SET
                piva = EXCLUDED.piva,
                owner_token = EXCLUDED.owner_token,
                acquired_at = EXCLUDED.acquired_at,
                expires_at = EXCLUDED.expires_at
            WHERE research_generation_leases.expires_at <= clock_timestamp()
            RETURNING owner_token
            """,
            (
                input_hash,
                *identity,
                research_input.piva,
                owner_token,
                lease_ttl.total_seconds(),
            ),
        ).fetchone()
        if row is None:
            return None
        returned_token = _uuid(row["owner_token"], "research lease")
        if returned_token != owner_token:  # pragma: no cover - SQL returns EXCLUDED token
            raise RuntimeError("research lease returned an unexpected owner")
        return ResearchGenerationLease(
            input_hash=input_hash,
            provider=identity[0],
            model=identity[1],
            prompt_version=identity[2],
            piva=research_input.piva,
            owner_token=owner_token,
        )

    def release_generation_lease(self, lease: ResearchGenerationLease) -> bool:
        with self._operation_lock:
            return self._release_generation_lease(lease)

    def _release_generation_lease(self, lease: ResearchGenerationLease) -> bool:
        """Release only a lease still owned by this exact worker token."""

        row = self._connection.execute(
            """
            DELETE FROM list_engine.research_generation_leases
            WHERE input_hash = %s
              AND provider = %s
              AND model = %s
              AND prompt_version = %s
              AND piva = %s
              AND owner_token = %s
            RETURNING owner_token
            """,
            (
                lease.input_hash,
                lease.provider,
                lease.model,
                lease.prompt_version,
                lease.piva,
                lease.owner_token,
            ),
        ).fetchone()
        return row is not None

    def get_cached(
        self,
        research_input: ResearchInput,
        *,
        provider: str,
        model: str,
        prompt_version: str,
    ) -> ApprovedDossier | None:
        with self._operation_lock:
            return self._get_cached(
                research_input,
                provider=provider,
                model=model,
                prompt_version=prompt_version,
            )

    def _get_cached(
        self,
        research_input: ResearchInput,
        *,
        provider: str,
        model: str,
        prompt_version: str,
    ) -> ApprovedDossier | None:
        """Read an unexpired eval-approved result for the exact agent identity."""

        identity = _identity(provider, model, prompt_version)
        row = self._connection.execute(
            """
            SELECT
                cache.piva,
                cache.generation,
                cache.evaluation,
                session.piva AS audited_piva,
                session.provider AS audited_provider,
                session.model AS audited_model,
                session.requested_model AS audited_requested_model,
                session.prompt_version AS audited_prompt_version,
                session.status AS audited_status,
                session.eval_passed AS audited_eval_passed,
                session.output AS audited_generation
            FROM list_engine.research_cache AS cache
            JOIN list_engine.agent_sessions AS session
              ON session.id = cache.agent_session_id
            WHERE cache.input_hash = %s
              AND cache.provider = %s
              AND cache.model = %s
              AND cache.prompt_version = %s
              AND cache.created_at <= clock_timestamp()
              AND cache.expires_at > clock_timestamp()
            """,
            (research_input_hash(research_input), *identity),
        ).fetchone()
        if row is None:
            return None

        # JSONB is decoded to Python lists/strings by psycopg. Strict Pydantic
        # models intentionally reject those Python coercions, while JSON-mode
        # validation correctly maps JSON arrays and decimal strings.
        generation = AgentGeneration.model_validate_json(
            json.dumps(row["generation"], ensure_ascii=False)
        )
        evaluation = DossierEvaluation.model_validate_json(
            json.dumps(row["evaluation"], ensure_ascii=False)
        )
        expected_identity = identity
        generation_identity = (
            generation.provider,
            generation.requested_model,
            generation.prompt_version,
        )
        audited_identity = (
            row["audited_provider"],
            row["audited_requested_model"],
            row["audited_prompt_version"],
        )
        if (
            row["piva"] != research_input.piva
            or row["audited_piva"] != research_input.piva
            or generation.dossier.piva != research_input.piva
        ):
            raise RuntimeError("research cache identity does not match its P.IVA")
        if generation_identity != expected_identity or audited_identity != expected_identity:
            raise RuntimeError("research cache identity does not match its lookup key")
        if (
            row["audited_model"] != generation.model
            or row["audited_status"] != "succeeded"
            or row["audited_eval_passed"] is not True
            or row["audited_generation"] != row["generation"]
        ):
            raise RuntimeError("research cache is detached from its immutable audit session")
        if not evaluation.passed or not CitationGate().evaluate(research_input, generation).passed:
            raise RuntimeError("research cache failed its mandatory citation gate")
        return ApprovedDossier(generation=generation, evaluation=evaluation)

    def record(
        self,
        research_input: ResearchInput,
        generation: AgentGeneration,
        evaluation: DossierEvaluation,
        *,
        lease: ResearchGenerationLease,
        cache_ttl: timedelta = timedelta(hours=24),
    ) -> PersistedResearchResult:
        with self._operation_lock:
            return self._record(
                research_input,
                generation,
                evaluation,
                lease=lease,
                cache_ttl=cache_ttl,
            )

    def _record(
        self,
        research_input: ResearchInput,
        generation: AgentGeneration,
        evaluation: DossierEvaluation,
        *,
        lease: ResearchGenerationLease,
        cache_ttl: timedelta,
    ) -> PersistedResearchResult:
        """Append an audit session and route pass/fail to cache or human review."""

        if generation.dossier.piva != research_input.piva:
            raise ValueError("generation P.IVA does not match the research input")
        if cache_ttl <= timedelta(0):
            raise ValueError("cache_ttl must be positive")
        self._validate_lease_identity(research_input, lease)
        if (
            generation.provider != lease.provider
            or generation.requested_model != lease.model
            or generation.prompt_version != lease.prompt_version
        ):
            raise ValueError("generation identity does not match its research lease")

        input_hash = research_input_hash(research_input)
        mandatory_evaluation = CitationGate().evaluate(research_input, generation)
        if not mandatory_evaluation.passed:
            effective_evaluation = mandatory_evaluation
        elif not evaluation.passed:
            # A semantic or business gate may make the deterministic pass stricter.
            effective_evaluation = evaluation
        else:
            # Never trust a caller-provided passing flag or reuse an eval from a
            # different generation. Recompute the mandatory trust boundary here.
            effective_evaluation = mandatory_evaluation

        status = "succeeded" if effective_evaluation.passed else "review"
        approved = (
            ApprovedDossier(generation=generation, evaluation=effective_evaluation)
            if effective_evaluation.passed
            else None
        )
        messages = [
            {
                "role": "input",
                "content": research_input.model_dump(mode="json"),
            },
            {
                "role": "sdk_result",
                "session_id": generation.session_id,
                "content": generation.dossier.model_dump(mode="json", by_alias=True),
            },
        ]
        generation_payload = generation.model_dump(mode="json", by_alias=True)
        evaluation_payload = effective_evaluation.model_dump(mode="json")

        with self._connection.transaction():
            self._assert_active_lease(lease)
            session = self._connection.execute(
                """
                INSERT INTO list_engine.agent_sessions (
                    piva, agent_name, provider, model, requested_model,
                    prompt_version, status,
                    messages, output, input_tokens, output_tokens, cost_eur,
                    eval_passed, started_at, completed_at, input_hash,
                    evaluation, cost_usd, lease_owner_token
                )
                VALUES (
                    %s, 'research_dossier', %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, 0, %s,
                    statement_timestamp(), statement_timestamp(), %s, %s, %s, %s
                )
                RETURNING id
                """,
                (
                    research_input.piva,
                    generation.provider,
                    generation.model,
                    generation.requested_model,
                    generation.prompt_version,
                    status,
                    Jsonb(messages),
                    Jsonb(generation_payload),
                    generation.input_tokens,
                    generation.output_tokens,
                    effective_evaluation.passed,
                    input_hash,
                    Jsonb(evaluation_payload),
                    generation.cost_usd,
                    lease.owner_token,
                ),
            ).fetchone()
            if session is None:  # pragma: no cover - INSERT always returns a row
                raise RuntimeError("research session insert returned no identifier")
            session_id = _uuid(session["id"], "agent session")

            review_queue_id: UUID | None = None
            if effective_evaluation.passed:
                self._cache_approved(
                    research_input,
                    generation,
                    effective_evaluation,
                    agent_session_id=session_id,
                    lease=lease,
                    input_hash=input_hash,
                    cache_ttl=cache_ttl,
                )
            else:
                issue_codes = (
                    ",".join(issue.code for issue in effective_evaluation.issues) or "failed"
                )
                review = self._connection.execute(
                    """
                    INSERT INTO list_engine.review_queue (piva, reason, payload)
                    VALUES (%s, %s, %s)
                    RETURNING id
                    """,
                    (
                        research_input.piva,
                        f"research_eval:{issue_codes}",
                        Jsonb(
                            {
                                "agent_session_id": str(session_id),
                                "input_hash": input_hash,
                                "generation": generation_payload,
                                "evaluation": evaluation_payload,
                            }
                        ),
                    ),
                ).fetchone()
                if review is None:  # pragma: no cover - INSERT always returns a row
                    raise RuntimeError("review queue insert returned no identifier")
                review_queue_id = _uuid(review["id"], "review queue")

        return PersistedResearchResult(
            session_id=session_id,
            approved=approved,
            review_queue_id=review_queue_id,
        )

    def record_failure(
        self,
        research_input: ResearchInput,
        *,
        lease: ResearchGenerationLease,
        error_type: str,
        attempts: tuple[AgentAttempt, ...] = (),
    ) -> PersistedResearchFailure:
        with self._operation_lock:
            return self._record_failure(
                research_input,
                lease=lease,
                error_type=error_type,
                attempts=attempts,
            )

    def _record_failure(
        self,
        research_input: ResearchInput,
        *,
        lease: ResearchGenerationLease,
        error_type: str,
        attempts: tuple[AgentAttempt, ...],
    ) -> PersistedResearchFailure:
        """Audit a failed/abstained generation without persisting raw error text."""

        self._validate_lease_identity(research_input, lease)
        error_type = error_type.strip()
        if not error_type or len(error_type) > 128:
            raise ValueError("error_type must be non-blank and at most 128 characters")
        input_hash = research_input_hash(research_input)
        attempt_payload = [attempt.model_dump(mode="json") for attempt in attempts]
        total_cost = sum((attempt.cost_usd for attempt in attempts), start=Decimal(0))
        input_tokens = _complete_usage(attempts, "input_tokens")
        output_tokens = _complete_usage(attempts, "output_tokens")
        messages = [
            {"role": "input", "content": research_input.model_dump(mode="json")},
            {
                "role": "failure",
                "error_type": error_type,
                "attempts": attempt_payload,
            },
        ]

        with self._connection.transaction():
            self._assert_active_lease(lease)
            session = self._connection.execute(
                """
                INSERT INTO list_engine.agent_sessions (
                    piva, agent_name, provider, model, requested_model,
                    prompt_version, status,
                    messages, output, input_tokens, output_tokens, cost_eur,
                    eval_passed, started_at, completed_at, input_hash,
                    evaluation, cost_usd, lease_owner_token
                )
                VALUES (
                    %s, 'research_dossier', %s, %s, %s, %s, 'failed',
                    %s, %s, %s, %s, 0, false,
                    statement_timestamp(), statement_timestamp(), %s, NULL, %s, %s
                )
                RETURNING id
                """,
                (
                    research_input.piva,
                    lease.provider,
                    lease.model,
                    lease.model,
                    lease.prompt_version,
                    Jsonb(messages),
                    Jsonb({"attempts": attempt_payload, "error_type": error_type}),
                    input_tokens,
                    output_tokens,
                    input_hash,
                    total_cost,
                    lease.owner_token,
                ),
            ).fetchone()
            if session is None:  # pragma: no cover - INSERT always returns a row
                raise RuntimeError("failed research session insert returned no identifier")
            session_id = _uuid(session["id"], "agent session")
            review = self._connection.execute(
                """
                INSERT INTO list_engine.review_queue (piva, reason, payload)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (
                    research_input.piva,
                    f"research_generation_failed:{error_type}",
                    Jsonb(
                        {
                            "agent_session_id": str(session_id),
                            "input_hash": input_hash,
                            "attempts": attempt_payload,
                        }
                    ),
                ),
            ).fetchone()
            if review is None:  # pragma: no cover - INSERT always returns a row
                raise RuntimeError("failed research review insert returned no identifier")
            review_queue_id = _uuid(review["id"], "review queue")

        return PersistedResearchFailure(
            session_id=session_id,
            review_queue_id=review_queue_id,
        )

    def record_superseded_generation(
        self,
        research_input: ResearchInput,
        generation: AgentGeneration,
        evaluation: DossierEvaluation,
        *,
        lease: ResearchGenerationLease,
    ) -> PersistedResearchResult:
        with self._operation_lock:
            return self._record_superseded_generation(
                research_input,
                generation,
                evaluation,
                lease=lease,
            )

    def _record_superseded_generation(
        self,
        research_input: ResearchInput,
        generation: AgentGeneration,
        evaluation: DossierEvaluation,
        *,
        lease: ResearchGenerationLease,
    ) -> PersistedResearchResult:
        """Audit a completed generation whose lease no longer permits publication."""

        if generation.dossier.piva != research_input.piva:
            raise ValueError("generation P.IVA does not match the research input")
        self._validate_lease_identity(research_input, lease)
        if (
            generation.provider != lease.provider
            or generation.requested_model != lease.model
            or generation.prompt_version != lease.prompt_version
        ):
            raise ValueError("generation identity does not match its research lease")

        mandatory_evaluation = CitationGate().evaluate(research_input, generation)
        effective_evaluation = (
            mandatory_evaluation
            if not mandatory_evaluation.passed or evaluation.passed
            else evaluation
        )
        input_hash = research_input_hash(research_input)
        generation_payload = generation.model_dump(mode="json", by_alias=True)
        evaluation_payload = effective_evaluation.model_dump(mode="json")
        messages = [
            {"role": "input", "content": research_input.model_dump(mode="json")},
            {
                "role": "sdk_result",
                "session_id": generation.session_id,
                "content": generation.dossier.model_dump(mode="json", by_alias=True),
            },
            {
                "role": "publication",
                "status": "superseded",
                "reason": "generation_lease_lost",
            },
        ]
        session = self._connection.execute(
            """
            INSERT INTO list_engine.agent_sessions (
                piva, agent_name, provider, model, requested_model,
                prompt_version, status,
                messages, output, input_tokens, output_tokens, cost_eur,
                eval_passed, started_at, completed_at, input_hash,
                evaluation, cost_usd, lease_owner_token
            )
            VALUES (
                %s, 'research_dossier', %s, %s, %s, %s, 'superseded',
                %s, %s, %s, %s, 0, %s,
                statement_timestamp(), statement_timestamp(), %s, %s, %s, %s
            )
            RETURNING id
            """,
            (
                research_input.piva,
                generation.provider,
                generation.model,
                generation.requested_model,
                generation.prompt_version,
                Jsonb(messages),
                Jsonb(generation_payload),
                generation.input_tokens,
                generation.output_tokens,
                effective_evaluation.passed,
                input_hash,
                Jsonb(evaluation_payload),
                generation.cost_usd,
                lease.owner_token,
            ),
        ).fetchone()
        if session is None:  # pragma: no cover - INSERT always returns a row
            raise RuntimeError("superseded research session insert returned no identifier")
        return PersistedResearchResult(
            session_id=_uuid(session["id"], "superseded agent session"),
            approved=None,
            review_queue_id=None,
            superseded=True,
        )

    def record_superseded_failure(
        self,
        research_input: ResearchInput,
        *,
        lease: ResearchGenerationLease,
        error_type: str,
        attempts: tuple[AgentAttempt, ...] = (),
    ) -> PersistedResearchFailure:
        with self._operation_lock:
            return self._record_superseded_failure(
                research_input,
                lease=lease,
                error_type=error_type,
                attempts=attempts,
            )

    def _record_superseded_failure(
        self,
        research_input: ResearchInput,
        *,
        lease: ResearchGenerationLease,
        error_type: str,
        attempts: tuple[AgentAttempt, ...],
    ) -> PersistedResearchFailure:
        """Audit a terminal failed call after ownership moved to another worker."""

        self._validate_lease_identity(research_input, lease)
        error_type = error_type.strip()
        if not error_type or len(error_type) > 128:
            raise ValueError("error_type must be non-blank and at most 128 characters")
        input_hash = research_input_hash(research_input)
        attempt_payload = [attempt.model_dump(mode="json") for attempt in attempts]
        total_cost = sum((attempt.cost_usd for attempt in attempts), start=Decimal(0))
        input_tokens = _complete_usage(attempts, "input_tokens")
        output_tokens = _complete_usage(attempts, "output_tokens")
        messages = [
            {"role": "input", "content": research_input.model_dump(mode="json")},
            {"role": "failure", "error_type": error_type, "attempts": attempt_payload},
            {
                "role": "publication",
                "status": "superseded",
                "reason": "generation_lease_lost",
            },
        ]
        resolved_model = attempts[-1].model if attempts else lease.model
        session = self._connection.execute(
            """
            INSERT INTO list_engine.agent_sessions (
                piva, agent_name, provider, model, requested_model,
                prompt_version, status,
                messages, output, input_tokens, output_tokens, cost_eur,
                eval_passed, started_at, completed_at, input_hash,
                evaluation, cost_usd, lease_owner_token
            )
            VALUES (
                %s, 'research_dossier', %s, %s, %s, %s, 'superseded',
                %s, %s, %s, %s, 0, false,
                statement_timestamp(), statement_timestamp(), %s, NULL, %s, %s
            )
            RETURNING id
            """,
            (
                research_input.piva,
                lease.provider,
                resolved_model,
                lease.model,
                lease.prompt_version,
                Jsonb(messages),
                Jsonb({"attempts": attempt_payload, "error_type": error_type}),
                input_tokens,
                output_tokens,
                input_hash,
                total_cost,
                lease.owner_token,
            ),
        ).fetchone()
        if session is None:  # pragma: no cover - INSERT always returns a row
            raise RuntimeError("superseded failure session insert returned no identifier")
        return PersistedResearchFailure(
            session_id=_uuid(session["id"], "superseded failure session"),
            review_queue_id=None,
            superseded=True,
        )

    def _cache_approved(
        self,
        research_input: ResearchInput,
        generation: AgentGeneration,
        evaluation: DossierEvaluation,
        *,
        agent_session_id: UUID,
        lease: ResearchGenerationLease,
        input_hash: str,
        cache_ttl: timedelta,
    ) -> None:
        row = self._connection.execute(
            """
            INSERT INTO list_engine.research_cache (
                agent_session_id, piva, input_hash, provider, model, prompt_version,
                generation, evaluation, created_at, expires_at
            )
            SELECT
                %s, %s, %s, %s, %s, %s, %s, %s,
                statement_timestamp(),
                statement_timestamp() + make_interval(secs => %s)
            FROM list_engine.research_generation_leases AS lease
            WHERE lease.input_hash = %s
              AND lease.provider = %s
              AND lease.model = %s
              AND lease.prompt_version = %s
              AND lease.piva = %s
              AND lease.owner_token = %s
              AND lease.expires_at > clock_timestamp()
            ON CONFLICT (input_hash, provider, model, prompt_version)
            DO UPDATE SET
                agent_session_id = EXCLUDED.agent_session_id,
                piva = EXCLUDED.piva,
                generation = EXCLUDED.generation,
                evaluation = EXCLUDED.evaluation,
                created_at = EXCLUDED.created_at,
                expires_at = EXCLUDED.expires_at
            WHERE research_cache.expires_at <= clock_timestamp()
            RETURNING id
            """,
            (
                agent_session_id,
                research_input.piva,
                input_hash,
                lease.provider,
                lease.model,
                lease.prompt_version,
                Jsonb(generation.model_dump(mode="json", by_alias=True)),
                Jsonb(evaluation.model_dump(mode="json")),
                cache_ttl.total_seconds(),
                lease.input_hash,
                lease.provider,
                lease.model,
                lease.prompt_version,
                lease.piva,
                lease.owner_token,
            ),
        ).fetchone()
        if row is None:
            raise ResearchLeaseLostError("research generation lease expired before cache publish")

    def _validate_lease_identity(
        self,
        research_input: ResearchInput,
        lease: ResearchGenerationLease,
    ) -> None:
        if lease.piva != research_input.piva or lease.input_hash != research_input_hash(
            research_input
        ):
            raise ValueError("research lease does not match its immutable input")

    def _assert_active_lease(self, lease: ResearchGenerationLease) -> None:
        row = self._connection.execute(
            """
            SELECT 1
            FROM list_engine.research_generation_leases
            WHERE input_hash = %s
              AND provider = %s
              AND model = %s
              AND prompt_version = %s
              AND piva = %s
              AND owner_token = %s
              AND expires_at > clock_timestamp()
            FOR UPDATE
            """,
            (
                lease.input_hash,
                lease.provider,
                lease.model,
                lease.prompt_version,
                lease.piva,
                lease.owner_token,
            ),
        ).fetchone()
        if row is None:
            raise ResearchLeaseLostError("research generation lease is no longer active")


def _identity(provider: str, model: str, prompt_version: str) -> tuple[str, str, str]:
    values = (provider.strip(), model.strip(), prompt_version.strip())
    if any(not value for value in values):
        raise ValueError("provider, model and prompt_version cannot be blank")
    return values


def _uuid(value: object, label: str) -> UUID:
    if not isinstance(value, UUID):  # pragma: no cover - psycopg adapter contract
        raise RuntimeError(f"PostgreSQL returned a non-UUID {label} ID")
    return value


def _complete_usage(
    attempts: tuple[AgentAttempt, ...],
    field: str,
) -> int | None:
    values: tuple[int | None, ...] = tuple(getattr(attempt, field) for attempt in attempts)
    if not values or any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


__all__ = [
    "PersistedResearchFailure",
    "PersistedResearchResult",
    "PostgresResearchStore",
    "ResearchGenerationLease",
    "ResearchLeaseLostError",
    "research_input_hash",
]
