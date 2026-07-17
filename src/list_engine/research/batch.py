"""Cache-first cold research batch bounded by deterministic workflow contracts."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from typing import Protocol

from list_engine.research.agent import ResearchAgent
from list_engine.research.gate import CitationGate
from list_engine.research.models import (
    AgentAttempt,
    AgentGeneration,
    ApprovedDossier,
    DossierEvaluation,
    ResearchInput,
)
from list_engine.research.postgres import (
    PersistedResearchFailure,
    PersistedResearchResult,
    ResearchGenerationLease,
    ResearchLeaseLostError,
)


@dataclass(frozen=True, slots=True)
class AgentIdentity:
    provider: str
    model: str
    prompt_version: str

    def __post_init__(self) -> None:
        if any(
            not value or value != value.strip()
            for value in (self.provider, self.model, self.prompt_version)
        ):
            raise ValueError("agent identity values must be non-blank and trimmed")


class ResearchStore(Protocol):
    """Synchronous persistence port called inside durable workflow steps."""

    def get_cached(
        self,
        research_input: ResearchInput,
        *,
        provider: str,
        model: str,
        prompt_version: str,
    ) -> ApprovedDossier | None: ...

    def acquire_generation_lease(
        self,
        research_input: ResearchInput,
        *,
        provider: str,
        model: str,
        prompt_version: str,
        lease_ttl: timedelta = timedelta(minutes=10),
    ) -> ResearchGenerationLease | None: ...

    def release_generation_lease(self, lease: ResearchGenerationLease) -> bool: ...

    def record(
        self,
        research_input: ResearchInput,
        generation: AgentGeneration,
        evaluation: DossierEvaluation,
        *,
        lease: ResearchGenerationLease,
        cache_ttl: timedelta = timedelta(hours=24),
    ) -> PersistedResearchResult: ...

    def record_failure(
        self,
        research_input: ResearchInput,
        *,
        lease: ResearchGenerationLease,
        error_type: str,
        attempts: tuple[AgentAttempt, ...] = (),
    ) -> PersistedResearchFailure: ...

    def record_superseded_generation(
        self,
        research_input: ResearchInput,
        generation: AgentGeneration,
        evaluation: DossierEvaluation,
        *,
        lease: ResearchGenerationLease,
    ) -> PersistedResearchResult: ...

    def record_superseded_failure(
        self,
        research_input: ResearchInput,
        *,
        lease: ResearchGenerationLease,
        error_type: str,
        attempts: tuple[AgentAttempt, ...] = (),
    ) -> PersistedResearchFailure: ...


class BatchItemStatus(StrEnum):
    CACHE_HIT = "cache_hit"
    DEFERRED = "deferred"
    FAILED = "failed"
    GENERATED = "generated"
    REVIEW = "review"


@dataclass(frozen=True, slots=True)
class BatchItemResult:
    piva: str
    status: BatchItemStatus
    dossier: ApprovedDossier | None
    error_type: str | None = None


@dataclass(frozen=True, slots=True)
class ColdBatchResult:
    items: tuple[BatchItemResult, ...]

    @property
    def cache_hits(self) -> int:
        return sum(item.status is BatchItemStatus.CACHE_HIT for item in self.items)

    @property
    def generated(self) -> int:
        return sum(item.status is BatchItemStatus.GENERATED for item in self.items)

    @property
    def routed_to_review(self) -> int:
        return sum(item.status is BatchItemStatus.REVIEW for item in self.items)

    @property
    def deferred(self) -> int:
        return sum(item.status is BatchItemStatus.DEFERRED for item in self.items)

    @property
    def failed(self) -> int:
        return sum(item.status is BatchItemStatus.FAILED for item in self.items)


class ColdResearchBatch:
    """Run cold dossiers cache-first; leave scheduling/retries to the orchestrator."""

    def __init__(
        self,
        *,
        agent: ResearchAgent,
        gate: CitationGate,
        store: ResearchStore,
        identity: AgentIdentity,
        cache_ttl: timedelta = timedelta(hours=24),
        lease_ttl: timedelta = timedelta(minutes=10),
        agent_timeout: timedelta = timedelta(minutes=8),
    ) -> None:
        if cache_ttl <= timedelta(0) or lease_ttl <= timedelta(0) or agent_timeout <= timedelta(0):
            raise ValueError("cache_ttl, lease_ttl, and agent_timeout must be positive")
        if agent_timeout >= lease_ttl:
            raise ValueError("agent_timeout must be strictly below lease_ttl")
        self._agent = agent
        self._gate = gate
        self._store = store
        self._identity = identity
        self._cache_ttl = cache_ttl
        self._lease_ttl = lease_ttl
        self._agent_timeout = agent_timeout

    async def run(self, inputs: tuple[ResearchInput, ...]) -> ColdBatchResult:
        results: list[BatchItemResult] = []
        for research_input in inputs:
            cached = self._store.get_cached(
                research_input,
                provider=self._identity.provider,
                model=self._identity.model,
                prompt_version=self._identity.prompt_version,
            )
            if cached is not None:
                results.append(
                    BatchItemResult(
                        piva=research_input.piva,
                        status=BatchItemStatus.CACHE_HIT,
                        dossier=cached,
                    )
                )
                continue

            lease = self._store.acquire_generation_lease(
                research_input,
                provider=self._identity.provider,
                model=self._identity.model,
                prompt_version=self._identity.prompt_version,
                lease_ttl=self._lease_ttl,
            )
            if lease is None:
                results.append(
                    BatchItemResult(
                        piva=research_input.piva,
                        status=BatchItemStatus.DEFERRED,
                        dossier=None,
                    )
                )
                continue

            try:
                # Another worker may have filled the cache after our first read and
                # released its lease just before this worker acquired it.
                cached = self._store.get_cached(
                    research_input,
                    provider=self._identity.provider,
                    model=self._identity.model,
                    prompt_version=self._identity.prompt_version,
                )
                if cached is not None:
                    results.append(
                        BatchItemResult(
                            piva=research_input.piva,
                            status=BatchItemStatus.CACHE_HIT,
                            dossier=cached,
                        )
                    )
                    continue

                try:
                    generation = await asyncio.wait_for(
                        self._agent.generate(research_input),
                        timeout=self._agent_timeout.total_seconds(),
                    )
                except Exception as error:
                    error_type = type(error).__name__
                    raw_attempts = getattr(error, "attempts", ())
                    attempts = (
                        raw_attempts
                        if isinstance(raw_attempts, tuple)
                        and all(isinstance(attempt, AgentAttempt) for attempt in raw_attempts)
                        else ()
                    )
                    try:
                        persisted_failure = self._store.record_failure(
                            research_input,
                            lease=lease,
                            error_type=error_type,
                            attempts=attempts,
                        )
                    except ResearchLeaseLostError as lease_error:
                        self._store.record_superseded_failure(
                            research_input,
                            lease=lease,
                            error_type=error_type,
                            attempts=attempts,
                        )
                        results.append(
                            BatchItemResult(
                                piva=research_input.piva,
                                status=BatchItemStatus.DEFERRED,
                                dossier=None,
                                error_type=type(lease_error).__name__,
                            )
                        )
                        continue
                    if persisted_failure.superseded:
                        results.append(
                            BatchItemResult(
                                piva=research_input.piva,
                                status=BatchItemStatus.DEFERRED,
                                dossier=None,
                                error_type="ResearchLeaseLostError",
                            )
                        )
                        continue
                    results.append(
                        BatchItemResult(
                            piva=research_input.piva,
                            status=BatchItemStatus.FAILED,
                            dossier=None,
                            error_type=error_type,
                        )
                    )
                    continue
                self._require_expected_identity(generation)
                evaluation = self._gate.evaluate(research_input, generation)
                try:
                    persisted = self._store.record(
                        research_input,
                        generation,
                        evaluation,
                        lease=lease,
                        cache_ttl=self._cache_ttl,
                    )
                except ResearchLeaseLostError as lease_error:
                    self._store.record_superseded_generation(
                        research_input,
                        generation,
                        evaluation,
                        lease=lease,
                    )
                    results.append(
                        BatchItemResult(
                            piva=research_input.piva,
                            status=BatchItemStatus.DEFERRED,
                            dossier=None,
                            error_type=type(lease_error).__name__,
                        )
                    )
                    continue
                if persisted.superseded:
                    results.append(
                        BatchItemResult(
                            piva=research_input.piva,
                            status=BatchItemStatus.DEFERRED,
                            dossier=None,
                            error_type="ResearchLeaseLostError",
                        )
                    )
                    continue
                status = (
                    BatchItemStatus.GENERATED
                    if persisted.approved is not None
                    else BatchItemStatus.REVIEW
                )
                results.append(
                    BatchItemResult(
                        piva=research_input.piva,
                        status=status,
                        dossier=persisted.approved,
                    )
                )
            finally:
                self._store.release_generation_lease(lease)
        return ColdBatchResult(items=tuple(results))

    def _require_expected_identity(self, generation: AgentGeneration) -> None:
        actual = (
            generation.provider,
            generation.requested_model,
            generation.prompt_version,
        )
        expected = (
            self._identity.provider,
            self._identity.model,
            self._identity.prompt_version,
        )
        if actual != expected:
            raise RuntimeError(
                "research agent identity changed during the batch; refusing to poison the cache"
            )


__all__ = [
    "AgentIdentity",
    "BatchItemResult",
    "BatchItemStatus",
    "ColdBatchResult",
    "ColdResearchBatch",
    "ResearchStore",
]
