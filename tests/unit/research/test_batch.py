from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import uuid4

import pytest

from list_engine.research.agent import (
    DEMO_MODEL,
    DEMO_PROMPT_VERSION,
    DEMO_PROVIDER,
    DemoResearchAgent,
    ResearchAbstentionError,
    ResearchAgent,
)
from list_engine.research.batch import (
    AgentIdentity,
    BatchItemStatus,
    ColdResearchBatch,
)
from list_engine.research.gate import CitationGate
from list_engine.research.models import (
    AgentAttempt,
    AgentGeneration,
    ApprovedDossier,
    DossierEvaluation,
    EvidenceItem,
    ResearchInput,
)
from list_engine.research.postgres import (
    PersistedResearchFailure,
    PersistedResearchResult,
    ResearchGenerationLease,
    ResearchLeaseLostError,
    research_input_hash,
)

NOW = datetime(2026, 7, 15, 8, tzinfo=UTC)


def packet(piva: str = "99000000002") -> ResearchInput:
    return ResearchInput(
        piva=piva,
        as_of=NOW.date(),
        evidence=(
            EvidenceItem(
                evidence_id="registry_1",
                source_url=f"https://registro.invalid/{piva}",
                source_kind="official_registry",
                title="Registro",
                excerpt="Impresa attiva con una nuova sede amministrativa.",
                observed_at=NOW,
                verified=True,
                tags=("signal:new_office",),
                valid_until=NOW.date(),
            ),
        ),
    )


class MemoryStore:
    def __init__(self) -> None:
        self.cached: dict[str, ApprovedDossier] = {}
        self.record_calls = 0
        self.force_review = False
        self.denied_leases: set[str] = set()
        self.failure_calls = 0
        self.failure_error_types: list[str] = []
        self.lose_lease_on_record: set[str] = set()
        self.lose_lease_on_failure: set[str] = set()
        self.superseded_generation_calls = 0
        self.superseded_failure_calls = 0

    def get_cached(
        self,
        research_input: ResearchInput,
        *,
        provider: str,
        model: str,
        prompt_version: str,
    ) -> ApprovedDossier | None:
        del provider, model, prompt_version
        return self.cached.get(research_input.piva)

    def record(
        self,
        research_input: ResearchInput,
        generation: AgentGeneration,
        evaluation: DossierEvaluation,
        *,
        lease: ResearchGenerationLease,
        cache_ttl: timedelta = timedelta(hours=24),
    ) -> PersistedResearchResult:
        del lease, cache_ttl
        if research_input.piva in self.lose_lease_on_record:
            raise ResearchLeaseLostError("expired")
        self.record_calls += 1
        approved = None
        if evaluation.passed and not self.force_review:
            approved = ApprovedDossier(generation=generation, evaluation=evaluation)
            self.cached[research_input.piva] = approved
        return PersistedResearchResult(
            session_id=uuid4(),
            approved=approved,
            review_queue_id=None if approved is not None else uuid4(),
        )

    def acquire_generation_lease(
        self,
        research_input: ResearchInput,
        *,
        provider: str,
        model: str,
        prompt_version: str,
        lease_ttl: timedelta = timedelta(minutes=10),
    ) -> ResearchGenerationLease | None:
        del lease_ttl
        if research_input.piva in self.denied_leases:
            return None
        return ResearchGenerationLease(
            input_hash=research_input_hash(research_input),
            provider=provider,
            model=model,
            prompt_version=prompt_version,
            piva=research_input.piva,
            owner_token=uuid4(),
        )

    def release_generation_lease(self, lease: ResearchGenerationLease) -> bool:
        del lease
        return True

    def record_failure(
        self,
        research_input: ResearchInput,
        *,
        lease: ResearchGenerationLease,
        error_type: str,
        attempts: tuple[AgentAttempt, ...] = (),
    ) -> PersistedResearchFailure:
        del lease, attempts
        if research_input.piva in self.lose_lease_on_failure:
            raise ResearchLeaseLostError("expired")
        self.failure_calls += 1
        self.failure_error_types.append(error_type)
        return PersistedResearchFailure(session_id=uuid4(), review_queue_id=uuid4())

    def record_superseded_generation(
        self,
        research_input: ResearchInput,
        generation: AgentGeneration,
        evaluation: DossierEvaluation,
        *,
        lease: ResearchGenerationLease,
    ) -> PersistedResearchResult:
        del research_input, generation, evaluation, lease
        self.superseded_generation_calls += 1
        return PersistedResearchResult(
            session_id=uuid4(),
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
        del research_input, lease, error_type, attempts
        self.superseded_failure_calls += 1
        return PersistedResearchFailure(
            session_id=uuid4(),
            review_queue_id=None,
            superseded=True,
        )


class WrongIdentityAgent:
    async def generate(self, research_input: ResearchInput) -> AgentGeneration:
        generation = await DemoResearchAgent().generate(research_input)
        return generation.model_copy(update={"requested_model": "unexpected-model"})


class SelectiveFailureAgent:
    async def generate(self, research_input: ResearchInput) -> AgentGeneration:
        if research_input.piva == "99000000002":
            raise ResearchAbstentionError("no verified support")
        return await DemoResearchAgent().generate(research_input)


class SelectiveTimeoutAgent:
    async def generate(self, research_input: ResearchInput) -> AgentGeneration:
        if research_input.piva == "99000000002":
            await asyncio.sleep(1)
        return await DemoResearchAgent().generate(research_input)


def batch(store: MemoryStore, agent: ResearchAgent | None = None) -> ColdResearchBatch:
    return ColdResearchBatch(
        agent=agent or DemoResearchAgent(),
        gate=CitationGate(),
        store=store,
        identity=AgentIdentity(
            provider=DEMO_PROVIDER,
            model=DEMO_MODEL,
            prompt_version=DEMO_PROMPT_VERSION,
        ),
    )


def test_cold_batch_generates_once_then_uses_the_approved_cache() -> None:
    store = MemoryStore()
    runner = batch(store)

    first = asyncio.run(runner.run((packet(),)))
    second = asyncio.run(runner.run((packet(),)))

    assert first.generated == 1
    assert first.items[0].status is BatchItemStatus.GENERATED
    assert second.cache_hits == 1
    assert second.items[0].status is BatchItemStatus.CACHE_HIT
    assert store.record_calls == 1


def test_reviewed_generation_is_never_returned_as_a_dossier() -> None:
    store = MemoryStore()
    store.force_review = True

    result = asyncio.run(batch(store).run((packet(),)))

    assert result.routed_to_review == 1
    assert result.items[0].dossier is None


def test_agent_identity_drift_fails_before_cache_write() -> None:
    store = MemoryStore()
    runner = batch(store, cast(ResearchAgent, WrongIdentityAgent()))

    with pytest.raises(RuntimeError, match="refusing to poison"):
        asyncio.run(runner.run((packet(),)))
    assert store.record_calls == 0


def test_contended_account_is_deferred_without_blocking_the_next_item() -> None:
    store = MemoryStore()
    store.denied_leases.add("99000000002")

    result = asyncio.run(batch(store).run((packet(), packet("99000000010"))))

    assert result.deferred == 1
    assert result.generated == 1
    assert [item.status for item in result.items] == [
        BatchItemStatus.DEFERRED,
        BatchItemStatus.GENERATED,
    ]


def test_failed_account_is_audited_without_aborting_the_rest_of_the_batch() -> None:
    store = MemoryStore()
    runner = batch(store, cast(ResearchAgent, SelectiveFailureAgent()))

    result = asyncio.run(runner.run((packet(), packet("99000000010"))))

    assert result.failed == 1
    assert result.generated == 1
    assert result.items[0].error_type == "ResearchAbstentionError"
    assert result.items[1].status is BatchItemStatus.GENERATED
    assert store.failure_calls == 1


def test_timed_out_account_is_audited_without_aborting_the_rest_of_the_batch() -> None:
    store = MemoryStore()
    runner = ColdResearchBatch(
        agent=cast(ResearchAgent, SelectiveTimeoutAgent()),
        gate=CitationGate(),
        store=store,
        identity=AgentIdentity(
            provider=DEMO_PROVIDER,
            model=DEMO_MODEL,
            prompt_version=DEMO_PROMPT_VERSION,
        ),
        agent_timeout=timedelta(milliseconds=10),
    )

    result = asyncio.run(runner.run((packet(), packet("99000000010"))))

    assert [item.status for item in result.items] == [
        BatchItemStatus.FAILED,
        BatchItemStatus.GENERATED,
    ]
    assert result.items[0].error_type == "TimeoutError"
    assert store.failure_calls == 1
    assert store.failure_error_types == ["TimeoutError"]


def test_lost_publish_lease_is_deferred_without_aborting_the_batch() -> None:
    store = MemoryStore()
    store.lose_lease_on_record.add("99000000002")

    result = asyncio.run(batch(store).run((packet(), packet("99000000010"))))

    assert [item.status for item in result.items] == [
        BatchItemStatus.DEFERRED,
        BatchItemStatus.GENERATED,
    ]
    assert result.items[0].error_type == "ResearchLeaseLostError"
    assert store.superseded_generation_calls == 1


def test_lost_failure_lease_is_deferred_with_a_superseded_audit() -> None:
    store = MemoryStore()
    store.lose_lease_on_failure.add("99000000002")
    runner = batch(store, cast(ResearchAgent, SelectiveFailureAgent()))

    result = asyncio.run(runner.run((packet(), packet("99000000010"))))

    assert result.deferred == 1
    assert result.failed == 0
    assert result.generated == 1
    assert result.items[0].error_type == "ResearchLeaseLostError"
    assert store.failure_calls == 0
    assert store.superseded_failure_calls == 1


def test_timed_out_account_with_lost_lease_is_deferred_and_batch_continues() -> None:
    store = MemoryStore()
    store.lose_lease_on_failure.add("99000000002")
    runner = ColdResearchBatch(
        agent=cast(ResearchAgent, SelectiveTimeoutAgent()),
        gate=CitationGate(),
        store=store,
        identity=AgentIdentity(
            provider=DEMO_PROVIDER,
            model=DEMO_MODEL,
            prompt_version=DEMO_PROMPT_VERSION,
        ),
        agent_timeout=timedelta(milliseconds=10),
    )

    result = asyncio.run(runner.run((packet(), packet("99000000010"))))

    assert [item.status for item in result.items] == [
        BatchItemStatus.DEFERRED,
        BatchItemStatus.GENERATED,
    ]
    assert result.items[0].error_type == "ResearchLeaseLostError"
    assert store.failure_calls == 0
    assert store.superseded_failure_calls == 1


def test_batch_rejects_non_positive_cache_ttl() -> None:
    with pytest.raises(ValueError, match="positive"):
        ColdResearchBatch(
            agent=DemoResearchAgent(),
            gate=CitationGate(),
            store=MemoryStore(),
            identity=AgentIdentity(
                provider=DEMO_PROVIDER,
                model=DEMO_MODEL,
                prompt_version=DEMO_PROMPT_VERSION,
            ),
            cache_ttl=timedelta(0),
        )


@pytest.mark.parametrize(
    "agent_timeout",
    [timedelta(0), timedelta(microseconds=-1)],
)
def test_batch_rejects_non_positive_agent_timeout(agent_timeout: timedelta) -> None:
    with pytest.raises(ValueError, match="positive"):
        ColdResearchBatch(
            agent=DemoResearchAgent(),
            gate=CitationGate(),
            store=MemoryStore(),
            identity=AgentIdentity(
                provider=DEMO_PROVIDER,
                model=DEMO_MODEL,
                prompt_version=DEMO_PROMPT_VERSION,
            ),
            agent_timeout=agent_timeout,
        )


@pytest.mark.parametrize(
    "agent_timeout",
    [timedelta(minutes=10), timedelta(minutes=11)],
)
def test_batch_requires_agent_timeout_strictly_below_lease(
    agent_timeout: timedelta,
) -> None:
    with pytest.raises(ValueError, match="strictly below"):
        ColdResearchBatch(
            agent=DemoResearchAgent(),
            gate=CitationGate(),
            store=MemoryStore(),
            identity=AgentIdentity(
                provider=DEMO_PROVIDER,
                model=DEMO_MODEL,
                prompt_version=DEMO_PROMPT_VERSION,
            ),
            lease_ttl=timedelta(minutes=10),
            agent_timeout=agent_timeout,
        )
