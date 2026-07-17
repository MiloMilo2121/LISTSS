"""Unit tests for the hot workflow, evidence builder, and event worker."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from list_engine.core.models import Company
from list_engine.core.repository import InMemoryCompanyRepository
from list_engine.hot.demo import demo_clay_signal, demo_company
from list_engine.hot.models import (
    HotOutcome,
    HotSignal,
    HotSignalType,
    OutboxEvent,
    OutboxStatus,
    SignalSource,
)
from list_engine.hot.outbox import InMemoryOutbox
from list_engine.hot.worker import (
    DirectResearchRunner,
    EvidenceUnavailableError,
    HotEventWorker,
    HotWorkflow,
    HotWorkflowResult,
    InMemoryHotTaskStore,
    ResearchOutcome,
    SignalEvidenceBuilder,
)
from list_engine.research.agent import DemoResearchAgent
from list_engine.research.gate import CitationGate

T0 = datetime(2026, 7, 17, 8, 0, 0, tzinfo=UTC)
NOW = datetime(2026, 7, 17, 8, 0, 3, tzinfo=UTC)


def event_for(signal: HotSignal) -> OutboxEvent:
    return OutboxEvent(
        id=uuid4(), signal=signal, available_at=signal.received_at, created_at=signal.received_at
    )


def direct_workflow(
    *, companies: InMemoryCompanyRepository, tasks: InMemoryHotTaskStore
) -> HotWorkflow:
    return HotWorkflow(
        companies=companies,
        research=DirectResearchRunner(DemoResearchAgent(), CitationGate()),
        tasks=tasks,
        evidence=SignalEvidenceBuilder(),
    )


class FakeResearchRunner:
    def __init__(self, outcome: ResearchOutcome) -> None:
        self._outcome = outcome

    async def research(self, research_input: object, *, now: datetime) -> ResearchOutcome:
        return self._outcome


class FakeWorkflowRunner:
    def __init__(
        self, *, outcome: HotOutcome | None = None, error: Exception | None = None
    ) -> None:
        self._outcome = outcome
        self._error = error

    async def run(self, event: OutboxEvent, *, now: datetime) -> HotWorkflowResult:
        if self._error is not None:
            raise self._error
        assert self._outcome is not None
        return HotWorkflowResult(event.piva, self._outcome, None, None)


# --- HotWorkflow branches -------------------------------------------------


def test_workflow_creates_task_then_is_idempotent() -> None:
    companies = InMemoryCompanyRepository([demo_company(observed_at=T0)])
    tasks = InMemoryHotTaskStore()
    workflow = direct_workflow(companies=companies, tasks=tasks)
    event = event_for(demo_clay_signal(received_at=T0))

    first = asyncio.run(workflow.run(event, now=NOW))
    second = asyncio.run(workflow.run(event, now=NOW))

    assert first.outcome is HotOutcome.task_created
    assert second.outcome is HotOutcome.task_exists
    assert len(tasks.all_tasks()) == 1


def test_workflow_unknown_company() -> None:
    workflow = direct_workflow(companies=InMemoryCompanyRepository(), tasks=InMemoryHotTaskStore())
    result = asyncio.run(workflow.run(event_for(demo_clay_signal(received_at=T0)), now=NOW))
    assert result.outcome is HotOutcome.unknown_company


def test_workflow_expired_signal() -> None:
    companies = InMemoryCompanyRepository([demo_company(observed_at=T0)])
    workflow = direct_workflow(companies=companies, tasks=InMemoryHotTaskStore())
    event = event_for(demo_clay_signal(received_at=T0))  # valid_until ~ 2026-10-15
    result = asyncio.run(workflow.run(event, now=datetime(2026, 12, 1, tzinfo=UTC)))
    assert result.outcome is HotOutcome.expired


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("review", HotOutcome.research_review),
        ("abstained", HotOutcome.research_abstained),
        ("deferred", HotOutcome.deferred),
        ("failed", HotOutcome.failed),
    ],
)
def test_workflow_maps_research_outcomes(status: str, expected: HotOutcome) -> None:
    companies = InMemoryCompanyRepository([demo_company(observed_at=T0)])
    outcome = ResearchOutcome(status=status, dossier=None, agent_session_id=None)  # type: ignore[arg-type]
    workflow = HotWorkflow(
        companies=companies,
        research=FakeResearchRunner(outcome),
        tasks=InMemoryHotTaskStore(),
        evidence=SignalEvidenceBuilder(),
    )
    result = asyncio.run(workflow.run(event_for(demo_clay_signal(received_at=T0)), now=NOW))
    assert result.outcome is expected


# --- SignalEvidenceBuilder ------------------------------------------------


def test_evidence_builder_produces_signal_tagged_input() -> None:
    research_input = SignalEvidenceBuilder().build(
        demo_company(observed_at=T0), demo_clay_signal(received_at=T0), as_of=T0.date()
    )
    assert len(research_input.evidence) == 1
    item = research_input.evidence[0]
    assert item.verified is True
    assert item.tags == ("signal:administrative_hiring",)
    assert item.valid_until == demo_clay_signal(received_at=T0).valid_until.date()
    assert research_input.piva == "99000000002"


def test_evidence_builder_abstains_without_citable_url() -> None:
    company = Company(
        piva="99000000002", legal_name="X", source="demo", is_demo=True, source_observed_at=T0
    )
    signal = HotSignal(
        source=SignalSource.clay,
        piva="99000000002",
        signal_type=HotSignalType.administrative_hiring,
        observed_at=T0,
        received_at=T0,
        valid_until=datetime(2026, 8, 16, tzinfo=UTC),
        source_url=None,
        payload={},
    )
    with pytest.raises(EvidenceUnavailableError):
        SignalEvidenceBuilder().build(company, signal, as_of=T0.date())


# --- HotEventWorker settle routing ---------------------------------------


def _worker(outbox: InMemoryOutbox, runner: FakeWorkflowRunner, **kwargs: object) -> HotEventWorker:
    return HotEventWorker(outbox=outbox, runner=runner, clock=lambda: T0, **kwargs)  # type: ignore[arg-type]


def test_worker_settles_done_on_task_created() -> None:
    outbox = InMemoryOutbox()
    signal = demo_clay_signal(received_at=T0)
    outbox.append(signal, now=T0)
    worker = _worker(outbox, FakeWorkflowRunner(outcome=HotOutcome.task_created))
    asyncio.run(worker.run_once())
    assert outbox.status_of(signal.natural_key_hash) is OutboxStatus.done


def test_worker_reschedules_on_deferred() -> None:
    outbox = InMemoryOutbox()
    signal = demo_clay_signal(received_at=T0)
    outbox.append(signal, now=T0)
    worker = _worker(outbox, FakeWorkflowRunner(outcome=HotOutcome.deferred))
    asyncio.run(worker.run_once())
    assert outbox.status_of(signal.natural_key_hash) is OutboxStatus.pending


def test_worker_dead_letters_unknown_company() -> None:
    outbox = InMemoryOutbox()
    signal = demo_clay_signal(received_at=T0)
    outbox.append(signal, now=T0)
    worker = _worker(outbox, FakeWorkflowRunner(outcome=HotOutcome.unknown_company))
    asyncio.run(worker.run_once())
    assert outbox.status_of(signal.natural_key_hash) is OutboxStatus.dead


def test_worker_retries_then_dead_letters() -> None:
    outbox = InMemoryOutbox()
    signal = demo_clay_signal(received_at=T0)
    outbox.append(signal, now=T0)
    worker = HotEventWorker(
        outbox=outbox,
        runner=FakeWorkflowRunner(outcome=HotOutcome.failed),
        clock=lambda: T0,
        max_attempts=2,
        backoff=lambda attempts: timedelta(0),
    )
    asyncio.run(worker.run_once())  # attempt 1 -> retry
    assert outbox.status_of(signal.natural_key_hash) is OutboxStatus.pending
    asyncio.run(worker.run_once())  # attempt 2 -> dead
    assert outbox.status_of(signal.natural_key_hash) is OutboxStatus.dead


def test_worker_treats_exception_as_failure() -> None:
    outbox = InMemoryOutbox()
    signal = demo_clay_signal(received_at=T0)
    outbox.append(signal, now=T0)
    worker = HotEventWorker(
        outbox=outbox,
        runner=FakeWorkflowRunner(error=RuntimeError("boom")),
        clock=lambda: T0,
        max_attempts=1,
    )
    result = asyncio.run(worker.run_once())
    assert result is not None
    assert result.outcome is HotOutcome.failed
    assert outbox.status_of(signal.natural_key_hash) is OutboxStatus.dead


def test_worker_idle_returns_none() -> None:
    worker = _worker(InMemoryOutbox(), FakeWorkflowRunner(outcome=HotOutcome.task_created))
    assert asyncio.run(worker.run_once()) is None
