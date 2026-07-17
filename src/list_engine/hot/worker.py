"""The signal-to-action core: assemble evidence, run research, create a task.

``HotWorkflow`` is the engine-agnostic domain unit (it implements
``WorkflowRunner``); ``HotEventWorker`` is the Postgres-poller shell that a
durable-execution engine replaces at M8. Research and task persistence sit
behind ports so the DEMO path (``DemoResearchAgent`` + in-memory stores) and the
production path (``ColdResearchBatch`` + Postgres) share one workflow.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Literal, Protocol
from uuid import UUID, uuid4

from pydantic import ValidationError

from list_engine.core.models import Company
from list_engine.core.repository import CompanyRepository
from list_engine.hot.models import (
    HotOutcome,
    HotSignal,
    HotTask,
    OutboxEvent,
    SignalSource,
)
from list_engine.hot.outbox import LeasedEvent, Outbox
from list_engine.ingestion.quality import canonical_payload_hash
from list_engine.research.agent import ResearchAbstentionError, ResearchAgent
from list_engine.research.batch import BatchItemStatus, ColdResearchBatch
from list_engine.research.gate import CitationGate
from list_engine.research.models import ApprovedDossier, EvidenceItem, ResearchInput, SourceKind

_SOURCE_KIND: dict[SignalSource, SourceKind] = {
    SignalSource.company_monitoring: "official_registry",
    SignalSource.hiring: "job_board",
    SignalSource.clay: "crm",
}

ResearchStatus = Literal["approved", "review", "abstained", "deferred", "failed"]


class EvidenceUnavailableError(RuntimeError):
    """A signal cannot be turned into a citable evidence packet (cita-o-astieniti)."""


@dataclass(frozen=True, slots=True)
class ResearchOutcome:
    status: ResearchStatus
    dossier: ApprovedDossier | None
    agent_session_id: UUID | None


@dataclass(frozen=True, slots=True)
class HotWorkflowResult:
    piva: str
    outcome: HotOutcome
    task: HotTask | None
    latency_seconds: Decimal | None


class ResearchRunner(Protocol):
    async def research(
        self, research_input: ResearchInput, *, now: datetime
    ) -> ResearchOutcome: ...


class HotTaskStore(Protocol):
    def create_task(
        self,
        *,
        piva: str,
        signal: HotSignal,
        agent_session_id: UUID | None,
        dossier_digest: str,
        now: datetime,
    ) -> tuple[HotTask, bool]: ...


class WorkflowRunner(Protocol):
    async def run(self, event: OutboxEvent, *, now: datetime) -> HotWorkflowResult: ...


class DirectResearchRunner:
    """Offline research: DemoResearchAgent + CitationGate, no cache/lease/store."""

    def __init__(self, agent: ResearchAgent, gate: CitationGate) -> None:
        self._agent = agent
        self._gate = gate

    async def research(self, research_input: ResearchInput, *, now: datetime) -> ResearchOutcome:
        try:
            generation = await self._agent.generate(research_input)
        except ResearchAbstentionError:
            return ResearchOutcome(status="abstained", dossier=None, agent_session_id=None)
        evaluation = self._gate.evaluate(research_input, generation)
        if evaluation.passed:
            dossier = ApprovedDossier(generation=generation, evaluation=evaluation)
            return ResearchOutcome(status="approved", dossier=dossier, agent_session_id=None)
        return ResearchOutcome(status="review", dossier=None, agent_session_id=None)


class BatchResearchRunner:
    """Production research: the cache-first, fenced-lease ColdResearchBatch."""

    def __init__(self, batch: ColdResearchBatch) -> None:
        self._batch = batch

    async def research(self, research_input: ResearchInput, *, now: datetime) -> ResearchOutcome:
        result = await self._batch.run((research_input,))
        item = result.items[0]
        if item.status in (BatchItemStatus.CACHE_HIT, BatchItemStatus.GENERATED):
            return ResearchOutcome(status="approved", dossier=item.dossier, agent_session_id=None)
        if item.status is BatchItemStatus.REVIEW:
            return ResearchOutcome(status="review", dossier=None, agent_session_id=None)
        if item.status is BatchItemStatus.DEFERRED:
            return ResearchOutcome(status="deferred", dossier=None, agent_session_id=None)
        return ResearchOutcome(status="failed", dossier=None, agent_session_id=None)


class SignalEvidenceBuilder:
    """Pure: turn a Company + HotSignal into a single-evidence ResearchInput."""

    def build(self, company: Company, signal: HotSignal, *, as_of: date) -> ResearchInput:
        source_url = self._citable_url(company, signal)
        if source_url is None:
            raise EvidenceUnavailableError("hot signal has no citable HTTP(S) source URL")
        excerpt = (
            f"Segnale {signal.signal_type.value} rilevato da {signal.source.value} "
            f"per {company.legal_name} in data {signal.observed_at.date().isoformat()}."
        )
        try:
            evidence = EvidenceItem(
                evidence_id="hot_signal",
                source_url=source_url,
                source_kind=_SOURCE_KIND[signal.source],
                title=f"Segnale {signal.signal_type.value}",
                excerpt=excerpt,
                observed_at=signal.observed_at,
                verified=True,
                tags=(f"signal:{signal.signal_type.value}",),
                valid_until=signal.valid_until.date(),
            )
            return ResearchInput(
                piva=signal.piva,
                as_of=as_of,
                website=None,
                hub_data=self._hub_data(company, signal),
                evidence=(evidence,),
            )
        except ValidationError as error:
            raise EvidenceUnavailableError(f"cannot build citable evidence: {error}") from error

    @staticmethod
    def _citable_url(company: Company, signal: HotSignal) -> str | None:
        if signal.source_url is not None:
            return signal.source_url
        website = company.website
        if website is not None and website.startswith("https://"):
            return website
        return None

    @staticmethod
    def _hub_data(company: Company, signal: HotSignal) -> dict[str, object]:
        data: dict[str, object] = {
            "legal_name": company.legal_name,
            "signal_type": signal.signal_type.value,
            "signal_source": signal.source.value,
        }
        for key, value in (
            ("ateco_code", company.ateco_code),
            ("region", company.region),
            ("city", company.city),
            ("website", company.website),
        ):
            if value is not None:
                data[key] = value
        if company.employees is not None:
            data["employees"] = company.employees
        return data


class HotWorkflow:
    """Engine-agnostic signal -> dossier -> task step function."""

    def __init__(
        self,
        *,
        companies: CompanyRepository,
        research: ResearchRunner,
        tasks: HotTaskStore,
        evidence: SignalEvidenceBuilder,
    ) -> None:
        self._companies = companies
        self._research = research
        self._tasks = tasks
        self._evidence = evidence

    async def run(self, event: OutboxEvent, *, now: datetime) -> HotWorkflowResult:
        signal = event.signal
        company = self._companies.get(signal.piva)
        if company is None:
            # A fresh webhook must not fabricate a golden record; hand to review.
            return HotWorkflowResult(signal.piva, HotOutcome.unknown_company, None, None)
        if signal.is_expired(at=now):
            return HotWorkflowResult(signal.piva, HotOutcome.expired, None, None)
        try:
            research_input = self._evidence.build(company, signal, as_of=now.date())
        except EvidenceUnavailableError:
            return HotWorkflowResult(signal.piva, HotOutcome.research_abstained, None, None)

        outcome = await self._research.research(research_input, now=now)
        if outcome.status == "approved" and outcome.dossier is not None:
            digest = canonical_payload_hash(outcome.dossier.model_dump(mode="json"))
            task, created = self._tasks.create_task(
                piva=signal.piva,
                signal=signal,
                agent_session_id=outcome.agent_session_id,
                dossier_digest=digest,
                now=now,
            )
            status = HotOutcome.task_created if created else HotOutcome.task_exists
            return HotWorkflowResult(signal.piva, status, task, task.signal_to_task_seconds)

        outcome_map: dict[ResearchStatus, HotOutcome] = {
            "approved": HotOutcome.failed,  # unreachable: approved without a dossier
            "review": HotOutcome.research_review,
            "abstained": HotOutcome.research_abstained,
            "deferred": HotOutcome.deferred,
            "failed": HotOutcome.failed,
        }
        return HotWorkflowResult(signal.piva, outcome_map[outcome.status], None, None)


class InMemoryHotTaskStore:
    """Deterministic DEMO task store; idempotent on the signal natural key."""

    def __init__(self) -> None:
        self._tasks: dict[str, HotTask] = {}

    def create_task(
        self,
        *,
        piva: str,
        signal: HotSignal,
        agent_session_id: UUID | None,
        dossier_digest: str,
        now: datetime,
    ) -> tuple[HotTask, bool]:
        key = signal.natural_key_hash
        existing = self._tasks.get(key)
        if existing is not None:
            return existing, False
        task = HotTask(
            id=uuid4(),
            piva=piva,
            signal_natural_key=key,
            source=signal.source,
            signal_type=signal.signal_type,
            agent_session_id=agent_session_id,
            dossier_digest=dossier_digest,
            signal_observed_at=signal.observed_at,
            signal_received_at=signal.received_at,
            task_created_at=now,
        )
        self._tasks[key] = task
        return task, True

    def all_tasks(self) -> tuple[HotTask, ...]:
        return tuple(self._tasks.values())


def default_backoff(attempts: int) -> timedelta:
    return timedelta(seconds=min(300, 2 ** max(0, attempts)))


class HotEventWorker:
    """Poll the outbox, run the workflow, and settle the lease (fenced)."""

    def __init__(
        self,
        *,
        outbox: Outbox,
        runner: WorkflowRunner,
        clock: Callable[[], datetime],
        lease_ttl: timedelta = timedelta(minutes=5),
        max_attempts: int = 8,
        backoff: Callable[[int], timedelta] = default_backoff,
    ) -> None:
        self._outbox = outbox
        self._runner = runner
        self._clock = clock
        self._lease_ttl = lease_ttl
        self._max_attempts = max_attempts
        self._backoff = backoff

    async def run_once(self) -> HotWorkflowResult | None:
        now = self._clock()
        lease = self._outbox.claim_next(owner_token=uuid4(), lease_ttl=self._lease_ttl, now=now)
        if lease is None:
            return None
        try:
            result = await self._runner.run(lease.event, now=now)
        except Exception as error:
            self._settle_failure(lease, error_type=type(error).__name__, now=now)
            return HotWorkflowResult(lease.event.piva, HotOutcome.failed, None, None)
        self._settle(lease, result, now=now)
        return result

    async def run_forever(self, *, poll_interval: timedelta, stop: asyncio.Event) -> None:
        while not stop.is_set():
            result = await self.run_once()
            if result is None:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=poll_interval.total_seconds())

    def _settle(self, lease: LeasedEvent, result: HotWorkflowResult, *, now: datetime) -> None:
        if result.outcome in (HotOutcome.task_created, HotOutcome.task_exists, HotOutcome.expired):
            self._outbox.settle_done(lease, now=now)
        elif result.outcome is HotOutcome.deferred:
            self._outbox.settle_retry(
                lease, backoff=self._backoff(lease.event.attempts), now=now, error_type="deferred"
            )
        elif result.outcome is HotOutcome.failed:
            self._settle_failure(lease, error_type="failed", now=now)
        else:
            # unknown_company / research_review / research_abstained: nothing to retry.
            self._outbox.settle_dead(
                lease, now=now, error_type=result.outcome.value, retryable=False
            )

    def _settle_failure(self, lease: LeasedEvent, *, error_type: str, now: datetime) -> None:
        if lease.event.attempts < self._max_attempts:
            self._outbox.settle_retry(
                lease, backoff=self._backoff(lease.event.attempts), now=now, error_type=error_type
            )
        else:
            self._outbox.settle_dead(lease, now=now, error_type=error_type, retryable=True)


__all__ = [
    "BatchResearchRunner",
    "DirectResearchRunner",
    "EvidenceUnavailableError",
    "HotEventWorker",
    "HotTaskStore",
    "HotWorkflow",
    "HotWorkflowResult",
    "InMemoryHotTaskStore",
    "ResearchOutcome",
    "ResearchRunner",
    "SignalEvidenceBuilder",
    "WorkflowRunner",
    "default_backoff",
]
