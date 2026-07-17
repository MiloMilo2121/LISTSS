"""Offline composition of the whole hot path: no network, no DB, no FastAPI.

Wires an in-memory hub, outbox, worker and the deterministic ``DemoResearchAgent``
so the signal -> event -> dossier -> task flow (and its signal->task latency) can
be exercised and asserted with an injected clock.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from list_engine.core.models import Company
from list_engine.core.repository import InMemoryCompanyRepository
from list_engine.hot.adapters import normalize_clay
from list_engine.hot.models import HotSignal
from list_engine.hot.outbox import InMemoryOutbox, SignalIngestService
from list_engine.hot.worker import (
    DirectResearchRunner,
    HotEventWorker,
    HotWorkflow,
    InMemoryHotTaskStore,
    SignalEvidenceBuilder,
)
from list_engine.research.agent import DemoResearchAgent
from list_engine.research.gate import CitationGate

DEMO_PIVA = "99000000002"


def demo_company(*, observed_at: datetime) -> Company:
    return Company(
        piva=DEMO_PIVA,
        legal_name="Azienda Dimostrativa S.r.l.",
        website="https://azienda.demo.invalid",
        ateco_code="41.20.00",
        city="Milano",
        province="MI",
        region="Lombardia",
        revenue_eur=Decimal("4500000"),
        employees=42,
        company_status="active",
        source="demo",
        is_demo=True,
        source_observed_at=observed_at,
    )


def demo_clay_signal(*, received_at: datetime) -> HotSignal:
    return normalize_clay(
        {
            "piva": DEMO_PIVA,
            "signal_type": "administrative_hiring",
            "observed_at": received_at.isoformat(),
            "source_url": "https://jobs.demo.invalid/amministrazione",
            "role": "Responsabile Amministrativo",
        },
        received_at=received_at,
    )


@dataclass(frozen=True, slots=True)
class DemoHotPipeline:
    companies: InMemoryCompanyRepository
    outbox: InMemoryOutbox
    ingest: SignalIngestService
    worker: HotEventWorker
    tasks: InMemoryHotTaskStore


def build_demo_pipeline(
    *, clock: Callable[[], datetime], seed_observed_at: datetime
) -> DemoHotPipeline:
    companies = InMemoryCompanyRepository([demo_company(observed_at=seed_observed_at)])
    outbox = InMemoryOutbox()
    tasks = InMemoryHotTaskStore()
    workflow = HotWorkflow(
        companies=companies,
        research=DirectResearchRunner(DemoResearchAgent(), CitationGate()),
        tasks=tasks,
        evidence=SignalEvidenceBuilder(),
    )
    worker = HotEventWorker(outbox=outbox, runner=workflow, clock=clock)
    return DemoHotPipeline(
        companies=companies,
        outbox=outbox,
        ingest=SignalIngestService(outbox),
        worker=worker,
        tasks=tasks,
    )


__all__ = [
    "DEMO_PIVA",
    "DemoHotPipeline",
    "build_demo_pipeline",
    "demo_clay_signal",
    "demo_company",
]
