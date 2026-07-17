"""Event-driven hot-signal pipeline: signal -> event -> dossier + task in minutes.

The Postgres store (``hot.postgres``) and the FastAPI ingress (``hot.ingress``,
needs the optional ``ingress`` extra) are imported by their full path and are
deliberately not re-exported here, so importing ``list_engine.hot`` never pulls
in a database connection or a web framework.
"""

from __future__ import annotations

from list_engine.hot.adapters import (
    HmacSignatureVerifier,
    SharedTokenVerifier,
    SignalNormalizationError,
    SignatureError,
    SignatureVerifier,
    normalize_clay,
    normalize_company_monitoring,
    normalize_hiring,
)
from list_engine.hot.models import (
    HotOutcome,
    HotSignal,
    HotSignalType,
    HotTask,
    HotTaskStatus,
    OutboxEvent,
    OutboxStatus,
    SignalSource,
)
from list_engine.hot.outbox import (
    InMemoryOutbox,
    LeasedEvent,
    Outbox,
    OutboxAppendResult,
    SignalIngestService,
)
from list_engine.hot.worker import (
    BatchResearchRunner,
    DirectResearchRunner,
    EvidenceUnavailableError,
    HotEventWorker,
    HotTaskStore,
    HotWorkflow,
    HotWorkflowResult,
    InMemoryHotTaskStore,
    ResearchOutcome,
    ResearchRunner,
    SignalEvidenceBuilder,
    WorkflowRunner,
    default_backoff,
)

__all__ = [
    "BatchResearchRunner",
    "DirectResearchRunner",
    "EvidenceUnavailableError",
    "HmacSignatureVerifier",
    "HotEventWorker",
    "HotOutcome",
    "HotSignal",
    "HotSignalType",
    "HotTask",
    "HotTaskStatus",
    "HotTaskStore",
    "HotWorkflow",
    "HotWorkflowResult",
    "InMemoryHotTaskStore",
    "InMemoryOutbox",
    "LeasedEvent",
    "Outbox",
    "OutboxAppendResult",
    "OutboxEvent",
    "OutboxStatus",
    "ResearchOutcome",
    "ResearchRunner",
    "SharedTokenVerifier",
    "SignalEvidenceBuilder",
    "SignalIngestService",
    "SignalNormalizationError",
    "SignalSource",
    "SignatureError",
    "SignatureVerifier",
    "WorkflowRunner",
    "default_backoff",
    "normalize_clay",
    "normalize_company_monitoring",
    "normalize_hiring",
]
