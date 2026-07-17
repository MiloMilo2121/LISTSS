"""Research-agent boundary and the deterministic DEMO implementation.

The production SDK adapter is deliberately kept outside this module.  Both the
production and DEMO implementations must satisfy :class:`ResearchAgent`, while
the citation gate remains the only path to a downstream-approved dossier.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Protocol, runtime_checkable

from list_engine.ingestion.quality import canonical_payload_hash
from list_engine.research.models import (
    INFERENCE_RATIONALE,
    AgentGeneration,
    CitedClaim,
    CitedInference,
    EvidenceItem,
    ResearchDossier,
    ResearchInput,
)

DEMO_PROVIDER = "list-engine-demo"
DEMO_MODEL = "deterministic-evidence-v1"
DEMO_PROMPT_VERSION = "demo-research-v1"
OPENING_HOOK_PREFIX = "Aprire dalla fonte verificata: "

_SOURCE_CONFIDENCE = {
    "official_registry": 0.99,
    "company_website": 0.95,
    "job_board": 0.85,
    "crm": 0.80,
    "public_directory": 0.75,
}


class ResearchAbstentionError(RuntimeError):
    """Raised when CITA-O-ASTIENITI leaves no verified evidence to cite."""


@runtime_checkable
class ResearchAgent(Protocol):
    """Minimal async port shared by DEMO and production research agents."""

    async def generate(self, research_input: ResearchInput) -> AgentGeneration:
        """Create an auditable but still-untrusted dossier generation."""
        ...


def _normalise_space(value: str) -> str:
    return " ".join(value.split())


def _clip(value: str, limit: int) -> str:
    """Return a normalized prefix that remains a substring of normalized evidence."""

    value = _normalise_space(value)
    if len(value) <= limit:
        return value
    prefix = value[:limit]
    if " " in prefix:
        prefix = prefix.rsplit(" ", maxsplit=1)[0]
    return prefix.rstrip(" ,;:-")


def _tag_value(item: EvidenceItem, prefix: str) -> str | None:
    for tag in item.tags:
        if tag.startswith(prefix):
            value = tag.removeprefix(prefix).strip()
            if value:
                return value
    return None


def _claim(item: EvidenceItem) -> CitedClaim:
    supporting_excerpt = _clip(item.excerpt, 500)
    return CitedClaim(
        claim=supporting_excerpt,
        evidence_id=item.evidence_id,
        source_url=item.source_url,
        supporting_excerpt=supporting_excerpt,
        confidence=_SOURCE_CONFIDENCE[item.source_kind],
    )


class DemoResearchAgent:
    """Produce a repeatable dossier from the supplied evidence packet only.

    The implementation performs no I/O, reads no environment state, and uses no
    clock or random identifier.  Unverified or blank evidence is ignored.  If
    that leaves no support for the required opening hook, it abstains instead of
    inventing copy.
    """

    async def generate(self, research_input: ResearchInput) -> AgentGeneration:
        verified = tuple(
            item
            for item in research_input.evidence
            if item.verified and _normalise_space(item.excerpt)
        )
        if not verified:
            raise ResearchAbstentionError(
                "CITA-O-ASTIENITI: no verified evidence is available for this dossier"
            )

        facts = tuple(_claim(item) for item in verified[:3])
        signal_items = tuple(item for item in verified if _tag_value(item, "signal:") is not None)
        active_signals = tuple(_claim(item) for item in signal_items)

        persona_item = next(
            (item for item in verified if _tag_value(item, "persona:") is not None),
            None,
        )
        probable_persona: CitedInference | None = None
        if persona_item is not None:
            persona = _clip(_tag_value(persona_item, "persona:") or "", 450)
            probable_persona = CitedInference(
                text=f"Persona probabile: {persona}",
                evidence_ids=(persona_item.evidence_id,),
                rationale=INFERENCE_RATIONALE,
            )

        objection_item = next(
            (item for item in verified if _tag_value(item, "objection:") is not None),
            None,
        )
        likely_objection: CitedInference | None = None
        if objection_item is not None:
            objection = _clip(_tag_value(objection_item, "objection:") or "", 450)
            likely_objection = CitedInference(
                text=f"Obiezione probabile: {objection}",
                evidence_ids=(objection_item.evidence_id,),
                rationale=INFERENCE_RATIONALE,
            )

        hook_item = next(
            (item for item in verified if _tag_value(item, "hook:") is not None),
            signal_items[0] if signal_items else verified[0],
        )
        opening_hook = f"{OPENING_HOOK_PREFIX}{_clip(hook_item.excerpt, 900)}"

        dossier = ResearchDossier(
            piva=research_input.piva,
            fatti=facts,
            segnali_attivi=active_signals,
            persona_probabile=probable_persona,
            hook_apertura=opening_hook,
            opening_hook_evidence_ids=(hook_item.evidence_id,),
            obiezione_probabile=likely_objection,
            cosa_non_dire=(
                "Non presentare ipotesi come fatti.",
                "Non citare dati assenti dalle fonti verificate.",
            ),
        )
        digest = canonical_payload_hash(research_input.model_dump(mode="json"))
        return AgentGeneration(
            dossier=dossier,
            provider=DEMO_PROVIDER,
            requested_model=DEMO_MODEL,
            model=DEMO_MODEL,
            prompt_version=DEMO_PROMPT_VERSION,
            session_id=f"demo:{digest}",
            cost_usd=Decimal("0"),
        )


__all__ = [
    "DEMO_MODEL",
    "DEMO_PROMPT_VERSION",
    "DEMO_PROVIDER",
    "OPENING_HOOK_PREFIX",
    "DemoResearchAgent",
    "ResearchAbstentionError",
    "ResearchAgent",
]
