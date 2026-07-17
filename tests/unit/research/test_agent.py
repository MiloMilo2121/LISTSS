from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from list_engine.research.agent import (
    DEMO_MODEL,
    DEMO_PROVIDER,
    DemoResearchAgent,
    ResearchAbstentionError,
    ResearchAgent,
)
from list_engine.research.models import AgentGeneration, EvidenceItem, ResearchInput

PIVA = "99000000002"
AS_OF = date(2026, 7, 15)


def evidence(
    evidence_id: str,
    excerpt: str,
    *,
    tags: tuple[str, ...] = (),
    verified: bool = True,
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        source_url=f"https://demo.invalid/{evidence_id}",
        source_kind="company_website",
        title=evidence_id.replace("_", " ").title(),
        excerpt=excerpt,
        observed_at=datetime(2026, 7, 15, tzinfo=UTC),
        verified=verified,
        tags=tags,
        valid_until=AS_OF if any(tag.startswith("signal:") for tag in tags) else None,
    )


def run_agent(research_input: ResearchInput) -> AgentGeneration:
    return asyncio.run(DemoResearchAgent().generate(research_input))


def test_demo_agent_satisfies_async_port_and_has_no_external_cost() -> None:
    agent = DemoResearchAgent()
    assert isinstance(agent, ResearchAgent)

    research_input = ResearchInput(
        piva=PIVA,
        as_of=AS_OF,
        evidence=(evidence("company_site", "La societa opera a Milano."),),
    )
    generation = asyncio.run(agent.generate(research_input))

    assert generation.provider == DEMO_PROVIDER
    assert generation.model == DEMO_MODEL
    assert generation.cost_usd == Decimal("0")
    assert generation.input_tokens is None
    assert generation.output_tokens is None


def test_demo_agent_uses_only_verified_evidence_and_caps_facts_at_three() -> None:
    research_input = ResearchInput(
        piva=PIVA,
        as_of=AS_OF,
        evidence=(
            evidence(
                "unverified_news",
                "La societa avrebbe aperto una sede a Roma.",
                tags=("signal:new_office", "hook:espansione"),
                verified=False,
            ),
            evidence("registry", "La societa e attiva dal 2018."),
            evidence(
                "jobs",
                "Sono aperte cinque posizioni amministrative.",
                tags=("signal:hiring", "persona:direttore amministrativo"),
            ),
            evidence(
                "website",
                "Il sito descrive servizi per imprese italiane.",
                tags=("hook:servizi per imprese",),
            ),
            evidence("directory", "La sede pubblicata e a Torino."),
        ),
    )

    dossier = run_agent(research_input).dossier

    assert [claim.evidence_id for claim in dossier.facts] == ["registry", "jobs", "website"]
    assert [claim.evidence_id for claim in dossier.active_signals] == ["jobs"]
    assert dossier.probable_persona is not None
    assert dossier.probable_persona.evidence_ids == ("jobs",)
    assert "direttore amministrativo" in dossier.probable_persona.text
    assert dossier.opening_hook_evidence_ids == ("website",)
    assert "Il sito descrive servizi per imprese italiane." in dossier.opening_hook
    assert "unverified_news" not in dossier.model_dump_json()


def test_demo_generation_is_reproducible_for_the_same_bounded_input() -> None:
    research_input = ResearchInput(
        piva=PIVA,
        as_of=AS_OF,
        hub_data={"tier": "T1"},
        evidence=(
            evidence(
                "signal_one",
                "  L'azienda   ricerca un responsabile finanza.  ",
                tags=("signal:hiring",),
            ),
        ),
    )

    first = run_agent(research_input)
    second = run_agent(research_input)

    assert first == second
    assert first.session_id.startswith("demo:")
    assert len(first.session_id) == len("demo:") + 64
    assert first.dossier.facts[0].claim == "L'azienda ricerca un responsabile finanza."
    assert first.dossier.active_signals[0].supporting_excerpt == (
        "L'azienda ricerca un responsabile finanza."
    )


def test_demo_session_identity_is_independent_of_json_object_key_order() -> None:
    item = evidence("registry", "La societa risulta attiva.")
    first_input = ResearchInput(
        piva=PIVA,
        as_of=AS_OF,
        hub_data={"tier": "T1", "score": 82},
        evidence=(item,),
    )
    reordered_input = ResearchInput(
        piva=PIVA,
        as_of=AS_OF,
        hub_data={"score": 82, "tier": "T1"},
        evidence=(item,),
    )

    assert run_agent(first_input).session_id == run_agent(reordered_input).session_id


def test_demo_hook_falls_back_to_a_verified_signal_then_first_evidence() -> None:
    signal_input = ResearchInput(
        piva=PIVA,
        as_of=AS_OF,
        evidence=(
            evidence("registry", "Impresa attiva."),
            evidence("signal", "Nuova ricerca personale.", tags=("signal:hiring",)),
        ),
    )
    plain_input = ResearchInput(
        piva=PIVA,
        as_of=AS_OF,
        evidence=(evidence("registry", "Impresa attiva."),),
    )

    assert run_agent(signal_input).dossier.opening_hook_evidence_ids == ("signal",)
    assert run_agent(plain_input).dossier.opening_hook_evidence_ids == ("registry",)


def test_demo_agent_abstains_when_no_usable_verified_evidence_exists() -> None:
    research_input = ResearchInput(
        piva=PIVA,
        as_of=AS_OF,
        evidence=(evidence("rumour", "Voce non confermata.", verified=False),),
    )

    with pytest.raises(ResearchAbstentionError, match="CITA-O-ASTIENITI"):
        run_agent(research_input)
