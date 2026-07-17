from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime

import pytest

from list_engine.research.agent import DemoResearchAgent
from list_engine.research.gate import CitationGate, DossierRejectedError
from list_engine.research.models import (
    INFERENCE_RATIONALE,
    AgentGeneration,
    CitedClaim,
    CitedInference,
    DossierEvaluation,
    EvidenceItem,
    ResearchInput,
)

PIVA = "99000000002"
OTHER_PIVA = "99000000010"
AS_OF = date(2026, 7, 15)


def evidence(
    *,
    source_url: str = "https://azienda.invalid/news",
    verified: bool = True,
    tags: tuple[str, ...] = ("signal:new_office", "persona:direttore finanziario"),
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id="company_news",
        source_url=source_url,
        source_kind="company_website",
        title="Nuova sede",
        excerpt="L'azienda ha aperto una nuova sede operativa a Milano nel 2026.",
        observed_at=datetime(2026, 7, 15, tzinfo=UTC),
        verified=verified,
        tags=tags,
        valid_until=AS_OF if any(tag.startswith("signal:") for tag in tags) else None,
    )


def packet(*, item: EvidenceItem | None = None) -> ResearchInput:
    return ResearchInput(piva=PIVA, as_of=AS_OF, evidence=(item or evidence(),))


def generation(research_input: ResearchInput | None = None) -> AgentGeneration:
    return asyncio.run(DemoResearchAgent().generate(research_input or packet()))


def with_first_fact(
    original: AgentGeneration,
    **updates: object,
) -> AgentGeneration:
    claim = original.dossier.facts[0].model_copy(update=updates)
    dossier = original.dossier.model_copy(update={"facts": (claim,)})
    return original.model_copy(update={"dossier": dossier})


def issue_codes(evaluation: DossierEvaluation) -> set[str]:
    return {issue.code for issue in evaluation.issues}


def test_gate_approves_a_fully_supported_demo_dossier() -> None:
    research_input = packet()
    result = generation(research_input)

    evaluation = CitationGate().evaluate(research_input, result)
    approved = CitationGate().approve(research_input, result)

    assert evaluation.passed
    assert evaluation.issues == ()
    assert evaluation.factual_support == 1.0
    assert evaluation.citation_accuracy == 1.0
    assert evaluation.source_quality == 0.9
    assert evaluation.hallucination_free == 1.0
    assert approved.generation == result
    assert approved.evaluation.passed


def test_gate_rejects_a_fabricated_url_even_when_evidence_id_exists() -> None:
    research_input = packet()
    tampered = with_first_fact(
        generation(research_input),
        source_url="https://attacker.invalid/fabricated",
    )

    evaluation = CitationGate().evaluate(research_input, tampered)

    assert not evaluation.passed
    assert "citation_url_mismatch" in issue_codes(evaluation)


def test_gate_rejects_a_fabricated_supporting_quote() -> None:
    research_input = packet()
    tampered = with_first_fact(
        generation(research_input),
        claim="L'azienda ha acquisito un concorrente.",
        supporting_excerpt="L'azienda ha acquisito un concorrente.",
    )

    evaluation = CitationGate().evaluate(research_input, tampered)

    assert "excerpt_not_found" in issue_codes(evaluation)


def test_gate_rejects_a_claim_not_lexically_supported_by_a_real_quote() -> None:
    research_input = packet()
    tampered = with_first_fact(
        generation(research_input),
        claim="Il fatturato certificato supera cento milioni di euro.",
    )

    evaluation = CitationGate().evaluate(research_input, tampered)

    assert "claim_not_supported" in issue_codes(evaluation)
    assert evaluation.factual_support < 0.6


def test_gate_rejects_a_negated_paraphrase_despite_high_lexical_overlap() -> None:
    research_input = packet()
    tampered = with_first_fact(
        generation(research_input),
        claim="L'azienda non ha aperto una nuova sede operativa a Milano nel 2026.",
    )

    evaluation = CitationGate().evaluate(research_input, tampered)

    assert "claim_not_extractive" in issue_codes(evaluation)
    assert not evaluation.passed


def test_gate_rejects_an_invented_hook_with_a_real_evidence_id() -> None:
    research_input = packet()
    original = generation(research_input)
    dossier = original.dossier.model_copy(
        update={"opening_hook": "Aprire dalla fonte verificata: Acquisizione completata."}
    )

    evaluation = CitationGate().evaluate(
        research_input, original.model_copy(update={"dossier": dossier})
    )

    assert "hook_not_extractive" in issue_codes(evaluation)


def test_gate_rejects_a_dossier_that_omits_supported_facts() -> None:
    research_input = packet()
    original = generation(research_input)
    dossier = original.dossier.model_copy(update={"facts": ()})

    evaluation = CitationGate().evaluate(
        research_input, original.model_copy(update={"dossier": dossier})
    )

    assert "facts_incomplete" in issue_codes(evaluation)
    assert evaluation.completeness < 1.0
    assert not evaluation.passed


def test_gate_rejects_omitted_tagged_signals_and_inferences() -> None:
    research_input = packet(
        item=evidence(
            tags=(
                "signal:new_office",
                "persona:direttore finanziario",
                "objection:processo gia coperto",
            )
        )
    )
    original = generation(research_input)
    dossier = original.dossier.model_copy(
        update={
            "active_signals": (),
            "probable_persona": None,
            "likely_objection": None,
        }
    )

    evaluation = CitationGate().evaluate(
        research_input, original.model_copy(update={"dossier": dossier})
    )

    assert {
        "signals_incomplete",
        "persona_incomplete",
        "objection_incomplete",
    } <= issue_codes(evaluation)
    assert evaluation.completeness < 1.0
    assert not evaluation.passed


def test_gate_requires_explicit_exact_tags_for_persona_and_objection() -> None:
    research_input = packet()
    original = generation(research_input)
    assert original.dossier.probable_persona is not None
    invented_persona = original.dossier.probable_persona.model_copy(
        update={"text": "Persona probabile: amministratore delegato"}
    )
    unsupported_objection = CitedInference(
        text="Obiezione probabile: processo gia coperto",
        evidence_ids=("company_news",),
        rationale=INFERENCE_RATIONALE,
    )
    dossier = original.dossier.model_copy(
        update={
            "probable_persona": invented_persona,
            "likely_objection": unsupported_objection,
        }
    )

    evaluation = CitationGate().evaluate(
        research_input, original.model_copy(update={"dossier": dossier})
    )

    assert {"inference_tag_not_supported", "objection_tag_missing"} <= issue_codes(evaluation)


def test_gate_rejects_unverified_evidence_for_claim_persona_and_hook() -> None:
    verified_packet = packet()
    result = generation(verified_packet)
    unverified_packet = packet(item=evidence(verified=False))

    evaluation = CitationGate().evaluate(unverified_packet, result)

    assert not evaluation.passed
    assert {
        "evidence_unverified",
        "inference_evidence_unverified",
        "hook_evidence_unverified",
    } <= issue_codes(evaluation)
    assert evaluation.hallucination_free == 0.0


def test_gate_rejects_wrong_dossier_piva() -> None:
    research_input = packet()
    result = generation(research_input)
    wrong_dossier = result.dossier.model_copy(update={"piva": OTHER_PIVA})
    tampered = result.model_copy(update={"dossier": wrong_dossier})

    evaluation = CitationGate().evaluate(research_input, tampered)

    assert issue_codes(evaluation) >= {"piva_mismatch"}


def test_gate_rejects_missing_claim_inference_and_hook_evidence_ids() -> None:
    research_input = packet()
    result = generation(research_input)
    missing_claim = CitedClaim(
        claim="Affermazione inventata",
        evidence_id="fabricated_source",
        source_url="https://fabricated.invalid/source",
        supporting_excerpt="Affermazione inventata",
        confidence=1.0,
    )
    assert result.dossier.probable_persona is not None
    missing_persona = result.dossier.probable_persona.model_copy(
        update={"evidence_ids": ("missing_persona",)}
    )
    dossier = result.dossier.model_copy(
        update={
            "facts": (missing_claim,),
            "probable_persona": missing_persona,
            "opening_hook_evidence_ids": ("missing_hook",),
        }
    )
    tampered = result.model_copy(update={"dossier": dossier})

    evaluation = CitationGate().evaluate(research_input, tampered)

    assert {
        "citation_evidence_missing",
        "inference_evidence_missing",
        "hook_evidence_missing",
    } <= issue_codes(evaluation)


def test_active_signal_must_cite_evidence_with_signal_tag() -> None:
    tagged_packet = packet()
    result = generation(tagged_packet)
    untagged_packet = packet(item=evidence(tags=("persona:direttore finanziario",)))

    evaluation = CitationGate().evaluate(untagged_packet, result)

    assert "signal_tag_missing" in issue_codes(evaluation)


def test_normalized_quote_matching_tolerates_case_and_whitespace_only() -> None:
    research_input = packet()
    result = generation(research_input)
    normalized = with_first_fact(
        result,
        supporting_excerpt="l'AZIENDA   ha aperto una nuova sede operativa a Milano nel 2026.",
    )

    evaluation = CitationGate().evaluate(research_input, normalized)

    assert evaluation.passed


def test_plain_http_evidence_fails_the_source_quality_floor() -> None:
    http_packet = packet(item=evidence(source_url="http://azienda.invalid/news"))
    result = generation(http_packet)

    evaluation = CitationGate().evaluate(http_packet, result)

    assert not evaluation.passed
    assert "low_source_quality" in issue_codes(evaluation)
    assert evaluation.source_quality == 0.45


def test_approve_is_blocking_and_exposes_the_evaluation() -> None:
    research_input = packet()
    tampered = with_first_fact(
        generation(research_input),
        source_url="https://attacker.invalid/fabricated",
    )

    with pytest.raises(DossierRejectedError) as captured:
        CitationGate().approve(research_input, tampered)

    assert not captured.value.evaluation.passed
    assert "citation_url_mismatch" in issue_codes(captured.value.evaluation)


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.1])
def test_gate_rejects_invalid_threshold_configuration(threshold: float) -> None:
    with pytest.raises(ValueError):
        CitationGate(lexical_support_threshold=threshold)
