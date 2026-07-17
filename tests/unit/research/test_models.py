from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any, cast

import pytest
from pydantic import ValidationError

from list_engine.research.models import (
    AgentGeneration,
    ApprovedDossier,
    CitedClaim,
    DossierEvaluation,
    EvidenceItem,
    ResearchDossier,
    ResearchInput,
)

AS_OF = date(2026, 7, 15)


def evidence() -> EvidenceItem:
    return EvidenceItem(
        evidence_id="company_site",
        source_url="https://demo.invalid/azienda",
        source_kind="company_website",
        title="Sito aziendale",
        excerpt="L'azienda ha aperto una nuova sede a Milano.",
        observed_at=datetime(2026, 7, 15, tzinfo=UTC),
        verified=True,
    )


def dossier() -> ResearchDossier:
    claim = CitedClaim(
        claim="L'azienda ha aperto una nuova sede a Milano.",
        evidence_id="company_site",
        source_url="https://demo.invalid/azienda",
        supporting_excerpt="L'azienda ha aperto una nuova sede a Milano.",
        confidence=1.0,
    )
    return ResearchDossier(
        piva="99000000002",
        fatti=(claim,),
        segnali_attivi=(claim,),
        persona_probabile=None,
        hook_apertura=(
            "Aprire dalla fonte verificata: L'azienda ha aperto una nuova sede a Milano."
        ),
        opening_hook_evidence_ids=("company_site",),
        obiezione_probabile=None,
        cosa_non_dire=("Non presentare ipotesi come fatti.",),
    )


def generation() -> AgentGeneration:
    return AgentGeneration(
        dossier=dossier(),
        provider="demo",
        requested_model="deterministic-v1",
        model="deterministic-v1",
        prompt_version="research-v1",
        session_id="session-1",
        cost_usd=Decimal("0"),
        input_tokens=0,
        output_tokens=0,
    )


def evaluation(*, passed: bool) -> DossierEvaluation:
    return DossierEvaluation(
        passed=passed,
        factual_support=1.0 if passed else 0.0,
        citation_accuracy=1.0 if passed else 0.0,
        completeness=1.0 if passed else 0.0,
        source_quality=1.0 if passed else 0.0,
        hallucination_free=1.0 if passed else 0.0,
        evaluator_version="citation-gate-v1",
    )


def test_research_input_normalizes_piva_and_rejects_duplicate_evidence() -> None:
    item = evidence()
    research_input = ResearchInput(piva="IT 99000000002", as_of=AS_OF, evidence=(item,))
    assert research_input.piva == "99000000002"

    with pytest.raises(ValidationError, match="evidence_id values must be unique"):
        ResearchInput(piva="99000000002", as_of=AS_OF, evidence=(item, item))


@pytest.mark.parametrize(
    "url",
    [
        "ftp://demo.invalid",
        "https://user:secret@demo.invalid",
        "https://demo.invalid/doc?access_token=secret",
        "https://demo.invalid/doc?X-Amz-Signature=secret",
        "https://demo.invalid/doc?refreshToken=secret",
        "https://demo.invalid/doc#section",
        "relative/path",
    ],
)
def test_evidence_rejects_non_public_or_credential_bearing_urls(url: str) -> None:
    with pytest.raises(ValidationError, match="source URL"):
        EvidenceItem(
            evidence_id="company_site",
            source_url=url,
            source_kind="company_website",
            title="Sito",
            excerpt="Contenuto",
            observed_at=datetime(2026, 7, 15, tzinfo=UTC),
            verified=True,
        )


def test_evidence_bounds_tags_and_requires_a_signal_window() -> None:
    payload = evidence().model_dump(mode="python")
    payload["tags"] = ("signal:new_office",)

    with pytest.raises(ValidationError, match="requires valid_until"):
        EvidenceItem.model_validate(payload)

    payload["valid_until"] = AS_OF
    payload["tags"] = tuple(f"tag:{index}" for index in range(21))
    with pytest.raises(ValidationError, match="at most 20"):
        EvidenceItem.model_validate(payload)

    payload["tags"] = ("tag:" + "x" * 129,)
    with pytest.raises(ValidationError, match="at most 128"):
        EvidenceItem.model_validate(payload)


def test_research_input_rejects_future_or_expired_signal_evidence() -> None:
    payload = evidence().model_dump(mode="python")
    payload.update(tags=("signal:new_office",), valid_until=date(2026, 7, 16))
    active_signal = EvidenceItem.model_validate(payload)

    with pytest.raises(ValidationError, match="observed after as_of"):
        ResearchInput(
            piva="99000000002",
            as_of=date(2026, 7, 14),
            evidence=(active_signal,),
        )

    with pytest.raises(ValidationError, match="expired at as_of"):
        ResearchInput(
            piva="99000000002",
            as_of=date(2026, 7, 17),
            evidence=(active_signal,),
        )


def test_dossier_caps_facts_at_three() -> None:
    claim = dossier().facts[0]
    payload = dossier().model_dump(mode="python", by_alias=True)
    payload["fatti"] = (claim, claim, claim, claim)
    with pytest.raises(ValidationError, match="at most 3"):
        ResearchDossier.model_validate(payload)


def test_only_a_passing_evaluation_can_create_an_approved_dossier() -> None:
    approved = ApprovedDossier(generation=generation(), evaluation=evaluation(passed=True))
    assert approved.generation.dossier.piva == "99000000002"

    with pytest.raises(ValidationError, match="passing eval gate"):
        ApprovedDossier(generation=generation(), evaluation=evaluation(passed=False))


def test_research_input_deep_freezes_hub_data_and_serializes_a_copy() -> None:
    payload = {"crm": {"labels": ["T1"]}}
    research_input = ResearchInput(
        piva="99000000002",
        as_of=AS_OF,
        hub_data=payload,
        evidence=(evidence(),),
    )
    payload["crm"] = {"labels": ["changed"]}

    assert research_input.model_dump(mode="json")["hub_data"] == {"crm": {"labels": ["T1"]}}
    with pytest.raises(TypeError):
        cast(Any, research_input.hub_data)["new"] = "value"


def test_research_input_rejects_credentials_and_unbounded_hub_data() -> None:
    for sensitive_key in ("api_token", "accessToken", "clientSecret", "credentials"):
        with pytest.raises(ValidationError, match="credential-like key"):
            ResearchInput(
                piva="99000000002",
                as_of=AS_OF,
                hub_data={sensitive_key: "do-not-send"},
                evidence=(evidence(),),
            )
    with pytest.raises(ValidationError, match="cannot exceed"):
        ResearchInput(
            piva="99000000002",
            as_of=AS_OF,
            hub_data={"notes": "x" * 50_001},
            evidence=(evidence(),),
        )


def test_dossier_rejects_free_form_call_cautions() -> None:
    payload = dossier().model_dump(mode="python", by_alias=True)
    payload["cosa_non_dire"] = ("L'azienda non ha bisogno del prodotto.",)

    with pytest.raises(ValidationError, match="Input should be"):
        ResearchDossier.model_validate(payload)
