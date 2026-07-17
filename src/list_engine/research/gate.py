"""Blocking, deterministic citation checks for research dossiers."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from list_engine.research.agent import OPENING_HOOK_PREFIX
from list_engine.research.models import (
    AgentGeneration,
    ApprovedDossier,
    CitedClaim,
    CitedInference,
    DossierEvaluation,
    EvaluationIssue,
    EvidenceItem,
    ResearchInput,
)

EVALUATOR_VERSION = "citation-gate-v1"
DEFAULT_LEXICAL_SUPPORT_THRESHOLD = 0.60
DEFAULT_MIN_SOURCE_QUALITY = 0.60

_SOURCE_QUALITY = {
    "official_registry": 1.00,
    "company_website": 0.90,
    "job_board": 0.80,
    "crm": 0.75,
    "public_directory": 0.65,
}
_STOP_WORDS = frozenset(
    {
        "a",
        "ad",
        "al",
        "alla",
        "alle",
        "con",
        "da",
        "dal",
        "dalla",
        "dei",
        "del",
        "della",
        "di",
        "e",
        "ed",
        "gli",
        "ha",
        "il",
        "in",
        "la",
        "le",
        "lo",
        "nel",
        "nella",
        "per",
        "si",
        "su",
        "un",
        "una",
        "the",
        "of",
        "and",
        "for",
        "to",
    }
)
_TOKEN = re.compile(r"[^\W_]+", flags=re.UNICODE)


class DossierRejectedError(ValueError):
    """Raised when a generation cannot cross the citation trust boundary."""

    def __init__(self, evaluation: DossierEvaluation) -> None:
        self.evaluation = evaluation
        codes = ", ".join(issue.code for issue in evaluation.issues)
        super().__init__(f"dossier rejected by {evaluation.evaluator_version}: {codes}")


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _lexical_support(claim: str, supporting_excerpt: str) -> float:
    claim_tokens = [
        token
        for token in _TOKEN.findall(_normalized(claim))
        if token not in _STOP_WORDS and len(token) > 1
    ]
    if not claim_tokens:
        return 0.0
    excerpt_tokens = set(_TOKEN.findall(_normalized(supporting_excerpt)))
    supported = sum(token in excerpt_tokens for token in claim_tokens)
    return supported / len(claim_tokens)


def _quality(item: EvidenceItem) -> float:
    quality = _SOURCE_QUALITY[item.source_kind]
    if not item.source_url.startswith("https://"):
        quality *= 0.5
    return quality


def _issue(code: str, message: str, field: str | None = None) -> EvaluationIssue:
    return EvaluationIssue(code=code, message=message, field=field)


class CitationGate:
    """Enforce CITA-O-ASTIENITI before a dossier can reach delivery code."""

    def __init__(
        self,
        *,
        lexical_support_threshold: float = DEFAULT_LEXICAL_SUPPORT_THRESHOLD,
        min_source_quality: float = DEFAULT_MIN_SOURCE_QUALITY,
    ) -> None:
        if not 0 < lexical_support_threshold <= 1:
            raise ValueError("lexical_support_threshold must be in (0, 1]")
        if not 0 < min_source_quality <= 1:
            raise ValueError("min_source_quality must be in (0, 1]")
        self._lexical_support_threshold = lexical_support_threshold
        self._min_source_quality = min_source_quality

    def evaluate(
        self,
        research_input: ResearchInput,
        generation: AgentGeneration,
    ) -> DossierEvaluation:
        issues: list[EvaluationIssue] = []
        dossier = generation.dossier
        evidence_by_id = {item.evidence_id: item for item in research_input.evidence}
        referenced: dict[str, EvidenceItem] = {}

        if dossier.piva != research_input.piva:
            issues.append(
                _issue(
                    "piva_mismatch",
                    "The dossier P.IVA does not match its bounded research input.",
                    "piva",
                )
            )

        claim_support: list[float] = []
        citation_checks: list[bool] = []
        for field, claims, require_signal_tag in (
            ("fatti", dossier.facts, False),
            ("segnali_attivi", dossier.active_signals, True),
        ):
            for index, claim in enumerate(claims):
                path = f"{field}.{index}"
                valid, support = self._check_claim(
                    claim,
                    path=path,
                    evidence_by_id=evidence_by_id,
                    referenced=referenced,
                    issues=issues,
                    require_signal_tag=require_signal_tag,
                )
                citation_checks.append(valid)
                claim_support.append(support)

        if dossier.probable_persona is not None:
            persona_valid = self._check_tagged_inference(
                dossier.probable_persona,
                path="persona_probabile",
                kind="inference",
                tag_prefix="persona:",
                evidence_by_id=evidence_by_id,
                referenced=referenced,
                issues=issues,
            )
            citation_checks.append(persona_valid)

        if dossier.likely_objection is not None:
            objection_valid = self._check_tagged_inference(
                dossier.likely_objection,
                path="obiezione_probabile",
                kind="objection",
                tag_prefix="objection:",
                evidence_by_id=evidence_by_id,
                referenced=referenced,
                issues=issues,
            )
            citation_checks.append(objection_valid)

        hook_valid = self._check_reference_ids(
            dossier.opening_hook_evidence_ids,
            path="opening_hook_evidence_ids",
            kind="hook",
            evidence_by_id=evidence_by_id,
            referenced=referenced,
            issues=issues,
        )
        hook_evidence = tuple(
            evidence_by_id[evidence_id]
            for evidence_id in dossier.opening_hook_evidence_ids
            if evidence_id in evidence_by_id and evidence_by_id[evidence_id].verified
        )
        normalized_hook = _normalized(dossier.opening_hook)
        normalized_prefix = _normalized(OPENING_HOOK_PREFIX)
        hook_quote = (
            normalized_hook.removeprefix(normalized_prefix).strip()
            if normalized_hook.startswith(normalized_prefix)
            else ""
        )
        if not hook_quote or not any(
            hook_quote in _normalized(item.excerpt) for item in hook_evidence
        ):
            issues.append(
                _issue(
                    "hook_not_extractive",
                    "The hook must be the approved prefix followed only by an exact "
                    "excerpt from one of its cited evidence items.",
                    "hook_apertura",
                )
            )
            hook_valid = False
        citation_checks.append(hook_valid)

        verified_ids = {item.evidence_id for item in research_input.evidence if item.verified}
        fact_ids = {claim.evidence_id for claim in dossier.facts}
        required_fact_count = min(3, len(verified_ids))
        covered_fact_count = min(required_fact_count, len(fact_ids.intersection(verified_ids)))
        fact_completeness = covered_fact_count / required_fact_count if required_fact_count else 0.0
        if covered_fact_count < required_fact_count:
            issues.append(
                _issue(
                    "facts_incomplete",
                    "The dossier must cite distinct verified evidence in up to three facts.",
                    "fatti",
                )
            )

        expected_signal_ids = {
            item.evidence_id
            for item in research_input.evidence
            if item.verified
            and any(
                tag.startswith("signal:") and tag.removeprefix("signal:").strip()
                for tag in item.tags
            )
        }
        actual_signal_ids = {claim.evidence_id for claim in dossier.active_signals}
        missing_signal_ids = expected_signal_ids.difference(actual_signal_ids)
        signal_completeness = (
            len(expected_signal_ids.intersection(actual_signal_ids)) / len(expected_signal_ids)
            if expected_signal_ids
            else 1.0
        )
        if missing_signal_ids:
            issues.append(
                _issue(
                    "signals_incomplete",
                    "Tagged active signals are missing from the dossier: "
                    + ", ".join(sorted(missing_signal_ids)),
                    "segnali_attivi",
                )
            )

        persona_expected = _has_verified_tag(research_input.evidence, "persona:")
        persona_complete = not persona_expected or dossier.probable_persona is not None
        if not persona_complete:
            issues.append(
                _issue(
                    "persona_incomplete",
                    "A verified persona:* tag requires a cited probable persona.",
                    "persona_probabile",
                )
            )

        objection_expected = _has_verified_tag(research_input.evidence, "objection:")
        objection_complete = not objection_expected or dossier.likely_objection is not None
        if not objection_complete:
            issues.append(
                _issue(
                    "objection_incomplete",
                    "A verified objection:* tag requires a cited likely objection.",
                    "obiezione_probabile",
                )
            )

        completeness = (
            sum(
                (
                    fact_completeness,
                    signal_completeness,
                    float(persona_complete),
                    float(objection_complete),
                    float(hook_valid),
                )
            )
            / 5
        )

        qualities = tuple(_quality(item) for item in referenced.values())
        source_quality = sum(qualities) / len(qualities) if qualities else 0.0
        low_quality_ids = tuple(
            item.evidence_id
            for item in referenced.values()
            if _quality(item) < self._min_source_quality
        )
        if not referenced:
            issues.append(
                _issue(
                    "no_cited_evidence",
                    "CITA-O-ASTIENITI requires at least one existing cited evidence item.",
                )
            )
        elif low_quality_ids:
            issues.append(
                _issue(
                    "low_source_quality",
                    "Cited evidence is below the source-quality floor: "
                    + ", ".join(low_quality_ids),
                )
            )

        factual_support = sum(claim_support) / len(claim_support) if claim_support else 1.0
        citation_accuracy = sum(citation_checks) / len(citation_checks) if citation_checks else 0.0
        hallucination_codes = {
            "citation_evidence_missing",
            "citation_url_mismatch",
            "claim_not_supported",
            "claim_not_extractive",
            "evidence_unverified",
            "excerpt_not_found",
            "hook_evidence_missing",
            "hook_evidence_unverified",
            "hook_not_extractive",
            "inference_evidence_missing",
            "inference_evidence_unverified",
            "inference_tag_missing",
            "inference_tag_not_supported",
            "no_cited_evidence",
            "objection_evidence_missing",
            "objection_evidence_unverified",
            "objection_tag_missing",
            "objection_tag_not_supported",
            "piva_mismatch",
            "signal_tag_missing",
        }
        hallucination_free = (
            0.0 if any(issue.code in hallucination_codes for issue in issues) else 1.0
        )
        return DossierEvaluation(
            passed=not issues,
            factual_support=round(factual_support, 6),
            citation_accuracy=round(citation_accuracy, 6),
            completeness=round(completeness, 6),
            source_quality=round(source_quality, 6),
            hallucination_free=hallucination_free,
            issues=tuple(issues),
            evaluator_version=EVALUATOR_VERSION,
        )

    def approve(
        self,
        research_input: ResearchInput,
        generation: AgentGeneration,
    ) -> ApprovedDossier:
        evaluation = self.evaluate(research_input, generation)
        if not evaluation.passed:
            raise DossierRejectedError(evaluation)
        return ApprovedDossier(generation=generation, evaluation=evaluation)

    def _check_claim(
        self,
        claim: CitedClaim,
        *,
        path: str,
        evidence_by_id: dict[str, EvidenceItem],
        referenced: dict[str, EvidenceItem],
        issues: list[EvaluationIssue],
        require_signal_tag: bool,
    ) -> tuple[bool, float]:
        valid = True
        item = evidence_by_id.get(claim.evidence_id)
        if item is None:
            issues.append(
                _issue(
                    "citation_evidence_missing",
                    f"Evidence '{claim.evidence_id}' is not in the research input.",
                    f"{path}.evidence_id",
                )
            )
            valid = False
        else:
            referenced[item.evidence_id] = item
            if claim.source_url != item.source_url:
                issues.append(
                    _issue(
                        "citation_url_mismatch",
                        "The claim URL is not the exact URL stored for its evidence ID.",
                        f"{path}.source_url",
                    )
                )
                valid = False
            if not item.verified:
                issues.append(
                    _issue(
                        "evidence_unverified",
                        f"Evidence '{item.evidence_id}' has not been verified.",
                        f"{path}.evidence_id",
                    )
                )
                valid = False
            if _normalized(claim.supporting_excerpt) not in _normalized(item.excerpt):
                issues.append(
                    _issue(
                        "excerpt_not_found",
                        "The normalized supporting excerpt is not present in the source evidence.",
                        f"{path}.supporting_excerpt",
                    )
                )
                valid = False
            if require_signal_tag and not any(
                tag.startswith("signal:") and tag.removeprefix("signal:").strip()
                for tag in item.tags
            ):
                issues.append(
                    _issue(
                        "signal_tag_missing",
                        "An active signal must cite evidence carrying a signal:* tag.",
                        f"{path}.evidence_id",
                    )
                )
                valid = False

        support = _lexical_support(claim.claim, claim.supporting_excerpt)
        if _normalized(claim.claim) != _normalized(claim.supporting_excerpt):
            issues.append(
                _issue(
                    "claim_not_extractive",
                    "A factual claim must exactly reproduce its verified supporting excerpt.",
                    f"{path}.claim",
                )
            )
            valid = False
        if support < self._lexical_support_threshold:
            issues.append(
                _issue(
                    "claim_not_supported",
                    f"Lexical support {support:.3f} is below "
                    f"{self._lexical_support_threshold:.3f}.",
                    f"{path}.claim",
                )
            )
            valid = False
        return valid, support

    @staticmethod
    def _check_reference_ids(
        evidence_ids: Iterable[str],
        *,
        path: str,
        kind: str,
        evidence_by_id: dict[str, EvidenceItem],
        referenced: dict[str, EvidenceItem],
        issues: list[EvaluationIssue],
    ) -> bool:
        valid = True
        for index, evidence_id in enumerate(evidence_ids):
            item = evidence_by_id.get(evidence_id)
            if item is None:
                issues.append(
                    _issue(
                        f"{kind}_evidence_missing",
                        f"Evidence '{evidence_id}' is not in the research input.",
                        f"{path}.{index}",
                    )
                )
                valid = False
                continue
            referenced[item.evidence_id] = item
            if not item.verified:
                issues.append(
                    _issue(
                        f"{kind}_evidence_unverified",
                        f"Evidence '{evidence_id}' has not been verified.",
                        f"{path}.{index}",
                    )
                )
                valid = False
        return valid

    @classmethod
    def _check_tagged_inference(
        cls,
        inference: CitedInference,
        *,
        path: str,
        kind: str,
        tag_prefix: str,
        evidence_by_id: dict[str, EvidenceItem],
        referenced: dict[str, EvidenceItem],
        issues: list[EvaluationIssue],
    ) -> bool:
        valid = cls._check_reference_ids(
            inference.evidence_ids,
            path=f"{path}.evidence_ids",
            kind=kind,
            evidence_by_id=evidence_by_id,
            referenced=referenced,
            issues=issues,
        )
        tag_values = tuple(
            tag.removeprefix(tag_prefix).strip()
            for evidence_id in inference.evidence_ids
            if (item := evidence_by_id.get(evidence_id)) is not None and item.verified
            for tag in item.tags
            if tag.startswith(tag_prefix) and tag.removeprefix(tag_prefix).strip()
        )
        if not tag_values:
            issues.append(
                _issue(
                    f"{kind}_tag_missing",
                    f"The inference requires an explicit {tag_prefix} evidence tag.",
                    path,
                )
            )
            return False

        label = "Persona probabile" if kind == "inference" else "Obiezione probabile"
        expected_texts = {_normalized(f"{label}: {value}") for value in tag_values}
        if _normalized(inference.text) not in expected_texts:
            issues.append(
                _issue(
                    f"{kind}_tag_not_supported",
                    "The inference text must exactly reproduce one explicitly tagged value.",
                    path,
                )
            )
            valid = False
        return valid


def _has_verified_tag(evidence: Iterable[EvidenceItem], prefix: str) -> bool:
    return any(
        item.verified
        and any(tag.startswith(prefix) and tag.removeprefix(prefix).strip() for tag in item.tags)
        for item in evidence
    )


__all__ = [
    "DEFAULT_LEXICAL_SUPPORT_THRESHOLD",
    "DEFAULT_MIN_SOURCE_QUALITY",
    "EVALUATOR_VERSION",
    "CitationGate",
    "DossierRejectedError",
]
