"""Continuous evaluation for research dossiers.

The deterministic citation gate is the primary trust boundary.  A semantic
judge may make the gate stricter, but can never turn a deterministic rejection
into an approval.  Golden-set expectations then guard prompt/model behaviour
that is valid yet semantically different from the approved baseline.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from list_engine.research.models import (
    AgentGeneration,
    DossierEvaluation,
    EvaluationIssue,
    ResearchInput,
)

EVALUATOR_VERSION = "research-eval-v1"

SEMANTIC_JUDGE_RUBRIC: dict[str, str] = {
    "factuality": "Every factual statement is entailed by the supplied evidence.",
    "citation_correctness": "Every citation resolves to the exact supporting source excerpt.",
    "completeness": "The dossier covers the supported facts, signals, hook and call cautions.",
    "source_quality": "Claims prefer first-party or official sources and expose weaker sources.",
    "hallucination_free": "Unsupported details, identities, intent and certainty are absent.",
}


class ResearchAgentPort(Protocol):
    """Structural port used by the golden runner."""

    async def generate(self, research_input: ResearchInput) -> AgentGeneration:
        """Generate one untrusted dossier."""
        ...


class CitationGatePort(Protocol):
    """The mandatory deterministic gate; deliberately synchronous and pure."""

    def evaluate(
        self,
        research_input: ResearchInput,
        generation: AgentGeneration,
    ) -> DossierEvaluation:
        """Return deterministic citation/evidence checks."""
        ...


class SemanticJudgeResult(BaseModel):
    """Structured output required from an optional LLM-as-judge adapter."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    factuality: float = Field(ge=0, le=1)
    citation_correctness: float = Field(ge=0, le=1)
    completeness: float = Field(ge=0, le=1)
    source_quality: float = Field(ge=0, le=1)
    hallucination_free: float = Field(ge=0, le=1)
    issues: tuple[EvaluationIssue, ...] = ()
    judge_version: str = Field(min_length=1, max_length=64)


class SemanticJudge(Protocol):
    """Adapter boundary for a model-backed judge used in production CI."""

    async def evaluate(
        self,
        *,
        case_id: str,
        research_input: ResearchInput,
        generation: AgentGeneration,
        rubric: dict[str, str],
    ) -> SemanticJudgeResult:
        """Score one dossier against evidence, not against model world knowledge."""
        ...


class EvaluationThresholds(BaseModel):
    """Blocking dataset-level and per-dimension thresholds."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    minimum_dimension_score: float = Field(default=0.80, ge=0, le=1)
    minimum_mean_score: float = Field(default=0.90, ge=0, le=1)
    minimum_case_pass_rate: float = Field(default=1.0, ge=0, le=1)
    baseline_mean_score: float | None = Field(default=None, ge=0, le=1)
    maximum_mean_regression: float = Field(default=0.02, ge=0, le=1)


class GoldenExpectedOutput(BaseModel):
    """Stable, evidence-oriented expectations for one generated dossier."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    fact_evidence_ids: tuple[str, ...] = Field(max_length=3)
    signal_evidence_ids: tuple[str, ...] = ()
    persona_evidence_ids: tuple[str, ...] | None = None
    hook_evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    omitted_evidence_ids: tuple[str, ...] = ()
    required_claim_fragments: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_disjoint_omissions(self) -> GoldenExpectedOutput:
        expected = {
            *self.fact_evidence_ids,
            *self.signal_evidence_ids,
            *self.hook_evidence_ids,
            *(self.persona_evidence_ids or ()),
        }
        overlap = expected.intersection(self.omitted_evidence_ids)
        if overlap:
            raise ValueError("omitted evidence cannot also be expected in dossier output")
        return self


class GoldenCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    case_id: str = Field(pattern=r"[a-z][a-z0-9_-]{1,63}")
    description: str = Field(min_length=1, max_length=500)
    research_input: ResearchInput
    expected: GoldenExpectedOutput


class GoldenDataset(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    dataset_version: str = Field(min_length=1, max_length=64)
    thresholds: EvaluationThresholds
    cases: tuple[GoldenCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_case_ids(self) -> GoldenDataset:
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("golden case IDs must be unique")
        return self


class GoldenCaseResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    case_id: str
    passed: bool
    quality_score: float = Field(ge=0, le=1)
    evaluation: DossierEvaluation
    provider: str | None = None
    model: str | None = None
    prompt_version: str | None = None


class GoldenSetReport(BaseModel):
    """Serializable CI artifact suitable for quality trend tracking."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    dataset_version: str
    passed: bool
    case_count: int = Field(ge=1)
    passed_case_count: int = Field(ge=0)
    pass_rate: float = Field(ge=0, le=1)
    mean_quality_score: float = Field(ge=0, le=1)
    regression_from_baseline: float | None = Field(default=None, ge=0, le=1)
    failure_reasons: tuple[str, ...] = ()
    results: tuple[GoldenCaseResult, ...]


def load_golden_dataset(path: Path) -> GoldenDataset:
    """Load and strictly validate a versioned JSON golden set."""

    return GoldenDataset.model_validate_json(path.read_text(encoding="utf-8"))


def _quality_score(evaluation: DossierEvaluation) -> float:
    return (
        evaluation.factual_support
        + evaluation.citation_accuracy
        + evaluation.completeness
        + evaluation.source_quality
        + evaluation.hallucination_free
    ) / 5


def _golden_expectation_issues(
    expected: GoldenExpectedOutput,
    generation: AgentGeneration,
) -> tuple[EvaluationIssue, ...]:
    dossier = generation.dossier
    issues: list[EvaluationIssue] = []

    actual_facts = tuple(claim.evidence_id for claim in dossier.facts)
    if actual_facts != expected.fact_evidence_ids:
        issues.append(
            EvaluationIssue(
                code="golden_fact_evidence_mismatch",
                message=f"expected fact evidence {expected.fact_evidence_ids}, got {actual_facts}",
                field="fatti",
            )
        )

    actual_signals = tuple(claim.evidence_id for claim in dossier.active_signals)
    if actual_signals != expected.signal_evidence_ids:
        issues.append(
            EvaluationIssue(
                code="golden_signal_evidence_mismatch",
                message=(
                    f"expected signal evidence {expected.signal_evidence_ids}, got {actual_signals}"
                ),
                field="segnali_attivi",
            )
        )

    actual_persona = (
        dossier.probable_persona.evidence_ids if dossier.probable_persona is not None else None
    )
    if actual_persona != expected.persona_evidence_ids:
        issues.append(
            EvaluationIssue(
                code="golden_persona_evidence_mismatch",
                message=(
                    f"expected persona evidence {expected.persona_evidence_ids}, "
                    f"got {actual_persona}"
                ),
                field="persona_probabile",
            )
        )

    if dossier.opening_hook_evidence_ids != expected.hook_evidence_ids:
        issues.append(
            EvaluationIssue(
                code="golden_hook_evidence_mismatch",
                message=(
                    f"expected hook evidence {expected.hook_evidence_ids}, "
                    f"got {dossier.opening_hook_evidence_ids}"
                ),
                field="opening_hook_evidence_ids",
            )
        )

    cited_evidence = {
        *actual_facts,
        *actual_signals,
        *dossier.opening_hook_evidence_ids,
        *(actual_persona or ()),
    }
    leaked_omissions = cited_evidence.intersection(expected.omitted_evidence_ids)
    if leaked_omissions:
        issues.append(
            EvaluationIssue(
                code="golden_omitted_evidence_cited",
                message=f"evidence expected to be omitted was cited: {sorted(leaked_omissions)}",
            )
        )

    claim_text = " ".join(
        claim.claim for claim in (*dossier.facts, *dossier.active_signals)
    ).casefold()
    for fragment in expected.required_claim_fragments:
        if fragment.casefold() not in claim_text:
            issues.append(
                EvaluationIssue(
                    code="golden_claim_fragment_missing",
                    message=f"required claim fragment is missing: {fragment!r}",
                    field="fatti",
                )
            )
    return tuple(issues)


def _dimension_issues(
    *,
    factual_support: float,
    citation_accuracy: float,
    completeness: float,
    source_quality: float,
    hallucination_free: float,
    minimum: float,
) -> tuple[EvaluationIssue, ...]:
    dimensions = {
        "factual_support": factual_support,
        "citation_accuracy": citation_accuracy,
        "completeness": completeness,
        "source_quality": source_quality,
        "hallucination_free": hallucination_free,
    }
    return tuple(
        EvaluationIssue(
            code=f"{name}_below_threshold",
            message=f"{name} score {score:.3f} is below required {minimum:.3f}",
            field=name,
        )
        for name, score in dimensions.items()
        if score < minimum
    )


def combine_evaluations(
    deterministic: DossierEvaluation,
    *,
    semantic: SemanticJudgeResult | None,
    expectation_issues: tuple[EvaluationIssue, ...],
    minimum_dimension_score: float,
) -> DossierEvaluation:
    """Combine gates using minima; semantic approval can never mask a rejection."""

    factual_support = deterministic.factual_support
    citation_accuracy = deterministic.citation_accuracy
    completeness = deterministic.completeness
    source_quality = deterministic.source_quality
    hallucination_free = deterministic.hallucination_free
    issues = [*deterministic.issues, *expectation_issues]

    if not deterministic.passed and not deterministic.issues:
        issues.append(
            EvaluationIssue(
                code="deterministic_gate_failed",
                message="the deterministic citation gate rejected the dossier",
            )
        )

    judge_version = "none"
    if semantic is not None:
        factual_support = min(factual_support, semantic.factuality)
        citation_accuracy = min(citation_accuracy, semantic.citation_correctness)
        completeness = min(completeness, semantic.completeness)
        source_quality = min(source_quality, semantic.source_quality)
        hallucination_free = min(hallucination_free, semantic.hallucination_free)
        issues.extend(semantic.issues)
        judge_version = semantic.judge_version

    issues.extend(
        _dimension_issues(
            factual_support=factual_support,
            citation_accuracy=citation_accuracy,
            completeness=completeness,
            source_quality=source_quality,
            hallucination_free=hallucination_free,
            minimum=minimum_dimension_score,
        )
    )
    return DossierEvaluation(
        passed=deterministic.passed and not issues,
        factual_support=factual_support,
        citation_accuracy=citation_accuracy,
        completeness=completeness,
        source_quality=source_quality,
        hallucination_free=hallucination_free,
        issues=tuple(issues),
        evaluator_version=(
            f"{EVALUATOR_VERSION}+{deterministic.evaluator_version}+{judge_version}"
        )[:64],
    )


async def evaluate_golden_case(
    case: GoldenCase,
    *,
    generation: AgentGeneration,
    citation_gate: CitationGatePort,
    semantic_judge: SemanticJudge | None,
    minimum_dimension_score: float,
) -> GoldenCaseResult:
    """Evaluate a generated case through all configured, non-bypassable gates."""

    deterministic = citation_gate.evaluate(case.research_input, generation)
    semantic = None
    # The model-backed judge can only make a valid result stricter. Short-circuit
    # deterministic rejects to avoid unnecessary cost and exposure of bad output.
    if deterministic.passed and semantic_judge is not None:
        semantic = await semantic_judge.evaluate(
            case_id=case.case_id,
            research_input=case.research_input,
            generation=generation,
            rubric=SEMANTIC_JUDGE_RUBRIC,
        )
    evaluation = combine_evaluations(
        deterministic,
        semantic=semantic,
        expectation_issues=_golden_expectation_issues(case.expected, generation),
        minimum_dimension_score=minimum_dimension_score,
    )
    return GoldenCaseResult(
        case_id=case.case_id,
        passed=evaluation.passed,
        quality_score=_quality_score(evaluation),
        evaluation=evaluation,
        provider=generation.provider,
        model=generation.model,
        prompt_version=generation.prompt_version,
    )


def _failed_generation_result(case_id: str, error: Exception) -> GoldenCaseResult:
    issue = EvaluationIssue(
        code="agent_generation_failed",
        message=f"{type(error).__name__}: {str(error)[:1_500]}",
    )
    evaluation = DossierEvaluation(
        passed=False,
        factual_support=0.0,
        citation_accuracy=0.0,
        completeness=0.0,
        source_quality=0.0,
        hallucination_free=0.0,
        issues=(issue,),
        evaluator_version=EVALUATOR_VERSION,
    )
    return GoldenCaseResult(
        case_id=case_id,
        passed=False,
        quality_score=0.0,
        evaluation=evaluation,
    )


async def run_golden_set(
    dataset: GoldenDataset,
    *,
    agent: ResearchAgentPort,
    citation_gate: CitationGatePort,
    semantic_judge: SemanticJudge | None = None,
) -> GoldenSetReport:
    """Run all cases, continue after case errors, and compute a blocking report."""

    results: list[GoldenCaseResult] = []
    for case in dataset.cases:
        try:
            generation = await agent.generate(case.research_input)
            result = await evaluate_golden_case(
                case,
                generation=generation,
                citation_gate=citation_gate,
                semantic_judge=semantic_judge,
                minimum_dimension_score=dataset.thresholds.minimum_dimension_score,
            )
        except Exception as error:  # A failed case must not abort the remaining regression set.
            result = _failed_generation_result(case.case_id, error)
        results.append(result)

    passed_case_count = sum(result.passed for result in results)
    pass_rate = round(passed_case_count / len(results), 6)
    mean_quality_score = round(
        sum(result.quality_score for result in results) / len(results),
        6,
    )
    baseline = dataset.thresholds.baseline_mean_score
    regression = None if baseline is None else round(max(0.0, baseline - mean_quality_score), 6)

    failure_reasons: list[str] = []
    if pass_rate < dataset.thresholds.minimum_case_pass_rate:
        failure_reasons.append("case_pass_rate_below_threshold")
    if mean_quality_score < dataset.thresholds.minimum_mean_score:
        failure_reasons.append("mean_quality_score_below_threshold")
    if regression is not None and regression > dataset.thresholds.maximum_mean_regression:
        failure_reasons.append("mean_quality_regression_exceeded")

    return GoldenSetReport(
        dataset_version=dataset.dataset_version,
        passed=not failure_reasons,
        case_count=len(results),
        passed_case_count=passed_case_count,
        pass_rate=pass_rate,
        mean_quality_score=mean_quality_score,
        regression_from_baseline=regression,
        failure_reasons=tuple(failure_reasons),
        results=tuple(results),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the blocking research golden-set eval")
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path("evals/golden/research_cases.json"),
        help="path to the versioned golden-set JSON",
    )
    parser.add_argument(
        "--agent",
        choices=("demo", "claude"),
        default="demo",
        help="dossier generator under evaluation",
    )
    parser.add_argument(
        "--agent-model",
        help="Claude model identifier required when --agent=claude",
    )
    parser.add_argument(
        "--agent-max-budget-usd",
        type=float,
        default=0.10,
        help="maximum Claude generation spend per golden case",
    )
    parser.add_argument(
        "--semantic-judge",
        choices=("none", "claude"),
        default="none",
        help="optional model-backed judge; deterministic citation checks always run first",
    )
    parser.add_argument(
        "--judge-model",
        help="Claude model identifier required when --semantic-judge=claude",
    )
    parser.add_argument(
        "--judge-max-budget-usd",
        type=float,
        default=0.05,
        help="maximum Claude spend per golden case",
    )
    return parser


async def _run_cli(
    golden_path: Path,
    *,
    agent: ResearchAgentPort | None = None,
    semantic_judge: SemanticJudge | None = None,
) -> GoldenSetReport:
    # Imports stay local so the reusable eval module only depends on structural ports.
    from list_engine.research.agent import DemoResearchAgent
    from list_engine.research.gate import CitationGate

    return await run_golden_set(
        load_golden_dataset(golden_path),
        agent=agent or DemoResearchAgent(),
        citation_gate=CitationGate(),
        semantic_judge=semantic_judge,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    agent: ResearchAgentPort | None = None
    if args.agent == "claude":
        if not args.agent_model:
            parser.error("--agent-model is required when --agent=claude")
        from list_engine.research.claude_sdk import ClaudeSdkResearchAgent

        agent = ClaudeSdkResearchAgent(
            model=args.agent_model,
            max_budget_usd=args.agent_max_budget_usd,
        )
    semantic_judge: SemanticJudge | None = None
    if args.semantic_judge == "claude":
        if not args.judge_model:
            parser.error("--judge-model is required when --semantic-judge=claude")
        from list_engine.research.claude_judge import ClaudeSemanticJudge

        semantic_judge = ClaudeSemanticJudge(
            model=args.judge_model,
            max_budget_usd=args.judge_max_budget_usd,
        )
    report = asyncio.run(
        _run_cli(
            args.golden,
            agent=agent,
            semantic_judge=semantic_judge,
        )
    )
    print(report.model_dump_json(indent=2))
    return 0 if report.passed else 1


if __name__ == "__main__":  # pragma: no cover - exercised as a CI command.
    raise SystemExit(main())


__all__ = [
    "EVALUATOR_VERSION",
    "SEMANTIC_JUDGE_RUBRIC",
    "CitationGatePort",
    "EvaluationThresholds",
    "GoldenCase",
    "GoldenCaseResult",
    "GoldenDataset",
    "GoldenExpectedOutput",
    "GoldenSetReport",
    "ResearchAgentPort",
    "SemanticJudge",
    "SemanticJudgeResult",
    "combine_evaluations",
    "evaluate_golden_case",
    "load_golden_dataset",
    "main",
    "run_golden_set",
]
