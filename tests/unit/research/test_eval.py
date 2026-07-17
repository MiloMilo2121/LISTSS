from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from list_engine.research.agent import DemoResearchAgent
from list_engine.research.eval import (
    SEMANTIC_JUDGE_RUBRIC,
    EvaluationThresholds,
    GoldenDataset,
    SemanticJudgeResult,
    evaluate_golden_case,
    load_golden_dataset,
    main,
    run_golden_set,
)
from list_engine.research.gate import CitationGate
from list_engine.research.models import (
    AgentGeneration,
    DossierEvaluation,
    EvaluationIssue,
    ResearchInput,
)

GOLDEN_PATH = Path("evals/golden/research_cases.json")


class AlwaysPassJudge:
    async def evaluate(
        self,
        *,
        case_id: str,
        research_input: ResearchInput,
        generation: AgentGeneration,
        rubric: dict[str, str],
    ) -> SemanticJudgeResult:
        assert case_id
        assert research_input.piva == generation.dossier.piva
        assert rubric == SEMANTIC_JUDGE_RUBRIC
        return SemanticJudgeResult(
            factuality=1.0,
            citation_correctness=1.0,
            completeness=1.0,
            source_quality=1.0,
            hallucination_free=1.0,
            judge_version="test-perfect-judge",
        )


class LowFactualityJudge:
    async def evaluate(
        self,
        *,
        case_id: str,
        research_input: ResearchInput,
        generation: AgentGeneration,
        rubric: dict[str, str],
    ) -> SemanticJudgeResult:
        del case_id, research_input, generation, rubric
        return SemanticJudgeResult(
            factuality=0.40,
            citation_correctness=1.0,
            completeness=1.0,
            source_quality=1.0,
            hallucination_free=1.0,
            judge_version="test-low-factuality",
        )


class ForcedGate:
    def __init__(self, evaluation: DossierEvaluation) -> None:
        self.evaluation = evaluation

    def evaluate(
        self,
        research_input: ResearchInput,
        generation: AgentGeneration,
    ) -> DossierEvaluation:
        del research_input, generation
        return self.evaluation


def _evaluation(*, passed: bool, score: float = 1.0) -> DossierEvaluation:
    issues: tuple[EvaluationIssue, ...] = ()
    if not passed:
        issues = (
            EvaluationIssue(
                code="forced_deterministic_rejection",
                message="forced by the deterministic test gate",
            ),
        )
    return DossierEvaluation(
        passed=passed,
        factual_support=score,
        citation_accuracy=score,
        completeness=score,
        source_quality=score,
        hallucination_free=score,
        issues=issues,
        evaluator_version="test-gate-v1",
    )


def _single_case_dataset(
    *,
    thresholds: EvaluationThresholds | None = None,
) -> GoldenDataset:
    loaded = load_golden_dataset(GOLDEN_PATH)
    return GoldenDataset(
        dataset_version="single-test-v1",
        thresholds=thresholds or loaded.thresholds,
        cases=(loaded.cases[0],),
    )


def test_versioned_golden_set_has_twenty_synthetic_cases() -> None:
    dataset = load_golden_dataset(GOLDEN_PATH)

    assert len(dataset.cases) == 20
    assert len({case.case_id for case in dataset.cases}) == 20
    assert all(
        evidence.source_url.endswith(".invalid") or ".invalid/" in evidence.source_url
        for case in dataset.cases
        for evidence in case.research_input.evidence
    )


def test_demo_agent_passes_complete_blocking_golden_set() -> None:
    report = asyncio.run(
        run_golden_set(
            load_golden_dataset(GOLDEN_PATH),
            agent=DemoResearchAgent(),
            citation_gate=CitationGate(),
        )
    )

    assert report.passed is True
    assert report.passed_case_count == 20
    assert report.pass_rate == 1.0
    assert report.mean_quality_score >= 0.90
    assert report.failure_reasons == ()


def test_semantic_judge_cannot_override_deterministic_rejection() -> None:
    dataset = _single_case_dataset()
    case = dataset.cases[0]
    generation = asyncio.run(DemoResearchAgent().generate(case.research_input))

    result = asyncio.run(
        evaluate_golden_case(
            case,
            generation=generation,
            citation_gate=ForcedGate(_evaluation(passed=False)),
            semantic_judge=AlwaysPassJudge(),
            minimum_dimension_score=0.65,
        )
    )

    assert result.passed is False
    assert "forced_deterministic_rejection" in {issue.code for issue in result.evaluation.issues}


def test_semantic_judge_can_make_a_valid_dossier_fail() -> None:
    dataset = _single_case_dataset()
    case = dataset.cases[0]
    generation = asyncio.run(DemoResearchAgent().generate(case.research_input))

    result = asyncio.run(
        evaluate_golden_case(
            case,
            generation=generation,
            citation_gate=CitationGate(),
            semantic_judge=LowFactualityJudge(),
            minimum_dimension_score=0.65,
        )
    )

    assert result.passed is False
    assert result.evaluation.factual_support == 0.40
    assert "factual_support_below_threshold" in {issue.code for issue in result.evaluation.issues}


def test_golden_expectation_mismatch_is_blocking() -> None:
    dataset = _single_case_dataset()
    original = dataset.cases[0]
    bad_expected = original.expected.model_copy(update={"fact_evidence_ids": ()})
    mismatched_case = original.model_copy(update={"expected": bad_expected})
    generation = asyncio.run(DemoResearchAgent().generate(original.research_input))

    result = asyncio.run(
        evaluate_golden_case(
            mismatched_case,
            generation=generation,
            citation_gate=CitationGate(),
            semantic_judge=None,
            minimum_dimension_score=0.65,
        )
    )

    assert result.passed is False
    assert "golden_fact_evidence_mismatch" in {issue.code for issue in result.evaluation.issues}


def test_regression_threshold_blocks_even_when_every_case_passes() -> None:
    thresholds = EvaluationThresholds(
        minimum_dimension_score=0.50,
        minimum_mean_score=0.70,
        minimum_case_pass_rate=1.0,
        baseline_mean_score=0.90,
        maximum_mean_regression=0.05,
    )
    dataset = _single_case_dataset(thresholds=thresholds)

    report = asyncio.run(
        run_golden_set(
            dataset,
            agent=DemoResearchAgent(),
            citation_gate=ForcedGate(_evaluation(passed=True, score=0.80)),
        )
    )

    assert report.passed_case_count == 1
    assert report.pass_rate == 1.0
    assert report.mean_quality_score == 0.80
    assert report.regression_from_baseline == 0.10
    assert report.failure_reasons == ("mean_quality_regression_exceeded",)


def test_agent_failure_is_recorded_and_remaining_cases_continue() -> None:
    class FailFirstAgent:
        def __init__(self) -> None:
            self.calls = 0

        async def generate(self, research_input: ResearchInput) -> AgentGeneration:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("synthetic generation failure")
            return await DemoResearchAgent().generate(research_input)

    loaded = load_golden_dataset(GOLDEN_PATH)
    dataset = GoldenDataset(
        dataset_version="continue-after-error-v1",
        thresholds=loaded.thresholds,
        cases=loaded.cases[:2],
    )
    report = asyncio.run(
        run_golden_set(dataset, agent=FailFirstAgent(), citation_gate=CitationGate())
    )

    assert report.passed is False
    assert report.results[0].evaluation.issues[0].code == "agent_generation_failed"
    assert report.results[1].passed is True


def test_cli_returns_nonzero_when_regression_is_blocking(tmp_path: Path) -> None:
    loaded = load_golden_dataset(GOLDEN_PATH)
    strict_thresholds = loaded.thresholds.model_copy(
        update={"baseline_mean_score": 1.0, "maximum_mean_regression": 0.0}
    )
    regressed = loaded.model_copy(update={"thresholds": strict_thresholds})
    path = tmp_path / "regressed-golden.json"
    path.write_text(regressed.model_dump_json(), encoding="utf-8")

    assert main(["--golden", str(path)]) == 1


def test_cli_requires_an_explicit_model_for_the_real_semantic_judge() -> None:
    with pytest.raises(SystemExit) as raised:
        main(["--golden", str(GOLDEN_PATH), "--semantic-judge", "claude"])

    assert raised.value.code == 2


def test_cli_requires_an_explicit_model_for_the_real_research_agent() -> None:
    with pytest.raises(SystemExit) as raised:
        main(["--golden", str(GOLDEN_PATH), "--agent", "claude"])

    assert raised.value.code == 2
