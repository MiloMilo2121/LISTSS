# Sibill List Engine

List Engine turns public Italian company data and fresh readiness signals into a
compliance-checked, prioritised call queue. The repository is being built in eight
independently verifiable milestones; the complete DEMO run and production wiring are
documented in the final milestone.

Current milestone: **M4 — cited research agent and blocking evaluation**.

The DEMO research path is deterministic and offline. Production generation uses
the optional Claude Agent SDK adapter with every built-in tool, MCP server, skill,
and ambient setting disabled: the model receives one bounded packet of verified
evidence and returns one Pydantic-validated dossier. Claims and call hooks are
extractive, inferences require explicit evidence tags, and only an
`ApprovedDossier` may cross the citation gate.

Run the versioned 20-case quality gate locally:

```bash
uv sync --extra dev --extra agent
uv run python -m list_engine.research.eval \
  --golden evals/golden/research_cases.json
```

When CI has `ANTHROPIC_API_KEY` plus the `ANTHROPIC_AGENT_MODEL` and
`ANTHROPIC_JUDGE_MODEL` repository variables, it runs the production Claude
adapter over the golden set and evaluates those outputs with the real Claude
semantic judge (hard ceiling: USD 3 per full 20-case run). Without credentials
the deterministic citation and regression gate remains mandatory, so DEMO and
pull requests from forks stay reproducible and offline.

PostgreSQL retains the complete successful retry history, cost, tokens, final
evaluation, and review failures. The disposable dossier cache is keyed by the
exact input/prompt/provider/model identity, linked to its audit session, and
protected by a short generation lease so hot and cold workers cannot pay twice.
