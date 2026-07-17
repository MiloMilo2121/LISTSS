# Sibill List Engine

List Engine turns public Italian company data and fresh readiness signals into a
compliance-checked, prioritised call queue. The repository is being built in eight
independently verifiable milestones; the complete DEMO run and production wiring are
documented in the final milestone.

Current milestone: **M5 — event-driven hot-signal pipeline**.

Fresh readiness signals (Company Monitoring, hiring, Clay webhooks) become a BDR
call task with a cited dossier in minutes. A signed webhook ingress verifies each
payload over its raw body, normalizes it to a neutral `HotSignal`, and enqueues it
in one transaction (append-only raw landing plus an outbox event). A leased worker
claims each event, assembles a verified evidence packet, runs the M4 research
pipeline, and writes an internal `hot_task` carrying the `ApprovedDossier`. HubSpot
delivery is M6; M5 stops at the durable task and measures the headline KPI —
signal→task latency (the `hot_signal_latency` view).

Every effect is idempotent (raw content hash → outbox dedupe key → task natural
key), so a redelivered webhook never creates a second event or a duplicate task.
The whole path runs offline with the deterministic DEMO agent and an in-memory
outbox; a durable-execution engine (DBOS/Hatchet) can replace the Postgres poller
behind the `Outbox`/`WorkflowRunner` ports at M8 without touching the domain
workflow.

Run the offline hot path (signal → event → dossier → task) with no DB or network:

```bash
uv run python -m list_engine.hot demo
```

The webhook ingress needs the optional `ingress` extra. Build the app with
`list_engine.hot.ingress.create_ingress_app(...)` at a composition root — injecting
the per-source signature verifiers (HMAC over the raw body, `SecretStr`) and a
`SignalIngestService` — and serve it under uvicorn. Process the queue with the
Postgres worker:

```bash
uv sync --extra dev --extra agent --extra ingress
uv run python -m list_engine.hot worker --dsn "$DATABASE_URL"
```

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
