"""Command-line entrypoints for the hot pipeline.

``demo`` runs the whole signal -> event -> dossier -> task path offline (no DB, no
network) and prints the task with its signal->task latency. ``worker`` polls the
Postgres outbox and processes hot events (offline DEMO research; production swaps
in the Claude agent behind the same ResearchRunner port).
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime, timedelta

from list_engine.hot.worker import HotEventWorker, HotWorkflowResult


def _print_result(result: HotWorkflowResult) -> None:
    payload: dict[str, object] = {"piva": result.piva, "outcome": result.outcome.value}
    if result.task is not None:
        payload["signal_to_task_seconds"] = str(result.task.signal_to_task_seconds)
        payload["dossier_digest"] = result.task.dossier_digest
    print(json.dumps(payload))


def _run_demo() -> int:
    from list_engine.hot.demo import build_demo_pipeline, demo_clay_signal

    received = datetime(2026, 7, 17, 8, 0, 0, tzinfo=UTC)
    task_time = received + timedelta(seconds=2)
    pipeline = build_demo_pipeline(clock=lambda: task_time, seed_observed_at=received)
    pipeline.ingest.accept(demo_clay_signal(received_at=received), now=received)
    result = asyncio.run(pipeline.worker.run_once())
    if result is None:
        print(json.dumps({"outcome": "no_event"}))
        return 1
    _print_result(result)
    return 0


async def _drain(worker: HotEventWorker) -> int:
    processed = 0
    while (result := await worker.run_once()) is not None:
        _print_result(result)
        processed += 1
    print(json.dumps({"drained": processed}))
    return 0


def _run_worker(dsn: str, *, follow: bool, poll_interval: float) -> int:
    from list_engine.adapters.postgres import PostgresCompanyRepository
    from list_engine.hot.postgres import PostgresHotStore
    from list_engine.hot.worker import DirectResearchRunner, HotWorkflow, SignalEvidenceBuilder
    from list_engine.research.agent import DemoResearchAgent
    from list_engine.research.gate import CitationGate

    store = PostgresHotStore.connect(dsn)
    companies = PostgresCompanyRepository.connect(dsn)
    workflow = HotWorkflow(
        companies=companies,
        research=DirectResearchRunner(DemoResearchAgent(), CitationGate()),
        tasks=store,
        evidence=SignalEvidenceBuilder(),
    )
    worker = HotEventWorker(outbox=store, runner=workflow, clock=lambda: datetime.now(UTC))
    try:
        if follow:
            asyncio.run(
                worker.run_forever(
                    poll_interval=timedelta(seconds=poll_interval), stop=asyncio.Event()
                )
            )
            return 0
        return asyncio.run(_drain(worker))
    finally:
        store.close()
        companies.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="list_engine.hot", description="Hot-signal pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="Run the offline in-memory hot path and print the task")
    worker = sub.add_parser("worker", help="Poll the Postgres outbox and process hot events")
    worker.add_argument("--dsn", required=True, help="PostgreSQL DSN")
    worker.add_argument(
        "--follow", action="store_true", help="Run continuously instead of draining once"
    )
    worker.add_argument(
        "--poll-interval", type=float, default=2.0, help="Idle poll seconds in --follow mode"
    )
    args = parser.parse_args(argv)
    if args.command == "demo":
        return _run_demo()
    return _run_worker(args.dsn, follow=args.follow, poll_interval=args.poll_interval)


if __name__ == "__main__":
    raise SystemExit(main())
