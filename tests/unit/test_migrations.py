from __future__ import annotations

import re
from pathlib import Path

from list_engine.core.piva import is_valid_piva

ROOT = Path(__file__).resolve().parents[2]


def test_initial_schema_contains_every_required_foundation_table() -> None:
    ddl = (ROOT / "supabase" / "migrations" / "202607150001_initial.sql").read_text()
    required_tables = {
        "companies",
        "company_firmographic_history",
        "source_records",
        "enrichment_log",
        "contacts",
        "signals",
        "scoring_weight_versions",
        "scores",
        "suppression_list",
        "campaign_outcomes",
        "agent_sessions",
        "review_queue",
        "dlq",
        "idempotency_keys",
        "quarantine",
        "provider_health",
        "compliance_checks",
        "audit_log",
    }

    declared = set(re.findall(r"CREATE TABLE IF NOT EXISTS ([a-z_]+)", ddl))
    assert required_tables <= declared
    assert "prevent_append_only_mutation" in ddl
    assert "source_records_replay_guard" in ddl


def test_demo_seed_has_exactly_twenty_unique_checksum_valid_pivas() -> None:
    seed = (ROOT / "seeds" / "demo.sql").read_text()
    pivas = re.findall(r"\('([0-9]{11})', '[^']+ Demo ", seed)

    assert len(pivas) == 20
    assert len(set(pivas)) == 20
    assert all(is_valid_piva(piva) for piva in pivas)
    assert seed.count("true, '2026-07-01T00:00:00Z'") == 20
    assert "list_engine.app_mode" in seed


def test_migrations_are_replay_safe_by_construction() -> None:
    initial = (ROOT / "supabase" / "migrations" / "202607150001_initial.sql").read_text()
    seed = (ROOT / "seeds" / "demo.sql").read_text()

    assert "CREATE TABLE " not in initial.replace("CREATE TABLE IF NOT EXISTS", "")
    assert "CREATE INDEX " not in initial.replace("CREATE INDEX IF NOT EXISTS", "")
    assert "ON CONFLICT (piva) DO NOTHING" in seed
    assert "ON CONFLICT (piva, source, valid_from) DO NOTHING" in seed
    assert "CREATE SCHEMA IF NOT EXISTS list_engine" in initial
    assert "REVOKE ALL ON SCHEMA list_engine FROM PUBLIC" in initial


def test_ingestion_migration_distinguishes_quality_empty_and_failure_states() -> None:
    ingestion = (ROOT / "supabase" / "migrations" / "202607150002_ingestion.sql").read_text()

    declared = set(re.findall(r"CREATE TABLE IF NOT EXISTS ([a-z_]+)", ingestion))
    assert {
        "ingestion_runs",
        "source_record_processing",
        "source_schema_snapshots",
        "data_quality_issues",
    } <= declared
    assert "'empty_verified'" in ingestion
    assert "empty_proof IS NOT NULL" in ingestion
    assert "warning_count integer" in ingestion
    assert "quarantine_replay_guard" in ingestion
    assert "data_quality_issue_replay_guard" in ingestion
    assert "data_quality_issues_append_only" in ingestion
    assert "source_record_processing_append_only" in ingestion
    assert "ON source_records (source, payload_hash)" in ingestion


def test_scoring_migration_scopes_activation_per_segment_and_freezes_scores() -> None:
    scoring = (ROOT / "supabase" / "migrations" / "202607150003_scoring.sql").read_text()

    assert "ADD COLUMN IF NOT EXISTS segment_id" in scoring
    assert "one_active_scoring_version_per_segment" in scoring
    assert "ON scoring_weight_versions (segment_id)" in scoring
    assert "scores_append_only" in scoring
    assert "BEFORE UPDATE OR DELETE ON scores" in scoring
