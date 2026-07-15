BEGIN;

SET LOCAL search_path = list_engine, public;

-- Raw evidence is content-addressed independently from any parser projection. A
-- parser fix may discover a P.IVA that was unavailable during the first read, so
-- including piva in this key would create a second copy of the same evidence.
DROP INDEX IF EXISTS source_records_replay_guard;
CREATE UNIQUE INDEX IF NOT EXISTS source_records_replay_guard
ON source_records (source, payload_hash);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source text NOT NULL,
    mode text NOT NULL CHECK (mode IN ('batch', 'event')),
    status text NOT NULL CHECK (status IN (
        'running', 'succeeded', 'empty_verified', 'failed'
    )),
    schema_hash text CHECK (schema_hash IS NULL OR schema_hash ~ '^[a-f0-9]{64}$'),
    records_seen integer NOT NULL DEFAULT 0 CHECK (records_seen >= 0),
    records_created integer NOT NULL DEFAULT 0 CHECK (records_created >= 0),
    records_updated integer NOT NULL DEFAULT 0 CHECK (records_updated >= 0),
    records_unchanged integer NOT NULL DEFAULT 0 CHECK (records_unchanged >= 0),
    records_quarantined integer NOT NULL DEFAULT 0 CHECK (records_quarantined >= 0),
    records_replayed integer NOT NULL DEFAULT 0 CHECK (records_replayed >= 0),
    warning_count integer NOT NULL DEFAULT 0 CHECK (warning_count >= 0),
    empty_proof jsonb,
    error_summary text,
    started_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz,
    CHECK (status <> 'empty_verified' OR empty_proof IS NOT NULL),
    CHECK (status = 'running' OR completed_at IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS ingestion_runs_source_started_idx
ON ingestion_runs (source, started_at DESC);

CREATE TABLE IF NOT EXISTS source_schema_snapshots (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source text NOT NULL,
    schema_hash text NOT NULL CHECK (schema_hash ~ '^[a-f0-9]{64}$'),
    field_names jsonb NOT NULL,
    required_fields jsonb NOT NULL,
    drift_status text NOT NULL CHECK (drift_status IN ('expected', 'warning', 'breaking')),
    first_seen_at timestamptz NOT NULL DEFAULT now(),
    last_seen_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (source, schema_hash)
);

ALTER TABLE quarantine ADD COLUMN IF NOT EXISTS record_key text;
ALTER TABLE quarantine ADD COLUMN IF NOT EXISTS payload_hash text;

CREATE UNIQUE INDEX IF NOT EXISTS quarantine_replay_guard
ON quarantine (source, COALESCE(record_key, ''), COALESCE(payload_hash, ''));

CREATE TABLE IF NOT EXISTS data_quality_issues (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_record_id uuid REFERENCES source_records(id) ON DELETE RESTRICT,
    source text NOT NULL,
    record_key text NOT NULL,
    severity text NOT NULL CHECK (severity IN ('error', 'warning')),
    code text NOT NULL,
    field_name text,
    message text NOT NULL,
    details jsonb NOT NULL DEFAULT '{}'::jsonb,
    observed_at timestamptz NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS data_quality_issue_replay_guard
ON data_quality_issues (
    COALESCE(source_record_id::text, ''), source, record_key, code, COALESCE(field_name, '')
);

CREATE TABLE IF NOT EXISTS source_record_processing (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_record_id uuid NOT NULL REFERENCES source_records(id) ON DELETE RESTRICT,
    contract_hash text NOT NULL CHECK (contract_hash ~ '^[a-f0-9]{64}$'),
    processing_hash text NOT NULL CHECK (processing_hash ~ '^[a-f0-9]{64}$'),
    outcome text NOT NULL CHECK (outcome IN (
        'created', 'updated', 'unchanged', 'quarantined'
    )),
    quality_report jsonb NOT NULL,
    company_projection jsonb,
    processed_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (source_record_id, processing_hash)
);

DROP TRIGGER IF EXISTS data_quality_issues_append_only ON data_quality_issues;
CREATE TRIGGER data_quality_issues_append_only
BEFORE UPDATE OR DELETE ON data_quality_issues
FOR EACH ROW EXECUTE FUNCTION prevent_append_only_mutation();

DROP TRIGGER IF EXISTS source_record_processing_append_only ON source_record_processing;
CREATE TRIGGER source_record_processing_append_only
BEFORE UPDATE OR DELETE ON source_record_processing
FOR EACH ROW EXECUTE FUNCTION prevent_append_only_mutation();

REVOKE ALL ON ALL TABLES IN SCHEMA list_engine FROM PUBLIC;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA list_engine FROM PUBLIC;

COMMIT;
