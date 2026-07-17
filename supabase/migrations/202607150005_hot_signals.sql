BEGIN;

SET LOCAL search_path = list_engine, public;

-- Raw hot-signal landing: append-only, content-addressed. Deliberately no FK to
-- companies, because a fresh signal may arrive for a P.IVA not yet in the hub.
-- This is the exact-bytes replay guard, the analogue of source_records.
CREATE TABLE IF NOT EXISTS hot_signals (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source text NOT NULL,
    piva text NOT NULL CHECK (is_valid_italian_vat(piva)),
    signal_type text NOT NULL,
    observed_at timestamptz NOT NULL,
    received_at timestamptz NOT NULL DEFAULT now(),
    valid_until timestamptz NOT NULL,
    confidence numeric(4, 3) NOT NULL DEFAULT 1 CHECK (confidence BETWEEN 0 AND 1),
    source_url text,
    payload jsonb NOT NULL,
    content_hash text NOT NULL CHECK (content_hash ~ '^[a-f0-9]{64}$'),
    natural_key_hash text NOT NULL CHECK (natural_key_hash ~ '^[a-f0-9]{64}$'),
    ingested_at timestamptz NOT NULL DEFAULT now(),
    CHECK (valid_until > observed_at),
    CHECK (received_at >= observed_at),
    UNIQUE (source, content_hash)
);

CREATE INDEX IF NOT EXISTS hot_signals_piva_idx ON hot_signals (piva, observed_at DESC);
CREATE INDEX IF NOT EXISTS hot_signals_natural_key_idx ON hot_signals (natural_key_hash);

DROP TRIGGER IF EXISTS hot_signals_append_only ON hot_signals;
CREATE TRIGGER hot_signals_append_only
BEFORE UPDATE OR DELETE ON hot_signals
FOR EACH ROW EXECUTE FUNCTION prevent_append_only_mutation();

-- One durable event per distinct real-world signal, carrying a fenced lease.
-- Kept separate from the immutable hot_signals fact: attempts/lease/status are
-- mutable work-queue state (the same separation M4 uses for research leases).
CREATE TABLE IF NOT EXISTS signal_outbox (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    hot_signal_id uuid NOT NULL REFERENCES hot_signals(id) ON DELETE RESTRICT,
    dedupe_key text NOT NULL CHECK (dedupe_key ~ '^[a-f0-9]{64}$'),
    piva text NOT NULL CHECK (is_valid_italian_vat(piva)),
    source text NOT NULL,
    signal_type text NOT NULL,
    signal jsonb NOT NULL,
    status text NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'leased', 'done', 'failed', 'dead')),
    attempts integer NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    owner_token uuid,
    lease_expires_at timestamptz,
    available_at timestamptz NOT NULL DEFAULT now(),
    leased_at timestamptz,
    done_at timestamptz,
    dead_at timestamptz,
    last_error text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (dedupe_key),
    CHECK (status <> 'leased' OR (owner_token IS NOT NULL AND lease_expires_at IS NOT NULL)),
    CHECK (status <> 'dead' OR last_error IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS signal_outbox_claim_idx
ON signal_outbox (available_at, created_at) WHERE status IN ('pending', 'leased');

DROP TRIGGER IF EXISTS signal_outbox_set_updated_at ON signal_outbox;
CREATE TRIGGER signal_outbox_set_updated_at
BEFORE UPDATE ON signal_outbox
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- Internal BDR call task carrying a gated dossier; M6 delivers it to HubSpot.
-- The signal->task latency is stored as generated columns (same idiom as
-- scores.total_score) so "measure it" needs no application arithmetic.
CREATE TABLE IF NOT EXISTS hot_tasks (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    piva text NOT NULL REFERENCES companies(piva) ON DELETE RESTRICT,
    signal_natural_key text NOT NULL CHECK (signal_natural_key ~ '^[a-f0-9]{64}$'),
    source text NOT NULL,
    signal_type text NOT NULL,
    agent_session_id uuid REFERENCES agent_sessions(id) ON DELETE RESTRICT,
    dossier_digest text NOT NULL CHECK (dossier_digest ~ '^[a-f0-9]{64}$'),
    status text NOT NULL DEFAULT 'ready' CHECK (status IN ('ready')),
    signal_observed_at timestamptz NOT NULL,
    signal_received_at timestamptz NOT NULL,
    task_created_at timestamptz NOT NULL,
    signal_to_task_seconds numeric GENERATED ALWAYS AS
        (EXTRACT(EPOCH FROM (task_created_at - signal_received_at))) STORED,
    source_to_task_seconds numeric GENERATED ALWAYS AS
        (EXTRACT(EPOCH FROM (task_created_at - signal_observed_at))) STORED,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CHECK (signal_received_at >= signal_observed_at),
    CHECK (task_created_at >= signal_received_at),
    UNIQUE (signal_natural_key)
);

CREATE INDEX IF NOT EXISTS hot_tasks_piva_idx ON hot_tasks (piva, task_created_at DESC);

DROP TRIGGER IF EXISTS hot_tasks_set_updated_at ON hot_tasks;
CREATE TRIGGER hot_tasks_set_updated_at
BEFORE UPDATE ON hot_tasks
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- The M5 headline KPI: signal->task latency percentiles per source and type.
CREATE OR REPLACE VIEW hot_signal_latency AS
SELECT
    source,
    signal_type,
    count(*) AS tasks,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY signal_to_task_seconds) AS p50_seconds,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY signal_to_task_seconds) AS p95_seconds
FROM hot_tasks
GROUP BY source, signal_type;

REVOKE ALL ON TABLE hot_signals, signal_outbox, hot_tasks FROM PUBLIC;
REVOKE ALL ON hot_signal_latency FROM PUBLIC;

COMMIT;
