BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE SCHEMA IF NOT EXISTS list_engine;
REVOKE ALL ON SCHEMA list_engine FROM PUBLIC;
ALTER DEFAULT PRIVILEGES IN SCHEMA list_engine REVOKE ALL ON TABLES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES IN SCHEMA list_engine REVOKE ALL ON SEQUENCES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES IN SCHEMA list_engine REVOKE EXECUTE ON FUNCTIONS FROM PUBLIC;
SET LOCAL search_path = list_engine, public;

CREATE OR REPLACE FUNCTION is_valid_italian_vat(value text)
RETURNS boolean
LANGUAGE plpgsql
IMMUTABLE
STRICT
AS $$
DECLARE
    total integer := 0;
    digit integer;
    doubled integer;
    position integer;
BEGIN
    IF value !~ '^[0-9]{11}$' THEN
        RETURN false;
    END IF;

    FOR position IN 1..10 LOOP
        digit := substring(value FROM position FOR 1)::integer;
        IF position % 2 = 1 THEN
            total := total + digit;
        ELSE
            doubled := digit * 2;
            total := total + CASE WHEN doubled > 9 THEN doubled - 9 ELSE doubled END;
        END IF;
    END LOOP;

    RETURN ((10 - (total % 10)) % 10) = substring(value FROM 11 FOR 1)::integer;
END;
$$;

CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION prevent_append_only_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION '% is append-only; mutation is not permitted', TG_TABLE_NAME;
END;
$$;

CREATE TABLE IF NOT EXISTS companies (
    piva text PRIMARY KEY CHECK (is_valid_italian_vat(piva)),
    legal_name text NOT NULL CHECK (length(trim(legal_name)) > 0),
    website text,
    ateco_code text,
    city text,
    province text CHECK (province IS NULL OR province ~ '^[A-Z]{2}$'),
    region text,
    revenue_eur numeric(18, 2) CHECK (revenue_eur IS NULL OR revenue_eur >= 0),
    employees integer CHECK (employees IS NULL OR employees >= 0),
    company_status text,
    source text NOT NULL,
    is_demo boolean NOT NULL DEFAULT false,
    source_observed_at timestamptz NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

DROP TRIGGER IF EXISTS companies_set_updated_at ON companies;
CREATE TRIGGER companies_set_updated_at
BEFORE UPDATE ON companies
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TABLE IF NOT EXISTS company_firmographic_history (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    piva text NOT NULL REFERENCES companies(piva) ON DELETE RESTRICT,
    revenue_eur numeric(18, 2) CHECK (revenue_eur IS NULL OR revenue_eur >= 0),
    employees integer CHECK (employees IS NULL OR employees >= 0),
    company_status text,
    source text NOT NULL,
    valid_from timestamptz NOT NULL,
    valid_to timestamptz,
    recorded_at timestamptz NOT NULL DEFAULT now(),
    CHECK (valid_to IS NULL OR valid_to > valid_from),
    UNIQUE (piva, source, valid_from)
);

CREATE UNIQUE INDEX IF NOT EXISTS one_current_firmographic_snapshot_per_company
ON company_firmographic_history (piva)
WHERE valid_to IS NULL;

CREATE TABLE IF NOT EXISTS source_records (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source text NOT NULL,
    source_record_id text,
    piva text CHECK (piva IS NULL OR is_valid_italian_vat(piva)),
    payload jsonb NOT NULL,
    payload_hash text NOT NULL CHECK (payload_hash ~ '^[a-f0-9]{64}$'),
    observed_at timestamptz,
    ingested_at timestamptz NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS source_records_replay_guard
ON source_records (source, COALESCE(piva, ''), payload_hash);

DROP TRIGGER IF EXISTS source_records_append_only ON source_records;
CREATE TRIGGER source_records_append_only
BEFORE UPDATE OR DELETE ON source_records
FOR EACH ROW EXECUTE FUNCTION prevent_append_only_mutation();

CREATE TABLE IF NOT EXISTS enrichment_log (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    piva text NOT NULL REFERENCES companies(piva) ON DELETE RESTRICT,
    field_name text NOT NULL,
    field_value jsonb,
    source text NOT NULL,
    source_url text,
    confidence numeric(4, 3) NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    verified boolean NOT NULL DEFAULT false,
    cost_eur numeric(12, 6) NOT NULL DEFAULT 0 CHECK (cost_eur >= 0),
    observed_at timestamptz NOT NULL,
    recorded_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS enrichment_log_piva_field_idx
ON enrichment_log (piva, field_name, observed_at DESC);

DROP TRIGGER IF EXISTS enrichment_log_append_only ON enrichment_log;
CREATE TRIGGER enrichment_log_append_only
BEFORE UPDATE OR DELETE ON enrichment_log
FOR EACH ROW EXECUTE FUNCTION prevent_append_only_mutation();

CREATE TABLE IF NOT EXISTS contacts (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    piva text NOT NULL REFERENCES companies(piva) ON DELETE RESTRICT,
    full_name text,
    role_title text,
    email text,
    phone text,
    phone_type text CHECK (phone_type IS NULL OR phone_type IN ('switchboard', 'mobile')),
    source text NOT NULL,
    source_url text,
    verified_at timestamptz,
    legal_basis text,
    rpo_status text NOT NULL DEFAULT 'unknown'
        CHECK (rpo_status IN ('unknown', 'pending', 'clear', 'blocked')),
    rpo_checked_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CHECK (email IS NOT NULL OR phone IS NOT NULL),
    UNIQUE (id, piva)
);

CREATE UNIQUE INDEX IF NOT EXISTS contacts_unique_email
ON contacts (piva, lower(email)) WHERE email IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS contacts_unique_phone
ON contacts (piva, phone) WHERE phone IS NOT NULL;

DROP TRIGGER IF EXISTS contacts_set_updated_at ON contacts;
CREATE TRIGGER contacts_set_updated_at
BEFORE UPDATE ON contacts
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TABLE IF NOT EXISTS signals (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    piva text NOT NULL REFERENCES companies(piva) ON DELETE RESTRICT,
    signal_type text NOT NULL,
    source text NOT NULL,
    source_url text,
    occurred_at timestamptz NOT NULL,
    received_at timestamptz NOT NULL DEFAULT now(),
    is_hot boolean NOT NULL DEFAULT false,
    confidence numeric(4, 3) NOT NULL DEFAULT 1 CHECK (confidence BETWEEN 0 AND 1),
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (piva, signal_type, source, occurred_at)
);

CREATE INDEX IF NOT EXISTS signals_hot_queue_idx
ON signals (received_at, piva) WHERE is_hot;

CREATE TABLE IF NOT EXISTS scoring_weight_versions (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    version text NOT NULL UNIQUE,
    weights jsonb NOT NULL,
    rationale text NOT NULL,
    is_active boolean NOT NULL DEFAULT false,
    valid_from timestamptz NOT NULL DEFAULT now(),
    valid_to timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (valid_to IS NULL OR valid_to > valid_from)
);

CREATE UNIQUE INDEX IF NOT EXISTS one_active_scoring_version
ON scoring_weight_versions (is_active) WHERE is_active;

CREATE TABLE IF NOT EXISTS scores (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    piva text NOT NULL REFERENCES companies(piva) ON DELETE RESTRICT,
    scoring_version text NOT NULL REFERENCES scoring_weight_versions(version),
    fit_score smallint NOT NULL CHECK (fit_score BETWEEN 0 AND 40),
    signal_score smallint NOT NULL CHECK (signal_score BETWEEN 0 AND 40),
    reachability_score smallint NOT NULL CHECK (reachability_score BETWEEN 0 AND 20),
    total_score smallint GENERATED ALWAYS AS
        (fit_score + signal_score + reachability_score) STORED,
    tier text NOT NULL CHECK (tier IN ('T0', 'T1', 'T2', 'T3', 'UNQUALIFIED')),
    inputs jsonb NOT NULL,
    scored_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (piva, scoring_version, scored_at)
);

CREATE INDEX IF NOT EXISTS scores_queue_idx
ON scores (tier, total_score DESC, scored_at DESC);

CREATE TABLE IF NOT EXISTS suppression_list (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    piva text CHECK (piva IS NULL OR is_valid_italian_vat(piva)),
    contact_value text,
    reason text NOT NULL,
    source text NOT NULL,
    active boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now(),
    removed_at timestamptz,
    CHECK (piva IS NOT NULL OR contact_value IS NOT NULL),
    CHECK ((active AND removed_at IS NULL) OR (NOT active))
);

CREATE UNIQUE INDEX IF NOT EXISTS active_company_suppression
ON suppression_list (piva) WHERE active AND piva IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS active_contact_suppression
ON suppression_list (contact_value) WHERE active AND contact_value IS NOT NULL;

CREATE TABLE IF NOT EXISTS campaign_outcomes (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    piva text NOT NULL REFERENCES companies(piva) ON DELETE RESTRICT,
    hubspot_task_id text NOT NULL,
    disposition text NOT NULL CHECK (disposition IN (
        'wrong_number', 'no_answer', 'gatekeeper_stop',
        'conversation', 'demo_booked', 'refusal_opt_out'
    )),
    source_at_send text NOT NULL,
    tier_at_send text NOT NULL CHECK (tier_at_send IN ('T1', 'T2', 'T3')),
    signal_at_send text,
    cost_at_send_eur numeric(12, 6) NOT NULL DEFAULT 0 CHECK (cost_at_send_eur >= 0),
    occurred_at timestamptz NOT NULL,
    ingested_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (hubspot_task_id, disposition, occurred_at)
);

CREATE TABLE IF NOT EXISTS agent_sessions (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    piva text NOT NULL REFERENCES companies(piva) ON DELETE RESTRICT,
    agent_name text NOT NULL,
    provider text NOT NULL,
    model text NOT NULL,
    prompt_version text NOT NULL,
    status text NOT NULL CHECK (status IN ('running', 'succeeded', 'failed', 'review')),
    messages jsonb NOT NULL DEFAULT '[]'::jsonb,
    output jsonb,
    input_tokens integer CHECK (input_tokens IS NULL OR input_tokens >= 0),
    output_tokens integer CHECK (output_tokens IS NULL OR output_tokens >= 0),
    cost_eur numeric(12, 6) NOT NULL DEFAULT 0 CHECK (cost_eur >= 0),
    eval_passed boolean,
    started_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz
);

CREATE TABLE IF NOT EXISTS review_queue (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    piva text REFERENCES companies(piva) ON DELETE RESTRICT,
    reason text NOT NULL,
    payload jsonb NOT NULL,
    status text NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'approved', 'rejected', 'expired')),
    created_at timestamptz NOT NULL DEFAULT now(),
    resolved_at timestamptz,
    resolved_by text
);

CREATE INDEX IF NOT EXISTS review_queue_pending_idx
ON review_queue (created_at) WHERE status = 'pending';

CREATE TABLE IF NOT EXISTS dlq (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_name text NOT NULL,
    step_name text NOT NULL,
    piva text CHECK (piva IS NULL OR is_valid_italian_vat(piva)),
    error_type text NOT NULL,
    error_message text NOT NULL,
    payload jsonb NOT NULL,
    retryable boolean NOT NULL DEFAULT false,
    attempts integer NOT NULL DEFAULT 1 CHECK (attempts > 0),
    status text NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'replaying', 'resolved', 'discarded')),
    failed_at timestamptz NOT NULL DEFAULT now(),
    resolved_at timestamptz
);

CREATE TABLE IF NOT EXISTS idempotency_keys (
    key text PRIMARY KEY,
    scope text NOT NULL,
    request_hash text NOT NULL,
    response jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz
);

CREATE INDEX IF NOT EXISTS idempotency_keys_expiry_idx
ON idempotency_keys (expires_at) WHERE expires_at IS NOT NULL;

CREATE TABLE IF NOT EXISTS quarantine (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source text NOT NULL,
    payload jsonb NOT NULL,
    reasons jsonb NOT NULL,
    severity text NOT NULL CHECK (severity IN ('error', 'warning')),
    status text NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'resolved', 'discarded')),
    created_at timestamptz NOT NULL DEFAULT now(),
    resolved_at timestamptz
);

CREATE TABLE IF NOT EXISTS provider_health (
    provider text PRIMARY KEY,
    state text NOT NULL DEFAULT 'closed' CHECK (state IN ('closed', 'open', 'half_open')),
    consecutive_failures integer NOT NULL DEFAULT 0 CHECK (consecutive_failures >= 0),
    last_status_code integer,
    cooldown_until timestamptz,
    last_success_at timestamptz,
    last_failure_at timestamptz,
    updated_at timestamptz NOT NULL DEFAULT now()
);

DROP TRIGGER IF EXISTS provider_health_set_updated_at ON provider_health;
CREATE TRIGGER provider_health_set_updated_at
BEFORE UPDATE ON provider_health
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TABLE IF NOT EXISTS compliance_checks (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    piva text NOT NULL REFERENCES companies(piva) ON DELETE RESTRICT,
    contact_id uuid,
    check_type text NOT NULL CHECK (check_type IN ('rpo', 'legal_basis', 'suppression')),
    result text NOT NULL CHECK (result IN ('clear', 'blocked', 'unknown')),
    provider text NOT NULL,
    evidence jsonb NOT NULL,
    checked_at timestamptz NOT NULL,
    valid_until timestamptz,
    recorded_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (contact_id, piva) REFERENCES contacts(id, piva) ON DELETE RESTRICT
);

DROP TRIGGER IF EXISTS compliance_checks_append_only ON compliance_checks;
CREATE TRIGGER compliance_checks_append_only
BEFORE UPDATE OR DELETE ON compliance_checks
FOR EACH ROW EXECUTE FUNCTION prevent_append_only_mutation();

CREATE TABLE IF NOT EXISTS audit_log (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    actor text NOT NULL,
    action text NOT NULL,
    object_type text NOT NULL,
    object_id text NOT NULL,
    piva text CHECK (piva IS NULL OR is_valid_italian_vat(piva)),
    details jsonb NOT NULL DEFAULT '{}'::jsonb,
    occurred_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS audit_log_object_idx
ON audit_log (object_type, object_id, occurred_at DESC);

DROP TRIGGER IF EXISTS audit_log_append_only ON audit_log;
CREATE TRIGGER audit_log_append_only
BEFORE UPDATE OR DELETE ON audit_log
FOR EACH ROW EXECUTE FUNCTION prevent_append_only_mutation();

REVOKE ALL ON ALL TABLES IN SCHEMA list_engine FROM PUBLIC;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA list_engine FROM PUBLIC;
REVOKE EXECUTE ON ALL FUNCTIONS IN SCHEMA list_engine FROM PUBLIC;

COMMIT;
