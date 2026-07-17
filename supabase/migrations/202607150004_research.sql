BEGIN;

SET LOCAL search_path = list_engine, public;

-- The SDK session identifier is ephemeral. The bounded input, structured output,
-- evaluation and cost stored here are the durable audit record.
ALTER TABLE agent_sessions
ADD COLUMN IF NOT EXISTS input_hash text;

ALTER TABLE agent_sessions
ADD COLUMN IF NOT EXISTS evaluation jsonb;

ALTER TABLE agent_sessions
ADD COLUMN IF NOT EXISTS cost_usd numeric(12, 6);

ALTER TABLE agent_sessions
ADD COLUMN IF NOT EXISTS requested_model text;

ALTER TABLE agent_sessions
ADD COLUMN IF NOT EXISTS lease_owner_token uuid;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = 'list_engine.agent_sessions'::regclass
          AND conname = 'agent_sessions_input_hash_check'
    ) THEN
        ALTER TABLE agent_sessions
        ADD CONSTRAINT agent_sessions_input_hash_check
        CHECK (input_hash IS NULL OR input_hash ~ '^[a-f0-9]{64}$');
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = 'list_engine.agent_sessions'::regclass
          AND conname = 'agent_sessions_cost_usd_check'
    ) THEN
        ALTER TABLE agent_sessions
        ADD CONSTRAINT agent_sessions_cost_usd_check
        CHECK (cost_usd IS NULL OR cost_usd >= 0);
    END IF;

    -- A superseded SDK call is still an immutable, billable audit event, but it
    -- is never eligible for cache or downstream delivery.
    ALTER TABLE agent_sessions
    DROP CONSTRAINT IF EXISTS agent_sessions_status_check;

    ALTER TABLE agent_sessions
    ADD CONSTRAINT agent_sessions_status_check
    CHECK (status IN ('running', 'succeeded', 'failed', 'review', 'superseded'));

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = 'list_engine.agent_sessions'::regclass
          AND conname = 'agent_sessions_research_identity_check'
    ) THEN
        ALTER TABLE agent_sessions
        ADD CONSTRAINT agent_sessions_research_identity_check
        CHECK (
            agent_name <> 'research_dossier'
            OR (requested_model IS NOT NULL AND lease_owner_token IS NOT NULL)
        );
    END IF;
END;
$$;

CREATE INDEX IF NOT EXISTS agent_sessions_research_history_idx
ON agent_sessions (piva, agent_name, completed_at DESC);

CREATE OR REPLACE FUNCTION protect_completed_agent_session()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'DELETE' OR OLD.completed_at IS NOT NULL THEN
        RAISE EXCEPTION 'completed agent sessions are immutable';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS agent_sessions_completed_immutable ON agent_sessions;
CREATE TRIGGER agent_sessions_completed_immutable
BEFORE UPDATE OR DELETE ON agent_sessions
FOR EACH ROW EXECUTE FUNCTION protect_completed_agent_session();

-- Cache rows are explicitly derived state. They may be refreshed after expiry;
-- every generation still remains in agent_sessions for audit and regression work.
CREATE TABLE IF NOT EXISTS research_cache (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_session_id uuid NOT NULL REFERENCES agent_sessions(id) ON DELETE RESTRICT,
    piva text NOT NULL REFERENCES companies(piva) ON DELETE RESTRICT,
    input_hash text NOT NULL CHECK (input_hash ~ '^[a-f0-9]{64}$'),
    provider text NOT NULL,
    model text NOT NULL,
    prompt_version text NOT NULL,
    generation jsonb NOT NULL,
    evaluation jsonb NOT NULL CHECK (evaluation @> '{"passed": true}'::jsonb),
    created_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz NOT NULL,
    CHECK (expires_at > created_at),
    UNIQUE (input_hash, provider, model, prompt_version)
);

CREATE INDEX IF NOT EXISTS research_cache_lookup_idx
ON research_cache (piva, expires_at DESC);

CREATE INDEX IF NOT EXISTS research_cache_session_idx
ON research_cache (agent_session_id);

-- A short fenced lease prevents cold cron and hot-event workers from publishing
-- the same generation concurrently. The agent deadline is shorter than this TTL;
-- no database transaction remains open while the external model runs.
CREATE TABLE IF NOT EXISTS research_generation_leases (
    input_hash text NOT NULL CHECK (input_hash ~ '^[a-f0-9]{64}$'),
    provider text NOT NULL,
    model text NOT NULL,
    prompt_version text NOT NULL,
    piva text NOT NULL REFERENCES companies(piva) ON DELETE RESTRICT,
    owner_token uuid NOT NULL,
    acquired_at timestamptz NOT NULL,
    expires_at timestamptz NOT NULL,
    CHECK (expires_at > acquired_at),
    PRIMARY KEY (input_hash, provider, model, prompt_version)
);

CREATE INDEX IF NOT EXISTS research_generation_leases_expiry_idx
ON research_generation_leases (expires_at);

REVOKE ALL ON TABLE agent_sessions, research_cache, research_generation_leases FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION protect_completed_agent_session() FROM PUBLIC;

COMMIT;
