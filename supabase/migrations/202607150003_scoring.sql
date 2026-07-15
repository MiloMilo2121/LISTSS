BEGIN;

SET LOCAL search_path = list_engine, public;

-- M1 could only keep one scoring version active for the entire application. A
-- segment is an independent commercial hypothesis, so activation must be scoped
-- to that segment. Existing rows are backfilled from the canonical
-- ``segment@version`` identifier; pre-M3 identifiers are kept together in a
-- conservative ``legacy`` segment instead of guessing their ownership.
ALTER TABLE scoring_weight_versions
ADD COLUMN IF NOT EXISTS segment_id text;

UPDATE scoring_weight_versions
SET segment_id = CASE
    WHEN version ~ '^[a-z][a-z0-9_-]{1,63}@.+'
        THEN split_part(version, '@', 1)
    ELSE 'legacy'
END
WHERE segment_id IS NULL;

ALTER TABLE scoring_weight_versions
ALTER COLUMN segment_id SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = 'list_engine.scoring_weight_versions'::regclass
          AND conname = 'scoring_weight_versions_segment_id_check'
    ) THEN
        ALTER TABLE scoring_weight_versions
        ADD CONSTRAINT scoring_weight_versions_segment_id_check
        CHECK (segment_id ~ '^[a-z][a-z0-9_-]{1,63}$');
    END IF;
END;
$$;

DROP INDEX IF EXISTS one_active_scoring_version;

CREATE UNIQUE INDEX IF NOT EXISTS one_active_scoring_version_per_segment
ON scoring_weight_versions (segment_id)
WHERE is_active;

CREATE INDEX IF NOT EXISTS scoring_versions_segment_history_idx
ON scoring_weight_versions (segment_id, valid_from DESC);

CREATE INDEX IF NOT EXISTS scores_version_queue_idx
ON scores (scoring_version, tier, total_score DESC, scored_at DESC);

-- Score rows are immutable observations. Version activation is mutable state,
-- but a historical score must only ever be superseded by a later INSERT.
DROP TRIGGER IF EXISTS scores_append_only ON scores;
CREATE TRIGGER scores_append_only
BEFORE UPDATE OR DELETE ON scores
FOR EACH ROW EXECUTE FUNCTION prevent_append_only_mutation();

REVOKE ALL ON TABLE scoring_weight_versions, scores FROM PUBLIC;

COMMIT;
