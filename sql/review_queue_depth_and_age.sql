-- Review queue depth (REVIEW/BLOCK decisions with no analyst label yet) and
-- the median age of those pending items. Served by ix_decisions_action (the
-- action IN (...) filter) combined with a NOT EXISTS anti-join against
-- users_reviewlabel (small table, its own decision_id index from Django's
-- FK -- db_constraint=False only skips the FK *constraint*, Django still
-- indexes the column for lookups).
SELECT
    count(*) AS queue_depth,
    round(
        percentile_cont(0.50) WITHIN GROUP (
            ORDER BY EXTRACT(EPOCH FROM (now() - scored_at))
        )::numeric, 0
    ) AS median_age_seconds
FROM decisions d
WHERE d.action IN ('REVIEW', 'BLOCK')
  AND NOT EXISTS (
      SELECT 1 FROM users_reviewlabel rl WHERE rl.decision_id = d.id
  );
