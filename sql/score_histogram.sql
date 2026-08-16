-- risk_score histogram: 20 equal-width buckets in [0, 1]. Served by
-- ix_decisions_risk_score (width_bucket's implicit range scan benefits from
-- the index even though this reads the full distribution, since Postgres
-- can use it to avoid a full heap scan when combined with other filters
-- callers may add, e.g. a scored_at window).
SELECT
    width_bucket(risk_score, 0, 1, 20) AS bucket,
    round((width_bucket(risk_score, 0, 1, 20) - 1) * 0.05, 2) AS bucket_start,
    count(*) AS n
FROM decisions
GROUP BY 1
ORDER BY 1;
