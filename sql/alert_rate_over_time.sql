-- Alert rate (share of decisions that are NOT ALLOW) per calendar day, over
-- the whole history in `decisions`. Served by ix_decisions_scored_at (the
-- date_trunc('day', scored_at) grouping scans in scored_at order).
SELECT
    date_trunc('day', scored_at) AS day,
    count(*) AS total,
    count(*) FILTER (WHERE action != 'ALLOW') AS alerts,
    round(
        100.0 * count(*) FILTER (WHERE action != 'ALLOW') / count(*), 2
    ) AS alert_rate_pct
FROM decisions
GROUP BY 1
ORDER BY 1;
