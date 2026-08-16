-- p50/p95/p99 scoring latency (decisions.latency_ms), overall and by source
-- (api/worker/replay -- worker latency includes the Redis round trip per
-- feature read, api does not, so mixing them would be misleading). Served
-- by ix_decisions_scored_at if callers add a recency filter; without one
-- this is a full-table PERCENTILE_CONT aggregate by design (a global
-- latency SLO figure, not a windowed one).
SELECT
    source,
    count(*) AS n,
    round(percentile_cont(0.50) WITHIN GROUP (ORDER BY latency_ms)::numeric, 2) AS p50_ms,
    round(percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)::numeric, 2) AS p95_ms,
    round(percentile_cont(0.99) WITHIN GROUP (ORDER BY latency_ms)::numeric, 2) AS p99_ms
FROM decisions
GROUP BY source
ORDER BY source;
