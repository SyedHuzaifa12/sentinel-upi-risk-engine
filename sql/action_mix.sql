-- Overall action mix (ALLOW/WARN/REVIEW/BLOCK counts + share of total) across
-- all logged decisions. Served by ix_decisions_action (the GROUP BY action
-- reads directly off that index rather than scanning the whole table).
SELECT
    action,
    count(*) AS n,
    round(100.0 * count(*) / sum(count(*)) OVER (), 2) AS pct
FROM decisions
GROUP BY action
ORDER BY n DESC;
