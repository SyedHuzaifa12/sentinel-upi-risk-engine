-- "Precision" restricted to reviewed cases: share of labeled decisions an
-- analyst confirmed as fraud. NOT the model's true precision -- see
-- DESIGN.md "Review queue and selective labelling bias" and the caveat
-- rendered on the review queue page itself. Served by users_reviewlabel's
-- own primary key scan (small table by construction -- only ever grows by
-- one row per analyst review, never by decision volume).
SELECT
    count(*) AS n_reviewed,
    count(*) FILTER (WHERE disposition = 'CONFIRMED_FRAUD') AS n_confirmed_fraud,
    round(
        100.0 * count(*) FILTER (WHERE disposition = 'CONFIRMED_FRAUD') / NULLIF(count(*), 0), 2
    ) AS precision_on_reviewed_pct
FROM users_reviewlabel;
