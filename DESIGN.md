# DESIGN

Design notes that don't fit naturally in code comments or PROGRESS.md's
phase-by-phase log -- things a future reader needs explained once, in one
place, rather than re-derived from scattered docstrings.

## Review queue and selective labelling bias

The analyst review queue (`/review/`, `backend/users/views/review.py`) lists
`decisions` rows with `action IN ('REVIEW', 'BLOCK')` and lets an analyst
attach a disposition -- `CONFIRMED_FRAUD`, `LEGIT`, or `UNCLEAR` -- as a new
`ReviewLabel` row. The dashboard and the queue page both surface a
"precision on reviewed cases" figure: the share of labeled decisions an
analyst confirmed as fraud.

**This number is not the model's precision, and it must never be reported as
such.** The mechanism is selective labelling bias, not a data quality
problem, and no amount of additional reviewing fixes it on its own:

- Only transactions the model already flagged (REVIEW/BLOCK) ever reach the
  queue. A transaction the model ALLOWed is never reviewed, so a false
  negative -- fraud the model missed entirely -- can never appear in this
  metric. "Precision on reviewed cases" says nothing about recall, and
  nothing about the ALLOW tier's real error rate.
- Reviewing is itself non-random: analysts tend to review higher-risk-score
  or higher-amount cases first, so even within the alerted population the
  reviewed subset is not a representative sample of it.
- The result is a metric that can look arbitrarily good or bad depending on
  which alerted cases happen to get reviewed, independent of how the model
  is actually performing on the traffic it never flagged.

**The only way to get an unbiased read on true precision/recall is to also
sample and review some ALLOWed traffic** (a holdout audit), not to review
more of the alerted queue. This system does not currently do that; it is a
known, accepted gap for a synthetic-data proof of concept, not an oversight.

**Where this caveat is enforced, not just documented:**
- The review queue page (`backend/users/templates/app/review_queue.html`)
  renders a visible banner next to the precision figure.
- The monitoring dashboard (`backend/users/templates/app/monitoring.html`)
  renders the same caveat next to its own reviewed-case precision figure.
- `pipelines/drift_check.py`'s "concept drift on reviewed cases" section
  carries the identical caveat in both its JSON output and printed report --
  calibration drift computed against `review_labels` inherits exactly the
  same bias, for exactly the same reason.
- `sql/precision_on_reviewed_cases.sql` and
  `sql/review_queue_depth_and_age.sql` state it in their header comments.

If a future phase adds an ALLOW-tier audit sample, this section should be
updated to describe how that sample is drawn and how it changes (or doesn't)
the interpretation of the existing reviewed-cases figure -- it should not
just be quietly retired.

## Cold-start artifact in feature drift (Phase 6)

`pipelines/drift_check.py`'s PSI-per-feature comparison can show enormous,
implausible drift on history-dependent features (`payer_txn_count_30d`,
`payee_age_hours_in_system`, pair-history counts, ...) whenever the "recent"
decisions came from a worker replay that started against an **empty**
history store -- exactly what Phase 6's capped 8,000-event replay did.
Training's `build_feature_dataframe` computes features in one continuous
pass warmed by the full 90 days of history; a fresh replay has no such
warm-up, so counts/ages/rates read artificially low or zero at first. This
is a replay-seeding artifact, not evidence of real population drift, until
the store has been running long enough to warm back up. `drift_check.py`
surfaces this explicitly as `feature_drift.cold_start_caveat` in its JSON
output, in its printed report, and on the monitoring dashboard -- it is not
something to "fix" by tuning PSI thresholds; it goes away on its own once
the system has run against continuous live traffic for a comparable span to
what the model was trained on.
