# DESIGN

Design notes that don't fit naturally in code comments or PROGRESS.md's
phase-by-phase log -- things a future reader needs explained once, in one
place, rather than re-derived from scattered docstrings.

## Problem framing: why card-fraud features don't apply

UPI fraud is overwhelmingly Authorised Push Payment (APP) fraud -- the victim
is socially engineered (fake customer support call, QR code swap, a "collect
request" disguised as a refund) and *willingly* authorises the payment. That
means device fingerprint, PIN entry, and geolocation are all genuine: the
real account holder, on their real device, in their real location, chose to
send the money. This is structurally different from card fraud, where the
signal is a *stolen* credential being used by someone who isn't the account
holder.

Two consequences drive this system's design, both stated as hard rules in
CLAUDE.md:
- **Chargebacks, foreign-transaction flags, and high-risk-country lists are
  forbidden features.** They don't exist on the UPI rail (there's no
  chargeback mechanism, and "foreign transaction" doesn't map onto a
  domestic instant-payments network), and even where an analogous field
  existed, it would be **post-outcome leakage** -- a signal that only exists
  *because* the fraud already happened (a chargeback is filed after the
  victim disputes a payment that already succeeded), not something knowable
  at decision time.
- **Detection has to come from behavioural deviation, not credential
  verification.** Two families carry the actual signal: **payee-side
  behaviour** (mule fan-in -- one account suddenly receiving from many
  distinct payers in a short window; account age; pass-through velocity --
  money arriving and leaving again fast) and **payer-side deviation** (first-
  ever payment to this VPA; amount z-score vs. this payer's own history;
  collect-request-initiated payments, which is how several APP scams solicit
  money without the payer typing an amount themselves).

## Event schema

`feature_lib/event.py`'s `UPIEvent` (a pydantic `BaseModel`) is the canonical
shape -- exactly the fields a payment app actually knows at request time, no
aggregates, no history, no forbidden fields:

```python
txn_id: str
timestamp: datetime
payer_vpa: str
payee_vpa: str
amount: float
txn_type: Literal["P2P", "P2M"]
initiation_mode: Literal["SCAN_QR", "INTENT", "COLLECT_REQUEST", "CONTACT"]
device_id: str
payer_bank: str
payee_bank: str
payer_account_age_days: int
label_is_fraud: bool
label_typology: Optional[str] = None
```

`feature_lib/event.py` has no dependency on `ml/` or `backend/` -- it's the
shared contract between the synthetic generator, training, the FastAPI
serving layer, and the stream worker. `label_is_fraud`/`label_typology` exist
on the type because the synthetic generator needs to carry ground truth
through the pipeline, but nothing in the live serving path (`service/`,
`worker/consumer.py`'s call into `score_event`) ever reads them for a
decision -- they're for training/backtesting only, never an input feature
(that would be the most literal form of the leakage this system is designed
to avoid).

## Feature definitions and time windows

All 51 features (`feature_lib/registry.py`'s `ALL_FEATURES`) are grouped by
what they're computed *from*, each with an explicit window bucket baked into
the name: `_1h`/`_24h`/`_30d` suffixes are real sliding windows evaluated
as-of the event's own timestamp (`payer_txn_count_1h/24h/30d`,
`payee_distinct_payers_1h/24h`, `device_distinct_payers_30d`, ...), not
calendar buckets. There is deliberately no single canonical window -- fan-in
scams show up fast (1h), a payer's typical amount is more stable over 30d,
and mixing those windows is what lets both a burst-detection feature and a
baseline-deviation feature exist side by side without duplicating logic.
34 of the 51 are `cold_safe` (computable without any payee history -- payer-
side, device-side, and event-intrinsic features); the remaining 17
(`payee_*` history features, pair-history features) are warm-only by
construction, since they need payee transaction history that a cold payee
doesn't have yet.

## Point-in-time correctness, enforced in code

CLAUDE.md's hard rule -- "features for an event at time t use ONLY rows with
timestamp < t" -- isn't just a convention here, it's enforced at the single
choke point every feature reads through: each `HistoryStore` implementation
(`feature_lib/store/{in_memory,postgres,redis_store}.py`) filters on
`as_of=event.timestamp` inside its own query logic, so a feature spec
(`feature_lib/registry.py`'s `FeatureSpec.compute`) has no way to accidentally
see a future row -- it only ever gets what the store's query returned, and
that query already excluded anything at or after `t`. `compute_features()`
additionally calls `store.record(event)` **after** computing features, never
before (`service/scoring.py:158`'s comment: "AFTER features are computed --
never before"), so an event can never see itself in its own history either.
`ml/tests/test_feature_leak.py` and `ml/tests/test_temporal_split.py` are the
regression tests for this; `ml/src/training/temporal_split.py`'s
`assert_temporal_integrity()` additionally asserts train/val/test day ranges
never overlap chronologically -- there is no `random_state` parameter
anywhere in that module, structurally nothing to pass one to.

## Cold/warm split and why threshold=3

A payee with 1-2 prior transactions is technically "seen," but its payee-
history features (fan-in, amount-std, P2M ratio) are computed from too few
observations to be statistically meaningful — so `feature_lib/compute.py`'s
`is_cold()` still routes it to the cold model, not just a literal
first-ever-payment check:

```python
COLD_PAYEE_TXN_THRESHOLD = 3

def is_cold(event, store, threshold=COLD_PAYEE_TXN_THRESHOLD):
    return store.payee_txn_count(event.payee_vpa, as_of=event.timestamp) < threshold
```

The threshold is overridable per call (useful for experimentation), but the
registered cold/warm split always uses this default at both training and
serving time, so a `decisions.model_version` string always resolves to the
model that was actually trained on the same routing rule that scored it.
Cold gets a **stricter** threshold set (see cost matrix below) precisely
because it has less signal to work with -- a cold decision is closer to a
prior than warm's is.

## Calibration: isotonic, on the validation slice

Both cold and warm models fit an `IsotonicRegression` calibrator
(`ml/src/training/lightgbm_pipeline.py::fit_calibrator`) on the **validation**
split's raw predicted probabilities, not the training split's. This is a
deliberate, documented choice (see PROGRESS.md's Phase 2 notes) rather than
an oversight: the validation slice is reused for both LightGBM's early
stopping *and* isotonic calibration, given the dataset's size (calibrating on
train data would just refit the training distribution's own miscalibration,
telling you nothing new; a three-way split reserved purely for calibration
would shrink an already-small cold-model validation set further). Isotonic
over Platt/sigmoid scaling because LightGBM's raw scores aren't guaranteed to
be monotonically related to true probability in a simple parametric way, and
isotonic makes no distributional assumption -- it just enforces monotonicity,
which is the one property actually needed for `ml/src/policy/decide.py`'s
threshold comparisons to be meaningful.

## Cost matrix and threshold derivation

`ml/src/policy/cost_matrix.py` defines the four costs a decision can incur:
`FN_COST` (missed fraud, scales with amount), `TP_BENEFIT` (fraud caught,
scales with amount), and flat `FP_COST_FRICTION`/`FP_COST_REVIEW`/
`FP_COST_BLOCK` (a false alarm costs more the more it inconveniences a
legitimate payer -- WARN's friction is cheap, an analyst REVIEW costs analyst
time, a wrongful BLOCK is the most expensive false positive). `ml/src/policy/
thresholds.py` derives `t_warn`/`t_review`/`t_block` via a **sequential
(greedy) sweep** -- optimize `t_review` first, then `t_block` given that
`t_review`, then `t_warn` given both -- rather than an exhaustive joint grid
search over all three simultaneously. This is a tractability decision
(~600 evaluations vs. ~8M for a joint 3D grid at the same resolution),
accepted as sufficient because the expected-cost surface is monotone in each
threshold given the others held fixed (see PROGRESS.md's Phase 3 "Decisions"
entry) -- not revisited without deliberately re-opening that assumption.
Cold's thresholds are asserted `>=` warm's at every tier after the sweep,
with a documented multiplier applied if the sequential sweep doesn't produce
that ordering naturally on its own.

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

## Known limitations of synthetic data

Reused verbatim from PROGRESS.md -- the honesty artifact for this project,
not softened for either audience:

- **Combined PR-AUC 0.8455 (warm-only 0.8385, cold-only 0.9111) is higher than
  real payment-fraud systems typically see (0.3-0.7)**, because the encoded
  typologies — even after the realism fixes below — are still more separable
  than real fraud. This is a property of the synthetic data, not a claim about
  how this model would perform on real UPI traffic.
- **Amount overlap coefficient 0.78**; rules baseline PR-AUC 0.1136 with
  amount-weighted recall@1% 0.5395 — used throughout as the sanity floor a real
  model must clear, not as a target to hit exactly.
- **QR_SWAP recall 0.82 vs a design target below 0.30.** The lookalike payee is
  seeded with background transaction history specifically so it isn't a virgin
  brand-new account, but it remains more detectable than the "near-invisible by
  construction" intent from Phase 0's DATA_CARD.md.
- **Three tuning iterations were run** (amount overlap between fraud/legit
  distributions, sloppy/imperfect typology variants, mule fan-in realism) before
  accepting the current state rather than continuing an open-ended parameter search.

## What breaks at 10x traffic

Reasoned projection, not measured -- this system has never been load-tested
at that volume:

- **Single Postgres primary, no read replicas.** `decisions` is append-only
  and every write goes through one instance; at 10x today's replay rate the
  monitoring dashboard's raw-SQL queries (full-table aggregates in `sql/*.sql`)
  would start contending with write throughput on the same instance. The fix
  is a read replica for `sql/` queries, not a rewrite -- none of those queries
  need transactional consistency with the latest write.
- **Single Redis instance, no cluster/Sentinel.** Both the online feature
  store (`RedisHistoryStore`) and the Streams consumer group depend on one
  process; it's a single point of failure and a single point of throughput
  ceiling. Consumer-group scaling (`docker compose up -d --scale worker=N`,
  demonstrated in Phase 6) scales *consumption*, not the Redis instance
  itself.
- **Worker scaling is real but bounded by Postgres write throughput**, not by
  Redis or the workers' own CPU -- Phase 6's 3-replica replay measured ~16
  events/sec aggregate against a single Postgres instance already showing
  contention from concurrent local test/build activity; the actual ceiling
  under dedicated hardware hasn't been isolated and measured.
- **MLflow's local FileStore isn't built for concurrent writers.** It's the
  right choice for this project's single-operator scale (no server process,
  simplest possible setup — see "MLflow" in PROGRESS.md's Phase 6 section),
  but multiple training/backtest processes writing runs concurrently at 10x
  scale would need a real tracking server or a database-backed store
  (`sqlite:///` at minimum, ideally a shared Postgres backend) instead.
- **`InMemoryHistoryStore` would be unusable at that volume** -- it's the
  zero-dependency default for training/local dev, not a serving-path option;
  only `PostgresHistoryStore`/`RedisHistoryStore` (both already the ones
  actually used by `service/`/`worker/`) scale past a single process's memory.
