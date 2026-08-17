# Interview notes — Sentinel UPI Risk Engine

Personal prep page. Every number here is pulled from `PROGRESS.md`,
`DESIGN.md`, or `ml/artifacts/metrics/metrics_v2.json` — nothing invented.
Where the project's own docs disagree with each other, the discrepancy is
called out rather than silently picking one (see the ablation figure in
section 2).

---

## 1. The 60-second pitch (written to be spoken aloud)

> UPI fraud in India is mostly Authorised Push Payment fraud — the victim is
> socially engineered and willingly sends the money, so device, PIN, and
> location all look completely genuine. That means the card-fraud playbook
> — chargebacks, foreign-transaction flags — doesn't apply at all, and is
> actually a data leak if you try to use it, since a chargeback is only
> known *after* the fraud already happened.
>
> So I built a real-time risk engine that derives its signal entirely from
> behavioural history instead: how this payee has been receiving money, how
> this payer's behaviour is deviating from their own baseline. A raw event
> comes in, features get computed server-side from that payer/payee/device's
> own history — never from the caller — and it routes to one of two LightGBM
> models depending on whether the payee has any transaction history at all.
> The raw score gets isotonic-calibrated into an actual probability, then a
> cost matrix — not a fixed 0.5 cutoff — decides allow, warn, review, or
> block. Every decision is logged immutably with its full feature snapshot,
> and anything above allow gets a real SHAP explanation, not a canned
> sentence.
>
> On synthetic data it lands at 0.8455 combined PR-AUC, which is honestly
> higher than a real system would see — I say that upfront, not when asked.
> The interesting part isn't the score, it's the systems work around it:
> point-in-time correctness enforced at the data-access layer, train/serve
> parity enforced by using the literal same feature function everywhere,
> and a cost-based decision policy that actually surfaces real operational
> tradeoffs — like the fact that at this system's precision, reviewing more
> transactions costs more than the fraud it catches, which is a genuine
> finding a cost matrix is supposed to produce, not a bug to hide.

---

## 2. Key numbers

| Metric | Value | What it means |
|---|---|---|
| Combined PR-AUC (cold+warm test set) | **0.8455** | The headline number, but it's cold+warm *together* — say this explicitly, don't let "PR-AUC 0.8455" get heard as "warm-only." |
| Warm-only PR-AUC | 0.8385 (95% CI 0.80–0.87) | The model used once a payee has ≥3 prior transactions. |
| Cold-only PR-AUC | 0.9111 (95% CI **0.79–1.00**) | Looks better than warm, but on only **22 test positives** — the CI is wide because the sample is small, not because cold fraud is easier. Say this before they ask. |
| Cold-only ROC-AUC | 0.9528 | Don't conflate this with cold's PR-AUC (0.9111) — they're different metrics and it's an easy slip mid-answer. |
| Rules-baseline PR-AUC | 0.1136 | A crude 5-rule score, used as the sanity floor a real model has to clear, not a target. |
| Amount-weighted recall @ chosen operating point | 0.6687 | Share of fraud *value* caught at the deployed thresholds (from a live backtest over 8,000 real logged decisions, not the offline test set). |
| Precision @ chosen operating point | 0.2338 | ~23% of alerts are real fraud. This is the number that makes the threshold-curve story make sense — see section 4. |
| p99 scoring latency | 27.32ms (predict 20.18 + features 1.79 + reason_codes 13.75 [non-ALLOW only] + calibrate 1.09 + decide 0.03) | Measured via `service/benchmark.py`, 1000 real test-set events. Reason-codes' own p99 is dragged up entirely by the non-ALLOW minority — median is 0.00ms because most events are ALLOW and skip SHAP entirely. |
| Net benefit vs. a naive 0.5 cutoff | ~2.14x (−1,270,877 vs. −593,930 expected cost, held-out test set) | "More negative cost" = more net benefit under this project's cost-matrix sign convention — say the ratio, not the raw signed numbers, or it reads backwards. |
| Total features / cold-safe | 51 / 34 | 17 features are warm-only (need payee history that a cold payee doesn't have). |
| Total tests | 73 | 61 across ml/feature_lib/service/worker/decisionlog/pipelines + 12 in backend/users. |
| Python lines | ~9,510 | Excludes venv/migrations/__pycache__. |
| docker-compose services | 6 | postgres, redis, api, worker, ui, generator. |

**Known discrepancy, worth knowing rather than getting caught by**: the
"Decisions (do not revisit)" section of `PROGRESS.md` cites an ablation
study (dropping `days_since_payer_last_paid_payee` and pair-history
features) retaining "100.1% of the full model's PR-AUC" — but that number
is from an *earlier* training run, before a later realism fix changed the
warm PR-AUC from 0.9940 to 0.8385. **The current `metrics_v2.json`'s own
`ablation_no_pair_history` field shows 95.4% retained** (PR-AUC 0.7999 vs.
0.8385) — still strong evidence the model isn't a single-feature crutch,
just a different, more current number than the prose narrative quotes. If
asked "does removing your top feature break the model," the honest answer
is 95.4%, not 100.1%.

---

## 3. Why this, not that

**Redis Streams, not Kafka.** Single-node scale (this system was never
going to run a Kafka cluster), and Redis Streams' consumer groups give the
exact same at-least-once, horizontally-scalable delivery semantics needed —
demonstrated directly by scaling the worker to 3 replicas with zero code
changes. Kafka would be solving a durability/throughput problem this
system's scale doesn't have.

**Hand-rolled Redis/Postgres store, not Feast.** Feast is a real feature
store, but its value is standardizing feature definitions *across many
models and teams*. This project already gets that with one shared
`feature_lib.compute_features()` used by training, the API, and the
worker — a feature-store framework on top would add operational surface
area (its own registry, its own deployment) without solving a problem this
single-codebase system still has.

**Two models (cold/warm), not one.** A payee with no transaction history
structurally cannot have payee-side features computed — they'd all be
missing, not just noisy. Rather than let one model learn to route around
missingness implicitly, cold gets a smaller, explicitly `cold_safe` feature
set and stricter thresholds, because it has less signal to work with and
should be more conservative as a result.

**Cold threshold = 3, not 1.** A payee with 1-2 prior transactions is
technically "seen," but their fan-in/amount-std/P2M-ratio features are
computed from too few observations to be statistically meaningful — so
`is_cold()` still routes them to cold. This is a deliberate choice about
when a feature has *enough* data to trust, not just whether it's
technically computable.

**Isotonic calibration, not Platt/sigmoid scaling.** LightGBM's raw scores
aren't guaranteed to relate to true probability in a simple parametric
(sigmoid) way. Isotonic makes no distributional assumption — it only
enforces monotonicity, which is the one property the decision policy's
threshold comparisons actually need. Fit on the validation split (reused
from LightGBM's own early stopping, a deliberate choice given dataset size
— not a fresh calibration-only split, which would shrink an already-small
cold-model validation set further).

**Greedy sequential threshold sweep, not a joint grid search.** Optimizing
`t_review`, then `t_block` given that, then `t_warn` given both, is ~600
evaluations vs. ~8M for a joint 3D grid at the same resolution. Accepted
because the expected-cost surface is monotone in each threshold given the
others fixed — not revisited without deliberately re-opening that
assumption.

**Plain `pipelines/` + a Makefile, not Airflow/Prefect.** Single node, no
DAG (each script — train, backtest, drift_check — is a straight-line batch
job), no recurring schedule to manage. A scheduler solves fan-out and
time-based orchestration across many interdependent jobs — problems this
system doesn't have yet.

**Django kept as the analyst console, not rewritten in React.** The
product surface here (review queue, monitoring dashboard, a sandbox form)
is server-rendered CRUD and tables — exactly what Django templates are
for. Django is explicitly a *client* of the FastAPI scoring service over
HTTP (Phase 4), not a model host, so it never needed to be more than a
thin console in the first place.

**LightGBM, not a neural network** *(no head-to-head evaluation against
deep learning is documented in this project's history — answering from
why LightGBM fits the actual constraints, not a false "we tried X and
rejected it" story)*: tabular, structured, moderate-cardinality features;
a genuinely small positive-class sample (22-321 positives depending on
split); a sub-30ms p99 latency budget; and a hard requirement for
per-transaction SHAP explanations, which `shap.TreeExplainer` gives
efficiently and exactly for tree ensembles. None of those point toward a
network needing far more data and much more inference latency to reach
parity, let alone beat, a well-tuned gradient-boosted tree on this problem
shape.

---

## 4. Likely questions, with real answers

**How do you prevent train/serve skew?**
One function, `feature_lib.compute_features()`, imported unmodified by
training (`ml/src/train.py`), the API (`service/scoring.py`), and the
worker (`worker/consumer.py`) — three call sites, zero reimplementations.
`feature_lib.frame.vector_to_frame()` is likewise the single shared
"feature values → model-ready DataFrame" step, used by both training and
serving, so column order/dtype casting can't silently diverge either.

**How do you guarantee point-in-time correctness?**
Enforced at the one choke point every feature reads through: each
`HistoryStore` implementation filters on `as_of=event.timestamp` inside its
own query logic, so a feature spec has no way to see a future row — it
only gets what the store's query already excluded the future from.
`compute_features()` also calls `store.record(event)` *after* computing
features, never before, so an event can't see itself in its own history.
Regression-tested in `ml/tests/test_feature_leak.py` and
`test_temporal_split.py`; `assert_temporal_integrity()` additionally
asserts train/val/test day ranges never overlap chronologically — there is
no `random_state` parameter anywhere in the split code.

**Why not accuracy?**
Fraud is ~0.7% of events — a model that predicts "never fraud" gets 99.3%
accuracy and is useless. PR-AUC, precision-at-alert-rate, and
amount-weighted recall are the metrics that actually reflect the
class-imbalanced, cost-asymmetric problem this is.

**Why calibrate at all — why not just rank?**
Because the decision policy needs an actual probability to compare against
a *cost-derived* threshold, not just a ranking. An uncalibrated raw score
can be a perfectly good ranker while being numerically meaningless as "this
transaction is 80% likely to be fraud" — and the whole cost-matrix approach
depends on that number being real.

**How did you choose your thresholds?**
`ml/src/policy/thresholds.py` sweeps `t_review` first, then `t_block` given
that, then `t_warn` given both — a greedy sequential sweep, not a joint
grid search, to minimize expected cost under `cost_matrix.py`'s FN/TP/FP
costs. Cold's thresholds are asserted `>=` warm's at every tier afterward,
with a documented multiplier applied if the sweep doesn't produce that
ordering naturally.

**What happens when the payee has no history?**
`is_cold()` routes it to the cold model, which only uses `cold_safe`
features (payer/device-side, never payee-history-dependent) and gets
stricter thresholds — cold has less signal, so it should be more
conservative, not equally confident with fewer inputs.

**How do you handle label lag?**
Honestly — this project doesn't have real label lag, since the synthetic
generator's ground truth is available instantly. In `DESIGN.md`'s "What I'd
do differently with real data," this is called out explicitly: real fraud
labels arrive weeks later via dispute windows, and only for what got
reported, compounding with the next answer.

**What is selective labelling bias and how does it affect your precision
number?**
Only transactions the model already flagged (REVIEW/BLOCK) ever reach the
analyst review queue. A transaction the model ALLOWed is never reviewed, so
a false negative can never appear in "precision on reviewed cases" — it
says nothing about recall or about the ALLOW tier's true error rate, and
reviewing more of the *same* alerted queue doesn't fix that; only sampling
some ALLOWed traffic for a holdout audit would. This is documented and
enforced with a visible banner on both the review queue page and the
monitoring dashboard, not just in a doc nobody reads.

**Your top feature was 43% of gain — how did you know it wasn't a leak?**
It had already been one once, at 65.5% of gain — a real generator artifact
(all four fraud typologies structurally targeted a payee brand-new to that
payer, so `days_since_payer_last_paid_payee` was a missingness pattern, not
a behavioural signal). After fixing the generator's payer archetypes and
regenerating, it dropped to 42.9%. To confirm this was a real signal and
not a residual leak, I ran an ablation: a warm model trained on all
features *except* pair-history retained 95.4% of the full model's PR-AUC
(0.7999 vs. 0.8385) — proof the model uses diverse signal across its full
feature set, not a single-feature crutch, so the remaining 42.9% is
legitimate importance, not leakage.

**Why is your PR-AUC higher than a real system's?**
The synthetic fraud typologies, even after three rounds of deliberate
realism tuning (amount-overlap coefficient 0.58→0.78, sloppier/imperfect
typology variants, softened mule fan-in), are still more separable than
real fraud. This is a property of the dataset, stated plainly in the README
and DESIGN.md, not a claim about real-world performance.

**How would you detect model degradation in production?**
`pipelines/drift_check.py` — three explicitly separate axes, because each
has a different cause and remedy: feature drift (PSI per feature vs. the
training distribution — retrain if significant), score drift (KS test on
`risk_score` vs. validation-time probabilities — re-tune thresholds), and
concept drift (calibration on reviewed cases only, carrying the same
selective-labelling caveat as above — get more labels). Found and disclosed
a real cold-start artifact doing this: a worker replay starting against an
empty history store produces enormous, implausible feature-drift PSI that
isn't real drift at all — flagged explicitly in the tool's own output, not
silently absorbed.

**What breaks at 10x traffic?**
Single Postgres primary with no read replicas (the monitoring dashboard's
full-table SQL aggregates would start contending with write throughput);
single Redis instance, no cluster/Sentinel; MLflow's local FileStore isn't
built for concurrent writers; `InMemoryHistoryStore` (the zero-dependency
default) would be unusable at that volume — only the Postgres/Redis-backed
stores scale past one process's memory. Stated as reasoned projection in
`DESIGN.md`, not measured — this system has never actually been load-tested
at 10x.

**How would you A/B test a new model?**
Not built, but the backtest harness (`pipelines/backtest.py`) is the
offline half of the answer: it re-scores a historical decision window from
*stored feature snapshots* — never recomputing features — so a candidate
model or threshold set can be compared against what was actually logged,
including a full threshold-vs-net-benefit curve, before ever touching live
traffic. A real A/B would need the online half (shadow-scoring or a
traffic split) on top of that, which isn't built.

**What's your biggest weakness in this project?**
Selective labelling bias, unresolved. The review queue's "precision on
reviewed cases" is the only feedback signal the system has, and it's
structurally biased — no amount of more reviewing fixes it, only a holdout
audit of ALLOWed traffic would, and that's not built. I'd rather name this
than let a reviewed-precision number pass as if it were the model's real
precision.

**Walk me through what happens in the 100ms after a payment request
arrives.**
Event in → `compute_features()` reads the payer/payee/device's history from
the store (as-of the event's own timestamp, ~0.4-1.8ms p99) → routed cold
or warm by payee transaction count → LightGBM `predict_proba` (~7.9-20ms
p99 — the dominant cost) → isotonic calibration (~0.2-1.1ms) → cost-based
decision (~0.01-0.03ms) → if the action isn't ALLOW, SHAP reason codes
(~13.75ms p99, but 0ms median — most events are ALLOW and skip this
entirely) → the full decision is logged immutably with its feature
snapshot → response returned. Total p99: 27.32ms, comfortably under the
100ms budget.

---

## 5. Bugs I found and fixed — 30-second stories

**1. Chargeback/foreign-transaction leakage (the reason this project exists
in its current form).** The original v1 system's RandomForest literally
trained on `DailyChargebackAvgAmt`, `Six_MonthChbkFreq`,
`isForeignTransaction`, and `isHighRiskCountry` as input features. All four
are structurally impossible for a real-time UPI system — a chargeback is
only known *after* a dispute, and none of those fields exist on the UPI
rail at all. Caught during the v1→v2 architectural rewrite, not a small
patch: it's why CLAUDE.md now has a hard rule forbidding them, and why
`DESIGN.md` opens with the reasoning instead of burying it.

**2. The "payee new to payer" missingness artifact (top feature at 65.5% of
gain).** All four synthetic fraud typologies structurally targeted a payee
brand-new to that specific payer. `days_since_payer_last_paid_payee` is
NaN exactly when that's true, and only ~9% of *legitimate* events were also
first-time payments — so the model had learned a missingness pattern from
the generator's own construction, not a behavioural signal. Caught by a
40%-dominance safety guard that halted training before saving any model
artifact — by design, not luck. Fixed by splitting payers into habitual
(~7% new-payee rate) vs. exploratory (~55%) archetypes in the generator,
which brought the feature down to 42.9% and, independently, verified via
an ablation (95.4% PR-AUC retained without it) that it wasn't still riding
on a residual leak.

**3. NaN feature values got silently rejected by Postgres, not silently
dropped.** `feature_snapshot` legitimately contains NaN for any feature a
payer/payee doesn't have enough history for yet — the common case, not an
edge case. `json.dumps(float('nan'))` emits the non-standard `NaN` token,
which Postgres's JSONB correctly refuses
(`psycopg.errors.InvalidTextRepresentation`). Found by live integration
tests actually hitting the real worker path, not a unit test with a
hand-picked "nice" feature vector — every single message failed until this
was fixed by recursively replacing NaN/Infinity with `None` before
serializing.

**4. Two independently-correct implementations disagreed at the 15th
significant digit.** Running the cross-store parity test for real (not
skipped) against a live Postgres container surfaced
`InMemoryHistoryStore` computing `0.8922178162191943` for a variance
feature where `PostgresHistoryStore` computed `...191938` — a ~5e-16
relative difference. Diagnosed as IEEE-754 floating-point
non-associativity (Python's incremental running-sum-of-squares vs. SQL's
`STDDEV_POP()` aggregate, same formula, different summation order), not a
logic bug. Fixed the *test*, not the code: switched float comparison to
`math.isclose(rel_tol=1e-9)` — seven orders of magnitude looser than the
observed noise floor and seven orders tighter than anything that could move
a calibrated probability across a decision threshold. Bools/NaN/missingness
still require exact match.

**5. A reason-code message literally said "nan days since this payer last
paid this payee."** Found by eye while capturing screenshots for the
README — a NaN-valued base feature can still be the top SHAP contributor
(LightGBM routes NaN through a learned split, so it's real, rankable
signal), but its own f-string template formatted the raw float directly.
Its `_is_missing` sibling feature already had the right wording; the fix
was falling back to that same "could not be reliably computed" message
whenever the *value* being formatted is NaN, regardless of which feature
SHAP ranked higher. Added a regression test asserting no reason-code
message ever contains the literal string "nan," for either model.

**6. Nearly destroyed a populated database by pointing a test env var at
the wrong instance.** Running the local test suite with `TEST_DATABASE_URL`
pointed at the same live Postgres database holding a real 8,000-row replay
— because no separate test database existed locally yet — triggered
`decisionlog/tests`' and `worker/tests`' fixtures, both of which
unconditionally `TRUNCATE decisions` against whatever that variable
resolves to. It worked exactly as designed against the wrong target and
wiped the table down to 1 row. Recovered by creating a dedicated
`sentinel_test` database and re-running the replay; the incident is now a
standing rule in `PROGRESS.md`'s "Decisions" section: that variable must
never point at live data, and CI is unaffected since its Postgres service
container is always disposable.

---

## 6. What I'd do with real data and six months

- **Cross-institution mule intelligence.** The strongest payee-side signal
  here (fan-in velocity) is entirely local to this system's own transaction
  history. Real mule networks receive from many *different* banks/PSPs —
  detecting that needs shared intelligence across institutions, which no
  single bank's data can provide alone.
- **Genuine label feedback with real dispute lag.** Not instant synthetic
  ground truth — labels that arrive weeks later, only for what got
  reported, modeled honestly as a lagging, partial signal rather than
  assumed complete.
- **An ALLOW-tier holdout audit**, to finally get an unbiased read on true
  precision/recall instead of the structurally biased "precision on
  reviewed cases" this system is stuck with today.
- **Deeper device fingerprinting.** `device_id` here is a flat identifier;
  real systems fingerprint OS version, sensor drift, emulator detection —
  catching fraud rings that reuse device profiles across many stolen
  identities, which a bare device ID alone misses entirely.
- **Load-test the 10x-traffic projections in DESIGN.md** instead of leaving
  them as reasoned-but-unmeasured — Postgres read replicas, Redis
  clustering, and an MLflow tracking server would all move from "probably
  needed" to "measured and either needed or not."
