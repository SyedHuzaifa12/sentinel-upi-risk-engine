# UPI Transaction Risk Engine (v2)

## What this is
Event-driven real-time risk scoring for UPI payments. A raw payment event arrives;
the system derives features from its OWN transaction history and returns a decision:
ALLOW / WARN / REVIEW / BLOCK. Data is SYNTHETIC and must always be labelled as such.

## Hard rules — never violate
- Features are NEVER supplied by a user. They are computed server-side from history.
- FORBIDDEN features: chargebacks, foreign transaction, high-risk country.
  Reason: post-outcome leakage, and these do not exist on the UPI rail.
- Point-in-time correctness: features for an event at time t use ONLY rows with
  timestamp < t. Never use future data.
- Temporal train/val/test splits only. NEVER random_state splits.
- ONE feature implementation in feature_lib/, imported by training, worker, and API.
  Never duplicate feature logic anywhere.
- Two models: COLD (payee unseen) and WARM (payee has history). Cold uses only
  cold-safe features and gets stricter thresholds.
- Metrics: PR-AUC, precision@alert-rate, amount-weighted recall. NEVER accuracy.
- Every decision is logged immutably with feature snapshot + model version.
- No hardcoded explanation text. Reason codes are generated per transaction.

## Stack
FastAPI (serving) · LightGBM + isotonic calibration · Redis (online store + streams) ·
Postgres · Django (analyst console only, calls the API over HTTP) · MLflow ·
Docker Compose · GitHub Actions

## Working style
- Propose a plan first. Wait for my approval before writing code.
- Small commits, one concern each.
- Tests alongside code, not after.
- Do not touch auth/, profile, or OAuth code unless I explicitly ask.


## Phase additions (do not skip)

### Phase 5
- Add docker-compose.yml with services: api, worker, redis, postgres, ui.
- Switch Django DB from hardcoded SQLite to Postgres via DATABASE_URL env var.
- Add explicit Postgres indices on (payer_vpa, timestamp), (payee_vpa, timestamp),
  (device_id, timestamp), (payer_vpa, payee_vpa, timestamp). Document why each exists.

### Phase 6
- Analyst review queue: list REVIEW-action decisions; analyst marks
  confirmed_fraud / legit / unclear; disposition written back as a label.
  Show precision on reviewed cases in the monitoring page.
- Add pipelines/ module (train.py, backtest.py, drift_check.py) + Makefile as the
  orchestration layer. No Airflow, no Prefect.
- CI adds a ruff lint step alongside tests and the PR-AUC quality gate.

### Phase 7
- Rewrite README problem-first: APP fraud framing, why card-fraud features don't
  apply, synthetic data disclosure. Tech stack goes last, not first.
- Delete dead code: UserPredictDataForm.
- Record a 60-90s demo of the full flow (replay -> auto-scoring -> alerts ->
  reason codes -> analyst labels).