# Thin wrappers around already-existing commands (docker compose, pytest,
# the pipelines/ scripts, service.benchmark) -- not a task runner in its own
# right. No Airflow/Prefect: this is a single-node system with no DAG (each
# pipelines/ script is a straight-line batch job: load -> compute -> write)
# and no recurring schedule to manage. A scheduler solves fan-out and
# time-based orchestration across many interdependent jobs -- problems this
# system doesn't have. Re-introduce one only if that changes.

.PHONY: train backtest drift test up down bench

train:
	python -m pipelines.train

# Usage: make backtest FROM=2026-08-01 TO=2026-08-16 MODEL=warm-20260816-9a6747c
backtest:
	python -m pipelines.backtest --from $(FROM) --to $(TO) --model $(MODEL) \
		--thresholds ml/artifacts/policy/thresholds.json

drift:
	python -m pipelines.drift_check

test:
	pytest ml/tests feature_lib/tests service/tests worker/tests decisionlog/tests \
		pipelines/tests backend/users/tests

up:
	docker compose up -d

down:
	docker compose down

bench:
	python -m service.benchmark
