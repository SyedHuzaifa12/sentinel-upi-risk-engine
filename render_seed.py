"""Idempotent startup seeding for the Render demo deploy target.

    python render_seed.py

Safe to run on every container start: skips entirely if `decisions` already
has >= SEED_MIN_ROWS rows (a redeploy/restart never re-seeds or duplicates,
and never overwrites -- decisionlog is append-only by design).

Reduced from an original 2000-generated/600-scored design (2026-08-17):
0.1 CPU makes scoring the slow part, and a startup timeout fails the whole
deploy, not just the seed -- 300 decisions is already plenty to populate
the monitoring dashboard and review queue.

Timing, measured on this dev machine (full CPU, no network latency to a
remote DB): load_models() ~4.6s, generating 1000 events ~0.4s, scoring 300
of them ~4.5s -- about 9.5s end to end. Render's free tier gives 0.1 vCPU
(a tenth of what this was measured on) plus real network round-trips to a
remote Neon Postgres for each of the 300 `decisionlog.record()` calls,
neither of which this local measurement captures. The 60s target may still
not hold on first deploy -- see the WALL_CLOCK_BUDGET_S safety valve below,
and DEPLOY_NOTES.md for what to do if it doesn't (lower RENDER_SEED_N_SCORE
further).
"""
import lightgbm  # noqa: F401,E402,I001 -- MUST be imported before pandas, see module docstring
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from decisionlog import DecisionLog  # noqa: E402
from feature_lib.store.in_memory import InMemoryHistoryStore  # noqa: E402
from ml.src.generator.generator import generate  # noqa: E402
from service import config, scoring  # noqa: E402

SEED_MIN_ROWS = 250
N_GENERATE = int(os.getenv("RENDER_SEED_N_GENERATE", "1000"))
N_SCORE = int(os.getenv("RENDER_SEED_N_SCORE", "300"))
N_WARM_UP = max(N_GENERATE - N_SCORE, 0)
# Hard safety valve, not just a hopeful parameter choice: stop scoring early
# (a partial seed) rather than risk blowing the whole startup budget if the
# free-tier CPU throttle + remote-DB latency turns out worse than measured
# locally. A partial seed still satisfies "fewer than SEED_MIN_ROWS -> seed"
# on the NEXT restart, so nothing is lost, just deferred.
WALL_CLOCK_BUDGET_S = float(os.getenv("RENDER_SEED_BUDGET_S", "45"))


def main():
    if not config.DATABASE_URL:
        print("render_seed: DATABASE_URL not set -- skipping (nothing to seed against).")
        return

    log = DecisionLog(config.DATABASE_URL)  # __init__ calls decisionlog.schema.ensure_schema()
    try:
        existing = log._conn.execute("SELECT count(*) FROM decisions").fetchone()[0]
        if existing >= SEED_MIN_ROWS:
            print(f"render_seed: {existing} decisions already present (>= {SEED_MIN_ROWS}) -- skipping seed.")
            return

        t_start = time.perf_counter()
        print(f"render_seed: {existing} existing rows -- seeding from scratch "
              f"(target: generate {N_GENERATE}, score up to {N_SCORE}, budget {WALL_CLOCK_BUDGET_S}s).")

        registry = scoring.load_models()
        print(f"render_seed: models loaded (cold={registry['cold_model']['model_version']}, "
              f"warm={registry['warm_model']['model_version']}).")

        events, summary = generate(days=15, n_payers=200, n_payees=100, seed=42, fraud_rate=0.02)
        events = events[:N_GENERATE]
        print(f"render_seed: generated {len(events)} synthetic events "
              f"({summary['fraud_event_count']} fraud, seed=42, deterministic).")

        store = InMemoryHistoryStore()
        for event in events[:N_WARM_UP]:
            store.record(event)
        print(f"render_seed: warmed history with {N_WARM_UP} unscored events "
              f"(so scored events see realistic prior history, not a cold start).")

        n_scored = 0
        action_counts = {}
        for event in events[N_WARM_UP:N_WARM_UP + N_SCORE]:
            if time.perf_counter() - t_start > WALL_CLOCK_BUDGET_S:
                print(f"render_seed: hit the {WALL_CLOCK_BUDGET_S}s budget after {n_scored} scored -- "
                      f"stopping early (a partial seed; below {SEED_MIN_ROWS} rows still re-attempts "
                      f"seeding on the next restart).")
                break

            result = scoring.score_event(event, store)
            action_counts[result.action] = action_counts.get(result.action, 0) + 1
            log.record(
                txn_id=result.txn_id,
                event={
                    "payer_vpa": event.payer_vpa, "payee_vpa": event.payee_vpa, "amount": event.amount,
                    "label_is_fraud": event.label_is_fraud, "label_typology": event.label_typology,
                },
                feature_snapshot=result.feature_snapshot,
                risk_score=result.risk_score,
                raw_score=result.raw_score,
                is_cold=result.is_cold,
                action=result.action,
                risk_tier=result.risk_tier,
                reason_codes=result.reason_codes,
                model_version=result.model_version,
                thresholds_version=result.thresholds_version,
                latency_ms=result.latency_ms,
                # `datetime.now()`, not the synthetic event's own ~Jan-2026
                # timestamp -- matches worker/consumer.py's convention and makes
                # the monitoring dashboard's "alert rate over time" show today's
                # date, not a stale-looking seed date, for anyone opening the
                # demo link months after this container first started.
                scored_at=datetime.now(timezone.utc),
                source="replay",
            )
            n_scored += 1

        elapsed = time.perf_counter() - t_start
        print(f"render_seed: done. Scored {n_scored} events in {elapsed:.1f}s (incl. model load). "
              f"Action mix: {action_counts}")
    finally:
        # try/finally, not just a trailing log.close() -- a mid-loop failure
        # (a bad event, a dropped Neon connection) must still release the
        # connection on the way out, not leak it while the process exits.
        log.close()


if __name__ == "__main__":
    # Fail fast, don't retry: this process is a one-shot startup step, not a
    # daemon, and this repo has exactly one failure mode worth distinguishing
    # -- render_entrypoint.sh's `set -e` already stops it from proceeding to
    # `exec uvicorn` after ANY non-zero exit here, so there is no internal
    # retry loop to remove. What this DOES add: a single, unambiguous FATAL
    # line before the traceback, so a crash-looping container (Render
    # restarting the whole container after a crash, not this script retrying
    # itself) produces a readable log instead of a bare traceback repeated
    # per restart. Note: this can only catch RUNTIME failures (a bad Neon
    # connection, a scoring error) -- an IMPORT-time crash (e.g. the
    # 2026-08-18 missing-libgomp1 incident, which failed on `import lightgbm`
    # above, before this block ever runs) can't be caught here at all; the
    # only real fix for that class is not letting the import fail in the
    # first place (see Dockerfile.render's libgomp1 comment).
    try:
        main()
    except Exception as exc:
        print(f"render_seed: FATAL -- seeding failed, exiting immediately, no retry: {exc}", file=sys.stderr)
        raise
