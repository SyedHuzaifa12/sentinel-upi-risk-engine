"""Benchmarks the scoring path's latency against the 100ms p99 budget.

Seeds a fresh InMemoryHistoryStore with train+val-range events (chronological
warm-up, matching the day ranges ml/src/training/temporal_split.py uses),
then scores up to 1000 real test-range events one at a time through
service.scoring.score_event, printing p50/p95/p99 per stage.

    python -m service.benchmark
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_lib.store.in_memory import InMemoryHistoryStore  # noqa: E402
from ml.src.data.load_synthetic import load_synthetic_events  # noqa: E402
from ml.src.training.temporal_split import TEST_DAY_RANGE, VAL_DAY_RANGE  # noqa: E402

from . import scoring  # noqa: E402
from .config import LATENCY_BUDGET_P99_MS  # noqa: E402

SAMPLE_SIZE = 1000
STAGES = ["features_ms", "predict_ms", "calibrate_ms", "decide_ms", "reason_codes_ms"]


def _percentile(values, p):
    values = sorted(values)
    idx = min(len(values) - 1, int(round(p * (len(values) - 1))))
    return values[idx]


def main():
    print("Loading synthetic events...")
    events = load_synthetic_events()
    sim_start = events[0].timestamp

    def day(event):
        return (event.timestamp - sim_start).days

    warm_up = [e for e in events if day(e) <= VAL_DAY_RANGE[1]]
    test_events = [e for e in events if day(e) >= TEST_DAY_RANGE[0]][:SAMPLE_SIZE]

    print(f"Warm-up events (train+val range, day <= {VAL_DAY_RANGE[1]}): {len(warm_up)}")
    print(f"Scoring {len(test_events)} test-range events (day >= {TEST_DAY_RANGE[0]})...")

    scoring.load_models()
    store = InMemoryHistoryStore()
    for event in warm_up:
        store.record(event)

    totals = []
    stage_samples = {stage: [] for stage in STAGES}
    for event in test_events:
        result = scoring.score_event(event, store)
        totals.append(result.latency_ms)
        for stage in STAGES:
            stage_samples[stage].append(result.stage_latency_ms[stage])

    print(f"\n{'stage':<16}{'p50':>10}{'p95':>10}{'p99':>10}   (ms)")
    for stage in STAGES:
        vals = stage_samples[stage]
        p50, p95, p99 = _percentile(vals, 0.50), _percentile(vals, 0.95), _percentile(vals, 0.99)
        print(f"{stage:<16}{p50:>10.2f}{p95:>10.2f}{p99:>10.2f}")
    t50, t95, t99 = _percentile(totals, 0.50), _percentile(totals, 0.95), _percentile(totals, 0.99)
    print(f"{'total':<16}{t50:>10.2f}{t95:>10.2f}{t99:>10.2f}")

    p99_total = _percentile(totals, 0.99)
    verdict = "within" if p99_total <= LATENCY_BUDGET_P99_MS else "OVER"
    print(f"\np99 total latency: {p99_total:.2f}ms against a {LATENCY_BUDGET_P99_MS:.0f}ms budget ({verdict} budget).")
    return {"totals": totals, "stage_samples": stage_samples}


if __name__ == "__main__":
    main()
