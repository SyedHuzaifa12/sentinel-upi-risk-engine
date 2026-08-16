"""Validation tests for the synthetic UPI event generator
(ml/src/generator/). Plain pytest, no Django/DB dependency -- run with:

    pytest ml/tests
"""
import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.src.generator.generator import events_to_hash_input, generate  # noqa: E402

# Deliberately the real default scale, not a shrunken "fast" approximation:
# per-column AUC is a large-sample statistic, and smaller scales proved
# genuinely flaky here (600/30 gave hour_of_day AUC 0.773; 1200/45 gave
# amount AUC 0.761; 3000/90 gave amount AUC 0.758) purely from finite-sample
# variance in which random fraud instances got which hour/amount, not from
# any actual change in generator behavior. SCAM_COLLECT's amount is 5-100x
# personal average by design, so amount's true AUC sits near (but under,
# ~0.73) the 0.75 threshold -- only a large enough sample estimates that
# stably. This makes the suite slower (~1 minute) but not flaky.
MEDIUM_DAYS = 90
MEDIUM_PAYERS = 5000
MEDIUM_PAYEES = 2000
FRAUD_RATE = 0.007
BAND_LOW, BAND_HIGH = 0.004, 0.012


@pytest.fixture(scope="module")
def medium_dataset():
    events, summary = generate(
        days=MEDIUM_DAYS, n_payers=MEDIUM_PAYERS, n_payees=MEDIUM_PAYEES,
        seed=42, fraud_rate=FRAUD_RATE,
    )
    return events, summary


def test_determinism_same_seed_identical_hash():
    events_a, _ = generate(days=7, n_payers=200, n_payees=80, seed=7, fraud_rate=FRAUD_RATE)
    events_b, _ = generate(days=7, n_payers=200, n_payees=80, seed=7, fraud_rate=FRAUD_RATE)

    hash_a = hashlib.sha256(events_to_hash_input(events_a)).hexdigest()
    hash_b = hashlib.sha256(events_to_hash_input(events_b)).hexdigest()
    assert hash_a == hash_b


def test_timestamps_strictly_non_decreasing(medium_dataset):
    events, _ = medium_dataset
    timestamps = [e.timestamp for e in events]
    assert all(t1 <= t2 for t1, t2 in zip(timestamps, timestamps[1:]))


def test_fraud_rate_within_configured_band(medium_dataset):
    _, summary = medium_dataset
    assert BAND_LOW <= summary["fraud_rate"] <= BAND_HIGH, (
        f"actual fraud rate {summary['fraud_rate']:.4f} outside [{BAND_LOW}, {BAND_HIGH}]"
    )


def test_no_single_feature_column_leaks(medium_dataset):
    """Only feature-candidate columns are tested here -- identifier columns
    (txn_id, payer_vpa, payee_vpa, device_id) are excluded entirely: they're
    never model features, and scoring them by per-category empirical fraud
    rate on the same data they're evaluated against is itself target leakage
    (e.g. every mule VPA would trivially score 1.0)."""
    events, _ = medium_dataset
    y = np.array([1 if e.label_is_fraud else 0 for e in events])

    numeric_columns = {
        "amount": [e.amount for e in events],
        "hour_of_day": [e.timestamp.hour for e in events],
        "payer_account_age_days": [e.payer_account_age_days for e in events],
    }
    categorical_columns = {
        "txn_type": [e.txn_type for e in events],
        "initiation_mode": [e.initiation_mode for e in events],
        "payer_bank": [e.payer_bank for e in events],
        "payee_bank": [e.payee_bank for e in events],
    }

    for name, values in numeric_columns.items():
        arr = np.array(values, dtype=float)
        auc = max(roc_auc_score(y, arr), roc_auc_score(y, -arr))
        assert auc <= 0.75, f"'{name}' alone achieves AUC {auc:.3f} (leak)"

    for name, values in categorical_columns.items():
        labels_by_category = {}
        for category, label in zip(values, y):
            labels_by_category.setdefault(category, []).append(label)
        rate_by_category = {
            category: sum(labels) / len(labels) for category, labels in labels_by_category.items()
        }
        scores = np.array([rate_by_category[category] for category in values])
        auc = roc_auc_score(y, scores)
        assert auc <= 0.75, f"'{name}' alone achieves AUC {auc:.3f} (leak)"


def test_mule_payees_have_higher_hourly_fanin_than_legit(medium_dataset):
    """Mule fan-in is now spread over 6-24h (not 1-6h) with 8-40 senders
    (not 30-200) -- deliberately less extreme, so mule velocity isn't
    orders of magnitude above any legitimate merchant's (see PROGRESS.md
    "Bugs found and fixed"). A same-calendar-hour bucket count is too
    narrow a window to reliably catch a fan-in spread this way, so this
    uses a rolling 24h window (max distinct payers in any trailing 24h)
    instead -- still a real, meaningfully higher, but no longer
    unrealistically extreme, signal."""
    events, _ = medium_dataset

    mule_payees = {e.payee_vpa for e in events if e.label_typology == "MULE_FANIN"}
    if not mule_payees:
        pytest.skip("no MULE_FANIN operation was injected at this scale/seed")

    from bisect import bisect_left
    from datetime import timedelta

    def max_rolling_24h_unique_payer_count(payee_vpa):
        payee_events = sorted(
            ((e.timestamp, e.payer_vpa) for e in events if e.payee_vpa == payee_vpa),
            key=lambda pair: pair[0],
        )
        timestamps = [t for t, _ in payee_events]
        best = 0
        for i, (ts, _) in enumerate(payee_events):
            lo = bisect_left(timestamps, ts - timedelta(hours=24))
            distinct = len({payer for _, payer in payee_events[lo:i + 1]})
            best = max(best, distinct)
        return best

    mule_max_fanins = [max_rolling_24h_unique_payer_count(vpa) for vpa in mule_payees]

    legit_payees = {e.payee_vpa for e in events if e.label_typology == "legit"}
    legit_max_fanins = [max_rolling_24h_unique_payer_count(vpa) for vpa in legit_payees]

    # Median, not min: with fan-in deliberately tamed to 8-40 senders (not
    # 30-200), the occasional weak mule instance overlapping a busy
    # merchant's upper tail is now expected and realistic -- mule detection
    # via fan-in alone should no longer be a guaranteed catch every time.
    # The typical mule instance should still stand out clearly, though.
    p95_legit = np.percentile(legit_max_fanins, 95)
    median_mule = float(np.median(mule_max_fanins))
    assert median_mule > p95_legit, (
        f"median mule fan-in ({median_mule}) does not clearly exceed "
        f"the 95th percentile of legit payee fan-in ({p95_legit})"
    )


def test_legit_new_payees_appear_in_every_10_day_window(medium_dataset):
    """Regression test for the required legit-payee-churn fix: mule/QR-swap
    VPAs must not be the only payees created after day 0, or payee-age
    becomes a perfect fraud separator."""
    events, _ = medium_dataset
    sim_start = min(e.timestamp for e in events)

    legit_events = [e for e in events if e.label_typology == "legit"]
    first_seen = {}
    for e in legit_events:
        first_seen.setdefault(e.payee_vpa, e.timestamp)
    first_seen_days = sorted((ts - sim_start).days for ts in first_seen.values())

    # Same stratification scheme as population.build_payees: 10-day windows
    # spanning day 1 .. MEDIUM_DAYS-1, ceiling-divided so a partial final
    # window is still covered.
    window_size = 10
    n_windows = max(1, math.ceil((MEDIUM_DAYS - 1) / window_size))
    for w in range(n_windows):
        window_start = 1 + w * window_size
        window_end = min(MEDIUM_DAYS - 1, window_start + window_size - 1)
        assert any(window_start <= d <= window_end for d in first_seen_days), (
            f"no new legit payee first-seen in day window [{window_start}, {window_end}]"
        )
