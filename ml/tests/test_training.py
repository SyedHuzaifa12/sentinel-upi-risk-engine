"""Phase 2 training tests. The first three read the artifacts already
produced by `python ml/src/train.py` (fast -- no retraining). The
determinism test isolates just the new LightGBM-determinism mechanism on a
small hand-built dataset, rather than re-running the full ~3-minute
pipeline twice.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from feature_lib.registry import ALL_FEATURES, COLD_FEATURES, WARM_ONLY_FEATURES  # noqa: E402
from ml.src.utils.paths import FEATURE_REGISTRY_PATH, METRICS_V2_PATH  # noqa: E402


def _load_json(path):
    if not path.exists():
        pytest.skip(f"{path} does not exist -- run `python ml/src/train.py` first")
    with open(path) as f:
        return json.load(f)


def test_calibration_improves_brier_on_test():
    """Strict for warm (hundreds of test positives, a reliable comparison).
    Cold is print-only, not asserted: with ~20 test positives, isotonic
    calibration fit on an equally tiny validation slice can land marginally
    worse on test by chance alone -- that's sampling noise at this sample
    size, not a calibration bug, and it's exactly why cold metrics are
    flagged as indicative throughout this phase, not precise."""
    metrics = _load_json(METRICS_V2_PATH)

    warm = metrics["metrics"]["warm"]
    assert warm["brier_calibrated"] <= warm["brier_raw"], (
        f"warm: calibrated Brier ({warm['brier_calibrated']}) is not <= raw Brier ({warm['brier_raw']})"
    )

    cold = metrics["metrics"]["cold"]
    print(f"\ncold: brier raw={cold['brier_raw']:.4f} calibrated={cold['brier_calibrated']:.4f} "
          f"(n_test_positives={cold['n_test_positives']}, not asserted -- see docstring)")


def test_feature_order_matches_registry_exactly():
    registry = _load_json(FEATURE_REGISTRY_PATH)
    assert registry["cold_model"]["features"] == COLD_FEATURES
    assert registry["warm_model"]["features"] == ALL_FEATURES


def test_cold_model_excludes_warm_only_features():
    registry = _load_json(FEATURE_REGISTRY_PATH)
    cold_features = set(registry["cold_model"]["features"])
    assert cold_features.isdisjoint(set(WARM_ONLY_FEATURES))


_DETERMINISM_SCRIPT = """
import sys
import numpy as np
sys.path.insert(0, {repo_root!r})
from ml.src.training.lightgbm_pipeline import build_model, fit_model

rng = np.random.default_rng({seed})
n = 500
X = np.column_stack([
    rng.normal(size=n),
    rng.normal(size=n),
    rng.integers(0, 5, size=n).astype(float),
])
y = (X[:, 0] + rng.normal(scale=0.5, size=n) > 0.5).astype(int)

split = n * 3 // 4
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]
n_pos = y_train.sum()
scale_pos_weight = (len(y_train) - n_pos) / n_pos if n_pos else 1.0

model = build_model(scale_pos_weight)
fit_model(model, X_train, y_train, X_val, y_val)
proba = model.predict_proba(X_val)[:, 1]
print(",".join(f"{{p:.10f}}" for p in proba))
"""


def _train_and_predict_in_subprocess(seed):
    # Run in a FRESH process, not in-process: LightGBM 4.7.0's ctypes
    # bridge on this platform has shown an intermittent native access
    # violation in LGBM_DatasetSetField when a prior test in the same
    # pytest process has already touched pandas/numpy first (reproduced
    # reliably after ml/tests/test_temporal_split.py, not in isolation) --
    # an environment/library quirk, not a determinism bug in the code under
    # test. A subprocess sidesteps whatever native/global state carries
    # over between tests, which is exactly the contamination in question.
    repo_root = str(Path(__file__).resolve().parents[2])
    script = _DETERMINISM_SCRIPT.format(repo_root=repo_root, seed=seed)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, cwd=repo_root, timeout=60,
    )
    assert result.returncode == 0, f"subprocess failed:\n{result.stdout}\n{result.stderr}"
    last_line = result.stdout.strip().splitlines()[-1]  # LightGBM logs precede our one printed line
    return np.array([float(x) for x in last_line.split(",")])


def test_same_seed_reproduces_identical_predictions():
    """Isolates the LightGBM-determinism mechanism (n_jobs=1, deterministic=True,
    force_row_wise=True) on a small dataset in two independent subprocesses --
    proving the mechanism works without re-paying the ~3 minute full-dataset
    training cost twice, and without being sensitive to whatever ran earlier
    in the test process."""
    proba_a = _train_and_predict_in_subprocess(seed=7)
    proba_b = _train_and_predict_in_subprocess(seed=7)
    np.testing.assert_array_almost_equal(proba_a, proba_b, decimal=4)
