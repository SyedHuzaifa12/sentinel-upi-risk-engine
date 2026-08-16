"""The model quality gate must fail on a degraded metrics fixture and pass
on the real one -- proving the gate would actually catch a PR-AUC
regression in CI, not just that it runs without crashing.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402

from ml.src.utils.paths import METRICS_V2_PATH  # noqa: E402
from pipelines.quality_gate import WARM_PR_AUC_MIN, check_quality_gate  # noqa: E402


def _write_metrics_fixture(tmp_path: Path, warm_pr_auc: float) -> Path:
    path = tmp_path / "metrics_v2.json"
    with open(path, "w") as f:
        json.dump({"metrics": {"warm": {"pr_auc": warm_pr_auc}}}, f)
    return path


def test_gate_fails_on_a_degraded_metrics_fixture(tmp_path):
    degraded_path = _write_metrics_fixture(tmp_path, warm_pr_auc=WARM_PR_AUC_MIN - 0.1)

    with pytest.raises(ValueError, match="quality gate FAILED"):
        check_quality_gate(degraded_path)


def test_gate_passes_on_a_healthy_metrics_fixture(tmp_path):
    healthy_path = _write_metrics_fixture(tmp_path, warm_pr_auc=WARM_PR_AUC_MIN + 0.1)

    result = check_quality_gate(healthy_path)

    assert result == pytest.approx(WARM_PR_AUC_MIN + 0.1)


def test_gate_passes_on_the_real_metrics_file():
    if not METRICS_V2_PATH.exists():
        pytest.skip(f"{METRICS_V2_PATH} not present -- run ml/src/train.py first")

    result = check_quality_gate(METRICS_V2_PATH)

    assert result >= WARM_PR_AUC_MIN
