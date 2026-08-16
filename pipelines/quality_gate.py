"""CI quality gate: model quality is a build artifact like any compiled
binary -- a regression should fail CI the same way a broken test does, not
surface later as a support ticket. Fails the build if the warm model's
PR-AUC (read from ml/artifacts/metrics/metrics_v2.json, the file
ml/src/train.py itself writes) drops below WARM_PR_AUC_MIN.

    python -m pipelines.quality_gate
    make quality-gate

0.75 is comfortably under the accepted 0.8455 (see PROGRESS.md), chosen to
catch a real regression rather than ordinary run-to-run noise.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.src.utils.paths import METRICS_V2_PATH  # noqa: E402

WARM_PR_AUC_MIN = 0.75


def check_quality_gate(metrics_v2_path: Path = METRICS_V2_PATH, min_warm_pr_auc: float = WARM_PR_AUC_MIN) -> float:
    """Returns the warm PR-AUC if it passes the gate; raises ValueError if not."""
    with open(metrics_v2_path) as f:
        metrics_v2 = json.load(f)

    warm_pr_auc = metrics_v2["metrics"]["warm"]["pr_auc"]
    if warm_pr_auc < min_warm_pr_auc:
        raise ValueError(
            f"Model quality gate FAILED: warm PR-AUC={warm_pr_auc:.4f} is below the "
            f"minimum {min_warm_pr_auc} (see {metrics_v2_path})."
        )
    return warm_pr_auc


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fail the build if warm PR-AUC has regressed.")
    parser.add_argument("--metrics-path", type=Path, default=METRICS_V2_PATH)
    parser.add_argument("--min-warm-pr-auc", type=float, default=WARM_PR_AUC_MIN)
    args = parser.parse_args(argv)

    try:
        warm_pr_auc = check_quality_gate(args.metrics_path, args.min_warm_pr_auc)
    except ValueError as e:
        print(str(e))
        sys.exit(1)

    print(f"Model quality gate PASSED: warm PR-AUC={warm_pr_auc:.4f} >= {args.min_warm_pr_auc}")


if __name__ == "__main__":
    main()
