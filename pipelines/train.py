"""Training pipeline entry point with MLflow experiment tracking.

    python -m pipelines.train
    make train

Wraps `ml.src.train.main()` -- the actual training logic lives there and is
NOT reimplemented here -- in an MLflow run: logs params/metrics/artifacts,
and registers the cold/warm models in the local Model Registry so a
`decisions.model_version` string can be resolved back to a registry entry.

Local, no server: tracking URI is `file:./mlruns` (confirmed to support the
Model Registry directly -- verified empirically before committing to this
design, not assumed). Never run `mlflow server`/`mlflow ui` as part of this
system; `mlflow-skinny` (the dependency pinned in requirements.txt) doesn't
even have Flask installed, so it structurally can't.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import mlflow  # noqa: E402
import mlflow.lightgbm  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from feature_lib.compute import COLD_PAYEE_TXN_THRESHOLD  # noqa: E402
from ml.src import train as ml_train  # noqa: E402
from ml.src.training.lightgbm_pipeline import LGBM_PARAMS  # noqa: E402
from ml.src.training.temporal_split import TEST_DAY_RANGE, TRAIN_DAY_RANGE, VAL_DAY_RANGE  # noqa: E402
from ml.src.utils.paths import (  # noqa: E402
    COLD_CALIBRATOR_PATH,
    COLD_MODEL_PATH,
    FEATURE_REGISTRY_PATH,
    METRICS_V2_PATH,
    RELIABILITY_COLD_PATH,
    RELIABILITY_WARM_PATH,
    THRESHOLD_CURVE_PATH,
    THRESHOLDS_PATH,
    WARM_CALIBRATOR_PATH,
    WARM_MODEL_PATH,
)

MLRUNS_DIR = Path(__file__).resolve().parents[1] / "mlruns"
EXPERIMENT_NAME = "sentinel-upi-risk-engine"


def _flatten_numeric(prefix: str, d: dict, out: dict) -> None:
    """Only numeric leaves become MLflow metrics (log_metrics requires
    floats). Non-numeric fields (formatted CI strings, notes, nested
    per-typology dicts with mixed types) are NOT force-flattened -- they're
    already fully captured in the logged metrics_v2.json artifact, so
    nothing is lost, just not force-fit into the wrong shape."""
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, bool):
            continue
        if isinstance(v, dict):
            _flatten_numeric(key, v, out)
        elif isinstance(v, (int, float)):
            out[key] = float(v)


def main():
    mlflow.set_tracking_uri(f"file:{MLRUNS_DIR.as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        metrics_v2 = ml_train.main()

        with open(FEATURE_REGISTRY_PATH) as f:
            feature_registry = json.load(f)
        cold_model_version = feature_registry["cold_model"]["model_version"]
        warm_model_version = feature_registry["warm_model"]["model_version"]

        mlflow.log_params({
            "generator_seed": ml_train.GENERATOR_SEED,
            "cold_payee_txn_threshold": COLD_PAYEE_TXN_THRESHOLD,
            "train_day_range": list(TRAIN_DAY_RANGE),
            "val_day_range": list(VAL_DAY_RANGE),
            "test_day_range": list(TEST_DAY_RANGE),
            **{f"lgbm.{k}": v for k, v in LGBM_PARAMS.items()},
        })

        flat_metrics = {}
        _flatten_numeric("", metrics_v2["metrics"], flat_metrics)
        mlflow.log_metrics(flat_metrics)

        mlflow.set_tags({
            "git_sha": metrics_v2.get("git_sha") or "unknown",
            "cold_model_version": cold_model_version,
            "warm_model_version": warm_model_version,
        })

        for path in (FEATURE_REGISTRY_PATH, METRICS_V2_PATH, RELIABILITY_COLD_PATH,
                     RELIABILITY_WARM_PATH, THRESHOLDS_PATH, THRESHOLD_CURVE_PATH):
            if path.exists():
                mlflow.log_artifact(str(path))

        cold_model = joblib.load(COLD_MODEL_PATH)
        warm_model = joblib.load(WARM_MODEL_PATH)
        mlflow.lightgbm.log_model(cold_model, artifact_path="cold_model")
        mlflow.lightgbm.log_model(warm_model, artifact_path="warm_model")
        # Calibrators are plain sklearn IsotonicRegression objects, not
        # LightGBM -- logged as generic artifacts (the pickle files already
        # on disk), not via a model flavor.
        mlflow.log_artifact(str(COLD_CALIBRATOR_PATH), artifact_path="cold_calibrator")
        mlflow.log_artifact(str(WARM_CALIBRATOR_PATH), artifact_path="warm_calibrator")

        client = MlflowClient()
        registered = {}
        for name, model_version_str, artifact_subpath in (
            ("sentinel-cold", cold_model_version, "cold_model"),
            ("sentinel-warm", warm_model_version, "warm_model"),
        ):
            model_uri = f"runs:/{run.info.run_id}/{artifact_subpath}"
            mv = mlflow.register_model(model_uri, name)
            # This is the resolution mechanism: given a decisions.model_version
            # string (e.g. "warm-20260816-9a6747c"), search_model_versions(
            # f"tags.model_version = '{model_version_str}'") finds this entry.
            # MLflow's own auto-incrementing version numbers are a separate,
            # unrelated numbering scheme -- never conflated with ours.
            client.set_model_version_tag(name, mv.version, "model_version", model_version_str)
            registered[name] = {"registry_version": mv.version, "model_version": model_version_str}
            print(f"Registered {name} v{mv.version} (model_version={model_version_str})")

        print(f"\nMLflow run: {run.info.run_id}")
        print(f"Tracking URI: {mlflow.get_tracking_uri()}")

    return {"metrics_v2": metrics_v2, "run_id": run.info.run_id, "registered": registered}


if __name__ == "__main__":
    main()
