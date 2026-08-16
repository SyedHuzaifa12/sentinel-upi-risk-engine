import json

import pytest

from service import scoring


def test_matching_registry_passes(tmp_path):
    real = scoring._assert_feature_registry_matches_code()
    assert "cold_model" in real and "warm_model" in real


def test_mismatched_registry_refuses_to_start(tmp_path, monkeypatch):
    bad_registry = {
        "cold_model": {"features": ["not_a_real_feature"], "model_version": "cold-fake"},
        "warm_model": {"features": ["not_a_real_feature"], "model_version": "warm-fake"},
    }
    bad_path = tmp_path / "feature_registry.json"
    bad_path.write_text(json.dumps(bad_registry))

    monkeypatch.setattr(scoring, "FEATURE_REGISTRY_PATH", bad_path)

    with pytest.raises(RuntimeError, match="Refusing to start"):
        scoring._assert_feature_registry_matches_code()
