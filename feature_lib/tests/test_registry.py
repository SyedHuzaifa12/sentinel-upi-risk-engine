"""Registry: COLD_FEATURES/WARM_ONLY_FEATURES/ALL_FEATURES must never
hand-drift from the per-feature cold_safe flags, and Group D must never
leak into COLD_FEATURES -- including via its _is_missing indicators, which
would otherwise be a constant-True giveaway on cold data.
"""
from feature_lib.registry import ALL_FEATURES, COLD_FEATURES, REGISTRY, WARM_ONLY_FEATURES


def test_cold_and_warm_only_features_match_independently_recomputed_filter():
    expected_cold = [f.name for f in REGISTRY if f.cold_safe]
    expected_warm_only = [f.name for f in REGISTRY if not f.cold_safe]
    expected_all = [f.name for f in REGISTRY]

    assert COLD_FEATURES == expected_cold
    assert WARM_ONLY_FEATURES == expected_warm_only
    assert ALL_FEATURES == expected_all


def test_cold_and_warm_only_are_disjoint_and_cover_every_feature():
    cold_set = set(COLD_FEATURES)
    warm_only_set = set(WARM_ONLY_FEATURES)
    all_names = {f.name for f in REGISTRY}

    assert cold_set.isdisjoint(warm_only_set)
    assert cold_set | warm_only_set == all_names
    assert set(ALL_FEATURES) == all_names


def test_no_group_d_feature_or_indicator_in_cold_features():
    group_d_names = {f.name for f in REGISTRY if f.group == "D"}
    assert group_d_names, "expected at least one Group D feature to exist"
    assert group_d_names.isdisjoint(set(COLD_FEATURES))
    assert group_d_names <= set(WARM_ONLY_FEATURES)


def test_warm_features_name_does_not_exist():
    """Regression guard: WARM_FEATURES was ambiguous (read as "features for
    the warm model" when it actually meant "warm-only features") and has
    been deleted entirely so nothing can import it by mistake."""
    import feature_lib.registry as registry_module
    assert not hasattr(registry_module, "WARM_FEATURES")
