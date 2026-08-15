"""Registry: COLD_FEATURES/WARM_FEATURES must never hand-drift from the
per-feature cold_safe flags, and Group D must never leak into COLD_FEATURES
-- including via its _is_missing indicators, which would otherwise be a
constant-True giveaway on cold data.
"""
from feature_lib.registry import COLD_FEATURES, REGISTRY, WARM_FEATURES


def test_cold_and_warm_features_match_independently_recomputed_filter():
    expected_cold = [f.name for f in REGISTRY if f.cold_safe]
    expected_warm = [f.name for f in REGISTRY if not f.cold_safe]

    assert COLD_FEATURES == expected_cold
    assert WARM_FEATURES == expected_warm


def test_cold_and_warm_are_disjoint_and_cover_every_feature():
    cold_set = set(COLD_FEATURES)
    warm_set = set(WARM_FEATURES)
    all_names = {f.name for f in REGISTRY}

    assert cold_set.isdisjoint(warm_set)
    assert cold_set | warm_set == all_names


def test_no_group_d_feature_or_indicator_in_cold_features():
    group_d_names = {f.name for f in REGISTRY if f.group == "D"}
    assert group_d_names, "expected at least one Group D feature to exist"
    assert group_d_names.isdisjoint(set(COLD_FEATURES))
    assert group_d_names <= set(WARM_FEATURES)
