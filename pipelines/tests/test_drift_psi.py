"""PSI unit tests, isolated from the database and the training pipeline --
_psi/_psi_categorical are pure functions of two numpy arrays, so these don't
need TEST_DATABASE_URL or a trained model.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from pipelines.drift_check import PSI_MODERATE, PSI_SIGNIFICANT, _psi, _psi_categorical  # noqa: E402


def test_psi_is_near_zero_for_identical_distributions():
    rng = np.random.default_rng(42)
    reference = rng.normal(loc=0.0, scale=1.0, size=5000)
    current = rng.normal(loc=0.0, scale=1.0, size=5000)

    psi = _psi(reference, current)

    assert psi < PSI_MODERATE, f"expected near-zero PSI for identical distributions, got {psi}"


def test_psi_is_significant_for_a_shifted_distribution():
    rng = np.random.default_rng(42)
    reference = rng.normal(loc=0.0, scale=1.0, size=5000)
    # A large mean shift plus a tighter variance -- a population that has
    # genuinely moved, not just sampling noise.
    current = rng.normal(loc=3.0, scale=0.3, size=5000)

    psi = _psi(reference, current)

    assert psi > PSI_SIGNIFICANT, f"expected significant PSI for a shifted distribution, got {psi}"


def test_psi_treats_nan_as_its_own_bin():
    rng = np.random.default_rng(42)
    reference = rng.normal(size=1000)
    current = np.concatenate([rng.normal(size=1000), np.full(1000, np.nan)])

    psi = _psi(reference, current)

    # Half the current sample is now missing when none of the reference was
    # -- a real missingness-rate shift, not noise.
    assert psi > PSI_MODERATE


def test_psi_categorical_is_near_zero_for_identical_category_mix():
    reference = np.array(["round"] * 300 + ["neither"] * 700)
    current = np.array(["round"] * 290 + ["neither"] * 710)

    psi = _psi_categorical(reference, current)

    assert psi < PSI_MODERATE


def test_psi_categorical_is_significant_for_a_flipped_category_mix():
    reference = np.array(["round"] * 900 + ["neither"] * 100)
    current = np.array(["round"] * 100 + ["neither"] * 900)

    psi = _psi_categorical(reference, current)

    assert psi > PSI_SIGNIFICANT
