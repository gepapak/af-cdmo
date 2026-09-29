from __future__ import annotations

import numpy as np

from scripts.benchmark_af_cdmo_real_v13 import (
    NEURAL_METHODS,
    PRIMARY_NORDIC_ZONES,
    _seed_sign_test,
)


def test_primary_panel_is_the_preregistered_ten_zone_panel() -> None:
    assert PRIMARY_NORDIC_ZONES == (
        "DK1",
        "DK2",
        "FI",
        "NO1",
        "NO2",
        "NO3",
        "NO5",
        "SE1",
        "SE3",
        "SE4",
    )


def test_all_preregistered_neural_controls_are_present() -> None:
    assert set(NEURAL_METHODS) == {
        "ambient_cqdm_residual",
        "slack_qdm_residual",
        "af_projected_raw_residual",
        "af_qdm_residual",
    }


def test_seed_sign_test_is_paired_and_tie_aware() -> None:
    baseline = np.asarray([3.0, 2.0, 1.0, 4.0])
    candidate = np.asarray([2.0, 3.0, 1.0, 1.0])
    result = _seed_sign_test(baseline, candidate)
    assert result["candidate_wins"] == 2
    assert result["candidate_losses"] == 1
    assert result["ties"] == 1
    assert result["effective_pairs"] == 3
    assert result["one_sided_exact_sign_test_p_value"] == 0.5
    assert result["two_sided_exact_sign_test_p_value"] == 1.0
