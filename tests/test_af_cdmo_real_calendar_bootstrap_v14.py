from __future__ import annotations

import numpy as np

from scripts.analyze_af_cdmo_real_v14_calendar_bootstrap import (
    _bootstrap_indices,
    _calendar_blocks,
    _paired_summary,
    _worst_fixed_summary,
)


def test_calendar_blocks_respect_duration_on_irregular_grid() -> None:
    minute = 60 * 1_000_000_000
    timestamps = np.asarray([0, 15, 30, 60, 75, 90], dtype=np.int64) * minute
    blocks, audit = _calendar_blocks(timestamps, block_days=1.0 / 24.0)
    assert [len(block) for block in blocks] == [3, 3, 3, 4, 4, 4]
    assert audit["gap_minutes"]["max"] == 30.0
    assert audit["calendar_block_observed_rows"]["min"] == 3


def test_calendar_bootstrap_indices_are_deterministic_and_bounded() -> None:
    blocks = [np.asarray([0, 1]), np.asarray([2, 3]), np.asarray([4])]
    first = _bootstrap_indices(blocks, rows=5, replicates=20, seed=7)
    second = _bootstrap_indices(blocks, rows=5, replicates=20, seed=7)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (20, 5)
    assert int(first.min()) >= 0
    assert int(first.max()) < 5


def test_paired_calendar_summary_detects_uniform_improvement() -> None:
    baseline = np.asarray([2.0, 3.0, 4.0, 5.0])
    candidate = baseline - 0.5
    indices = np.tile(np.arange(4), (200, 1))
    summary = _paired_summary(baseline, candidate, indices)
    assert summary["candidate_minus_baseline_mean_loss"] == -0.5
    assert summary["calendar_block_probability_of_improvement"] == 1.0
    assert summary["improvement_percent_calendar_block_95ci"][0] > 0.0


def test_worst_fixed_calendar_summary_uses_stresswise_maximum() -> None:
    baseline = np.asarray([[2.0, 2.0, 2.0], [5.0, 5.0, 5.0]])
    candidate = np.asarray([[1.0, 1.0, 1.0], [4.0, 4.0, 4.0]])
    indices = np.tile(np.arange(3), (200, 1))
    summary = _worst_fixed_summary(baseline, candidate, indices)
    assert summary["baseline_worst_fixed_mae_eur_mwh"] == 5.0
    assert summary["candidate_worst_fixed_mae_eur_mwh"] == 4.0
    assert summary["candidate_minus_baseline_mean_loss"] == -1.0
    assert summary["calendar_block_probability_of_improvement"] == 1.0
