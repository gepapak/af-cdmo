"""Synthetic contracts for portable secondary audits; no provider data needed."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import audit_paper2_operational_triage_portable_v1 as triage
from scripts import paper2_portable_audit_utils as inputs
from scripts import paper2_secondary_temporal_metrics as metrics


def test_manifest_verification_uses_logical_identity_after_relocation(tmp_path):
    source = tmp_path / "renamed_private_input.json"
    source.write_text("frozen bytes", encoding="utf-8")
    manifest = {"files": [{"path": "reports/original.json", "sha256": inputs.sha256(source)}]}
    assert inputs.manifest_check(source, manifest, "reports/original.json")["verified_against_frozen_manifest"]
    source.write_text("changed bytes", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        inputs.manifest_check(source, manifest, "reports/original.json")


def test_windows_input_path_relocation_preserves_frozen_report(tmp_path):
    data = tmp_path / "data" / "source.csv.gz"
    data.parent.mkdir()
    data.write_bytes(b"authorized input")
    original = {"sources": {"dual_archive": "Q:/study/Paper2/data/source.csv.gz", "dual_archive_sha256": inputs.sha256(data)}}
    args = argparse.Namespace(input_root=tmp_path, dual_archive=None)
    mapped = inputs.map_report_sources(original, args, tmp_path, names=("dual_archive",))
    assert mapped["sources"]["dual_archive"] == str(data.resolve())
    assert original["sources"]["dual_archive"] == "Q:/study/Paper2/data/source.csv.gz"
    assert mapped["sources"]["dual_archive_sha256"] == original["sources"]["dual_archive_sha256"]
    with pytest.raises(ValueError, match="escapes"):
        inputs.relocated("Paper2/../elsewhere", tmp_path)


def test_exact_float32_threshold_and_score_alignment():
    threshold = float(np.float32(5.5921735763549805))
    magnitude = np.array([threshold, np.nextafter(threshold, np.inf)])
    times = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    target = np.array([0., 1.])
    assert inputs.validate_predictions(times, target, magnitude, {"method": np.array([.1, .9])}, threshold).equals(times)
    with pytest.raises(ValueError, match="exact original threshold"):
        inputs.validate_predictions(times, 1 - target, magnitude, {}, threshold)
    with pytest.raises(ValueError, match="unaligned"):
        inputs.validate_predictions(times, target, magnitude, {"method": np.array([.1])}, threshold)
    with pytest.raises(ValueError, match="unique"):
        inputs.validate_predictions([times[0], times[0]], target, magnitude, {}, threshold)


def test_supplied_hgb_predictions_require_identical_cohort_order(tmp_path):
    times = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
    target, magnitude, probability = np.array([0., 1., 0.]), np.array([1., 8., 2.]), np.array([.1, .9, .2])
    path = tmp_path / "private.npz"
    np.savez(path, timestamp_utc=np.asarray([item.isoformat() for item in times]), target=target,
             residual_magnitude_eur_mwh=magnitude, probability=probability)
    assert np.array_equal(inputs.aligned_hgb_csv(path, times, target, magnitude), probability)
    with pytest.raises(ValueError, match="cohort order"):
        inputs.aligned_hgb_csv(path, times[::-1], target, magnitude)
    with pytest.raises(ValueError, match="targets differ"):
        inputs.aligned_hgb_csv(path, times, 1 - target, magnitude)


def test_fixed_december_threshold_is_strict_and_global_budget_is_floor():
    target = np.array([1., 0., 1., 0., 0.])
    probability = np.array([.5, .5, .8, .2, .1])
    magnitude = np.array([8., 1., 9., 2., 3.])
    fixed = metrics.threshold_operational_metrics(target, magnitude, probability, .5)
    assert fixed["flagged_rows"] == 1 and fixed["missed_tail_rows"] == 1
    assert np.array_equal(metrics.exact_budget_flags(probability, .4), [1, 0, 1, 0, 0])
    assert metrics.decision_cost(target, probability, .4, missed_tail_cost=1., audit_cost=.05) == pytest.approx(.02)


def test_daily_policy_preserves_calendar_gaps_and_ceil_workload():
    scores, days = np.array([.8, .8, .1, .9, .4]), np.array([0, 0, 0, 2, 2])
    decision = triage.daily_budget(scores, days, .5)
    assert np.array_equal(decision, [1, 1, 0, 1, 0])
    counts = triage.aggregate_days(decision, np.array([1., 0., 1., 1., 0.]), np.ones(5), days, 3)
    assert np.array_equal(counts[1], np.zeros(5))
    weights = triage.calendar_weights(17, 200, 7)
    assert np.all(weights.sum(axis=1) == 17)
    assert triage.ratios(counts.sum(axis=0))["tail_recall"] == pytest.approx(2 / 3)


def test_temporal_paired_statistics_have_known_difference_and_determinism():
    times = pd.date_range("2026-01-01", periods=30, freq="D", tz="UTC")
    target = np.tile([0., 1.], 15)
    comparator = np.where(target > 0, .8, .2)
    candidate = np.where(target > 0, .7, .3)
    result = metrics.compare_brier(times, target, candidate, comparator, block_days=[7, 14, 28],
        replicates=100, seed=20261003, absolute_margin=.0025, relative_margin=.05)
    assert result["candidate_minus_comparator_brier"] == pytest.approx(.05)
    assert result["effective_margin"] == pytest.approx(.002)
    assert not result["noninferior_all_block_lengths"]
    for block in result["calendar_block_bootstrap"].values():
        assert block["ci95"] == pytest.approx([.05, .05])
    a = metrics.paired_calendar_block_bootstrap(times, np.arange(30), block_days=7, replicates=100, seed=4)
    b = metrics.paired_calendar_block_bootstrap(times, np.arange(30), block_days=7, replicates=100, seed=4)
    assert a == b


@pytest.mark.parametrize("filename", [
    "audit_paper2_operational_triage_portable_v1.py", "audit_paper2_learning_value_portable_v1.py",
    "audit_paper2_later_temporal_portable_v1.py", "audit_paper2_later_hgb_portable_v1.py",
])
def test_portable_cli_help_needs_no_provider_files_or_sibling_project(filename):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / "scripts" / filename), "--help"],
        capture_output=True, text=True, timeout=45, cwd=root)
    assert result.returncode == 0, result.stderr
    assert "--output" in result.stdout and "--predictions" in result.stdout
    text = (root / "scripts" / filename).read_text(encoding="utf-8")
    assert 'parent / "Paper3"' not in text and "PAPER3 /" not in text


def test_learning_self_test_runs_without_optional_psutil():
    root = Path(__file__).resolve().parents[1]
    driver = root / "scripts/audit_paper2_learning_value_portable_v1.py"
    command = "import runpy,sys; sys.modules['psutil']=None; sys.argv=[sys.argv[1],'--self-test']; runpy.run_path(sys.argv[0],run_name='__main__')"
    result = subprocess.run([sys.executable, "-c", command, str(driver)], capture_output=True,
                            text=True, timeout=45, cwd=root)
    assert result.returncode == 0, result.stderr
    assert "checks passed" in result.stdout
