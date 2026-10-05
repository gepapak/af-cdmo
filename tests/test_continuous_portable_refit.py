"""Meaningful cache-contract and numerical tests without provider data."""
from __future__ import annotations

import numpy as np
import pytest
import importlib.util
import sys
from pathlib import Path

_scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_scripts))
_spec = importlib.util.spec_from_file_location("continuous_refit_under_test", _scripts / "audit_paper2_continuous_refit_portable_v1.py")
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


def _caches(tmp_path):
    rng = np.random.default_rng(231)
    times = np.asarray(["2026-01-01T00:00:00+00:00", "2026-01-03T00:00:00+00:00", "2026-01-04T00:00:00+00:00"])
    analytic = audit.metrics.center(rng.normal(size=(3, 10)))
    target = audit.metrics.center(rng.normal(size=(3, 10)))
    residual = audit.metrics.center(target - analytic)
    features = np.zeros((3, 84))
    features[:, 0] = np.log1p(1.0)
    features[:, 1] = 1.0
    features[:, 2:12] = analytic
    magnitude = np.abs(audit.metrics.pairs(residual)).mean(axis=1)
    feature_path, target_path = tmp_path / "features.npz", tmp_path / "targets.npz"
    np.savez(feature_path, features=features, target=np.zeros(3), residual_magnitude_eur_mwh=magnitude, timestamp_utc=times)
    target_data = dict(target_field=target, analytic_field=analytic, residual_field=residual, timestamp_utc=times)
    np.savez(target_path, **target_data)
    return feature_path, target_path, target_data


def test_exact_safe_numeric_cache_alignment_and_gap(tmp_path):
    features, targets, _ = _caches(tmp_path)
    matrix, target, residual, identity = audit.load_window(features, targets)
    assert identity["rows"] == 3 and identity["maximum_analytic_magnitude_difference"] == 0
    assert matrix[3][1] - matrix[3][0] == audit.pd.Timedelta(days=2)
    np.testing.assert_array_equal(residual, audit.metrics.center(target - matrix[0][:, 2:12]))


def test_object_timestamp_pickle_is_rejected(tmp_path):
    features, targets, data = _caches(tmp_path)
    data["timestamp_utc"] = data["timestamp_utc"].astype(object)
    np.savez(targets, **data)
    with pytest.raises(ValueError):
        audit.load_window(features, targets)


def test_target_timestamp_and_analytic_identity_fail_closed(tmp_path):
    features, targets, data = _caches(tmp_path)
    data["timestamp_utc"] = data["timestamp_utc"][::-1]
    np.savez(targets, **data)
    with pytest.raises(ValueError):
        audit.load_window(features, targets)
    features, targets, data = _caches(tmp_path)
    data["analytic_field"] = data["analytic_field"].copy()
    data["analytic_field"][0, 0] += 1e-3
    np.savez(targets, **data)
    with pytest.raises(ValueError, match="Analytic"):
        audit.load_window(features, targets)


def test_weighted_p90_matches_explicit_expansion_and_fit_only_scaler():
    audit.self_test()
    values = np.asarray([0., 2., 2., 9.])
    weights = np.asarray([3, 0, 2, 1])
    assert audit.metrics.weighted_quantile(values, weights, .90) == pytest.approx(np.quantile(np.repeat(values, weights), .90))
