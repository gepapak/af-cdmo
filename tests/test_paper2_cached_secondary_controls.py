"""Expose provider-free cached-audit arithmetic contracts to public pytest."""

from scripts import audit_paper2_fixed_reconstruction_portable_cached_v1 as reconstruction
from scripts import audit_paper2_nested_learning_value_portable_cached_v1 as nested


def test_nested_control_mapping_weighted_auc_and_calendar_contracts():
    # Includes an independent sklearn weighted-AUC comparison, tied rankings,
    # complete-current/physical-field identity, and a missing calendar day.
    nested.self_test()


def test_reconstruction_weighted_quantiles_and_zero_error_bootstrap_contracts():
    # Weighted quantiles are compared with explicit repeated samples; pairwise
    # invariance to centering and zero-error paired intervals are checked.
    reconstruction.self_test()
