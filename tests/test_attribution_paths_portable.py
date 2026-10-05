"""Data-free functional, margin and exact-zero decision contracts."""
import numpy as np
from scripts import audit_paper2_attribution_paths_portable_v1 as audit
from scripts import paper2_attribution_paths_metrics as metrics


def test_unit_rescaling_centering_and_routing_contracts():
    audit.self_test()


def test_large_margin_certifies_stable_top_tag():
    a = np.asarray([[[8., 2., 1.]]])
    b = np.asarray([[[7.9, 2.1, .9]]])
    context = metrics.routing_arrays(a, b)
    context.update(tags=["A", "B", "C"], times=["synthetic"])
    detail = metrics.descriptive_detail(context)
    assert detail["individual_margin_certified_nonzero"] == 1
    assert detail["global_margin_certified_nonzero"] == 1
    assert detail["routing_flips"] == 0


def test_exact_zero_screen_can_change_without_counting_primary_flip():
    a = np.zeros((1, 1, 2))
    b = np.asarray([[[0., 1e-30]]])
    context = metrics.routing_arrays(a, b)
    context.update(tags=["A", "B"], times=["synthetic"])
    detail = metrics.descriptive_detail(context)
    assert detail["routing_flips"] == 0  # Original route-A nonzero estimand.
    assert detail["exact_all_zero_screen_membership_changes"] == 1
    assert detail["individual_margin_certified_nonzero"] == 0


def test_exact_ties_follow_declared_tag_order():
    context = metrics.routing_arrays(np.ones((1, 2, 3)), np.ones((1, 2, 3)))
    assert np.array_equal(context["selected_a"], np.zeros((1, 2), dtype=int))
    assert np.array_equal(context["margins"], np.zeros((1, 2)))
