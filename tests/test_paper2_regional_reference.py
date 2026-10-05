"""Independent geometry and fail-closed query checks; no market data."""
import numpy as np
import pytest

from scripts import audit_paper2_regional_reference_v1 as audit
from scripts import audit_paper2_regional_reference_census_v1 as census

ZONES = ["DK1", "DK1_CO", "DK1_DE", "DK1_KS", "DK1_SB", "DK1_SK", "DK2", "DK2_KO", "DK2_SB",
         "FI", "FI_EL", "FI_FS", "NO1", "NO2", "NO2_ND", "NO2_NK", "NO2_SK", "NO3", "NO4", "NO5",
         "SE1", "SE2", "SE3", "SE3_FS", "SE3_KS", "SE3_SWL", "SE4", "SE4_BC", "SE4_NB", "SE4_SP", "SE4_SWL"]


def test_regional_gauge_is_outside_original_geometry_and_removed_by_declared_geometry():
    e5, e6, regional = audit.declared_geometries(ZONES)
    assert np.linalg.matrix_rank(e5) == 5 and np.linalg.matrix_rank(e6) == 6
    assert np.linalg.norm(regional @ audit.projector(e5)) > 2.0
    assert np.linalg.norm(regional @ audit.projector(e6)) < 1e-12
    normal = np.zeros(31)
    normal[ZONES.index("DK2")] = 1.
    before = audit.geometric_row(normal, 1152., e6)
    after = audit.geometric_row(normal - regional, 1152., e6)
    assert np.allclose(before["direction"], after["direction"], atol=1e-12)
    assert before["normalized_rhs"] == pytest.approx(after["normalized_rhs"], abs=1e-8)


def test_cross_synchronous_query_fails_closed_and_within_area_query_passes():
    e5, e6, _ = audit.declared_geometries(ZONES)
    cross = np.zeros(31)
    cross[ZONES.index("DK1")], cross[ZONES.index("DK2")] = 1., -1.
    audit.require_tangent_query(cross, e5)
    with pytest.raises(ValueError, match="outside the declared tangent"):
        audit.require_tangent_query(cross, e6)
    within = np.zeros(31)
    within[ZONES.index("FI")], within[ZONES.index("SE3")] = 1., -1.
    audit.require_tangent_query(within, e6)
    queries = audit.query_audit(ZONES, e6)
    assert queries["accepted_count"] == 36 and queries["rejected_count"] == 9


def test_geometric_equivalence_does_not_hide_changed_auxiliary_mark_or_dual_mass():
    _, e6, regional = audit.declared_geometries(ZONES)
    normal = np.zeros(31)
    normal[ZONES.index("DK2")] = 1.
    rows = []
    for current, shadow, fref in ((normal, 2., 0.), (normal - regional, 3., 19.)):
        length = np.linalg.norm(current)
        row = {"ptdf_l2_norm": length, "ram": 1152., "shadowPrice": shadow,
               "flowFb": 1152., "fmax": 1152., "fref": fref, "fall": 0.}
        row.update({"unit_ptdf_" + zone: value / length for zone, value in zip(ZONES, current)})
        rows.append(row)
    result = audit.row_comparison(rows[0], rows[1], ZONES, e6)
    assert result["geometric_row_support_equivalent"]
    assert not result["retained_auxiliary_mark_equal"]
    assert not result["shadow_price_equal"]
    assert not result["full_single_atom_measure_equal"]


def test_census_reports_constant_positive_dual_rows_without_normalizing_them():
    _, e6, regional = audit.declared_geometries(ZONES)
    rows = []
    for current in (regional, 2 * regional):
        length = np.linalg.norm(current)
        row = {"ptdf_l2_norm": length, "ram": 0., "shadowPrice": 1.}
        row.update({"unit_ptdf_" + zone: value / length for zone, value in zip(ZONES, current)})
        rows.append(row)
    result = census.classify(rows[0], rows[1], ZONES, e6)
    assert result["status"] == "fail_closed_projected_zero_normal"
    assert result["geometric_row_support_equivalent"] is None
    assert not result["normalized_direction_or_rhs_evaluated"]
