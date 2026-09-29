from __future__ import annotations

import numpy as np

from scripts.audit_af_cdmo_real_geometry_v13 import (
    ZERO_LOSS_INTERNAL_HVDC,
    build_loss_safe_geometry,
    tangent_projector,
)


ZONES = [
    "DK1",
    "DK1_CO",
    "DK1_DE",
    "DK1_KS",
    "DK1_SB",
    "DK1_SK",
    "DK2",
    "DK2_KO",
    "DK2_SB",
    "FI",
    "FI_EL",
    "FI_FS",
    "NO1",
    "NO2",
    "NO2_ND",
    "NO2_NK",
    "NO2_SK",
    "NO3",
    "NO4",
    "NO5",
    "SE1",
    "SE2",
    "SE3",
    "SE3_FS",
    "SE3_KS",
    "SE3_SWL",
    "SE4",
    "SE4_BC",
    "SE4_NB",
    "SE4_SP",
    "SE4_SWL",
]


def test_loss_safe_geometry_has_expected_rank() -> None:
    equality, names = build_loss_safe_geometry(ZONES)
    assert equality.shape == (1 + len(ZERO_LOSS_INTERNAL_HVDC), len(ZONES))
    assert names[0] == "global_net_position_balance"
    assert np.linalg.matrix_rank(equality) == 5


def test_all_physical_transfer_queries_are_tangent() -> None:
    equality, _ = build_loss_safe_geometry(ZONES)
    physical = ("DK1", "DK2", "FI", "NO1", "NO2", "NO3", "NO5", "SE1", "SE3", "SE4")
    eye = np.eye(len(ZONES))
    index = {zone: position for position, zone in enumerate(ZONES)}
    for left_index, left in enumerate(physical):
        for right in physical[left_index + 1 :]:
            query = eye[index[left]] - eye[index[right]]
            np.testing.assert_allclose(equality @ query, 0.0, atol=1.0e-12)


def test_projector_removes_arbitrary_equality_gauge() -> None:
    equality, _ = build_loss_safe_geometry(ZONES)
    projector = tangent_projector(equality)
    rng = np.random.default_rng(7)
    normals = rng.normal(size=(64, len(ZONES)))
    gauge = rng.normal(size=(64, equality.shape[0]))
    rewritten = normals + gauge @ equality
    np.testing.assert_allclose(
        rewritten @ projector,
        normals @ projector,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(projector @ projector, projector, atol=1.0e-12)


def test_projector_is_invariant_to_full_rank_equality_basis_rewrite() -> None:
    equality, _ = build_loss_safe_geometry(ZONES)
    projector = tangent_projector(equality)
    rng = np.random.default_rng(42)
    transform, _ = np.linalg.qr(rng.normal(size=(len(equality), len(equality))))
    transform = np.diag(np.linspace(0.4, 2.2, len(equality))) @ transform
    rewritten = transform @ equality
    assert np.linalg.matrix_rank(rewritten) == np.linalg.matrix_rank(equality)
    np.testing.assert_allclose(
        tangent_projector(rewritten), projector, atol=1.0e-12, rtol=1.0e-12
    )
