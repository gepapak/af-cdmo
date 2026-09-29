from __future__ import annotations

import numpy as np
import pytest
import torch

from af_cdmo import (
    AffineFeasibleGeometry,
    DegenerateConstraintError,
    GeometryError,
    MassMeasureCotangentNet,
    canonicalize_certificate,
    collate_certificate_batch,
    cotangent_gauge_rewrite,
    equality_gauge_scale_rewrite,
    feasible_transfer_value,
    measure_restricted_lagrangian,
    permute_certificate,
    restricted_lagrangian,
    split_certificate,
)


def _geometry() -> AffineFeasibleGeometry:
    equality = np.asarray(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0, 0.0],
        ]
    )
    rhs = np.asarray([0.0, 0.4, -0.2])
    return AffineFeasibleGeometry.from_equalities(equality, rhs)


def _certificate() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    geometry = _geometry()
    tangent = geometry.project(
        np.asarray(
            [
                [1.3, -0.2, 0.8, -0.7, 0.1],
                [-0.5, 1.1, 0.2, 0.9, -0.8],
                [0.4, -0.9, 1.2, -0.1, 0.7],
                [0.8, 0.3, -0.6, 1.4, -0.4],
            ]
        )
    )
    base_gauge = np.asarray(
        [
            [0.2, -0.3, 0.1],
            [-0.4, 0.2, 0.5],
            [0.7, -0.1, -0.2],
            [0.1, 0.4, -0.6],
        ]
    )
    normals = tangent + base_gauge @ geometry.equality_matrix
    normalized_rhs = np.asarray([1.2, -0.4, 0.7, 1.8])
    projected_norm = np.linalg.norm(tangent, axis=1)
    rhs = normalized_rhs * projected_norm + normals @ geometry.base_point
    multipliers = np.asarray([3.0, 1.5, 2.2, 0.8])
    tail = np.asarray(
        [[0.1, 0.3], [0.2, -0.1], [-0.4, 0.6], [0.7, 0.2]]
    )
    return normals, rhs, multipliers, tail


def _coalesced_measure(measure, decimals: int = 9) -> tuple[np.ndarray, np.ndarray]:
    atoms = np.round(measure.atoms(rhs_scale=2.0), decimals=decimals)
    unique, inverse = np.unique(atoms, axis=0, return_inverse=True)
    mass = np.zeros(len(unique), dtype=np.float64)
    np.add.at(mass, inverse, measure.masses)
    order = np.lexsort(unique.T[::-1])
    return unique[order], mass[order]


def _combined_rewrite():
    geometry = _geometry()
    normals, rhs, multipliers, tail = _certificate()
    scales = np.asarray([0.2, 3.0, 1.7, 8.0])
    gauge = np.asarray(
        [[1.1, -0.7, 0.4], [-0.2, 0.3, 1.0], [0.8, 0.6, -0.5], [0.1, -1.2, 0.9]]
    )
    normals, rhs, multipliers = equality_gauge_scale_rewrite(
        geometry,
        normals,
        rhs,
        multipliers,
        scales=scales,
        gauge_coefficients=gauge,
    )
    normals, rhs, multipliers, tail = split_certificate(
        normals,
        rhs,
        multipliers,
        tail,
        fractions=np.asarray([0.15, 0.35, 0.65, 0.82]),
    )
    order = np.asarray([6, 1, 4, 7, 0, 5, 3, 2])
    return permute_certificate(
        normals, rhs, multipliers, tail, order=order
    )


def _network_item(measure, target: np.ndarray) -> dict:
    return {
        "atoms": measure.atoms(rhs_scale=2.0).astype(np.float32),
        "weights": measure.weights.astype(np.float32),
        "total_mass": measure.total_mass,
        "raw": np.zeros((len(measure.weights), 1), dtype=np.float32),
        "analytic": measure.analytic_cotangent.astype(np.float32),
        "target": target.astype(np.float32),
        "timestamp": "synthetic",
    }


def test_projector_is_symmetric_idempotent_and_annihilates_equalities() -> None:
    geometry = _geometry()
    projector = geometry.tangent_projector
    np.testing.assert_allclose(projector, projector.T, atol=1.0e-13)
    np.testing.assert_allclose(projector @ projector, projector, atol=1.0e-13)
    np.testing.assert_allclose(
        geometry.equality_matrix @ projector, 0.0, atol=1.0e-13
    )
    np.testing.assert_allclose(
        geometry.equality_matrix @ geometry.base_point,
        geometry.equality_rhs,
        atol=1.0e-13,
    )


def test_full_certificate_quotient_is_invariant_to_combined_orbit() -> None:
    geometry = _geometry()
    original = canonicalize_certificate(geometry, *_certificate())
    rewritten = canonicalize_certificate(geometry, *_combined_rewrite())
    original_atoms, original_mass = _coalesced_measure(original)
    rewritten_atoms, rewritten_mass = _coalesced_measure(rewritten)
    np.testing.assert_allclose(rewritten_atoms, original_atoms, atol=1.0e-9)
    np.testing.assert_allclose(rewritten_mass, original_mass, atol=1.0e-10)
    np.testing.assert_allclose(
        rewritten.dual_current, original.dual_current, atol=1.0e-11
    )
    np.testing.assert_allclose(
        rewritten.moment_signature(rhs_scale=2.0),
        original.moment_signature(rhs_scale=2.0),
        atol=1.0e-10,
    )


def test_quotient_is_invariant_to_equality_basis_change() -> None:
    geometry = _geometry()
    transform = np.asarray(
        [[1.0, 0.2, -0.1], [0.4, 1.3, 0.5], [-0.2, 0.1, 0.9]]
    )
    equivalent = geometry.equivalent_basis(transform)
    first = canonicalize_certificate(geometry, *_certificate())
    second = canonicalize_certificate(equivalent, *_certificate())
    np.testing.assert_allclose(
        equivalent.tangent_projector, geometry.tangent_projector, atol=1.0e-12
    )
    np.testing.assert_allclose(
        equivalent.base_point, geometry.base_point, atol=1.0e-12
    )
    np.testing.assert_allclose(
        second.moment_signature(), first.moment_signature(), atol=1.0e-11
    )


def test_measure_preserves_restricted_lagrangian_and_dual_current() -> None:
    geometry = _geometry()
    normals, rhs, multipliers, tail = _certificate()
    measure = canonicalize_certificate(geometry, normals, rhs, multipliers, tail)
    displacement = geometry.project(np.asarray([0.7, -0.4, 0.3, 1.2, -0.2]))
    direct = restricted_lagrangian(
        geometry, normals, rhs, multipliers, displacement
    )
    quotient = measure_restricted_lagrangian(geometry, measure, displacement)
    assert quotient == pytest.approx(direct, abs=1.0e-11)
    rewritten_normals, rewritten_rhs, rewritten_dual, rewritten_tail = (
        _combined_rewrite()
    )
    rewritten_direct = restricted_lagrangian(
        geometry,
        rewritten_normals,
        rewritten_rhs,
        rewritten_dual,
        displacement,
    )
    assert rewritten_direct == pytest.approx(direct, abs=1.0e-11)
    rewritten_measure = canonicalize_certificate(
        geometry,
        rewritten_normals,
        rewritten_rhs,
        rewritten_dual,
        rewritten_tail,
    )
    assert measure_restricted_lagrangian(
        geometry, rewritten_measure, displacement
    ) == pytest.approx(direct, abs=1.0e-11)
    np.testing.assert_allclose(
        measure.dual_current,
        geometry.project(multipliers @ normals),
        atol=1.0e-12,
    )


def test_output_is_a_feasible_cotangent_equivalence_class() -> None:
    geometry = _geometry()
    field = np.asarray([3.0, -1.0, 0.4, 2.0, -0.7])
    rewritten = cotangent_gauge_rewrite(
        geometry, field, np.asarray([0.8, -0.3, 1.1])
    )
    np.testing.assert_allclose(
        geometry.project(rewritten), geometry.project(field), atol=1.0e-12
    )
    displacement = geometry.project(np.asarray([-0.2, 0.5, 1.4, -0.6, 0.1]))
    assert feasible_transfer_value(
        geometry, rewritten, displacement
    ) == pytest.approx(
        feasible_transfer_value(geometry, field, displacement), abs=1.0e-12
    )


def test_network_prediction_is_invariant_to_complete_certificate_orbit() -> None:
    torch.manual_seed(7)
    geometry = _geometry()
    original = canonicalize_certificate(geometry, *_certificate())
    rewritten = canonicalize_certificate(geometry, *_combined_rewrite())
    target = geometry.project(np.asarray([1.5, -0.4, 0.7, -1.0, 0.2]))
    model = MassMeasureCotangentNet(
        original.atoms(rhs_scale=2.0).shape[1], geometry.tangent_projector
    ).eval()
    with torch.no_grad():
        first = model(collate_certificate_batch([_network_item(original, target)]))
        second = model(collate_certificate_batch([_network_item(rewritten, target)]))
    torch.testing.assert_close(first, second, atol=3.0e-6, rtol=3.0e-6)
    residual = (
        first.detach().cpu().numpy()[0] @ geometry.equality_matrix.T
    )
    np.testing.assert_allclose(residual, 0.0, atol=3.0e-6)


def test_network_prediction_is_invariant_to_equality_basis_change() -> None:
    torch.manual_seed(42)
    geometry = _geometry()
    equivalent = geometry.equivalent_basis(
        np.asarray([[1.0, 0.3, -0.2], [-0.1, 1.2, 0.4], [0.25, -0.15, 0.9]])
    )
    original = canonicalize_certificate(geometry, *_certificate())
    rewritten = canonicalize_certificate(equivalent, *_certificate())
    target = geometry.project(np.asarray([0.2, 1.0, -0.8, 0.3, -0.4]))
    model = MassMeasureCotangentNet(
        original.atoms(rhs_scale=2.0).shape[1], geometry.tangent_projector
    ).eval()
    with torch.no_grad():
        first = model(collate_certificate_batch([_network_item(original, target)]))
        second = model(collate_certificate_batch([_network_item(rewritten, target)]))
    torch.testing.assert_close(first, second, atol=3.0e-6, rtol=3.0e-6)


def test_global_slack_quotient_does_not_remove_higher_rank_gauge() -> None:
    full = _geometry()
    normals, rhs, multipliers, tail = _certificate()
    global_only = AffineFeasibleGeometry.from_equalities(
        full.equality_matrix[:1], full.equality_rhs[:1]
    )
    before = canonicalize_certificate(global_only, normals, rhs, multipliers, tail)
    gauge = np.zeros((len(normals), 3), dtype=np.float64)
    gauge[:, 1] = np.asarray([0.5, -0.3, 0.8, -0.2])
    changed_normals, changed_rhs, changed_dual = equality_gauge_scale_rewrite(
        full,
        normals,
        rhs,
        multipliers,
        scales=np.ones(len(normals)),
        gauge_coefficients=gauge,
    )
    after = canonicalize_certificate(
        global_only, changed_normals, changed_rhs, changed_dual, tail
    )
    assert not np.allclose(
        before.moment_signature(), after.moment_signature(), atol=1.0e-6
    )


def test_positive_dual_constraint_constant_on_manifold_is_rejected() -> None:
    geometry = _geometry()
    rowspace_normal = geometry.equality_matrix[1:2].copy()
    rhs = rowspace_normal @ geometry.base_point
    with pytest.raises(DegenerateConstraintError, match="constant on the feasible"):
        canonicalize_certificate(
            geometry,
            rowspace_normal,
            rhs.ravel(),
            np.asarray([1.0]),
            np.zeros((1, 0)),
        )


def test_inconsistent_equality_system_is_rejected() -> None:
    with pytest.raises(GeometryError, match="inconsistent"):
        AffineFeasibleGeometry.from_equalities(
            np.asarray([[1.0, 0.0], [1.0, 0.0]]),
            np.asarray([0.0, 1.0]),
        )


def test_negative_dual_is_rejected() -> None:
    geometry = _geometry()
    normals, rhs, multipliers, tail = _certificate()
    multipliers[2] = -0.1
    with pytest.raises(GeometryError, match="non-negative"):
        canonicalize_certificate(geometry, normals, rhs, multipliers, tail)
