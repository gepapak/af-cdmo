from __future__ import annotations

import numpy as np

from af_cdmo import (
    AffineFeasibleGeometry,
    canonicalize_certificate,
    equality_gauge_scale_rewrite,
)


def _geometry() -> AffineFeasibleGeometry:
    return AffineFeasibleGeometry.from_equalities(
        np.asarray(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, -1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -1.0, 0.0],
            ]
        ),
        np.asarray([0.0, 0.4, -0.2]),
    )


def test_tall_full_column_rank_equality_rewrite_preserves_geometry() -> None:
    geometry = _geometry()
    transform = np.asarray(
        [
            [1.0, 0.2, -0.1],
            [0.3, 1.1, 0.4],
            [-0.2, 0.5, 0.9],
            [0.7, -0.4, 0.6],
            [1.2, 0.3, -0.8],
        ]
    )
    assert np.linalg.matrix_rank(transform) == 3
    rewritten = geometry.equivalent_basis(transform)
    assert rewritten.rank == geometry.rank
    np.testing.assert_allclose(
        rewritten.tangent_projector, geometry.tangent_projector, atol=2.0e-12
    )
    np.testing.assert_allclose(
        rewritten.base_point, geometry.base_point, atol=2.0e-12
    )


def test_equal_single_row_atom_recovers_registered_rewrite() -> None:
    geometry = _geometry()
    normal = np.asarray([[1.2, -0.4, 0.8, -0.9, 0.3]])
    projected = geometry.project(normal)[0]
    assert np.linalg.norm(projected) > 0.2
    rhs = np.asarray([normal[0] @ geometry.base_point + 0.7])
    dual = np.asarray([2.3])
    tail = np.asarray([[0.2, -0.5]])
    scale = np.asarray([3.7])
    gauge = np.asarray([[0.4, -0.8, 1.1]])
    changed_normal, changed_rhs, changed_dual = equality_gauge_scale_rewrite(
        geometry,
        normal,
        rhs,
        dual,
        scales=scale,
        gauge_coefficients=gauge,
    )

    first = canonicalize_certificate(geometry, normal, rhs, dual, tail)
    second = canonicalize_certificate(
        geometry, changed_normal, changed_rhs, changed_dual, tail
    )
    np.testing.assert_allclose(second.directions, first.directions, atol=1.0e-12)
    np.testing.assert_allclose(
        second.normalized_rhs, first.normalized_rhs, atol=1.0e-12
    )
    np.testing.assert_allclose(second.masses, first.masses, atol=1.0e-12)

    recovered_scale = (
        np.linalg.norm(geometry.project(changed_normal)[0])
        / np.linalg.norm(geometry.project(normal)[0])
    )
    residual_normal = changed_normal[0] - recovered_scale * normal[0]
    recovered_eta = np.linalg.lstsq(
        geometry.equality_matrix.T, residual_normal, rcond=None
    )[0]
    np.testing.assert_allclose(
        residual_normal,
        geometry.equality_matrix.T @ recovered_eta,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        changed_rhs[0],
        recovered_scale * rhs[0] + recovered_eta @ geometry.equality_rhs,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        changed_dual[0], dual[0] / recovered_scale, atol=1.0e-12
    )


def test_atom_stability_bounds_hold_away_from_degeneracy() -> None:
    geometry = _geometry()
    normal = np.asarray([1.2, -0.4, 0.8, -0.9, 0.3])
    rhs = float(normal @ geometry.base_point + 0.7)
    dual = 2.3
    perturbation = np.asarray([0.002, -0.001, 0.0005, 0.0015, -0.0008])
    changed_normal = normal + perturbation
    changed_rhs = rhs - 0.0012
    changed_dual = dual + 0.003

    p = geometry.project(normal)
    changed_p = geometry.project(changed_normal)
    r = float(np.linalg.norm(p))
    changed_r = float(np.linalg.norm(changed_p))
    epsilon = min(r, changed_r)
    assert epsilon > 0.1
    u = p / r
    changed_u = changed_p / changed_r
    s = rhs - normal @ geometry.base_point
    changed_s = changed_rhs - changed_normal @ geometry.base_point
    beta = s / r
    changed_beta = changed_s / changed_r
    mass = dual * r
    changed_mass = changed_dual * changed_r

    normal_delta = float(np.linalg.norm(normal - changed_normal))
    projected_delta = float(np.linalg.norm(p - changed_p))
    direction_bound = 2.0 * projected_delta / epsilon
    beta_bound = (
        abs(rhs - changed_rhs)
        + np.linalg.norm(geometry.base_point) * normal_delta
    ) / epsilon + abs(changed_s) * projected_delta / (epsilon**2)
    mass_bound = r * abs(dual - changed_dual) + abs(changed_dual) * projected_delta

    assert np.linalg.norm(u - changed_u) <= direction_bound + 1.0e-14
    assert abs(beta - changed_beta) <= beta_bound + 1.0e-14
    assert abs(mass - changed_mass) <= mass_bound + 1.0e-14


def test_zero_dual_rows_do_not_change_the_positive_measure() -> None:
    geometry = _geometry()
    normals = np.asarray(
        [
            [1.2, -0.4, 0.8, -0.9, 0.3],
            [-0.6, 0.2, 0.1, 0.7, -0.3],
        ]
    )
    rhs = normals @ geometry.base_point + np.asarray([0.7, -0.2])
    tail = np.asarray([[0.2], [99.0]])
    with_zero = canonicalize_certificate(
        geometry, normals, rhs, np.asarray([2.3, 0.0]), tail
    )
    without_zero = canonicalize_certificate(
        geometry, normals[:1], rhs[:1], np.asarray([2.3]), tail[:1]
    )
    np.testing.assert_allclose(with_zero.directions, without_zero.directions)
    np.testing.assert_allclose(with_zero.normalized_rhs, without_zero.normalized_rhs)
    np.testing.assert_allclose(with_zero.masses, without_zero.masses)
    np.testing.assert_allclose(with_zero.tail, without_zero.tail)
