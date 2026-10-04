from __future__ import annotations

import numpy as np
import pytest

from af_cdmo import (
    AffineFeasibleGeometry,
    DegenerateConstraintError,
    GeometryError,
    canonicalize_certificate,
    equality_gauge_scale_rewrite,
)


def _balanced_geometry() -> AffineFeasibleGeometry:
    return AffineFeasibleGeometry.from_equalities(
        np.asarray([[1.0, 1.0, 1.0]]), np.zeros(1)
    )


def test_gauge_that_erases_the_tangent_component_is_rejected() -> None:
    geometry = _balanced_geometry()
    normal = np.asarray([[1.0, -1.0, 0.0]])
    changed, rhs, dual = equality_gauge_scale_rewrite(
        geometry,
        normal,
        np.zeros(1),
        np.ones(1),
        scales=np.ones(1),
        gauge_coefficients=np.asarray([[1.0e16]]),
    )
    # At this magnitude float64 stores all three coefficients identically.
    np.testing.assert_array_equal(changed, np.full((1, 3), 1.0e16))
    with pytest.raises(DegenerateConstraintError, match="numerically unresolved"):
        canonicalize_certificate(geometry, changed, rhs, dual)


def test_large_resolved_gauge_still_preserves_the_measure() -> None:
    geometry = _balanced_geometry()
    normal = np.asarray([[1.0, -1.0, 0.0]])
    original = canonicalize_certificate(geometry, normal, np.zeros(1), np.ones(1))
    changed, rhs, dual = equality_gauge_scale_rewrite(
        geometry,
        normal,
        np.zeros(1),
        np.ones(1),
        scales=np.asarray([2.0]),
        gauge_coefficients=np.asarray([[1.0e3]]),
    )
    rewritten = canonicalize_certificate(geometry, changed, rhs, dual)
    np.testing.assert_allclose(rewritten.directions, original.directions, atol=1.0e-10)
    np.testing.assert_allclose(rewritten.masses, original.masses, atol=1.0e-10)


def test_direct_svd_preserves_a_resolved_anisotropic_basis() -> None:
    geometry = AffineFeasibleGeometry.from_equalities(
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), np.zeros(2)
    )
    equivalent = geometry.equivalent_basis(np.diag([1.0, 1.0e-8]))
    assert equivalent.rank == 2
    np.testing.assert_allclose(
        equivalent.tangent_projector, geometry.tangent_projector, atol=1.0e-13
    )


def test_full_rank_map_that_loses_resolved_affine_geometry_is_rejected() -> None:
    geometry = AffineFeasibleGeometry.from_equalities(
        np.asarray([[1.0e-8, 0.0, 0.0], [0.0, 1.0, 0.0]]), np.zeros(2)
    )
    transform = np.diag([1.0e-8, 1.0])
    assert np.linalg.matrix_rank(transform) == 2
    # The composite equality system loses its first resolved singular value.
    with pytest.raises(GeometryError, match="numerically resolved affine geometry"):
        geometry.equivalent_basis(transform)


def test_retained_mass_underflow_is_rejected() -> None:
    geometry = AffineFeasibleGeometry.from_equalities(np.empty((0, 2)), np.empty(0))
    with pytest.raises(GeometryError, match="Every retained canonical dual mass"):
        canonicalize_certificate(
            geometry,
            np.asarray([[1.0e-20, 0.0], [1.0, 0.0]]),
            np.zeros(2),
            np.asarray([1.0e-310, 1.0]),
            norm_tolerance=1.0e-30,
        )


def test_positive_rescaling_below_the_absolute_cutoff_fails_closed() -> None:
    geometry = _balanced_geometry()
    original = canonicalize_certificate(
        geometry, np.asarray([[1.0, -1.0, 0.0]]), np.zeros(1), np.ones(1)
    )
    assert original.total_mass == pytest.approx(np.sqrt(2.0))
    with pytest.raises(DegenerateConstraintError, match="numerically unresolved"):
        canonicalize_certificate(
            geometry,
            np.asarray([[1.0e-12, -1.0e-12, 0.0]]),
            np.zeros(1),
            np.asarray([1.0e12]),
        )


def test_finite_raw_bound_that_overflows_after_normalization_is_rejected() -> None:
    geometry = AffineFeasibleGeometry.from_equalities(np.empty((0, 2)), np.empty(0))
    with np.errstate(over="ignore"):
        with pytest.raises(GeometryError, match="Normalized certificate atoms"):
            canonicalize_certificate(
                geometry,
                np.asarray([[1.0e-10, 0.0]]),
                np.asarray([1.0e308]),
                np.ones(1),
            )
