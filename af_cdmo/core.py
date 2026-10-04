"""Algebra for an affine-feasible certificate dual-measure operator.

The code in this package is deliberately isolated from the frozen Slack-QDM
v11 campaign.  It implements the generic mathematics only; it does not assert
that the candidate seven-row Nordic equality model is operationally exact.

For a feasible affine set M = {x : E x = e}, an inequality certificate row
(a, b, lambda) has the same restriction to M after

    a'      = c a + eta E,
    b'      = c b + eta e,
    lambda' = lambda / c,

for c > 0.  AF-CDMO projects a onto ker(E), translates b relative to the
minimum-norm point of M, and then applies the CQDM scale/refinement quotient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class GeometryError(ValueError):
    """Raised when an affine feasible geometry is malformed or inconsistent."""


class DegenerateConstraintError(ValueError):
    """Raised when a positive-dual row is constant on the feasible manifold."""


def _finite_array(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise GeometryError(f"{name} must have {ndim} dimensions; got {array.shape}")
    if not np.isfinite(array).all():
        raise GeometryError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True)
class AffineFeasibleGeometry:
    """Canonical Euclidean geometry of ``{x : E x = e}``.

    ``tangent_projector`` is the unique orthogonal projector onto ``ker(E)``.
    ``base_point`` is the unique minimum-norm point in the affine set.  Both
    depend only on the represented affine set, not on the chosen equality-row
    basis, including redundant rows.
    """

    equality_matrix: np.ndarray
    equality_rhs: np.ndarray
    rowspace_projector: np.ndarray
    tangent_projector: np.ndarray
    base_point: np.ndarray
    rank: int
    singular_tolerance: float
    consistency_tolerance: float

    @classmethod
    def from_equalities(
        cls,
        equality_matrix: Any,
        equality_rhs: Any,
        *,
        singular_tolerance: float | None = None,
        consistency_tolerance: float = 1.0e-10,
    ) -> "AffineFeasibleGeometry":
        matrix = _finite_array(equality_matrix, name="equality_matrix", ndim=2)
        rhs = _finite_array(equality_rhs, name="equality_rhs", ndim=1)
        if matrix.shape[0] != rhs.shape[0]:
            raise GeometryError(
                "equality_rhs length must match equality_matrix row count"
            )
        if matrix.shape[1] < 2:
            raise GeometryError("The ambient decision space must have dimension >= 2")
        if matrix.shape[0] == 0:
            projector = np.eye(matrix.shape[1], dtype=np.float64)
            return cls(
                equality_matrix=matrix.copy(),
                equality_rhs=rhs.copy(),
                rowspace_projector=np.zeros_like(projector),
                tangent_projector=projector,
                base_point=np.zeros(matrix.shape[1], dtype=np.float64),
                rank=0,
                singular_tolerance=0.0,
                consistency_tolerance=float(consistency_tolerance),
            )
        if consistency_tolerance <= 0.0:
            raise GeometryError("consistency_tolerance must be positive")

        _, singular_values, right_vectors = np.linalg.svd(matrix, full_matrices=True)
        automatic_tolerance = (
            max(matrix.shape)
            * np.finfo(np.float64).eps
            * float(singular_values.max(initial=0.0))
        )
        tolerance = (
            automatic_tolerance
            if singular_tolerance is None
            else float(singular_tolerance)
        )
        if tolerance < 0.0:
            raise GeometryError("singular_tolerance must be non-negative")
        rank = int(np.sum(singular_values > tolerance))
        row_basis = right_vectors[:rank].T
        rowspace = row_basis @ row_basis.T
        rowspace = 0.5 * (rowspace + rowspace.T)
        tangent = np.eye(matrix.shape[1], dtype=np.float64) - rowspace
        tangent = 0.5 * (tangent + tangent.T)
        largest_singular = float(singular_values.max(initial=0.0))
        relative_rcond = (
            tolerance / largest_singular
            if largest_singular > 0.0
            else 1.0e-15
        )
        base_point = np.linalg.pinv(
            matrix, rcond=max(relative_rcond, 1.0e-15)
        ) @ rhs

        residual = matrix @ base_point - rhs
        residual_scale = max(
            1.0,
            float(np.linalg.norm(rhs)),
            float(np.linalg.norm(matrix, ord=2) * np.linalg.norm(base_point)),
        )
        if float(np.linalg.norm(residual)) > consistency_tolerance * residual_scale:
            raise GeometryError(
                "Equality system is inconsistent: minimum-norm residual exceeds "
                f"tolerance ({np.linalg.norm(residual):.3e})"
            )

        identity_error = float(np.max(np.abs(tangent @ tangent - tangent)))
        orthogonality_error = float(np.max(np.abs(matrix @ tangent)))
        numerical_limit = 100.0 * max(automatic_tolerance, np.finfo(float).eps)
        if identity_error > max(numerical_limit, 1.0e-11):
            raise GeometryError("Failed to construct an idempotent tangent projector")
        if orthogonality_error > max(numerical_limit, 1.0e-10):
            raise GeometryError("Tangent projector is not in the equality nullspace")

        return cls(
            equality_matrix=matrix.copy(),
            equality_rhs=rhs.copy(),
            rowspace_projector=rowspace,
            tangent_projector=tangent,
            base_point=base_point,
            rank=rank,
            singular_tolerance=tolerance,
            consistency_tolerance=float(consistency_tolerance),
        )

    @property
    def ambient_dimension(self) -> int:
        return int(self.equality_matrix.shape[1])

    @property
    def tangent_dimension(self) -> int:
        return self.ambient_dimension - self.rank

    def project(self, values: Any) -> np.ndarray:
        """Project vectors in the final dimension onto feasible displacements."""

        array = _finite_array(values, name="values")
        if array.shape[-1] != self.ambient_dimension:
            raise GeometryError(
                f"Expected final dimension {self.ambient_dimension}; got {array.shape}"
            )
        return array @ self.tangent_projector

    def assert_feasible_displacement(
        self, displacement: Any, *, tolerance: float = 1.0e-9
    ) -> np.ndarray:
        value = _finite_array(displacement, name="displacement", ndim=1)
        if value.shape != (self.ambient_dimension,):
            raise GeometryError("Displacement has the wrong ambient dimension")
        residual = self.equality_matrix @ value
        if float(np.max(np.abs(residual), initial=0.0)) > tolerance:
            raise GeometryError("Displacement is not tangent to the feasible manifold")
        return value

    def equivalent_basis(self, transform: Any) -> "AffineFeasibleGeometry":
        """Return an equivalent equality representation after a full-rank row map."""

        change = _finite_array(transform, name="transform", ndim=2)
        if change.shape[1] != self.equality_matrix.shape[0]:
            raise GeometryError("Equality-basis transform has incompatible width")
        if np.linalg.matrix_rank(change) < self.equality_matrix.shape[0]:
            raise GeometryError("Equality-basis transform must have full column rank")
        equivalent = AffineFeasibleGeometry.from_equalities(
            change @ self.equality_matrix,
            change @ self.equality_rhs,
            consistency_tolerance=self.consistency_tolerance,
        )
        # Full column rank is an exact-arithmetic condition.  A numerically
        # resolved row map must also preserve the geometry actually computed.
        geometry_tolerance = max(
            self.consistency_tolerance,
            100.0 * np.finfo(np.float64).eps * self.ambient_dimension,
        )
        point_scale = max(1.0, float(np.linalg.norm(self.base_point)))
        if (
            equivalent.rank != self.rank
            or np.linalg.norm(
                equivalent.tangent_projector - self.tangent_projector, ord=2
            ) > geometry_tolerance
            or np.linalg.norm(equivalent.base_point - self.base_point)
            > geometry_tolerance * point_scale
        ):
            raise GeometryError(
                "Equality-basis rewrite does not preserve the numerically "
                "resolved affine geometry"
            )
        return equivalent


@dataclass(frozen=True)
class AffineFeasibleMeasure:
    """Canonical positive dual measure on affine-feasible constraint atoms."""

    directions: np.ndarray
    normalized_rhs: np.ndarray
    tail: np.ndarray
    masses: np.ndarray
    weights: np.ndarray
    total_mass: float
    dual_current: np.ndarray
    analytic_cotangent: np.ndarray
    active_indices: np.ndarray

    def atoms(self, *, rhs_scale: float = 1.0) -> np.ndarray:
        if rhs_scale <= 0.0:
            raise ValueError("rhs_scale must be positive")
        transformed_rhs = np.arcsinh(self.normalized_rhs / rhs_scale)
        return np.column_stack([self.directions, transformed_rhs, self.tail])

    def moment_signature(self, *, rhs_scale: float = 1.0) -> np.ndarray:
        """Permutation/refinement-invariant diagnostics, not a complete invariant."""

        atoms = self.atoms(rhs_scale=rhs_scale)
        if not len(self.masses):
            return np.zeros(2 + 2 * atoms.shape[1] + len(self.dual_current))
        mean = self.weights @ atoms
        second = self.weights @ np.square(atoms)
        return np.concatenate(
            [
                np.asarray([self.total_mass, np.log1p(self.total_mass)]),
                mean,
                second,
                self.dual_current,
            ]
        )


def canonicalize_certificate(
    geometry: AffineFeasibleGeometry,
    normals: Any,
    rhs: Any,
    multipliers: Any,
    tail: Any | None = None,
    *,
    norm_tolerance: float = 1.0e-11,
    dual_tolerance: float = 0.0,
) -> AffineFeasibleMeasure:
    """Construct the AF-CDMO quotient of a positive inequality-dual certificate."""

    matrix = _finite_array(normals, name="normals", ndim=2)
    bounds = _finite_array(rhs, name="rhs", ndim=1)
    dual = _finite_array(multipliers, name="multipliers", ndim=1)
    if matrix.shape[1] != geometry.ambient_dimension:
        raise GeometryError("Certificate normals have the wrong ambient dimension")
    if matrix.shape[0] != len(bounds) or len(bounds) != len(dual):
        raise GeometryError("Certificate row arrays must have equal lengths")
    if (
        not np.isfinite(norm_tolerance)
        or not np.isfinite(dual_tolerance)
        or norm_tolerance <= 0.0
        or dual_tolerance < 0.0
    ):
        raise ValueError("Tolerances must be non-negative with norm_tolerance > 0")
    if np.any(dual < -dual_tolerance):
        raise GeometryError("Inequality multipliers must be non-negative")

    if tail is None:
        row_tail = np.empty((len(matrix), 0), dtype=np.float64)
    else:
        row_tail = _finite_array(tail, name="tail", ndim=2)
        if row_tail.shape[0] != len(matrix):
            raise GeometryError("tail must have one row per certificate row")

    active = np.flatnonzero(dual > dual_tolerance)
    if not len(active):
        dimension = geometry.ambient_dimension
        return AffineFeasibleMeasure(
            directions=np.empty((0, dimension), dtype=np.float64),
            normalized_rhs=np.empty(0, dtype=np.float64),
            tail=np.empty((0, row_tail.shape[1]), dtype=np.float64),
            masses=np.empty(0, dtype=np.float64),
            weights=np.empty(0, dtype=np.float64),
            total_mass=0.0,
            dual_current=np.zeros(dimension, dtype=np.float64),
            analytic_cotangent=np.zeros(dimension, dtype=np.float64),
            active_indices=active,
        )

    active_normals = matrix[active]
    active_rhs = bounds[active]
    active_dual = dual[active]
    projected = geometry.project(active_normals)
    effective_rhs = active_rhs - active_normals @ geometry.base_point
    norms = np.linalg.norm(projected, axis=1)
    ambient_norms = np.linalg.norm(active_normals, axis=1)
    if (
        not np.isfinite(projected).all()
        or not np.isfinite(effective_rhs).all()
        or not np.isfinite(norms).all()
        or not np.isfinite(ambient_norms).all()
    ):
        raise GeometryError("Certificate projection produced non-finite values")
    # An enormous equality gauge can destroy a small tangent component before
    # normalization.  A nonzero computed norm alone cannot detect that loss.
    # This conservative roundoff-resolution floor is an acceptance guard,
    # not a claim of a uniform error bound for arbitrary presentations.
    resolution_floor = (
        32.0 * np.finfo(np.float64).eps * geometry.ambient_dimension * ambient_norms
    )
    degenerate = norms <= np.maximum(norm_tolerance, resolution_floor)
    if np.any(degenerate):
        rows = active[degenerate].tolist()
        raise DegenerateConstraintError(
            "Positive-dual rows are constant on the feasible manifold or their "
            f"projected normals are numerically unresolved for rows {rows}"
        )
    tangent_residual = np.linalg.norm(
        projected @ geometry.rowspace_projector, axis=1
    )
    tangent_tolerance = max(
        1.0e-10, 100.0 * np.finfo(np.float64).eps * geometry.ambient_dimension
    )
    if np.any(tangent_residual > tangent_tolerance * norms):
        raise GeometryError("Projected certificate normals failed the tangent check")

    directions = projected / norms[:, None]
    normalized_rhs = effective_rhs / norms
    if not np.isfinite(directions).all() or not np.isfinite(normalized_rhs).all():
        raise GeometryError("Normalized certificate atoms must remain finite")
    masses = active_dual * norms
    if not np.isfinite(masses).all() or np.any(masses <= 0.0):
        raise GeometryError(
            "Every retained canonical dual mass must be finite and positive"
        )
    total_mass = float(masses.sum())
    if not np.isfinite(total_mass) or total_mass <= 0.0:
        raise GeometryError("Canonical dual mass must be finite and positive")
    weights = masses / total_mass
    dual_current = masses @ directions
    analytic = -geometry.project(dual_current)
    return AffineFeasibleMeasure(
        directions=directions,
        normalized_rhs=normalized_rhs,
        tail=row_tail[active],
        masses=masses,
        weights=weights,
        total_mass=total_mass,
        dual_current=dual_current,
        analytic_cotangent=analytic,
        active_indices=active,
    )


def equality_gauge_scale_rewrite(
    geometry: AffineFeasibleGeometry,
    normals: Any,
    rhs: Any,
    multipliers: Any,
    *,
    scales: Any,
    gauge_coefficients: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the complete positive-scale/equality-gauge row equivalence."""

    matrix = _finite_array(normals, name="normals", ndim=2)
    bounds = _finite_array(rhs, name="rhs", ndim=1)
    dual = _finite_array(multipliers, name="multipliers", ndim=1)
    scale = _finite_array(scales, name="scales", ndim=1)
    gauge = _finite_array(
        gauge_coefficients, name="gauge_coefficients", ndim=2
    )
    row_count = matrix.shape[0]
    if matrix.shape[1] != geometry.ambient_dimension:
        raise GeometryError("Certificate normals have the wrong ambient dimension")
    if any(len(value) != row_count for value in (bounds, dual, scale)):
        raise GeometryError("Rewrite vectors must match certificate row count")
    if gauge.shape != (row_count, geometry.equality_matrix.shape[0]):
        raise GeometryError("Gauge coefficients have the wrong shape")
    if np.any(scale <= 0.0):
        raise GeometryError("All row scales must be strictly positive")
    rewritten_normals = scale[:, None] * matrix + gauge @ geometry.equality_matrix
    rewritten_rhs = scale * bounds + gauge @ geometry.equality_rhs
    rewritten_dual = dual / scale
    return rewritten_normals, rewritten_rhs, rewritten_dual


def split_certificate(
    normals: Any,
    rhs: Any,
    multipliers: Any,
    tail: Any,
    *,
    fractions: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split every row into two coincident atoms while conserving dual mass."""

    matrix = _finite_array(normals, name="normals", ndim=2)
    bounds = _finite_array(rhs, name="rhs", ndim=1)
    dual = _finite_array(multipliers, name="multipliers", ndim=1)
    row_tail = _finite_array(tail, name="tail", ndim=2)
    alpha = _finite_array(fractions, name="fractions", ndim=1)
    if not (
        len(matrix) == len(bounds) == len(dual) == len(row_tail) == len(alpha)
    ):
        raise GeometryError("Split arrays must share a row count")
    if np.any((alpha <= 0.0) | (alpha >= 1.0)):
        raise GeometryError("Split fractions must lie strictly inside (0, 1)")
    return (
        np.concatenate([matrix, matrix], axis=0),
        np.concatenate([bounds, bounds]),
        np.concatenate([dual * alpha, dual * (1.0 - alpha)]),
        np.concatenate([row_tail, row_tail], axis=0),
    )


def permute_certificate(
    normals: Any,
    rhs: Any,
    multipliers: Any,
    tail: Any,
    *,
    order: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matrix = _finite_array(normals, name="normals", ndim=2)
    bounds = _finite_array(rhs, name="rhs", ndim=1)
    dual = _finite_array(multipliers, name="multipliers", ndim=1)
    row_tail = _finite_array(tail, name="tail", ndim=2)
    permutation = np.asarray(order, dtype=np.int64)
    if permutation.shape != (len(matrix),) or set(permutation.tolist()) != set(
        range(len(matrix))
    ):
        raise GeometryError("order must be a complete row permutation")
    return (
        matrix[permutation],
        bounds[permutation],
        dual[permutation],
        row_tail[permutation],
    )


def restricted_lagrangian(
    geometry: AffineFeasibleGeometry,
    normals: Any,
    rhs: Any,
    multipliers: Any,
    displacement: Any,
) -> float:
    delta = geometry.assert_feasible_displacement(displacement)
    matrix = _finite_array(normals, name="normals", ndim=2)
    bounds = _finite_array(rhs, name="rhs", ndim=1)
    dual = _finite_array(multipliers, name="multipliers", ndim=1)
    if not (len(matrix) == len(bounds) == len(dual)):
        raise GeometryError("Certificate arrays must share a row count")
    point = geometry.base_point + delta
    return float(dual @ (matrix @ point - bounds))


def measure_restricted_lagrangian(
    geometry: AffineFeasibleGeometry,
    measure: AffineFeasibleMeasure,
    displacement: Any,
) -> float:
    delta = geometry.assert_feasible_displacement(displacement)
    if not len(measure.masses):
        return 0.0
    return float(
        measure.masses
        @ (measure.directions @ delta - measure.normalized_rhs)
    )


def cotangent_gauge_rewrite(
    geometry: AffineFeasibleGeometry, cotangent: Any, coefficients: Any
) -> np.ndarray:
    field = _finite_array(cotangent, name="cotangent", ndim=1)
    gauge = _finite_array(coefficients, name="coefficients", ndim=1)
    if field.shape != (geometry.ambient_dimension,):
        raise GeometryError("Cotangent has the wrong ambient dimension")
    if gauge.shape != (geometry.equality_matrix.shape[0],):
        raise GeometryError("Cotangent gauge coefficients have the wrong dimension")
    return field + gauge @ geometry.equality_matrix


def feasible_transfer_value(
    geometry: AffineFeasibleGeometry, cotangent: Any, displacement: Any
) -> float:
    field = _finite_array(cotangent, name="cotangent", ndim=1)
    delta = geometry.assert_feasible_displacement(displacement)
    if field.shape != (geometry.ambient_dimension,):
        raise GeometryError("Cotangent has the wrong ambient dimension")
    return float(geometry.project(field) @ delta)
