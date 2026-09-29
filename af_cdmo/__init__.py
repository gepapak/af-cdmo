"""Isolated affine-feasible certificate dual-measure research extension."""

from .core import (
    AffineFeasibleGeometry,
    AffineFeasibleMeasure,
    DegenerateConstraintError,
    GeometryError,
    canonicalize_certificate,
    cotangent_gauge_rewrite,
    equality_gauge_scale_rewrite,
    feasible_transfer_value,
    measure_restricted_lagrangian,
    permute_certificate,
    restricted_lagrangian,
    split_certificate,
)
from .models import (
    MassMeasureCotangentNet,
    ProjectedRawCotangentDeepSet,
    collate_certificate_batch,
)

__all__ = [
    "AffineFeasibleGeometry",
    "AffineFeasibleMeasure",
    "DegenerateConstraintError",
    "GeometryError",
    "MassMeasureCotangentNet",
    "ProjectedRawCotangentDeepSet",
    "canonicalize_certificate",
    "collate_certificate_batch",
    "cotangent_gauge_rewrite",
    "equality_gauge_scale_rewrite",
    "feasible_transfer_value",
    "measure_restricted_lagrangian",
    "permute_certificate",
    "restricted_lagrangian",
    "split_certificate",
]
