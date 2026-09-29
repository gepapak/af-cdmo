"""Algebra and datasets for the isolated Slack-QDM v11 extension.

Slack-QDM quotients each published PTDF row by the rank-one global-balance
gauge before applying the existing CQDM scale/refinement quotient.  For a
balanced net-position vector ``x`` with ``1.T @ x = 0``, the presentations
``a`` and ``a + eta * 1`` define the same inequality.  The unique orthogonal
representative is ``P0 @ a`` where ``P0 = I - 11.T / d``.

This module intentionally imports the frozen v6 model/collator.  It adds a new
input representation without changing completed v6/v10 engines or evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from torch.utils.data import Dataset

try:
    from scripts.benchmark_certificate_governance_v5 import _stress_certificate
    from scripts.multizone_quotient_gauge_v6 import center_price_field, field_to_pairwise
except ModuleNotFoundError:
    from benchmark_certificate_governance_v5 import _stress_certificate
    from multizone_quotient_gauge_v6 import center_price_field, field_to_pairwise


REPRESENTATIONS = ("ambient", "slack")
BASE_PRESENTATIONS = (
    "original",
    "combined",
    "projected_representative",
    "combined_projected_representative",
)


def global_balance_project(normals: np.ndarray) -> np.ndarray:
    """Project row normals onto the globally balanced net-position subspace."""

    values = np.asarray(normals, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("PTDF normals must be a two-dimensional d>=2 array")
    if not np.isfinite(values).all():
        raise ValueError("PTDF normals must be finite")
    return values - values.mean(axis=1, keepdims=True)


def global_slack_rewrite(normals: np.ndarray, slack_index: int) -> np.ndarray:
    """Express every row relative to one registered global slack zone."""

    values = np.asarray(normals, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("PTDF normals must be two-dimensional")
    if not 0 <= int(slack_index) < values.shape[1]:
        raise IndexError(f"Slack index {slack_index} outside [0, {values.shape[1]})")
    return values - values[:, [int(slack_index)]]


def presentation_names(zone_names: Iterable[str]) -> tuple[str, ...]:
    """Return the preregistered global-slack and row-rewrite stress orbit."""

    names = tuple(zone_names)
    if len(names) != len(set(names)) or len(names) < 2:
        raise ValueError("Global-slack zone names must be unique and non-trivial")
    return (
        *BASE_PRESENTATIONS,
        *(f"slack_{name}" for name in names),
        *(f"combined_slack_{name}" for name in names),
    )


@dataclass(frozen=True)
class Presentation:
    name: str
    combined: bool
    slack_index: int | None = None
    projected_representative: bool = False


def parse_presentation(name: str, zone_names: Iterable[str]) -> Presentation:
    zones = tuple(zone_names)
    if name == "original":
        return Presentation(name=name, combined=False)
    if name == "combined":
        return Presentation(name=name, combined=True)
    if name == "projected_representative":
        return Presentation(name=name, combined=False, projected_representative=True)
    if name == "combined_projected_representative":
        return Presentation(name=name, combined=True, projected_representative=True)
    prefix = "combined_slack_" if name.startswith("combined_slack_") else "slack_"
    if not name.startswith(prefix):
        raise ValueError(f"Unknown v11 certificate presentation: {name}")
    zone = name[len(prefix) :]
    if zone not in zones:
        raise ValueError(f"Unknown global slack zone in presentation {name}: {zone}")
    return Presentation(
        name=name,
        combined=prefix == "combined_slack_",
        slack_index=zones.index(zone),
    )


def _apply_combined_row_rewrite(
    shadow: np.ndarray,
    normals: np.ndarray,
    rhs: np.ndarray,
    raw_tail: np.ndarray,
    canonical_tail: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply the frozen v5 scaling/split/permutation rewrite in explicit form."""

    rng = np.random.default_rng(seed)
    scale = np.exp(rng.uniform(np.log(0.05), np.log(20.0), size=len(normals)))
    shadow = shadow / scale
    normals = normals * scale[:, None]
    rhs = rhs * scale

    alpha = rng.uniform(0.1, 0.9, size=len(normals))
    shadow = np.concatenate([shadow * alpha, shadow * (1.0 - alpha)])
    normals = np.concatenate([normals, normals], axis=0)
    rhs = np.concatenate([rhs, rhs])
    raw_tail = np.concatenate([raw_tail, raw_tail], axis=0)
    canonical_tail = np.concatenate([canonical_tail, canonical_tail], axis=0)

    order = rng.permutation(len(normals))
    return (
        shadow[order],
        normals[order],
        rhs[order],
        raw_tail[order],
        canonical_tail[order],
    )


class SlackGaugeDataset(Dataset):
    """Expose ambient or Slack-QDM views of the same certificate samples.

    ``samples`` must be produced by ``_build_vector_samples`` from the frozen
    v6 benchmark.  The raw layout is ``[lambda, a..., ram, flowFb, ...]`` and
    the canonical layout is ``[a/||a||..., asinh(ram/||a||/scale), ...]``.
    Degenerate projected rows are rejected rather than silently regularized.
    """

    def __init__(
        self,
        samples: list[dict[str, Any]],
        raw_scale: np.ndarray,
        target_scale: float,
        presentation: str,
        representation: str,
        real_zone_indices: np.ndarray,
        ptdf_count: int,
        zone_names: Iterable[str],
        canonical_ram_scale: float,
        *,
        norm_tolerance: float = 1.0e-12,
    ) -> None:
        if representation not in REPRESENTATIONS:
            raise ValueError(f"Unknown representation: {representation}")
        if target_scale <= 0.0 or canonical_ram_scale <= 0.0:
            raise ValueError("Target and canonical-RAM scales must be positive")
        if norm_tolerance <= 0.0:
            raise ValueError("Projected-norm tolerance must be positive")
        self.samples = samples
        self.raw_scale = np.asarray(raw_scale, dtype=np.float64)
        self.target_scale = float(target_scale)
        self.presentation = parse_presentation(presentation, zone_names)
        self.representation = representation
        self.real_zone_indices = np.asarray(real_zone_indices, dtype=np.int64)
        self.ptdf_count = int(ptdf_count)
        self.zone_names = tuple(zone_names)
        self.canonical_ram_scale = float(canonical_ram_scale)
        self.norm_tolerance = float(norm_tolerance)
        self.epoch = 0

        expected_raw_dim = 1 + self.ptdf_count + 5
        if self.raw_scale.shape != (expected_raw_dim,):
            raise ValueError(
                f"raw_scale has shape {self.raw_scale.shape}; expected {(expected_raw_dim,)}"
            )
        if np.any(self.raw_scale <= 0.0) or not np.isfinite(self.raw_scale).all():
            raise ValueError("Raw scales must be finite and strictly positive")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def _empty(self, sample: dict[str, Any]) -> dict[str, Any]:
        target = np.asarray(sample["target_vector"], dtype=np.float64)
        raw_dim = 1 + self.ptdf_count + 5
        canonical_dim = self.ptdf_count + 5
        analytic = np.zeros(len(self.real_zone_indices), dtype=np.float64)
        return {
            **sample,
            "raw": np.empty((0, raw_dim), dtype=np.float32),
            "raw_unscaled": np.empty((0, raw_dim), dtype=np.float32),
            "raw_scaled": np.empty((0, raw_dim), dtype=np.float32),
            "canonical": np.empty((0, canonical_dim), dtype=np.float32),
            "weights": np.empty(0, dtype=np.float32),
            "total": 0.0,
            "analytic_vector": analytic.astype(np.float32),
            "analytic_scaled": (analytic / self.target_scale).astype(np.float32),
            "target_scaled": (target / self.target_scale).astype(np.float32),
            "residual_scaled": ((target - analytic) / self.target_scale).astype(np.float32),
            "pair_target_scaled": (field_to_pairwise(target) / self.target_scale).astype(np.float32),
            "pair_residual_scaled": (field_to_pairwise(target) / self.target_scale).astype(np.float32),
            "total_scaled": np.float32(0.0),
        }

    def _ambient_frozen_view(self, sample: dict[str, Any], index: int) -> dict[str, Any]:
        row_scaled_columns = np.asarray(
            [*range(1, 1 + self.ptdf_count), 1 + self.ptdf_count], dtype=np.int64
        )
        raw, canonical, weights = _stress_certificate(
            sample["raw"],
            sample["canonical"],
            sample["weights"],
            self.presentation.name,
            20260904 + index,
            row_scaled_columns,
        )
        total = float(sample["total"])
        if len(weights):
            potential = total * (
                weights.astype(np.float64)
                @ canonical[:, self.real_zone_indices].astype(np.float64)
            )
            analytic = center_price_field(-potential)
        else:
            analytic = np.zeros(len(self.real_zone_indices), dtype=np.float64)
        return self._finish(sample, raw, canonical, weights, total, analytic)

    def _present(self, sample: dict[str, Any], index: int) -> dict[str, Any]:
        raw_base = np.asarray(sample["raw"], dtype=np.float64)
        if not len(raw_base):
            return self._empty(sample)
        if self.representation == "ambient" and self.presentation.name in {
            "original",
            "combined",
        }:
            return self._ambient_frozen_view(sample, index)

        shadow = raw_base[:, 0].copy()
        normals = raw_base[:, 1 : 1 + self.ptdf_count].copy()
        rhs = raw_base[:, 1 + self.ptdf_count].copy()
        raw_tail = raw_base[:, 2 + self.ptdf_count :].copy()
        canonical_tail = np.asarray(
            sample["canonical"][:, self.ptdf_count + 1 :], dtype=np.float64
        ).copy()

        if self.presentation.slack_index is not None:
            normals = global_slack_rewrite(normals, self.presentation.slack_index)
        elif self.presentation.projected_representative:
            normals = global_balance_project(normals)

        if self.presentation.combined:
            shadow, normals, rhs, raw_tail, canonical_tail = _apply_combined_row_rewrite(
                shadow,
                normals,
                rhs,
                raw_tail,
                canonical_tail,
                seed=20260904 + index,
            )

        represented_normals = (
            global_balance_project(normals)
            if self.representation == "slack"
            else normals
        )
        norms = np.linalg.norm(represented_normals, axis=1)
        if np.any(norms <= self.norm_tolerance):
            count = int(np.sum(norms <= self.norm_tolerance))
            raise RuntimeError(
                f"{count} rows are constant on the global-balance subspace in "
                f"presentation={self.presentation.name}"
            )
        directions = represented_normals / norms[:, None]
        canonical_ram = np.arcsinh((rhs / norms) / self.canonical_ram_scale)
        canonical = np.column_stack(
            [directions, canonical_ram, canonical_tail]
        ).astype(np.float32)
        mass = shadow * norms
        if np.any(mass <= 0.0) or not np.isfinite(mass).all():
            raise RuntimeError("Canonical dual masses must remain finite and positive")
        total = float(mass.sum())
        weights = (mass / total).astype(np.float32)
        raw = np.column_stack(
            [shadow, represented_normals, rhs, raw_tail]
        ).astype(np.float32)
        potential = mass @ directions[:, self.real_zone_indices]
        analytic = center_price_field(-potential)
        return self._finish(sample, raw, canonical, weights, total, analytic)

    def _finish(
        self,
        sample: dict[str, Any],
        raw: np.ndarray,
        canonical: np.ndarray,
        weights: np.ndarray,
        total: float,
        analytic: np.ndarray,
    ) -> dict[str, Any]:
        target = np.asarray(sample["target_vector"], dtype=np.float64)
        analytic = np.asarray(analytic, dtype=np.float64)
        if raw.shape[1] != len(self.raw_scale):
            raise RuntimeError("Raw certificate width changed unexpectedly")
        if not np.isfinite(canonical).all() or not np.isfinite(analytic).all():
            raise RuntimeError("Non-finite canonical certificate representation")
        return {
            **sample,
            "raw": raw,
            "raw_unscaled": raw,
            "raw_scaled": np.arcsinh(raw / self.raw_scale).astype(np.float32),
            "canonical": canonical,
            "weights": weights,
            "total": total,
            "analytic_vector": analytic.astype(np.float32),
            "analytic_scaled": (analytic / self.target_scale).astype(np.float32),
            "target_scaled": (target / self.target_scale).astype(np.float32),
            "residual_scaled": ((target - analytic) / self.target_scale).astype(np.float32),
            "pair_target_scaled": (field_to_pairwise(target) / self.target_scale).astype(np.float32),
            "pair_residual_scaled": (
                (field_to_pairwise(target) - field_to_pairwise(analytic))
                / self.target_scale
            ).astype(np.float32),
            "total_scaled": np.float32(np.log1p(total)),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._present(self.samples[index], index)


def raw_rows_for_scale(
    samples: list[dict[str, Any]],
    *,
    representation: str,
    ptdf_count: int,
    zone_names: Iterable[str],
    canonical_ram_scale: float,
) -> np.ndarray:
    """Collect original-presentation raw rows for representation-specific scaling."""

    raw_dim = 1 + int(ptdf_count) + 5
    if not samples:
        raise RuntimeError("No samples available to estimate raw feature scales")
    target_width = len(np.asarray(samples[0]["target_vector"]))
    if target_width > ptdf_count:
        raise RuntimeError("Target width exceeds the available PTDF channels")
    probe = SlackGaugeDataset(
        samples,
        raw_scale=np.ones(raw_dim, dtype=np.float64),
        target_scale=1.0,
        presentation="original",
        representation=representation,
        # Only the raw rows are consumed here. Matching the target width keeps
        # the otherwise unused analytic-vector shape internally consistent.
        real_zone_indices=np.arange(target_width, dtype=np.int64),
        ptdf_count=ptdf_count,
        zone_names=zone_names,
        canonical_ram_scale=canonical_ram_scale,
    )
    rows = [probe[index]["raw_unscaled"] for index in range(len(probe))]
    nonempty = [row for row in rows if len(row)]
    if not nonempty:
        raise RuntimeError("No certificate rows available to estimate raw feature scales")
    return np.concatenate(nonempty, axis=0).astype(np.float64)
