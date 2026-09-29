"""Loss-safe real-certificate representations for AF-CDMO v13.

The module is intentionally data-only.  It exposes matched Slack-QDM and
rank-five AF-QDM views of the same v6/v10 certificate samples without fitting a
model or touching frozen artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from torch.utils.data import Dataset

try:
    from scripts.audit_af_cdmo_real_geometry_v13 import build_loss_safe_geometry
    from scripts.slack_qdm_extension_v11 import (
        _apply_combined_row_rewrite,
        global_balance_project,
        global_slack_rewrite,
        presentation_names as slack_presentation_names,
    )
    from scripts.multizone_quotient_gauge_v6 import (
        center_price_field,
        field_to_pairwise,
    )
except ModuleNotFoundError:
    from audit_af_cdmo_real_geometry_v13 import build_loss_safe_geometry
    from slack_qdm_extension_v11 import (
        _apply_combined_row_rewrite,
        global_balance_project,
        global_slack_rewrite,
        presentation_names as slack_presentation_names,
    )
    from multizone_quotient_gauge_v6 import center_price_field, field_to_pairwise


REPRESENTATIONS = ("ambient", "slack", "af")
AF_EXTRA_PRESENTATIONS = (
    "af_projected_representative",
    "combined_af_projected_representative",
    "equality_reflection",
    "combined_equality_reflection",
    "random_equality_gauge",
    "combined_random_equality_gauge",
)


@dataclass(frozen=True)
class RealAffineGeometry:
    zone_names: tuple[str, ...]
    equality_names: tuple[str, ...]
    equality_matrix: np.ndarray
    tangent_projector: np.ndarray
    rank: int

    @classmethod
    def loss_safe(cls, zone_names: Iterable[str]) -> "RealAffineGeometry":
        zones = tuple(zone_names)
        equality, equality_names = build_loss_safe_geometry(list(zones))
        rowspace = equality.T @ np.linalg.pinv(equality @ equality.T) @ equality
        tangent = np.eye(len(zones), dtype=np.float64) - rowspace
        tangent = 0.5 * (tangent + tangent.T)
        return cls(
            zone_names=zones,
            equality_names=tuple(equality_names),
            equality_matrix=equality,
            tangent_projector=tangent,
            rank=int(np.linalg.matrix_rank(equality)),
        )

    def project(self, normals: np.ndarray) -> np.ndarray:
        values = np.asarray(normals, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.zone_names):
            raise ValueError("PTDF normals have the wrong shape for this geometry")
        return values @ self.tangent_projector


@dataclass(frozen=True)
class Presentation:
    name: str
    combined: bool = False
    global_slack_index: int | None = None
    kind: str = "identity"
    equality_row_index: int | None = None


def presentation_names(geometry: RealAffineGeometry) -> tuple[str, ...]:
    shared = slack_presentation_names(geometry.zone_names)
    row_stresses = tuple(
        name
        for row_name in geometry.equality_names[1:]
        for name in (f"gauge_{row_name}", f"combined_gauge_{row_name}")
    )
    names = (*shared, *AF_EXTRA_PRESENTATIONS, *row_stresses)
    if len(names) != len(set(names)):
        raise RuntimeError("AF-CDMO v13 presentation names are not unique")
    return names


def parse_presentation(name: str, geometry: RealAffineGeometry) -> Presentation:
    combined = name.startswith("combined_") or name == "combined"
    base = name.removeprefix("combined_") if combined else name
    if base in {"original", "combined"}:
        return Presentation(name=name, combined=combined)
    if base == "projected_representative":
        return Presentation(name=name, combined=combined, kind="global_projected")
    if base.startswith("slack_"):
        zone = base.removeprefix("slack_")
        if zone not in geometry.zone_names:
            raise ValueError(f"Unknown global slack zone in {name}: {zone}")
        return Presentation(
            name=name,
            combined=combined,
            global_slack_index=geometry.zone_names.index(zone),
            kind="global_slack",
        )
    if base == "af_projected_representative":
        return Presentation(name=name, combined=combined, kind="af_projected")
    if base == "equality_reflection":
        return Presentation(name=name, combined=combined, kind="equality_reflection")
    if base == "random_equality_gauge":
        return Presentation(name=name, combined=combined, kind="random_equality_gauge")
    if base.startswith("gauge_"):
        row_name = base.removeprefix("gauge_")
        if row_name not in geometry.equality_names:
            raise ValueError(f"Unknown equality row in {name}: {row_name}")
        return Presentation(
            name=name,
            combined=combined,
            kind="row_gauge",
            equality_row_index=geometry.equality_names.index(row_name),
        )
    raise ValueError(f"Unknown AF-CDMO v13 presentation: {name}")


def _apply_presentation(
    presentation: Presentation,
    geometry: RealAffineGeometry,
    shadow: np.ndarray,
    normals: np.ndarray,
    rhs: np.ndarray,
    raw_tail: np.ndarray,
    canonical_tail: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if presentation.kind == "global_slack":
        if presentation.global_slack_index is None:
            raise RuntimeError("Global-slack presentation omitted its index")
        normals = global_slack_rewrite(normals, presentation.global_slack_index)
    elif presentation.kind == "global_projected":
        normals = global_balance_project(normals)
    elif presentation.kind == "af_projected":
        normals = geometry.project(normals)
    elif presentation.kind == "equality_reflection":
        projected = geometry.project(normals)
        normals = 2.0 * projected - normals
    elif presentation.kind == "random_equality_gauge":
        rng = np.random.default_rng(seed + 1_000_003)
        coefficients = rng.normal(
            0.0, 1.5, size=(len(normals), geometry.equality_matrix.shape[0])
        )
        normals = normals + coefficients @ geometry.equality_matrix
    elif presentation.kind == "row_gauge":
        if presentation.equality_row_index is None:
            raise RuntimeError("Row-gauge presentation omitted its equality row")
        equality_row = geometry.equality_matrix[presentation.equality_row_index]
        nonzero = np.flatnonzero(np.abs(equality_row) > 0.0)
        if not len(nonzero):
            raise RuntimeError("Cannot gauge-fix against an empty equality row")
        pivot = int(nonzero[0])
        coefficient = -normals[:, pivot] / equality_row[pivot]
        normals = normals + coefficient[:, None] * equality_row[None, :]
    elif presentation.kind != "identity":
        raise RuntimeError(f"Unhandled presentation kind: {presentation.kind}")

    if presentation.combined:
        shadow, normals, rhs, raw_tail, canonical_tail = _apply_combined_row_rewrite(
            shadow,
            normals,
            rhs,
            raw_tail,
            canonical_tail,
            seed=seed,
        )
    return shadow, normals, rhs, raw_tail, canonical_tail


class RealAffineGaugeDataset(Dataset):
    """Matched Slack-QDM or AF-QDM view of real certificate samples."""

    def __init__(
        self,
        samples: list[dict[str, Any]],
        raw_scale: np.ndarray,
        target_scale: float,
        presentation: str,
        representation: str,
        real_zone_indices: np.ndarray,
        geometry: RealAffineGeometry,
        canonical_ram_scale: float,
        *,
        norm_tolerance: float = 1.0e-12,
        analytic_tolerance: float = 1.0e-3,
    ) -> None:
        if representation not in REPRESENTATIONS:
            raise ValueError(f"Unknown representation: {representation}")
        if target_scale <= 0.0 or canonical_ram_scale <= 0.0:
            raise ValueError("Target and canonical-RAM scales must be positive")
        if norm_tolerance <= 0.0 or analytic_tolerance <= 0.0:
            raise ValueError("Numerical tolerances must be positive")
        self.samples = samples
        self.raw_scale = np.asarray(raw_scale, dtype=np.float64)
        self.target_scale = float(target_scale)
        self.presentation = parse_presentation(presentation, geometry)
        self.representation = representation
        self.real_zone_indices = np.asarray(real_zone_indices, dtype=np.int64)
        self.geometry = geometry
        self.ptdf_count = len(geometry.zone_names)
        self.canonical_ram_scale = float(canonical_ram_scale)
        self.norm_tolerance = float(norm_tolerance)
        self.analytic_tolerance = float(analytic_tolerance)
        self.epoch = 0

        expected_raw_dim = 1 + self.ptdf_count + 5
        if self.raw_scale.shape != (expected_raw_dim,):
            raise ValueError(
                f"raw_scale has shape {self.raw_scale.shape}; expected {(expected_raw_dim,)}"
            )
        if np.any(self.raw_scale <= 0.0) or not np.isfinite(self.raw_scale).all():
            raise ValueError("Raw scales must be finite and strictly positive")
        if np.any(self.real_zone_indices < 0) or np.any(
            self.real_zone_indices >= self.ptdf_count
        ):
            raise ValueError("Observed-zone indices fall outside the PTDF coordinates")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def _empty(self, sample: dict[str, Any]) -> dict[str, Any]:
        target = np.asarray(sample["target_vector"], dtype=np.float64)
        raw_dim = 1 + self.ptdf_count + 5
        canonical_dim = self.ptdf_count + 5
        analytic = np.zeros(len(self.real_zone_indices), dtype=np.float64)
        return self._finish(
            sample,
            np.empty((0, raw_dim), dtype=np.float64),
            np.empty((0, canonical_dim), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            0.0,
            analytic,
            target,
        )

    def _finish(
        self,
        sample: dict[str, Any],
        raw: np.ndarray,
        canonical: np.ndarray,
        weights: np.ndarray,
        total: float,
        analytic: np.ndarray,
        target: np.ndarray,
        analytic_pair_drift: float = 0.0,
    ) -> dict[str, Any]:
        if raw.shape[1] != len(self.raw_scale):
            raise RuntimeError("Raw certificate width changed unexpectedly")
        if not np.isfinite(canonical).all() or not np.isfinite(analytic).all():
            raise RuntimeError("Non-finite AF certificate representation")
        return {
            **sample,
            "raw": raw.astype(np.float32),
            "raw_unscaled": raw.astype(np.float32),
            "raw_scaled": np.arcsinh(raw / self.raw_scale).astype(np.float32),
            "canonical": canonical.astype(np.float32),
            "weights": weights.astype(np.float32),
            "total": float(total),
            "analytic_vector": analytic.astype(np.float32),
            "analytic_scaled": (analytic / self.target_scale).astype(np.float32),
            "target_scaled": (target / self.target_scale).astype(np.float32),
            "residual_scaled": ((target - analytic) / self.target_scale).astype(np.float32),
            "pair_target_scaled": (
                field_to_pairwise(target) / self.target_scale
            ).astype(np.float32),
            "pair_residual_scaled": (
                (field_to_pairwise(target) - field_to_pairwise(analytic))
                / self.target_scale
            ).astype(np.float32),
            "total_scaled": np.float32(np.log1p(total)),
            "analytic_pair_drift_eur_mwh": float(analytic_pair_drift),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        raw_base = np.asarray(sample["raw"], dtype=np.float64)
        target = np.asarray(sample["target_vector"], dtype=np.float64)
        if not len(raw_base):
            return self._empty(sample)

        # Preserve the frozen v11 ambient input exactly on the clean
        # presentation.  Reconstructed views are used only when a registered
        # null rewrite actually changes the certificate syntax.
        if self.representation == "ambient" and self.presentation.name == "original":
            return self._finish(
                sample,
                raw_base,
                np.asarray(sample["canonical"], dtype=np.float64),
                np.asarray(sample["weights"], dtype=np.float64),
                float(sample["total"]),
                np.asarray(sample["analytic_vector"], dtype=np.float64),
                target,
            )

        shadow = raw_base[:, 0].copy()
        normals = raw_base[:, 1 : 1 + self.ptdf_count].copy()
        rhs = raw_base[:, 1 + self.ptdf_count].copy()
        raw_tail = raw_base[:, 2 + self.ptdf_count :].copy()
        canonical_tail = np.asarray(
            sample["canonical"][:, self.ptdf_count + 1 :], dtype=np.float64
        ).copy()
        shadow, normals, rhs, raw_tail, canonical_tail = _apply_presentation(
            self.presentation,
            self.geometry,
            shadow,
            normals,
            rhs,
            raw_tail,
            canonical_tail,
            seed=20260920 + index,
        )

        if self.representation == "ambient":
            represented = normals
        elif self.representation == "slack":
            represented = global_balance_project(normals)
        else:
            represented = self.geometry.project(normals)
        norms = np.linalg.norm(represented, axis=1)
        if np.any(norms <= self.norm_tolerance):
            count = int(np.sum(norms <= self.norm_tolerance))
            raise RuntimeError(
                f"{count} positive-dual rows are constant after "
                f"representation={self.representation}, presentation={self.presentation.name}"
            )
        directions = represented / norms[:, None]
        canonical_ram = np.arcsinh((rhs / norms) / self.canonical_ram_scale)
        canonical = np.column_stack([directions, canonical_ram, canonical_tail])
        masses = shadow * norms
        if np.any(masses <= 0.0) or not np.isfinite(masses).all():
            raise RuntimeError("Canonical dual masses must remain finite and positive")
        total = float(masses.sum())
        weights = masses / total
        raw = np.column_stack([shadow, represented, rhs, raw_tail])
        potential = masses @ directions[:, self.real_zone_indices]
        analytic = center_price_field(-potential)

        ambient_analytic = np.asarray(sample["analytic_vector"], dtype=np.float64)
        pair_drift = float(
            np.max(
                np.abs(
                    field_to_pairwise(analytic)
                    - field_to_pairwise(ambient_analytic)
                ),
                initial=0.0,
            )
        )
        if pair_drift > self.analytic_tolerance:
            raise RuntimeError(
                "Projected analytic transfer potential changed by "
                f"{pair_drift:.3e}; observed queries are not tangent"
            )
        return self._finish(
            sample,
            raw,
            canonical,
            weights,
            total,
            analytic,
            target,
            analytic_pair_drift=pair_drift,
        )


def raw_rows_for_scale(
    samples: list[dict[str, Any]],
    *,
    representation: str,
    real_zone_indices: np.ndarray,
    geometry: RealAffineGeometry,
    canonical_ram_scale: float,
) -> np.ndarray:
    if not samples:
        raise RuntimeError("No samples available to estimate raw feature scales")
    raw_dim = 1 + len(geometry.zone_names) + 5
    probe = RealAffineGaugeDataset(
        samples,
        raw_scale=np.ones(raw_dim, dtype=np.float64),
        target_scale=1.0,
        presentation="original",
        representation=representation,
        real_zone_indices=real_zone_indices,
        geometry=geometry,
        canonical_ram_scale=canonical_ram_scale,
    )
    rows = [probe[index]["raw_unscaled"] for index in range(len(probe))]
    nonempty = [row for row in rows if len(row)]
    if not nonempty:
        raise RuntimeError("No certificate rows available to estimate raw feature scales")
    return np.concatenate(nonempty, axis=0).astype(np.float64)
