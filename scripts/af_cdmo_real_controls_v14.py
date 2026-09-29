"""Frozen fairness controls for the real AF-CDMO v14 confirmation study.

This module is deliberately separate from the completed v13 engines.  It adds
two controls without changing the v13 implementation or artifacts:

* ``UniformMassDataset`` keeps the AF projection and neural architecture but
  removes the conserved dual-mass base measure from attention.
* ``GaugeAugmentedAmbientDataset`` trains ambient CQDM on randomly sampled
  equality-row-space gauges while preserving the same sample and update count.

Both wrappers are deterministic for a fixed seed, epoch, and sample index.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from torch.utils.data import Dataset

try:
    from scripts.af_cdmo_real_extension_v13 import RealAffineGaugeDataset
    from scripts.multizone_quotient_gauge_v6 import center_price_field, field_to_pairwise
    from scripts.slack_qdm_extension_v11 import global_balance_project
except ModuleNotFoundError:
    from af_cdmo_real_extension_v13 import RealAffineGaugeDataset
    from multizone_quotient_gauge_v6 import center_price_field, field_to_pairwise
    from slack_qdm_extension_v11 import global_balance_project


REGISTERED_AUGMENTATION_AMPLITUDES = (0.25, 0.75)
HELD_OUT_RANDOM_GAUGE_AMPLITUDE = 1.5


class UniformMassDataset(Dataset):
    """Use uniform token mass with an otherwise unchanged AF dataset.

    The wrapper leaves canonical atoms, analytic channel, target, total mass,
    and model architecture unchanged.  Only the attention base measure is
    replaced.  It is therefore parameter matched to AF-QDM but is not exactly
    invariant to mass-conserving split/merge rewrites.
    """

    def __init__(self, base: RealAffineGaugeDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def set_epoch(self, epoch: int) -> None:
        self.base.set_epoch(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.base[index])
        count = len(item["weights"])
        if count:
            item["weights"] = np.full(count, 1.0 / count, dtype=np.float32)
        return item


class NonuniformRefinementAFDataset(Dataset):
    """Mass-conserving AF refinement with nonuniform row multiplicities.

    The frozen combined rewrite splits every row exactly twice, under which an
    unweighted mean is accidentally unchanged.  This diagnostic instead gives
    each row a deterministic multiplicity in ``{1, 2, 3}`` and partitions its
    dual mass across identical canonical atoms.  The represented measure is
    unchanged, but token multiplicity is not.
    """

    def __init__(self, base: RealAffineGaugeDataset, *, seed: int = 20260921) -> None:
        if base.representation != "af":
            raise ValueError("Nonuniform refinement requires the AF representation")
        self.base = base
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.base)

    def set_epoch(self, epoch: int) -> None:
        self.base.set_epoch(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.base[index])
        count = len(item["weights"])
        if not count:
            return item
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, int(index)])
        )
        multiplicities = 1 + rng.integers(0, 3, size=count)
        # Ensure the diagnostic is nontrivial even for an unlikely constant draw.
        if np.all(multiplicities == multiplicities[0]) and count > 1:
            multiplicities[0] = 1
            multiplicities[1] = 3

        refined_raw = []
        refined_canonical = []
        refined_weights = []
        for row, multiplicity in enumerate(multiplicities):
            shares = rng.dirichlet(np.ones(int(multiplicity), dtype=np.float64))
            for share in shares:
                raw = np.asarray(item["raw_unscaled"][row], dtype=np.float64).copy()
                raw[0] *= share
                refined_raw.append(raw)
                refined_canonical.append(item["canonical"][row])
                refined_weights.append(float(item["weights"][row]) * share)

        raw_unscaled = np.asarray(refined_raw, dtype=np.float64)
        item["raw_unscaled"] = raw_unscaled.astype(np.float32)
        item["raw"] = item["raw_unscaled"]
        item["raw_scaled"] = np.arcsinh(
            raw_unscaled / self.base.raw_scale
        ).astype(np.float32)
        item["canonical"] = np.asarray(refined_canonical, dtype=np.float32)
        item["weights"] = np.asarray(refined_weights, dtype=np.float32)
        item["weights"] /= item["weights"].sum()
        return item


def _fixed_gauge_item(
    base: RealAffineGaugeDataset,
    index: int,
    *,
    amplitude: float,
    seed: int,
) -> dict[str, Any]:
    sample = base.samples[index]
    raw_base = np.asarray(sample["raw"], dtype=np.float64)
    if not len(raw_base):
        return base[index]

    ptdf_count = base.ptdf_count
    shadow = raw_base[:, 0].copy()
    normals = raw_base[:, 1 : 1 + ptdf_count].copy()
    rhs = raw_base[:, 1 + ptdf_count].copy()
    raw_tail = raw_base[:, 2 + ptdf_count :].copy()
    canonical_tail = np.asarray(
        sample["canonical"][:, ptdf_count + 1 :], dtype=np.float64
    ).copy()
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(index)]))
    coefficients = rng.normal(
        0.0,
        amplitude,
        size=(len(normals), base.geometry.equality_matrix.shape[0]),
    )
    normals = normals + coefficients @ base.geometry.equality_matrix

    if base.representation == "ambient":
        represented = normals
    elif base.representation == "slack":
        represented = global_balance_project(normals)
    elif base.representation == "af":
        represented = base.geometry.project(normals)
    else:
        raise RuntimeError(f"Unsupported representation: {base.representation}")
    norms = np.linalg.norm(represented, axis=1)
    if np.any(norms <= base.norm_tolerance):
        raise RuntimeError("Fixed equality gauge produced a degenerate row")
    directions = represented / norms[:, None]
    canonical_ram = np.arcsinh((rhs / norms) / base.canonical_ram_scale)
    canonical = np.column_stack([directions, canonical_ram, canonical_tail])
    masses = shadow * norms
    if np.any(masses <= 0.0) or not np.isfinite(masses).all():
        raise RuntimeError("Fixed equality gauge produced invalid dual masses")
    total = float(masses.sum())
    weights = masses / total
    raw = np.column_stack([shadow, represented, rhs, raw_tail])
    potential = masses @ directions[:, base.real_zone_indices]
    analytic = center_price_field(-potential)
    reference = np.asarray(sample["analytic_vector"], dtype=np.float64)
    pair_drift = float(
        np.max(
            np.abs(field_to_pairwise(analytic) - field_to_pairwise(reference)),
            initial=0.0,
        )
    )
    if pair_drift > base.analytic_tolerance:
        raise RuntimeError(
            "Fixed equality gauge changed the conserved analytic transfer "
            f"potential by {pair_drift:.3e}"
        )
    return base._finish(
        sample,
        raw,
        canonical,
        weights,
        total,
        analytic,
        np.asarray(sample["target_vector"], dtype=np.float64),
        analytic_pair_drift=pair_drift,
    )


class FixedEqualityGaugeDataset(Dataset):
    """Apply a fixed held-out equality-gauge amplitude at evaluation time."""

    def __init__(
        self,
        base: RealAffineGaugeDataset,
        *,
        amplitude: float,
        seed: int = 20260922,
    ) -> None:
        if amplitude < 0.0 or not np.isfinite(amplitude):
            raise ValueError("Gauge amplitude must be finite and nonnegative")
        self.base = base
        self.amplitude = float(amplitude)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.base)

    def set_epoch(self, epoch: int) -> None:
        self.base.set_epoch(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return _fixed_gauge_item(
            self.base,
            index,
            amplitude=self.amplitude,
            seed=self.seed,
        )


class NonEquivalentRAMDataset(Dataset):
    """Perturb AF canonical RAM coordinates without claiming equivalence.

    This is a negative control: changing normalized RAM changes the represented
    feasible domain.  Exact invariance is neither expected nor desirable.
    """

    def __init__(self, base: RealAffineGaugeDataset, *, delta: float) -> None:
        if base.representation != "af":
            raise ValueError("RAM sensitivity requires the AF representation")
        if delta == 0.0 or not np.isfinite(delta):
            raise ValueError("RAM perturbation must be finite and nonzero")
        self.base = base
        self.delta = float(delta)

    def __len__(self) -> int:
        return len(self.base)

    def set_epoch(self, epoch: int) -> None:
        self.base.set_epoch(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.base[index])
        if not len(item["canonical"]):
            return item
        canonical = np.asarray(item["canonical"], dtype=np.float32).copy()
        # The canonical RAM coordinate follows the PTDF direction coordinates.
        canonical[:, self.base.ptdf_count] += self.delta
        item["canonical"] = canonical
        return item


class GaugeAugmentedAmbientDataset(Dataset):
    """Ambient CQDM training view with causal equality-gauge augmentation.

    Each epoch exposes exactly one representation per chronological sample, so
    the optimizer update budget is identical to the unaugmented ambient arm.
    Training uses amplitudes 0.25 and 0.75 only.  The registered random-gauge
    evaluation amplitude 1.5 and all deterministic row gauges, reflections,
    slack choices, and split/merge rewrites remain held out.
    """

    def __init__(
        self,
        base: RealAffineGaugeDataset,
        *,
        amplitudes: Sequence[float] = REGISTERED_AUGMENTATION_AMPLITUDES,
        seed: int = 20260921,
    ) -> None:
        if base.representation != "ambient" or base.presentation.name != "original":
            raise ValueError("Gauge augmentation requires the original ambient view")
        amplitudes = tuple(float(value) for value in amplitudes)
        if not amplitudes or any(value <= 0.0 for value in amplitudes):
            raise ValueError("Augmentation amplitudes must be finite and positive")
        if not np.isfinite(amplitudes).all():
            raise ValueError("Augmentation amplitudes must be finite")
        if any(np.isclose(value, HELD_OUT_RANDOM_GAUGE_AMPLITUDE) for value in amplitudes):
            raise ValueError("The held-out random-gauge amplitude cannot be trained on")
        self.base = base
        self.amplitudes = amplitudes
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.base)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.base.set_epoch(epoch)

    def _rng(self, index: int) -> np.random.Generator:
        sequence = np.random.SeedSequence([self.seed, self.epoch, int(index)])
        return np.random.default_rng(sequence)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.base.samples[index]
        raw_base = np.asarray(sample["raw"], dtype=np.float64)
        if not len(raw_base):
            return self.base[index]

        ptdf_count = self.base.ptdf_count
        shadow = raw_base[:, 0].copy()
        normals = raw_base[:, 1 : 1 + ptdf_count].copy()
        rhs = raw_base[:, 1 + ptdf_count].copy()
        raw_tail = raw_base[:, 2 + ptdf_count :].copy()
        canonical_tail = np.asarray(
            sample["canonical"][:, ptdf_count + 1 :], dtype=np.float64
        ).copy()

        amplitude = self.amplitudes[(self.epoch + index) % len(self.amplitudes)]
        coefficients = self._rng(index).normal(
            0.0,
            amplitude,
            size=(len(normals), self.base.geometry.equality_matrix.shape[0]),
        )
        normals = normals + coefficients @ self.base.geometry.equality_matrix

        norms = np.linalg.norm(normals, axis=1)
        if np.any(norms <= self.base.norm_tolerance):
            raise RuntimeError("Gauge augmentation produced a degenerate ambient row")
        directions = normals / norms[:, None]
        canonical_ram = np.arcsinh(
            (rhs / norms) / self.base.canonical_ram_scale
        )
        canonical = np.column_stack([directions, canonical_ram, canonical_tail])
        masses = shadow * norms
        if np.any(masses <= 0.0) or not np.isfinite(masses).all():
            raise RuntimeError("Gauge augmentation produced invalid dual masses")
        total = float(masses.sum())
        weights = masses / total
        raw = np.column_stack([shadow, normals, rhs, raw_tail])

        potential = masses @ directions[:, self.base.real_zone_indices]
        analytic = center_price_field(-potential)
        reference = np.asarray(sample["analytic_vector"], dtype=np.float64)
        pair_drift = float(
            np.max(
                np.abs(field_to_pairwise(analytic) - field_to_pairwise(reference)),
                initial=0.0,
            )
        )
        if pair_drift > self.base.analytic_tolerance:
            raise RuntimeError(
                "Gauge augmentation changed the conserved analytic transfer "
                f"potential by {pair_drift:.3e}"
            )

        return self.base._finish(
            sample,
            raw,
            canonical,
            weights,
            total,
            analytic,
            np.asarray(sample["target_vector"], dtype=np.float64),
            analytic_pair_drift=pair_drift,
        )


def control_contract() -> dict[str, Any]:
    return {
        "uniform_mass": {
            "changed_component": "attention base measure only",
            "unchanged_components": [
                "AF projection",
                "canonical atoms",
                "analytic channel",
                "neural architecture",
                "optimizer",
                "sample count",
            ],
            "held_out_nonuniform_refinement": True,
        },
        "ambient_gauge_augmentation": {
            "amplitudes": list(REGISTERED_AUGMENTATION_AMPLITUDES),
            "held_out_random_gauge_amplitude": HELD_OUT_RANDOM_GAUGE_AMPLITUDE,
            "same_samples_per_epoch": True,
            "held_out_families": [
                "deterministic equality-row gauges",
                "equality reflection",
                "slack changes",
                "positive rescaling",
                "mass-conserving split/merge",
                "equality-basis rewrites",
            ],
        },
    }
