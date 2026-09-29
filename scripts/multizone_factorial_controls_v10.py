"""Matched raw-certificate controls for the CQDM multi-zone experiment.

These controls separate three factors that were coupled in the original v6
study: certificate representation, analytic-residual learning, and output
geometry.  The module is deliberately separate from the frozen v6 engine so
that adding controls does not change the hashes of completed evidence.
"""

from __future__ import annotations

import copy
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from scripts.multizone_quotient_gauge_v6 import (
        center_price_field,
        collate_multizone,
        field_to_pairwise,
        pair_indices,
    )
except ModuleNotFoundError:
    from multizone_quotient_gauge_v6 import (
        center_price_field,
        collate_multizone,
        field_to_pairwise,
        pair_indices,
    )


CONTROL_MODES = (
    "raw_deepset_pairwise",
    "raw_deepset_analytic_residual_gauge",
    "raw_deepset_analytic_residual_pairwise",
)


def pair_incidence_matrix(zone_count: int) -> np.ndarray:
    """Return B such that ``field_to_pairwise(p) == p @ B.T``."""

    pairs = pair_indices(zone_count)
    incidence = np.zeros((len(pairs), zone_count), dtype=np.float64)
    for row, (left, right) in enumerate(pairs):
        incidence[row, left] = 1.0
        incidence[row, right] = -1.0
    return incidence


def project_pairwise_to_gauge_field(values: np.ndarray, zone_count: int) -> np.ndarray:
    """Least-squares project arbitrary pair predictions onto a zero-sum field.

    The complete-graph incidence matrix has the constant vector as its only
    null direction.  Its Moore-Penrose solution therefore selects the unique
    minimum-norm, zero-sum price field.
    """

    pair_values = np.asarray(values, dtype=np.float64)
    expected_pairs = len(pair_indices(zone_count))
    if pair_values.ndim < 1 or pair_values.shape[-1] != expected_pairs:
        raise ValueError(
            f"Expected final pair dimension {expected_pairs}, got {pair_values.shape}"
        )
    if not np.isfinite(pair_values).all():
        raise ValueError("Pairwise values must be finite")
    incidence = pair_incidence_matrix(zone_count)
    field = pair_values @ np.linalg.pinv(incidence).T
    return center_price_field(field)


def project_pairwise_to_consistent(values: np.ndarray, zone_count: int) -> dict[str, np.ndarray]:
    """Return the gauge field and cycle-consistent pair vector projection."""

    field = project_pairwise_to_gauge_field(values, zone_count)
    return {"field": field, "pairwise": field_to_pairwise(field)}


def decompose_pairwise_hodge(values: np.ndarray, zone_count: int) -> dict[str, np.ndarray]:
    """Orthogonally split pair values into gradient and cycle-space components."""

    pair_values = np.asarray(values, dtype=np.float64)
    projected = project_pairwise_to_consistent(pair_values, zone_count)
    cycle_component = pair_values - projected["pairwise"]
    return {
        "field": projected["field"],
        "gradient_pairwise": projected["pairwise"],
        "cycle_pairwise": cycle_component,
    }


class RawFactorialControlRegressor(nn.Module):
    """Raw-row controls matched by target and analytic information.

    ``raw_deepset_pairwise`` completes the raw-input/direct-pairwise cell.
    The two analytic-residual modes receive exactly the same analytic KKT field
    exposed to the CQDM residual models.  They therefore test whether gains are
    due to the quotient representation rather than residualization alone.
    """

    def __init__(self, mode: str, raw_dim: int, zone_count: int) -> None:
        super().__init__()
        if mode not in CONTROL_MODES:
            raise ValueError(f"Unknown factorial-control mode: {mode}")
        self.mode = mode
        self.zone_count = int(zone_count)
        self.pair_count = len(pair_indices(zone_count))
        self.pair_output = mode.endswith("pairwise")
        self.residual_output = "analytic_residual" in mode
        self.encoder = nn.Sequential(
            nn.Linear(raw_dim, 48), nn.GELU(), nn.Linear(48, 32), nn.GELU()
        )
        output_dim = self.pair_count if self.pair_output else self.zone_count
        if self.residual_output:
            self.head = nn.Sequential(
                nn.Linear(32 + 2 + self.zone_count, 64),
                nn.GELU(),
                nn.Linear(64, 48),
                nn.GELU(),
                nn.Linear(48, output_dim),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(32, 48), nn.GELU(), nn.Linear(48, output_dim)
            )

    def forward(self, batch: dict[str, Any], device: torch.device) -> torch.Tensor:
        atoms = batch["raw"].to(device)
        mask = batch["raw_mask"].to(device)
        pooled = (self.encoder(atoms) * mask.unsqueeze(-1)).sum(dim=1)
        if self.residual_output:
            pooled = torch.cat(
                [
                    pooled,
                    batch["total"].to(device),
                    batch["present"].to(device),
                    batch["analytic"].to(device),
                ],
                dim=-1,
            )
        output = self.head(pooled)
        if not self.pair_output:
            output = output - output.mean(dim=-1, keepdim=True)
        return output


def _target_key(mode: str) -> str:
    if mode == "raw_deepset_pairwise":
        return "pair_target"
    if mode == "raw_deepset_analytic_residual_pairwise":
        return "pair_residual"
    if mode == "raw_deepset_analytic_residual_gauge":
        return "residual"
    raise ValueError(f"Unknown factorial-control mode: {mode}")


def fit_factorial_control(
    mode: str,
    seed: int,
    train: Dataset,
    validation: Dataset,
    *,
    raw_dim: int,
    zone_count: int,
    batch_size: int,
    epochs: int,
    device: torch.device,
) -> tuple[RawFactorialControlRegressor, dict[str, Any]]:
    """Fit one deterministic control with the frozen v6 optimization recipe."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = RawFactorialControlRegressor(mode, raw_dim, zone_count).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8.0e-4, weight_decay=2.0e-3)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_multizone,
    )
    validation_loader = DataLoader(
        validation,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_multizone,
    )
    target_key = _target_key(mode)
    best = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    stale = 0
    for epoch in range(1, epochs + 1):
        if hasattr(train, "set_epoch"):
            train.set_epoch(epoch)
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch, device)
            loss = F.smooth_l1_loss(prediction, batch[target_key].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        model.eval()
        objective_sum = 0.0
        rows = 0
        with torch.no_grad():
            for batch in validation_loader:
                prediction = model(batch, device)
                count = len(prediction)
                objective_sum += count * float(
                    F.smooth_l1_loss(prediction, batch[target_key].to(device)).cpu()
                )
                rows += count
        objective = objective_sum / rows
        if objective < best - 1.0e-6:
            best = objective
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= 8:
            break
    model.load_state_dict(best_state)
    return model, {
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_objective": best,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_rows_per_epoch": len(train),
        "validation_rows": len(validation),
    }


def predict_factorial_control(
    model: RawFactorialControlRegressor,
    dataset: Dataset,
    *,
    batch_size: int,
    target_scale: float,
    device: torch.device,
) -> dict[str, Any]:
    """Predict fields/pairs with the same reconstruction rules as v6."""

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_multizone
    )
    fields: list[np.ndarray] = []
    pairs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    timestamps: list[Any] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            output = model(batch, device).cpu().numpy() * target_scale
            analytic = batch["analytic_raw"]
            if model.pair_output:
                pair_prediction = output
                if model.residual_output:
                    pair_prediction = field_to_pairwise(analytic) + pair_prediction
                pairs.append(pair_prediction)
            else:
                field_prediction = output
                if model.residual_output:
                    field_prediction = analytic + field_prediction
                field_prediction = center_price_field(field_prediction)
                fields.append(field_prediction)
                pairs.append(field_to_pairwise(field_prediction))
            targets.append(batch["target_raw"])
            timestamps.extend(batch["timestamp"])
    return {
        "field": np.concatenate(fields, axis=0) if fields else None,
        "pairwise": np.concatenate(pairs, axis=0),
        "target": np.concatenate(targets, axis=0),
        "timestamps": timestamps,
    }
