"""Shared algebra and models for the CQDM multi-zone quotient-gauge study."""

from __future__ import annotations

import copy
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


REAL_NORDIC_ZONES = (
    "DK1",
    "DK2",
    "FI",
    "NO1",
    "NO2",
    "NO3",
    "NO4",
    "NO5",
    "SE1",
    "SE2",
    "SE3",
    "SE4",
)


def center_price_field(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 1:
        raise ValueError("Price field must have at least one dimension")
    return values - values.mean(axis=-1, keepdims=True)


def pair_indices(zone_count: int) -> tuple[tuple[int, int], ...]:
    if zone_count < 2:
        raise ValueError("At least two zones are required")
    return tuple((left, right) for left in range(zone_count) for right in range(left + 1, zone_count))


def field_to_pairwise(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    pairs = pair_indices(values.shape[-1])
    return np.stack([values[..., left] - values[..., right] for left, right in pairs], axis=-1)


def pairwise_to_matrix(values: np.ndarray, zone_count: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    pairs = pair_indices(zone_count)
    if values.shape[-1] != len(pairs):
        raise ValueError("Pairwise vector does not match zone count")
    matrix = np.zeros((*values.shape[:-1], zone_count, zone_count), dtype=np.float64)
    for pair_index, (left, right) in enumerate(pairs):
        matrix[..., left, right] = values[..., pair_index]
        matrix[..., right, left] = -values[..., pair_index]
    return matrix


def cycle_inconsistency(values: np.ndarray, zone_count: int) -> np.ndarray:
    """Absolute cycle residuals for every unordered zone triplet."""

    matrix = pairwise_to_matrix(values, zone_count)
    cycles = []
    for first in range(zone_count):
        for second in range(first + 1, zone_count):
            for third in range(second + 1, zone_count):
                cycles.append(
                    np.abs(
                        matrix[..., first, second]
                        + matrix[..., second, third]
                        + matrix[..., third, first]
                    )
                )
    return np.stack(cycles, axis=-1)


def quotient_feature(
    canonical: np.ndarray,
    weights: np.ndarray,
    total: float,
    real_zone_indices: np.ndarray,
) -> np.ndarray:
    """Fixed invariant moments for the multi-zone classical control."""

    canonical = np.asarray(canonical, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if not len(weights) or total <= 0.0:
        dimension = canonical.shape[1]
        return np.zeros(2 + len(real_zone_indices) + 2 * dimension, dtype=np.float64)
    normalized = weights / weights.sum()
    mean = normalized @ canonical
    second = normalized @ np.square(canonical)
    dual_potential = total * mean[real_zone_indices]
    analytic = center_price_field(-dual_potential)
    return np.concatenate(
        [np.asarray([np.log1p(total), 1.0]), analytic, mean, second]
    )


def collate_multizone(batch: list[dict[str, Any]]) -> dict[str, Any]:
    raw_dim = batch[0]["raw_scaled"].shape[1]
    canonical_dim = batch[0]["canonical"].shape[1]
    max_rows = max(max(len(item["raw_scaled"]), len(item["weights"]), 1) for item in batch)
    raw = np.zeros((len(batch), max_rows, raw_dim), dtype=np.float32)
    raw_mask = np.zeros((len(batch), max_rows), dtype=bool)
    canonical = np.zeros((len(batch), max_rows, canonical_dim), dtype=np.float32)
    weights = np.zeros((len(batch), max_rows), dtype=np.float32)
    mask = np.zeros((len(batch), max_rows), dtype=bool)
    for index, item in enumerate(batch):
        raw_count = len(item["raw_scaled"])
        canonical_count = len(item["weights"])
        if raw_count:
            raw[index, :raw_count] = item["raw_scaled"]
            raw_mask[index, :raw_count] = True
        if canonical_count:
            canonical[index, :canonical_count] = item["canonical"]
            weights[index, :canonical_count] = item["weights"]
            mask[index, :canonical_count] = True
    return {
        "raw": torch.from_numpy(raw),
        "raw_mask": torch.from_numpy(raw_mask),
        "canonical": torch.from_numpy(canonical),
        "weights": torch.from_numpy(weights),
        "mask": torch.from_numpy(mask),
        "total": torch.tensor([[item["total_scaled"]] for item in batch]),
        "present": torch.tensor([[float(len(item["weights"]) > 0)] for item in batch]),
        "analytic": torch.tensor(np.stack([item["analytic_scaled"] for item in batch])),
        "target": torch.tensor(np.stack([item["target_scaled"] for item in batch])),
        "residual": torch.tensor(np.stack([item["residual_scaled"] for item in batch])),
        "pair_target": torch.tensor(np.stack([item["pair_target_scaled"] for item in batch])),
        "pair_residual": torch.tensor(np.stack([item["pair_residual_scaled"] for item in batch])),
        "target_raw": np.stack([item["target_vector"] for item in batch]),
        "analytic_raw": np.stack([item["analytic_vector"] for item in batch]),
        "timestamp": [item["timestamp"] for item in batch],
    }


class MultiZoneRegressor(nn.Module):
    """Raw, quotient-gauge, or independent-pair certificate regressor."""

    def __init__(
        self,
        mode: str,
        raw_dim: int,
        canonical_dim: int,
        zone_count: int,
    ) -> None:
        super().__init__()
        allowed = {
            "raw_deepset_gauge",
            "raw_deepset_augmented_gauge",
            "cqdm_gauge_residual",
            "cqdm_pairwise_residual",
        }
        if mode not in allowed:
            raise ValueError(f"Unknown multi-zone mode: {mode}")
        self.mode = mode
        self.zone_count = zone_count
        self.pair_count = len(pair_indices(zone_count))
        self.raw_input = mode.startswith("raw_deepset")
        self.pair_output = mode == "cqdm_pairwise_residual"
        atom_dim = raw_dim if self.raw_input else canonical_dim
        self.encoder = nn.Sequential(
            nn.Linear(atom_dim, 48), nn.GELU(), nn.Linear(48, 32), nn.GELU()
        )
        if self.raw_input:
            self.head = nn.Sequential(
                nn.Linear(32, 48), nn.GELU(), nn.Linear(48, zone_count)
            )
        else:
            self.query_count = 4
            self.attention_dim = 32
            self.queries = nn.Parameter(0.02 * torch.randn(self.query_count, self.attention_dim))
            self.keys = nn.Linear(32, self.attention_dim)
            self.values = nn.Linear(32, self.attention_dim)
            output_dim = self.pair_count if self.pair_output else zone_count
            self.head = nn.Sequential(
                nn.Linear(self.query_count * self.attention_dim + 2 + zone_count, 64),
                nn.GELU(),
                nn.Linear(64, 48),
                nn.GELU(),
                nn.Linear(48, output_dim),
            )

    def forward(self, batch: dict[str, Any], device: torch.device) -> torch.Tensor:
        if self.raw_input:
            atoms = batch["raw"].to(device)
            mask = batch["raw_mask"].to(device)
            hidden = (self.encoder(atoms) * mask.unsqueeze(-1)).sum(dim=1)
            field = self.head(hidden)
            return field - field.mean(dim=-1, keepdim=True)

        atoms = batch["canonical"].to(device)
        masses = batch["weights"].to(device)
        mask = batch["mask"].to(device)
        embedding = self.encoder(atoms)
        keys = self.keys(embedding)
        values = self.values(embedding)
        scores = torch.einsum("qd,bnd->bqn", self.queries, keys) / np.sqrt(self.attention_dim)
        masked_scores = scores.masked_fill(~mask.unsqueeze(1), -1.0e9)
        shifted = masked_scores - masked_scores.amax(dim=-1, keepdim=True)
        numerator = torch.exp(shifted) * masses.unsqueeze(1) * mask.unsqueeze(1)
        attention = numerator / numerator.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        pooled = torch.einsum("bqn,bnd->bqd", attention, values).flatten(1)
        hidden = torch.cat(
            [
                pooled,
                batch["total"].to(device),
                batch["present"].to(device),
                batch["analytic"].to(device),
            ],
            dim=-1,
        )
        output = self.head(hidden)
        if not self.pair_output:
            output = output - output.mean(dim=-1, keepdim=True)
        return output


def fit_multizone(
    mode: str,
    seed: int,
    train: Dataset,
    validation: Dataset,
    *,
    raw_dim: int,
    canonical_dim: int,
    zone_count: int,
    batch_size: int,
    epochs: int,
    device: torch.device,
) -> tuple[MultiZoneRegressor, dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = MultiZoneRegressor(mode, raw_dim, canonical_dim, zone_count).to(device)
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
    target_key = (
        "pair_residual"
        if mode == "cqdm_pairwise_residual"
        else "residual"
        if mode == "cqdm_gauge_residual"
        else "target"
    )
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


def predict_multizone(
    model: MultiZoneRegressor,
    dataset: Dataset,
    *,
    batch_size: int,
    target_scale: float,
    device: torch.device,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_multizone)
    fields: list[np.ndarray] = []
    pairs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    timestamps: list[Any] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            output = model(batch, device).cpu().numpy() * target_scale
            analytic = batch["analytic_raw"]
            if model.mode == "cqdm_pairwise_residual":
                pair_prediction = field_to_pairwise(analytic) + output
                pairs.append(pair_prediction)
            else:
                field_prediction = (
                    analytic + output if model.mode == "cqdm_gauge_residual" else output
                )
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
