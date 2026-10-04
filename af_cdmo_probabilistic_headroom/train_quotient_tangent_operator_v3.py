"""Train an invariant quotient-tangent neural operator for attribution risk.

This is an isolated, development-only experiment.  It never reads timestamps
on or after 2026-03-01 UTC and does not modify the frozen AF-CDMO study.

The model integrates over the exact affine-feasible certificate quotient with
mass-conserving atom-to-atom attention.  It jointly learns the centered
price-gauge tangent residual and nested residual-tail events.  The richer
auxiliary target is intended to avoid fitting a rare binary event from only
twenty thousand development rows.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]

try:
    from af_cdmo_learning_headroom.audit_residual_headroom import (
        CONFIRMATION_START,
        DEVELOPMENT_END,
        FIT_END,
        PRIMARY_ZONES,
        _build_vector_samples,
        _load_price_panel,
        _read_certificate,
        DEFAULT_DUAL,
        DEFAULT_PRICE_ROOT,
    )
    from scripts.af_cdmo_real_extension_v13 import (
        RealAffineGaugeDataset,
        RealAffineGeometry,
        presentation_names,
        raw_rows_for_scale,
    )
    from scripts.benchmark_af_cdmo_real_v14 import (
        DEFAULT_GEOMETRY_AUDIT,
        _canonical_ram_scale,
    )
    from scripts.benchmark_certificate_governance_v5 import _positive_scales
    from scripts.multizone_quotient_gauge_v6 import field_to_pairwise, quotient_feature
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(ROOT))
    from af_cdmo_learning_headroom.audit_residual_headroom import (
        CONFIRMATION_START,
        DEVELOPMENT_END,
        FIT_END,
        PRIMARY_ZONES,
        _build_vector_samples,
        _load_price_panel,
        _read_certificate,
        DEFAULT_DUAL,
        DEFAULT_PRICE_ROOT,
    )
    from scripts.af_cdmo_real_extension_v13 import (
        RealAffineGaugeDataset,
        RealAffineGeometry,
        presentation_names,
        raw_rows_for_scale,
    )
    from scripts.benchmark_af_cdmo_real_v14 import (
        DEFAULT_GEOMETRY_AUDIT,
        _canonical_ram_scale,
    )
    from scripts.benchmark_certificate_governance_v5 import _positive_scales
    from scripts.multizone_quotient_gauge_v6 import field_to_pairwise, quotient_feature


TUNING_END = pd.Timestamp("2025-12-01", tz="UTC")
CALIBRATION_END = pd.Timestamp("2026-01-01", tz="UTC")
TAIL_QUANTILE = 0.90
EPSILON = 1.0e-7
BLOCK_ROWS = 672
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261001


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dual", type=Path, default=DEFAULT_DUAL)
    parser.add_argument("--price_root", type=Path, default=DEFAULT_PRICE_ROOT)
    parser.add_argument("--geometry_audit", type=Path, default=DEFAULT_GEOMETRY_AUDIT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "af_cdmo_probabilistic_headroom"
        / "quotient_tangent_operator_development_v3.json",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123, 2025, 3007])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--variant",
        choices=("full", "no_interactions", "no_tangent"),
        default="full",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _timestamp_mask(
    timestamps: pd.DatetimeIndex,
    start: pd.Timestamp | None,
    end: pd.Timestamp,
) -> np.ndarray:
    mask = timestamps < end
    if start is not None:
        mask &= timestamps >= start
    return np.asarray(mask)


def _residual_magnitude(item: dict[str, Any], target_scale: float) -> float:
    pair_residual = np.asarray(item["pair_residual_scaled"], dtype=np.float64)
    return float(np.mean(np.abs(pair_residual)) * target_scale)


def _calendar(timestamp: pd.Timestamp) -> np.ndarray:
    quarter = timestamp.hour * 4 + timestamp.minute // 15
    weekday = timestamp.dayofweek
    day = timestamp.dayofyear
    return np.asarray(
        [
            math.sin(2.0 * math.pi * quarter / 96.0),
            math.cos(2.0 * math.pi * quarter / 96.0),
            math.sin(2.0 * math.pi * weekday / 7.0),
            math.cos(2.0 * math.pi * weekday / 7.0),
            math.sin(2.0 * math.pi * day / 365.25),
            math.cos(2.0 * math.pi * day / 365.25),
        ],
        dtype=np.float32,
    )


class TailDataset(Dataset):
    def __init__(self, base: RealAffineGaugeDataset, threshold: float) -> None:
        self.base = base
        self.threshold = float(threshold)
        # Canonicalization is deterministic and immutable for this experiment.
        # Cache each row after first access so training epochs do not repeatedly
        # execute the affine projection and certificate reconstruction.
        self._cache: list[dict[str, Any] | None] = [None] * len(base)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        cached = self._cache[index]
        if cached is not None:
            return cached
        item = self.base[index]
        magnitude = _residual_magnitude(item, self.base.target_scale)
        result = {
            **item,
            "tail_target": np.float32(magnitude > self.threshold),
            "residual_magnitude": np.float32(magnitude),
            "calendar": _calendar(pd.Timestamp(item["timestamp"])),
        }
        self._cache[index] = result
        return result


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    canonical_dim = batch[0]["canonical"].shape[1]
    max_rows = max(max(len(item["weights"]), 1) for item in batch)
    canonical = np.zeros((len(batch), max_rows, canonical_dim), dtype=np.float32)
    weights = np.zeros((len(batch), max_rows), dtype=np.float32)
    mask = np.zeros((len(batch), max_rows), dtype=bool)
    for index, item in enumerate(batch):
        count = len(item["weights"])
        if count:
            canonical[index, :count] = item["canonical"]
            weights[index, :count] = item["weights"]
            mask[index, :count] = True
    return {
        "canonical": torch.from_numpy(canonical),
        "weights": torch.from_numpy(weights),
        "mask": torch.from_numpy(mask),
        "total": torch.tensor([[item["total_scaled"]] for item in batch]),
        "present": torch.tensor([[float(len(item["weights"]) > 0)] for item in batch]),
        "analytic": torch.tensor(np.stack([item["analytic_scaled"] for item in batch])),
        "residual": torch.tensor(np.stack([item["residual_scaled"] for item in batch])),
        "pair_residual": torch.tensor(
            np.stack([item["pair_residual_scaled"] for item in batch])
        ),
        "calendar": torch.tensor(np.stack([item["calendar"] for item in batch])),
        "target": torch.tensor([item["tail_target"] for item in batch]),
        "magnitude": np.asarray([item["residual_magnitude"] for item in batch]),
        "timestamp": [item["timestamp"] for item in batch],
    }


class MassConservingAttentionBlock(nn.Module):
    """Atom interaction that is invariant to permutation and mass splitting."""

    def __init__(self, hidden: int, heads: int = 4) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden width must be divisible by attention heads")
        self.heads = heads
        self.width = hidden // heads
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.output = nn.Linear(hidden, hidden)
        self.norm_attention = nn.LayerNorm(hidden)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden, 2 * hidden),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(2 * hidden, hidden),
        )
        self.norm_feed_forward = nn.LayerNorm(hidden)

    def forward(
        self,
        hidden: torch.Tensor,
        weights: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, atoms, width = hidden.shape
        query = self.query(hidden).reshape(batch, atoms, self.heads, self.width)
        key = self.key(hidden).reshape(batch, atoms, self.heads, self.width)
        value = self.value(hidden).reshape(batch, atoms, self.heads, self.width)
        scores = torch.einsum("bihd,bjhd->bhij", query, key) / math.sqrt(self.width)
        scores = scores.masked_fill(~mask[:, None, None, :], -1.0e9)
        shifted = scores - scores.amax(dim=-1, keepdim=True)
        numerator = (
            torch.exp(shifted)
            * weights[:, None, None, :]
            * mask[:, None, None, :]
        )
        attention = numerator / numerator.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        message = torch.einsum("bhij,bjhd->bihd", attention, value).reshape(
            batch, atoms, width
        )
        valid = mask.unsqueeze(-1)
        hidden = self.norm_attention(hidden + self.output(message)) * valid
        hidden = self.norm_feed_forward(hidden + self.feed_forward(hidden)) * valid
        return hidden


class QuotientTangentOperator(nn.Module):
    """Exact-quotient measure operator with gauge-tangent auxiliary output."""

    def __init__(
        self,
        canonical_dim: int,
        zone_count: int,
        *,
        variant: str,
        atom_mean: np.ndarray,
        atom_std: np.ndarray,
    ) -> None:
        super().__init__()
        hidden = 64
        query_count = 6
        self.variant = variant
        self.zone_count = zone_count
        self.register_buffer("atom_mean", torch.tensor(atom_mean, dtype=torch.float32))
        self.register_buffer("atom_std", torch.tensor(atom_std, dtype=torch.float32))
        self.encoder = nn.Sequential(
            nn.Linear(canonical_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        block_count = 0 if variant == "no_interactions" else 2
        self.interactions = nn.ModuleList(
            [MassConservingAttentionBlock(hidden) for _ in range(block_count)]
        )
        self.queries = nn.Parameter(0.02 * torch.randn(query_count, hidden))
        self.pool_key = nn.Linear(hidden, hidden)
        self.pool_value = nn.Linear(hidden, hidden)
        summary_dim = query_count * hidden + 2 * canonical_dim + zone_count + 2
        self.trunk = nn.Sequential(
            nn.Linear(summary_dim, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(192, 96),
            nn.GELU(),
        )
        self.residual_head = nn.Linear(96, zone_count)
        self.tail_head = nn.Sequential(
            nn.Linear(96 + 1, 48),
            nn.GELU(),
            nn.Linear(48, 3),
        )

    @staticmethod
    def _pairwise(field: torch.Tensor) -> torch.Tensor:
        indices = torch.triu_indices(field.shape[-1], field.shape[-1], offset=1)
        return field[:, indices[0]] - field[:, indices[1]]

    def forward(self, batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
        atoms = batch["canonical"].to(device)
        weights = batch["weights"].to(device)
        mask = batch["mask"].to(device)
        normalized_atoms = (atoms - self.atom_mean) / self.atom_std
        encoded = self.encoder(normalized_atoms) * mask.unsqueeze(-1)
        for interaction in self.interactions:
            encoded = interaction(encoded, weights, mask)

        keys = self.pool_key(encoded)
        values = self.pool_value(encoded)
        scores = torch.einsum("qd,bnd->bqn", self.queries, keys) / math.sqrt(keys.shape[-1])
        scores = scores.masked_fill(~mask.unsqueeze(1), -1.0e9)
        shifted = scores - scores.amax(dim=-1, keepdim=True)
        numerator = torch.exp(shifted) * weights.unsqueeze(1) * mask.unsqueeze(1)
        attention = numerator / numerator.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        pooled = torch.einsum("bqn,bnd->bqd", attention, values).flatten(1)

        normalized_mass = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        first_moment = torch.einsum("bn,bnd->bd", normalized_mass, atoms)
        second_moment = torch.einsum("bn,bnd->bd", normalized_mass, torch.square(atoms))
        summary = torch.cat(
            [
                pooled,
                first_moment,
                second_moment,
                batch["total"].to(device),
                batch["present"].to(device),
                batch["analytic"].to(device),
            ],
            dim=-1,
        )
        latent = self.trunk(summary)
        residual = self.residual_head(latent)
        residual = residual - residual.mean(dim=-1, keepdim=True)
        predicted_pair_magnitude = self._pairwise(residual).abs().mean(dim=-1, keepdim=True)
        tail_logits = self.tail_head(torch.cat([latent, predicted_pair_magnitude], dim=-1))
        return {
            "tail_logits": tail_logits,
            "residual": residual,
            "pair_residual": self._pairwise(residual),
        }


class MomentMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _brier_from_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.square(torch.sigmoid(logits) - target))


def _fit_sentinel(
    seed: int,
    train: TailDataset,
    tuning: TailDataset,
    *,
    canonical_dim: int,
    zone_count: int,
    variant: str,
    atom_mean: np.ndarray,
    atom_std: np.ndarray,
    tail_thresholds: np.ndarray,
    epochs: int,
    batch_size: int,
    patience: int,
    device: torch.device,
) -> tuple[QuotientTangentOperator, dict[str, Any]]:
    _seed_everything(seed)
    model = QuotientTangentOperator(
        canonical_dim,
        zone_count,
        variant=variant,
        atom_mean=atom_mean,
        atom_std=atom_std,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=3.0e-3)
    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=_collate,
    )
    tuning_loader = DataLoader(tuning, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    best_loss = float("inf")
    best_average_precision = -float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    history: list[dict[str, float]] = []
    threshold_tensor = torch.tensor(tail_thresholds, dtype=torch.float32, device=device)
    ordinal_weights = torch.tensor([0.20, 0.60, 0.20], dtype=torch.float32, device=device)
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            magnitude = torch.tensor(batch["magnitude"], dtype=torch.float32, device=device)
            ordinal_target = (magnitude[:, None] > threshold_tensor[None, :]).float()
            output = model(batch, device)
            ordinal_loss = nn.functional.binary_cross_entropy_with_logits(
                output["tail_logits"], ordinal_target, reduction="none"
            )
            classification_loss = (ordinal_loss * ordinal_weights[None, :]).mean()
            monotonic_loss = (
                torch.relu(output["tail_logits"][:, 1] - output["tail_logits"][:, 0]).mean()
                + torch.relu(output["tail_logits"][:, 2] - output["tail_logits"][:, 1]).mean()
            )
            if variant == "no_tangent":
                tangent_loss = torch.zeros((), dtype=torch.float32, device=device)
            else:
                zone_loss = nn.functional.smooth_l1_loss(
                    output["residual"], batch["residual"].to(device), beta=0.25
                )
                pair_loss = nn.functional.smooth_l1_loss(
                    output["pair_residual"], batch["pair_residual"].to(device), beta=0.25
                )
                tangent_loss = zone_loss + 0.5 * pair_loss
            loss = classification_loss + 0.35 * tangent_loss + 0.05 * monotonic_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        tuning_logits: list[np.ndarray] = []
        tuning_targets: list[np.ndarray] = []
        with torch.no_grad():
            for batch in tuning_loader:
                output = model(batch, device)
                tuning_logits.append(output["tail_logits"][:, 1].cpu().numpy())
                tuning_targets.append(batch["target"].numpy())
        tuning_logit = np.concatenate(tuning_logits)
        tuning_target = np.concatenate(tuning_targets)
        tuning_probability = 1.0 / (1.0 + np.exp(-np.clip(tuning_logit, -30.0, 30.0)))
        tuning_loss = float(brier_score_loss(tuning_target, tuning_probability))
        tuning_average_precision = float(
            average_precision_score(tuning_target, tuning_probability)
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_log_loss": float(np.mean(train_losses)),
                "tuning_brier": tuning_loss,
                "tuning_average_precision": tuning_average_precision,
            }
        )
        if (
            tuning_average_precision > best_average_precision + 1.0e-5
            or (
                abs(tuning_average_precision - best_average_precision) <= 1.0e-5
                and tuning_loss < best_loss - 1.0e-6
            )
        ):
            best_loss = tuning_loss
            best_average_precision = tuning_average_precision
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_tuning_brier": best_loss,
        "best_tuning_average_precision": best_average_precision,
        "loss": "nested unweighted BCE + gauge-tangent SmoothL1 + ordinal monotonicity",
        "tail_thresholds_eur_mwh": [float(value) for value in tail_thresholds],
        "history": history,
    }


def _predict_sentinel(
    model: QuotientTangentOperator,
    dataset: TailDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    logits: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    magnitudes: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    pair_targets: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            output = model(batch, device)
            logits.append(output["tail_logits"][:, 1].cpu().numpy())
            targets.append(batch["target"].numpy())
            magnitudes.append(batch["magnitude"])
            residuals.append(output["pair_residual"].cpu().numpy())
            pair_targets.append(batch["pair_residual"].numpy())
    return (
        np.concatenate(logits),
        np.concatenate(targets),
        np.concatenate(magnitudes),
        np.concatenate(residuals),
        np.concatenate(pair_targets),
    )


def _atom_standardization(dataset: TailDataset) -> tuple[np.ndarray, np.ndarray]:
    dimension = dataset[0]["canonical"].shape[1]
    first = np.zeros(dimension, dtype=np.float64)
    second = np.zeros(dimension, dtype=np.float64)
    mass = 0.0
    for index in range(len(dataset)):
        item = dataset[index]
        atoms = np.asarray(item["canonical"], dtype=np.float64)
        weights = np.asarray(item["weights"], dtype=np.float64)
        if not len(weights):
            continue
        normalized = weights / weights.sum()
        first += normalized @ atoms
        second += normalized @ np.square(atoms)
        mass += 1.0
    if mass <= 0.0:
        raise RuntimeError("Cannot standardize an empty quotient measure")
    mean = first / mass
    variance = np.maximum(second / mass - np.square(mean), 1.0e-6)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _pair_residual_metrics(
    target_scaled: np.ndarray,
    prediction_scaled: np.ndarray,
    target_scale: float,
) -> dict[str, float]:
    error = (prediction_scaled - target_scaled) * target_scale
    timestamp_mae = np.mean(np.abs(error), axis=1)
    return {
        "mean_pairwise_mae_eur_mwh": float(np.mean(timestamp_mae)),
        "median_timestamp_pairwise_mae_eur_mwh": float(np.median(timestamp_mae)),
        "p95_timestamp_pairwise_mae_eur_mwh": float(np.quantile(timestamp_mae, 0.95)),
        "pairwise_rmse_eur_mwh": float(np.sqrt(np.mean(np.square(error)))),
    }


def _numpy_pairwise(field: np.ndarray) -> np.ndarray:
    left, right = np.triu_indices(field.shape[1], k=1)
    return field[:, left] - field[:, right]


def _logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability, EPSILON, 1.0 - EPSILON)
    return np.log(probability / (1.0 - probability))


def _platt(
    calibration_logits: np.ndarray,
    calibration_target: np.ndarray,
    evaluation_logits: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    model = LogisticRegression(C=1.0, solver="lbfgs", random_state=BOOTSTRAP_SEED)
    model.fit(calibration_logits[:, None], calibration_target)
    probability = model.predict_proba(evaluation_logits[:, None])[:, 1]
    return probability, {
        "intercept": float(model.intercept_[0]),
        "logit_coefficient": float(model.coef_[0, 0]),
    }


def _metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    probability = np.clip(probability, EPSILON, 1.0 - EPSILON)
    return {
        "prevalence": float(np.mean(target)),
        "brier": float(brier_score_loss(target, probability)),
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(target, probability)),
        "average_precision": float(average_precision_score(target, probability)),
        "mean_probability": float(np.mean(probability)),
    }


def _risk_coverage(
    target: np.ndarray,
    probability: np.ndarray,
    magnitude: np.ndarray,
) -> dict[str, dict[str, float]]:
    order = np.argsort(probability)
    output: dict[str, dict[str, float]] = {}
    for coverage in (0.50, 0.70, 0.80, 0.90, 0.95, 1.00):
        count = max(1, int(math.floor(coverage * len(order))))
        kept = order[:count]
        output[f"{coverage:.2f}"] = {
            "rows": int(count),
            "tail_rate": float(np.mean(target[kept])),
            "mean_residual_magnitude_eur_mwh": float(np.mean(magnitude[kept])),
            "p95_residual_magnitude_eur_mwh": float(np.quantile(magnitude[kept], 0.95)),
        }
    return output


def _circular_indices(length: int, rng: np.random.Generator) -> np.ndarray:
    blocks = int(math.ceil(length / BLOCK_ROWS))
    starts = rng.integers(0, length, size=blocks)
    offsets = np.arange(BLOCK_ROWS)
    return ((starts[:, None] + offsets[None, :]) % length).ravel()[:length]


def _paired_brier(
    target: np.ndarray,
    baseline_probability: np.ndarray,
    candidate_probability: np.ndarray,
) -> dict[str, Any]:
    baseline_loss = np.square(baseline_probability - target)
    candidate_loss = np.square(candidate_probability - target)
    difference = candidate_loss - baseline_loss
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        rows = _circular_indices(len(target), rng)
        draws[index] = float(np.mean(difference[rows]))
    return {
        "candidate_minus_baseline_brier": float(np.mean(difference)),
        "block_bootstrap_95ci": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "probability_of_improvement": float(np.mean(draws < 0.0)),
        "block_rows": BLOCK_ROWS,
        "replicates": BOOTSTRAP_REPLICATES,
    }


def _fixed_quotient_features(
    dataset: TailDataset,
    real_indices: np.ndarray,
) -> np.ndarray:
    values = []
    for index in range(len(dataset)):
        item = dataset[index]
        # RealAffineGaugeDataset keeps the source sample under ``total`` for
        # provenance, while ``total_scaled`` is recomputed after projecting
        # the certificate into the affine-feasible quotient.  The matched
        # classical control must use the latter mass, just like the neural
        # model, or it would receive a different market representation.
        projected_total = float(np.expm1(float(item["total_scaled"])))
        values.append(
            quotient_feature(
                item["canonical"], item["weights"], projected_total, real_indices
            )
        )
    return np.stack(values)


def _fit_moment_mlp(
    seed: int,
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    tuning_x: np.ndarray,
    tuning_y: np.ndarray,
    calibration_x: np.ndarray,
    calibration_y: np.ndarray,
    evaluation_x: np.ndarray,
    *,
    epochs: int,
    patience: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    _seed_everything(seed)
    scaler = StandardScaler().fit(fit_x)
    fit_tensor = torch.tensor(scaler.transform(fit_x), dtype=torch.float32)
    fit_target = torch.tensor(fit_y, dtype=torch.float32)
    tuning_tensor = torch.tensor(scaler.transform(tuning_x), dtype=torch.float32, device=device)
    tuning_target = torch.tensor(tuning_y, dtype=torch.float32, device=device)
    model = MomentMLP(fit_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8.0e-4, weight_decay=2.0e-3)
    loader = DataLoader(
        torch.utils.data.TensorDataset(fit_tensor, fit_target),
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    best = float("inf")
    best_average_precision = -float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    stale = 0
    positive_weight = float(
        max(np.sum(fit_y == 0.0), 1.0) / max(np.sum(fit_y == 1.0), 1.0)
    )
    pos_weight_tensor = torch.tensor(positive_weight, dtype=torch.float32, device=device)
    for epoch in range(1, epochs + 1):
        model.train()
        for values, target in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(values.to(device))
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, target.to(device), pos_weight=pos_weight_tensor
            )
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            tuning_logits = model(tuning_tensor)
            tuning_brier = float(_brier_from_logits(tuning_logits, tuning_target).cpu())
            tuning_probability = torch.sigmoid(tuning_logits).cpu().numpy()
            tuning_average_precision = float(
                average_precision_score(tuning_y, tuning_probability)
            )
        if (
            tuning_average_precision > best_average_precision + 1.0e-5
            or (
                abs(tuning_average_precision - best_average_precision) <= 1.0e-5
                and tuning_brier < best - 1.0e-6
            )
        ):
            best = tuning_brier
            best_average_precision = tuning_average_precision
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        calibration_logits = model(
            torch.tensor(scaler.transform(calibration_x), dtype=torch.float32, device=device)
        ).cpu().numpy()
        evaluation_logits = model(
            torch.tensor(scaler.transform(evaluation_x), dtype=torch.float32, device=device)
        ).cpu().numpy()
    probability, calibration = _platt(calibration_logits, calibration_y, evaluation_logits)
    return probability, {
        "best_epoch": best_epoch,
        "best_tuning_brier": best,
        "best_tuning_average_precision": best_average_precision,
        "fit_positive_weight": positive_weight,
        "platt": calibration,
    }


def _rewrite_invariance(
    model: QuotientTangentOperator,
    samples: list[dict[str, Any]],
    *,
    raw_scale: np.ndarray,
    target_scale: float,
    real_indices: np.ndarray,
    geometry: RealAffineGeometry,
    canonical_ram_scale: float,
    threshold: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    probe_samples = samples[: min(64, len(samples))]

    def make(presentation: str) -> TailDataset:
        return TailDataset(
            RealAffineGaugeDataset(
                probe_samples,
                raw_scale=raw_scale,
                target_scale=target_scale,
                presentation=presentation,
                representation="af",
                real_zone_indices=real_indices,
                geometry=geometry,
                canonical_ram_scale=canonical_ram_scale,
            ),
            threshold,
        )

    original_logits, _, _, original_tangent, _ = _predict_sentinel(
        model, make("original"), batch_size=batch_size, device=device
    )
    logit_records: dict[str, float] = {}
    tangent_records: dict[str, float] = {}
    for presentation in presentation_names(geometry):
        stressed_logits, _, _, stressed_tangent, _ = _predict_sentinel(
            model, make(presentation), batch_size=batch_size, device=device
        )
        logit_records[presentation] = float(
            np.max(np.abs(stressed_logits - original_logits), initial=0.0)
        )
        tangent_records[presentation] = float(
            np.max(np.abs(stressed_tangent - original_tangent), initial=0.0)
        )
    return {
        "rows": len(probe_samples),
        "max_abs_logit_drift": float(max(logit_records.values(), default=0.0)),
        "max_abs_pair_residual_drift_scaled": float(
            max(tangent_records.values(), default=0.0)
        ),
        "logit_by_presentation": logit_records,
        "pair_residual_by_presentation_scaled": tangent_records,
    }


@dataclass(frozen=True)
class SplitData:
    fit: TailDataset
    tuning: TailDataset
    calibration: TailDataset
    evaluation: TailDataset


def main() -> int:
    args = _parse_args()
    if DEVELOPMENT_END != CONFIRMATION_START:
        raise RuntimeError("Development boundary is not the frozen confirmation start")
    if not (FIT_END < TUNING_END < CALIBRATION_END < DEVELOPMENT_END):
        raise RuntimeError("Invalid chronological split")

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    dual_path = args.dual.expanduser().resolve()
    price_root = args.price_root.expanduser().resolve()
    geometry_audit_path = args.geometry_audit.expanduser().resolve()
    geometry_audit = json.loads(geometry_audit_path.read_text(encoding="utf-8"))
    if not geometry_audit["observed_transfer_queries"]["all_tangent"]:
        raise RuntimeError("Frozen geometry audit did not certify tangent queries")
    if geometry_audit["registered_gauge_rewrite"]["status"] != "passed":
        raise RuntimeError("Frozen geometry audit did not certify gauge rewrites")

    frame, certificate_audit = _read_certificate(dual_path, 60.0)
    unit_columns = certificate_audit["unit_columns"]
    input_zones = tuple(column.removeprefix("unit_ptdf_") for column in unit_columns)
    geometry = RealAffineGeometry.loss_safe(input_zones)
    if list(input_zones) != geometry_audit["geometry"]["zone_order"]:
        raise RuntimeError("PTDF order differs from the frozen geometry audit")
    zones = PRIMARY_ZONES
    real_indices = np.asarray(
        [unit_columns.index(f"unit_ptdf_{zone}") for zone in zones], dtype=np.int64
    )
    price_panel, price_provenance = _load_price_panel(price_root, zones)
    samples = _build_vector_samples(
        frame, price_panel, unit_columns, zones, FIT_END, DEVELOPMENT_END
    )
    if not samples or max(pd.Timestamp(sample["timestamp"]) for sample in samples) >= CONFIRMATION_START:
        raise RuntimeError("This isolated extension attempted to enter confirmation")

    fit_samples = [sample for sample in samples if sample["timestamp"] < FIT_END]
    tuning_samples = [
        sample for sample in samples if FIT_END <= sample["timestamp"] < TUNING_END
    ]
    calibration_samples = [
        sample
        for sample in samples
        if TUNING_END <= sample["timestamp"] < CALIBRATION_END
    ]
    evaluation_samples = [
        sample
        for sample in samples
        if CALIBRATION_END <= sample["timestamp"] < DEVELOPMENT_END
    ]
    if args.smoke:
        fit_samples = fit_samples[-2400:]
        tuning_samples = tuning_samples[-1200:]
        calibration_samples = calibration_samples[-1200:]
        evaluation_samples = evaluation_samples[:1600]
        args.seeds = args.seeds[:1]
        args.epochs = min(args.epochs, 4)
        args.patience = min(args.patience, 2)
    if min(map(len, (fit_samples, tuning_samples, calibration_samples, evaluation_samples))) == 0:
        raise RuntimeError("At least one chronological split is empty")

    target_values = np.stack([sample["target_vector"] for sample in fit_samples])
    target_scale = max(float(np.median(np.abs(target_values))), 1.0)
    canonical_ram_scale = _canonical_ram_scale(frame, FIT_END)
    raw_dim = 1 + len(unit_columns) + 5
    raw_scale = _positive_scales(
        raw_rows_for_scale(
            fit_samples,
            representation="af",
            real_zone_indices=real_indices,
            geometry=geometry,
            canonical_ram_scale=canonical_ram_scale,
        )
    )

    def base(split_samples: list[dict[str, Any]]) -> RealAffineGaugeDataset:
        return RealAffineGaugeDataset(
            split_samples,
            raw_scale=raw_scale,
            target_scale=target_scale,
            presentation="original",
            representation="af",
            real_zone_indices=real_indices,
            geometry=geometry,
            canonical_ram_scale=canonical_ram_scale,
        )

    fit_base = base(fit_samples)
    fit_magnitude = np.asarray(
        [_residual_magnitude(fit_base[index], target_scale) for index in range(len(fit_base))]
    )
    tail_thresholds = np.quantile(fit_magnitude, [0.75, TAIL_QUANTILE, 0.95])
    threshold = float(tail_thresholds[1])
    split = SplitData(
        fit=TailDataset(fit_base, threshold),
        tuning=TailDataset(base(tuning_samples), threshold),
        calibration=TailDataset(base(calibration_samples), threshold),
        evaluation=TailDataset(base(evaluation_samples), threshold),
    )
    canonical_dim = fit_base[0]["canonical"].shape[1]
    atom_mean, atom_std = _atom_standardization(split.fit)

    quotient = {
        name: _fixed_quotient_features(dataset, real_indices)
        for name, dataset in (
            ("fit", split.fit),
            ("tuning", split.tuning),
            ("calibration", split.calibration),
            ("evaluation", split.evaluation),
        )
    }
    targets = {
        name: np.asarray([dataset[index]["tail_target"] for index in range(len(dataset))])
        for name, dataset in (
            ("fit", split.fit),
            ("tuning", split.tuning),
            ("calibration", split.calibration),
            ("evaluation", split.evaluation),
        )
    }
    residual_targets = {
        name: np.stack([dataset[index]["residual_scaled"] for index in range(len(dataset))])
        for name, dataset in (
            ("fit", split.fit),
            ("tuning", split.tuning),
            ("calibration", split.calibration),
            ("evaluation", split.evaluation),
        )
    }
    evaluation_magnitude = np.asarray(
        [split.evaluation[index]["residual_magnitude"] for index in range(len(split.evaluation))]
    )

    hgb = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=220,
        max_leaf_nodes=15,
        min_samples_leaf=50,
        l2_regularization=3.0,
        random_state=BOOTSTRAP_SEED,
    ).fit(quotient["fit"], targets["fit"])
    hgb_calibration_raw = hgb.predict_proba(quotient["calibration"])[:, 1]
    hgb_evaluation_raw = hgb.predict_proba(quotient["evaluation"])[:, 1]
    hgb_calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=BOOTSTRAP_SEED)
    hgb_calibrator.fit(_logit(hgb_calibration_raw)[:, None], targets["calibration"])
    hgb_probability = hgb_calibrator.predict_proba(_logit(hgb_evaluation_raw)[:, None])[:, 1]

    hgb_residual_columns: list[np.ndarray] = []
    for zone_index in range(len(zones)):
        regressor = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.05,
            max_iter=180,
            max_leaf_nodes=15,
            min_samples_leaf=50,
            l2_regularization=3.0,
            random_state=BOOTSTRAP_SEED + zone_index,
        ).fit(quotient["fit"], residual_targets["fit"][:, zone_index])
        hgb_residual_columns.append(regressor.predict(quotient["evaluation"]))
    hgb_residual = np.column_stack(hgb_residual_columns)
    hgb_residual -= hgb_residual.mean(axis=1, keepdims=True)
    hgb_pair_residual = _numpy_pairwise(hgb_residual)
    evaluation_pair_target = _numpy_pairwise(residual_targets["evaluation"])

    seed_results: dict[str, Any] = {}
    sentinel_probabilities: list[np.ndarray] = []
    tangent_predictions: list[np.ndarray] = []
    moment_probabilities: list[np.ndarray] = []
    invariant_records: list[dict[str, Any]] = []
    for seed in args.seeds:
        print(f"[QCT-RISK] seed={seed} device={device}", flush=True)
        model, training = _fit_sentinel(
            seed,
            split.fit,
            split.tuning,
            canonical_dim=canonical_dim,
            zone_count=len(zones),
            variant=args.variant,
            atom_mean=atom_mean,
            atom_std=atom_std,
            tail_thresholds=tail_thresholds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            device=device,
        )
        calibration_logits, calibration_target, _, _, _ = _predict_sentinel(
            model, split.calibration, batch_size=args.batch_size, device=device
        )
        (
            evaluation_logits,
            evaluation_target,
            _,
            evaluation_tangent,
            evaluation_pair_target_check,
        ) = _predict_sentinel(
            model, split.evaluation, batch_size=args.batch_size, device=device
        )
        if not np.allclose(evaluation_pair_target_check, evaluation_pair_target, atol=1.0e-6):
            raise RuntimeError("Pair-residual target drifted between dataset paths")
        probability, calibration = _platt(
            calibration_logits, calibration_target, evaluation_logits
        )
        moment_probability, moment_training = _fit_moment_mlp(
            seed,
            quotient["fit"],
            targets["fit"],
            quotient["tuning"],
            targets["tuning"],
            quotient["calibration"],
            targets["calibration"],
            quotient["evaluation"],
            epochs=args.epochs,
            patience=args.patience,
            device=device,
        )
        invariance = _rewrite_invariance(
            model,
            evaluation_samples,
            raw_scale=raw_scale,
            target_scale=target_scale,
            real_indices=real_indices,
            geometry=geometry,
            canonical_ram_scale=canonical_ram_scale,
            threshold=threshold,
            batch_size=args.batch_size,
            device=device,
        )
        sentinel_probabilities.append(probability)
        tangent_predictions.append(evaluation_tangent)
        moment_probabilities.append(moment_probability)
        invariant_records.append(invariance)
        seed_results[str(seed)] = {
            "sentinel": _metrics(evaluation_target, probability),
            "tangent_residual": _pair_residual_metrics(
                evaluation_pair_target,
                evaluation_tangent,
                target_scale,
            ),
            "moment_mlp": _metrics(evaluation_target, moment_probability),
            "training": training,
            "moment_training": moment_training,
            "platt": calibration,
            "rewrite_invariance": invariance,
        }
        print(
            "[QCT-RISK] "
            f"seed={seed} done brier={seed_results[str(seed)]['sentinel']['brier']:.6f} "
            f"auc={seed_results[str(seed)]['sentinel']['roc_auc']:.4f} "
            f"rewrite_drift={invariance['max_abs_logit_drift']:.3e}",
            flush=True,
        )

    target = targets["evaluation"]
    sentinel_ensemble = np.mean(np.stack(sentinel_probabilities), axis=0)
    tangent_ensemble = np.mean(np.stack(tangent_predictions), axis=0)
    moment_ensemble = np.mean(np.stack(moment_probabilities), axis=0)
    prevalence = float(np.mean(targets["fit"]))
    unconditional = np.full(len(target), prevalence, dtype=np.float64)
    report = {
        "status": "development_only_neural_extension",
        "claim_boundary": (
            "No timestamp on or after 2026-03-01 UTC was read. Results justify only "
            "an isolated tail-risk extension, not a revision of frozen AF-CDMO claims."
        ),
        "protocol": {
            "name": "quotient_tangent_measure_operator_v3",
            "fit_end_exclusive": FIT_END.isoformat(),
            "tuning_end_exclusive": TUNING_END.isoformat(),
            "calibration_end_exclusive": CALIBRATION_END.isoformat(),
            "development_end_exclusive": DEVELOPMENT_END.isoformat(),
            "confirmation_start": CONFIRMATION_START.isoformat(),
            "tail_quantile": TAIL_QUANTILE,
            "tail_threshold_eur_mwh": threshold,
            "nested_tail_thresholds_eur_mwh": [
                float(value) for value in tail_thresholds
            ],
            "variant": args.variant,
            "seeds": args.seeds,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "device": str(device),
            "smoke": bool(args.smoke),
        },
        "rows": {
            "fit": len(split.fit),
            "tuning": len(split.tuning),
            "calibration": len(split.calibration),
            "development_evaluation": len(split.evaluation),
        },
        "sources": {
            "dual_archive": str(dual_path),
            "dual_archive_sha256": _sha256(dual_path),
            "price_root": str(price_root),
            "price_provenance": price_provenance,
            "geometry_audit": str(geometry_audit_path),
            "geometry_audit_sha256": _sha256(geometry_audit_path),
            "engine": str(Path(__file__).resolve()),
            "engine_sha256": _sha256(Path(__file__).resolve()),
        },
        "architecture": {
            "input": "exact affine-feasible market-equivalence quotient measure",
            "aggregation": (
                "mass-conserving atom-to-atom attention followed by dual-mass "
                "query integration and exact quotient-moment skip"
            ),
            "output": "centered gauge-tangent residual plus nested q75/q90/q95 tail logits",
            "target": (
                "centered zonal KKT residual and mean absolute pairwise residual "
                "spread thresholds"
            ),
            "selection_metric": "November average precision; December Platt calibration",
            "parameter_count": int(
                sum(
                    parameter.numel()
                    for parameter in QuotientTangentOperator(
                        canonical_dim,
                        len(zones),
                        variant=args.variant,
                        atom_mean=atom_mean,
                        atom_std=atom_std,
                    ).parameters()
                )
            ),
        },
        "evaluation_target": {
            "prevalence": float(np.mean(target)),
            "mean_residual_magnitude_eur_mwh": float(np.mean(evaluation_magnitude)),
            "p90_residual_magnitude_eur_mwh": float(np.quantile(evaluation_magnitude, 0.90)),
            "p95_residual_magnitude_eur_mwh": float(np.quantile(evaluation_magnitude, 0.95)),
        },
        "controls": {
            "unconditional": _metrics(target, unconditional),
            "hgb_quotient_moments": _metrics(target, hgb_probability),
            "moment_mlp_ensemble": _metrics(target, moment_ensemble),
            "zero_tangent_residual": _pair_residual_metrics(
                evaluation_pair_target,
                np.zeros_like(evaluation_pair_target),
                target_scale,
            ),
            "hgb_quotient_tangent_residual": _pair_residual_metrics(
                evaluation_pair_target,
                hgb_pair_residual,
                target_scale,
            ),
        },
        "sentinel": {
            "ensemble": _metrics(target, sentinel_ensemble),
            "risk_coverage": _risk_coverage(
                target, sentinel_ensemble, evaluation_magnitude
            ),
            "paired_brier_vs_unconditional": _paired_brier(
                target, unconditional, sentinel_ensemble
            ),
            "paired_brier_vs_hgb_quotient_moments": _paired_brier(
                target, hgb_probability, sentinel_ensemble
            ),
            "paired_brier_vs_moment_mlp": _paired_brier(
                target, moment_ensemble, sentinel_ensemble
            ),
            "max_registered_rewrite_logit_drift": float(
                max(
                    record["max_abs_logit_drift"]
                    for record in invariant_records
                )
            ),
            "max_registered_rewrite_pair_residual_drift_scaled": float(
                max(
                    record["max_abs_pair_residual_drift_scaled"]
                    for record in invariant_records
                )
            ),
            "tangent_residual": _pair_residual_metrics(
                evaluation_pair_target,
                tangent_ensemble,
                target_scale,
            ),
        },
        "seed_results": seed_results,
    }
    _atomic_json(args.output.expanduser().resolve(), report)
    print(f"[OK] wrote {args.output.expanduser().resolve()}")
    print(
        json.dumps(
            {
                "sentinel": report["sentinel"]["ensemble"],
                "hgb": report["controls"]["hgb_quotient_moments"],
                "moment_mlp": report["controls"]["moment_mlp_ensemble"],
                "rewrite_drift": report["sentinel"]["max_registered_rewrite_logit_drift"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
