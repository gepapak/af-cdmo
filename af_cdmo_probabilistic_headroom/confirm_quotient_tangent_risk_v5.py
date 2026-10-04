"""Frozen confirmation of quotient-conditioned tangent risk (QCT-Risk).

This companion script evaluates the development-selected v3 architecture on
the untouched AF-CDMO confirmation chronology.  It deliberately imports the
development engine instead of changing it, verifies the frozen development
artifact and q90 threshold, and never uses confirmation labels for model
selection or probability calibration.

The task is same-delivery market-certificate assurance, not forecasting.
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
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]

try:
    from af_cdmo_probabilistic_headroom import train_quotient_tangent_operator_v3 as qct
    from scripts.af_cdmo_real_controls_v14 import (
        NonEquivalentRAMDataset,
        NonuniformRefinementAFDataset,
    )
    from scripts.af_cdmo_real_extension_v13 import RealAffineGaugeDataset, RealAffineGeometry, presentation_names, raw_rows_for_scale
    from scripts.benchmark_af_cdmo_real_v14 import DEFAULT_GEOMETRY_AUDIT, _canonical_ram_scale
    from scripts.benchmark_certificate_governance_v5 import _positive_scales
    from scripts.multizone_quotient_gauge_v6 import quotient_feature
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(ROOT))
    from af_cdmo_probabilistic_headroom import train_quotient_tangent_operator_v3 as qct
    from scripts.af_cdmo_real_controls_v14 import (
        NonEquivalentRAMDataset,
        NonuniformRefinementAFDataset,
    )
    from scripts.af_cdmo_real_extension_v13 import RealAffineGaugeDataset, RealAffineGeometry, presentation_names, raw_rows_for_scale
    from scripts.benchmark_af_cdmo_real_v14 import DEFAULT_GEOMETRY_AUDIT, _canonical_ram_scale
    from scripts.benchmark_certificate_governance_v5 import _positive_scales
    from scripts.multizone_quotient_gauge_v6 import quotient_feature


DEVELOPMENT_ARTIFACT = ROOT / "af_cdmo_probabilistic_headroom" / "quotient_tangent_operator_development_v3.json"
CONFIRMATION_REPORT = ROOT / "official_jao_dual_measure" / "af_cdmo_real_confirmation_v14.json"
DEFAULT_OUTPUT = ROOT / "af_cdmo_probabilistic_headroom" / "quotient_tangent_risk_confirmation_v5.json"
DEFAULT_CHECKPOINT_DIR = ROOT / "af_cdmo_probabilistic_headroom" / "confirmation_checkpoints_v5"
PROTOCOL_NOTE = ROOT / "af_cdmo_probabilistic_headroom" / "QCT_RISK_CONFIRMATION_PROTOCOL_V5.md"
EXPECTED_DEVELOPMENT_ARTIFACT_SHA256 = "18cc7313a78afeddd6a90fc60394a6bd269128ebcee25a93dd5fbc9b97c6f821"
EXPECTED_DEVELOPMENT_ENGINE_SHA256 = "d88d6cb50f476f7646bd0397f20c419ae2332477604d4470f956edf79d81f558"
EXPECTED_THRESHOLD = 5.592173604525909
THRESHOLD_REPRODUCTION_ATOL = 1.0e-6
REGISTERED_SEEDS = (7, 42, 123, 2025, 3007)
REGISTERED_METHODS = (
    "quotient_deepset",
    "uniform_set_transformer",
    "qct_full",
    "qct_no_interactions",
    "qct_no_tangent",
)
RAM_PERTURBATIONS = (-0.20, -0.05, 0.05, 0.20)
EPSILON = 1.0e-7
BLOCK_ROWS = 672
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261002


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dual", type=Path, default=qct.DEFAULT_DUAL)
    parser.add_argument("--price_root", type=Path, default=qct.DEFAULT_PRICE_ROOT)
    parser.add_argument("--geometry_audit", type=Path, default=DEFAULT_GEOMETRY_AUDIT)
    parser.add_argument("--development_artifact", type=Path, default=DEVELOPMENT_ARTIFACT)
    parser.add_argument("--confirmation_report", type=Path, default=CONFIRMATION_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint_dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(REGISTERED_SEEDS))
    parser.add_argument("--methods", nargs="+", choices=REGISTERED_METHODS, default=list(REGISTERED_METHODS))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_full_rewrite_sweep", action="store_true")
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


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _as_utc(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("Chronology boundary must include a timezone")
    return timestamp.tz_convert("UTC")


class RiskDataset(Dataset):
    """Add frozen tail and calendar targets to any certificate dataset."""

    def __init__(self, base: Dataset, *, target_scale: float, threshold: float) -> None:
        self.base = base
        self.target_scale = float(target_scale)
        self.threshold = float(threshold)
        self._cache: list[dict[str, Any] | None] = [None] * len(base)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        cached = self._cache[index]
        if cached is not None:
            return cached
        item = dict(self.base[index])
        magnitude = qct._residual_magnitude(item, self.target_scale)
        item["tail_target"] = np.float32(magnitude > self.threshold)
        item["residual_magnitude"] = np.float32(magnitude)
        item["calendar"] = qct._calendar(pd.Timestamp(item["timestamp"]))
        self._cache[index] = item
        return item


class MassDeepSetOperator(nn.Module):
    """Mass-weighted Deep Sets control on the exact quotient measure."""

    def __init__(self, canonical_dim: int, zone_count: int, atom_mean: np.ndarray, atom_std: np.ndarray) -> None:
        super().__init__()
        hidden = 64
        self.zone_count = zone_count
        self.register_buffer("atom_mean", torch.tensor(atom_mean, dtype=torch.float32))
        self.register_buffer("atom_std", torch.tensor(atom_std, dtype=torch.float32))
        self.encoder = nn.Sequential(
            nn.Linear(canonical_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU()
        )
        summary_dim = hidden + 2 * canonical_dim + zone_count + 2
        self.trunk = nn.Sequential(
            nn.Linear(summary_dim, 192), nn.LayerNorm(192), nn.GELU(), nn.Dropout(0.05), nn.Linear(192, 96), nn.GELU()
        )
        self.residual_head = nn.Linear(96, zone_count)
        self.tail_head = nn.Sequential(nn.Linear(97, 48), nn.GELU(), nn.Linear(48, 3))

    @staticmethod
    def _pairwise(field: torch.Tensor) -> torch.Tensor:
        indices = torch.triu_indices(field.shape[-1], field.shape[-1], offset=1)
        return field[:, indices[0]] - field[:, indices[1]]

    def forward(self, batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
        atoms = batch["canonical"].to(device)
        weights = batch["weights"].to(device)
        mask = batch["mask"].to(device)
        encoded = self.encoder((atoms - self.atom_mean) / self.atom_std) * mask.unsqueeze(-1)
        normalized_mass = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        pooled = torch.einsum("bn,bnd->bd", normalized_mass, encoded)
        first = torch.einsum("bn,bnd->bd", normalized_mass, atoms)
        second = torch.einsum("bn,bnd->bd", normalized_mass, torch.square(atoms))
        summary = torch.cat(
            [pooled, first, second, batch["total"].to(device), batch["present"].to(device), batch["analytic"].to(device)], dim=-1
        )
        latent = self.trunk(summary)
        residual = self.residual_head(latent)
        residual = residual - residual.mean(dim=-1, keepdim=True)
        pair = self._pairwise(residual)
        magnitude = pair.abs().mean(dim=-1, keepdim=True)
        return {"tail_logits": self.tail_head(torch.cat([latent, magnitude], dim=-1)), "residual": residual, "pair_residual": pair}


class UniformAttentionBlock(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden, 4, dropout=0.05, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Dropout(0.05), nn.Linear(2 * hidden, hidden))
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        safe_mask = mask.clone()
        empty = ~safe_mask.any(dim=1)
        safe_mask[empty, 0] = True
        message, _ = self.attention(hidden, hidden, hidden, key_padding_mask=~safe_mask, need_weights=False)
        hidden = self.norm1(hidden + message)
        hidden = self.norm2(hidden + self.ff(hidden))
        return hidden * mask.unsqueeze(-1)


class UniformSetTransformerOperator(nn.Module):
    """Ordinary token-count Set Transformer; not split/merge invariant."""

    def __init__(self, canonical_dim: int, zone_count: int, atom_mean: np.ndarray, atom_std: np.ndarray) -> None:
        super().__init__()
        hidden = 64
        query_count = 6
        self.zone_count = zone_count
        self.register_buffer("atom_mean", torch.tensor(atom_mean, dtype=torch.float32))
        self.register_buffer("atom_std", torch.tensor(atom_std, dtype=torch.float32))
        self.encoder = nn.Sequential(nn.Linear(canonical_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.blocks = nn.ModuleList([UniformAttentionBlock(hidden), UniformAttentionBlock(hidden)])
        self.queries = nn.Parameter(0.02 * torch.randn(query_count, hidden))
        self.pool = nn.MultiheadAttention(hidden, 4, dropout=0.05, batch_first=True)
        summary_dim = query_count * hidden + 2 * canonical_dim + zone_count + 2
        self.trunk = nn.Sequential(
            nn.Linear(summary_dim, 192), nn.LayerNorm(192), nn.GELU(), nn.Dropout(0.05), nn.Linear(192, 96), nn.GELU()
        )
        self.residual_head = nn.Linear(96, zone_count)
        self.tail_head = nn.Sequential(nn.Linear(97, 48), nn.GELU(), nn.Linear(48, 3))

    @staticmethod
    def _pairwise(field: torch.Tensor) -> torch.Tensor:
        indices = torch.triu_indices(field.shape[-1], field.shape[-1], offset=1)
        return field[:, indices[0]] - field[:, indices[1]]

    def forward(self, batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
        atoms = batch["canonical"].to(device)
        mask = batch["mask"].to(device)
        hidden = self.encoder((atoms - self.atom_mean) / self.atom_std) * mask.unsqueeze(-1)
        for block in self.blocks:
            hidden = block(hidden, mask)
        safe_mask = mask.clone()
        empty = ~safe_mask.any(dim=1)
        safe_mask[empty, 0] = True
        queries = self.queries.unsqueeze(0).expand(hidden.shape[0], -1, -1)
        pooled, _ = self.pool(queries, hidden, hidden, key_padding_mask=~safe_mask, need_weights=False)
        count = mask.sum(dim=-1, keepdim=True).clamp_min(1).to(atoms.dtype)
        uniform = mask.to(atoms.dtype) / count
        first = torch.einsum("bn,bnd->bd", uniform, atoms)
        second = torch.einsum("bn,bnd->bd", uniform, torch.square(atoms))
        summary = torch.cat(
            [pooled.flatten(1), first, second, batch["total"].to(device), batch["present"].to(device), batch["analytic"].to(device)], dim=-1
        )
        latent = self.trunk(summary)
        residual = self.residual_head(latent)
        residual = residual - residual.mean(dim=-1, keepdim=True)
        pair = self._pairwise(residual)
        magnitude = pair.abs().mean(dim=-1, keepdim=True)
        return {"tail_logits": self.tail_head(torch.cat([latent, magnitude], dim=-1)), "residual": residual, "pair_residual": pair}


@dataclass(frozen=True)
class PlattMap:
    intercept: float
    coefficient: float

    def probability(self, logits: np.ndarray) -> np.ndarray:
        value = self.intercept + self.coefficient * np.asarray(logits, dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def _fit_platt(logits: np.ndarray, target: np.ndarray) -> PlattMap:
    model = LogisticRegression(C=1.0, solver="lbfgs", random_state=BOOTSTRAP_SEED)
    model.fit(np.asarray(logits)[:, None], target)
    return PlattMap(float(model.intercept_[0]), float(model.coef_[0, 0]))


def _model_factory(method: str, canonical_dim: int, zone_count: int, atom_mean: np.ndarray, atom_std: np.ndarray) -> tuple[nn.Module, bool]:
    if method == "qct_full":
        return qct.QuotientTangentOperator(canonical_dim, zone_count, variant="full", atom_mean=atom_mean, atom_std=atom_std), True
    if method == "qct_no_interactions":
        return qct.QuotientTangentOperator(canonical_dim, zone_count, variant="no_interactions", atom_mean=atom_mean, atom_std=atom_std), True
    if method == "qct_no_tangent":
        return qct.QuotientTangentOperator(canonical_dim, zone_count, variant="no_tangent", atom_mean=atom_mean, atom_std=atom_std), False
    if method == "quotient_deepset":
        return MassDeepSetOperator(canonical_dim, zone_count, atom_mean, atom_std), True
    if method == "uniform_set_transformer":
        return UniformSetTransformerOperator(canonical_dim, zone_count, atom_mean, atom_std), True
    raise ValueError(f"Unknown method: {method}")


def _fit_neural(
    method: str,
    seed: int,
    train: Dataset,
    tuning: Dataset,
    *,
    canonical_dim: int,
    zone_count: int,
    atom_mean: np.ndarray,
    atom_std: np.ndarray,
    tail_thresholds: np.ndarray,
    epochs: int,
    batch_size: int,
    patience: int,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    _seed_everything(seed)
    model, use_tangent = _model_factory(method, canonical_dim, zone_count, atom_mean, atom_std)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=3.0e-3)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed), collate_fn=qct._collate)
    tuning_loader = DataLoader(tuning, batch_size=batch_size, shuffle=False, collate_fn=qct._collate)
    threshold_tensor = torch.tensor(tail_thresholds, dtype=torch.float32, device=device)
    ordinal_weights = torch.tensor([0.20, 0.60, 0.20], dtype=torch.float32, device=device)
    best_ap = -float("inf")
    best_brier = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            magnitude = torch.tensor(batch["magnitude"], dtype=torch.float32, device=device)
            ordinal_target = (magnitude[:, None] > threshold_tensor[None, :]).float()
            output = model(batch, device)
            ordinal = nn.functional.binary_cross_entropy_with_logits(output["tail_logits"], ordinal_target, reduction="none")
            classification_loss = (ordinal * ordinal_weights[None, :]).mean()
            monotonic_loss = torch.relu(output["tail_logits"][:, 1] - output["tail_logits"][:, 0]).mean() + torch.relu(output["tail_logits"][:, 2] - output["tail_logits"][:, 1]).mean()
            if use_tangent:
                zone_loss = nn.functional.smooth_l1_loss(output["residual"], batch["residual"].to(device), beta=0.25)
                pair_loss = nn.functional.smooth_l1_loss(output["pair_residual"], batch["pair_residual"].to(device), beta=0.25)
                tangent_loss = zone_loss + 0.5 * pair_loss
            else:
                tangent_loss = torch.zeros((), dtype=torch.float32, device=device)
            loss = classification_loss + 0.35 * tangent_loss + 0.05 * monotonic_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        logits: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        with torch.no_grad():
            for batch in tuning_loader:
                output = model(batch, device)
                logits.append(output["tail_logits"][:, 1].cpu().numpy())
                targets.append(batch["target"].numpy())
        tuning_logits = np.concatenate(logits)
        tuning_target = np.concatenate(targets)
        probability = 1.0 / (1.0 + np.exp(-np.clip(tuning_logits, -30.0, 30.0)))
        brier = float(brier_score_loss(tuning_target, probability))
        ap = float(average_precision_score(tuning_target, probability))
        history.append({"epoch": float(epoch), "train_loss": float(np.mean(train_losses)), "tuning_brier": brier, "tuning_average_precision": ap})
        if ap > best_ap + 1.0e-5 or (abs(ap - best_ap) <= 1.0e-5 and brier < best_brier - 1.0e-6):
            best_ap, best_brier, best_epoch = ap, brier, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_tuning_brier": best_brier,
        "best_tuning_average_precision": best_ap,
        "use_tangent_auxiliary": use_tangent,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "history": history,
    }


def _predict(model: nn.Module, dataset: Dataset, *, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=qct._collate)
    logits: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    magnitudes: list[np.ndarray] = []
    pair_predictions: list[np.ndarray] = []
    pair_targets: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            output = model(batch, device)
            logits.append(output["tail_logits"][:, 1].cpu().numpy())
            targets.append(batch["target"].numpy())
            magnitudes.append(batch["magnitude"])
            pair_predictions.append(output["pair_residual"].cpu().numpy())
            pair_targets.append(batch["pair_residual"].numpy())
    return np.concatenate(logits), np.concatenate(targets), np.concatenate(magnitudes), np.concatenate(pair_predictions), np.concatenate(pair_targets)


def _logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    return np.log(probability / (1.0 - probability))


def _calibrated_hgb_probability(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    calibration_x: np.ndarray,
    calibration_y: np.ndarray,
    evaluation_x: np.ndarray,
    *,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=220,
        max_leaf_nodes=15,
        min_samples_leaf=50,
        l2_regularization=3.0,
        random_state=random_state,
    ).fit(fit_x, fit_y)
    calibration_raw = model.predict_proba(calibration_x)[:, 1]
    evaluation_raw = model.predict_proba(evaluation_x)[:, 1]
    platt = _fit_platt(_logit(calibration_raw), calibration_y)
    return (
        platt.probability(_logit(calibration_raw)),
        platt.probability(_logit(evaluation_raw)),
        {"platt": platt.__dict__, "model": "HistGradientBoostingClassifier"},
    )


def _fixed_quotient_features(dataset: Dataset, real_indices: np.ndarray) -> np.ndarray:
    values = []
    for index in range(len(dataset)):
        item = dataset[index]
        projected_total = float(np.expm1(float(item["total_scaled"])))
        values.append(quotient_feature(item["canonical"], item["weights"], projected_total, real_indices))
    return np.stack(values)


def _simple_structure_features(dataset: Dataset, target_scale: float) -> np.ndarray:
    values = []
    for index in range(len(dataset)):
        item = dataset[index]
        analytic = np.asarray(item["analytic_scaled"], dtype=np.float64) * target_scale
        pair = qct._numpy_pairwise(analytic[None, :])[0]
        values.append(
            [
                float(len(item["weights"]) > 0),
                float(item["total_scaled"]),
                float(np.mean(np.abs(pair))),
                float(np.max(np.abs(pair), initial=0.0)),
            ]
        )
    return np.asarray(values, dtype=np.float64)


def _calendar_features(dataset: Dataset) -> np.ndarray:
    return np.stack([np.asarray(dataset[index]["calendar"], dtype=np.float64) for index in range(len(dataset))])


def _targets(dataset: Dataset) -> np.ndarray:
    return np.asarray([dataset[index]["tail_target"] for index in range(len(dataset))], dtype=np.float64)


def _magnitudes(dataset: Dataset) -> np.ndarray:
    return np.asarray([dataset[index]["residual_magnitude"] for index in range(len(dataset))], dtype=np.float64)


def _pair_targets(dataset: Dataset) -> np.ndarray:
    return np.stack([np.asarray(dataset[index]["pair_residual_scaled"], dtype=np.float64) for index in range(len(dataset))])


def _ece(target: np.ndarray, probability: np.ndarray, bins: int = 10) -> tuple[float, list[dict[str, Any]]]:
    target = np.asarray(target, dtype=np.float64)
    probability = np.asarray(probability, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    records: list[dict[str, Any]] = []
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (probability >= edges[index]) & (probability <= edges[index + 1])
        else:
            mask = (probability >= edges[index]) & (probability < edges[index + 1])
        count = int(mask.sum())
        if not count:
            records.append({"lower": float(edges[index]), "upper": float(edges[index + 1]), "rows": 0})
            continue
        confidence = float(probability[mask].mean())
        frequency = float(target[mask].mean())
        value += count / len(target) * abs(confidence - frequency)
        records.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "rows": count,
                "mean_probability": confidence,
                "observed_frequency": frequency,
            }
        )
    return float(value), records


def _calibration_diagnostic(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    model = LogisticRegression(C=1.0e6, solver="lbfgs", random_state=BOOTSTRAP_SEED)
    model.fit(_logit(probability)[:, None], target)
    return {"intercept": float(model.intercept_[0]), "slope": float(model.coef_[0, 0])}


def _classification_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    probability = np.clip(np.asarray(probability, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    ece, reliability = _ece(target, probability)
    return {
        "prevalence": float(np.mean(target)),
        "brier": float(brier_score_loss(target, probability)),
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(target, probability)),
        "average_precision": float(average_precision_score(target, probability)),
        "mean_probability": float(np.mean(probability)),
        "expected_calibration_error_10bin": ece,
        "calibration": _calibration_diagnostic(target, probability),
        "reliability_bins": reliability,
    }


def _risk_coverage(target: np.ndarray, probability: np.ndarray, magnitude: np.ndarray) -> dict[str, Any]:
    order = np.argsort(probability, kind="stable")
    total_tails = max(float(target.sum()), 1.0)
    records: dict[str, dict[str, float | int]] = {}
    for coverage in (0.50, 0.70, 0.80, 0.90, 0.95, 1.00):
        count = max(1, int(math.floor(coverage * len(order))))
        kept = order[:count]
        records[f"{coverage:.2f}"] = {
            "rows": int(count),
            "coverage": float(count / len(order)),
            "escalation_rate": float(1.0 - count / len(order)),
            "tail_rate": float(np.mean(target[kept])),
            "tail_false_negative_fraction": float(target[kept].sum() / total_tails),
            "mean_residual_magnitude_eur_mwh": float(np.mean(magnitude[kept])),
            "p95_residual_magnitude_eur_mwh": float(np.quantile(magnitude[kept], 0.95)),
        }
    cumulative_tails = np.cumsum(target[order]) / np.arange(1, len(order) + 1)
    cumulative_magnitude = np.cumsum(magnitude[order]) / np.arange(1, len(order) + 1)
    return {
        "operating_points": records,
        "area_under_tail_risk_coverage": float(np.mean(cumulative_tails)),
        "area_under_mean_magnitude_coverage": float(np.mean(cumulative_magnitude)),
    }


def _paired_brier(target: np.ndarray, baseline: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    difference = np.square(candidate - target) - np.square(baseline - target)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    blocks = int(math.ceil(len(target) / BLOCK_ROWS))
    offsets = np.arange(BLOCK_ROWS)
    for index in range(BOOTSTRAP_REPLICATES):
        starts = rng.integers(0, len(target), size=blocks)
        rows = ((starts[:, None] + offsets[None, :]) % len(target)).ravel()[: len(target)]
        draws[index] = float(np.mean(difference[rows]))
    return {
        "candidate_minus_baseline_brier": float(np.mean(difference)),
        "block_bootstrap_95ci": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "probability_of_improvement": float(np.mean(draws < 0.0)),
        "block_rows": BLOCK_ROWS,
        "replicates": BOOTSTRAP_REPLICATES,
    }


def _decision_metrics(target: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float | int]:
    flagged = probability > threshold
    true_tail = target > 0.5
    return {
        "probability_threshold_from_december": float(threshold),
        "flagged_rows": int(flagged.sum()),
        "flag_rate": float(flagged.mean()),
        "tail_recall": float(np.sum(flagged & true_tail) / max(np.sum(true_tail), 1)),
        "tail_precision": float(np.sum(flagged & true_tail) / max(np.sum(flagged), 1)),
        "accepted_tail_rate": float(np.mean(target[~flagged])) if np.any(~flagged) else float("nan"),
    }


def _pair_residual_metrics(target_scaled: np.ndarray, prediction_scaled: np.ndarray, target_scale: float) -> dict[str, float]:
    return qct._pair_residual_metrics(target_scaled, prediction_scaled, target_scale)


def _ensemble_predict(
    members: list[nn.Module],
    calibrators: list[PlattMap],
    dataset: Dataset,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = []
    tangent = []
    for model, calibrator in zip(members, calibrators, strict=True):
        logits, _, _, pair_prediction, _ = _predict(model, dataset, batch_size=batch_size, device=device)
        probabilities.append(calibrator.probability(logits))
        tangent.append(pair_prediction)
    return np.mean(np.stack(probabilities), axis=0), np.mean(np.stack(tangent), axis=0)


def _dataset_factory(
    samples: list[dict[str, Any]],
    *,
    raw_scale: np.ndarray,
    target_scale: float,
    real_indices: np.ndarray,
    geometry: RealAffineGeometry,
    canonical_ram_scale: float,
    threshold: float,
    presentation: str = "original",
) -> RiskDataset:
    base = RealAffineGaugeDataset(
        samples,
        raw_scale=raw_scale,
        target_scale=target_scale,
        presentation=presentation,
        representation="af",
        real_zone_indices=real_indices,
        geometry=geometry,
        canonical_ram_scale=canonical_ram_scale,
    )
    return RiskDataset(base, target_scale=target_scale, threshold=threshold)


def _verify_frozen_inputs(development_path: Path, confirmation_path: Path, seeds: list[int], smoke: bool) -> tuple[dict[str, Any], pd.Timestamp]:
    artifact_hash = _sha256(development_path)
    engine_hash = _sha256(Path(qct.__file__).resolve())
    if artifact_hash != EXPECTED_DEVELOPMENT_ARTIFACT_SHA256:
        raise RuntimeError(f"Development artifact hash changed: {artifact_hash}")
    if engine_hash != EXPECTED_DEVELOPMENT_ENGINE_SHA256:
        raise RuntimeError(f"Development engine hash changed: {engine_hash}")
    development = json.loads(development_path.read_text(encoding="utf-8"))
    protocol = development["protocol"]
    if protocol["name"] != "quotient_tangent_measure_operator_v3" or protocol["variant"] != "full":
        raise RuntimeError("Development-selected architecture is not the frozen v3 full model")
    if not smoke and tuple(seeds) != REGISTERED_SEEDS:
        raise RuntimeError(f"Confirmation seeds must be {REGISTERED_SEEDS}")
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    if confirmation["protocol"]["study_role"] != "confirmation":
        raise RuntimeError("AF-CDMO reference report is not the confirmation study")
    return development, _as_utc(confirmation["protocol"]["evaluation_end_exclusive"])


def main() -> int:
    args = _parse_args()
    development_path = args.development_artifact.expanduser().resolve()
    confirmation_path = args.confirmation_report.expanduser().resolve()
    development, confirmation_end = _verify_frozen_inputs(
        development_path, confirmation_path, args.seeds, args.smoke
    )
    if not PROTOCOL_NOTE.exists():
        raise RuntimeError("The preregistered confirmation protocol note is missing")

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    if args.smoke:
        args.seeds = args.seeds[:1]
        args.epochs = min(args.epochs, 3)
        args.patience = min(args.patience, 2)

    dual_path = args.dual.expanduser().resolve()
    price_root = args.price_root.expanduser().resolve()
    geometry_audit_path = args.geometry_audit.expanduser().resolve()
    geometry_audit = json.loads(geometry_audit_path.read_text(encoding="utf-8"))
    if not geometry_audit["observed_transfer_queries"]["all_tangent"]:
        raise RuntimeError("Frozen geometry audit did not certify tangent queries")
    if geometry_audit["registered_gauge_rewrite"]["status"] != "passed":
        raise RuntimeError("Frozen geometry audit did not certify gauge rewrites")

    frame, certificate_audit = qct._read_certificate(dual_path, 60.0)
    unit_columns = certificate_audit["unit_columns"]
    input_zones = tuple(column.removeprefix("unit_ptdf_") for column in unit_columns)
    geometry = RealAffineGeometry.loss_safe(input_zones)
    if list(input_zones) != geometry_audit["geometry"]["zone_order"]:
        raise RuntimeError("PTDF order differs from the frozen geometry audit")
    zones = qct.PRIMARY_ZONES
    real_indices = np.asarray([unit_columns.index(f"unit_ptdf_{zone}") for zone in zones], dtype=np.int64)
    price_panel, price_provenance = qct._load_price_panel(price_root, zones)
    samples = qct._build_vector_samples(frame, price_panel, unit_columns, zones, qct.FIT_END, confirmation_end)

    fit_samples = [sample for sample in samples if sample["timestamp"] < qct.FIT_END]
    tuning_samples = [sample for sample in samples if qct.FIT_END <= sample["timestamp"] < qct.TUNING_END]
    calibration_samples = [sample for sample in samples if qct.TUNING_END <= sample["timestamp"] < qct.CALIBRATION_END]
    development_samples = [sample for sample in samples if qct.CALIBRATION_END <= sample["timestamp"] < qct.DEVELOPMENT_END]
    confirmation_samples = [sample for sample in samples if qct.CONFIRMATION_START <= sample["timestamp"] < confirmation_end]
    if min(map(len, (fit_samples, tuning_samples, calibration_samples, development_samples, confirmation_samples))) == 0:
        raise RuntimeError("At least one chronological split is empty")
    if max(pd.Timestamp(sample["timestamp"]) for sample in calibration_samples) >= qct.CONFIRMATION_START:
        raise RuntimeError("Training or calibration crossed the confirmation boundary")
    if min(pd.Timestamp(sample["timestamp"]) for sample in confirmation_samples) < qct.CONFIRMATION_START:
        raise RuntimeError("Confirmation includes a development timestamp")
    if args.smoke:
        fit_samples = fit_samples[-2400:]
        tuning_samples = tuning_samples[-1200:]
        calibration_samples = calibration_samples[-1200:]
        confirmation_samples = confirmation_samples[:800]

    target_values = np.stack([sample["target_vector"] for sample in fit_samples])
    target_scale = max(float(np.median(np.abs(target_values))), 1.0)
    canonical_ram_scale = _canonical_ram_scale(frame, qct.FIT_END)
    raw_scale = _positive_scales(
        raw_rows_for_scale(
            fit_samples,
            representation="af",
            real_zone_indices=real_indices,
            geometry=geometry,
            canonical_ram_scale=canonical_ram_scale,
        )
    )

    def make(split_samples: list[dict[str, Any]], presentation: str = "original", threshold: float = EXPECTED_THRESHOLD) -> RiskDataset:
        return _dataset_factory(
            split_samples,
            raw_scale=raw_scale,
            target_scale=target_scale,
            real_indices=real_indices,
            geometry=geometry,
            canonical_ram_scale=canonical_ram_scale,
            threshold=threshold,
            presentation=presentation,
        )

    fit_base_for_threshold = make(fit_samples, threshold=EXPECTED_THRESHOLD)
    fit_magnitude = _magnitudes(fit_base_for_threshold)
    tail_thresholds = np.quantile(fit_magnitude, [0.75, qct.TAIL_QUANTILE, 0.95])
    threshold = float(tail_thresholds[1])
    expected = float(development["protocol"]["tail_threshold_eur_mwh"])
    if not args.smoke and (
        abs(threshold - EXPECTED_THRESHOLD) > THRESHOLD_REPRODUCTION_ATOL
        or abs(threshold - expected) > THRESHOLD_REPRODUCTION_ATOL
    ):
        raise RuntimeError(f"Frozen q90 threshold did not reproduce: {threshold} versus {expected}")

    fit = make(fit_samples, threshold=threshold)
    tuning = make(tuning_samples, threshold=threshold)
    calibration = make(calibration_samples, threshold=threshold)
    confirmation = make(confirmation_samples, threshold=threshold)
    canonical_dim = fit[0]["canonical"].shape[1]
    atom_mean, atom_std = qct._atom_standardization(fit)
    confirmation_target = _targets(confirmation)
    confirmation_magnitude = _magnitudes(confirmation)
    confirmation_pair_target = _pair_targets(confirmation)

    features: dict[str, dict[str, np.ndarray]] = {}
    for feature_name, builder in (
        ("calendar", _calendar_features),
        ("simple_structure", lambda dataset: _simple_structure_features(dataset, target_scale)),
        ("quotient", lambda dataset: _fixed_quotient_features(dataset, real_indices)),
    ):
        features[feature_name] = {
            "fit": builder(fit),
            "calibration": builder(calibration),
            "confirmation": builder(confirmation),
        }
    fit_target = _targets(fit)
    calibration_target = _targets(calibration)

    control_probabilities: dict[str, np.ndarray] = {}
    control_calibration_probabilities: dict[str, np.ndarray] = {}
    control_details: dict[str, Any] = {}
    fit_prevalence = float(np.mean(fit_target))
    control_probabilities["unconditional"] = np.full(len(confirmation_target), fit_prevalence)
    control_calibration_probabilities["unconditional"] = np.full(len(calibration_target), fit_prevalence)
    control_details["unconditional"] = {"fit_prevalence": fit_prevalence}
    for name in ("calendar", "simple_structure", "quotient"):
        cal_probability, eval_probability, details = _calibrated_hgb_probability(
            features[name]["fit"],
            fit_target,
            features[name]["calibration"],
            calibration_target,
            features[name]["confirmation"],
            random_state=BOOTSTRAP_SEED,
        )
        method_name = f"hgb_{name}"
        control_calibration_probabilities[method_name] = cal_probability
        control_probabilities[method_name] = eval_probability
        control_details[method_name] = details

    fit_residual = np.stack([fit[index]["residual_scaled"] for index in range(len(fit))])
    hgb_residual_columns = []
    for zone_index in range(len(zones)):
        regressor = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.05,
            max_iter=180,
            max_leaf_nodes=15,
            min_samples_leaf=50,
            l2_regularization=3.0,
            random_state=BOOTSTRAP_SEED + zone_index,
        ).fit(features["quotient"]["fit"], fit_residual[:, zone_index])
        hgb_residual_columns.append(regressor.predict(features["quotient"]["confirmation"]))
    hgb_residual = np.column_stack(hgb_residual_columns)
    hgb_residual -= hgb_residual.mean(axis=1, keepdims=True)
    hgb_pair_residual = qct._numpy_pairwise(hgb_residual)

    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output.expanduser().resolve().with_suffix(".progress.json")
    neural_models: dict[str, list[nn.Module]] = {method: [] for method in args.methods}
    neural_calibrators: dict[str, list[PlattMap]] = {method: [] for method in args.methods}
    neural_probabilities: dict[str, np.ndarray] = {}
    neural_calibration_probabilities: dict[str, np.ndarray] = {}
    neural_pair_predictions: dict[str, np.ndarray] = {}
    seed_results: dict[str, dict[str, Any]] = {method: {} for method in args.methods}
    checkpoint_records: list[dict[str, Any]] = []

    for method in args.methods:
        member_probabilities = []
        member_calibration_probabilities = []
        member_pair_predictions = []
        for seed in args.seeds:
            print(f"[QCT-CONFIRM] method={method} seed={seed} device={device}", flush=True)
            checkpoint_path = checkpoint_dir / f"{method}__seed{seed}.pt"
            if args.resume and checkpoint_path.exists():
                payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
                if payload.get("method") != method or int(payload.get("seed", -1)) != seed:
                    raise RuntimeError(f"Checkpoint identity mismatch: {checkpoint_path}")
                if payload.get("development_artifact_sha256") != EXPECTED_DEVELOPMENT_ARTIFACT_SHA256:
                    raise RuntimeError(f"Checkpoint development provenance mismatch: {checkpoint_path}")
                if not np.allclose(payload.get("tail_thresholds_eur_mwh"), tail_thresholds, atol=1.0e-9):
                    raise RuntimeError(f"Checkpoint threshold mismatch: {checkpoint_path}")
                model, _ = _model_factory(method, canonical_dim, len(zones), atom_mean, atom_std)
                model = model.to(device)
                model.load_state_dict(payload["state_dict"])
                training = payload["training"]
                saved_platt = PlattMap(**payload["platt"])
                print(f"[QCT-CONFIRM] resumed {checkpoint_path.name}", flush=True)
            else:
                model, training = _fit_neural(
                    method,
                    seed,
                    fit,
                    tuning,
                    canonical_dim=canonical_dim,
                    zone_count=len(zones),
                    atom_mean=atom_mean,
                    atom_std=atom_std,
                    tail_thresholds=tail_thresholds,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    patience=args.patience,
                    device=device,
                )
                saved_platt = None
            calibration_logits, calibration_check, _, _, _ = _predict(model, calibration, batch_size=args.batch_size, device=device)
            confirmation_logits, target_check, magnitude_check, pair_prediction, pair_target_check = _predict(model, confirmation, batch_size=args.batch_size, device=device)
            if not np.array_equal(calibration_check, calibration_target):
                raise RuntimeError("Calibration target drifted between model paths")
            if not np.array_equal(target_check, confirmation_target) or not np.allclose(magnitude_check, confirmation_magnitude):
                raise RuntimeError("Confirmation target drifted between model paths")
            if not np.allclose(pair_target_check, confirmation_pair_target, atol=1.0e-6):
                raise RuntimeError("Confirmation tangent target drifted between model paths")
            platt = saved_platt or _fit_platt(calibration_logits, calibration_target)
            cal_probability = platt.probability(calibration_logits)
            probability = platt.probability(confirmation_logits)
            member_calibration_probabilities.append(cal_probability)
            member_probabilities.append(probability)
            member_pair_predictions.append(pair_prediction)
            neural_models[method].append(model)
            neural_calibrators[method].append(platt)
            if not (args.resume and checkpoint_path.exists()):
                torch.save(
                    {
                        "method": method,
                        "seed": seed,
                        "state_dict": model.state_dict(),
                        "platt": platt.__dict__,
                        "tail_thresholds_eur_mwh": tail_thresholds,
                        "target_scale": target_scale,
                        "canonical_ram_scale": canonical_ram_scale,
                        "development_artifact_sha256": EXPECTED_DEVELOPMENT_ARTIFACT_SHA256,
                        "training": training,
                    },
                    checkpoint_path,
                )
            checkpoint_records.append({"method": method, "seed": seed, "path": str(checkpoint_path), "sha256": _sha256(checkpoint_path)})
            seed_results[method][str(seed)] = {
                "classification": _classification_metrics(confirmation_target, probability),
                "tangent_residual": _pair_residual_metrics(confirmation_pair_target, pair_prediction, target_scale),
                "training": training,
                "platt": platt.__dict__,
            }
            _atomic_json(
                progress_path,
                {
                    "status": "running",
                    "last_completed": {"method": method, "seed": seed},
                    "completed_checkpoints": checkpoint_records,
                },
            )
        neural_calibration_probabilities[method] = np.mean(np.stack(member_calibration_probabilities), axis=0)
        neural_probabilities[method] = np.mean(np.stack(member_probabilities), axis=0)
        neural_pair_predictions[method] = np.mean(np.stack(member_pair_predictions), axis=0)

    all_probabilities = {**control_probabilities, **neural_probabilities}
    all_calibration_probabilities = {**control_calibration_probabilities, **neural_calibration_probabilities}
    method_results: dict[str, Any] = {}
    for method, probability in all_probabilities.items():
        decision_threshold = float(np.quantile(all_calibration_probabilities[method], 0.80))
        method_results[method] = {
            "classification": _classification_metrics(confirmation_target, probability),
            "risk_coverage": _risk_coverage(confirmation_target, probability, confirmation_magnitude),
            "decision_at_calibration_80pct_coverage": _decision_metrics(confirmation_target, probability, decision_threshold),
        }
    method_results["hgb_quotient"]["tangent_residual"] = _pair_residual_metrics(
        confirmation_pair_target, hgb_pair_residual, target_scale
    )
    for method, prediction in neural_pair_predictions.items():
        method_results[method]["tangent_residual"] = _pair_residual_metrics(
            confirmation_pair_target, prediction, target_scale
        )

    paired: dict[str, Any] = {}
    if "qct_full" in neural_probabilities:
        for baseline in ("unconditional", "hgb_calendar", "hgb_simple_structure", "hgb_quotient", "quotient_deepset", "uniform_set_transformer", "qct_no_interactions", "qct_no_tangent"):
            if baseline in all_probabilities:
                paired[f"{baseline}_to_qct_full"] = _paired_brier(
                    confirmation_target, all_probabilities[baseline], neural_probabilities["qct_full"]
                )

    invariance: dict[str, Any] = {}
    if "qct_full" in neural_models:
        original_probability = neural_probabilities["qct_full"]
        original_tangent = neural_pair_predictions["qct_full"]
        qct_threshold = float(np.quantile(neural_calibration_probabilities["qct_full"], 0.80))
        if args.skip_full_rewrite_sweep or args.smoke:
            registered_presentations = [
                "combined",
                "af_projected_representative",
                "equality_reflection",
                "random_equality_gauge",
                "gauge_internal_hvdc_storebaelt",
            ]
        else:
            registered_presentations = [name for name in presentation_names(geometry) if name != "original"]
        rewrite_records: dict[str, Any] = {}
        for presentation in registered_presentations:
            print(f"[QCT-CONFIRM] rewrite={presentation}", flush=True)
            stressed = make(confirmation_samples, presentation=presentation, threshold=threshold)
            probability, tangent = _ensemble_predict(
                neural_models["qct_full"], neural_calibrators["qct_full"], stressed, batch_size=args.batch_size, device=device
            )
            rewrite_records[presentation] = {
                "max_abs_probability_drift": float(np.max(np.abs(probability - original_probability), initial=0.0)),
                "mean_abs_probability_drift": float(np.mean(np.abs(probability - original_probability))),
                "max_abs_pair_residual_drift_scaled": float(np.max(np.abs(tangent - original_tangent), initial=0.0)),
                "decision_flip_rate": float(np.mean((probability > qct_threshold) != (original_probability > qct_threshold))),
            }
            del stressed
        invariance["registered_rewrites"] = {
            "rows_per_presentation": len(confirmation),
            "presentations": rewrite_records,
            "maximum_probability_drift": float(max(record["max_abs_probability_drift"] for record in rewrite_records.values())),
            "maximum_pair_residual_drift_scaled": float(max(record["max_abs_pair_residual_drift_scaled"] for record in rewrite_records.values())),
            "maximum_decision_flip_rate": float(max(record["decision_flip_rate"] for record in rewrite_records.values())),
        }

        original_base = RealAffineGaugeDataset(
            confirmation_samples,
            raw_scale=raw_scale,
            target_scale=target_scale,
            presentation="original",
            representation="af",
            real_zone_indices=real_indices,
            geometry=geometry,
            canonical_ram_scale=canonical_ram_scale,
        )
        refined = RiskDataset(
            NonuniformRefinementAFDataset(original_base), target_scale=target_scale, threshold=threshold
        )
        refinement_records = {}
        for method in ("qct_full", "quotient_deepset", "uniform_set_transformer"):
            if method not in neural_models:
                continue
            probability, tangent = _ensemble_predict(
                neural_models[method], neural_calibrators[method], refined, batch_size=args.batch_size, device=device
            )
            method_threshold = float(np.quantile(neural_calibration_probabilities[method], 0.80))
            refinement_records[method] = {
                "max_abs_probability_drift": float(np.max(np.abs(probability - neural_probabilities[method]), initial=0.0)),
                "mean_abs_probability_drift": float(np.mean(np.abs(probability - neural_probabilities[method]))),
                "max_abs_pair_residual_drift_scaled": float(np.max(np.abs(tangent - neural_pair_predictions[method]), initial=0.0)),
                "decision_flip_rate": float(np.mean((probability > method_threshold) != (neural_probabilities[method] > method_threshold))),
            }
        invariance["held_out_nonuniform_mass_refinement"] = {
            "rows": len(refined),
            "training_exposure": False,
            "methods": refinement_records,
        }

        ram_records = {}
        for delta in RAM_PERTURBATIONS:
            changed = RiskDataset(NonEquivalentRAMDataset(original_base, delta=delta), target_scale=target_scale, threshold=threshold)
            probability, tangent = _ensemble_predict(
                neural_models["qct_full"], neural_calibrators["qct_full"], changed, batch_size=args.batch_size, device=device
            )
            ram_records[str(delta)] = {
                "mean_abs_probability_change": float(np.mean(np.abs(probability - original_probability))),
                "max_abs_probability_change": float(np.max(np.abs(probability - original_probability), initial=0.0)),
                "mean_abs_pair_residual_change_scaled": float(np.mean(np.abs(tangent - original_tangent))),
                "decision_flip_rate": float(np.mean((probability > qct_threshold) != (original_probability > qct_threshold))),
            }
        invariance["non_equivalent_ram_sensitivity"] = {
            "interpretation": "Changing normalized RAM changes the represented feasible domain; invariance is not expected.",
            "methods": {"qct_full": ram_records},
        }

    primary_controls = [name for name in ("unconditional", "hgb_calendar") if f"{name}_to_qct_full" in paired]
    primary_pass = bool(primary_controls) and all(
        paired[f"{name}_to_qct_full"]["block_bootstrap_95ci"][1] < 0.0 for name in primary_controls
    )
    structural_pass = False
    if "registered_rewrites" in invariance:
        structural_pass = (
            invariance["registered_rewrites"]["maximum_probability_drift"] <= 1.0e-4
            and invariance["registered_rewrites"]["maximum_decision_flip_rate"] == 0.0
        )
    hgb_superiority = None
    if "hgb_quotient_to_qct_full" in paired:
        hgb_superiority = paired["hgb_quotient_to_qct_full"]["block_bootstrap_95ci"][1] < 0.0

    report = {
        "status": "confirmation_complete",
        "claim_boundary": {
            "supported_if_gates_pass": "same-delivery invariant selective assurance for flow-based market-certificate monitoring",
            "not_supported": [
                "future-price forecasting superiority",
                "trading profitability",
                "worldwide priority for generic set or Transformer architectures",
            ],
        },
        "protocol": {
            "name": "quotient_tangent_risk_confirmation_v5",
            "fit_end_exclusive": qct.FIT_END.isoformat(),
            "tuning_end_exclusive": qct.TUNING_END.isoformat(),
            "calibration_end_exclusive": qct.CALIBRATION_END.isoformat(),
            "development_end_exclusive": qct.DEVELOPMENT_END.isoformat(),
            "confirmation_start": qct.CONFIRMATION_START.isoformat(),
            "confirmation_end_exclusive": confirmation_end.isoformat(),
            "tail_thresholds_eur_mwh": [float(value) for value in tail_thresholds],
            "methods": args.methods,
            "seeds": args.seeds,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "device": str(device),
            "smoke": bool(args.smoke),
        },
        "rows": {
            "fit": len(fit),
            "tuning": len(tuning),
            "calibration": len(calibration),
            "development_not_reused": len(development_samples),
            "confirmation": len(confirmation),
        },
        "sources": {
            "dual_archive": str(dual_path),
            "dual_archive_sha256": _sha256(dual_path),
            "price_root": str(price_root),
            "price_provenance": price_provenance,
            "geometry_audit": str(geometry_audit_path),
            "geometry_audit_sha256": _sha256(geometry_audit_path),
            "development_artifact": str(development_path),
            "development_artifact_sha256": _sha256(development_path),
            "development_engine_sha256": _sha256(Path(qct.__file__).resolve()),
            "confirmation_reference": str(confirmation_path),
            "confirmation_reference_sha256": _sha256(confirmation_path),
            "confirmation_engine": str(Path(__file__).resolve()),
            "confirmation_engine_sha256": _sha256(Path(__file__).resolve()),
            "protocol_note": str(PROTOCOL_NOTE),
            "protocol_note_sha256": _sha256(PROTOCOL_NOTE),
        },
        "confirmation_target": {
            "prevalence": float(np.mean(confirmation_target)),
            "mean_residual_magnitude_eur_mwh": float(np.mean(confirmation_magnitude)),
            "p90_residual_magnitude_eur_mwh": float(np.quantile(confirmation_magnitude, 0.90)),
            "p95_residual_magnitude_eur_mwh": float(np.quantile(confirmation_magnitude, 0.95)),
        },
        "control_details": control_details,
        "methods": method_results,
        "paired_brier": paired,
        "seed_results": seed_results,
        "invariance_and_sensitivity": invariance,
        "checkpoints": checkpoint_records,
        "decision": {
            "primary_qct_vs_unconditional_and_calendar_pass": primary_pass,
            "registered_rewrite_contract_pass": structural_pass,
            "qct_significantly_better_than_hgb_quotient": hgb_superiority,
            "include_extension_in_tempr_paper": bool(primary_pass and structural_pass),
            "allow_dl_superiority_claim": bool(primary_pass and structural_pass and hgb_superiority),
        },
    }
    output_path = args.output.expanduser().resolve()
    _atomic_json(output_path, report)
    if progress_path.exists():
        progress_path.unlink()
    print(f"[OK] wrote {output_path}")
    print(json.dumps(report["decision"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
