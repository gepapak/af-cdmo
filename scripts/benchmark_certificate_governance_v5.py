"""Benchmark market-certificate representations and monitoring stability.

The task reconstructs the observed DK1-DK2 day-ahead spread from the published
JAO active-constraint certificate. It compares an ordinary raw-row Deep Set with
canonical constraint-quotient models under row permutation, positive row
rescaling/inverse dual scaling, and duplicate split/merge transformations. The v5
extension adds a capacity-matched augmentation-trained Set Transformer and a
validation-calibrated certificate-price consistency monitor. The chronology is
explicit and configurable so the same frozen benchmark can also test the real
Nordic hourly-to-quarter-hour day-ahead transition without changing the model or
stress definitions.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from torch import nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

try:
    from scripts.audit_dual_measure_conservation import _load_day_ahead
    from scripts.cqdm_utils import (
        expand_native_hourly_certificates,
        paired_block_bootstrap,
    )
except ModuleNotFoundError:
    from audit_dual_measure_conservation import _load_day_ahead
    from cqdm_utils import (
        expand_native_hourly_certificates,
        paired_block_bootstrap,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DUAL = (
    ROOT
    / "official_jao_dual_measure"
    / "NordicBindingDualMeasure_2025_2026_full_history.csv.gz"
)
DEFAULT_MARKET_ROOT = ROOT / "official_post_golive_market"
DEFAULT_OUTPUT = (
    ROOT / "official_jao_dual_measure" / "certificate_formulation_shift_report.json"
)
DEFAULT_FIT_END = pd.Timestamp("2025-11-01", tz="UTC")
DEFAULT_VALIDATION_END = pd.Timestamp("2026-01-01", tz="UTC")
DEFAULT_EVALUATION_END = pd.Timestamp("2026-03-01", tz="UTC")
STRESSES = (
    "original",
    "permutation",
    "row_scaling",
    "duplicate_split",
    "selective_duplicate_split",
    "combined",
)
AUGMENTATION_STRESSES = (
    "original",
    "row_scaling",
    "duplicate_split",
    "selective_duplicate_split",
    "combined",
)
NEURAL_MODES = (
    "raw_deepset",
    "raw_deepset_augmented",
    "raw_set_transformer",
    "raw_set_transformer_augmented",
    "raw_set_transformer_pma",
    "raw_set_transformer_pma_augmented",
    "cqdm_direct",
    "cqdm_conservation_residual",
    "cqdm_mass_attention_residual",
    "cqdm_mass_attention_uniform_weights_residual",
    "cqdm_mass_attention_no_potential",
)
RESIDUAL_MODES = (
    "cqdm_conservation_residual",
    "cqdm_mass_attention_residual",
    "cqdm_mass_attention_uniform_weights_residual",
)
RAW_SCALAR_COLUMNS = ("ram", "flowFb", "fmax", "fref", "fall")
CANONICAL_SCALAR_COLUMNS = (
    "canonical_ram_mw",
    "flowFb",
    "fmax",
    "fref",
    "fall",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _positive_scales(values: np.ndarray) -> np.ndarray:
    scales = np.ones(values.shape[1], dtype=np.float32)
    for index in range(values.shape[1]):
        finite = np.abs(values[:, index])
        finite = finite[np.isfinite(finite) & (finite > 0.0)]
        if len(finite):
            scales[index] = max(float(np.median(finite)), 1e-6)
    return scales


def _invariant_moment_feature(
    canonical: np.ndarray,
    weights: np.ndarray,
    total: float,
    dk1_index: int,
    dk2_index: int,
) -> np.ndarray:
    """Fixed classical summary of the quotient measure, excluding row count."""

    dimension = canonical.shape[1]
    if not len(weights) or total <= 0.0:
        return np.zeros(3 + 2 * dimension, dtype=np.float64)
    normalized = weights.astype(np.float64)
    normalized /= normalized.sum()
    atoms = canonical.astype(np.float64)
    mean = normalized @ atoms
    second_moment = normalized @ np.square(atoms)
    potential = float(total * (mean[dk2_index] - mean[dk1_index]))
    return np.concatenate(
        [
            np.asarray([potential, np.log1p(total), 1.0], dtype=np.float64),
            mean,
            second_moment,
        ]
    )


def _invariant_moment_matrix(
    samples: list[dict[str, Any]], dk1_index: int, dk2_index: int
) -> np.ndarray:
    return np.stack(
        [
            _invariant_moment_feature(
                sample["canonical"],
                sample["weights"],
                sample["total"],
                dk1_index,
                dk2_index,
            )
            for sample in samples
        ],
        axis=0,
    )


def _read_certificate(path: Path, lead_minutes: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    digest = _sha256(path)
    if digest != metadata.get("sha256"):
        raise RuntimeError("Dual-measure SHA-256 mismatch")
    frame = pd.read_csv(path, low_memory=False)
    frame["delivery_utc"] = pd.to_datetime(
        frame["delivery_utc"], utc=True, errors="raise", format="mixed"
    )
    frame["publication_utc"] = pd.to_datetime(
        frame["publication_utc"], utc=True, errors="raise", format="mixed"
    )
    frame = expand_native_hourly_certificates(frame)
    decision = frame["delivery_utc"] - pd.to_timedelta(lead_minutes, unit="m")
    frame = frame.loc[frame["publication_utc"] <= decision].copy()
    unit_columns = sorted(column for column in frame if column.startswith("unit_ptdf_"))
    numeric = list(
        dict.fromkeys(
            [
                "shadowPrice",
                "ptdf_l2_norm",
                "dual_mass_eur_mwh",
                *RAW_SCALAR_COLUMNS,
                *CANONICAL_SCALAR_COLUMNS,
                *unit_columns,
            ]
        )
    )
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(
        subset=["shadowPrice", "ptdf_l2_norm", "dual_mass_eur_mwh", *unit_columns]
    )
    frame = frame.loc[frame["dual_mass_eur_mwh"] > 0.0]
    return frame, {
        "path": str(path),
        "sha256": digest,
        "timely_rows_after_resolution_expansion": int(len(frame)),
        "unit_columns": unit_columns,
    }


def _load_full_day_ahead(root: Path, area: str) -> pd.Series:
    """Compose official hourly spot history with the later quarter-hour DA file."""

    imbalance_path = root / f"ImbalancePrice_{area}_post_golive.csv.gz"
    frame = pd.read_csv(imbalance_path, usecols=["TimeUTC", "SpotPriceEUR"])
    frame["delivery_utc"] = pd.to_datetime(
        frame["TimeUTC"], utc=True, errors="raise", format="mixed"
    )
    hourly = pd.Series(
        pd.to_numeric(frame["SpotPriceEUR"], errors="coerce").to_numpy(),
        index=frame["delivery_utc"],
        name=area,
    )
    hourly = hourly[~hourly.index.duplicated(keep="last")].sort_index()
    quarter_hour = _load_day_ahead(root, area)
    # The dedicated DayAheadPrices dataset is authoritative where available;
    # SpotPriceEUR supplies the same day-ahead market price before that file's
    # quarter-hour coverage begins.
    return quarter_hour.combine_first(hourly).sort_index()


def _build_samples(
    frame: pd.DataFrame,
    market_root: Path,
    unit_columns: list[str],
    fit_end: pd.Timestamp,
    evaluation_end: pd.Timestamp,
) -> list[dict[str, Any]]:
    fit = frame.loc[frame["delivery_utc"] < fit_end]
    canonical_scales: dict[str, float] = {}
    for column in CANONICAL_SCALAR_COLUMNS:
        finite = np.abs(fit[column].to_numpy(dtype=np.float64))
        finite = finite[np.isfinite(finite) & (finite > 0.0)]
        canonical_scales[column] = max(float(np.median(finite)), 1e-6) if len(finite) else 1.0

    groups: dict[pd.Timestamp, dict[str, Any]] = {}
    dk1_index = unit_columns.index("unit_ptdf_DK1")
    dk2_index = unit_columns.index("unit_ptdf_DK2")
    for delivery, group in frame.groupby("delivery_utc", sort=False):
        unit = group[unit_columns].fillna(0.0).to_numpy(dtype=np.float64)
        norm = group["ptdf_l2_norm"].to_numpy(dtype=np.float64)
        shadow = group["shadowPrice"].to_numpy(dtype=np.float64)
        raw_scalars = group[list(RAW_SCALAR_COLUMNS)].fillna(0.0).to_numpy(dtype=np.float64)
        raw = np.column_stack([shadow, unit * norm[:, None], raw_scalars]).astype(np.float32)
        canonical_scalars = np.column_stack(
            [
                np.arcsinh(
                    group[column].fillna(0.0).to_numpy(dtype=np.float64)
                    / canonical_scales[column]
                )
                for column in CANONICAL_SCALAR_COLUMNS
            ]
        )
        canonical = np.column_stack([unit, canonical_scalars]).astype(np.float32)
        mass = group["dual_mass_eur_mwh"].to_numpy(dtype=np.float64)
        total = float(mass.sum())
        weights = (mass / total).astype(np.float32)
        potential = float(np.dot(mass, unit[:, dk2_index] - unit[:, dk1_index]))
        groups[pd.Timestamp(delivery)] = {
            "raw": raw,
            "canonical": canonical,
            "weights": weights,
            "total": total,
            "potential": potential,
        }

    actual = (
        _load_full_day_ahead(market_root, "DK1")
        - _load_full_day_ahead(market_root, "DK2")
    ).dropna()
    start = frame["delivery_utc"].min()
    samples: list[dict[str, Any]] = []
    empty_raw_dim = 1 + len(unit_columns) + len(RAW_SCALAR_COLUMNS)
    empty_canonical_dim = len(unit_columns) + len(CANONICAL_SCALAR_COLUMNS)
    for timestamp, target in actual.items():
        timestamp = pd.Timestamp(timestamp)
        if timestamp < start or timestamp >= evaluation_end:
            continue
        certificate = groups.get(
            timestamp,
            {
                "raw": np.empty((0, empty_raw_dim), dtype=np.float32),
                "canonical": np.empty((0, empty_canonical_dim), dtype=np.float32),
                "weights": np.empty(0, dtype=np.float32),
                "total": 0.0,
                "potential": 0.0,
            },
        )
        samples.append(
            {
                "timestamp": timestamp,
                "target": float(target),
                **certificate,
            }
        )
    return samples


def _stress_certificate(
    raw: np.ndarray,
    canonical: np.ndarray,
    weights: np.ndarray,
    stress: str,
    seed: int,
    row_scaled_columns: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if stress == "original" or len(raw) == 0:
        return raw.copy(), canonical.copy(), weights.copy()
    rng = np.random.default_rng(seed)
    value = raw.copy().astype(np.float64)
    canonical_value = canonical.copy().astype(np.float64)
    weight_value = weights.copy().astype(np.float64)
    if stress in ("row_scaling", "combined"):
        scale = np.exp(rng.uniform(np.log(0.05), np.log(20.0), size=len(value)))
        value[:, 0] /= scale
        if row_scaled_columns is None:
            value[:, 1:] *= scale[:, None]
        else:
            value[:, row_scaled_columns] *= scale[:, None]
    if stress in ("duplicate_split", "combined"):
        alpha = rng.uniform(0.1, 0.9, size=len(value))
        left = value.copy()
        right = value.copy()
        left[:, 0] *= alpha
        right[:, 0] *= 1.0 - alpha
        value = np.concatenate([left, right], axis=0)
        canonical_value = np.concatenate([canonical_value, canonical_value], axis=0)
        weight_value = np.concatenate(
            [weight_value * alpha, weight_value * (1.0 - alpha)], axis=0
        )
    elif stress == "selective_duplicate_split":
        selected = rng.random(len(value)) < 0.5
        if not selected.any():
            selected[rng.integers(0, len(value))] = True
        if len(value) > 1 and selected.all():
            selected[rng.integers(0, len(value))] = False
        alpha = rng.uniform(0.1, 0.9, size=int(selected.sum()))
        left_weight = weight_value.copy()
        left_weight[selected] *= alpha
        right = value[selected].copy()
        right[:, 0] *= 1.0 - alpha
        value[selected, 0] *= alpha
        value = np.concatenate([value, right], axis=0)
        canonical_value = np.concatenate(
            [canonical_value, canonical_value[selected]], axis=0
        )
        weight_value = np.concatenate(
            [left_weight, weight_value[selected] * (1.0 - alpha)], axis=0
        )
    if stress in ("permutation", "combined"):
        order = rng.permutation(len(value))
        value = value[order]
        canonical_value = canonical_value[order]
        weight_value = weight_value[order]
    return (
        value.astype(np.float32),
        canonical_value.astype(np.float32),
        weight_value.astype(np.float32),
    )


def _stress_raw(raw: np.ndarray, stress: str, seed: int) -> np.ndarray:
    """Compatibility helper used by the algebraic stress test."""

    canonical = np.zeros((len(raw), 1), dtype=np.float32)
    weights = np.full(len(raw), 1.0 / max(len(raw), 1), dtype=np.float32)
    return _stress_certificate(raw, canonical, weights, stress, seed)[0]


class CertificateShiftDataset(Dataset):
    def __init__(
        self,
        samples: list[dict[str, Any]],
        raw_scale: np.ndarray,
        target_scale: float,
        stress: str,
        dk1_index: int,
        dk2_index: int,
        ptdf_count: int | None = None,
    ) -> None:
        self.samples = samples
        self.raw_scale = raw_scale
        self.target_scale = target_scale
        self.stress = stress
        self.dk1_index = dk1_index
        self.dk2_index = dk2_index
        if ptdf_count is None:
            ptdf_count = max(len(raw_scale) - 1 - len(RAW_SCALAR_COLUMNS), 0)
        self.row_scaled_columns = np.asarray(
            [*range(1, 1 + ptdf_count), 1 + ptdf_count], dtype=np.int64
        )
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        stress = self.stress
        stress_seed = 20260820 + index
        if stress == "augmentation":
            stress = AUGMENTATION_STRESSES[(index + self.epoch) % len(AUGMENTATION_STRESSES)]
            stress_seed += self.epoch * max(len(self.samples), 1)
        raw, canonical, weights = _stress_certificate(
            sample["raw"],
            sample["canonical"],
            sample["weights"],
            stress,
            stress_seed,
            self.row_scaled_columns,
        )
        raw = np.arcsinh(raw / self.raw_scale).astype(np.float32)
        potential = (
            float(
                sample["total"]
                * np.dot(
                    weights.astype(np.float64),
                    canonical[:, self.dk2_index].astype(np.float64)
                    - canonical[:, self.dk1_index].astype(np.float64),
                )
            )
            if len(weights)
            else 0.0
        )
        return {
            **sample,
            "raw_scaled": raw,
            "canonical": canonical,
            "weights": weights,
            "potential": potential,
            "potential_scaled": np.float32(potential / self.target_scale),
            "target_scaled": np.float32(sample["target"] / self.target_scale),
            "residual_scaled": np.float32(
                (sample["target"] - potential) / self.target_scale
            ),
            "total_scaled": np.float32(np.log1p(sample["total"])),
        }


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    raw_dim = batch[0]["raw_scaled"].shape[1]
    canonical_dim = batch[0]["canonical"].shape[1]
    max_rows = max(
        max(len(sample["weights"]), len(sample["raw_scaled"]), 1) for sample in batch
    )
    raw = np.zeros((len(batch), max_rows, raw_dim), dtype=np.float32)
    canonical = np.zeros((len(batch), max_rows, canonical_dim), dtype=np.float32)
    weights = np.zeros((len(batch), max_rows), dtype=np.float32)
    mask = np.zeros((len(batch), max_rows), dtype=bool)
    raw_mask = np.zeros((len(batch), max_rows), dtype=bool)
    for index, sample in enumerate(batch):
        count = len(sample["weights"])
        raw_count = len(sample["raw_scaled"])
        # Split stresses can contain twice as many raw rows as canonical rows.
        if raw_count > max_rows:
            raise RuntimeError("Collate row capacity does not cover stressed raw rows")
        if raw_count:
            raw[index, :raw_count] = sample["raw_scaled"]
            raw_mask[index, :raw_count] = True
        if count:
            canonical[index, :count] = sample["canonical"]
            weights[index, :count] = sample["weights"]
            mask[index, :count] = True
    return {
        "raw": torch.from_numpy(raw),
        "raw_mask": torch.from_numpy(raw_mask),
        "canonical": torch.from_numpy(canonical),
        "weights": torch.from_numpy(weights),
        "mask": torch.from_numpy(mask),
        "total": torch.tensor([[sample["total_scaled"]] for sample in batch]),
        "present": torch.tensor([[float(len(sample["weights"]) > 0)] for sample in batch]),
        "potential_scaled": torch.tensor(
            [[sample["potential_scaled"]] for sample in batch]
        ),
        "target": torch.tensor([sample["target_scaled"] for sample in batch]),
        "residual": torch.tensor([sample["residual_scaled"] for sample in batch]),
        "target_raw": np.asarray([sample["target"] for sample in batch]),
        "potential": np.asarray([sample["potential"] for sample in batch]),
        "timestamp": [sample["timestamp"] for sample in batch],
    }


class CertificateRegressor(nn.Module):
    def __init__(self, mode: str, raw_dim: int, canonical_dim: int) -> None:
        super().__init__()
        self.mode = mode
        self.raw_set_transformer = mode.startswith("raw_set_transformer")
        self.raw_set_transformer_pma = mode.startswith("raw_set_transformer_pma")
        self.raw_input = mode.startswith("raw_deepset") or self.raw_set_transformer
        self.mass_attention = mode in (
            "cqdm_mass_attention_residual",
            "cqdm_mass_attention_uniform_weights_residual",
            "cqdm_mass_attention_no_potential",
        )
        self.uniform_attention_weights = (
            mode == "cqdm_mass_attention_uniform_weights_residual"
        )
        self.include_potential = mode != "cqdm_mass_attention_no_potential"
        atom_dim = raw_dim if self.raw_input else canonical_dim
        self.encoder = nn.Sequential(
            nn.Linear(atom_dim, 48), nn.GELU(), nn.Linear(48, 32), nn.GELU()
        )
        if self.raw_set_transformer:
            self.set_attention = nn.MultiheadAttention(
                embed_dim=32, num_heads=4, dropout=0.0, batch_first=True
            )
            self.set_attention_norm = nn.LayerNorm(32)
            if self.raw_set_transformer_pma:
                self.pma_seed = nn.Parameter(0.02 * torch.randn(1, 1, 32))
                self.pma_attention = nn.MultiheadAttention(
                    embed_dim=32, num_heads=4, dropout=0.0, batch_first=True
                )
                self.pma_norm = nn.LayerNorm(32)
        if self.mass_attention:
            self.mass_attention_query_count = 4
            self.mass_attention_dim = 32
            self.mass_attention_queries = nn.Parameter(
                0.02 * torch.randn(self.mass_attention_query_count, self.mass_attention_dim)
            )
            self.mass_attention_keys = nn.Linear(32, self.mass_attention_dim)
            self.mass_attention_values = nn.Linear(32, self.mass_attention_dim)
            head_dim = (
                self.mass_attention_query_count * self.mass_attention_dim
                + 2
                + int(self.include_potential)
            )
            self.head = nn.Sequential(
                nn.Linear(head_dim, 64),
                nn.GELU(),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )
        else:
            head_dim = 32 if self.raw_input else 35
            self.head = nn.Sequential(
                nn.Linear(head_dim, 32), nn.GELU(), nn.Linear(32, 1)
            )

    def forward(self, batch: dict[str, Any], device: torch.device) -> torch.Tensor:
        if self.raw_set_transformer:
            atoms = batch["raw"].to(device)
            raw_mask = batch["raw_mask"].to(device)
            embedding = self.encoder(atoms)
            safe_mask = raw_mask.clone()
            empty = ~safe_mask.any(dim=1)
            safe_mask[empty, 0] = True
            attended, _ = self.set_attention(
                embedding,
                embedding,
                embedding,
                key_padding_mask=~safe_mask,
                need_weights=False,
            )
            attended = self.set_attention_norm(attended + embedding)
            if self.raw_set_transformer_pma:
                seed = self.pma_seed.expand(len(atoms), -1, -1)
                pooled, _ = self.pma_attention(
                    seed,
                    attended,
                    attended,
                    key_padding_mask=~safe_mask,
                    need_weights=False,
                )
                hidden = self.pma_norm(pooled + seed).squeeze(1)
                hidden = hidden.masked_fill(empty.unsqueeze(1), 0.0)
            else:
                hidden = (attended * raw_mask.unsqueeze(-1)).sum(dim=1)
        elif self.mode.startswith("raw_deepset"):
            atoms = batch["raw"].to(device)
            # Raw-row Deep Sets is permutation invariant, but sum aggregation is
            # not invariant to duplicate constraint splitting.
            raw_mask = batch["raw_mask"].to(device)
            hidden = (self.encoder(atoms) * raw_mask.unsqueeze(-1)).sum(dim=1)
        elif self.mass_attention:
            atoms = batch["canonical"].to(device)
            weights = batch["weights"].to(device)
            mask = batch["mask"].to(device)
            if self.uniform_attention_weights:
                weights = mask.to(weights.dtype)
                weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
            embedding = self.encoder(atoms)
            keys = self.mass_attention_keys(embedding)
            values = self.mass_attention_values(embedding)
            scores = torch.einsum(
                "qd,bnd->bqn", self.mass_attention_queries, keys
            ) / np.sqrt(self.mass_attention_dim)
            # Multiplying the attention kernel by dual mass makes each query an
            # integral over Q. Splitting one atom's mass across coincident rows
            # leaves both numerator and denominator unchanged.
            masked_scores = scores.masked_fill(~mask.unsqueeze(1), -1.0e9)
            shifted = masked_scores - masked_scores.amax(dim=-1, keepdim=True)
            numerator = (
                torch.exp(shifted)
                * weights.unsqueeze(1)
                * mask.unsqueeze(1)
            )
            attention = numerator / numerator.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            pooled = torch.einsum("bqn,bnd->bqd", attention, values).flatten(1)
            auxiliaries = [
                pooled,
                batch["total"].to(device),
                batch["present"].to(device),
            ]
            if self.include_potential:
                auxiliaries.append(batch["potential_scaled"].to(device))
            hidden = torch.cat(auxiliaries, dim=-1)
        else:
            atoms = batch["canonical"].to(device)
            weights = batch["weights"].to(device)
            mask = batch["mask"].to(device)
            embedding = self.encoder(atoms)
            hidden = (embedding * (weights * mask).unsqueeze(-1)).sum(dim=1)
            hidden = torch.cat(
                [
                    hidden,
                    batch["total"].to(device),
                    batch["present"].to(device),
                    batch["potential_scaled"].to(device),
                ],
                dim=-1,
            )
        return self.head(hidden).squeeze(-1)


def _fit(
    mode: str,
    seed: int,
    train: Dataset,
    validation: Dataset,
    raw_dim: int,
    canonical_dim: int,
    batch_size: int,
    epochs: int,
    device: torch.device,
) -> tuple[CertificateRegressor, dict[str, Any]]:
    _seed_everything(seed)
    model = CertificateRegressor(mode, raw_dim, canonical_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-3)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, generator=generator, collate_fn=_collate
    )
    validation_loader = DataLoader(
        validation, batch_size=batch_size, shuffle=False, collate_fn=_collate
    )
    best = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    stale = 0
    target_key = "residual" if mode in RESIDUAL_MODES else "target"
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
        total = 0.0
        rows = 0
        with torch.no_grad():
            for batch in validation_loader:
                prediction = model(batch, device)
                count = len(batch[target_key])
                total += count * float(
                    F.smooth_l1_loss(prediction, batch[target_key].to(device)).cpu()
                )
                rows += count
        objective = total / rows
        if objective < best - 1e-6:
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


def _predict(
    model: CertificateRegressor,
    dataset: Dataset,
    batch_size: int,
    target_scale: float,
    device: torch.device,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    prediction: list[float] = []
    target: list[float] = []
    timestamps: list[pd.Timestamp] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            value = model(batch, device).cpu().numpy() * target_scale
            if model.mode in RESIDUAL_MODES:
                value = value + batch["potential"]
            prediction.extend(value.tolist())
            target.extend(batch["target_raw"].tolist())
            timestamps.extend(batch["timestamp"])
    return {
        "prediction": np.asarray(prediction, dtype=np.float64),
        "target": np.asarray(target, dtype=np.float64),
        "timestamps": timestamps,
    }


def _market_monitor_metrics(
    target: np.ndarray,
    original_prediction: np.ndarray,
    stressed_predictions: dict[str, np.ndarray],
    threshold_eur_mwh: float,
    standardized_transfer_mwh: float,
) -> dict[str, Any]:
    """Measure whether an equivalent certificate rewrite changes an alarm decision."""

    target = np.asarray(target, dtype=np.float64)
    original_prediction = np.asarray(original_prediction, dtype=np.float64)
    if target.shape != original_prediction.shape:
        raise ValueError("target and original_prediction must have the same shape")
    if threshold_eur_mwh < 0.0:
        raise ValueError("threshold_eur_mwh must be non-negative")
    if standardized_transfer_mwh <= 0.0:
        raise ValueError("standardized_transfer_mwh must be positive")

    original_score = np.abs(target - original_prediction)
    original_alert = original_score > threshold_eur_mwh
    original_clear = ~original_alert
    stress_results: dict[str, Any] = {}
    for stress, prediction in stressed_predictions.items():
        prediction = np.asarray(prediction, dtype=np.float64)
        if prediction.shape != target.shape:
            raise ValueError(f"prediction shape mismatch for stress={stress}")
        score = np.abs(target - prediction)
        alert = score > threshold_eur_mwh
        false_alarm = original_clear & alert
        suppressed_alarm = original_alert & ~alert
        prediction_drift = np.abs(prediction - original_prediction)
        score_drift = np.abs(score - original_score)
        stress_results[stress] = {
            "alert_rate": float(np.mean(alert)),
            "alert_flip_rate": float(np.mean(alert != original_alert)),
            "induced_false_alarm_rate": float(np.mean(false_alarm)),
            "suppressed_alarm_rate": float(np.mean(suppressed_alarm)),
            "conditional_false_alarm_rate": float(
                false_alarm.sum() / max(int(original_clear.sum()), 1)
            ),
            "conditional_suppressed_alarm_rate": float(
                suppressed_alarm.sum() / max(int(original_alert.sum()), 1)
            ),
            "mean_abs_monitoring_score_drift_eur_mwh": float(score_drift.mean()),
            "max_abs_monitoring_score_drift_eur_mwh": float(score_drift.max()),
            "mean_abs_prediction_drift_eur_mwh": float(prediction_drift.mean()),
            "max_abs_prediction_drift_eur_mwh": float(prediction_drift.max()),
            "mean_abs_standardized_transfer_valuation_drift_eur": float(
                standardized_transfer_mwh * prediction_drift.mean()
            ),
            "max_abs_standardized_transfer_valuation_drift_eur": float(
                standardized_transfer_mwh * prediction_drift.max()
            ),
        }
    return {
        "threshold_eur_mwh": float(threshold_eur_mwh),
        "original_alert_rate": float(np.mean(original_alert)),
        "original_alert_count": int(original_alert.sum()),
        "evaluation_rows": int(len(target)),
        "stresses": stress_results,
    }


def _alert_flip_vector(
    target: np.ndarray,
    original_prediction: np.ndarray,
    stressed_prediction: np.ndarray,
    threshold_eur_mwh: float,
) -> np.ndarray:
    original_alert = np.abs(target - original_prediction) > threshold_eur_mwh
    stressed_alert = np.abs(target - stressed_prediction) > threshold_eur_mwh
    return (original_alert != stressed_alert).astype(np.float64)


def _paired_monitoring_bootstrap(
    baseline: np.ndarray, candidate: np.ndarray
) -> dict[str, Any]:
    if float(np.mean(baseline)) <= 0.0:
        return {
            "status": "baseline_has_zero_alert_flips",
            "baseline_mean_alert_flip_rate": 0.0,
            "candidate_mean_alert_flip_rate": float(np.mean(candidate)),
        }
    return paired_block_bootstrap(
        {"alert_flip": baseline}, {"alert_flip": candidate}
    )["alert_flip"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dual", type=Path, default=DEFAULT_DUAL)
    parser.add_argument("--market_root", type=Path, default=DEFAULT_MARKET_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 123])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--monitor_threshold_quantile", type=float, default=0.95)
    parser.add_argument("--standardized_transfer_mwh", type=float, default=100.0)
    parser.add_argument(
        "--study_role",
        choices=("development", "frozen_transition", "confirmation"),
        default="development",
    )
    parser.add_argument("--fit_end", default=DEFAULT_FIT_END.isoformat())
    parser.add_argument(
        "--validation_end", default=DEFAULT_VALIDATION_END.isoformat()
    )
    parser.add_argument(
        "--evaluation_end", default=DEFAULT_EVALUATION_END.isoformat()
    )
    args = parser.parse_args()

    if not 0.0 < args.monitor_threshold_quantile < 1.0:
        raise ValueError("monitor_threshold_quantile must be strictly between 0 and 1")
    if args.standardized_transfer_mwh <= 0.0:
        raise ValueError("standardized_transfer_mwh must be positive")

    fit_end = pd.Timestamp(args.fit_end)
    validation_end = pd.Timestamp(args.validation_end)
    evaluation_end = pd.Timestamp(args.evaluation_end)
    if fit_end.tzinfo is None or validation_end.tzinfo is None or evaluation_end.tzinfo is None:
        raise ValueError("fit/validation/evaluation boundaries must include a timezone")
    fit_end = fit_end.tz_convert("UTC")
    validation_end = validation_end.tz_convert("UTC")
    evaluation_end = evaluation_end.tz_convert("UTC")
    if not fit_end < validation_end < evaluation_end:
        raise ValueError("Require fit_end < validation_end < evaluation_end")

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    frame, audit = _read_certificate(args.dual.resolve(), 60.0)
    samples = _build_samples(
        frame,
        args.market_root.resolve(),
        audit["unit_columns"],
        fit_end,
        evaluation_end,
    )
    train_samples = [sample for sample in samples if sample["timestamp"] < fit_end]
    validation_samples = [
        sample
        for sample in samples
        if fit_end <= sample["timestamp"] < validation_end
    ]
    evaluation_samples = [
        sample
        for sample in samples
        if validation_end <= sample["timestamp"] < evaluation_end
    ]
    if not train_samples or not validation_samples or not evaluation_samples:
        raise RuntimeError(
            "Chronology produced an empty fit, validation, or evaluation split: "
            f"fit={len(train_samples)}, validation={len(validation_samples)}, "
            f"evaluation={len(evaluation_samples)}"
        )
    raw_values = np.concatenate(
        [sample["raw"] for sample in train_samples if len(sample["raw"])], axis=0
    ).astype(np.float64)
    raw_scale = _positive_scales(raw_values)
    train_target = np.asarray([sample["target"] for sample in train_samples])
    target_scale = max(float(np.median(np.abs(train_target))), 1.0)
    dk1_index = audit["unit_columns"].index("unit_ptdf_DK1")
    dk2_index = audit["unit_columns"].index("unit_ptdf_DK2")
    train_dataset = CertificateShiftDataset(
        train_samples,
        raw_scale,
        target_scale,
        "original",
        dk1_index,
        dk2_index,
        ptdf_count=len(audit["unit_columns"]),
    )
    validation_dataset = CertificateShiftDataset(
        validation_samples,
        raw_scale,
        target_scale,
        "original",
        dk1_index,
        dk2_index,
        ptdf_count=len(audit["unit_columns"]),
    )
    augmented_train_dataset = CertificateShiftDataset(
        train_samples,
        raw_scale,
        target_scale,
        "augmentation",
        dk1_index,
        dk2_index,
        ptdf_count=len(audit["unit_columns"]),
    )
    augmented_validation_dataset = ConcatDataset(
        [
            CertificateShiftDataset(
                validation_samples,
                raw_scale,
                target_scale,
                stress,
                dk1_index,
                dk2_index,
                ptdf_count=len(audit["unit_columns"]),
            )
            for stress in AUGMENTATION_STRESSES
        ]
    )
    raw_dim = train_samples[0]["raw"].shape[1]
    canonical_dim = train_samples[0]["canonical"].shape[1]
    modes = NEURAL_MODES
    models: dict[str, list[CertificateRegressor]] = {mode: [] for mode in modes}
    fits: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}
    for mode in modes:
        for seed in args.seeds:
            mode_train = (
                augmented_train_dataset if mode.endswith("_augmented") else train_dataset
            )
            mode_validation = (
                augmented_validation_dataset
                if mode.endswith("_augmented")
                else validation_dataset
            )
            model, fit = _fit(
                mode,
                seed,
                mode_train,
                mode_validation,
                raw_dim,
                canonical_dim,
                args.batch_size,
                args.epochs,
                device,
            )
            models[mode].append(model)
            fits[mode].append(fit)
            print(
                f"[FIT] mode={mode} seed={seed} epoch={fit['best_epoch']} "
                f"validation={fit['validation_objective']:.6f}",
                flush=True,
            )

    validation_predictions: dict[str, np.ndarray] = {}
    validation_targets: np.ndarray | None = None
    for mode in modes:
        members = [
            _predict(
                model,
                validation_dataset,
                args.batch_size,
                target_scale,
                device,
            )
            for model in models[mode]
        ]
        validation_predictions[mode] = np.mean(
            np.stack([member["prediction"] for member in members], axis=0), axis=0
        )
        if validation_targets is None:
            validation_targets = members[0]["target"]
    assert validation_targets is not None

    predictions: dict[str, dict[str, np.ndarray]] = {mode: {} for mode in modes}
    predictions_by_seed: dict[str, dict[str, np.ndarray]] = {
        mode: {} for mode in modes
    }
    targets: np.ndarray | None = None
    timestamps: list[pd.Timestamp] = []
    for stress in STRESSES:
        dataset = CertificateShiftDataset(
            evaluation_samples,
            raw_scale,
            target_scale,
            stress,
            dk1_index,
            dk2_index,
            ptdf_count=len(audit["unit_columns"]),
        )
        for mode in modes:
            members = [
                _predict(model, dataset, args.batch_size, target_scale, device)
                for model in models[mode]
            ]
            member_predictions = np.stack(
                [member["prediction"] for member in members], axis=0
            )
            predictions_by_seed[mode][stress] = member_predictions
            predictions[mode][stress] = np.mean(member_predictions, axis=0)
            if targets is None:
                targets = members[0]["target"]
                timestamps = members[0]["timestamps"]
    assert targets is not None
    analytic = np.asarray([sample["potential"] for sample in evaluation_samples])

    train_target_raw = np.asarray(
        [sample["target"] for sample in train_samples], dtype=np.float64
    )
    train_potential = np.asarray(
        [[sample["potential"]] for sample in train_samples], dtype=np.float64
    )
    validation_potential = np.asarray(
        [[sample["potential"]] for sample in validation_samples], dtype=np.float64
    )
    evaluation_potential = analytic.reshape(-1, 1)
    affine_model = LinearRegression().fit(train_potential, train_target_raw)
    validation_affine_prediction = affine_model.predict(validation_potential)
    affine_prediction = affine_model.predict(evaluation_potential)

    train_moments = _invariant_moment_matrix(
        train_samples, dk1_index, dk2_index
    )
    evaluation_moments = _invariant_moment_matrix(
        evaluation_samples, dk1_index, dk2_index
    )
    validation_moments = _invariant_moment_matrix(
        validation_samples, dk1_index, dk2_index
    )
    moment_model = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=20260820,
    ).fit(train_moments, train_target_raw)
    validation_moment_prediction = moment_model.predict(validation_moments)
    moment_prediction = moment_model.predict(evaluation_moments)

    validation_predictions_all: dict[str, np.ndarray] = {
        "analytic_congestion_potential": validation_potential[:, 0],
        "affine_congestion_calibration": validation_affine_prediction,
        "invariant_moment_hist_gradient_boosting": validation_moment_prediction,
        **validation_predictions,
    }
    evaluation_predictions_all: dict[str, dict[str, np.ndarray]] = {
        "analytic_congestion_potential": {
            stress: analytic for stress in STRESSES
        },
        "affine_congestion_calibration": {
            stress: affine_prediction for stress in STRESSES
        },
        "invariant_moment_hist_gradient_boosting": {
            stress: moment_prediction for stress in STRESSES
        },
        **predictions,
    }

    losses: dict[str, np.ndarray] = {
        "analytic_congestion_potential": np.abs(targets - analytic),
        "affine_congestion_calibration": np.abs(targets - affine_prediction),
        "invariant_moment_hist_gradient_boosting": np.abs(
            targets - moment_prediction
        ),
    }
    stress_losses: dict[str, dict[str, np.ndarray]] = {}
    results: dict[str, Any] = {
        "analytic_congestion_potential": {
            "mae_eur_mwh": float(mean_absolute_error(targets, analytic))
        },
        "affine_congestion_calibration": {
            "mae_eur_mwh": float(
                mean_absolute_error(targets, affine_prediction)
            ),
            "coefficient": float(affine_model.coef_[0]),
            "intercept_eur_mwh": float(affine_model.intercept_),
            "formulation_invariant": True,
        },
        "invariant_moment_hist_gradient_boosting": {
            "mae_eur_mwh": float(
                mean_absolute_error(targets, moment_prediction)
            ),
            "feature_count": int(train_moments.shape[1]),
            "formulation_invariant": True,
            "feature_note": (
                "Conserved potential, log total dual mass, presence, and "
                "normalized first and second canonical-atom moments; row count "
                "is excluded because duplicate splitting changes it."
            ),
        },
    }
    for mode in modes:
        original = predictions[mode]["original"]
        losses[mode] = np.abs(targets - original)
        stress_losses[mode] = {}
        stress_results: dict[str, Any] = {}
        for stress in STRESSES:
            current = predictions[mode][stress]
            stress_losses[mode][stress] = np.abs(targets - current)
            member_predictions = predictions_by_seed[mode][stress]
            original_members = predictions_by_seed[mode]["original"]
            stress_results[stress] = {
                "mae_eur_mwh": float(mean_absolute_error(targets, current)),
                "mean_abs_prediction_drift_eur_mwh": float(np.mean(np.abs(current - original))),
                "max_abs_prediction_drift_eur_mwh": float(np.max(np.abs(current - original))),
                "seed_members": [
                    {
                        "seed": int(seed),
                        "mae_eur_mwh": float(
                            mean_absolute_error(targets, member_prediction)
                        ),
                        "mean_abs_prediction_drift_eur_mwh": float(
                            np.mean(np.abs(member_prediction - original_member))
                        ),
                        "max_abs_prediction_drift_eur_mwh": float(
                            np.max(np.abs(member_prediction - original_member))
                        ),
                    }
                    for seed, member_prediction, original_member in zip(
                        args.seeds, member_predictions, original_members
                    )
                ],
            }
        results[mode] = stress_results
    monitoring_methods: dict[str, Any] = {}
    monitoring_thresholds: dict[str, float] = {}
    for method, stress_predictions in evaluation_predictions_all.items():
        threshold = float(
            np.quantile(
                np.abs(validation_targets - validation_predictions_all[method]),
                args.monitor_threshold_quantile,
            )
        )
        monitoring_thresholds[method] = threshold
        monitoring_methods[method] = _market_monitor_metrics(
            targets,
            stress_predictions["original"],
            stress_predictions,
            threshold,
            args.standardized_transfer_mwh,
        )
    results["market_monitoring"] = {
        "definition": (
            "Raise a certificate-price consistency alert when absolute spread "
            "reconstruction error exceeds the method-specific threshold fitted "
            "only on the internal validation block. Equivalent certificate "
            "rewrites must not change the alert decision."
        ),
        "threshold_quantile": float(args.monitor_threshold_quantile),
        "threshold_calibration_split": "internal_validation_original_formulation",
        "standardized_transfer_mwh": float(args.standardized_transfer_mwh),
        "methods": monitoring_methods,
    }
    monitoring_comparisons: dict[str, Any] = {}
    for baseline_mode in (
        "raw_set_transformer_augmented",
        "raw_set_transformer_pma_augmented",
    ):
        baseline_flips = {
            stress: _alert_flip_vector(
                targets,
                predictions[baseline_mode]["original"],
                predictions[baseline_mode][stress],
                monitoring_thresholds[baseline_mode],
            )
            for stress in STRESSES
        }
        baseline_worst_flip = np.max(
            np.stack(list(baseline_flips.values())), axis=0
        )
        for candidate_mode in ("cqdm_direct", "cqdm_mass_attention_residual"):
            candidate_flips = {
                stress: _alert_flip_vector(
                    targets,
                    predictions[candidate_mode]["original"],
                    predictions[candidate_mode][stress],
                    monitoring_thresholds[candidate_mode],
                )
                for stress in STRESSES
            }
            candidate_worst_flip = np.max(
                np.stack(list(candidate_flips.values())), axis=0
            )
            key = f"{candidate_mode}_minus_{baseline_mode}"
            monitoring_comparisons[key] = {
                "combined_stress": _paired_monitoring_bootstrap(
                    baseline_flips["combined"], candidate_flips["combined"]
                ),
                "worst_registered_stress_per_row": _paired_monitoring_bootstrap(
                    baseline_worst_flip, candidate_worst_flip
                ),
            }
    results["market_monitoring"]["paired_alert_flip_comparisons"] = (
        monitoring_comparisons
    )
    results["paired_original_loss"] = {
        "cqdm_residual_minus_raw": paired_block_bootstrap(
            {"mae": losses["raw_deepset"]},
            {"mae": losses["cqdm_conservation_residual"]},
        )["mae"],
        "cqdm_residual_minus_analytic": paired_block_bootstrap(
            {"mae": losses["analytic_congestion_potential"]},
            {"mae": losses["cqdm_conservation_residual"]},
        )["mae"],
        "cqdm_residual_minus_affine": paired_block_bootstrap(
            {"mae": losses["affine_congestion_calibration"]},
            {"mae": losses["cqdm_conservation_residual"]},
        )["mae"],
        "cqdm_residual_minus_invariant_moment_hgb": paired_block_bootstrap(
            {"mae": losses["invariant_moment_hist_gradient_boosting"]},
            {"mae": losses["cqdm_conservation_residual"]},
        )["mae"],
        "cqdm_mass_attention_minus_analytic": paired_block_bootstrap(
            {"mae": losses["analytic_congestion_potential"]},
            {"mae": losses["cqdm_mass_attention_residual"]},
        )["mae"],
        "cqdm_mass_attention_minus_invariant_moment_hgb": paired_block_bootstrap(
            {"mae": losses["invariant_moment_hist_gradient_boosting"]},
            {"mae": losses["cqdm_mass_attention_residual"]},
        )["mae"],
        "cqdm_direct_minus_raw_set_transformer": paired_block_bootstrap(
            {"mae": losses["raw_set_transformer"]},
            {"mae": losses["cqdm_direct"]},
        )["mae"],
        "cqdm_mass_attention_minus_raw_set_transformer": paired_block_bootstrap(
            {"mae": losses["raw_set_transformer"]},
            {"mae": losses["cqdm_mass_attention_residual"]},
        )["mae"],
        "cqdm_direct_minus_augmented_raw_set_transformer": paired_block_bootstrap(
            {"mae": losses["raw_set_transformer_augmented"]},
            {"mae": losses["cqdm_direct"]},
        )["mae"],
        "cqdm_mass_attention_minus_augmented_raw_set_transformer": paired_block_bootstrap(
            {"mae": losses["raw_set_transformer_augmented"]},
            {"mae": losses["cqdm_mass_attention_residual"]},
        )["mae"],
        "cqdm_direct_minus_augmented_raw_set_transformer_pma": paired_block_bootstrap(
            {"mae": losses["raw_set_transformer_pma_augmented"]},
            {"mae": losses["cqdm_direct"]},
        )["mae"],
        "cqdm_mass_attention_minus_augmented_raw_set_transformer_pma": paired_block_bootstrap(
            {"mae": losses["raw_set_transformer_pma_augmented"]},
            {"mae": losses["cqdm_mass_attention_residual"]},
        )["mae"],
    }
    mechanism_baselines = (
        "cqdm_mass_attention_uniform_weights_residual",
        "cqdm_mass_attention_no_potential",
    )
    full_mode = "cqdm_mass_attention_residual"
    results["mechanism_ablations"] = {}
    for baseline_mode in mechanism_baselines:
        baseline_worst = np.max(
            np.stack(
                [stress_losses[baseline_mode][stress] for stress in STRESSES]
            ),
            axis=0,
        )
        full_worst = np.max(
            np.stack([stress_losses[full_mode][stress] for stress in STRESSES]),
            axis=0,
        )
        results["mechanism_ablations"][f"full_minus_{baseline_mode}"] = {
            "original": paired_block_bootstrap(
                {"mae": stress_losses[baseline_mode]["original"]},
                {"mae": stress_losses[full_mode]["original"]},
            )["mae"],
            "worst_formulation": paired_block_bootstrap(
                {"mae": baseline_worst}, {"mae": full_worst}
            )["mae"],
            "baseline_worst_case_mae_eur_mwh": float(baseline_worst.mean()),
            "full_worst_case_mae_eur_mwh": float(full_worst.mean()),
        }
    raw_worst = np.max(
        np.stack([stress_losses["raw_deepset"][stress] for stress in STRESSES]),
        axis=0,
    )
    cqdm_worst = np.max(
        np.stack([stress_losses["cqdm_direct"][stress] for stress in STRESSES]),
        axis=0,
    )
    results["formulation_robustness"] = {
        "raw_deepset_worst_case_mae_eur_mwh": float(raw_worst.mean()),
        "raw_deepset_augmented_worst_case_mae_eur_mwh": float(
            np.max(
                np.stack(
                    [
                        stress_losses["raw_deepset_augmented"][stress]
                        for stress in STRESSES
                    ]
                ),
                axis=0,
            ).mean()
        ),
        "cqdm_direct_worst_case_mae_eur_mwh": float(cqdm_worst.mean()),
        "cqdm_direct_minus_raw_by_stress": {
            stress: paired_block_bootstrap(
                {"mae": stress_losses["raw_deepset"][stress]},
                {"mae": stress_losses["cqdm_direct"][stress]},
            )["mae"]
            for stress in STRESSES
        },
        "cqdm_direct_minus_raw_worst_case": paired_block_bootstrap(
            {"mae": raw_worst}, {"mae": cqdm_worst}
        )["mae"],
    }
    augmented_worst = np.max(
        np.stack(
            [stress_losses["raw_deepset_augmented"][stress] for stress in STRESSES]
        ),
        axis=0,
    )
    results["formulation_robustness"][
        "cqdm_direct_minus_augmented_raw_by_stress"
    ] = {
        stress: paired_block_bootstrap(
            {"mae": stress_losses["raw_deepset_augmented"][stress]},
            {"mae": stress_losses["cqdm_direct"][stress]},
        )["mae"]
        for stress in STRESSES
    }
    results["formulation_robustness"][
        "cqdm_direct_minus_augmented_raw_worst_case"
    ] = paired_block_bootstrap(
        {"mae": augmented_worst}, {"mae": cqdm_worst}
    )["mae"]
    set_transformer_worst = np.max(
        np.stack(
            [stress_losses["raw_set_transformer"][stress] for stress in STRESSES]
        ),
        axis=0,
    )
    augmented_set_transformer_worst = np.max(
        np.stack(
            [
                stress_losses["raw_set_transformer_augmented"][stress]
                for stress in STRESSES
            ]
        ),
        axis=0,
    )
    augmented_set_transformer_pma_worst = np.max(
        np.stack(
            [
                stress_losses["raw_set_transformer_pma_augmented"][stress]
                for stress in STRESSES
            ]
        ),
        axis=0,
    )
    mass_attention_worst = np.max(
        np.stack(
            [
                stress_losses["cqdm_mass_attention_residual"][stress]
                for stress in STRESSES
            ]
        ),
        axis=0,
    )
    results["formulation_robustness"].update(
        {
            "raw_set_transformer_worst_case_mae_eur_mwh": float(
                set_transformer_worst.mean()
            ),
            "raw_set_transformer_augmented_worst_case_mae_eur_mwh": float(
                augmented_set_transformer_worst.mean()
            ),
            "raw_set_transformer_pma_augmented_worst_case_mae_eur_mwh": float(
                augmented_set_transformer_pma_worst.mean()
            ),
            "cqdm_mass_attention_worst_case_mae_eur_mwh": float(
                mass_attention_worst.mean()
            ),
            "cqdm_direct_minus_raw_set_transformer_worst_case": paired_block_bootstrap(
                {"mae": set_transformer_worst}, {"mae": cqdm_worst}
            )["mae"],
            "cqdm_mass_attention_minus_raw_set_transformer_worst_case": paired_block_bootstrap(
                {"mae": set_transformer_worst}, {"mae": mass_attention_worst}
            )["mae"],
            "cqdm_direct_minus_augmented_raw_set_transformer_worst_case": paired_block_bootstrap(
                {"mae": augmented_set_transformer_worst}, {"mae": cqdm_worst}
            )["mae"],
            "cqdm_mass_attention_minus_augmented_raw_set_transformer_worst_case": paired_block_bootstrap(
                {"mae": augmented_set_transformer_worst},
                {"mae": mass_attention_worst},
            )["mae"],
            "cqdm_direct_minus_augmented_raw_set_transformer_pma_worst_case": paired_block_bootstrap(
                {"mae": augmented_set_transformer_pma_worst}, {"mae": cqdm_worst}
            )["mae"],
            "cqdm_mass_attention_minus_augmented_raw_set_transformer_pma_worst_case": paired_block_bootstrap(
                {"mae": augmented_set_transformer_pma_worst},
                {"mae": mass_attention_worst},
            )["mae"],
        }
    )

    augmented_set_transformer_drift = np.max(
        np.stack(
            [
                np.abs(
                    predictions["raw_set_transformer_augmented"][stress]
                    - predictions["raw_set_transformer_augmented"]["original"]
                )
                for stress in STRESSES
            ]
        ),
        axis=0,
    )
    cqdm_direct_drift = np.max(
        np.stack(
            [
                np.abs(
                    predictions["cqdm_direct"][stress]
                    - predictions["cqdm_direct"]["original"]
                )
                for stress in STRESSES
            ]
        ),
        axis=0,
    )
    augmented_set_transformer_pma_drift = np.max(
        np.stack(
            [
                np.abs(
                    predictions["raw_set_transformer_pma_augmented"][stress]
                    - predictions["raw_set_transformer_pma_augmented"]["original"]
                )
                for stress in STRESSES
            ]
        ),
        axis=0,
    )
    mass_attention_drift = np.max(
        np.stack(
            [
                np.abs(
                    predictions["cqdm_mass_attention_residual"][stress]
                    - predictions["cqdm_mass_attention_residual"]["original"]
                )
                for stress in STRESSES
            ]
        ),
        axis=0,
    )
    results["formulation_robustness"]["paired_worst_prediction_drift"] = {
        "cqdm_direct_minus_augmented_raw_set_transformer": paired_block_bootstrap(
            {"absolute_prediction_drift": augmented_set_transformer_drift},
            {"absolute_prediction_drift": cqdm_direct_drift},
        )["absolute_prediction_drift"],
        "cqdm_mass_attention_minus_augmented_raw_set_transformer": paired_block_bootstrap(
            {"absolute_prediction_drift": augmented_set_transformer_drift},
            {"absolute_prediction_drift": mass_attention_drift},
        )["absolute_prediction_drift"],
        "cqdm_direct_minus_augmented_raw_set_transformer_pma": paired_block_bootstrap(
            {"absolute_prediction_drift": augmented_set_transformer_pma_drift},
            {"absolute_prediction_drift": cqdm_direct_drift},
        )["absolute_prediction_drift"],
        "cqdm_mass_attention_minus_augmented_raw_set_transformer_pma": paired_block_bootstrap(
            {"absolute_prediction_drift": augmented_set_transformer_pma_drift},
            {"absolute_prediction_drift": mass_attention_drift},
        )["absolute_prediction_drift"],
    }

    target_source_files = []
    for area in ("DK1", "DK2"):
        for prefix in ("ImbalancePrice", "DayAheadPrices"):
            path = (args.market_root / f"{prefix}_{area}_post_golive.csv.gz").resolve()
            target_source_files.append(
                {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
            )

    report = {
        "protocol": {
            "purpose": "formulation-invariant reconstruction of a solved market certificate",
            "study_role": args.study_role,
            "extension_design_status": (
                "post-audit robustness extension using the frozen v4 chronology; "
                "the confirmation window is reused and is not represented as an "
                "independent second confirmation sample"
            ),
            "fit_end_exclusive": fit_end.isoformat(),
            "internal_validation_end_exclusive": validation_end.isoformat(),
            "evaluation_end_exclusive": evaluation_end.isoformat(),
            "lead_minutes": 60.0,
            "seeds": args.seeds,
            "stresses": list(STRESSES),
            "target": "observed DK1 minus DK2 day-ahead price spread in EUR/MWh",
            "architecture_version": "quotient_dual_mass_attention_v5_governance_extension",
            "architecture_note": (
                "Full canonical neural arms receive the exact invariant congestion "
                "potential explicitly; residual arms add their predictions to the "
                "same conserved potential. The registered no-potential ablation "
                "omits both operations. Mass-attention arms use learned queries "
                "whose attention kernel is integrated against dual mass, except "
                "for the registered uniform-weight ablation."
            ),
            "matched_augmented_set_transformer": {
                "modes": [
                    "raw_set_transformer_augmented",
                    "raw_set_transformer_pma_augmented",
                ],
                "architecture": (
                    "Each augmented arm is parameter-identical to its ordinary "
                    "counterpart. The PMA pair adds the standard learned pooling "
                    "control absent from the legacy sum-pooled comparator."
                ),
                "training_stresses": list(AUGMENTATION_STRESSES),
                "purpose": (
                    "Tests whether data augmentation alone can recover formulation "
                    "robustness at matched neural capacity."
                ),
            },
            "market_monitoring": {
                "decision": "certificate-price consistency alert",
                "threshold_quantile": float(args.monitor_threshold_quantile),
                "threshold_fit": "internal validation, original formulation only",
                "standardized_transfer_mwh": float(args.standardized_transfer_mwh),
                "primary_stability_outcomes": [
                    "alert flip rate",
                    "formulation-induced false-alarm rate",
                    "formulation-induced suppressed-alarm rate",
                    "standardized transfer valuation drift",
                ],
            },
            "mechanism_ablations": {
                "cqdm_mass_attention_uniform_weights_residual": (
                    "Same residual architecture and conserved potential, but the "
                    "attention integral uses uniform active-row weights instead of "
                    "normalized dual mass."
                ),
                "cqdm_mass_attention_no_potential": (
                    "Same dual-mass attention architecture, trained directly on the "
                    "spread without receiving or adding the conserved potential."
                ),
            },
            "classical_invariant_baselines": [
                "affine calibration of the conserved congestion potential",
                "fixed histogram gradient boosting on invariant quotient moments",
            ],
            "target_sources": [
                "ImbalancePrice SpotPriceEUR for March-September 2025",
                "DayAheadPrices DayAheadPriceEUR where the dedicated quarter-hour file is available",
            ],
            "no_claims": [
                "not a future-price forecast",
                "not an imbalance-tail prediction",
                "not a trading-profit experiment",
                "not an independent second confirmation study",
            ],
        },
        "source": audit,
        "reproducibility": {
            "benchmark_engine": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "target_source_files": target_source_files,
        },
        "rows": {
            "fit": len(train_samples),
            "internal_validation": len(validation_samples),
            "evaluation": len(evaluation_samples),
        },
        "target_scale_eur_mwh": target_scale,
        "fits": fits,
        "results": results,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
