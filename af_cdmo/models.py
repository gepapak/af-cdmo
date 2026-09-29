"""Continuous neural maps for AF-CDMO structural experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn


def collate_certificate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    atom_dim = int(np.asarray(batch[0]["atoms"]).shape[1])
    raw_dim = int(np.asarray(batch[0]["raw"]).shape[1])
    output_dim = int(np.asarray(batch[0]["target"]).shape[0])
    max_rows = max(max(len(item["atoms"]), len(item["raw"]), 1) for item in batch)
    atoms = np.zeros((len(batch), max_rows, atom_dim), dtype=np.float32)
    weights = np.zeros((len(batch), max_rows), dtype=np.float32)
    mask = np.zeros((len(batch), max_rows), dtype=bool)
    raw = np.zeros((len(batch), max_rows, raw_dim), dtype=np.float32)
    raw_mask = np.zeros((len(batch), max_rows), dtype=bool)
    for batch_index, item in enumerate(batch):
        atom_count = len(item["atoms"])
        raw_count = len(item["raw"])
        if atom_count:
            atoms[batch_index, :atom_count] = item["atoms"]
            weights[batch_index, :atom_count] = item["weights"]
            mask[batch_index, :atom_count] = True
        if raw_count:
            raw[batch_index, :raw_count] = item["raw"]
            raw_mask[batch_index, :raw_count] = True
    return {
        "atoms": torch.from_numpy(atoms),
        "weights": torch.from_numpy(weights),
        "mask": torch.from_numpy(mask),
        "raw": torch.from_numpy(raw),
        "raw_mask": torch.from_numpy(raw_mask),
        "total": torch.tensor(
            [[np.log1p(float(item["total_mass"]))] for item in batch],
            dtype=torch.float32,
        ),
        "present": torch.tensor(
            [[float(len(item["atoms"]) > 0)] for item in batch],
            dtype=torch.float32,
        ),
        "analytic": torch.tensor(
            np.stack([item["analytic"] for item in batch]), dtype=torch.float32
        ),
        "target": torch.tensor(
            np.stack([item["target"] for item in batch]), dtype=torch.float32
        ),
        "timestamp": [item.get("timestamp") for item in batch],
    }


class _CotangentProjection(nn.Module):
    def __init__(self, tangent_projector: np.ndarray) -> None:
        super().__init__()
        projector = np.asarray(tangent_projector, dtype=np.float32)
        if projector.ndim != 2 or projector.shape[0] != projector.shape[1]:
            raise ValueError("tangent_projector must be square")
        if not np.isfinite(projector).all():
            raise ValueError("tangent_projector must be finite")
        self.register_buffer("tangent_projector", torch.from_numpy(projector))

    @property
    def output_dim(self) -> int:
        return int(self.tangent_projector.shape[0])

    def project(self, values: torch.Tensor) -> torch.Tensor:
        return values @ self.tangent_projector.T


class MassMeasureCotangentNet(_CotangentProjection):
    """Mass-refinement-invariant attention with a feasible-cotangent output."""

    def __init__(
        self,
        atom_dim: int,
        tangent_projector: np.ndarray,
        *,
        query_count: int = 4,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__(tangent_projector)
        if atom_dim < 1 or query_count < 1 or hidden_dim < 4:
            raise ValueError("Invalid AF-CDMO network dimensions")
        self.encoder = nn.Sequential(
            nn.Linear(atom_dim, 48),
            nn.GELU(),
            nn.Linear(48, hidden_dim),
            nn.GELU(),
        )
        self.query_count = int(query_count)
        self.hidden_dim = int(hidden_dim)
        self.queries = nn.Parameter(0.02 * torch.randn(query_count, hidden_dim))
        self.keys = nn.Linear(hidden_dim, hidden_dim)
        self.values = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(query_count * hidden_dim + 2 + self.output_dim, 64),
            nn.GELU(),
            nn.Linear(64, 48),
            nn.GELU(),
            nn.Linear(48, self.output_dim),
        )

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        atoms = batch["atoms"].to(self.tangent_projector.device)
        masses = batch["weights"].to(self.tangent_projector.device)
        mask = batch["mask"].to(self.tangent_projector.device)
        embedded = self.encoder(atoms)
        keys = self.keys(embedded)
        values = self.values(embedded)
        scores = torch.einsum("qd,bnd->bqn", self.queries, keys)
        scores = scores / np.sqrt(self.hidden_dim)
        scores = scores.masked_fill(~mask.unsqueeze(1), -1.0e9)
        shifted = scores - scores.amax(dim=-1, keepdim=True)
        numerator = torch.exp(shifted) * masses.unsqueeze(1) * mask.unsqueeze(1)
        attention = numerator / numerator.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        pooled = torch.einsum("bqn,bnd->bqd", attention, values).flatten(1)
        analytic = batch["analytic"].to(self.tangent_projector.device)
        hidden = torch.cat(
            [
                pooled,
                batch["total"].to(self.tangent_projector.device),
                batch["present"].to(self.tangent_projector.device),
                analytic,
            ],
            dim=-1,
        )
        residual = self.project(self.head(hidden))
        return self.project(analytic + residual)


class ProjectedRawCotangentDeepSet(_CotangentProjection):
    """Projection-only control without the scale/refinement quotient."""

    def __init__(
        self, raw_dim: int, tangent_projector: np.ndarray, *, hidden_dim: int = 32
    ) -> None:
        super().__init__(tangent_projector)
        if raw_dim < 1 or hidden_dim < 4:
            raise ValueError("Invalid projection-only network dimensions")
        self.encoder = nn.Sequential(
            nn.Linear(raw_dim, 48),
            nn.GELU(),
            nn.Linear(48, hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + self.output_dim, 64),
            nn.GELU(),
            nn.Linear(64, self.output_dim),
        )

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        raw = batch["raw"].to(self.tangent_projector.device)
        mask = batch["raw_mask"].to(self.tangent_projector.device)
        encoded = self.encoder(raw) * mask.unsqueeze(-1)
        pooled = encoded.sum(dim=1)
        analytic = batch["analytic"].to(self.tangent_projector.device)
        residual = self.project(self.head(torch.cat([pooled, analytic], dim=-1)))
        return self.project(analytic + residual)
