from __future__ import annotations

import numpy as np
import pytest

from scripts.evaluate_af_cdmo_physical_ram_sensitivity_v15 import (
    PhysicalPercentageRAMDataset,
)


class _DummyAFDataset:
    representation = "af"
    ptdf_count = 2

    def __init__(self) -> None:
        self.canonical = np.asarray(
            [
                [1.0, 0.0, np.arcsinh(2.0), 4.0],
                [0.0, 1.0, np.arcsinh(-0.5), 5.0],
            ],
            dtype=np.float32,
        )
        self.epoch = 0

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        assert index == 0
        return {"canonical": self.canonical}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def test_percentage_ram_transform_precedes_asinh() -> None:
    base = _DummyAFDataset()
    wrapped = PhysicalPercentageRAMDataset(base, delta=0.20)

    item = wrapped[0]
    transformed = item["canonical"][:, base.ptdf_count]
    expected = np.arcsinh(1.20 * np.asarray([2.0, -0.5]))

    np.testing.assert_allclose(transformed, expected, rtol=0.0, atol=1.0e-6)
    np.testing.assert_allclose(
        base.canonical[:, base.ptdf_count],
        np.arcsinh(np.asarray([2.0, -0.5])),
        rtol=0.0,
        atol=1.0e-6,
    )


def test_percentage_ram_identity_and_validation() -> None:
    base = _DummyAFDataset()
    identity = PhysicalPercentageRAMDataset(base, delta=0.0)[0]["canonical"]
    np.testing.assert_array_equal(identity, base.canonical)

    with pytest.raises(ValueError):
        PhysicalPercentageRAMDataset(base, delta=-1.0)
