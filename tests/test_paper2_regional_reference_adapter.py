"""Synthetic operational adapter and provenance contracts; no provider data."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts import audit_paper2_regional_reference_adapter_v1 as adapter
from scripts import audit_paper2_regional_reference_learner_v1 as learner
from scripts import audit_paper2_regional_reference_v1 as reference

ZONES = "DK1 DK1_CO DK1_DE DK1_KS DK1_SB DK1_SK DK2 DK2_KO DK2_SB FI FI_EL FI_FS NO1 NO2 NO2_ND NO2_NK NO2_SK NO3 NO4 NO5 SE1 SE2 SE3 SE3_FS SE3_KS SE3_SWL SE4 SE4_BC SE4_NB SE4_SP SE4_SWL".split()


def example():
    normals = np.zeros((2, len(ZONES)))
    normals[0, ZONES.index("DK2")] = 1.
    normals[0, ZONES.index("SE2")] = .4
    normals[1, ZONES.index("FI")] = .7
    normals[1, ZONES.index("SE2")] = -.3
    return dict(normals=normals, ram=np.asarray([100., 150.]), dual=np.asarray([2., 3.]),
                auxiliary=np.asarray([[50., 100., 40., 0.], [30., 80., 35., 1.]]))


def test_complete_marked_measure_changes_under_rank5_and_is_preserved_by_regional_adapter():
    values = example()
    e5, e6, regional = reference.declared_geometries(ZONES)
    rewritten = dict(values, normals=values["normals"] - values["normals"][:, ZONES.index("SE2"), None] * regional)
    original = adapter.compare(values, rewritten, ZONES, e5, e6)
    adapted = adapter.compare(values, rewritten, ZONES, e6, e6)
    assert not original["corresponding_fine_measure_atoms_equal"]
    assert adapted["corresponding_fine_measure_atoms_equal"]
    assert adapted["projected_mass_max_abs_difference"] < 1e-12
    assert adapted["retained_auxiliary_marks_exactly_equal"]


def test_direct_analytic_within_area_queries_already_annihilate_regional_reference_rule():
    values = example()
    e5, e6, regional = reference.declared_geometries(ZONES)
    rewritten = dict(values, normals=values["normals"] + 2. * regional)
    original = adapter.compare(values, rewritten, ZONES, e5, e6)
    assert original["analytic_field_max_abs_difference"] > 1.
    assert original["admissible_pair_count"] == 36
    assert original["analytic_spread_max_abs_difference_eur_mwh"] < 1e-12


def test_projected_constant_positive_dual_rows_fail_closed_before_normalization():
    _, e6, regional = reference.declared_geometries(ZONES)
    values = dict(normals=regional[None], ram=np.asarray([0.]), dual=np.asarray([1.]), auxiliary=np.zeros((1, 4)))
    with pytest.raises(ValueError, match="fail closed"):
        adapter.canonical(values, e6)
    with pytest.raises(ValueError, match="fail closed"):
        learner.learner_input(values, e6, ZONES, 10., 100., np.ones(4))


def test_learner_float32_input_contract_and_explicit_36_pair_emission():
    values = example()
    _, e6, regional = reference.declared_geometries(ZONES)
    rewritten = dict(values, normals=values["normals"] + .5 * regional)
    inputs = [learner.learner_input(item, e6, ZONES, 17., 900., np.asarray([800., 865., 685., 128.]))
              for item in (values, rewritten)]
    assert inputs[0]["canonical"].dtype == np.float32
    assert inputs[0]["weights"].dtype == np.float32
    assert np.allclose(inputs[0]["canonical"], inputs[1]["canonical"], atol=1e-7)
    field = np.arange(10.)
    emitted = learner.emitted_spreads(field, ZONES, e6)
    assert len(emitted) == 36
    assert all("DK1" not in key for key in emitted)
    assert emitted["DK2-FI"] == -1.


def write_archive(path: Path, duplicate=False):
    row = dict(delivery_utc="2025-05-28 03:00:00+00:00", mrId="one", cnecName="synthetic",
        ptdf_l2_norm=1., ram=100., shadowPrice=2., flowFb=50., fmax=100., fref=40., fall=0.)
    row.update({"unit_ptdf_" + zone: float(zone == "DK2") for zone in ZONES})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
        if duplicate:
            writer.writerow(row)


def test_duplicate_archived_identity_is_rejected_instead_of_silently_coalesced(tmp_path):
    path = tmp_path / "rows.csv"
    write_archive(path, duplicate=True)
    with pytest.raises(ValueError, match="unique retained row identities"):
        adapter.certificate(path, "2025-05-28 03:00:00+00:00", ZONES)


@pytest.mark.parametrize("column,value", [("shadowPrice", "nan"), ("ram", "inf"), ("unit_ptdf_DK2", "nan")])
def test_nonfinite_provider_row_fails_before_measure_or_prediction(tmp_path, column, value):
    path = tmp_path / "rows.csv"
    write_archive(path)
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0][column] = value
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="numerical fields must be finite"):
        adapter.certificate(path, "2025-05-28 03:00:00+00:00", ZONES)


@pytest.mark.parametrize("value", ["0", "-1"])
def test_nonpositive_multiplier_cannot_enter_positive_measure_contract(tmp_path, value):
    path = tmp_path / "rows.csv"
    write_archive(path)
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["shadowPrice"] = value
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="strictly positive multipliers"):
        adapter.certificate(path, "2025-05-28 03:00:00+00:00", ZONES)


def test_cli_freeze_binds_authorized_input_and_blocks_mutation_before_computation(tmp_path):
    archive, geometry = tmp_path / "rows.csv", tmp_path / "geometry.json"
    write_archive(archive)
    geometry.write_text(json.dumps({"geometry": {"zone_order": ZONES}}), encoding="utf-8")
    output = tmp_path / "audit"
    command = [sys.executable, str(Path(adapter.__file__)), "--dual-archive", str(archive),
        "--geometry-audit", str(geometry), "--delivery", "2025-05-28 03:00:00+00:00", "--output", str(output)]
    frozen = subprocess.run(command + ["--plan"], capture_output=True, text=True, check=False)
    assert frozen.returncode == 0, frozen.stderr
    assert (output / "protocol.json").is_file()
    archive.write_text(archive.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    changed = subprocess.run(command + ["--run"], capture_output=True, text=True, check=False)
    assert changed.returncode != 0
    assert "Source, input or protocol changed after freeze" in changed.stderr
    assert not (output / "regional_reference_adapter.json").exists()
