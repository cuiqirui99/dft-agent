"""Reference-tool contract tests only; no mock result is scientific evidence."""
import json
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from compare_reference import artifact_receipts, compare, read_incar
from prepare_reference import prepare


PROTOCOL = Path(__file__).resolve().parents[1] / "protocol.v1.json"


@pytest.fixture
def poscar(tmp_path):
    source = tmp_path / "POSCAR"
    source.write_text("Si fixture\n1\n0 2.7155 2.7155\n2.7155 0 2.7155\n2.7155 2.7155 0\nSi\n2\nDirect\n0 0 0\n0.25 0.25 0.25\n")
    return source


def test_manual_recipe_preserves_geometry_and_uses_protocol(tmp_path, poscar):
    destination = tmp_path / "reference"
    receipt = prepare(poscar, destination, "Si", "scf", "baseline", PROTOCOL)
    original, copied = read(poscar, format="vasp"), read(destination / "POSCAR", format="vasp")
    np.testing.assert_allclose(original.cell.array, copied.cell.array, atol=1e-12)
    np.testing.assert_allclose(original.positions, copied.positions, atol=1e-12)
    settings = json.loads(PROTOCOL.read_text())["parameters"]
    incar = read_incar(destination / "INCAR")
    assert incar["ENCUT"] == settings["encut"]
    assert incar["NSW"] == 0 and incar["IBRION"] == -1 and incar["LCHARG"] is True
    assert receipt["execution_status"] == "not_run"
    assert not (destination / "POTCAR").exists()


def test_displacement_exercises_internal_motion_at_fixed_cell(tmp_path, poscar):
    destination = tmp_path / "perturbed"
    prepare(poscar, destination, "Si", "displaced-relax", "baseline", PROTOCOL)
    original, changed = read(poscar, format="vasp"), read(destination / "POSCAR", format="vasp")
    np.testing.assert_allclose(changed.positions - original.positions, [[0, 0, 0], [0.1, 0, 0]], atol=1e-12)
    np.testing.assert_allclose(changed.cell.array, original.cell.array)
    assert read_incar(destination / "INCAR")["ISIF"] == 2


def test_refuse_material_mismatch_and_executed_directory(tmp_path, poscar):
    destination = tmp_path / "reference"
    with pytest.raises(ValueError, match="elements differ"):
        prepare(poscar, destination, "MgO", "scf", "baseline", PROTOCOL)
    destination.mkdir()
    (destination / "OUTCAR").write_text("existing run")
    with pytest.raises(ValueError, match="executed"):
        prepare(poscar, destination, "Si", "scf", "baseline", PROTOCOL)


def test_missing_receipt_and_self_comparison_cannot_pass(tmp_path):
    assert artifact_receipts(tmp_path)["verified"] is False
    with pytest.raises(ValueError, match="own independently executed reference"):
        compare(tmp_path, tmp_path, PROTOCOL, "same-model")
