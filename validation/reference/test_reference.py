"""Reference-tool contract tests only; no mock result is scientific evidence."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from ase.io import read
from ase.calculators.singlepoint import SinglePointCalculator

import compare_reference
from compare_reference import artifact_receipts, compare, read_incar, reference_identity
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
    with pytest.raises(ValueError, match="expected_source"):
        compare(tmp_path / "agent", tmp_path / "reference", PROTOCOL, "same-model")


@pytest.fixture
def comparison_inputs(tmp_path, poscar, monkeypatch):
    """Synthetic comparator contract only; these stubs are not solver evidence."""
    agent = tmp_path / "agent"
    prepare(poscar, agent, "Si", "scf", "baseline", PROTOCOL)
    (agent / "execution.json").write_text(json.dumps({"job_id": "synthetic-agent"}))
    atoms = read(poscar, format="vasp")
    atoms.calc = SinglePointCalculator(atoms, energy=-10.0, forces=np.zeros((len(atoms), 3)))
    original_read = compare_reference.read
    monkeypatch.setattr(compare_reference, "read", lambda path, **kwargs: atoms if Path(path).name == "OUTCAR" else original_read(path, **kwargs))
    monkeypatch.setattr(compare_reference, "audit", lambda *args, **kwargs: {"passed": True, "potcar_sha256": "a" * 64})

    def build(variant):
        inputs = tmp_path / f"{variant}-inputs"
        reference = tmp_path / f"{variant}-outputs"
        prepare(poscar, inputs, "Si", "scf", variant, PROTOCOL)
        reference.mkdir()
        for name in ("INCAR", "KPOINTS", "POSCAR"):
            shutil.copyfile(inputs / name, reference / name)
        (reference / "execution.json").write_text(json.dumps({"job_id": "synthetic-reference"}))
        return agent, reference, inputs / "reference_input.json", poscar
    return build


@pytest.mark.parametrize("variant,mode,status", [
    ("baseline", "same-model", "passed"),
    ("combined", "sensitivity", "measured"),
    ("higher-cutoff", "sensitivity", "measured"),
    ("denser-mesh", "sensitivity", "measured"),
    ("baseline", "sensitivity", "invalid_comparison"),
])
def test_declared_reference_variants_and_unchanged_sensitivity(comparison_inputs, variant, mode, status):
    agent, reference, receipt, source = comparison_inputs(variant)
    result = compare(agent, reference, PROTOCOL, mode, expected_source=source, reference_receipt=receipt)
    assert result["status"] == status
    if mode == "same-model":
        assert result["energy_tolerance_ev_per_atom"] == 0.001
        assert result["force_tolerance_ev_per_angstrom"] == 0.01


@pytest.mark.parametrize("field", ["source_poscar_sha256", "protocol_sha256", "method"])
def test_reference_receipt_binds_source_protocol_and_authorship(comparison_inputs, field):
    agent, reference, receipt, source = comparison_inputs("baseline")
    data = json.loads(receipt.read_text())
    data[field] = "invalid"
    receipt.write_text(json.dumps(data))
    result = compare(agent, reference, PROTOCOL, "same-model", expected_source=source, reference_receipt=receipt)
    assert result["passed"] is False
    assert result["reference_input_identity"]["verified"] is False


@pytest.mark.parametrize("location", ["authored", "executed"])
def test_reference_receipt_binds_authored_and_executed_hashes(comparison_inputs, location):
    agent, reference, receipt, source = comparison_inputs("baseline")
    target = (receipt.parent if location == "authored" else reference) / "INCAR"
    target.write_text(target.read_text() + "\n# changed after freezing\n")
    identity = reference_identity(reference, receipt, source, PROTOCOL, "same-model")
    assert identity["verified"] is False
    assert identity["checks"][f"INCAR_{location}_hash_matches"] is False


def test_declaring_combined_without_changing_settings_is_rejected(comparison_inputs):
    agent, reference, receipt, source = comparison_inputs("baseline")
    data = json.loads(receipt.read_text())
    data.update(variant="combined", encut_ev=520.0, kpoint_grid=[6, 6, 6])
    receipt.write_text(json.dumps(data))
    result = compare(agent, reference, PROTOCOL, "sensitivity", expected_source=source, reference_receipt=receipt)
    assert result["status"] == "invalid_comparison"
    assert result["reference_settings_match_declared_variant"] is False


def test_copied_execution_is_not_a_distinct_reference(comparison_inputs):
    agent, reference, receipt, source = comparison_inputs("baseline")
    shutil.copyfile(agent / "execution.json", reference / "execution.json")
    result = compare(agent, reference, PROTOCOL, "same-model", expected_source=source, reference_receipt=receipt)
    assert result["passed"] is False
    assert result["distinct_solver_jobs"] is False
