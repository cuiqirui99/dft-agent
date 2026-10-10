"""VASP's zero ionic E0 field must not flatten a real relaxation history."""

import csv
import hashlib
import json
import re
from importlib.resources import files
from types import SimpleNamespace

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Vasprun

from vasp_slurm_agent.vasp import _export_plots, _export_tables, _ionic_energy_ev


def parser_energy(step):
    run = object.__new__(Vasprun)
    run.ionic_steps = [step]
    return float(run.final_energy)


@pytest.mark.parametrize("step,expected", [
    ({"e_0_energy": -10.2}, -10.2),
    ({"e_0_energy": 0.0}, 0.0),
    ({"e_0_energy": -10.2, "e_fr_energy": -10.25,
      "electronic_steps": [{"e_0_energy": -10.2, "e_fr_energy": -10.25}]}, -10.2),
    ({"e_0_energy": 0.0, "e_fr_energy": -.1,
      "electronic_steps": [{"e_0_energy": 0.0, "e_fr_energy": -.1}]}, 0.0),
])
def test_valid_ionic_energy_including_true_zero_is_preserved(step, expected):
    assert _ionic_energy_ev(step) == expected
    if step.get("electronic_steps"):
        assert _ionic_energy_ev(step) == parser_energy(step)


def test_vasp5_zero_field_keeps_smearing_correction_and_ionic_offset():
    # Ionic free energy can differ from the last electronic value. Neither F
    # nor the last electronic E0 alone is the requested ionic sigma-to-zero E0.
    step = {"e_0_energy": -0.0, "e_fr_energy": -10.5,
            "electronic_steps": [{"e_0_energy": -10.4, "e_fr_energy": -10.6}]}
    assert _ionic_energy_ev(step) == -10.3
    assert _ionic_energy_ev(step) == parser_energy(step)
    assert _ionic_energy_ev(step) not in (step["e_fr_energy"], step["electronic_steps"][-1]["e_0_energy"])


def test_inconsistent_nonzero_xml_energy_uses_the_same_parser_correction():
    step = {"e_0_energy": -3.0, "e_fr_energy": -5.0,
            "electronic_steps": [{"e_0_energy": -4.8, "e_fr_energy": -5.0}]}
    assert _ionic_energy_ev(step) == parser_energy(step) == -4.8


@pytest.mark.parametrize("step", [
    {}, {"e_fr_energy": -10.0}, {"e_0_energy": None}, {"e_0_energy": float("nan")},
    {"e_0_energy": float("inf")},
    {"e_fr_energy": -10.5, "electronic_steps": [{"e_0_energy": -10.4, "e_fr_energy": -10.6}]},
])
def test_missing_or_nonfinite_ionic_energy_is_not_zero(step):
    assert _ionic_energy_ev(step) is None


def test_unavailable_energy_is_blank_in_table_and_cannot_make_a_curve(tmp_path):
    structure = Structure(Lattice.cubic(4), ["Si"], [[0, 0, 0]])
    run = SimpleNamespace(final_structure=structure, ionic_steps=[{"e_fr_energy": -10.0}])
    _export_tables(run, tmp_path)
    row = next(csv.DictReader((tmp_path / "ionic_steps.csv").open()))
    assert row["energy_ev"] == "" and float(row["free_energy_ev"]) == -10.0
    with pytest.raises(ValueError, match="No finite ionic energy history"):
        _export_plots(run, tmp_path, "relax")
    assert not (tmp_path / "relax_energy.csv").exists()


def test_real_srtio3_relaxation_history_matches_outcar_and_final_energy(tmp_path):
    source = files("vasp_slurm_agent").joinpath("samples", "srtio3-pbe-chain", "01_relax", "outputs")
    xml = source.joinpath("vasprun.xml")
    before = hashlib.sha256(xml.read_bytes()).hexdigest()
    run = Vasprun(str(xml), parse_potcar_file=False, parse_eigen=False, parse_dos=False)
    assert [step["e_0_energy"] for step in run.ionic_steps] == [0.0, 0.0, 0.0]
    expected = [-39.96213511, -39.96623414, -39.99141810]
    outcar = source.joinpath("OUTCAR").read_text()
    observed = [float(value) for value in re.findall(
        r"energy  without entropy=.*?energy\(sigma->0\)\s*=\s*([-+0-9.Ee]+)", outcar)]
    np.testing.assert_allclose(observed, expected, atol=1e-8, rtol=0)
    _export_plots(run, tmp_path, "relax")
    _export_tables(run, tmp_path)
    for name in ("relax_energy.csv", "ionic_steps.csv"):
        with (tmp_path / name).open(newline="") as source_csv:
            energies = [float(row["energy_ev"]) for row in csv.DictReader(source_csv)]
        np.testing.assert_allclose(energies, observed, atol=1e-8, rtol=0)
        assert energies[-1] == float(run.final_energy)
    assert json.loads((tmp_path / "plot_metadata.json").read_text())["energy_definition"] == "sigma_to_zero"
    assert run.converged_electronic and run.converged_ionic
    assert "reached required accuracy" in outcar
    assert hashlib.sha256(xml.read_bytes()).hexdigest() == before
