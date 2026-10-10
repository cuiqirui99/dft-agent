"""Exports retain calculated numbers and use their recorded units and reference."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.electronic_structure.dos import CompleteDos, Dos

from vasp_slurm_agent import vasp


def test_bands_use_recorded_fermi_and_keep_disconnected_segments(tmp_path, monkeypatch):
    points = [[0, 0, 0], [0, 0, .5], [.5, 0, 0], [.5, .5, 0]]
    metadata = {"scf_fermi_energy_ev": 4.0, "band_path_count": 4,
                "band_path": {"reciprocal_fractional": points, "distance_inv_angstrom": [0, 1, 1, 2],
                              "segments": [[0, 2], [2, 4]], "labels": ["GAMMA", "X", "M", "R_2"]}}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    values = np.asarray([[[i + 3, 1], [i + 5, 0]] for i in range(4)])
    run = SimpleNamespace(efermi=123.0, actual_kpoints=points, eigenvalues={Spin.up: values})
    captured = {}
    original = vasp._save_figure
    def inspect(fig, output, name):
        ax = fig.axes[0]
        captured["lines"] = [(list(line.get_xdata()), list(line.get_ydata())) for line in ax.lines[:4]]
        captured["ticks"] = [label.get_text() for label in ax.get_xticklabels()]
        return original(fig, output, name)
    monkeypatch.setattr(vasp, "_save_figure", inspect)
    artifacts = vasp._export_plots(run, tmp_path, "bands")
    assert captured["lines"] == [([0, 1], [-1, 0]), ([0, 1], [1, 2]), ([1, 2], [1, 2]), ([1, 2], [3, 4])]
    assert captured["ticks"] == ["Γ", "X|M", "R$_{2}$"]
    assert {"bands.csv", "bands.png", "bands.pdf", "bands.svg", "plot_metadata.json"} <= set(artifacts)
    for name in artifacts:
        assert (tmp_path / name).stat().st_size > 100
    rows = list(csv.DictReader((tmp_path / "bands.csv").open()))
    assert len(rows) == 8 and float(rows[0]["energy_minus_fermi_ev"]) == -1.0
    details = json.loads((tmp_path / "plot_metadata.json").read_text())
    assert details["energy_reference_ev"] == 4.0 and details["interpolation"] == "none"
    assert details["segments"] == [[0, 2], [2, 4]]


def test_bands_refuse_invalid_segment_before_plot(tmp_path):
    points = [[0, 0, 0], [0, 0, .5]]
    metadata = {"scf_fermi_energy_ev": 0.0, "band_path_count": 2,
                "band_path": {"reciprocal_fractional": points, "distance_inv_angstrom": [0, 1],
                              "segments": [[0, 3]], "labels": ["GAMMA", "X"]}}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    run = SimpleNamespace(actual_kpoints=points, eigenvalues={Spin.up: np.zeros((2, 2, 2))})
    with pytest.raises(ValueError, match="segment bounds"):
        vasp._export_plots(run, tmp_path, "bands")
    assert not (tmp_path / "bands.png").exists()


def test_projected_dos_exports_actual_elements_orbitals_and_unsigned_spin(tmp_path):
    structure = Structure(Lattice.cubic(4), ["Mg", "O"], [[0, 0, 0], [.5, .5, .5]])
    energies = np.array([-2, 0, 2, 4, 6])
    up, down = np.array([0, 1, 3, 1, 0]), np.array([0, 2, 1, 2, 0])
    total = Dos(2.0, energies, {Spin.up: 4 * up, Spin.down: 4 * down})
    projections = {site: {Orbital.s: {Spin.up: up, Spin.down: down},
                          Orbital.px: {Spin.up: up / 2, Spin.down: down / 2}}
                   for site in structure}
    dos = CompleteDos(structure, total, projections)
    artifacts = vasp._export_plots(SimpleNamespace(complete_dos=dos, efermi=2.0), tmp_path, "dos")
    assert {"dos_elements.csv", "dos_orbitals.csv", "dos_element_orbitals.csv", "dos_elements.pdf", "dos_orbitals.svg"} <= set(artifacts)
    rows = list(csv.DictReader((tmp_path / "dos_elements.csv").open()))
    assert len(rows) == 20
    assert {row["projection"] for row in rows} == {"Mg", "O"}
    assert min(float(row["dos_states_per_ev"]) for row in rows) >= 0
    down_row = next(row for row in rows if row["projection"] == "Mg" and row["spin"] == "-1" and float(row["energy_minus_fermi_ev"]) == -2.0)
    assert float(down_row["dos_states_per_ev"]) == 3.0
    details = json.loads((tmp_path / "plot_metadata.json").read_text())
    assert details["projected_dos_available"] and details["energy_reference_ev"] == 2.0
    assert "PAW sphere" in details["projection_note"]


def test_no_projected_dos_is_reported_without_inventing_data(tmp_path):
    dos = Dos(0.0, np.array([-1, 0, 1]), {Spin.up: np.array([1, 0, 1])})
    artifacts = vasp._export_plots(SimpleNamespace(complete_dos=dos, efermi=0.0), tmp_path, "dos")
    assert "dos_elements.csv" not in artifacts
    assert not json.loads((tmp_path / "plot_metadata.json").read_text())["projected_dos_available"]


def test_numerical_tables_preserve_units_site_order_and_scf_steps(tmp_path):
    structure = Structure(Lattice.cubic(4), ["Mg", "O"], [[0, 0, 0], [.5, .5, .5]])
    step = {"e_0_energy": -12.0, "e_fr_energy": -12.1, "structure": structure,
            "forces": [[.1, 0, 0], [0, -.2, 0]], "stress": [[-10, 1, 0], [1, 20, 0], [0, 0, 30]],
            "electronic_steps": [{"e_fr_energy": -10.0, "e_0_energy": -9.9}, {"e_fr_energy": -12.1, "e_0_energy": -12.0}]}
    artifacts = vasp._export_tables(SimpleNamespace(final_structure=structure, ionic_steps=[step]), tmp_path)
    assert set(artifacts) == {"structure.json", "ionic_steps.csv", "electronic_steps.csv", "forces.csv", "stress.csv"}
    forces = list(csv.DictReader((tmp_path / "forces.csv").open()))
    assert forces[1]["element"] == "O" and float(forces[1]["fy_ev_angstrom"]) == -.2
    stress = list(csv.DictReader((tmp_path / "stress.csv").open()))
    assert stress[0] == {"component": "xx", "stress_kbar": "-10.0"}
    electronic = list(csv.DictReader((tmp_path / "electronic_steps.csv").open()))
    assert electronic[0]["delta_free_energy_ev"] == ""
    assert float(electronic[1]["delta_free_energy_ev"]) == pytest.approx(-2.1)
    crystal = json.loads((tmp_path / "structure.json").read_text())
    assert crystal["sites"][1]["cartesian_angstrom"] == [2.0, 2.0, 2.0]


def test_unavailable_optional_tables_are_not_fabricated(tmp_path):
    structure = Structure(Lattice.cubic(4), ["Mg"], [[0, 0, 0]])
    run = SimpleNamespace(final_structure=structure, ionic_steps=[{"e_0_energy": -1.0, "forces": [[float("nan"), 0, 0]]}])
    artifacts = vasp._export_tables(run, tmp_path)
    assert artifacts == ["structure.json", "ionic_steps.csv"]
    row = next(csv.DictReader((tmp_path / "ionic_steps.csv").open()))
    assert row["max_force_ev_angstrom"] == "" and row["volume_angstrom3"] == ""


def test_failed_reanalysis_removes_previous_vector_and_table_exports(tmp_path):
    for name in ("bands.pdf", "bands.svg", "dos_elements.csv", "plot_metadata.json", "structure.json", "forces.csv", "stress.csv"):
        (tmp_path / name).write_text("stale")
    source = Path(__file__).resolve().parents[1] / "examples/Si.cif"
    result = vasp.analyze_outputs(tmp_path, "bands", source)
    assert not result["success"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["result.json"]


def test_new_band_inputs_use_dense_path_and_write_projections(tmp_path):
    from pymatgen.io.vasp import Incar
    source = Path(__file__).resolve().parents[1] / "examples/Si.cif"
    metadata = vasp.prepare_inputs(source, tmp_path, "bands", {})
    assert metadata["parameters"]["line_density"] == 60
    assert metadata["band_path_count"] > 100
    assert Incar.from_file(tmp_path / "INCAR")["LORBIT"] == 11
    path = metadata["band_path"]
    distance = np.asarray(path["distance_inv_angstrom"])
    for start, stop in path["segments"]:
        # Endpoint rounding can make short segments slightly coarser than 0.025.
        assert np.diff(distance[start:stop]).max() < 0.03


def _combined_inputs(tmp_path):
    bands, dos = tmp_path / "bands", tmp_path / "dos"
    structure = Structure(Lattice.cubic(4), ["Si"], [[0, 0, 0]])
    for folder in (bands, dos):
        folder.mkdir()
        (folder / "metadata.json").write_text(json.dumps({"method_fingerprint": "pbe-fixture"}))
        structure.to(filename=str(folder / "final_structure.vasp"), fmt="poscar")
    (bands / "plot_metadata.json").write_text(json.dumps({"task": "bands", "energy_reference_ev": 4.0,
        "energy_reference_source": "preceding_scf", "kpoint_count": 2, "segments": [[0, 2]],
        "ticks": [{"distance_inv_angstrom": 0, "label": "GAMMA"}, {"distance_inv_angstrom": 1, "label": "X"}]}))
    (dos / "plot_metadata.json").write_text(json.dumps({"task": "dos", "energy_reference_ev": 3.0}))
    (bands / "bands.csv").write_text("kpoint,distance_inv_angstrom,spin,band,energy_minus_fermi_ev\n0,0,1,1,-1\n1,1,1,1,0\n")
    (dos / "dos.csv").write_text("energy_minus_fermi_ev,dos_spin_1\n-1,0\n0,1\n1,0\n")
    return bands, dos


def test_combined_plot_aligns_references_without_changing_numerical_sources(tmp_path, monkeypatch):
    import hashlib
    bands, dos = _combined_inputs(tmp_path)
    source_bytes = (dos / "dos.csv").read_bytes()
    captured = {}
    original = vasp._save_figure
    def inspect(fig, output, name):
        captured["dos_energies"] = list(fig.axes[1].lines[1].get_ydata())
        return original(fig, output, name)
    monkeypatch.setattr(vasp, "_save_figure", inspect)
    artifacts = vasp.export_bands_dos(bands, dos)
    assert captured["dos_energies"] == [-2.0, -1.0, 0.0]
    assert (dos / "dos.csv").read_bytes() == source_bytes
    details = json.loads((bands / "bands_dos_metadata.json").read_text())
    assert details["dos_energy_shift_ev"] == -1.0 and details["energy_reference_ev"] == 4.0
    assert details["source_sha256"]["dos.csv"] == hashlib.sha256(source_bytes).hexdigest()
    assert artifacts == ["bands_dos.png", "bands_dos.pdf", "bands_dos.svg", "bands_dos_metadata.json"]
    assert all((bands / name).stat().st_size > 100 for name in artifacts)


@pytest.mark.parametrize("change", ["method", "structure"])
def test_combined_plot_rejects_incompatible_stages(tmp_path, change):
    bands, dos = _combined_inputs(tmp_path)
    if change == "method":
        (dos / "metadata.json").write_text(json.dumps({"method_fingerprint": "hse-fixture"}))
    else:
        Structure(Lattice.cubic(5), ["Si"], [[0, 0, 0]]).to(filename=str(dos / "final_structure.vasp"), fmt="poscar")
    with pytest.raises(ValueError, match="different"):
        vasp.export_bands_dos(bands, dos)
    assert not (bands / "bands_dos.png").exists()
