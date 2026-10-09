import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter
from pymatgen.io.vasp import Poscar

from vasp_slurm_agent.structures import (
    MAX_ATOMS,
    StructureReviewError,
    apply_structure_plan,
    preview_structure,
)


@pytest.fixture
def silicon(tmp_path):
    structure = Structure(Lattice.cubic(5.43), ["Si", "Si"], [[0, 0, 0], [.25, .25, .25]])
    path = tmp_path / "Si.cif"
    CifWriter(structure, symprec=None).write_file(path)
    return path


def write_poscar(tmp_path, structure, name="input.vasp"):
    path = tmp_path / name
    Poscar(structure, sort_structure=False).write_file(path)
    return path


def coords(preview):
    return np.array([site["fractional_coordinates"] for site in preview["output"]["sites"]])


def test_supercell_preview_is_deterministic_and_preserves_source(silicon):
    original = silicon.read_bytes()
    operations = [{"type": "supercell", "matrix": [3, 1, 1]},
                  {"type": "replace_sites", "indices": [0], "species": "Ge"},
                  {"type": "remove_sites", "indices": [5]}]
    first = preview_structure(silicon, operations)
    second = preview_structure(silicon, operations)
    assert first == second
    assert first["input"]["number_of_sites"] == 2
    assert first["output"]["number_of_sites"] == 5
    assert first["output"]["volume_angstrom3"] == pytest.approx(first["input"]["volume_angstrom3"] * 3)
    assert [site["element"] for site in first["output"]["sites"]] == ["Ge", "Si", "Si", "Si", "Si"]
    assert [site["source_index"] for site in first["output"]["sites"]] == [0, 0, 0, 1, 1]
    assert first["operations"][0]["matrix"] == [[3, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert silicon.read_bytes() == original
    assert list(silicon.parent.iterdir()) == [silicon]


def test_general_supercell_matrix(silicon):
    matrix = [[1, 1, 0], [-1, 1, 0], [0, 0, 1]]
    preview = preview_structure(silicon, [{"type": "supercell", "matrix": matrix}])
    assert preview["output"]["number_of_sites"] == 4
    np.testing.assert_allclose(preview["output"]["lattice_angstrom"],
                               np.array(matrix) @ np.array(preview["input"]["lattice_angstrom"]))


def test_species_replacement_preserves_site_order_and_origin(tmp_path):
    source = Structure(Lattice.cubic(6), ["O", "Ni", "O", "Fe"],
                       [[0, 0, 0], [.25, .25, .25], [.5, .5, .5], [.75, .75, .75]])
    path = tmp_path / "unsorted.cif"
    CifWriter(source, symprec=None).write_file(path)
    preview = preview_structure(path, [{"type": "replace_species", "from": "O", "to": "S"}])
    assert [site["element"] for site in preview["input"]["sites"]] == ["O", "Ni", "O", "Fe"]
    assert [site["element"] for site in preview["output"]["sites"]] == ["S", "Ni", "S", "Fe"]
    assert [site["source_index"] for site in preview["output"]["sites"]] == list(range(4))


def test_cartesian_translation_in_nonorthogonal_cell(tmp_path):
    lattice = [[4, .5, .3], [0, 5, .4], [.2, .1, 6]]
    structure = Structure(lattice, ["Si", "Ge"], [[.9, .1, .2], [.4, .6, .7]])
    path = write_poscar(tmp_path, structure)
    vector = [.6, -.2, .1]
    preview = preview_structure(path, [{"type": "translate_sites", "indices": [0],
                                        "vector": vector, "cartesian": True}])
    expected = np.mod(structure[0].frac_coords + structure.lattice.get_fractional_coords(vector), 1)
    np.testing.assert_allclose(coords(preview)[0], expected)
    np.testing.assert_allclose(coords(preview)[1], structure[1].frac_coords)
    assert preview["output"]["sites"][0]["source_index"] == 0


def test_fractional_translation_and_set_site(silicon):
    preview = preview_structure(silicon, [
        {"type": "translate_sites", "indices": [0], "vector": [-.1, 1.1, .1], "cartesian": False},
        {"type": "set_site", "index": 1, "fractional_coordinates": [1.4, -.4, .6]},
    ])
    np.testing.assert_allclose(coords(preview), [[.9, .1, .1], [.4, .6, .6]])
    assert [site["source_index"] for site in preview["output"]["sites"]] == [0, 1]


@pytest.mark.parametrize("operations", [
    [{"type": "relax"}],
    [{"type": "supercell", "matrix": [1, 0, 1]}],
    [{"type": "supercell", "matrix": [MAX_ATOMS, 1, 1]}],
    [{"type": "supercell", "matrix": [[1, 4096, 0], [0, 1, 4096], [0, 0, 1]]}],
    [{"type": "supercell", "matrix": [True, 1, 1]}],
    [{"type": "supercell", "matrix": [2, 1, 1], "extra": "ignored?"}],
    [{"type": "replace_sites", "indices": [], "species": "Ge"}],
    [{"type": "replace_sites", "indices": [0, 0], "species": "Ge"}],
    [{"type": "replace_sites", "indices": [2], "species": "Ge"}],
    [{"type": "replace_sites", "indices": [False], "species": "Ge"}],
    [{"type": "replace_sites", "indices": [0], "species": "Unknown"}],
    [{"type": "replace_species", "from": "Fe", "to": "Ge"}],
    [{"type": "remove_sites", "indices": [0, 1]}],
    [{"type": "set_site", "index": 1, "fractional_coordinates": [0, 0, 0]}],
    [{"type": "set_site", "index": 1, "fractional_coordinates": [float("nan"), 0, 0]}],
    [{"type": "translate_sites", "indices": [0], "vector": [0, 0, .1], "cartesian": "yes"}],
    [{"type": "slab", "miller_index": [0, 0, 1], "min_slab_size": 8}],
])
def test_invalid_edits_require_review(silicon, operations):
    original = silicon.read_bytes()
    with pytest.raises(StructureReviewError) as error:
        preview_structure(silicon, operations)
    assert error.value.question
    assert silicon.read_bytes() == original


def test_disordered_source_requires_review(tmp_path):
    structure = Structure(Lattice.cubic(5), [{"Si": .5, "Ge": .5}], [[0, 0, 0]])
    source = tmp_path / "mixed.cif"
    CifWriter(structure).write_file(source)
    with pytest.raises(StructureReviewError, match="ordered"):
        preview_structure(source, [])


def test_slab_single_termination_and_explicit_sizes(tmp_path):
    source = write_poscar(tmp_path, Structure(Lattice.cubic(3.6), ["Cu"], [[0, 0, 0]]))
    operation = {"type": "slab", "miller_index": [0, 0, 2], "min_slab_size": 8,
                 "min_vacuum_size": 10, "termination": None}
    preview = preview_structure(source, [operation])
    assert preview["operations"][0]["miller_index"] == [0, 0, 1]
    assert preview["operations"][0]["termination"] == 0
    assert preview["output"]["number_of_sites"] == 3
    assert len(preview["slab_terminations"][0]["choices"]) == 1
    assert any("not a validated surface model" in warning for warning in preview["warnings"])
    assert preview["output"]["lattice_angstrom"][2][2] >= 18


def test_slab_ambiguous_termination_needs_explicit_choice(tmp_path):
    structure = Structure(Lattice.hexagonal(3.25, 5.2), ["Zn", "O", "Zn", "O"],
                          [[1/3, 2/3, 0], [1/3, 2/3, .375], [2/3, 1/3, .5], [2/3, 1/3, .875]])
    source = write_poscar(tmp_path, structure)
    operation = {"type": "slab", "miller_index": [0, 0, 1], "min_slab_size": 8,
                 "min_vacuum_size": 10, "termination": None}
    with pytest.raises(StructureReviewError) as error:
        preview_structure(source, [operation])
    assert len(error.value.choices) == 2
    assert {choice["index"] for choice in error.value.choices} == {0, 1}
    preview = preview_structure(source, [{**operation, "termination": 1}])
    repeated = preview_structure(source, preview["operations"])
    assert preview == repeated
    assert preview["slab_terminations"][0]["selected"] == 1
    with pytest.raises(StructureReviewError, match="available"):
        preview_structure(source, [{**operation, "termination": 2}])


@pytest.mark.parametrize("output_format", ["cif", "poscar", "both"])
def test_export_preserves_geometry_order_source_and_portable_metadata(tmp_path, output_format):
    structure = Structure([[4, .5, .3], [0, 5, .4], [.2, .1, 6]], ["O", "Ni", "O", "Fe"],
                          [[0, 0, 0], [.25, .25, .25], [.3333333429999996, .6666666870000029, .5], [.75, .75, .75]])
    source = write_poscar(tmp_path, structure)
    original = source.read_bytes()
    plan = {**preview_structure(source, []), "output_format": output_format}
    result = apply_structure_plan(source, tmp_path / "export", plan)
    assert set(result["files"]) == {"source", "cif", "poscar", "metadata"}
    assert source.read_bytes() == original
    assert Path(result["files"]["source"]).read_bytes() == original
    poscar = Poscar.from_file(result["files"]["poscar"]).structure
    np.testing.assert_allclose(poscar.lattice.matrix, structure.lattice.matrix, rtol=0, atol=1e-14)
    np.testing.assert_allclose(poscar.frac_coords, structure.frac_coords, rtol=0, atol=1e-14)
    assert [site.specie.symbol for site in poscar] == ["O", "Ni", "O", "Fe"]
    cif_preview = preview_structure(result["files"]["cif"], [])
    assert [site["element"] for site in cif_preview["output"]["sites"]] == ["O", "Ni", "O", "Fe"]
    np.testing.assert_allclose(coords(cif_preview), structure.frac_coords, rtol=0, atol=1e-11)
    metadata_text = Path(result["files"]["metadata"]).read_text()
    metadata = json.loads(metadata_text)
    assert str(tmp_path) not in metadata_text
    assert metadata["output_format"] == output_format
    for key, entry in metadata["files"].items():
        assert Path(entry["name"]).name == entry["name"]
        assert hashlib.sha256(Path(result["files"][key]).read_bytes()).hexdigest() == entry["sha256"]


def test_apply_binds_plan_to_source_and_preview(silicon, tmp_path):
    plan = preview_structure(silicon, [])
    with pytest.raises(StructureReviewError, match="source differs"):
        apply_structure_plan(silicon, tmp_path / "out", {**plan, "source_sha256": "stale"})
    with pytest.raises(StructureReviewError, match="differs from the preview"):
        apply_structure_plan(silicon, tmp_path / "out", {**plan, "operations": [{"type": "supercell", "matrix": [2, 1, 1]}]})
    with pytest.raises(StructureReviewError, match="review"):
        apply_structure_plan(silicon, tmp_path / "out", {**plan, "status": "needs_input"})
    assert not (tmp_path / "out").exists()


def test_apply_rejects_source_and_occupied_output(silicon, tmp_path):
    plan = preview_structure(silicon, [])
    original = silicon.read_bytes()
    with pytest.raises(StructureReviewError, match="separate"):
        apply_structure_plan(silicon, silicon.parent, plan)
    with pytest.raises(StructureReviewError, match="separate"):
        apply_structure_plan(silicon, silicon, plan)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep").write_text("existing result")
    with pytest.raises(StructureReviewError, match="empty"):
        apply_structure_plan(silicon, occupied, plan)
    assert silicon.read_bytes() == original
    assert (occupied / "keep").read_text() == "existing result"
    empty = tmp_path / "empty"
    empty.mkdir()
    result = apply_structure_plan(silicon, empty, plan)
    assert Path(result["files"]["poscar"]).is_file()
