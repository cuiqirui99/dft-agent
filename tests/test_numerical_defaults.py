"""Geometry-derived grids and declared material types, without running VASP."""

import math

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Incar, Kpoints, Poscar

from vasp_slurm_agent import vasp
from vasp_slurm_agent.numerical_defaults import potcar_choices, vacuum_axes


def prepare(tmp_path, lattice=None, coordinates=None, task="scf", species=None, **parameters):
    structure = Structure(lattice or Lattice.cubic(5), species or ["Si", "Si"],
                          coordinates or [[0, 0, 0], [.25, .25, .25]])
    source = tmp_path / "POSCAR.source"
    Poscar(structure, sort_structure=False).write_file(source)
    destination = tmp_path / "inputs"
    metadata = vasp.prepare_inputs(source, destination, task, parameters)
    return metadata, Incar.from_file(destination / "INCAR"), Kpoints.from_file(destination / "KPOINTS")


def test_spacing_has_physical_two_pi_convention_and_scales_with_cell(tmp_path):
    meta, _, kpoints = prepare(tmp_path, kspacing=.25)
    assert kpoints.kpts == [(6, 6, 6)]
    assert all(2 * math.pi / 5 / count <= .25 for count in meta["parameters"]["mesh"])
    larger = tmp_path / "larger"
    larger.mkdir()
    meta, _, _ = prepare(larger, Lattice.cubic(10), kspacing=.25)
    assert meta["parameters"]["mesh"] == [3, 3, 3]


def test_anisotropic_oblique_grid_uses_reciprocal_vector_lengths(tmp_path):
    lattice = Lattice([[4, 0, 0], [3, 5, 0], [0, 0, 9]])
    meta, _, _ = prepare(tmp_path, lattice, kspacing=.2)
    assert meta["parameters"]["mesh"] == [math.ceil(length / .2) for length in lattice.reciprocal_lattice.abc]
    assert meta["parameters"]["mesh"][0] != math.ceil(2 * math.pi / lattice.a / .2)


@pytest.mark.parametrize("offset", [0, .5, .95])
def test_vacuum_detection_is_periodic_translation_invariant(tmp_path, offset):
    coordinates = [[0, 0, (.45 + offset) % 1], [.5, .5, (.55 + offset) % 1]]
    meta, _, kpoints = prepare(tmp_path, Lattice.orthorhombic(4, 5, 30), coordinates, kspacing=.1)
    assert kpoints.kpts == [(16, 13, 1)]
    assert [item["axis"] for item in meta["numerical_choices"]["vacuum_axes"]] == [2]
    assert meta["numerical_choices"]["vacuum_axes"][0]["empty_plane_gap_angstrom"] == pytest.approx(27)


def test_vacuum_uses_perpendicular_height_and_rotation_not_lattice_length():
    structure = Structure([[4, 0, 0], [1, 5, 0], [24, 0, 8]], ["Si", "Si"], [[0, 0, .45], [.5, .5, .55]])
    assert vacuum_axes(structure) == []
    rotated = structure.copy()
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    rotated.lattice = Lattice(structure.lattice.matrix @ rotation)
    assert vacuum_axes(rotated) == []


def test_long_bulk_axis_is_not_automatically_vacuum(tmp_path):
    coordinates = [[0, 0, index / 10] for index in range(10)]
    meta, _, _ = prepare(tmp_path, Lattice.orthorhombic(4, 5, 30), coordinates,
                         species=["Si"] * 10, kspacing=.1)
    assert meta["parameters"]["mesh"] == [16, 13, 3]
    assert meta["numerical_choices"]["vacuum_axes"] == []


def test_explicit_mesh_overrides_density_and_vacuum(tmp_path):
    meta, _, points = prepare(tmp_path, Lattice.orthorhombic(4, 5, 30),
                              [[0, 0, .45], [.5, .5, .55]], mesh=[3, 4, 5], kspacing=.1)
    assert points.kpts == [(3, 4, 5)]
    assert meta["numerical_choices"]["mesh_mode"] == "explicit"


def test_legacy_grid_alias_overrides_automatic_mesh(tmp_path):
    meta, _, points = prepare(tmp_path, mesh=None, kpoint_grid=[3, 4, 5])
    assert points.kpts == [(3, 4, 5)]
    assert meta["parameters"]["mesh"] == [3, 4, 5]


@pytest.mark.parametrize("task,kind,expected,sigma", [
    ("relax", "metal", 1, .2), ("scf", "metal", 1, .2),
    ("scf", "auto", 0, .05), ("relax", "insulator", 0, .05),
    ("dos", "insulator", -5, .05), ("dos", "auto", 0, .05),
    ("bands", "metal", 0, .05),
])
def test_smearing_uses_declared_type_and_task(tmp_path, task, kind, expected, sigma):
    meta, incar, _ = prepare(tmp_path, task=task, electronic_type=kind, mesh=[4, 4, 4], line_density=1)
    assert incar["ISMEAR"] == expected and incar["SIGMA"] == sigma
    assert meta["parameters"]["ismear"] == expected


def test_composition_does_not_guess_metallicity(tmp_path):
    meta, incar, _ = prepare(tmp_path, species=["Al", "Al"])
    assert incar["ISMEAR"] == 0
    assert meta["numerical_choices"]["electronic_type"] == "auto"


@pytest.mark.parametrize("vacuum", [True, False])
def test_insulator_dos_falls_back_to_gaussian_for_non_3d_mesh(tmp_path, vacuum):
    lattice = Lattice.orthorhombic(4, 5, 30) if vacuum else Lattice.cubic(5)
    meta, incar, _ = prepare(tmp_path, lattice, task="dos", electronic_type="insulator", mesh=[4, 4, 1])
    assert incar["ISMEAR"] == 0
    assert any("tetrahedra" in note for note in meta["numerical_choices"]["notes"])


def test_explicit_smearing_and_sigma_are_preserved(tmp_path):
    meta, incar, _ = prepare(tmp_path, task="relax", electronic_type="metal", ismear=-1, sigma=.08)
    assert incar["ISMEAR"] == -1 and incar["SIGMA"] == .08
    assert meta["numerical_choices"]["smearing_mode"] == "explicit"


@pytest.mark.parametrize("parameters", [
    {"kspacing": 0}, {"kspacing": float("nan")}, {"kspacing": True},
    {"sigma": 0}, {"ismear": True}, {"electronic_type": "semimetal"},
    {"ismear": -5, "mesh": [4, 4, 1]},
])
def test_invalid_numerical_settings_fail_before_writing_inputs(tmp_path, parameters):
    with pytest.raises(ValueError):
        prepare(tmp_path, **parameters)
    assert not (tmp_path / "inputs").exists()


def test_recommended_potentials_and_explicit_overrides_do_not_fix_f_valence():
    labels, source = potcar_choices(["Ti", "Fe", "Ni", "W", "Sm", "Si"], {"Fe": "Fe", "Sm": "Sm_3"})
    assert labels == ["Ti_pv", "Fe", "Ni_pv", "W_sv", "Sm_3", "Si"]
    assert potcar_choices(["Sm"])[0] == ["Sm"]
    assert source["overrides"] == {"Fe": "Fe", "Sm": "Sm_3"}


@pytest.mark.parametrize("task,functional,soc,isym", [
    ("relax", "PBE", False, 2), ("scf", "PBE", False, 2),
    ("scf", "HSE06", False, 3), ("bands", "PBE", False, 0),
    ("bands", "HSE06", False, -1), ("scf", "PBE", True, -1),
])
def test_symmetry_respects_method_and_path(tmp_path, task, functional, soc, isym):
    _, incar, _ = prepare(tmp_path, task=task, functional=functional, soc=soc, mesh=[2, 2, 2], line_density=1)
    assert incar["ISYM"] == isym
