"""The crystal view must not change calculation inputs or distort geometry."""

import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Poscar

from vasp_slurm_agent.structure_view import (
    cell_edges, display_repeat, display_structure, structure_figure, viewer_html,
)


def silicon():
    return Structure.from_file(Path(__file__).parents[1] / "examples" / "Si.cif")


def test_display_is_exact_periodic_copy_and_does_not_mutate_input():
    structure = silicon()
    original = structure.as_dict()
    shown = display_structure(structure)
    assert structure.as_dict() == original
    assert len(structure) == 2 and len(shown) == 16
    assert shown.volume == 8 * structure.volume
    # Every shown atom maps back to a real input site under lattice translation.
    wrapped = np.mod(structure.lattice.get_fractional_coords(shown.cart_coords), 1)
    for coord in wrapped:
        distances = np.abs((structure.frac_coords - coord + 0.5) % 1 - 0.5)
        assert np.any(np.all(distances < 1e-8, axis=1))
    assert shown.composition.reduced_formula == structure.composition.reduced_formula


def test_slab_is_not_repeated_through_vacuum():
    slab = Structure(Lattice.orthorhombic(3, 3, 22), ["C", "C"], [[0, 0, 0.49], [0.5, 0.5, 0.51]])
    assert display_repeat(slab) == (2, 2, 1)
    shown = display_structure(slab)
    assert len(shown) == 8
    assert shown.lattice.c == slab.lattice.c


def test_large_cell_is_not_automatically_multiplied():
    structure = silicon()
    structure.make_supercell([3, 3, 3])
    assert display_repeat(structure) == (1, 1, 1)
    assert len(display_structure(structure)) == len(structure)


def test_twelve_edges_match_actual_skewed_cell():
    structure = silicon()
    edges = cell_edges(structure)
    assert edges.shape == (12, 2, 3)
    vectors = structure.lattice.get_fractional_coords(edges[:, 1] - edges[:, 0])
    assert np.allclose(np.abs(vectors).sum(axis=1), 1)
    assert np.allclose(vectors, np.round(vectors))


def test_static_view_preserves_scale_and_separates_axes():
    structure = silicon()
    fig = structure_figure(structure)
    try:
        main, orientation = fig.axes
        spans = np.asarray([np.diff(main.get_xlim())[0], np.diff(main.get_ylim())[0], np.diff(main.get_zlim())[0]])
        scale = main.get_box_aspect() / spans
        assert np.allclose(scale, scale[0])
        assert not main.axison and not orientation.axison
        assert main.get_position().x1 < orientation.get_position().x0
        assert main.get_legend().get_texts()[0].get_text() == "Si"
    finally:
        plt.close(fig)


def test_interactive_cell_toggle_contains_exact_input_and_display_copies():
    structure = silicon()
    html = viewer_html(structure, "primary.js", "fallback.js", "Offline")
    models = json.loads(re.search(r"var models = (.*?), colors =", html).group(1))
    assert [len(Poscar.from_str(model["poscar"]).structure) for model in models] == [16, 2]
    assert len(models[0]["edges"]) == 12
    assert "2 × 2 × 2 view · 2 atoms in input" in html
    assert '<button id="cell">Input cell</button>' in html
    assert "addUnitCell" not in html  # Default labels/arrows would overlap atoms.
    assert 'id="orientation"' in html
    assert "setViewChangeCallback(orientation)" in html
