from types import SimpleNamespace

import numpy as np
import pytest

from vasp_slurm_agent.vasp import _band_gap


def spectrum(values, second=None):
    eigenvalues = {1: np.asarray(values)}
    if second is not None:
        eigenvalues[-1] = np.asarray(second)
    return SimpleNamespace(eigenvalues=eigenvalues, actual_kpoints=[[0, 0, i / len(values)] for i in range(len(values))])


def test_indirect_sampled_gap_and_same_k_separation():
    result = _band_gap(spectrum([[[-2, 1], [1, 0]], [[-1, 1], [2, 0]]]), "scf", {})
    assert result["available"] and result["status"] == "gapped"
    assert result["gap_ev"] == 2 and result["direct_gap_ev"] == 3
    assert result["vbm_ev"] == -1 and result["cbm_ev"] == 1
    assert result["scope"] == "sampled_kpoints" and not result["full_bz_validated"]


def test_hybrid_path_excludes_self_consistent_mesh():
    run = spectrum([[[-.1, 1], [.1, 0]], [[-2, 1], [1, 0]], [[-1, 1], [2, 0]]])
    result = _band_gap(run, "bands", {"band_path_offset": 1, "band_path_count": 2})
    assert result["gap_ev"] == 2 and result["scope"] == "sampled_path"
    assert result["kpoint_count"] == 2


def test_spin_channels_use_global_edges():
    run = spectrum([[[-3, 1], [3, 0]], [[-2, 1], [4, 0]]],
                   [[[-1, 1], [2, 0]], [[-2, 1], [1, 0]]])
    result = _band_gap(run, "dos", {})
    assert result["gap_ev"] == 2 and result["vbm_ev"] == -1


def test_partial_occupations_do_not_fabricate_gap_or_definitive_metal():
    result = _band_gap(spectrum([[[-1, 1], [0, .2], [2, 0]]]), "scf", {})
    assert result["status"] == "metallic_or_partially_occupied"
    assert result["gap_ev"] is None and not result["available"]
    assert result["partial_occupations"]


@pytest.mark.parametrize("values", [
    [[[-1, 1], [2, 0]], [[-2, 1], [-.5, 1], [3, 0]]],
    [[[0, 1], [2, 0]], [[-2, 1], [-1, 0]]],
])
def test_crossing_or_overlap_is_zero_sampled_gap(values):
    # Pad the crossing fixture to the same band count without changing its edges.
    if len(values[0]) != len(values[1]):
        values[0].append([4, 0])
    result = _band_gap(spectrum(values), "scf", {})
    assert result["status"] == "metallic" and result["gap_ev"] == 0


@pytest.mark.parametrize("run,task,metadata", [
    (SimpleNamespace(), "scf", {}),
    (spectrum([[[-1, 1], [1, 0]]]), "bands", {}),
    (spectrum([[[-1, 1], [1, 0]]]), "bands", {"band_path_offset": 0, "band_path_count": 2}),
    (spectrum([[[-1, 2], [1, 0]]]), "scf", {}),
    (spectrum([[[-1, 1], [float("nan"), 0]]]), "scf", {}),
    (spectrum([[[-1, 1], [1, 1]]]), "scf", {}),
])
def test_missing_or_ambiguous_eigenvalues_leave_gap_unavailable(run, task, metadata):
    result = _band_gap(run, task, metadata)
    assert not result["available"] and result["gap_ev"] is None
