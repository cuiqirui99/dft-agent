import json
from pathlib import Path
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest
from pymatgen.io.vasp import Incar

from vasp_slurm_agent import restart, workflow
from test_plan import setup


def xml(path, spec, *, target=False, cold=False, nbands=8, iterations=2, swap=False):
    tags = spec["target_tags" if target else "seed_tags"]
    tree = ET.Element("modeling")
    def tag(parent, name, value):
        vector = isinstance(value, (tuple, list))
        values = value if vector else [value]
        if vector and values and isinstance(values[0], list):
            values = [item for row in values for item in row]
        kind = "logical" if type(values[0]) is bool else "int" if type(values[0]) is int else "string" if isinstance(values[0], str) else "float"
        element = ET.SubElement(parent, "v" if vector else "i", name=name, type=kind)
        element.text = " ".join("T" if value is True else "F" if value is False else str(value) for value in values)
    incar = ET.SubElement(tree, "incar")
    for name, value in tags.items():
        tag(incar, name, value)
    parameters = ET.SubElement(tree, "parameters")
    for name, value in {**tags, "NBANDS": nbands, "NELECT": 8., "LNONCOLLINEAR": False, "LSORBIT": False,
                        "ISTART": 0 if cold else tags["ISTART"]}.items():
        tag(parameters, name, value)
    def array(parent, name, rows):
        parent = ET.SubElement(parent, "varray", name=name)
        for row in rows:
            ET.SubElement(parent, "v").text = " ".join(str(item) for item in row)
    points = [[0., 0., 0.], [.5, 0., 0.]]
    kpoints = ET.SubElement(tree, "kpoints")
    array(kpoints, "kpointlist", points[::-1] if swap else points)
    array(kpoints, "weights", [[.5], [.5]])
    for name in ("initialpos", "finalpos"):
        structure = ET.SubElement(tree, "structure", name=name)
        array(ET.SubElement(structure, "crystal"), "basis", spec["lattice"])
        array(structure, "positions", spec["positions"])
    calculation = ET.SubElement(tree, "calculation")
    tag(ET.SubElement(calculation, "energy"), "e_0_energy", -10.)
    for _ in range(iterations):
        ET.SubElement(calculation, "scstep")
    ET.ElementTree(tree).write(path)


def wave(path, spec, *, nbands=8, spin=1, encut=None):
    recl, nk = 512, 2
    data = bytearray(recl * (2 + spin * nk * (nbands + 1)))
    struct.pack_into("<3d", data, 0, recl, spin, 45200)
    struct.pack_into("<12d", data, recl, nk, nbands, encut or spec["target_tags"]["ENCUT"],
                     *[v for row in spec["lattice"] for v in row])
    for channel in range(spin):
        for index, point in enumerate([[0., 0., 0.], [.5, 0., 0.]]):
            struct.pack_into("<4d", data, recl * (2 + (channel * nk + index) * (nbands + 1)), 10, *point)
    path.write_bytes(data)


@pytest.fixture
def warm(setup):
    source, root, config = setup
    state = workflow.prepare_plan(source, root, config, ["scf"], {"functional": "HSE06", "mesh": [2, 2, 2]})
    inputs = root / "01_scf/inputs"
    (inputs / "POTCAR").write_text("test licensed potential")
    restart.initialize(inputs)
    spec = json.loads((inputs / "warm_start.spec.json").read_text())
    seed = inputs / ".pbe_seed"
    xml(seed / "vasprun.xml", spec)
    (seed / "OUTCAR").write_text("seed test output")
    wave(seed / "WAVECAR", spec)
    return inputs, spec, state, config


@pytest.mark.parametrize("task", ["relax", "scf", "bands", "dos"])
def test_new_hybrid_inputs_require_warm_start_on_the_same_basis(setup, task):
    source, root, config = setup
    state = workflow.prepare_plan(source, root, config, [task], {"functional": "HSE06", "mesh": [2, 2, 2]})
    stage = state["stages"][0]
    inputs = root / stage["folder"] / "inputs"
    target, seed = Incar.from_file(inputs / "INCAR"), Incar.from_file(inputs / "seed.INCAR")
    assert stage["warm_start"] and stage["metadata"]["warm_start"]
    assert target["ISTART"] == 1 and target["ICHARG"] == 0
    assert seed["LWAVE"] and seed["NSW"] == 0 and seed["ICHARG"] == 2
    assert not seed.get("LHFCALC", False) and target["LHFCALC"]
    assert seed["ISYM"] == target["ISYM"] and seed["ENCUT"] == target["ENCUT"]
    spec = json.loads((inputs / "warm_start.spec.json").read_text())
    assert spec["target_inputs"]["KPOINTS"] == restart.sha256(inputs / "KPOINTS")
    if task == "bands":
        assert stage["metadata"]["band_path_offset"] == 8
    assert json.loads((root / "plan.json").read_text())["stages"][0]["warm_start"]


def test_remote_seed_copy_and_actual_restart_receipts(warm):
    inputs, spec, _, _ = warm
    original = (inputs / "INCAR").read_bytes()
    restart.accept_seed(inputs, 0)
    receipt = json.loads((inputs / "warm_start.json").read_text())
    assert receipt["seed_accepted"] and receipt["wavecar"]["nbands"] == 8
    assert restart.sha256(inputs / "WAVECAR") == restart.sha256(inputs / ".pbe_seed/WAVECAR")
    xml(inputs / "vasprun.xml", spec, target=True)
    (inputs / "hybrid.stdout").write_text("WAVECAR successfully read\n")
    restart.finish(inputs, 0)
    assert json.loads((inputs / "warm_start.json").read_text())["restart_verified"]
    assert (inputs / "INCAR").read_bytes() == original


def test_vasp_runtime_cutoff_spelling_and_magnetic_vector_shape(warm):
    inputs, spec, _, _ = warm
    tree = ET.parse(inputs / ".pbe_seed/vasprun.xml")
    tree.find("parameters/i[@name='ENCUT']").set("name", "ENMAX")
    tree.write(inputs / ".pbe_seed/vasprun.xml")
    restart.accept_seed(inputs, 0)
    assert json.loads((inputs / "warm_start.json").read_text())["seed_accepted"]
    restart._check_tags({"MAGMOM": [1, 2, 3, -1, -2, -3], "LDAUTYPE": [2]},
                        {"MAGMOM": [[1, 2, 3], [-1, -2, -3]], "LDAUTYPE": 2})
    with pytest.raises(ValueError, match="MAGMOM"):
        restart._check_tags({"MAGMOM": [1, 2, 3, 1, 2, 3]}, {"MAGMOM": [[1, 2, 3], [-1, -2, -3]]})


@pytest.mark.parametrize("key", ["ENCUT", "ISPIN", "ISYM", "LSORBIT", "LNONCOLLINEAR", "NBANDS", "NELECT"])
def test_restart_requires_runtime_values_not_only_requested_incar(warm, key):
    inputs, spec, _, _ = warm
    seed = restart.xml_summary(inputs / ".pbe_seed/vasprun.xml")
    seed["incar"][key] = seed["actual"].pop(key)
    with pytest.raises(ValueError, match="actual"):
        restart._effective(seed, key)
    if key == "ENCUT":
        seed["actual"][key] = 100.
        with pytest.raises(ValueError, match="actual ENCUT"):
            restart._check_seed(seed, spec)


@pytest.mark.parametrize("issue", ["exit", "convergence", "header_bands", "cutoff", "potential", "input", "truncated"])
def test_incompatible_seed_never_copies_wavecar(warm, issue):
    inputs, spec, _, _ = warm
    if issue == "convergence":
        xml(inputs / ".pbe_seed/vasprun.xml", spec, iterations=spec["seed_tags"]["NELM"])
    elif issue == "header_bands":
        wave(inputs / ".pbe_seed/WAVECAR", spec, nbands=9)
    elif issue == "cutoff":
        wave(inputs / ".pbe_seed/WAVECAR", spec, encut=100)
    elif issue == "potential":
        (inputs / ".pbe_seed/POTCAR").write_text("changed potential")
    elif issue == "input":
        (inputs / "INCAR").write_text("changed target")
    elif issue == "truncated":
        (inputs / ".pbe_seed/WAVECAR").write_bytes(b"incomplete")
    with pytest.raises((ValueError, struct.error)):
        restart.accept_seed(inputs, 1 if issue == "exit" else 0)
    assert not (inputs / "WAVECAR").exists()
    assert not json.loads((inputs / "warm_start.json").read_text())["seed_accepted"]


@pytest.mark.parametrize("issue", ["cold", "bands", "kpoint_order", "missing_read", "changed_wave", "exit"])
def test_hybrid_cold_fallback_or_changed_basis_is_rejected(warm, issue):
    inputs, spec, _, _ = warm
    restart.accept_seed(inputs, 0)
    xml(inputs / "vasprun.xml", spec, target=True, cold=issue == "cold", nbands=9 if issue == "bands" else 8,
        swap=issue == "kpoint_order")
    (inputs / "hybrid.stdout").write_text("WAVECAR not read" if issue == "missing_read" else "WAVECAR successfully read")
    if issue == "changed_wave":
        (inputs / "WAVECAR").write_bytes(b"changed")
    with pytest.raises(ValueError):
        restart.finish(inputs, 1 if issue == "exit" else 0)
    assert not json.loads((inputs / "warm_start.json").read_text())["restart_verified"]


def test_helper_is_standalone_stdlib_and_failure_is_recorded(warm):
    inputs, _, _, _ = warm
    process = subprocess.run([sys.executable, "-I", str(Path(restart.__file__)), "seed", "3"], cwd=inputs,
                             capture_output=True, text=True)
    assert process.returncode != 0
    assert json.loads((inputs / "warm_start.json").read_text())["seed_exit_code"] == 3


@pytest.mark.skipif(sys.platform == "win32", reason="Executes a Linux job script with local Bash and POSIX paths")
def test_seed_failure_records_job_exit_and_does_not_launch_hybrid(warm, tmp_path):
    from dataclasses import replace
    inputs, _, state, config = warm
    # initialize must create a fresh seed directory in this simulated job.
    import shutil
    shutil.rmtree(inputs / ".pbe_seed")
    potential = tmp_path / "potentials/Si"
    potential.mkdir(parents=True)
    (potential / "POTCAR").write_text("TITEL test Si\n")
    (inputs / "restart.py").write_bytes(Path(restart.__file__).read_bytes())
    config = replace(config, potcar_root=str(potential.parent), vasp_command="echo launched >> ../launches; false")
    script = inputs / "submit.sh"
    script.write_text(workflow._script(config, state["stages"][0], None), encoding="utf-8", newline="\n")
    process = subprocess.run(["bash", str(script)], cwd=inputs, capture_output=True, text=True)
    assert process.returncode != 0
    receipt = json.loads((inputs / "execution.json").read_text())
    assert receipt["exit_code"] != 0 and receipt["finished_at"]
    assert (inputs / "launches").read_text().count("launched") == 1
    assert not (inputs / "hybrid.stdout").exists()
