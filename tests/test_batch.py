"""Batch orchestration and comparisons retain per-run evidence boundaries."""

from copy import deepcopy
import csv
import json
import zipfile

import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Poscar

from vasp_slurm_agent import batch, explanation, workflow
from vasp_slurm_agent.config import ClusterConfig
from test_plan import fake_analysis
from test_workflow import FakeTransport, completed_files


@pytest.fixture
def setup(tmp_path, monkeypatch):
    config = ClusterConfig(host="cluster.invalid", user="test", remote_root="/scratch/test",
                           vasp_command="srun vasp_std", vasp_ncl_command="srun vasp_ncl",
                           potcar_root="/licensed/pbe", partition="cpu")
    source = tmp_path / "POSCAR"
    Poscar(Structure(Lattice.cubic(5), ["O", "Fe"], [[0, 0, 0], [.5, .5, .5]])).write_file(source)
    monkeypatch.setattr(workflow, "SSHTransport", lambda *a, **k: pytest.fail("No real SSH"))
    monkeypatch.setattr(batch, "SSHTransport", lambda *a, **k: pytest.fail("No real SSH"))
    return source, tmp_path / "batch", config


def test_variants_have_isolated_inputs_methods_and_stage_recipes(setup):
    source, root, config = setup
    variants = [
        {"label": "FM", "parameters": {"spin": "collinear", "magmom": [.1, 4]}},
        {"label": "AFM stripe", "parameters": {"spin": "collinear", "magmom": [-.1, 4]}},
        {"label": "U4", "stage_parameters": [{}, {"functional": "HSE06", "hubbard_u": {"Fe": {"l": 2, "u": 4, "j": 0}}}]},
        {"label": "strain 2%", "strain": [.02, 0, 0]},
    ]
    state = batch.prepare_batch(source, root, config, ["relax", "scf"], variants,
                                parameters={"encut": 400}, stage_parameters=[{}, {"encut": 500}])
    assert state["status"] == "planned"
    assert len({member["run_id"] for member in state["runs"]}) == 4
    children = [workflow.read_state(root / member["folder"]) for member in state["runs"]]
    assert children[0]["stages"][0]["metadata"]["method"]["magmom"] == [4, .1]
    assert children[1]["stages"][0]["metadata"]["method"]["magmom"] == [4, -.1]
    assert children[2]["stages"][1]["parameters"]["functional"] == "HSE06"
    assert children[2]["stages"][1]["parameters"]["encut"] == 500
    strained = Structure.from_file(root / state["runs"][3]["folder"] / "01_relax/inputs/POSCAR")
    assert strained.lattice.a == pytest.approx(5.1)
    assert Structure.from_file(source).lattice.a == 5
    first = root / state["runs"][0]["folder"] / "01_relax/inputs/INCAR"
    second = root / state["runs"][1]["folder"] / "01_relax/inputs/INCAR"
    original = second.read_bytes()
    first.write_text("changed first variant")
    assert second.read_bytes() == original


@pytest.mark.parametrize("variant", [
    {"label": "bad", "strain": -1},
    {"label": "bad", "strain": True},
    {"label": "bad", "parameters": {"spin": "collinear", "magmom": [1]}},
    {"label": "bad", "operations": {}},
])
def test_one_invalid_variant_leaves_no_partial_batch(setup, variant):
    source, root, config = setup
    with pytest.raises(ValueError):
        batch.prepare_batch(source, root, config, ["scf"], [{"label": "valid"}, variant])
    assert not root.exists()


def test_strain_cannot_relax_away_its_cell_constraint(setup):
    source, root, config = setup
    with pytest.raises(ValueError, match="cell_relax"):
        batch.prepare_batch(source, root, config, ["relax"],
                            [{"label": "strain", "strain": .01}], parameters={"cell_relax": True})
    assert not root.exists()


def test_structure_operations_and_site_moments_use_edited_structure(setup):
    source, root, config = setup
    state = batch.prepare_batch(source, root, config, ["scf"], [{
        "label": "two cells", "operations": [{"type": "supercell", "matrix": [[2, 0, 0], [0, 1, 0], [0, 0, 1]]}],
        "parameters": {"spin": "collinear", "magmom": [.1, -.1, 4, -4]},
    }])
    child = workflow.read_state(root / state["runs"][0]["folder"])
    assert child["stages"][0]["metadata"]["number_of_atoms"] == 4
    assert len(child["stages"][0]["metadata"]["method"]["magmom"]) == 4


def test_failed_member_does_not_stop_another_job(setup, monkeypatch):
    source, root, config = setup
    state = batch.prepare_batch(source, root, config, ["scf"], [{"label": "a"}, {"label": "b"}])
    remote = FakeTransport()
    state = batch.advance_batch(root, remote)
    assert [member["status"] for member in state["runs"]] == ["queued", "queued"]
    assert len(remote.remote_jobs) == 2
    first, second = [root / member["folder"] for member in state["runs"]]
    failed = workflow.read_state(first)
    failed["status"] = failed["stages"][0]["status"] = "needs_attention"
    failed["last_error"] = "Failed calculation."
    workflow._write(first / "run.json", failed)
    monkeypatch.setattr(workflow, "analyze_outputs", fake_analysis)
    completed_files(second, remote)
    state = batch.advance_batch(root, remote)
    assert [member["status"] for member in state["runs"]] == ["needs_attention", "succeeded"]
    assert len(remote.remote_jobs) == 2
    assert batch.advance_batch(root, remote)["status"] == "needs_attention"
    assert len(remote.remote_jobs) == 2


def test_cancel_before_submit_and_watcher_do_not_submit(setup):
    source, root, config = setup
    batch.prepare_batch(source, root, config, ["scf"], [{"label": "a"}, {"label": "b"}])
    remote = FakeTransport()
    state = batch.cancel_batch(root, remote)
    assert state["status"] == "cancelled"
    assert all(member["status"] == "cancelled" for member in state["runs"])
    assert not remote.commands
    assert batch.watch_batch(root, interval=0, transport=remote)["status"] == "cancelled"
    assert not remote.commands and (root / "results.zip").is_file()


def test_batch_worker_uses_existing_singleton_lock(setup):
    import fcntl
    source, root, config = setup
    batch.prepare_batch(source, root, config, ["scf"], [{"label": "a"}])
    with (root / ".worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert batch.watch_batch(root)["status"] == "planned"


@pytest.mark.parametrize("change", ["config", "plan", "member"])
def test_tampered_batch_never_connects_or_submits(setup, change):
    source, root, config = setup
    state = batch.prepare_batch(source, root, config, ["scf"], [{"label": "a"}])
    if change == "config":
        (root / "config.json").write_text("{}")
    elif change == "plan":
        (root / "batch-plan.json").write_text("{}")
    else:
        state["runs"][0]["folder"] = "../unrelated"
        workflow._write(root / "batch.json", state)
    remote = FakeTransport()
    with pytest.raises(ValueError):
        batch.advance_batch(root, remote)
    assert not remote.commands
    with pytest.raises(ValueError):
        batch.watch_batch(root)


def accept_child(root, energy, results):
    state = workflow.read_state(root)
    stage = state["stages"][0]
    inputs, output = root / stage["folder"] / "inputs", root / stage["folder"] / "outputs"
    output.mkdir()
    for name in ("POSCAR", "INCAR", "KPOINTS"):
        (output / name).write_bytes((inputs / name).read_bytes())
    (output / "vasprun.xml").write_text(state["run_id"])
    (output / "OUTCAR").write_text("retained solver fixture")
    if stage["metadata"].get("warm_start"):
        for name in ("seed.INCAR", "seed.metadata.json", "warm_start.spec.json"):
            (output / name).write_bytes((inputs / name).read_bytes())
        for name in ("warm_start.json", "seed.vasprun.xml", "seed.OUTCAR", "hybrid.stdout", "potcar_hash.sha256"):
            (output / name).write_text("warm-start fixture; parsing is stubbed in this test")
    (output / "input_hashes.sha256").write_text("\n".join(
        f"{workflow._digest(inputs / name)}  {name}" for name in ("INCAR", "KPOINTS", "POSCAR")))
    structure = Structure.from_file(inputs / "POSCAR")
    structure.to(filename=output / "final_structure.cif")
    workflow._write(output / "artifact_manifest.json", {
        path.name: {"sha256": workflow._digest(path)} for path in output.iterdir()})
    result = {"success": True, "converged_electronic": True, "converged_ionic": True,
              "input_identity_verified": True, "scheduler_state": "COMPLETED", "final_energy_ev": energy,
              "final_energy_ev_per_atom": energy / len(structure), "final_max_force_ev_angstrom": .001,
              "fermi_energy_ev": 2., "vasprun_sha256": workflow._digest(output / "vasprun.xml"),
              "final_structure_sha256": workflow._digest(output / "final_structure.cif"),
              "method_fingerprint": stage["metadata"]["method_fingerprint"],
              "method": stage["metadata"]["method"],
              "band_gap": {"gap_ev": 1.2, "scope": "sampled_kpoints", "status": "resolved"}}
    results[state["run_id"]] = deepcopy(result)
    workflow._write(output / "result.json", result)
    stage.update(status="succeeded", scheduler_state="COMPLETED", result=result)
    state["status"] = "succeeded"
    workflow._write(root / "run.json", state)
    return output


def test_summary_rechecks_outputs_and_only_ranks_compatible_variants(setup, monkeypatch):
    source, root, config = setup
    variants = [
        {"label": "FM", "parameters": {"spin": "collinear", "magmom": [.1, 4]}},
        {"label": "AFM", "parameters": {"spin": "collinear", "magmom": [-.1, 4]}},
        {"label": "U4", "parameters": {"hubbard_u": {"Fe": {"l": 2, "u": 4, "j": 0}}}},
        {"label": "HSE06", "parameters": {"functional": "HSE06"}},
        {"label": "strain", "strain": .01},
    ]
    state = batch.prepare_batch(source, root, config, ["scf"], variants)
    results = {}
    outputs = [accept_child(root / member["folder"], -10 - index, results)
               for index, member in enumerate(state["runs"])]
    monkeypatch.setattr(explanation, "analyze_outputs", lambda path, *args: deepcopy(
        results[(path / "vasprun.xml").read_text()]))
    batch._refresh(root, batch.read_batch(root))
    summary = batch.summarize_batch(root)
    assert summary["status"] == "succeeded"
    rows = {row["label"]: row for row in summary["rows"]}
    assert len(summary["comparisons"]) == 1
    assert summary["comparisons"][0]["labels"] == ["AFM", "FM"]
    assert rows["AFM"]["rank"] == 1 and rows["FM"]["rank"] == 2
    assert rows["FM"]["relative_energy_mev_per_atom"] == 500
    assert all(rows[name]["rank"] is None for name in ("U4", "HSE06", "strain"))
    assert all(row["band_gap_ev"] == 1.2 and row["band_gap_scope"] == "sampled_kpoints"
               and row["band_gap_status"] == "resolved" for row in rows.values())
    assert (root / "summary.json").is_file()
    with (root / "summary.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 5
    # Editing cached energy plus state cannot override the retained solver values.
    child_root = root / state["runs"][0]["folder"]
    child = workflow.read_state(child_root)
    child["stages"][0]["result"]["final_energy_ev_per_atom"] = -999
    workflow._write(child_root / "run.json", child)
    workflow._write(outputs[0] / "result.json", child["stages"][0]["result"])
    changed = batch.summarize_batch(root)
    assert changed["rows"][0]["energy_ev_per_atom"] is None
    assert changed["rows"][0]["status"] == "needs_attention"
    assert changed["status"] == "needs_attention"
    assert changed["comparisons"] == []


def test_archive_includes_task_and_parent_without_potcar(setup):
    source, root, config = setup
    state = batch.prepare_batch(source, root, config, ["scf"], [{"label": "a"}])
    (root / "proposal.json").write_text('{"goal":"SCF"}')
    (root / "parent.json").write_text('{"run_id":"parent"}')
    (root / "structure_edit").mkdir()
    (root / "structure_edit/POSCAR").write_bytes(source.read_bytes())
    child = root / state["runs"][0]["folder"]
    (child / "parent.json").write_text('{"run_id":"parent"}')
    (child / "01_scf/inputs/POTCAR").write_text("licensed file")
    archive = batch.bundle_batch(root)
    with zipfile.ZipFile(archive) as saved:
        assert {"proposal.json", "parent.json", "structure_edit/POSCAR", "source/POSCAR"} <= set(saved.namelist())
    with zipfile.ZipFile(child / "results.zip") as saved:
        assert "parent.json" in saved.namelist()
        assert not any(name.endswith("POTCAR") for name in saved.namelist())
