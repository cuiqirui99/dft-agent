"""Prepare without a cluster, attach settings later, then submit."""

from importlib.resources import files
import json
import sys

import pytest

from vasp_slurm_agent import batch, cli, workflow
from vasp_slurm_agent.config import ClusterConfig


@pytest.fixture
def silicon(tmp_path):
    source = tmp_path / "Si.cif"
    source.write_bytes(files("vasp_slurm_agent").joinpath("examples", "Si.cif").read_bytes())
    return source


def cluster(**options):
    return ClusterConfig(host="cluster.example.invalid", user="researcher", remote_root="/scratch/test",
                         vasp_command="srun vasp_std", potcar_root="/licensed/pbe", partition="cpu", **options)


def snapshot(root):
    """Compare every user file, excluding the runtime's empty lock files."""
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*")
            if path.is_file() and path.name not in {".state.lock", ".worker.lock"}}


def test_prepare_without_config_then_attach(silicon, tmp_path):
    run = tmp_path / "run"
    state = workflow.prepare_plan(silicon, run, None, ["scf"], {"encut": 400})
    assert state["remote_root"] is None and state["config_sha256"] is None
    assert not (run / "config.json").exists()
    assert (run / "01_scf" / "inputs" / "INCAR").is_file()
    assert state["stages"][0]["metadata"]["potcar_labels"] == ["Si"]
    assert not workflow.has_config(run)
    with pytest.raises(ValueError, match="no cluster settings"):
        workflow.watch(run)
    attached = workflow.attach_config(run, cluster(potcar_symbols={"Si": "Si_sv"}))
    assert attached["remote_root"] == "/scratch/test/" + state["run_id"]
    assert attached["config_sha256"] == workflow._digest(run / "config.json")
    assert attached["stages"][0]["metadata"]["potcar_labels"] == ["Si_sv"]
    assert attached["history"][-1]["event"] == "Cluster settings attached."
    assert workflow._digest(run / "plan.json") == attached["plan_sha256"]
    workflow._check_config(run, attached)


@pytest.mark.parametrize("previous_config", [None, cluster(vasp_ncl_command="srun vasp_ncl")])
def test_attach_requires_launch_command_for_soc_without_changing_files(silicon, tmp_path, previous_config):
    run = tmp_path / "soc"
    workflow.prepare_plan(silicon, run, previous_config, ["scf"], {"spin": "noncollinear", "soc": True, "magmom": [[0, 0, 1], [0, 0, -1]]})
    before = snapshot(run)
    with pytest.raises(ValueError, match="vasp_ncl_command"):
        workflow.attach_config(run, cluster())
    assert snapshot(run) == before
    assert workflow.has_config(run) is (previous_config is not None)
    state = workflow.attach_config(run, cluster(vasp_ncl_command="srun vasp_ncl"))
    assert state["stages"][0]["metadata"]["requires_ncl"]


def test_attach_checks_later_soc_before_any_stage_is_submitted(silicon, tmp_path):
    run = tmp_path / "pipeline"
    soc = {"spin": "noncollinear", "soc": True, "magmom": [[0, 0, 1], [0, 0, -1]]}
    workflow.prepare_plan(silicon, run, None, ["relax", "scf"], stage_parameters=[{}, soc])
    before = snapshot(run)
    with pytest.raises(ValueError, match="vasp_ncl_command"):
        workflow.attach_config(run, cluster())
    assert snapshot(run) == before
    assert all(stage["job_id"] is None for stage in workflow.read_state(run)["stages"])
    state = workflow.attach_config(run, cluster(vasp_ncl_command="srun vasp_ncl"))
    assert state["stages"][0]["materialized"]
    assert not state["stages"][1]["materialized"]
    assert not (run / "02_scf" / "inputs").exists()
    workflow._check_config(run, state)


def test_attach_checks_later_method_with_new_potcar_mapping(silicon, tmp_path, monkeypatch):
    run = tmp_path / "hybrid"
    workflow.prepare_plan(silicon, run, None, ["relax", "scf"],
                          stage_parameters=[{}, {"functional": "HSE06"}])
    original = workflow.prepare_warm_inputs
    seen = []

    def validate(source, output, task, parameters, symbols):
        seen.append((parameters["functional"], symbols))
        return original(source, output, task, parameters, symbols)

    monkeypatch.setattr(workflow, "prepare_warm_inputs", validate)
    workflow.attach_config(run, cluster(potcar_symbols={"Si": "Si_sv"}))
    assert seen == [("HSE06", {"Si": "Si_sv"})]
    assert not (run / "02_scf" / "inputs").exists()


def test_attach_rolls_back_original_files_when_commit_fails(silicon, tmp_path, monkeypatch):
    run = tmp_path / "rollback"
    workflow.prepare_plan(silicon, run, cluster(), ["scf"])
    (run / "notes.txt").write_text("Keep my review notes.")
    before = snapshot(run)
    replace = workflow.os.replace
    failed = False

    def fail_commit(source, target):
        nonlocal failed
        if target == run / "run.json" and not failed:
            failed = True
            raise OSError("simulated write failure")
        return replace(source, target)

    monkeypatch.setattr(workflow.os, "replace", fail_commit)
    with pytest.raises(OSError, match="simulated write failure"):
        workflow.attach_config(run, cluster(potcar_symbols={"Si": "Si_sv"}))
    assert failed
    assert snapshot(run) == before
    workflow._check_config(run, workflow.read_state(run))


def test_attach_retains_originals_if_the_filesystem_also_blocks_rollback(silicon, tmp_path, monkeypatch):
    run = tmp_path / "blocked-rollback"
    workflow.prepare_plan(silicon, run, cluster(), ["scf"])
    original_state = (run / "run.json").read_bytes()
    replace = workflow.os.replace

    def block_run_file(source, target):
        if target == run / "run.json":
            raise OSError("run.json is temporarily inaccessible")
        return replace(source, target)

    monkeypatch.setattr(workflow.os, "replace", block_run_file)
    with pytest.raises(OSError, match="Originals are retained at"):
        workflow.attach_config(run, cluster(potcar_symbols={"Si": "Si_sv"}))
    retained = list(tmp_path.glob(".dft-attach-*/original/run.json"))
    assert len(retained) == 1 and retained[0].read_bytes() == original_state


def test_attach_refuses_after_submission(silicon, tmp_path):
    run = tmp_path / "submitted"
    state = workflow.prepare_plan(silicon, run, cluster(), ["scf"])
    state["stages"][0]["job_id"] = "123"
    state["status"] = "queued"
    workflow._write(run / "run.json", state)
    with pytest.raises(ValueError, match="before the first submission"):
        workflow.attach_config(run, cluster())
    with pytest.raises(ValueError, match="Save Cluster setup"):
        workflow.attach_config(run, None)


def test_attach_refuses_while_a_worker_owns_the_run(silicon, tmp_path):
    run = tmp_path / "monitored"
    workflow.prepare_plan(silicon, run, cluster(), ["scf"])
    before = snapshot(run)
    with (run / ".worker.lock").open("a") as worker:
        workflow._acquire(worker, blocking=False)
        with pytest.raises(ValueError, match="monitoring is running"):
            workflow.attach_config(run, cluster(potcar_symbols={"Si": "Si_sv"}))
    assert snapshot(run) == before


def test_batch_prepared_without_config_attaches_everywhere(silicon, tmp_path):
    root = tmp_path / "batch"
    state = batch.prepare_batch(silicon, root, None, ["scf"], [{"label": "A"}, {"label": "B", "strain": 0.01}])
    assert not (root / "config.json").exists()
    assert json.loads((root / "batch-plan.json").read_text())["config_sha256"] is None
    with pytest.raises(ValueError, match="no cluster settings"):
        batch.watch_batch(root)
    attached = batch.attach_batch_config(root, cluster())
    plan = json.loads((root / "batch-plan.json").read_text())
    assert plan["config_sha256"] == workflow._digest(root / "config.json")
    assert attached["plan_sha256"] == workflow._digest(root / "batch-plan.json")
    for member in state["runs"]:
        child = workflow.read_state(root / member["folder"])
        assert child["config_sha256"] == plan["config_sha256"]
    batch._children(root, attached)


def test_batch_attach_validation_failure_preserves_every_member(silicon, tmp_path):
    root = tmp_path / "batch"
    soc = {"spin": "noncollinear", "soc": True, "magmom": [[0, 0, 1], [0, 0, -1]]}
    batch.prepare_batch(silicon, root, None, ["scf"],
                        [{"label": "PBE"}, {"label": "SOC", "parameters": soc}])
    before = snapshot(root)
    with pytest.raises(ValueError, match="vasp_ncl_command"):
        batch.attach_batch_config(root, cluster())
    assert snapshot(root) == before
    assert not workflow.has_config(root)
    batch._children(root, batch.read_batch(root))
    assert batch.summarize_batch(root)["status"] == "planned"
    state = batch.attach_batch_config(root, cluster(vasp_ncl_command="srun vasp_ncl"))
    batch._children(root, state)


def test_batch_attach_rolls_back_all_members_on_commit_failure(silicon, tmp_path, monkeypatch):
    root = tmp_path / "batch"
    state = batch.prepare_batch(silicon, root, cluster(), ["scf"], [{"label": "A"}, {"label": "B"}])
    before = snapshot(root)
    replace = workflow.os.replace
    failed = False

    def fail_commit(source, target):
        nonlocal failed
        if target == root / "batch.json" and not failed:
            failed = True
            raise OSError("simulated batch write failure")
        return replace(source, target)

    monkeypatch.setattr(workflow.os, "replace", fail_commit)
    with pytest.raises(OSError, match="simulated batch write failure"):
        batch.attach_batch_config(root, cluster(potcar_symbols={"Si": "Si_sv"}))
    assert failed
    assert snapshot(root) == before
    batch._children(root, state)
    for member in state["runs"]:
        child = root / member["folder"]
        workflow._check_config(child, workflow.read_state(child))


def test_batch_attach_refuses_when_one_child_has_a_worker(silicon, tmp_path):
    root = tmp_path / "batch"
    state = batch.prepare_batch(silicon, root, cluster(), ["scf"], [{"label": "A"}, {"label": "B"}])
    before = snapshot(root)
    child = root / state["runs"][1]["folder"]
    with (child / ".worker.lock").open("a") as worker:
        workflow._acquire(worker, blocking=False)
        with pytest.raises(ValueError, match="monitoring is running"):
            batch.attach_batch_config(root, cluster(potcar_symbols={"Si": "Si_sv"}))
    assert snapshot(root) == before
    batch._children(root, state)


def test_cli_prepare_without_settings_reports_and_attach_adds_them(silicon, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DFT_AGENT_CONFIG", str(tmp_path / "missing.json"))
    for name in ("SSHTransport", "watch", "advance", "start_worker"):
        monkeypatch.setattr(workflow, name, lambda *a, **k: pytest.fail("no execution"))

    def invoke(*arguments):
        monkeypatch.setattr(sys, "argv", ["dft-agent", *map(str, arguments)])
        return cli.main()

    run = tmp_path / "run"
    assert invoke("prepare", silicon, run, "--task", "scf") == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["remote_root"] is None
    assert "prepared for review only" in captured.err
    assert invoke("attach", run, "--config", tmp_path / "nowhere.json") == 1
    assert "File not found" in capsys.readouterr().err
    config = cluster().save(tmp_path / "cluster.json")
    assert invoke("attach", run, "--config", config) == 0
    assert json.loads(capsys.readouterr().out)["remote_root"].startswith("/scratch/test/")
    assert invoke("status", run) == 0
