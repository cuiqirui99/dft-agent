"""CLI planning and preparation with a stubbed model and no remote execution."""

import hashlib
from importlib.resources import files
import json
import sys
import zipfile

import pytest

from vasp_slurm_agent import agent, cli, workflow
from vasp_slurm_agent.config import ClusterConfig


def model_response(tasks=None, missing_moments=False):
    tasks = tasks or ["relax", "scf"]
    return {
        "status": "ready", "summary": "Prepare the requested calculation.", "tasks": tasks,
        "parameters": {**dict.fromkeys(agent.DEFAULTS), "spin": "collinear" if missing_moments else "none",
                       "magmom": None, "soc": False, "saxis": [0, 0, 1], "functional": "PBE",
                       "hubbard_u": {"Si": None}},
        "magnetic_states": [], "initial_moment": None, "questions": [], "notes": [],
        "intent": {"requested_tasks": tasks, "forbidden_tasks": [], "spin": None, "soc": None,
                   "functional": None, "hubbard_u": None, "unsupported": []},
    }


@pytest.fixture
def local_cli(tmp_path, monkeypatch):
    source = tmp_path / "Si.cif"
    source.write_bytes(files("vasp_slurm_agent").joinpath("examples", "Si.cif").read_bytes())
    config = ClusterConfig(host="cluster.invalid", user="test", remote_root="/scratch/test",
                           vasp_command="srun vasp_std", potcar_root="/licensed/pbe", partition="cpu")
    config_path = config.save(tmp_path / "cluster.json")

    def no_execution(*args, **kwargs):
        pytest.fail("Plan and prepare must not execute a job or contact SSH")

    for name in ("SSHTransport", "watch", "advance", "start_worker"):
        monkeypatch.setattr(workflow, name, no_execution)

    def invoke(*arguments):
        monkeypatch.setattr(sys, "argv", ["dft-agent", *map(str, arguments)])
        return cli.main()

    return source, config_path, tmp_path / "run", invoke


def test_cli_model_proposal_binds_source_and_preserves_frozen_plan(local_cli, monkeypatch, capsys):
    source, config, root, invoke = local_cli
    proposal_path = root.parent / "proposal.json"
    monkeypatch.setattr(agent, "_request_plan", lambda *args: json.dumps(model_response()))
    assert invoke("plan", source, "Relax, then SCF.", "--provider", "codex", "--output", proposal_path) == 0
    proposal = json.loads(proposal_path.read_text())
    assert proposal["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert not root.exists()
    capsys.readouterr()
    assert invoke("prepare", source, root, "--config", config, "--plan", proposal_path) == 0
    state = json.loads(capsys.readouterr().out)
    assert state["status"] == "planned" and all(stage["job_id"] is None for stage in state["stages"])
    assert json.loads((root / "proposal.json").read_text()) == proposal
    frozen = json.loads((root / "plan.json").read_text())
    assert frozen["stages"] and "status" not in frozen
    assert state["plan_sha256"] == hashlib.sha256((root / "plan.json").read_bytes()).hexdigest()
    workflow._check_config(root, state)
    with zipfile.ZipFile(workflow.bundle_run(root)) as bundle:
        assert bundle.read("proposal.json") == (root / "proposal.json").read_bytes()
        assert bundle.read("plan.json") == (root / "plan.json").read_bytes()


@pytest.mark.parametrize("problem", ["changed_structure", "missing_hash", "needs_input"])
def test_cli_refuses_unbound_or_incomplete_proposal(local_cli, capsys, problem):
    source, config, root, invoke = local_cli
    proposal = {"status": "ready", "tasks": ["scf"], "parameters": {},
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    if problem == "changed_structure":
        source.write_bytes(source.read_bytes() + b"\n")
    elif problem == "missing_hash":
        proposal.pop("source_sha256")
    else:
        proposal["status"] = "needs_input"
    path = root.parent / "proposal.json"
    path.write_text(json.dumps(proposal))
    assert invoke("prepare", source, root, "--config", config, "--plan", path) == 1
    assert "Error:" in capsys.readouterr().err
    assert not root.exists()


def test_cli_needs_input_is_saved_without_preparation(local_cli, monkeypatch, capsys):
    source, config, root, invoke = local_cli
    proposal_path = root.parent / "proposal.json"
    monkeypatch.setattr(agent, "_request_plan", lambda *args: json.dumps(model_response(["scf"], missing_moments=True)))
    assert invoke("plan", source, "Spin-polarized SCF.", "--provider", "codex", "--output", proposal_path) == 1
    proposal = json.loads(proposal_path.read_text())
    assert proposal["status"] == "needs_input" and proposal["questions"]
    assert not root.exists()
    capsys.readouterr()
    assert invoke("prepare", source, root, "--config", config, "--plan", proposal_path) == 1
    assert "questions" in capsys.readouterr().err
    assert not root.exists()
