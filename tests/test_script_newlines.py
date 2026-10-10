"""Generated Linux job files keep LF bytes on every client platform."""

from dataclasses import replace
from pathlib import Path

import pytest

from vasp_slurm_agent import workflow
from vasp_slurm_agent.samples import install_samples
from test_plan import setup
from test_workflow import FakeTransport


@pytest.mark.parametrize("functional", ["PBE", "HSE06"])
@pytest.mark.parametrize("batch_upload", [False, True])
def test_uploaded_job_files_use_lf_bytes(setup, functional, batch_upload):
    source, root, config = setup
    config = replace(config, setup_commands=["module purge\r\nmodule load vasp"])
    workflow.prepare_plan(source, root, config, ["scf"], {"functional": functional, "mesh": [2, 2, 2]})
    uploaded = {}

    class InspectUploads(FakeTransport):
        def upload(self, source, destination):
            uploaded[Path(source).name] = Path(source).read_bytes()
            return super().upload(source, destination)

    transport = InspectUploads()
    if batch_upload:
        transport.upload_many = lambda paths, remote: {
            path.name: transport.upload(path, remote + "/" + path.name) for path in paths
        }
    state = workflow.advance(root, transport)
    assert state["status"] == "queued", state.get("last_error")
    assert {"submit.sh", "dispatch.py", "POSCAR", "INCAR", "KPOINTS"} <= uploaded.keys()
    assert uploaded["submit.sh"].startswith(b"#!/bin/bash\n")
    assert b"module purge\nmodule load vasp\n" in uploaded["submit.sh"]
    if functional == "HSE06":
        assert {"restart.py", "seed.INCAR", "seed.metadata.json", "warm_start.spec.json"} <= uploaded.keys()
    for name, data in uploaded.items():
        assert b"\r\n" not in data, name


def test_bundled_job_scripts_use_lf_bytes(tmp_path):
    scripts = [script for run in install_samples(tmp_path) for script in run.rglob("submit.sh")]
    assert scripts
    for script in scripts:
        data = script.read_bytes()
        assert data.startswith(b"#!/bin/bash\n"), script
        assert b"\r\n" not in data, script
