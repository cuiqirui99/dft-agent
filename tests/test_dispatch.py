from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import subprocess

import pytest

from vasp_slurm_agent import dispatch as remote_dispatch


def completed(argv, stdout="", returncode=0, stderr=""):
    return subprocess.CompletedProcess(argv, returncode, stdout, stderr)


def test_receipt_replay_submits_only_once(monkeypatch, tmp_path):
    calls = []

    def slurm(argv, **kwargs):
        calls.append(argv)
        assert argv[0] == "sbatch"
        assert kwargs["cwd"] == tmp_path
        return completed(argv, "40721;cluster\n")

    monkeypatch.setattr(remote_dispatch.subprocess, "run", slurm)
    first = remote_dispatch.dispatch(tmp_path, "vsa-example-relax")
    second = remote_dispatch.dispatch(tmp_path, "vsa-example-relax")
    assert first == second
    assert first["job_id"] == "40721"
    assert len(calls) == 1
    assert (tmp_path / "submit_intent.json").exists()
    assert json.loads((tmp_path / "job.json").read_text()) == first


def test_lost_receipt_reconstructs_original_job_without_resubmission(monkeypatch, tmp_path):
    calls = []
    name = "vsa-recover-relax"

    def slurm(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "sbatch":
            # Scheduler accepted the job, but the process lost its response.
            raise subprocess.TimeoutExpired(argv, 60)
        if argv[0] == "squeue":
            return completed(argv, f"50721|{name}\n")
        assert argv[0] == "sacct"
        # Duplicate observations and batch-step records must not duplicate jobs.
        return completed(argv, f"50721|{name}|\n50721.batch|{name}|\n")

    monkeypatch.setattr(remote_dispatch.subprocess, "run", slurm)
    with pytest.raises(subprocess.TimeoutExpired):
        remote_dispatch.dispatch(tmp_path, name)
    assert (tmp_path / "submit_intent.json").exists()
    assert not (tmp_path / "job.json").exists()
    recovered = remote_dispatch.dispatch(tmp_path, name)
    assert recovered == {"job_id": "50721", "job_name": name, "reconciled": True}
    assert [call[0] for call in calls].count("sbatch") == 1
    assert remote_dispatch.dispatch(tmp_path, name) == recovered
    assert [call[0] for call in calls].count("sbatch") == 1


@pytest.mark.parametrize("jobs", [[], ["50721", "50722"]])
def test_uncertain_intent_refuses_automatic_resubmission(monkeypatch, tmp_path, jobs):
    name = "vsa-uncertain-relax"
    (tmp_path / "submit_intent.json").write_text(json.dumps({"job_name": name, "date": "2026-01-01"}))
    calls = []

    def slurm(argv, **kwargs):
        calls.append(argv)
        assert argv[0] in {"squeue", "sacct"}
        return completed(argv, "".join(f"{job}|{name}\n" for job in jobs))

    monkeypatch.setattr(remote_dispatch.subprocess, "run", slurm)
    with pytest.raises(RuntimeError, match="outcome uncertain"):
        remote_dispatch.dispatch(tmp_path, name)
    assert len(calls) == 2
    assert not (tmp_path / "job.json").exists()


def test_failed_reconciliation_does_not_submit(monkeypatch, tmp_path):
    name = "vsa-unavailable-relax"
    (tmp_path / "submit_intent.json").write_text(json.dumps({"job_name": name, "date": "2026-01-01"}))

    def slurm(argv, **kwargs):
        assert argv[0] == "squeue"
        return completed(argv, returncode=1, stderr="slurmctld unavailable")

    monkeypatch.setattr(remote_dispatch.subprocess, "run", slurm)
    with pytest.raises(RuntimeError, match="query failed"):
        remote_dispatch.dispatch(tmp_path, name)
    assert not (tmp_path / "job.json").exists()


def test_rejected_submission_remains_recorded_and_not_blindly_retried(monkeypatch, tmp_path):
    calls = []

    def slurm(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "sbatch":
            return completed(argv, returncode=1, stderr="Invalid partition")
        return completed(argv)

    monkeypatch.setattr(remote_dispatch.subprocess, "run", slurm)
    with pytest.raises(RuntimeError, match="did not acknowledge"):
        remote_dispatch.dispatch(tmp_path, "vsa-rejected-relax")
    assert json.loads((tmp_path / "submission_error.json").read_text())["stderr"] == "Invalid partition"
    with pytest.raises(RuntimeError, match="outcome uncertain"):
        remote_dispatch.dispatch(tmp_path, "vsa-rejected-relax")
    assert [call[0] for call in calls].count("sbatch") == 1


def test_concurrent_dispatchers_share_one_receipt(monkeypatch, tmp_path):
    calls = []

    def slurm(argv, **kwargs):
        calls.append(argv)
        assert argv[0] == "sbatch"
        return completed(argv, "60721\n")

    monkeypatch.setattr(remote_dispatch.subprocess, "run", slurm)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(remote_dispatch.dispatch, tmp_path, "vsa-concurrent-relax") for _ in range(2)]
        values = [future.result(timeout=5) for future in futures]
    assert values[0] == values[1]
    assert len(calls) == 1


def test_reconcile_only_never_submits_without_recorded_intent(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("Reconciliation without an intent must not invoke Slurm")

    monkeypatch.setattr(remote_dispatch.subprocess, "run", forbidden)
    with pytest.raises(RuntimeError, match="cannot submit a new job"):
        remote_dispatch.dispatch(tmp_path, "vsa-recover-only-relax", reconcile_only=True)
    assert not (tmp_path / "submit_intent.json").exists()
    assert not (tmp_path / "job.json").exists()


def test_reconcile_only_returns_original_receipt_without_slurm(monkeypatch, tmp_path):
    value = {"job_name": "vsa-recover-only-relax", "job_id": "11721", "reconciled": False}
    (tmp_path / "job.json").write_text(json.dumps(value))

    def forbidden(*args, **kwargs):
        pytest.fail("A durable receipt already identifies the original job")

    monkeypatch.setattr(remote_dispatch.subprocess, "run", forbidden)
    assert remote_dispatch.dispatch(tmp_path, value["job_name"], reconcile_only=True) == value


def test_reconcile_only_reconstructs_existing_intent(monkeypatch, tmp_path):
    name = "vsa-reconcile-only-relax"
    (tmp_path / "submit_intent.json").write_text(json.dumps({"job_name": name, "date": "2026-01-01"}))

    def query(argv, **kwargs):
        assert argv[0] in {"squeue", "sacct"}
        return completed(argv, f"22721|{name}|\n")

    monkeypatch.setattr(remote_dispatch.subprocess, "run", query)
    result = remote_dispatch.dispatch(tmp_path, name, reconcile_only=True)
    assert result == {"job_id": "22721", "job_name": name, "reconciled": True}
    assert json.loads((tmp_path / "job.json").read_text()) == result


@pytest.mark.parametrize("record", ["job.json", "submit_intent.json"])
def test_recovery_rejects_other_runs_record(monkeypatch, tmp_path, record):
    (tmp_path / record).write_text(json.dumps({"job_name": "a-different-run", "job_id": "33721", "date": "2026-01-01"}))

    def forbidden(*args, **kwargs):
        pytest.fail("Mismatched records must not cause any Slurm action")

    monkeypatch.setattr(remote_dispatch.subprocess, "run", forbidden)
    with pytest.raises(RuntimeError, match="does not match this run"):
        remote_dispatch.dispatch(tmp_path, "vsa-expected-relax", reconcile_only=True)
