"""Additional output collection keeps accepted evidence and private files safe."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import zipfile

import pytest
from streamlit.testing.v1 import AppTest

from vasp_slurm_agent import postprocessing, workflow
from vasp_slurm_agent.transport import Result


def metadata(data):
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


class LocalOutputTransport:
    """Execute only the inventory Python code against a temporary remote folder."""
    def __init__(self):
        self.commands = []
        self.downloads = []
        self.inventory_override = None
        self.corrupt = None

    def run(self, command, timeout=60):
        self.commands.append(command)
        if self.inventory_override is not None:
            return Result(0, json.dumps(self.inventory_override))
        args = shlex.split(command)
        assert args[:2] == ["python3", "-c"] and len(args) == 3
        result = subprocess.run([sys.executable, "-c", args[2]], capture_output=True, text=True, timeout=timeout)
        return Result(result.returncode, result.stdout, result.stderr)

    def download_many(self, remote, local, names):
        self.downloads.append(list(names))
        fetched = {}
        for name in names:
            data = (Path(remote) / name).read_bytes()
            fetched[name] = metadata(data)
            (Path(local) / name).write_bytes(data + b"corruption" if self.corrupt == name else data)
        return fetched


@pytest.fixture
def saved_stage(tmp_path):
    root, remote = tmp_path / "run", tmp_path / "remote"
    output = root / "01_scf" / "outputs"
    output.mkdir(parents=True)
    (root / "01_scf" / "inputs").mkdir()
    (remote / "01_scf").mkdir(parents=True)
    stage = {"folder": "01_scf", "name": "scf", "job_id": "101", "scheduler_state": "FAILED",
             "status": "failed", "result": {"success": False, "reason": "Electronic convergence was not reached."}}
    (root / "run.json").write_text(json.dumps({"status": "needs_attention", "remote_root": str(remote),
                                              "stages": [stage], "history": []}))
    return root, remote / "01_scf", output, LocalOutputTransport()


def make_symlink(link, source):
    try:
        link.symlink_to(source, target_is_directory=source.is_dir())
    except OSError:
        pytest.skip("This host does not allow creating symlinks.")


def test_inventory_lists_only_approved_regular_output_files(saved_stage):
    root, remote, _, wire = saved_stage
    for name, data in {"OUTCAR": b"error details", "CHGCAR": b"charge", "PROCAR": b"projection",
                       "POTCAR": b"licensed data", "secret.txt": b"private"}.items():
        (remote / name).write_bytes(data)
    rows = postprocessing.list_remote_outputs(root, "01_scf", wire)
    assert {row["name"]: row["bytes"] for row in rows} == {"OUTCAR": 13, "CHGCAR": 6, "PROCAR": 10}
    assert not wire.downloads


def test_inventory_excludes_symlink_targets(saved_stage, tmp_path):
    root, remote, _, wire = saved_stage
    outside = tmp_path / "private"
    outside.write_bytes(b"secret")
    make_symlink(remote / "OUTCAR", outside)
    assert postprocessing.list_remote_outputs(root, "01_scf", wire) == []


@pytest.mark.parametrize("rows", [
    [{"name": "POTCAR", "bytes": 10}], [{"name": "../OUTCAR", "bytes": 10}],
    [{"name": "OUTCAR", "bytes": -1}], [{"name": "OUTCAR", "bytes": True}],
    [{"name": "OUTCAR", "bytes": 1}, {"name": "OUTCAR", "bytes": 1}], {"OUTCAR": 1},
])
def test_inventory_rejects_malformed_or_unapproved_remote_rows(saved_stage, rows):
    root, _, _, wire = saved_stage
    wire.inventory_override = rows
    with pytest.raises(ValueError, match="Invalid remote file list"):
        postprocessing.list_remote_outputs(root, "01_scf", wire)


@pytest.mark.parametrize("names", [["POTCAR"], ["../OUTCAR"], ["/OUTCAR"], ["OUTCAR", "OUTCAR"], []])
def test_unapproved_collection_refused_before_remote_access(saved_stage, names):
    root, _, _, wire = saved_stage
    with pytest.raises(ValueError, match="Select output files"):
        postprocessing.fetch_outputs(root, "01_scf", names, wire)
    assert not wire.commands and not wire.downloads


def test_missing_selected_file_does_not_change_saved_results(saved_stage):
    root, _, output, wire = saved_stage
    before = (root / "run.json").read_bytes()
    with pytest.raises(ValueError, match="no longer available"):
        postprocessing.fetch_outputs(root, "01_scf", ["OUTCAR"], wire)
    assert (root / "run.json").read_bytes() == before
    assert not (output / "artifact_manifest.json").exists() and not wire.downloads


@pytest.mark.parametrize("state", sorted(workflow.SLURM_TERMINAL))
def test_every_terminal_job_can_return_diagnostic_files(saved_stage, state):
    root, remote, output, wire = saved_stage
    saved = workflow.read_state(root)
    saved["stages"][0]["scheduler_state"] = state
    (root / "run.json").write_text(json.dumps(saved))
    data = b"VASP diagnostic log\n"
    (remote / "OUTCAR").write_bytes(data)
    fetched = postprocessing.fetch_outputs(root, "01_scf", ["OUTCAR"], wire)
    assert fetched == {"OUTCAR": metadata(data)}
    assert (output / "OUTCAR").read_bytes() == data
    after = workflow.read_state(root)
    assert after["status"] == saved["status"] and after["stages"] == saved["stages"]
    assert "Collected additional files" in after["history"][-1]["event"]


def test_running_job_cannot_be_collected_as_final_evidence(saved_stage):
    root, _, _, wire = saved_stage
    saved = workflow.read_state(root)
    saved["stages"][0]["scheduler_state"] = "RUNNING"
    (root / "run.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="Wait for the stage to finish"):
        postprocessing.list_remote_outputs(root, "01_scf", wire)
    assert not wire.commands


def test_remote_changed_file_preserves_accepted_copy_manifest_and_state(saved_stage):
    root, remote, output, wire = saved_stage
    accepted = b"accepted VASP output"
    (output / "OUTCAR").write_bytes(accepted)
    (output / "artifact_manifest.json").write_text(json.dumps({"OUTCAR": metadata(accepted)}))
    (remote / "OUTCAR").write_bytes(b"different remote output")
    before = {path: path.read_bytes() for path in [root / "run.json", output / "artifact_manifest.json", output / "OUTCAR"]}
    with pytest.raises(ValueError, match="changed since collection"):
        postprocessing.fetch_outputs(root, "01_scf", ["OUTCAR"], wire)
    assert {path: path.read_bytes() for path in before} == before
    assert not list(output.glob(".extra-*"))


def test_corrupt_download_publishes_no_partial_batch(saved_stage):
    root, remote, output, wire = saved_stage
    accepted = b"accepted output"
    for folder in (remote, output):
        (folder / "OUTCAR").write_bytes(accepted)
    (remote / "PROCAR").write_bytes(b"new projection")
    (output / "artifact_manifest.json").write_text(json.dumps({"OUTCAR": metadata(accepted)}))
    before = {path: path.read_bytes() for path in [root / "run.json", output / "artifact_manifest.json", output / "OUTCAR"]}
    wire.corrupt = "PROCAR"
    with pytest.raises(ValueError, match="Checksum mismatch"):
        postprocessing.fetch_outputs(root, "01_scf", ["OUTCAR", "PROCAR"], wire)
    assert {path: path.read_bytes() for path in before} == before
    assert not (output / "PROCAR").exists() and not list(output.glob(".extra-*"))


def test_verified_download_extends_manifest_without_changing_existing_entry(saved_stage):
    root, remote, output, wire = saved_stage
    accepted = b"accepted output"
    (output / "OUTCAR").write_bytes(accepted)
    (output / "artifact_manifest.json").write_text(json.dumps({"OUTCAR": metadata(accepted)}))
    data = b"wavefunction bytes\x00\xff"
    (remote / "WAVECAR").write_bytes(data)
    postprocessing.fetch_outputs(root, "01_scf", ["WAVECAR"], wire)
    assert (output / "OUTCAR").read_bytes() == accepted
    assert json.loads((output / "artifact_manifest.json").read_text()) == {"OUTCAR": metadata(accepted), "WAVECAR": metadata(data)}


def test_zip_includes_nested_exports_and_omits_potcar_and_staging(saved_stage):
    root, _, output, _ = saved_stage
    nested = output / "vaspkit" / "abc" / "projected_dos"
    nested.mkdir(parents=True)
    (nested / "PDOS_O.dat").write_bytes(b"numerical projection")
    (nested / "POTCAR").write_bytes(b"licensed")
    (output / ".extra-incomplete").mkdir()
    (output / ".extra-incomplete" / "OUTCAR").write_bytes(b"partial")
    (output / "PROCAR").write_bytes(b"raw projection")
    with zipfile.ZipFile(workflow.bundle_run(root)) as archive:
        names = set(archive.namelist())
        assert "01_scf/outputs/vaspkit/abc/projected_dos/PDOS_O.dat" in names
        assert "01_scf/outputs/PROCAR" in names
        assert all("POTCAR" not in name and ".extra-" not in name for name in names)


def test_zip_omits_file_and_directory_symlinks_in_outputs_and_sources(saved_stage, tmp_path):
    root, _, output, _ = saved_stage
    outside = tmp_path / "private"
    outside.mkdir()
    (outside / "token").write_bytes(b"private credential")
    make_symlink(output / "OUTCAR", outside / "token")
    make_symlink(output / "nested", outside)
    (root / "source").mkdir()
    (root / "plan.json").write_text("{}")
    make_symlink(root / "source" / "source.cif", outside / "token")
    with zipfile.ZipFile(workflow.bundle_run(root)) as archive:
        assert all(name not in archive.namelist() for name in ["01_scf/outputs/OUTCAR", "01_scf/outputs/nested/token", "source/source.cif"])
        assert all(b"private credential" not in archive.read(name) for name in archive.namelist())


def test_failed_stage_ui_still_exposes_raw_files_and_nested_exports(saved_stage, monkeypatch):
    from vasp_slurm_agent import app as app_module
    root, _, output, _ = saved_stage
    (output / "OUTCAR").write_text("failure details")
    (output / "POTCAR").write_text("licensed")
    nested = output / "vaspkit" / "trial"
    nested.mkdir(parents=True)
    (nested / "dos.csv").write_text("energy,dos\n0,1\n")
    monkeypatch.setattr(app_module, "has_config", lambda root: False)
    stage = workflow.read_state(root)["stages"][0]
    script = f"from pathlib import Path\nfrom vasp_slurm_agent.app import _stage_downloads\n_stage_downloads(Path({str(root)!r}), {stage!r})\n"
    app = AppTest.from_string(script).run()
    assert not app.exception
    assert not any(item.label == "Plot" for item in app.selectbox)
    files = next(item for item in app.selectbox if item.label == "File")
    assert files.options == ["OUTCAR", "vaspkit/trial/dos.csv"]
    assert next(item for item in app.download_button if item.label == "Download file").proto.url


def collect_fixture(saved_stage, monkeypatch, *, same_reference=True, export_error=False):
    from vasp_slurm_agent import vasp
    root, _, _, wire = saved_stage
    remote_root = Path(workflow.read_state(root)["remote_root"])
    state = {"status": "collecting", "current_stage": 2, "remote_root": str(remote_root), "history": [], "stages": [
        {"folder": "01_scf", "name": "scf", "status": "succeeded", "result": {"success": True, "fermi_energy_ev": 4.0}},
        {"folder": "02_bands", "name": "bands", "status": "succeeded", "charge_from": 0,
         "result": {"success": True, "final_energy_ev": -39.1, "artifacts": ["bands.csv"], "vasprun_sha256": "accepted-evidence"}},
        {"folder": "03_dos", "name": "dos", "status": "collecting", "scheduler_state": "COMPLETED", "job_id": "103",
         "charge_from": 0 if same_reference else 1, "metadata": {"input_sha256": {name: "a" * 64 for name in ("INCAR", "KPOINTS", "POSCAR")}}},
    ]}
    for stage in state["stages"]:
        (root / stage["folder"] / "outputs").mkdir(parents=True, exist_ok=True)
        (root / stage["folder"] / "inputs").mkdir(exist_ok=True)
    band_output = root / "02_bands" / "outputs"
    (band_output / "result.json").write_text(json.dumps(state["stages"][1]["result"]))
    (band_output / "vasprun.xml").write_bytes(b"accepted raw band evidence")
    (band_output / "artifact_manifest.json").write_text(json.dumps({"vasprun.xml": metadata(b"accepted raw band evidence")}))
    remote = remote_root / "03_dos"
    remote.mkdir()
    (remote / "input_hashes.sha256").write_text("\n".join("a" * 64 + "  " + name for name in ("INCAR", "KPOINTS", "POSCAR")))
    monkeypatch.setattr(workflow, "analyze_outputs", lambda *args: {"success": True, "reason": "Accepted", "final_energy_ev": -39.2, "artifacts": ["dos.csv"]})
    calls = []
    def export(bands_dir, dos_dir):
        calls.append((bands_dir, dos_dir))
        if export_error:
            raise ValueError("Mismatched spectrum metadata")
        (bands_dir / "bands_dos.png").write_bytes(b"derived plot")
        return ["bands_dos.png"]
    monkeypatch.setattr(vasp, "export_bands_dos", export)
    return root, state, wire, calls


def test_collection_registers_combined_plot_without_changing_accepted_raw_evidence(saved_stage, monkeypatch):
    root, state, wire, calls = collect_fixture(saved_stage, monkeypatch)
    bands = root / "02_bands" / "outputs"
    original = {name: (bands / name).read_bytes() for name in ("vasprun.xml", "artifact_manifest.json")}
    accepted = dict(state["stages"][1]["result"])
    after = workflow._collect(root, state, wire, state["stages"][2])
    assert calls == [(bands, root / "03_dos" / "outputs")]
    assert {name: (bands / name).read_bytes() for name in original} == original
    result = json.loads((bands / "result.json").read_text())
    assert result == {**accepted, "artifacts": ["bands.csv", "bands_dos.png"]}
    assert after["status"] == "succeeded" and after["stages"][2]["result"]["success"]


def test_collection_does_not_pair_spectra_from_different_scf_stages(saved_stage, monkeypatch):
    root, state, wire, calls = collect_fixture(saved_stage, monkeypatch, same_reference=False)
    accepted = (root / "02_bands" / "outputs" / "result.json").read_bytes()
    workflow._collect(root, state, wire, state["stages"][2])
    assert not calls
    assert (root / "02_bands" / "outputs" / "result.json").read_bytes() == accepted


def test_optional_combined_plot_error_does_not_reject_successful_calculation(saved_stage, monkeypatch):
    root, state, wire, calls = collect_fixture(saved_stage, monkeypatch, export_error=True)
    accepted = (root / "02_bands" / "outputs" / "result.json").read_bytes()
    after = workflow._collect(root, state, wire, state["stages"][2])
    assert len(calls) == 1 and after["status"] == "succeeded"
    assert after["stages"][2]["result"]["success"]
    assert "Mismatched spectrum metadata" in after["stages"][2]["analysis_warnings"][0]
    assert (root / "02_bands" / "outputs" / "result.json").read_bytes() == accepted


@pytest.mark.parametrize("fetch", [False, True])
def test_cli_files_routes_only_requested_action(saved_stage, monkeypatch, capsys, fetch):
    from vasp_slurm_agent import cli
    root, _, _, _ = saved_stage
    monkeypatch.setattr(cli, "migrate_legacy_config", lambda: None)
    calls = []
    def listing(path, stage):
        calls.append(("list", path, stage))
        return [{"name": "WAVECAR", "bytes": 8}]
    def collecting(path, stage, names):
        calls.append(("fetch", path, stage, names))
        return {"WAVECAR": metadata(b"wavedata")}
    monkeypatch.setattr(postprocessing, "list_remote_outputs", listing)
    monkeypatch.setattr(postprocessing, "fetch_outputs", collecting)
    args = ["dft-agent", "files", str(root), "01_scf"] + (["--fetch", "WAVECAR"] if fetch else [])
    monkeypatch.setattr(sys, "argv", args)
    assert cli.main() == 0
    assert calls == [("fetch", root, "01_scf", ["WAVECAR"])] if fetch else calls == [("list", root, "01_scf")]
    payload = json.loads(capsys.readouterr().out)
    assert "WAVECAR" in payload if fetch else payload == [{"name": "WAVECAR", "bytes": 8}]


def test_cli_files_rejects_potcar_with_clear_error_before_config_access(saved_stage, monkeypatch, capsys):
    from vasp_slurm_agent import cli
    root, _, _, _ = saved_stage
    monkeypatch.setattr(cli, "migrate_legacy_config", lambda: None)
    monkeypatch.setattr(sys, "argv", ["dft-agent", "files", str(root), "01_scf", "--fetch", "POTCAR"])
    assert cli.main() == 1
    assert "POTCAR is not exported" in capsys.readouterr().err


@pytest.mark.parametrize("status,exit_code", [("complete", 0), ("partial", 1), ("unavailable", 1)])
def test_cli_postprocess_reports_incomplete_analysis_as_nonzero(saved_stage, monkeypatch, capsys, status, exit_code):
    from vasp_slurm_agent import cli
    root, _, _, _ = saved_stage
    monkeypatch.setattr(cli, "migrate_legacy_config", lambda: None)
    calls = []
    def process(path, stage, tasks, executable):
        calls.append((path, stage, tasks, executable))
        return {"status": status}
    monkeypatch.setattr(postprocessing, "postprocess_run", process)
    monkeypatch.setattr(sys, "argv", ["dft-agent", "postprocess", str(root), "03_dos", "--tasks", "total_dos", "--executable", "/tools/my vaspkit"])
    assert cli.main() == exit_code
    assert calls == [(root, "03_dos", ["total_dos"], "/tools/my vaspkit")]
    assert json.loads(capsys.readouterr().out)["status"] == status


def test_dos_ui_identifies_its_own_fermi_reference(saved_stage):
    root, _, _, _ = saved_stage
    stage = {"name": "dos", "folder": "04_dos", "metadata": {"spectral_charge_mode": "fixed_charge"},
             "result": {"success": True, "fermi_energy_ev": 3.25699118, "energy_reference_ev": 3.25699118,
                        "energy_reference_source": "dos_run_fermi"}}
    script = f"from pathlib import Path\nfrom vasp_slurm_agent.app import _stage_results\n_stage_results(Path({str(root)!r}), {stage!r}, 3.45106126)\n"
    app = AppTest.from_string(script).run()
    assert not app.exception
    assert [(item.label, item.value) for item in app.metric] == [("DOS Fermi energy (eV)", "3.256991")]
