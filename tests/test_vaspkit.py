"""VASPKIT exports preserve calculation files and report failures accurately."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from vasp_slurm_agent import vaspkit


@pytest.fixture
def outputs(tmp_path):
    root = tmp_path / "completed run"
    root.mkdir()
    (root / "INCAR").write_text("ISPIN = 1\nLORBIT = 11\n")
    (root / "POSCAR").write_text("Si\n1\n1 0 0\n0 1 0\n0 0 1\nSi\n1\nDirect\n0 0 0\n")
    (root / "KPOINTS").write_text("Path\n20\nLine-mode\nReciprocal\n0 0 0 ! G\n0.5 0 0 ! X\n")
    (root / "DOSCAR").write_text("1\nheader\nheader\nheader\nheader\n10 -10 2 5.5 1\n-10 0 0\n10 1 2\n")
    (root / "EIGENVAL").write_text("fixture\n")
    (root / "PROCAR").write_text("fixture\n")
    return root


@pytest.fixture
def fake_run(monkeypatch):
    def execute(executable, folder, stdin, timeout, environment):
        task = stdin.splitlines()[0]
        names = {"211": "BAND.dat", "213": "PBAND_Si.dat", "111": "TDOS.dat", "113": "PDOS_Si.dat"}
        if folder.name.startswith("segment_"):
            count = int((folder / "KPOINTS").read_text().splitlines()[1])
            data = "# Energy density\n# Band-Index: 1\n" + "".join(f"{index / (count - 1)} {index}\n" for index in range(count))
        else:
            data = "# Energy density\n0 1\n1 2\n"
        (folder / names[task]).write_text(data)
        (folder / "stdout.txt").write_text("VASPKIT Standard Edition 1.5.1\n")
        (folder / "stderr.txt").write_text("")
        assert Path(environment["HOME"]).parent == (folder.parent if folder.name.startswith("segment_") else folder)
        assert "PLOT_MATPLOTLIB .FALSE." in (Path(environment["HOME"]) / ".vaspkit").read_text()
        return 0, ""
    monkeypatch.setattr(vaspkit, "_run", execute)
    return execute


def test_tasks_match_documented_noninteractive_commands(outputs):
    result = vaspkit.available_tasks(outputs)
    assert {key: value["task_id"] for key, value in result.items()} == {
        "bands": 211, "projected_bands": 213, "total_dos": 111, "projected_dos": 113,
    }
    assert all(value["available"] for value in result.values())


def test_export_preserves_inputs_and_records_reference(outputs, tmp_path, fake_run):
    original = {item.name: item.read_bytes() for item in outputs.iterdir()}
    destination = tmp_path / "exports"
    result = vaspkit.run_vaspkit(outputs, destination, list(vaspkit.TASKS), executable=sys.executable, fermi_energy_ev=4.2)
    assert result["status"] == "complete"
    assert result["energy_reference"] == "provided_fermi"
    assert {item.name: item.read_bytes() for item in outputs.iterdir()} == original
    assert (destination / "bands" / "FERMI_ENERGY.in").read_text().splitlines()[1] == "4.2"
    for task in result["tasks"]:
        assert task["version"] == "1.5.1"
        assert task["inputs"]["DOSCAR"] == task["prepared_inputs"]["DOSCAR"]
        assert all(len(item["sha256"]) == 64 for item in task["outputs"].values())
        assert task["stdin"] == str(task["task_id"]) + "\n0\n"
    assert json.loads((destination / "receipt.json").read_text()) == result


@pytest.mark.parametrize("setting,reason", [("LSORBIT=.TRUE.", "SOC"), ("LNONCOLLINEAR=T", "SOC"), ("LHFCALC = .TRUE.", "non-hybrid")])
def test_unsupported_band_formats_do_not_launch(outputs, tmp_path, monkeypatch, setting, reason):
    with (outputs / "INCAR").open("a") as target:
        target.write(setting + "\n")
    monkeypatch.setattr(vaspkit, "_run", lambda *args: pytest.fail("Unsupported format was launched"))
    result = vaspkit.run_vaspkit(outputs, tmp_path / "exports", executable=sys.executable)
    assert result["status"] == "unsupported"
    assert reason in result["tasks"][0]["reason"]


def test_projected_require_projection_settings_and_files(outputs):
    (outputs / "INCAR").write_text("ISPIN=1\n")
    result = vaspkit.available_tasks(outputs)
    assert result["bands"]["available"]
    assert not result["projected_dos"]["available"]
    assert "LORBIT" in result["projected_dos"]["reason"]
    (outputs / "PROCAR").unlink()
    assert "PROCAR" in vaspkit.available_tasks(outputs)["projected_bands"]["reason"]


def test_missing_binary_has_actionable_receipt(outputs, tmp_path):
    result = vaspkit.run_vaspkit(outputs, tmp_path / "exports", executable=str(tmp_path / "not-installed"))
    assert result["status"] == "unavailable"
    assert "Install" in result["reason"]
    assert (tmp_path / "exports" / "receipt.json").is_file()


@pytest.mark.parametrize("tasks", [("103",), ("bands", "bands"), ("../bands",), (), "bands"])
def test_task_allowlist_rejected_before_creating_files(outputs, tmp_path, tasks):
    with pytest.raises(ValueError, match="unique VASPKIT tasks"):
        vaspkit.run_vaspkit(outputs, tmp_path / "exports", tasks=tasks)
    assert not (tmp_path / "exports").exists()


@pytest.mark.parametrize("timeout", [0, 601, float("nan"), True])
def test_timeout_limits(outputs, tmp_path, timeout):
    with pytest.raises(ValueError, match="timeout"):
        vaspkit.run_vaspkit(outputs, tmp_path / "exports", timeout=timeout)


def test_existing_output_is_not_overwritten(outputs, tmp_path):
    target = tmp_path / "exports"
    target.mkdir()
    (target / "receipt.json").write_text("old result")
    with pytest.raises(FileExistsError):
        vaspkit.run_vaspkit(outputs, target)
    assert (target / "receipt.json").read_text() == "old result"


def test_no_files_is_failure_even_on_zero_exit(outputs, tmp_path, monkeypatch, fake_run):
    def empty(*args):
        result = fake_run(*args)
        (args[1] / "BAND.dat").unlink()
        return result
    monkeypatch.setattr(vaspkit, "_run", empty)
    result = vaspkit.run_vaspkit(outputs, tmp_path / "exports", executable=sys.executable)
    assert result["status"] == "failed"
    assert "expected data" in result["tasks"][0]["reason"]


@pytest.mark.parametrize("data", ["NaN 2\n", "hello world\n", "# nothing\n"])
def test_invalid_numeric_output_is_not_success(outputs, tmp_path, monkeypatch, fake_run, data):
    def invalid(*args):
        result = fake_run(*args)
        (args[1] / "BAND.dat").write_text(data)
        return result
    monkeypatch.setattr(vaspkit, "_run", invalid)
    assert vaspkit.run_vaspkit(outputs, tmp_path / "exports", executable=sys.executable)["status"] == "failed"


def test_mixed_success_is_partial(outputs, tmp_path, fake_run):
    (outputs / "PROCAR").unlink()
    result = vaspkit.run_vaspkit(outputs, tmp_path / "exports", ("bands", "projected_bands"), executable=sys.executable)
    assert result["status"] == "partial"
    assert [task["status"] for task in result["tasks"]] == ["complete", "unsupported"]


def test_timeout_keeps_partial_logs(tmp_path):
    folder = tmp_path / "worker"
    folder.mkdir()
    code, error = vaspkit._run(sys.executable, folder, "import time; print('started', flush=True); time.sleep(10)\n", 1, os.environ.copy())
    assert code != 0
    assert "timed out" in error
    assert "started" in (folder / "stdout.txt").read_text()


@pytest.mark.skipif(os.name == "nt", reason="POSIX remote cluster protocol")
def test_remote_helper_runs_without_installing_dft_agent(outputs, tmp_path):
    binary = tmp_path / "vaspkit"
    binary.write_text("#!/usr/bin/env python3\nfrom pathlib import Path\nimport sys\n"
                      "assert sys.stdin.readline().strip() == '211'\n"
                      "Path('BAND.dat').write_text('#K Energy\\n0 -1\\n1 1\\n')\n"
                      "print('VASPKIT 1.5.1')\n")
    binary.chmod(0o700)
    command = vaspkit.remote_command(outputs, tmp_path / "remote exports", ["bands"], executable=str(binary))
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(command, shell=True, cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout.split(vaspkit.RECEIPT_PREFIX)[-1])
    assert receipt["status"] == "complete"
    assert receipt["tasks"][0]["version"] == "1.5.1"


@pytest.mark.skipif(os.name == "nt", reason="Symlink creation needs extra Windows privileges")
def test_symlink_input_and_output_are_not_accepted(outputs, tmp_path, monkeypatch, fake_run):
    def linked(*args):
        result = fake_run(*args)
        path = args[1] / "BAND.dat"
        path.unlink()
        path.symlink_to(outputs / "DOSCAR")
        return result
    monkeypatch.setattr(vaspkit, "_run", linked)
    receipt = vaspkit.run_vaspkit(outputs, tmp_path / "exports", executable=sys.executable)
    assert receipt["status"] == "failed"
    assert not receipt["tasks"][0]["outputs"]
    (outputs / "PROCAR").unlink()
    (outputs / "PROCAR").symlink_to(outputs / "DOSCAR")
    assert not vaspkit.available_tasks(outputs)["projected_bands"]["available"]


@pytest.fixture
def explicit_outputs(outputs):
    points = [[0, 0, 0], [0.25, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0]]
    path = {"reciprocal_fractional": points, "labels": ["GAMMA", "", "X", "M"], "segments": [[0, 3], [2, 4]],
            "distance_inv_angstrom": [0, 0.25, 0.5, 1.0]}
    (outputs / "metadata.json").write_text(json.dumps({"band_path": path}))
    (outputs / "KPOINTS").write_text("Explicit\n4\nReciprocal\n" + "".join(" ".join(map(str, point)) + " 1\n" for point in points))
    eigen = "1 1 1 2\nheader\nheader\nheader\nheader\n2 4 2\n"
    procar = "PROCAR\n"
    for spin in (1, 2):
        procar += "# of k-points: 4 # of bands: 2 # of ions: 1\n"
        for index, point in enumerate(points):
            procar += "\nk-point " + str(index + 1) + " : " + " ".join(map(str, point)) + " weight = 1\n"
            procar += f"band 1 # energy {index + spin}.0 # occ. 1\nion s p d tot\n1 0.1 0.2 0.3 0.6\n"
    for index, point in enumerate(points):
        eigen += "\n" + " ".join(map(str, point)) + f" 1\n1 {index}.1 {index}.2 1 1\n2 {index}.3 {index}.4 0 0\n"
    (outputs / "EIGENVAL").write_text(eigen)
    (outputs / "PROCAR").write_text(procar)
    return outputs, path


def test_explicit_path_duplicates_only_existing_samples(explicit_outputs, tmp_path, fake_run):
    outputs, path = explicit_outputs
    original = (outputs / "EIGENVAL").read_bytes()
    destination = tmp_path / "exports"
    result = vaspkit.run_vaspkit(outputs, destination, ["projected_bands"], executable=sys.executable)
    assert result["status"] == "complete"
    folder = destination / "projected_bands"
    mapping = json.loads((folder / "kpoint_mapping.json").read_text())
    assert mapping["original_kpoint_indices_zero_based"] == [0, 1, 2, 2, 3]
    assert mapping["interpolation"] == "none"
    assert (folder / "segment_001/KPOINTS").read_text().splitlines()[1:3] == ["3", "Line-mode"]
    eigen = (folder / "segment_001/EIGENVAL").read_text()
    assert eigen.splitlines()[5] == "2 3 2"
    assert eigen.count("1 1.1 1.2 1 1") == 1
    assert eigen.count("2 2.3 2.4 0 0") == 1
    procar = (folder / "segment_002/PROCAR").read_text()
    assert procar.count("# of k-points: 2") == 2
    assert procar.count("k-point 2 :") == 2
    assert procar.endswith("\n\n")
    assert (outputs / "EIGENVAL").read_bytes() == original
    assert "kpoint_mapping.json" in result["tasks"][0]["logs"]


def test_explicit_path_requires_matching_eigenvalues(explicit_outputs, tmp_path, fake_run):
    outputs, path = explicit_outputs
    data = (outputs / "EIGENVAL").read_text().replace("0.25 0 0 1", "0.125 0 0 1")
    (outputs / "EIGENVAL").write_text(data)
    result = vaspkit.run_vaspkit(outputs, tmp_path / "exports", executable=sys.executable)
    assert result["status"] == "failed"
    assert "coordinates do not match" in result["tasks"][0]["reason"]


def test_remote_explicit_path_can_receive_metadata_argument(explicit_outputs, tmp_path, fake_run):
    outputs, path = explicit_outputs
    (outputs / "metadata.json").unlink()
    assert not vaspkit.available_tasks(outputs)["bands"]["available"]
    result = vaspkit.run_vaspkit(outputs, tmp_path / "exports", executable=sys.executable, band_path=path)
    assert result["status"] == "complete"


def test_irregular_explicit_segment_is_rejected_before_running(explicit_outputs, tmp_path, monkeypatch):
    outputs, path = explicit_outputs
    path["reciprocal_fractional"][1] = [0.125, 0, 0]
    monkeypatch.setattr(vaspkit, "_run", lambda *args: pytest.fail("Invalid path was launched"))
    result = vaspkit.run_vaspkit(outputs, tmp_path / "exports", executable=sys.executable, band_path=path)
    assert result["status"] == "unsupported"
    assert "evenly sampled" in result["tasks"][0]["reason"]


def test_segment_merge_preserves_energy_columns_and_breaks(tmp_path):
    path = {"labels": ["GAMMA", "X", "M"], "segments": [[0, 2], [1, 3]], "distance_inv_angstrom": [0, 0.5, 1.5]}
    children = [tmp_path / "one", tmp_path / "two"]
    for folder, data in zip(children, ["#k E\n# Band-Index: 1\n0 3\n0.5 4\n# Band-Index: 2\n0.5 6\n0 5\n",
                                       "#k E\n# Band-Index: 1\n0 4\n1 7\n# Band-Index: 2\n1 8\n0 6\n"]):
        folder.mkdir()
        (folder / "BAND.dat").write_text(data)
    vaspkit._merge_segments(tmp_path, children, path, r"BAND\.dat")
    result = vaspkit._band_blocks(tmp_path / "BAND.dat")
    assert [[float(row[1]) for row in block] for block in result] == [[3, 4, 4, 7], [5, 6, 6, 8]]
    assert [[float(row[0]) for row in block] for block in result] == [[0, 0.5, 0.5, 1.5]] * 2
    assert "\n\n0.5 4\n" in (tmp_path / "BAND.dat").read_text()
