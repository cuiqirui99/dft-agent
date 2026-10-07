"""Exercise user actions through Streamlit; all remote execution is stubbed."""

from importlib.resources import files
import json
from pathlib import Path
from unittest.mock import Mock

from pymatgen.core import Structure
import pytest
from streamlit.testing.v1 import AppTest

from vasp_slurm_agent import workflow
from vasp_slurm_agent.config import ClusterConfig


def widget(elements, label):
    return next(element for element in elements if element.label == label)


@pytest.fixture
def workbench(tmp_path, monkeypatch):
    config = ClusterConfig(
        host="cluster.example.invalid", user="researcher", remote_root="/scratch/test",
        vasp_command="srun vasp_std", potcar_root="/licensed/pbe", partition="cpu",
    )
    config_path = config.save(tmp_path / "cluster.json")
    monkeypatch.setenv("VASP_AGENT_CONFIG", str(config_path))

    def no_remote(*args, **kwargs):
        pytest.fail("UI smoke tests must never contact an SSH server")

    monkeypatch.setattr(workflow, "SSHTransport", no_remote)
    worker = Mock(return_value=43210)
    monkeypatch.setattr(workflow, "start_worker", worker)
    app = AppTest.from_file(str(files("vasp_slurm_agent").joinpath("app.py")), default_timeout=20).run()
    widget(app.text_input, "任务保存目录").set_value(str(tmp_path / "runs")).run()
    assert not app.exception
    return app, worker, tmp_path / "runs"


def silicon_bytes():
    return files("vasp_slurm_agent").joinpath("examples", "Si.cif").read_bytes()


@pytest.mark.parametrize("file_format", ["cif", "poscar"])
def test_structure_import_and_review_gate(workbench, file_format):
    app, worker, runs_root = workbench
    assert widget(app.button, "生成输入并预览").disabled
    if file_format == "cif":
        filename, payload = "Si.cif", silicon_bytes()
    else:
        structure = Structure.from_str(silicon_bytes().decode(), fmt="cif")
        filename, payload = "POSCAR", structure.to(fmt="poscar").encode()
    app.file_uploader[0].upload(filename, payload).run()
    assert not app.exception
    assert widget(app.metric, "化学式").value == "Si"
    assert not widget(app.button, "生成输入并预览").disabled
    assert not runs_root.exists()
    worker.assert_not_called()

    widget(app.button, "生成输入并预览").click().run()
    assert not app.exception
    run_dir = Path(app.session_state["active_run"])
    state = workflow.read_state(run_dir)
    assert state["status"] == "planned"
    assert state["stages"][0]["job_id"] is None
    assert (run_dir / "01_relax" / "inputs" / "POSCAR").is_file()
    assert widget(app.button, "确认并提交").disabled
    worker.assert_not_called()

    widget(app.checkbox, "已检查结构、计算参数、集群账户和每阶段资源设置").check().run()
    assert not widget(app.button, "确认并提交").disabled
    worker.assert_not_called()
    widget(app.button, "确认并提交").click().run()
    assert not app.exception
    worker.assert_called_once_with(run_dir)
    assert any("43210" in message.value for message in app.success)


def test_invalid_structure_cannot_be_prepared(workbench):
    app, worker, runs_root = workbench
    app.file_uploader[0].upload("POSCAR", b"This is not a crystal structure").run()
    assert not app.exception
    assert any("结构无法解析" in message.value for message in app.error)
    assert widget(app.button, "生成输入并预览").disabled
    assert not runs_root.exists()
    worker.assert_not_called()


@pytest.mark.parametrize("accepted", [True, False])
def test_completed_result_is_visible_without_resubmission(workbench, accepted):
    app, worker, _ = workbench
    app.file_uploader[0].upload("Si.cif", silicon_bytes()).run()
    widget(app.button, "生成输入并预览").click().run()
    run_dir = Path(app.session_state["active_run"])
    state = workflow.read_state(run_dir)
    state["status"] = "succeeded" if accepted else "needs_attention"
    stage = state["stages"][0]
    stage.update(status=state["status"], job_id="12345", result={
        "success": accepted, "reason": "Electronic convergence was not reached." if not accepted else "Accepted test fixture",
        "final_energy_ev": -10.123456, "final_max_force_ev_angstrom": 0.002,
    })
    (run_dir / "run.json").write_text(json.dumps(state))
    output = run_dir / stage["folder"] / "outputs"
    output.mkdir()
    # Even an existing file must not appear as an accepted result after refusal.
    (output / "final_structure.cif").write_bytes(silicon_bytes())
    app.run()
    assert not app.exception
    assert not any(button.label == "确认并提交" for button in app.button)
    if accepted:
        assert widget(app.metric, "最终总能量 / eV").value == "-10.123456"
        assert widget(app.download_button, "下载最终结构 / CIF").proto.url
        assert any("Si" in item.value and "个原子" in item.value for item in app.markdown)
    else:
        assert any("Electronic convergence" in item.value for item in app.warning)
        assert not any(button.label == "下载最终结构 / CIF" for button in app.download_button)
    widget(app.button, "打包当前结果").click().run()
    assert not app.exception
    assert (run_dir / "results.zip").is_file()
    assert widget(app.download_button, "下载 ZIP").proto.url
    worker.assert_not_called()


def test_installed_examples_are_readable_crystal_inputs():
    examples = files("vasp_slurm_agent").joinpath("examples")
    expected = {"Si", "C", "Ge", "Al", "Cu", "MgO", "NaCl", "SiC"}
    structures = {sample.name[:-4]: Structure.from_str(sample.read_text(), fmt="cif")
                  for sample in examples.iterdir() if sample.name.endswith(".cif")}
    assert structures.keys() == expected
    for structure in structures.values():
        assert structure.is_ordered and len(structure) > 0 and structure.volume > 0


def test_paused_job_can_reconnect_from_the_result_panel(workbench):
    app, worker, _ = workbench
    app.file_uploader[0].upload("Si.cif", silicon_bytes()).run()
    widget(app.button, "生成输入并预览").click().run()
    run_dir = Path(app.session_state["active_run"])
    state = workflow.read_state(run_dir)
    state.update(status="needs_attention", last_error="Connection unavailable", remote_failures=3)
    state["stages"][0].update(status="running", job_id="12345", staged=True)
    (run_dir / "run.json").write_text(json.dumps(state))
    app.run()
    worker.assert_not_called()
    widget(app.button, "重新连接 / 回收").click().run()
    assert not app.exception
    worker.assert_called_once_with(run_dir)
    restored = workflow.read_state(run_dir)
    assert restored["stages"][0]["job_id"] == "12345"
    assert restored["status"] not in workflow.TERMINAL


@pytest.mark.parametrize("task", ["bands", "dos"])
@pytest.mark.parametrize("scf_fermi", [None, 5.25])
def test_spectral_results_use_scf_reference_not_ground_state_metrics(workbench, task, scf_fermi):
    app, worker, _ = workbench
    app.file_uploader[0].upload("Si.cif", silicon_bytes()).run()
    widget(app.selectbox, "任务").select(task)
    widget(app.button, "生成输入并预览").click().run()
    run_dir = Path(app.session_state["active_run"])
    state = workflow.read_state(run_dir)
    assert state["task"] == task
    state.update(status="succeeded", current_stage=1)
    state["stages"][0].update(status="succeeded", job_id="12345", result={
        "success": True, "final_energy_ev": -10.0, "final_max_force_ev_angstrom": 0.002,
        "fermi_energy_ev": scf_fermi,
    })
    state["stages"][1].update(status="succeeded", job_id="12346", result={
        "success": True, "final_energy_ev": 123.0, "final_max_force_ev_angstrom": 99.0,
        "fermi_energy_ev": 8.88,
    })
    state_path = run_dir / "run.json"
    state_path.write_text(json.dumps(state))
    original = state_path.read_bytes()
    app.run()
    assert not app.exception
    assert [item.value for item in app.metric if item.label == "最终总能量 / eV"] == ["-10.000000"]
    assert [item.value for item in app.metric if item.label == "最大原子力 / eV Å⁻¹"] == ["0.002000"]
    references = [item.value for item in app.metric if item.label == "前序 SCF 费米能 / eV"]
    assert references == (["5.250000"] if scf_fermi is not None else [])
    assert any("固定电荷谱计算" in item.value for item in app.caption)
    assert state_path.read_bytes() == original
    worker.assert_not_called()


@pytest.mark.parametrize("scheduler_state", [None, "RUNNING", "COMPLETED", "CANCELLED", "FAILED"])
def test_paused_job_can_request_cancel_only_before_confirmed_scheduler_end(workbench, monkeypatch, scheduler_state):
    app, worker, _ = workbench
    app.file_uploader[0].upload("Si.cif", silicon_bytes()).run()
    widget(app.button, "生成输入并预览").click().run()
    run_dir = Path(app.session_state["active_run"])
    state = workflow.read_state(run_dir)
    state.update(status="needs_attention", last_error="Connection unavailable", remote_failures=3)
    state["stages"][0].update(status="running", job_id="12345", staged=True, scheduler_state=scheduler_state)
    state_path = run_dir / "run.json"
    state_path.write_text(json.dumps(state))
    original = state_path.read_bytes()
    cancellation = Mock(return_value=state)
    monkeypatch.setattr(workflow, "cancel", cancellation)
    app.run()
    assert not app.exception
    if scheduler_state in {None, "RUNNING"}:
        widget(app.button, "取消此任务").click().run()
        assert not app.exception
        cancellation.assert_called_once_with(run_dir)
    else:
        assert not any(button.label == "取消此任务" for button in app.button)
        cancellation.assert_not_called()
    assert state_path.read_bytes() == original
    worker.assert_not_called()
