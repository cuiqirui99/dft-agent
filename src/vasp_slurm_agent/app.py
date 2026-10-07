"""Local, single-user Streamlit workbench for reviewed VASP/Slurm runs."""

from __future__ import annotations

from datetime import datetime
from itertools import product
import json
import os
from pathlib import Path
import tempfile
import uuid

from pymatgen.core import Structure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from vasp_slurm_agent.cli import doctor
from vasp_slurm_agent.config import ClusterConfig
from vasp_slurm_agent.workflow import (
    SLURM_TERMINAL,
    bundle_run,
    cancel,
    prepare_run,
    read_state,
    resume,
    start_worker,
)


TERMINAL = {"succeeded", "needs_attention", "failed", "cancelled"}
STATUS_LABELS = {
    "planned": "待确认提交",
    "submitting": "提交中",
    "queued": "排队中",
    "running": "计算中",
    "collecting": "收集结果",
    "succeeded": "计算流程完成",
    "needs_attention": "需要检查",
    "failed": "失败",
    "cancelled": "已取消",
}


def _with_password(function, *args):
    """Provide a session-only password to one local action and its worker."""
    previous = os.environ.get("DFT_AGENT_SSH_PASSWORD")
    password = st.session_state.get("ssh_password", "")
    if password:
        os.environ["DFT_AGENT_SSH_PASSWORD"] = password
    try:
        return function(*args)
    finally:
        if previous is None:
            os.environ.pop("DFT_AGENT_SSH_PASSWORD", None)
        else:
            os.environ["DFT_AGENT_SSH_PASSWORD"] = previous


def _load_config(path: Path) -> ClusterConfig | None:
    if not path.is_file():
        return None
    try:
        return ClusterConfig.load(path)
    except Exception as exc:
        st.error(f"无法读取集群配置：{exc}")
        return None


def _config_editor(path: Path, config: ClusterConfig | None) -> None:
    st.subheader("集群配置")
    st.caption("配置只保存在本机。密码不写入配置；需要密码时在侧栏填写，仅保留在当前会话内存。")

    def value(name: str, default):
        return getattr(config, name, default)

    with st.form("cluster_config"):
        left, right = st.columns(2)
        with left:
            host = st.text_input("SSH 主机", value=value("host", ""))
            user = st.text_input("SSH 用户", value=value("user", ""))
            port = st.number_input("SSH 端口", 1, 65535, int(value("port", 22)))
            remote_root = st.text_input("远端任务根目录（绝对路径）", value=value("remote_root", ""))
            potcar_root = st.text_input("远端 POTCAR 根目录", value=value("potcar_root", ""))
            vasp_command = st.text_input("VASP 启动命令", value=value("vasp_command", "srun vasp_std"))
        with right:
            partition = st.text_input("Slurm partition", value=value("partition", ""))
            account = st.text_input("Slurm account（可留空）", value=value("account", ""))
            tasks = st.number_input("MPI task 数", 1, 4096, int(value("tasks", 8)),
                                    help="小晶胞可先用 8 个 MPI task；核数更多不一定更快。按集群要求和体系大小调整。")
            walltime = st.text_input("每阶段最长运行时间", value=value("walltime", "00:30:00"))
            timeout = st.number_input("SSH 连接超时 / 秒", 1, 120, int(value("connect_timeout", 15)))
            setup = st.text_area("集群环境命令（每行一条）", value="\n".join(value("setup_commands", [])),
                                 placeholder="module load your-vasp-module")
        symbols = st.text_area("POTCAR 元素映射 / JSON", value=json.dumps(value("potcar_symbols", {}), ensure_ascii=False),
                               help='例如 {"Ti": "Ti_pv"}。请按已授权的赝势目录设置。')
        save = st.form_submit_button("保存集群配置", type="primary")
    if save:
        try:
            mapping = json.loads(symbols)
            if not isinstance(mapping, dict):
                raise ValueError("POTCAR 元素映射必须是 JSON 对象")
            candidate = ClusterConfig(
                host=host.strip(), user=user.strip(), remote_root=remote_root.strip(),
                vasp_command=vasp_command.strip(), potcar_root=potcar_root.strip(),
                partition=partition.strip(), port=int(port), tasks=int(tasks), walltime=walltime.strip(),
                account=account.strip(), setup_commands=[line for line in setup.splitlines() if line.strip()],
                connect_timeout=int(timeout), potcar_symbols=mapping,
            )
            candidate.save(path)
        except Exception as exc:
            st.error(f"配置未保存：{exc}")
        else:
            st.success(f"已保存：{path}")
            st.rerun()
    if st.button("检查 SSH / Slurm / VASP 环境", disabled=config is None):
        try:
            with st.spinner("正在检查已保存的集群配置；不会提交作业……"):
                report = _with_password(doctor, config)
            if report.get("ok"):
                st.success("环境检查通过。真实作业和科学结果仍需单独验证。")
            else:
                st.warning("部分检查未通过，请检查以下结果。")
            st.json(report)
        except Exception as exc:
            st.error(f"环境检查失败：{exc}")


def _structure_plot(structure: Structure) -> None:
    fig = plt.figure(figsize=(6, 4))
    try:
        axis = fig.add_subplot(111, projection="3d")
        corners = np.asarray(list(product((0, 1), repeat=3)))
        cartesian = structure.lattice.get_cartesian_coords(corners)
        for i, first in enumerate(corners):
            for j in range(i + 1, len(corners)):
                if np.abs(first - corners[j]).sum() == 1:
                    edge = cartesian[[i, j]]
                    axis.plot(edge[:, 0], edge[:, 1], edge[:, 2], color="#8794a3", linewidth=0.8)
        species = sorted({site.species_string for site in structure})
        for name in species:
            positions = np.asarray([site.coords for site in structure if site.species_string == name])
            axis.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=75, label=name)
        axis.set(xlabel="x / Å", ylabel="y / Å", zlabel="z / Å")
        axis.set_box_aspect(np.maximum(np.ptp(cartesian, axis=0), 1e-6))
        axis.legend(loc="upper left")
        st.pyplot(fig)
        st.caption("晶胞与原子坐标预览；球大小仅用于显示，不代表原子半径。")
    finally:
        plt.close(fig)


def _preview(upload) -> Structure | None:
    suffix = ".cif" if upload.name.lower().endswith(".cif") else ".vasp"
    try:
        with tempfile.TemporaryDirectory(prefix="vasp-agent-preview-") as directory:
            source = Path(directory) / f"structure{suffix}"
            source.write_bytes(upload.getvalue())
            structure = Structure.from_file(source)
        cols = st.columns(3)
        cols[0].metric("化学式", structure.composition.reduced_formula)
        cols[1].metric("原子数", len(structure))
        cols[2].metric("晶胞体积 / Å³", f"{structure.volume:.3f}")
        st.caption("晶格 a, b, c / Å：" + ", ".join(f"{v:.4f}" for v in structure.lattice.abc))
        _structure_plot(structure)
        with st.expander("检查元素和分数坐标"):
            st.dataframe([
                {"元素": str(site.species_string), "x": site.a, "y": site.b, "z": site.c}
                for site in structure
            ], hide_index=True)
        return structure
    except Exception as exc:
        st.error(f"结构无法解析：{exc}")
        return None


def _stage_results(run_dir: Path, stage: dict, scf_fermi_energy: float | None = None) -> None:
    result = stage.get("result")
    if not result:
        return
    st.subheader(f"{stage['name']} · 结果")
    if not result.get("success"):
        st.warning(result.get("reason", "结果尚未通过检查。"))
        if result.get("recovery_hint"):
            st.info(result["recovery_hint"])
        return
    if stage["name"] in {"relax", "scf"}:
        metrics = st.columns(2)
        energy = result.get("final_energy_ev")
        force = result.get("final_max_force_ev_angstrom")
        if energy is not None:
            metrics[0].metric("最终总能量 / eV", f"{energy:.6f}")
        if force is not None:
            metrics[1].metric("最大原子力 / eV Å⁻¹", f"{force:.6f}")
    elif stage["name"] in {"bands", "dos"}:
        if result.get("energy_reference_source") == "preceding_scf":
            scf_fermi_energy = result.get("energy_reference_ev", scf_fermi_energy)
        if scf_fermi_energy is not None:
            st.metric("前序 SCF 费米能 / eV", f"{scf_fermi_energy:.6f}")
        st.caption("固定电荷谱计算；基态总能量和原子力请查看前序 SCF，此处仅列其费米能作为参考。")
    output = run_dir / stage["folder"] / "outputs"
    final_structure = output / "final_structure.cif"
    if final_structure.is_file():
        try:
            structure = Structure.from_file(final_structure)
            with st.expander("查看最终结构", expanded=True):
                st.write(f"{structure.composition.reduced_formula} · {len(structure)} 个原子 · "
                         f"晶胞体积 {structure.volume:.3f} Å³")
                _structure_plot(structure)
            st.download_button("下载最终结构 / CIF", final_structure.read_bytes(),
                               file_name=f"{run_dir.name}-{stage['name']}.cif",
                               mime="chemical/x-cif", key=f"structure_{run_dir.name}_{stage['folder']}")
        except Exception as exc:
            st.error(f"最终结构无法显示：{exc}")


def _new_run(config: ClusterConfig | None, runs_root: Path) -> None:
    st.subheader("准备计算")
    st.write("上传结构，选择任务并生成本地输入。确认后才会向 Slurm 提交作业。")
    st.caption("当前方法：非磁性 PBE（ISPIN=1）。bands 和 dos 会先执行同一结构的 SCF，不会自动优化结构。")
    upload = st.file_uploader("结构文件 / CIF 或 POSCAR", help="POSCAR 可以没有扩展名；不上传 POTCAR。")
    structure = _preview(upload) if upload is not None else None
    with st.form("prepare_run"):
        task = st.selectbox("任务", ["relax", "scf", "bands", "dos"],
                            format_func=lambda key: {"relax": "relax · 结构优化", "scf": "scf · 自洽计算",
                                                     "bands": "bands · 能带", "dos": "dos · 态密度"}[key])
        st.caption("以下为起始参数，不是经过收敛测试的推荐精度。任务所需阶段会在提交前展示。")
        left, right = st.columns(2)
        with left:
            encut = st.number_input("ENCUT / eV", 100.0, 2000.0, 520.0, 10.0)
            ediff = st.number_input("EDIFF / eV", min_value=1e-10, max_value=1e-2, value=1e-5,
                                    format="%.1e")
            ismear = st.selectbox("ISMEAR", [-5, -1, 0, 1, 2], index=2)
            sigma = st.number_input("SIGMA / eV", 0.001, 1.0, 0.05, 0.01, format="%.3f")
        with right:
            k_grid = st.text_input("Γ 中心 k 网格", "4 4 4", help="三个正整数，例如 4 4 4。")
            nsw = st.number_input("优化步数上限 / NSW", 1, 1000, 100)
            ediffg = st.number_input("力收敛阈值 / eV Å⁻¹", 0.0001, 1.0, 0.03, 0.005, format="%.4f")
            cell_relax = st.checkbox("结构优化同时优化晶胞", value=False)
        prepared = st.form_submit_button("生成输入并预览", type="primary",
                                         disabled=structure is None or config is None)
    if config is None:
        st.info("先在「集群配置」页保存有效配置，再准备任务。")
    if prepared:
        try:
            grid = [int(part) for part in k_grid.replace(",", " ").split()]
            if len(grid) != 3 or any(part < 1 for part in grid):
                raise ValueError("k 网格需要三个正整数")
            parameters = dict(encut=encut, ediff=ediff, ediffg=-ediffg, nsw=int(nsw),
                              kpoint_grid=grid, ismear=ismear, sigma=sigma, cell_relax=cell_relax)
            run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
            run_dir = runs_root / run_id
            suffix = ".cif" if upload.name.lower().endswith(".cif") else ".vasp"
            with tempfile.TemporaryDirectory(prefix="vasp-agent-input-") as directory:
                source = Path(directory) / f"structure{suffix}"
                source.write_bytes(upload.getvalue())
                prepare_run(source, run_dir, config, task=task, parameters=parameters)
            st.session_state["active_run"] = str(run_dir)
            st.success("输入已生成。下方核对参数和集群后，点击「确认并提交」。")
        except Exception as exc:
            st.error(f"准备失败：{exc}")


@st.fragment(run_every="5s")
def _run_panel(run_dir: Path) -> None:
    try:
        state = read_state(run_dir)
    except Exception as exc:
        st.error(f"无法读取任务：{exc}")
        return
    status = state.get("status", "unknown")
    stages = state.get("stages", [])
    awaiting_submission = status == "planned" and not any(stage.get("job_id") for stage in stages)
    st.subheader(f"{state.get('formula', '')} · {state.get('task', '')}")
    st.caption(str(run_dir))
    status_label = "准备下一阶段" if status == "planned" and not awaiting_submission else STATUS_LABELS.get(status, status)
    st.write(f"状态：**{status_label}**")
    if state.get("last_error"):
        st.warning(str(state["last_error"]))
    st.dataframe([
        {"阶段": stage.get("name", ""), "状态": STATUS_LABELS.get(stage.get("status"), stage.get("status", "")),
         "Slurm job": stage.get("job_id") or "—"}
        for stage in stages
    ], hide_index=True)
    with st.expander("核对参数与集群", expanded=awaiting_submission):
        st.json(state.get("parameters", {}))
        try:
            run_config = ClusterConfig.load(run_dir / "config.json")
            st.json(run_config.to_dict())
        except Exception as exc:
            st.error(f"无法读取任务配置：{exc}")
    with st.expander("阶段结果与运行记录"):
        st.json({"stages": stages, "history": state.get("history", [])})
    with st.expander("检查实际生成的 INCAR 和 KPOINTS"):
        for stage in stages:
            folder = stage.get("folder")
            if not folder:
                continue
            st.write(stage.get("name", folder))
            for name in ("INCAR", "KPOINTS"):
                source = run_dir / folder / "inputs" / name
                if source.is_file():
                    st.caption(name)
                    st.code(source.read_text(), language="text")
    scf_fermi_energy = None
    for stage in stages:
        if stage.get("name") == "scf" and (stage.get("result") or {}).get("success"):
            scf_fermi_energy = stage["result"].get("fermi_energy_ev")
        _stage_results(run_dir, stage, scf_fermi_energy)
        folder = stage.get("folder")
        if folder:
            for plot in sorted((run_dir / folder / "outputs").glob("*.png")):
                st.image(str(plot), caption=f"{stage.get('name', '')} · {plot.name}")

    actions = st.columns(3)
    if awaiting_submission:
        reviewed = st.checkbox("已检查结构、计算参数、集群账户和每阶段资源设置", key=f"review_{run_dir.name}")
        if actions[0].button("确认并提交", type="primary", disabled=not reviewed, key=f"submit_{run_dir.name}"):
            try:
                pid = _with_password(start_worker, run_dir)
                st.success(f"后台监控已启动（PID {pid}）。可关闭页面，稍后从运行记录查看。")
            except Exception as exc:
                st.error(f"未能启动：{exc}")
    elif status not in TERMINAL:
        if actions[0].button("恢复后台监控", key=f"watch_{run_dir.name}",
                             help="用于本机重启或监控进程退出后恢复；已提交的任务由状态文件关联。"):
            try:
                st.success(f"后台监控 PID：{_with_password(start_worker, run_dir)}")
            except Exception as exc:
                st.error(f"恢复失败：{exc}")
    elif status in {"needs_attention", "failed"}:
        if actions[0].button("重新连接 / 回收", key=f"reconnect_{run_dir.name}",
                             help="重新关联原作业并收集输出，不修改计算参数。未收敛结果仍会被拒绝。"):
            try:
                restored = _with_password(resume, run_dir)
                if restored.get("status") not in TERMINAL:
                    st.success(f"后台监控 PID：{_with_password(start_worker, run_dir)}")
                else:
                    st.warning(restored.get("last_error") or "任务仍需检查，请查看运行记录。")
            except Exception as exc:
                st.error(f"重新连接失败：{exc}")
    current_stage = stages[state.get("current_stage", 0)] if stages else {}
    has_job = bool(current_stage.get("job_id"))
    can_cancel = status not in {"succeeded", "cancelled"} and (
        (has_job and current_stage.get("scheduler_state") not in SLURM_TERMINAL)
        or (not has_job and status not in TERMINAL)
    )
    if can_cancel:
        if actions[1].button("取消此任务", key=f"cancel_{run_dir.name}"):
            try:
                _with_password(cancel, run_dir)
                st.rerun()
            except Exception as exc:
                st.error(f"取消未确认：{exc}")
    bundle_key = f"bundle_{run_dir}"
    if actions[2].button("打包当前结果", key=f"bundle_btn_{run_dir.name}"):
        try:
            st.session_state[bundle_key] = str(bundle_run(run_dir))
        except Exception as exc:
            st.error(f"打包失败：{exc}")
    if bundle_key in st.session_state:
        archive = Path(st.session_state[bundle_key])
        if archive.is_file():
            st.download_button("下载 ZIP", archive.read_bytes(), file_name=archive.name,
                               mime="application/zip", key=f"download_{run_dir.name}")
    if status == "succeeded":
        st.info("流程完成表示程序判据通过；具体物性仍需检查收敛、模型与适用范围。")
    elif status == "needs_attention":
        st.info("请查看阶段结果和错误记录；本程序不会自行更改科学参数继续计算。")


def main() -> None:
    st.set_page_config(page_title="VASP Slurm Agent", page_icon="⚛", layout="wide")
    st.title("VASP Slurm Agent")
    st.caption("本地单用户 · 0.1.0a1 开发预览 · VASP + Slurm · 无需 LLM")
    with st.sidebar:
        st.header("本地文件")
        config_path = Path(st.text_input("集群配置文件", value=os.environ.get(
            "VASP_AGENT_CONFIG", str(Path.home() / ".config/vasp-slurm-agent/cluster.json")))).expanduser()
        runs_root = Path(st.text_input("任务保存目录", value=str(Path.home() / "vasp-slurm-agent-runs"))).expanduser()
        st.text_input("SSH 密码（可选）", type="password", key="ssh_password",
                      help="优先使用 SSH 密钥。密码只保留在当前会话内存，用于显式发起的检查、提交、恢复和取消。")
        st.button("清除会话密码", on_click=lambda: st.session_state.update(ssh_password=""))
        st.caption("界面仅用于受信任的本机单用户。SSH 登录、VASP 授权及 Slurm 账户须预先具备。")
    config = _load_config(config_path)
    task_tab, history_tab, config_tab = st.tabs(["新任务", "运行记录", "集群配置"])
    with config_tab:
        _config_editor(config_path, config)
    with task_tab:
        _new_run(config, runs_root)
    with history_tab:
        available = sorted(runs_root.glob("*/run.json"), reverse=True) if runs_root.is_dir() else []
        selected = st.selectbox("选择已有任务", [str(path.parent) for path in available], index=None,
                                placeholder="选择任务目录")
        if st.button("查看任务", disabled=selected is None):
            st.session_state["active_run"] = selected
        manual = st.text_input("或输入任务目录")
        if st.button("打开目录", disabled=not manual.strip()):
            st.session_state["active_run"] = str(Path(manual).expanduser())
    if st.session_state.get("active_run"):
        st.divider()
        _run_panel(Path(st.session_state["active_run"]))


if __name__ == "__main__":
    main()
