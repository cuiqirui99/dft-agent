from pathlib import Path

import pytest

from vasp_slurm_agent.config import ClusterConfig


def config(**options):
    return ClusterConfig(host="cluster.example", user="researcher", remote_root="/scratch/runs",
                         vasp_command="srun vasp_std", potcar_root="/licensed/pbe", partition="cpu", **options)


@pytest.mark.parametrize("path", ["socket", "../socket", "/tmp/socket\nnext", "/tmp/socket\r", "/tmp/socket\x00", None])
def test_invalid_control_socket(path):
    with pytest.raises(ValueError, match="SSH control socket"):
        config(ssh_control_path=path)


def test_control_socket_expands_and_survives_save(tmp_path):
    original = config(ssh_control_path="~/.ssh/master.sock")
    assert original.ssh_control_path == str(Path.home() / ".ssh/master.sock")
    path = original.save(tmp_path / "cluster.json")
    assert ClusterConfig.load(path) == original
    assert path.stat().st_mode & 0o777 == 0o600


def test_legacy_config_keeps_private_transport():
    assert config().ssh_control_path == ""


def test_multinode_resources_survive_save(tmp_path):
    value = config(nodes=2, tasks=16, extra_sbatch=["--mem=32G", "#SBATCH --qos normal",
                   "--ntasks-per-node=8", "--cpus-per-task=2", "--constraint=zen4", "--gpus=2"])
    assert ClusterConfig.load(value.save(tmp_path / "cluster.json")) == value
    assert value.sbatch_resource_lines()[1] == "#SBATCH --qos=normal"


@pytest.mark.parametrize("options", [
    {"nodes": 0}, {"nodes": 9}, {"extra_sbatch": "--mem=1G"},
    {"tasks": 1.5}, {"tasks": True}, {"port": 22.5},
    {"extra_sbatch": ["--mem=1G\n#SBATCH --output=elsewhere"]},
    {"extra_sbatch": ["--output=elsewhere"]}, {"extra_sbatch": ["--wrap=command"]},
    {"extra_sbatch": ["--nodes=3"]}, {"extra_sbatch": ["--mem=1G", "--mem-per-cpu=1G"]},
    {"extra_sbatch": ["--qos"]}, {"extra_sbatch": ["--cpus-per-task=0"]},
    {"extra_sbatch": ["--gres=gpu:1", "--gpus-per-node=1"]},
    {"nodes": 2, "tasks": 16, "extra_sbatch": ["--ntasks-per-node=4"]},
    {"setup_commands": ["module load vasp\n#SBATCH --mem=2G"]},
])
def test_invalid_resource_config(options):
    with pytest.raises(ValueError):
        config(**options)


def test_cpu_binding_belongs_to_srun():
    with pytest.raises(ValueError, match="with srun in the VASP command"):
        config(extra_sbatch=["--cpu-bind=cores"])


def test_slurm_directives_precede_shell_commands():
    from vasp_slurm_agent.workflow import _script
    value = config(nodes=2, tasks=16, extra_sbatch=["--mem=32G", "--qos=normal"],
                   setup_commands=["module load vasp"])
    script = _script(value, {"name": "scf", "metadata": {"potcar_labels": ["Si"]}}, None)
    assert "#SBATCH --nodes=2" in script
    assert script.index("#SBATCH --qos=normal") < script.index("set -e") < script.index("module load vasp")
