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
