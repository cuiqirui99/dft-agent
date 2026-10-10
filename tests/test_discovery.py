"""Cluster probing is read-only and its output is parsed defensively."""

import json
import sys

import pytest

from vasp_slurm_agent import cli, discovery
from vasp_slurm_agent.transport import Result

OUTPUT = """
__DFT_AGENT_SECTION__ home
/home/alice
__DFT_AGENT_SECTION__ user
alice
__DFT_AGENT_SECTION__ slurm
sbatch
squeue
sacct
scancel
sinfo
__DFT_AGENT_SECTION__ partitions
main*|up|7-00:00:00|512
gpu|up|1-00:00:00|8
bad name|up|x|1
__DFT_AGENT_SECTION__ accounts
naiss2026-1-123
proj; rm -rf /
__DFT_AGENT_SECTION__ modules
VASP/6.4.3
VASP/6.3.2
__DFT_AGENT_SECTION__ executables
vasp_gam=/sw/bin/vasp_gam
__DFT_AGENT_SECTION__ potcar
/home/alice/potcar_LDA
/sw/vasp/potpaw_PBE.64
__DFT_AGENT_SECTION__ scratch
/scratch/alice
/home/alice
__DFT_AGENT_SECTION__ unknown
ignored
"""


class FakeTransport:
    def __init__(self, output="", returncode=0, stderr=""):
        self.output, self.returncode, self.stderr = output, returncode, stderr
        self.commands = []
        self.closed = False

    def run(self, command, timeout=60):
        self.commands.append(command)
        return Result(self.returncode, self.output, self.stderr)

    def close(self):
        self.closed = True


def test_parse_and_suggest():
    facts = discovery.parse_probe(OUTPUT)
    assert [item["name"] for item in facts["partitions"]] == ["main", "gpu"]
    assert facts["partitions"][0]["default"] and not facts["partitions"][1]["default"]
    assert facts["accounts"] == ["naiss2026-1-123"]
    assert facts["modules"] == ["VASP/6.4.3", "VASP/6.3.2"]
    assert facts["executables"] == {"vasp_gam": "/sw/bin/vasp_gam"}
    result = discovery.suggest(facts)
    assert result["suggested"] == {"partition": "main", "account": "naiss2026-1-123", "potcar_root": "/sw/vasp/potpaw_PBE.64",
                                   "remote_root": "/scratch/alice/dft-agent-runs", "setup_commands": ["module load VASP/6.4.3"],
                                   "vasp_command": "srun vasp_std", "vasp_ncl_command": "srun vasp_ncl"}
    assert any("7-00:00:00" in note for note in result["notes"])


def test_suggest_with_nothing_found_explains():
    result = discovery.suggest(discovery.parse_probe("__DFT_AGENT_SECTION__ home\n/home/x\n"))
    assert result["suggested"] == {}
    assert any("partitions" in note for note in result["notes"])
    assert any("POTCAR" in note for note in result["notes"])
    assert any("VASP was not found" in note for note in result["notes"])
    assert any("Missing Slurm commands" in note for note in result["notes"])


def test_probe_runs_one_read_only_script_and_closes():
    transport = FakeTransport(OUTPUT)
    config = discovery.probe_config("cluster.invalid", "alice", 2222, setup_commands=["module load slurm"])
    assert config.port == 2222 and config.partition == "probe"
    report = discovery.probe_cluster(config, transport)
    assert report["suggested"]["partition"] == "main"
    assert len(transport.commands) == 1
    assert transport.commands[0].startswith("module load slurm\n")
    assert "sbatch " not in transport.commands[0].replace("command -v", "").replace("for n in sbatch", "")
    assert not transport.closed  # borrowed transports stay open
    with pytest.raises(discovery.DiscoveryError, match="did not answer"):
        discovery.probe_cluster(config, FakeTransport("", 255, "Permission denied"))


def test_presets_load_with_values():
    presets = discovery.load_presets()
    assert {item["id"] for item in presets} >= {"dardel", "tetralith", "generic", "bscc"}
    assert presets[0]["id"] == "generic"
    dardel = next(item for item in presets if item["id"] == "dardel")
    values = discovery.preset_values(dardel)
    assert values["host"] == "dardel.pdc.kth.se" and values["partition"] == "main"
    assert "potcar_root" not in values and "account" not in values
    assert isinstance(values["setup_commands"], list)


def test_probe_keeps_environment_prerequisites():
    setup = ["source /site/modules.sh", "module load compiler/2026"]
    config = discovery.probe_config("cluster.invalid", "alice", setup_commands=setup)
    report = discovery.probe_cluster(config, FakeTransport(OUTPUT))
    assert report["suggested"]["setup_commands"] == [*setup, "module load VASP/6.4.3"]
    merged = discovery.merge_suggestions({"setup_commands": setup, "vasp_command": "mpprun vasp_std"}, report["suggested"])
    assert merged["setup_commands"] == [*setup, "module load VASP/6.4.3"]
    assert merged["vasp_command"] == "mpprun vasp_std"


def test_cli_detect_prints_suggestions(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(discovery, "SSHTransport", None, raising=False)
    monkeypatch.setattr("vasp_slurm_agent.transport.SSHTransport", lambda config: FakeTransport(OUTPUT))
    monkeypatch.setattr(sys, "argv", ["dft-agent", "detect", "--host", "cluster.invalid", "--user", "alice",
                                      "--config", str(tmp_path / "none.json")])
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["suggested"]["account"] == "naiss2026-1-123"
    monkeypatch.setattr(sys, "argv", ["dft-agent", "detect", "--config", str(tmp_path / "none.json")])
    assert cli.main() == 1
    assert "--host" in capsys.readouterr().err
