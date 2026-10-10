"""Bundled sample runs load as verified, placeholder-only results."""

import json
import re
import sys

from vasp_slurm_agent import cli
from vasp_slurm_agent.explanation import load_run_context
from vasp_slurm_agent.samples import install_samples, list_samples, sample_card
from vasp_slurm_agent.workflow import read_state

PERSONAL = re.compile(r"/Users/|/public1/home|/nobackup/proj|naiss|qiruic|kth\.se", re.IGNORECASE)


def test_samples_are_described_and_install_once(tmp_path):
    samples = list_samples()
    assert [item["id"] for item in samples] == ["si-pbe-chain", "mgo-cell-relax", "fe-seed-comparison"]
    assert all(item["title"] and item["description"] for item in samples)
    installed = install_samples(tmp_path / "runs")
    assert [path.name for path in installed] == ["sample-si-pbe-chain", "sample-mgo-cell-relax", "sample-fe-seed-comparison"]
    assert install_samples(tmp_path / "runs") == []
    assert sample_card(installed[0])["title"].startswith("Si")
    assert sample_card(tmp_path) is None


def test_samples_verify_and_carry_no_personal_details(tmp_path):
    for path in install_samples(tmp_path / "runs"):
        state = read_state(path)
        assert state["status"] == "succeeded"
        assert state["remote_root"].startswith("/home/researcher/dft-agent-runs/")
        context = load_run_context(path)
        for index in range(1, len(state["stages"]) + 1):
            assert context["facts"][f"stage_{index}.accepted"]["value"] is True
        for file in path.rglob("*"):
            if file.is_file() and file.suffix != ".png":
                assert not PERSONAL.search(file.read_text(errors="replace")), file
        assert not (path / "results.zip").exists()
    assert any((tmp_path / "runs" / "sample-si-pbe-chain").rglob("bands.png"))


def test_cli_lists_and_copies_samples(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["dft-agent", "samples", "--list"])
    assert cli.main() == 0
    assert len(json.loads(capsys.readouterr().out)) == 3
    monkeypatch.setattr(sys, "argv", ["dft-agent", "samples", "--runs", str(tmp_path / "runs")])
    assert cli.main() == 0
    assert "sample-mgo-cell-relax" in capsys.readouterr().out
    assert cli.main() == 0
    assert "already" in capsys.readouterr().err
