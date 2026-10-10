"""The ui command, cluster checklist rows and friendly errors."""

import sys

from vasp_slurm_agent import cli


def test_ui_opens_the_browser_and_hides_the_toolbar(monkeypatch):
    seen = {}
    opened = []

    class FakeStreamlit:
        @staticmethod
        def main():
            seen["argv"] = list(sys.argv)
            return 0

    monkeypatch.setitem(sys.modules, "streamlit.web.cli", FakeStreamlit)
    monkeypatch.setattr(cli, "_open_browser_later", lambda url, delay=1.5: opened.append(url))
    monkeypatch.delenv("DFT_AGENT_NO_BROWSER")
    monkeypatch.setattr(sys, "argv", ["dft-agent", "ui", "--port", "8600"])
    assert cli.main() == 0
    assert "--client.toolbarMode=minimal" in seen["argv"] and "--server.port=8600" in seen["argv"]
    assert "--server.address=127.0.0.1" in seen["argv"] and "--browser.gatherUsageStats=false" in seen["argv"]
    assert opened == ["http://127.0.0.1:8600"]
    monkeypatch.setattr(sys, "argv", ["dft-agent", "ui", "--no-browser"])
    assert cli.main() == 0
    assert opened == ["http://127.0.0.1:8600"]


def test_doctor_rows_carry_hints():
    report = {"ok": False, "checks": {"sbatch": {"ok": True, "detail": "/usr/bin/sbatch"},
                                      "potcar_root": {"ok": False, "detail": "/licensed/pbe"},
                                      "mystery": {"ok": False, "detail": ""}}}
    rows = cli.doctor_rows(report)
    assert rows[0] == {"check": "Slurm: sbatch", "ok": True, "detail": "/usr/bin/sbatch", "hint": ""}
    assert rows[1]["check"] == "POTCAR folder" and "POTCAR" in rows[1]["hint"]
    assert rows[2]["hint"]


def test_describe_errors():
    assert cli._describe(FileNotFoundError(2, "No such file", "/x/y.json")) == "File not found: /x/y.json"
    assert cli._describe(PermissionError(13, "Permission denied", "/x")) == "Permission denied: /x"
    assert cli._describe(ValueError("Plain message")) == "Plain message"
    assert cli._describe(FileExistsError("Choose a new run directory.")) == "Choose a new run directory."
