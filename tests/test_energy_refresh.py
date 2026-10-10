"""Old plots can be corrected without changing saved calculation evidence."""

import csv
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from vasp_slurm_agent.postprocessing import refresh_energy_exports


@pytest.fixture
def old_run(tmp_path):
    sample = Path(__file__).parents[1] / "src/vasp_slurm_agent/samples/srtio3-pbe-chain"
    shutil.copy(sample / "run.json", tmp_path / "run.json")
    shutil.copytree(sample / "01_relax", tmp_path / "01_relax")
    output = tmp_path / "01_relax/outputs"
    (output / "plot_metadata.json").write_text('{"task":"relax"}')
    (output / "relax_energy.csv").write_text("ionic_step,energy_ev\n1,0\n2,0\n3,0\n")
    return tmp_path, output


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_existing_zero_plot_is_corrected_without_changing_raw_evidence(old_run):
    root, output = old_run
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    unchanged = [output / name for name in manifest] + [root / "run.json", output / "result.json"]
    before = {path: digest(path) for path in unchanged}
    assert refresh_energy_exports(root, "01_relax")
    with (output / "relax_energy.csv").open() as stream:
        energies = [float(row["energy_ev"]) for row in csv.DictReader(stream)]
    assert energies == pytest.approx([-39.96213511, -39.96623414, -39.99141810], abs=1e-8)
    assert {path: digest(path) for path in unchanged} == before
    assert (output / "relax_energy.png").read_bytes().startswith(b"\x89PNG")
    assert (output / "relax_energy.pdf").read_bytes().startswith(b"%PDF")
    after = digest(output / "relax_energy.png")
    assert not refresh_energy_exports(root, "01_relax")
    assert digest(output / "relax_energy.png") == after


def test_changed_xml_does_not_regenerate_a_trusted_plot(old_run):
    root, output = old_run
    before = digest(output / "relax_energy.csv")
    with (output / "vasprun.xml").open("a") as stream:
        stream.write("\n<!-- changed -->\n")
    with pytest.raises(ValueError, match="XML changed"):
        refresh_energy_exports(root, "01_relax")
    assert digest(output / "relax_energy.csv") == before


def test_unaccepted_run_is_not_promoted_by_plot_refresh(old_run):
    root, output = old_run
    state = json.loads((root / "run.json").read_text())
    state["stages"][0]["status"] = "needs_attention"
    (root / "run.json").write_text(json.dumps(state))
    before = digest(output / "relax_energy.csv")
    assert not refresh_energy_exports(root, "01_relax")
    assert digest(output / "relax_energy.csv") == before
