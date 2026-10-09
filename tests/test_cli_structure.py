import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
from pymatgen.core import Structure
from vasp_slurm_agent import agent, cli, workflow
from vasp_slurm_agent.structures import apply_structure_plan
from test_cli_plan import local_cli


def test_convert_without_model_or_cluster(local_cli, monkeypatch, capsys):
    source, _, run, invoke = local_cli
    original = source.read_bytes()
    monkeypatch.setattr(agent, "_request_structured", lambda *a, **k: (_ for _ in ()).throw(AssertionError("No model needed")))
    assert invoke("convert", source, run, "--format", "poscar") == 0
    result = json.loads(capsys.readouterr().out)
    actual = Structure.from_file(result["files"]["poscar"])
    expected = Structure.from_file(source)
    assert actual.species == expected.species
    assert np.allclose(actual.lattice.matrix, expected.lattice.matrix)
    assert np.allclose(actual.frac_coords, expected.frac_coords)
    assert source.read_bytes() == original
    assert not (run / "run.json").exists()
    assert invoke("convert", source, run) == 1


def test_structure_cli_revision_replaces_supercell(local_cli, monkeypatch, capsys):
    source, _, run, invoke = local_cli
    first = run.parent / "first.json"
    second = run.parent / "second.json"
    requests = []

    def model(payload, schema, settings, **kwargs):
        requests.append(json.loads(payload))
        repeats = [2, 1, 1] if len(requests) == 1 else [3, 1, 1]
        return json.dumps({"status": "ready", "summary": "Repeat the cell.",
                           "operations": [{"type": "supercell", "matrix": repeats}],
                           "output_format": "both", "questions": [], "notes": []})

    monkeypatch.setattr(agent, "_request_structured", model)
    assert invoke("structure-plan", source, "Make a 2x1x1 supercell.", "--provider", "codex", "--output", first) == 0
    capsys.readouterr()
    assert invoke("structure-plan", source, "Change it to 3x1x1.", "--previous", first,
                  "--provider", "codex", "--output", second) == 0
    capsys.readouterr()
    assert requests[1]["goal"] == "Make a 2x1x1 supercell."
    assert requests[1]["history"][-1]["content"] == "Change it to 3x1x1."
    assert invoke("prepare-structure", source, run, "--plan", second) == 0
    result = json.loads(capsys.readouterr().out)
    assert len(Structure.from_file(result["files"]["poscar"])) == 3 * len(Structure.from_file(source))
    assert not (run / "run.json").exists()
    source.write_text(source.read_text() + "\n")
    assert invoke("structure-plan", source, "Change to 4x1x1.", "--previous", second,
                  "--provider", "codex") == 1
    assert len(requests) == 2


def test_bundle_keeps_structure_edit_provenance(local_cli):
    source, config, run, invoke = local_cli
    edited = run.parent / "edited"
    result = apply_structure_plan(source, edited, {"status": "ready", "operations": [],
                                  "output_format": "both", "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    assert invoke("prepare", result["files"]["poscar"], run, "--config", config, "--task", "scf") == 0
    shutil.copytree(edited, run / "structure_edit")
    with zipfile.ZipFile(workflow.bundle_run(run)) as bundle:
        for name in ["source.cif", "edited.cif", "POSCAR", "structure.json"]:
            assert bundle.read("structure_edit/" + name) == (edited / name).read_bytes()
        metadata = json.loads(bundle.read("structure_edit/structure.json"))
        assert metadata["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert str(run.parent) not in json.dumps(metadata)
