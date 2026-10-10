"""Keep the data behind each plot easy to download without exposing private files."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from streamlit.testing.v1 import AppTest

from vasp_slurm_agent.app import _plot_data_files


def render_files(root, stage="relax", success=True):
    state = {"name": stage, "folder": "01_" + stage, "result": {"success": success}}
    script = f"""
from pathlib import Path
from vasp_slurm_agent.app import _stage_downloads
_stage_downloads(Path({str(root)!r}), {state!r})
"""
    app = AppTest.from_string(script).run()
    assert not app.exception
    return app


def plot(output, stem):
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig.savefig(output / f"{stem}.png")
    plt.close(fig)


@pytest.mark.parametrize("stem,stage", [("relax_energy", "relax"), ("bands", "bands"),
                                       ("dos", "dos"), ("dos_elements", "dos"), ("dos_orbitals", "dos")])
def test_plot_csv_is_directly_available_with_raw_files(tmp_path, stem, stage, monkeypatch):
    output = tmp_path / ("01_" + stage) / "outputs"
    plot(output, stem)
    csv = output / f"{stem}.csv"
    csv.write_text("step,energy\n1,-10.25\n")
    (output / "OUTCAR").write_text("solver output")
    (output / "vasprun.xml").write_text("<modeling/>")
    (output / "POTCAR").write_text("licensed potential")
    read_files = []
    original = Path.read_bytes

    def read_bytes(path):
        read_files.append(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    app = render_files(tmp_path, stage)
    downloads = {item.label: item for item in app.download_button}
    assert downloads["CSV"].proto.url and not downloads["CSV"].disabled
    assert csv in read_files
    assert downloads["Download file"].proto.url
    assert any(item.value == "Raw data and files" for item in app.markdown)
    assert not any(item.get("selectbox") or item.get("download_button") for item in app.expander)
    files = next(item for item in app.selectbox if item.label == "File")
    assert files.options[:2] == ["vasprun.xml", "OUTCAR"]
    assert "POTCAR" not in files.options


def test_combined_plot_links_only_its_actual_csv_sources(tmp_path):
    output = tmp_path / "01_bands" / "outputs"
    plot(output, "bands_dos")
    bands = output / "bands.csv"
    bands.write_text("distance,energy\n0,-1\n")
    wrong = tmp_path / "02_dos" / "outputs"
    correct = tmp_path / "03_dos" / "outputs"
    wrong.mkdir(parents=True)
    correct.mkdir(parents=True)
    (wrong / "dos.csv").write_text("energy,dos\n0,999\n")
    (correct / "dos.csv").write_text("energy,dos\n0,2\n")
    (correct / "dos_elements.csv").write_text("element,energy,dos\nSi,0,2\n")
    sources = [bands, correct / "dos.csv", correct / "dos_elements.csv"]
    metadata = output / "bands_dos_metadata.json"
    metadata.write_text(json.dumps({"source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}))
    linked = _plot_data_files(tmp_path, output / "bands_dos.png", list(output.iterdir()))
    assert linked == list(zip(["Bands CSV", "DOS CSV", "Element DOS CSV"], sources))
    app = render_files(tmp_path, "bands")
    labels = {item.label for item in app.download_button}
    assert {"Bands CSV", "DOS CSV", "Element DOS CSV"} <= labels
    (correct / "dos.csv").write_text("changed after plot creation")
    assert "DOS CSV" not in {label for label, _ in _plot_data_files(tmp_path, output / "bands_dos.png", list(output.iterdir()))}


def test_shortcut_and_picker_exclude_symlinks_and_hidden_files(tmp_path):
    root = tmp_path / "run"
    output = root / "01_relax" / "outputs"
    plot(output, "relax_energy")
    outside = tmp_path / "private"
    outside.mkdir()
    (outside / "data.csv").write_text("private data")
    try:
        (output / "relax_energy.csv").symlink_to(outside / "data.csv")
        (output / "nested").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("This host does not allow creating symlinks.")
    (output / ".secret.csv").write_text("secret")
    (output / "POTCAR").write_text("licensed potential")
    (output / "OUTCAR").write_text("public output")
    app = render_files(root)
    assert "CSV" not in {item.label for item in app.download_button}
    files = next(item for item in app.selectbox if item.label == "File")
    assert files.options == ["OUTCAR", "relax_energy.png"]


def test_large_csv_is_not_loaded_for_download_or_preview(tmp_path, monkeypatch):
    output = tmp_path / "01_relax" / "outputs"
    plot(output, "relax_energy")
    csv = output / "relax_energy.csv"
    with csv.open("wb") as stream:
        stream.truncate(50 * 1024 * 1024 + 1)
    original = Path.read_bytes

    def read_bytes(path):
        assert path != csv, "Large files must not be loaded into the UI"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    app = render_files(tmp_path)
    downloads = {item.label: item for item in app.download_button}
    assert downloads["CSV"].disabled
    assert "Download file" not in downloads
    assert not app.dataframe
    assert any("Download results (.zip)" in item.value for item in app.caption)


@pytest.mark.parametrize("refresh_error", [None, ValueError("Changed XML")])
def test_energy_refresh_clears_cached_zip_or_hides_stale_plot(tmp_path, monkeypatch, refresh_error):
    from vasp_slurm_agent import postprocessing

    output = tmp_path / "01_relax" / "outputs"
    plot(output, "relax_energy")
    (output / "relax_energy.csv").write_text("step,energy\n1,-10\n")
    (output / "OUTCAR").write_text("original solver output")
    (tmp_path / "run.json").write_text("{}")
    calls = []

    def refresh(root, folder):
        calls.append((root, folder))
        if refresh_error:
            raise refresh_error
        return True

    monkeypatch.setattr(postprocessing, "refresh_energy_exports", refresh)
    app = render_files(tmp_path)
    app.session_state[f"bundle_{tmp_path}"] = "old.zip"
    app.run()
    assert not app.exception
    assert calls and calls[-1] == (tmp_path, "01_relax")
    assert (output / "OUTCAR").read_text() == "original solver output"
    assert next(item for item in app.download_button if item.label == "Download file").proto.url
    if refresh_error:
        assert not app.get("image")
        assert not any(item.label == "Plot" for item in app.selectbox)
        assert any("Could not refresh" in item.value for item in app.warning)
    else:
        assert f"bundle_{tmp_path}" not in app.session_state
        assert any(item.label == "CSV" for item in app.download_button)
