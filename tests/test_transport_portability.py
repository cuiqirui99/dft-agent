"""File-transfer behavior when Windows has no Unix-only os APIs."""
import hashlib
import io
import os
import tarfile

from vasp_slurm_agent.transport import _extract_checked_archive


def test_verified_archive_extracts_without_fchmod(tmp_path, monkeypatch):
    content = b"binary output\x00\xff\r\n"
    expected = {"OUTCAR": {"sha256": hashlib.sha256(content).hexdigest(), "size": len(content)}}
    archive_path = tmp_path / "outputs.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo("OUTCAR")
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    target = tmp_path / "output"
    target.mkdir()
    monkeypatch.delattr(os, "fchmod", raising=False)
    assert _extract_checked_archive(archive_path, target, expected) == expected
    assert (target / "OUTCAR").read_bytes() == content
