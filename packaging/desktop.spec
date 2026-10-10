from pathlib import Path
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

root = Path(SPECPATH).parent
sys.path.insert(0, str(root / "src"))
from vasp_slurm_agent import __version__

datas = [(str(root / "src/vasp_slurm_agent"), "vasp_slurm_agent"),
         (str(root / "LICENSE"), "."), (str(root / "NOTICE.md"), ".")]
hiddenimports = collect_submodules("vasp_slurm_agent")
for package in ("streamlit", "pymatgen", "monty", "seekpath", "spglib", "webview"):
    datas += collect_data_files(package)
for package in ("streamlit", "pymatgen", "pymatgen-core", "monty", "pywebview", "keyring"):
    try:
        datas += copy_metadata(package, recursive=True)
    except Exception:
        if package != "pymatgen-core":
            raise
for package in ("streamlit", "pymatgen.io.vasp", "pymatgen.symmetry", "keyring.backends"):
    hiddenimports += collect_submodules(package)
if sys.platform == "win32":
    hiddenimports += collect_submodules("paramiko")

a = Analysis([str(root / "packaging/desktop_entry.py")], pathex=[str(root / "src")],
             binaries=[], datas=datas, hiddenimports=hiddenimports,
             excludes=["PyQt5", "PyQt6", "PySide2", "PySide6", "IPython", "notebook", "pytest", "tkinter"],
             noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="DFT Agent", debug=False,
          bootloader_ignore_signals=False, strip=False, upx=False, console=True,
          hide_console="hide-early" if sys.platform == "win32" else None)
collection = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="DFT Agent")
if sys.platform == "darwin":
    app = BUNDLE(collection, name="DFT Agent.app", bundle_identifier="se.kth.qiruic.dftagent",
                 version=__version__, info_plist={"NSHighResolutionCapable": True,
                 "CFBundleDisplayName": "DFT Agent", "NSHumanReadableCopyright": "Qirui Cui",
                 "LSMinimumSystemVersion": "14.0"})
