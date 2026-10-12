"""Build a standalone executable containing all three assistant packages."""
import os
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
os.environ["PYINSTALLER_CONFIG_DIR"] = str(root / "build" / "cache")
for name, folder in (("TCL_LIBRARY", "tcl8.6"), ("TK_LIBRARY", "tk8.6")):
    location = Path(sys.base_prefix) / "tcl" / folder
    if location.exists():
        os.environ[name] = location.as_posix()
os.chdir(root)
import PyInstaller.__main__

arguments = ["--noconfirm", "--onefile", "--windowed", "--name", "SudokuAssistant",
             "--manifest", "app.manifest", "--additional-hooks-dir", "hooks",
             "--exclude-module", "matplotlib", "--exclude-module", "pandas",
             "--exclude-module", "scipy", "--exclude-module", "IPython"]
for key in ("universe", "jigsaw", "killer"):
    asset_path = f"sudoku_assistant/{key}/assets"
    arguments.extend(("--add-data", f"{asset_path};{asset_path}"))
PyInstaller.__main__.run([*arguments, "app.py"])
