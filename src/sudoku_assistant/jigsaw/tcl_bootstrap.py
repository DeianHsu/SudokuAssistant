"""Initialize the native Tcl runtime before a portable Python's tkinter shim."""
import ctypes
from pathlib import Path
import sys


def warm_tcl():
    root = Path(getattr(sys, "_MEIPASS", sys.base_prefix))
    dll_path = root / "tcl86t.dll" if getattr(sys, "frozen", False) else root / "DLLs" / "tcl86t.dll"
    if not dll_path.exists():
        return
    runtime = ctypes.WinDLL(str(dll_path))
    runtime.Tcl_CreateInterp.restype = ctypes.c_void_p
    runtime.Tcl_Init.argtypes = [ctypes.c_void_p]
    runtime.Tcl_DeleteInterp.argtypes = [ctypes.c_void_p]
    interp = runtime.Tcl_CreateInterp()
    try:
        runtime.Tcl_Init(interp)
    finally:
        runtime.Tcl_DeleteInterp(interp)
