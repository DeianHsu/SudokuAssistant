"""Collect Tk from the bundled portable Python after native initialization."""
from PyInstaller import isolated
from PyInstaller.utils.hooks import tcl_tk


@isolated.decorate
def warmed_tcl_info():
    from pathlib import Path
    import sys
    import ctypes
    dll = ctypes.WinDLL(str(Path(sys.base_prefix) / "DLLs" / "tcl86t.dll"))
    dll.Tcl_CreateInterp.restype = ctypes.c_void_p
    dll.Tcl_Init.argtypes = [ctypes.c_void_p]
    dll.Tcl_DeleteInterp.argtypes = [ctypes.c_void_p]
    interp = dll.Tcl_CreateInterp()
    dll.Tcl_Init(interp)
    dll.Tcl_DeleteInterp(interp)
    import tkinter
    import _tkinter
    tcl = tkinter.Tcl()
    return {"available": True, "tkinter_extension_file": _tkinter.__file__,
            "tcl_version": _tkinter.TCL_VERSION, "tk_version": _tkinter.TK_VERSION,
            "tcl_threaded": tcl.eval("info exists tcl_platform(threaded)") == "1",
            "tcl_data_dir": tcl.eval("info library"), "_tcl_runtime_version": tcl.eval("info tclversion")}


def pre_find_module_path(hook_api):
    tcl_tk._get_tcl_tk_info = warmed_tcl_info
    if not tcl_tk.tcltk_info.available:
        raise RuntimeError("图形界面配套文件不可用，停止打包。")
