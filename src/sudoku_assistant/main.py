"""Desktop interface for the three supported Sudoku games."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
import threading
import time

from PIL import Image, ImageGrab

from .coordinator import Coordinator
from .settings import CLICK_INTERVAL_MS
from .universe import app as universe_app
from .jigsaw import app as jigsaw_app
from .killer import app as killer_app


VERSION = "1.0.3"
ASSISTANTS = {
    "universe": (universe_app.DISPLAY_NAME, universe_app),
    "jigsaw": (jigsaw_app.DISPLAY_NAME, jigsaw_app),
    "killer": (killer_app.DISPLAY_NAME, killer_app),
}


def base_dir():
    return Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]


def configure_tk():
    from tcl_bootstrap import warm_tcl
    warm_tcl()
    root = Path(getattr(sys, "_MEIPASS", sys.base_prefix))
    for name, folder, frozen_folder in (("TCL_LIBRARY", "tcl8.6", "_tcl_data"),
                                         ("TK_LIBRARY", "tk8.6", "_tk_data")):
        location = root / frozen_folder if getattr(sys, "frozen", False) else root / "tcl" / folder
        if location.exists():
            os.environ[name] = location.as_posix()


def managed_page(base_class, coordinator, key):
    class ManagedPage(base_class):
        def start(self):
            started = coordinator.start(key)
            if not started and coordinator.last_reason:
                self.log(coordinator.last_reason)
                self.status.set(coordinator.last_reason)

        def pause(self):
            coordinator.pause()

        def stop(self):
            coordinator.stop_all()

    return ManagedPage


class AssistantWindow:
    def __init__(self, root, data_root=None, selected="universe"):
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.closed = threading.Event()
        self.coordinator = Coordinator(selected)
        self.pages = {}
        self.frames = {}
        self.data_root = Path(data_root) if data_root is not None else base_dir() / "data"
        self.data_root.mkdir(parents=True, exist_ok=True)

        root.title("数独自动助手（SudokuAssistant）")
        root.configure(bg="#f5f7fa")
        style = ttk.Style(root)
        style.theme_use("clam")
        header = ttk.Frame(root, padding=(18, 12, 18, 8))
        header.pack(fill="x")
        ttk.Label(header, text="选择助手", font=("Microsoft YaHei UI", 11, "bold")).pack(side="left", padx=(0, 12))
        self.labels = {key: value[0] for key, value in ASSISTANTS.items()}
        self.keys = {label: key for key, label in self.labels.items()}
        self.choice = tk.StringVar(root, value=self.labels[selected])
        self.selector = ttk.Combobox(header, textvariable=self.choice, values=tuple(self.keys),
                                     state="readonly", width=39, font=("Microsoft YaHei UI", 10))
        self.selector.pack(side="left")
        ttk.Label(header, text="切换助手会停止当前任务。", foreground="#536575").pack(side="left", padx=16)
        self.host = ttk.Frame(root)
        self.host.pack(fill="both", expand=True)

        for key, (_, module) in ASSISTANTS.items():
            frame = ttk.Frame(self.host)
            cls = managed_page(module.App, self.coordinator, key)
            page = cls(frame, register_hotkeys=False, embedded=True, data_dir=self.data_root / key)
            self.frames[key] = frame
            self.pages[key] = page
            self.coordinator.register(key, page,
                lambda page=page, cls=module.App: cls.start(page),
                lambda page=page, cls=module.App: cls.pause(page),
                lambda page=page, cls=module.App: cls.stop(page))

        self.frames[selected].pack(fill="both", expand=True)
        self.choice.trace_add("write", self.selection_changed)
        root.geometry("820x1000")
        root.minsize(760, 940)
        root.protocol("WM_DELETE_WINDOW", self.close)

    def selection_changed(self, *_):
        if self.closed.is_set():
            return
        key = self.keys.get(self.choice.get())
        if key is None:
            return
        previous = self.coordinator.selected
        active = self.coordinator.active_key()
        if not self.coordinator.select(key):
            return
        # Cancellation is sent before showing the newly selected interface.
        if active is not None:
            page = self.pages[active]
            page.log("已切换助手，当前任务已停止。")
            page.status.set("已切换助手，当前任务已停止。")
        self.frames[previous].pack_forget()
        self.frames[key].pack(fill="both", expand=True)

    def close(self):
        if self.closed.is_set():
            return
        self.closed.set()
        self.coordinator.close()
        for page in self.pages.values():
            page.closed.set()
            page.root.after_cancel(page.poll_id)
        deadline = time.monotonic() + 2
        for page in self.pages.values():
            if page.active():
                page.worker.join(timeout=max(0, deadline-time.monotonic()))
        self.root.destroy()

    def describe(self):
        return {"version": VERSION, "interface": "dropdown", "window_title": self.root.title(),
                "selected": self.coordinator.selected,
                "assistants": {key: {"label": self.labels[key], "limit": int(page.limit.get()),
                                      "continuous": bool(page.continuous.get()),
                                      "interval_ms": CLICK_INTERVAL_MS,
                                      "data_dir": str(page.data_dir)} for key, page in self.pages.items()}}


def inspect_image(key, path, menu=False):
    module = ASSISTANTS[key][1]
    if not menu:
        return {"ok": True, "assistant": key, **module.analyze(path)}
    with Image.open(path) as image:
        frame = module.canonical(image)
    if key == "universe" and module.endless_menu_visible(frame):
        return {"ok": True, "assistant": key, "mode": "无尽模式", "choices": module.ENDLESS_CHOICES}
    catalog = module.inspect_level_menu(frame)
    return {"ok": True, "assistant": key, "group": catalog.group, "remaining": catalog.remaining,
            "completed": catalog.completed, "total": catalog.total}


def smoke_test():
    from .universe import vision as universe_vision
    from .jigsaw import vision as jigsaw_vision
    from .killer import vision as killer_vision
    from .universe.wincontrol import enable_physical_pixels, Input
    enable_physical_pixels()
    configure_tk()
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    app = None
    try:
        app = AssistantWindow(root)
        root.update()
        for module in (universe_vision, jigsaw_vision, killer_vision):
            module.Recognizer()
        universe_vision.completion_template()
        jigsaw_vision.menu_templates()
        killer_vision.menu_templates()
        universe = app.pages["universe"]
        universe.mode.set("无尽模式")
        universe.change_mode()
        assert universe.endless_difficulty_widget.instate(["readonly"])
        assert universe.difficulty_widget.instate(["disabled"])
        app.choice.set(app.labels["jigsaw"])
        app.choice.set(app.labels["killer"])
        app.choice.set(app.labels["universe"])
        assert universe.mode.get() == "无尽模式"
        universe.mode.set("关卡模式")
        universe.change_mode()
        return {"ok": True, **app.describe(), "templates": "all three loaded",
                "endless_controls": "ok", "input_structure_bytes": ctypes.sizeof(Input)}
    finally:
        if app is not None:
            app.close()
        else:
            root.destroy()


def preview(directory):
    """Capture only this application's three interfaces; never operate a game."""
    from .universe.wincontrol import enable_physical_pixels
    enable_physical_pixels()
    configure_tk()
    import tkinter as tk
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    root = tk.Tk()
    root.withdraw()
    app = AssistantWindow(root, data_root=output / "preview-data")
    files = []
    try:
        root.geometry("820x1000+70+40")
        root.attributes("-topmost", True)
        root.deiconify()
        for key in ASSISTANTS:
            app.choice.set(app.labels[key])
            root.update()
            time.sleep(.15)
            root.update()
            path = output / f"{key}.png"
            left, top = root.winfo_rootx(), root.winfo_rooty()
            ImageGrab.grab(bbox=(left, top, left+root.winfo_width(), top+root.winfo_height()),
                           all_screens=True).save(path)
            files.append(str(path))
        return {"ok": True, "previews": files}
    finally:
        app.close()


def main():
    parser = argparse.ArgumentParser(description="选择数独宇宙、拼图数独或杀手数独助手。")
    parser.add_argument("--assistant", choices=tuple(ASSISTANTS), default="universe")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--analyze", help="仅识别并求解棋盘截图")
    group.add_argument("--inspect-menu", help="仅检查选关截图")
    group.add_argument("--smoke-test", action="store_true", help="仅检查三个界面和资源初始化")
    group.add_argument("--preview-dir", help="保存三个界面的预览图片，不操作游戏")
    parser.add_argument("--report", help="保存检查结果")
    args = parser.parse_args()
    if args.analyze or args.inspect_menu or args.smoke_test or args.preview_dir:
        try:
            if args.smoke_test:
                report = smoke_test()
            elif args.preview_dir:
                report = preview(args.preview_dir)
            else:
                report = inspect_image(args.assistant, args.analyze or args.inspect_menu, menu=bool(args.inspect_menu))
        except Exception as error:
            report = {"ok": False, "error": str(error)}
        if args.report:
            Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ok"] else 1
    from .universe.wincontrol import enable_physical_pixels
    enable_physical_pixels()
    configure_tk()
    import tkinter as tk
    root = tk.Tk()
    AssistantWindow(root, selected=args.assistant)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
