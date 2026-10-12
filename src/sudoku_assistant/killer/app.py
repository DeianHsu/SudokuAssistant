"""Chinese desktop launcher; all automation requires an explicit start."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time

from PIL import Image

from .solver import solve_unique
from .vision import Recognizer, LEVEL_GROUPS, canonical, inspect_level_menu
from ..settings import CLICK_INTERVAL

DISPLAY_NAME = "杀手数独（Sudoku Killer）自动助手"


def base_dir():
    return Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent


def configure_tk():
    # Some portable Python distributions do not initialize Tcl's search path.
    from .tcl_bootstrap import warm_tcl
    warm_tcl()
    root = Path(getattr(sys, "_MEIPASS", sys.base_prefix))
    tcl = root / "_tcl_data" if getattr(sys, "frozen", False) else root / "tcl" / "tcl8.6"
    tk = root / "_tk_data" if getattr(sys, "frozen", False) else root / "tcl" / "tk8.6"
    if (tcl / "init.tcl").exists():
        os.environ["TCL_LIBRARY"] = tcl.as_posix()
    if (tk / "tk.tcl").exists():
        os.environ["TK_LIBRARY"] = tk.as_posix()


def analyze(path):
    with Image.open(path) as image:
        reading = Recognizer().read(image)
    solution = solve_unique(reading.board, reading.cages)
    return {"board": reading.board, "solution": solution,
            "givens": sum(n > 0 for n in reading.board),
            "cages": [{"total": c.total, "cells": c.cells} for c in reading.cages],
            "chooser": reading.chooser,
            "min_confidence": min((s for n, s in zip(reading.board, reading.scores) if n), default=1.0)}


class App:
    def __init__(self, root, register_hotkeys=True, embedded=False, data_dir=None):
        import tkinter as tk
        from tkinter import ttk
        from .wincontrol import WindowsGame
        self.tk, self.ttk, self.root = tk, ttk, root
        self.events = queue.Queue()
        self.closed = threading.Event()
        self.worker = None
        self.control = None
        self.backend = WindowsGame()
        self.data_dir = Path(data_dir) if data_dir is not None else base_dir()/"data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if not embedded:
            root.title(DISPLAY_NAME)
        if not embedded:
            root.geometry("760x840")
        if not embedded:
            root.minsize(700, 780)
        if not embedded:
            root.configure(bg="#f5f7fa")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f5f7fa")
        style.configure("TLabel", background="#f5f7fa", font=("Microsoft YaHei UI", 10))
        style.configure("TButton", font=("Microsoft YaHei UI", 10), padding=6)
        style.configure("TCheckbutton", background="#f5f7fa", font=("Microsoft YaHei UI", 10))
        frame = ttk.Frame(root, padding=18)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text=DISPLAY_NAME, font=("Microsoft YaHei UI", 20, "bold")).pack(anchor="w")
        ttk.Label(frame, text="识别虚线框与总和 → 确认唯一解 → 自动填数 → 继续选关").pack(anchor="w", pady=(4, 12))
        options = ttk.Frame(frame)
        options.pack(fill="x")
        self.difficulty = tk.StringVar(value="当前难度（手动选择）")
        self.continuous = tk.BooleanVar(value=True)
        self.limit = tk.StringVar(value="0")
        ttk.Label(options, text="难度").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.difficulty_widget = ttk.Combobox(options, textvariable=self.difficulty, state="readonly", width=23,
            values=("当前难度（手动选择）", *LEVEL_GROUPS, "依次完成三个难度"))
        self.difficulty_widget.grid(row=0, column=1, sticky="w")
        self.continuous_widget=ttk.Checkbutton(options,text="连续通关",variable=self.continuous)
        self.continuous_widget.grid(row=0,column=2,padx=14)
        ttk.Label(options, text="最多通关（0 = 完成所选难度）").grid(row=1, column=0, columnspan=2, sticky="w", pady=8)
        self.limit_widget=ttk.Spinbox(options,from_=0,to=1000,textvariable=self.limit,width=8)
        self.limit_widget.grid(row=1,column=2,sticky="w")
        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=12)
        self.start_button = ttk.Button(buttons, text="开始", command=self.start)
        self.start_button.pack(side="left", padx=(0, 8))
        ttk.Button(buttons, text="暂停 / 继续", command=self.pause).pack(side="left", padx=(0, 8))
        ttk.Button(buttons, text="停止", command=self.stop).pack(side="left", padx=(0, 8))
        self.check_button = ttk.Button(buttons, text="检查截图", command=self.check_image)
        self.check_button.pack(side="left")
        self.status = tk.StringVar(value="待机 · 打开游戏选关界面后点击“开始”。")
        ttk.Label(frame, textvariable=self.status, wraplength=700, foreground="#24567a").pack(anchor="w", pady=(0, 8))
        preview = ttk.Frame(frame)
        preview.pack(fill="x")
        self.canvas = tk.Canvas(preview, width=342, height=342, bg="white", highlightthickness=0)
        self.canvas.pack(side="left")
        ttk.Label(preview, text="棋盘预览\n\n深色：识别到的已有数字\n蓝色：准备填写的解\n\n开始：留 5 秒切回游戏\n暂停 / 继续\n停止当前任务\n\n切出游戏会暂停。\n鼠标移到屏幕左上角也会停止。",
                  justify="left", foreground="#536575").pack(side="left", padx=20)
        self.draw_board([0] * 81, None)
        ttk.Label(frame, text="运行记录").pack(anchor="w", pady=(12, 4))
        self.log_widget = tk.Text(frame, height=7, state="disabled", font=("Microsoft YaHei UI", 9),
            relief="flat", bg="white", wrap="word")
        self.log_widget.pack(fill="both", expand=True)
        self.log("适用于 2560×1440、16:9 全屏。默认连续通关，最多通关 0 表示完成所选难度。")
        self.log("识别总和角标及虚线框；空棋盘和展开的数字菜单均可读取。")
        self.log("“检查截图”仅识别和求解，不操作游戏。")
        if not embedded:
            root.protocol("WM_DELETE_WINDOW", self.close)
        self.poll_id = root.after(80, self.poll)

    def emit(self, kind, value):
        self.events.put((kind, value))

    def log(self, message):
        line = f"{time.strftime('%H:%M:%S')}  {message}\n"
        self.log_widget.configure(state="normal")
        self.log_widget.insert("end", line)
        self.log_widget.see("end")
        self.log_widget.configure(state="disabled")
        try:
            with (self.data_dir / "运行记录.txt").open("a", encoding="utf-8") as file:
                file.write(line)
        except OSError:
            pass

    def draw_board(self, board, solution, cages=None):
        self.canvas.delete("all")
        for i in range(10):
            width = 3 if i % 3 == 0 else 1
            self.canvas.create_line(i * 38, 0, i * 38, 342, width=width, fill="#91adbf")
            self.canvas.create_line(0, i * 38, 342, i * 38, width=width, fill="#91adbf")
        if cages:
            for cage in cages:
                total = cage["total"] if isinstance(cage, dict) else cage.total
                cells = set(cage["cells"] if isinstance(cage, dict) else cage.cells)
                for i in cells:
                    r, c = divmod(i, 9)
                    x, y = c*38, r*38
                    for neighbor, valid, points in (
                        (i-9, r>0, (x+3,y+3,x+35,y+3)),
                        (i+9, r<8, (x+3,y+35,x+35,y+35)),
                        (i-1, c>0, (x+3,y+3,x+3,y+35)),
                        (i+1, c<8, (x+35,y+3,x+35,y+35))):
                        if not valid or neighbor not in cells:
                            self.canvas.create_line(*points, fill="#ce975b", dash=(3,2))
                anchor = min(cells, key=lambda i:(i%9,i//9))
                r, c = divmod(anchor, 9)
                self.canvas.create_text(c*38+7,r*38+6,text=str(total),fill="#ba721d",font=("Arial",7))
        for i, given in enumerate(board):
            n = given or (solution[i] if solution else 0)
            if n:
                r, c = divmod(i, 9)
                self.canvas.create_text(c * 38 + 19, r * 38 + 19, text=str(n),
                    fill="#343b44" if given else "#4084b1", font=("Microsoft YaHei UI", 14))

    def active(self):
        return self.worker is not None and self.worker.is_alive()

    def start(self):
        from tkinter import messagebox
        from .runner import Control, Runner, Settings
        if self.active():
            return
        try:
            limit=int(self.limit.get())
            if not 0<=limit<=1000:
                raise ValueError()
        except ValueError:
            messagebox.showerror("设置有误","通关盘数请输入 0–1000。")
            return
        self.control = Control()
        settings=Settings(self.continuous.get(),self.difficulty.get(),limit,CLICK_INTERVAL)
        runner = Runner(self.backend, self.control, settings, self.emit, self.data_dir)
        self.start_button.configure(state="disabled")
        self.check_button.configure(state="disabled")
        self.status.set("即将开始，请切回游戏。")
        self.worker = threading.Thread(target=runner.run, daemon=True)
        self.worker.start()

    def pause(self):
        if self.active() and self.control:
            self.control.toggle_pause()
            self.status.set("已暂停，点击“暂停 / 继续” 继续。" if self.control.paused.is_set() else "继续前重新识别。")

    def stop(self):
        if self.control:
            self.control.stop.set()

    def check_image(self):
        from tkinter import filedialog, messagebox
        path = filedialog.askopenfilename(title="选择游戏棋盘截图", filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp")])
        if not path:
            return
        try:
            result = analyze(path)
            self.draw_board(result["board"], result["solution"],result["cages"])
            self.log(f"截图检查通过：{len(result['cages'])} 个虚线框，{result['givens']} 个已有数字，已确认唯一解。")
            self.status.set("截图识别通过；本次没有操作游戏。")
        except Exception as error:
            self.log(str(error))
            messagebox.showerror("截图检查失败", str(error))


    def poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "log":
                    self.log(value)
                    self.status.set(value)
                elif kind == "error":
                    self.log("已停止：" + value)
                    self.log(f"最后的游戏截图：{self.data_dir / 'last_game.png'}")
                    self.status.set("已停止：" + value)
                elif kind == "board":
                    self.draw_board(*value)
                elif kind == "progress":
                    self.status.set(f"正在填写：{value[0]} / {value[1]} 个空格")
                elif kind == "done":
                    self.start_button.configure(state="normal")
                    self.check_button.configure(state="normal")
        except queue.Empty:
            pass
        if not self.closed.is_set():
            self.poll_id = self.root.after(80, self.poll)

    def close(self):
        self.stop()
        self.closed.set()
        self.root.after_cancel(self.poll_id)
        if self.active():
            self.worker.join(timeout=1)
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", help="仅识别给定图片，不点击游戏")
    parser.add_argument("--inspect-menu", help="仅检查选关截图，不操作游戏")
    parser.add_argument("--report", help="将诊断报告写入 JSON 文件")
    parser.add_argument("--smoke-test", action="store_true", help="仅检查程序初始化，不截图、不点击")
    args = parser.parse_args()
    if args.analyze or args.smoke_test or args.inspect_menu:
        try:
            if args.inspect_menu:
                with Image.open(args.inspect_menu) as image:
                    frame=canonical(image)
                catalog=inspect_level_menu(frame)
                report={"ok":True,"group":catalog.group,"remaining":list(catalog.remaining),
                        "completed":catalog.completed,"total":catalog.total}
            elif args.analyze:
                report = {"ok": True, **analyze(args.analyze)}
            else:
                from .wincontrol import enable_physical_pixels, WindowsGame, Input
                enable_physical_pixels()
                Recognizer()
                WindowsGame()
                configure_tk()
                import tkinter as tk
                root = tk.Tk()
                root.withdraw()
                app = App(root, register_hotkeys=False)
                root.update()
                window_title=root.title()
                default_limit=int(app.limit.get())
                app.close()
                report = {"ok": True, "templates": 9, "input_structure_bytes": ctypes.sizeof(Input), "interface": "ok", "game":"Sudoku Killer",
                          "window_title":window_title,"default_limit":default_limit}
        except Exception as error:
            report = {"ok": False, "error": str(error)}
        if args.report:
            Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ok"] else 1
    from .wincontrol import enable_physical_pixels
    enable_physical_pixels()
    configure_tk()
    import tkinter as tk
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
