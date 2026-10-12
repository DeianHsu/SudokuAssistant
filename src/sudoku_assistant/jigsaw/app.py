"""Chinese launcher for Sudoku Jigsaw. Automation starts from its Start button."""
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
from .vision import Recognizer, canonical, inspect_level_menu, LEVEL_GROUPS
from ..settings import CLICK_INTERVAL, CLICK_INTERVAL_MS

DISPLAY_NAME = "拼图数独（Sudoku Jigsaw）自动助手"


def base_dir():
    return Path(sys.executable).resolve().parent if getattr(sys,"frozen",False) else Path(__file__).resolve().parent


def configure_tk():
    from .tcl_bootstrap import warm_tcl
    warm_tcl()
    root=Path(getattr(sys,"_MEIPASS",sys.base_prefix))
    for name,folder in (("TCL_LIBRARY","tcl8.6"),("TK_LIBRARY","tk8.6")):
        path=root/("_tcl_data" if name=="TCL_LIBRARY" else "_tk_data") if getattr(sys,"frozen",False) else root/"tcl"/folder
        if path.exists():
            os.environ[name]=path.as_posix()


def analyze(path):
    with Image.open(path) as image:
        reading=Recognizer().read(image)
    solution=solve_unique(reading.board,reading.regions)
    scores=[s for n,s in zip(reading.board,reading.scores) if n]
    return {"board":reading.board,"regions":reading.regions,"solution":solution,
            "givens":sum(n>0 for n in reading.board),"chooser":reading.chooser,
            "min_confidence":min(scores) if scores else None}


class App:
    def __init__(self, root, register_hotkeys=True, embedded=False, data_dir=None):
        import tkinter as tk
        from tkinter import ttk
        from .wincontrol import WindowsGame
        self.tk,self.root=tk,root
        self.backend=WindowsGame()
        self.events=queue.Queue()
        self.closed=threading.Event()
        self.worker=self.control=None
        self.data_dir = Path(data_dir) if data_dir is not None else base_dir()/"data"
        self.data_dir.mkdir(parents=True,exist_ok=True)
        if not embedded:
            root.title(DISPLAY_NAME)
        if not embedded:
            root.geometry("780x800")
        if not embedded:
            root.minsize(730,740)
        style=ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel",font=("Microsoft YaHei UI",10))
        style.configure("TButton",font=("Microsoft YaHei UI",10),padding=6)
        style.configure("TCheckbutton",font=("Microsoft YaHei UI",10))
        frame=ttk.Frame(root,padding=18)
        frame.pack(fill="both",expand=True)
        ttk.Label(frame,text=DISPLAY_NAME,font=("Microsoft YaHei UI",20,"bold")).pack(anchor="w")
        ttk.Label(frame,text="识别数字与粗线异形宫 → 求唯一解 → 连续快速填写 → 返回选关继续").pack(anchor="w",pady=(5,14))
        options=ttk.Frame(frame)
        options.pack(fill="x")
        self.difficulty=tk.StringVar(value="当前难度（手动选择）")
        self.continuous=tk.BooleanVar(value=True)
        self.limit=tk.StringVar(value="0")
        ttk.Label(options,text="难度").grid(row=0,column=0,sticky="w")
        self.difficulty_widget=ttk.Combobox(options,textvariable=self.difficulty,state="readonly",width=23,
            values=("当前难度（手动选择）",*LEVEL_GROUPS,"依次完成三个难度"))
        self.difficulty_widget.grid(row=0,column=1,padx=12,sticky="w")
        self.continuous_widget=ttk.Checkbutton(options,text="连续通关",variable=self.continuous)
        self.continuous_widget.grid(row=0,column=2,padx=12)
        ttk.Label(options,text="最多通关（0 = 完成所选难度）").grid(row=1,column=0,columnspan=2,sticky="w",pady=10)
        self.limit_widget=ttk.Spinbox(options,from_=0,to=1000,textvariable=self.limit,width=8)
        self.limit_widget.grid(row=1,column=2,sticky="w")
        buttons=ttk.Frame(frame)
        buttons.pack(fill="x",pady=14)
        self.start_button=ttk.Button(buttons,text="开始",command=self.start)
        self.start_button.pack(side="left",padx=(0,8))
        ttk.Button(buttons,text="暂停 / 继续",command=self.pause).pack(side="left",padx=(0,8))
        ttk.Button(buttons,text="停止",command=self.stop).pack(side="left",padx=(0,8))
        self.check_button=ttk.Button(buttons,text="检查截图",command=self.check_image)
        self.check_button.pack(side="left")
        self.status=tk.StringVar(value="待机 · 打开选关界面或棋盘后点击“开始”。")
        ttk.Label(frame,textvariable=self.status,wraplength=710,foreground="#24567a").pack(anchor="w",pady=(0,10))
        preview=ttk.Frame(frame)
        preview.pack(fill="x")
        self.canvas=tk.Canvas(preview,width=342,height=342,bg="white",highlightthickness=0)
        self.canvas.pack(side="left")
        ttk.Label(preview,text="棋盘与异形宫预览\n\n深色：已有数字\n蓝色：准备填写的解\n粗线：识别出的异形宫\n\n开始：留 5 秒切回游戏\n暂停 / 继续\n停止当前任务\n\n切出游戏会暂停。\n鼠标移至屏幕左上角会停止。",justify="left",foreground="#536575").pack(side="left",padx=20)
        self.draw_board([0]*81,None,None)
        ttk.Label(frame,text="运行记录").pack(anchor="w",pady=(12,4))
        self.log_widget=tk.Text(frame,height=7,state="disabled",font=("Microsoft YaHei UI",9),bg="white",relief="flat",wrap="word")
        self.log_widget.pack(fill="both",expand=True)
        self.log("默认连续通关，上限 0 表示完成所选难度；游戏建议为 2560×1440、16:9 全屏。")
        self.log("支持任意背景色；异形宫须由清晰粗线分隔。检查截图不会操作游戏。")
        if not embedded:
            root.protocol("WM_DELETE_WINDOW",self.close)
        self.poll_id=root.after(80,self.poll)

    def emit(self,kind,value):
        self.events.put((kind,value))

    def log(self,message):
        line=f"{time.strftime('%H:%M:%S')}  {message}\n"
        self.log_widget.configure(state="normal")
        self.log_widget.insert("end",line)
        self.log_widget.see("end")
        self.log_widget.configure(state="disabled")
        try:
            with (self.data_dir/"运行记录.txt").open("a",encoding="utf-8") as file:
                file.write(line)
        except OSError:
            pass

    def draw_board(self,board,solution,regions):
        self.canvas.delete("all")
        for i in range(10):
            self.canvas.create_line(i*38,0,i*38,342,width=1,fill="#ccd5df")
            self.canvas.create_line(0,i*38,342,i*38,width=1,fill="#ccd5df")
        if regions:
            for r in range(9):
                for c in range(9):
                    i=r*9+c
                    if c<8 and regions[i]!=regions[i+1]:
                        self.canvas.create_line((c+1)*38,r*38,(c+1)*38,(r+1)*38,width=3,fill="#374e68")
                    if r<8 and regions[i]!=regions[i+9]:
                        self.canvas.create_line(c*38,(r+1)*38,(c+1)*38,(r+1)*38,width=3,fill="#374e68")
        self.canvas.create_rectangle(1,1,341,341,width=3,outline="#374e68")
        for i,given in enumerate(board):
            n=given or (solution[i] if solution else 0)
            if n:
                r,c=divmod(i,9)
                self.canvas.create_text(c*38+19,r*38+19,text=str(n),fill="#343b44" if given else "#4084b1",font=("Microsoft YaHei UI",14))

    def active(self):
        return self.worker is not None and self.worker.is_alive()

    def start(self):
        from tkinter import messagebox
        from .runner import Runner,Control,Settings
        if self.active():
            return
        try:
            limit=int(self.limit.get())
            if not 0<=limit<=1000:
                raise ValueError()
        except ValueError:
            messagebox.showerror("设置有误","最多通关请输入 0–1000。")
            return
        self.control=Control()
        settings=Settings(self.continuous.get(),self.difficulty.get(),limit,CLICK_INTERVAL)
        runner=Runner(self.backend,self.control,settings,self.emit,self.data_dir)
        for widget in (self.start_button,self.check_button,self.difficulty_widget,self.limit_widget,self.continuous_widget):
            widget.configure(state="disabled")
        self.status.set("即将开始，请切回游戏。")
        self.worker=threading.Thread(target=runner.run,daemon=True)
        self.worker.start()

    def pause(self):
        if self.active() and self.control:
            self.control.toggle_pause()
            self.status.set("已暂停，点击“暂停 / 继续” 继续。" if self.control.paused.is_set() else "继续前重新读取棋盘。")

    def stop(self):
        if self.control:
            self.control.stop.set()

    def check_image(self):
        from tkinter import filedialog,messagebox
        path=filedialog.askopenfilename(title="选择完整游戏截图",filetypes=[("图片","*.png *.jpg *.jpeg *.bmp")])
        if not path:
            return
        try:
            with Image.open(path) as image:
                frame=canonical(image)
            from .vision import level_menu_visible
            if level_menu_visible(frame):
                catalog=inspect_level_menu(frame)
                self.log(f"选关识别通过：{catalog.group}，剩余 {len(catalog.remaining)}/{catalog.total} 关。")
            else:
                result=analyze(path)
                self.draw_board(result["board"],result["solution"],result["regions"])
                self.log(f"截图检查通过：{result['givens']} 个已有数字，9 个异形宫，唯一解。")
            self.status.set("截图识别通过；本次没有操作游戏。")
        except Exception as error:
            self.log(str(error))
            messagebox.showerror("截图检查失败",str(error))


    def poll(self):
        try:
            while True:
                kind,value=self.events.get_nowait()
                if kind=="log":
                    self.log(value)
                    self.status.set(value)
                elif kind=="error":
                    self.log("已停止："+value)
                    self.log(f"诊断截图和识别结果保存在：{self.data_dir}")
                    self.status.set("已停止："+value)
                elif kind=="board":
                    self.draw_board(*value)
                elif kind=="progress":
                    self.status.set(f"正在填写：{value[0]} / {value[1]} 个空格")
                elif kind=="done":
                    for widget in (self.start_button,self.check_button,self.limit_widget,self.continuous_widget):
                        widget.configure(state="normal")
                    self.difficulty_widget.configure(state="readonly")
        except queue.Empty:
            pass
        if not self.closed.is_set():
            self.poll_id=self.root.after(80,self.poll)

    def close(self):
        self.stop()
        self.closed.set()
        self.root.after_cancel(self.poll_id)
        if self.active():
            self.worker.join(timeout=1)
        self.root.destroy()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--analyze",help="仅识别、求解图片，不操作游戏")
    parser.add_argument("--inspect-menu",help="仅检查选关截图")
    parser.add_argument("--report",help="将诊断报告保存为 JSON 文件")
    parser.add_argument("--smoke-test",action="store_true",help="仅检查初始化，不截图、不点击")
    args=parser.parse_args()
    if args.analyze or args.inspect_menu or args.smoke_test:
        try:
            if args.analyze:
                report={"ok":True,**analyze(args.analyze)}
            elif args.inspect_menu:
                with Image.open(args.inspect_menu) as image:
                    catalog=inspect_level_menu(canonical(image))
                report={"ok":True,"group":catalog.group,"remaining":catalog.remaining,
                        "completed":catalog.completed,"total":catalog.total}
            else:
                from .wincontrol import enable_physical_pixels,Input
                enable_physical_pixels()
                Recognizer()
                configure_tk()
                import tkinter as tk
                root=tk.Tk()
                root.withdraw()
                app=App(root,register_hotkeys=False)
                root.update()
                defaults={"limit":int(app.limit.get()),"continuous":bool(app.continuous.get()),
                          "interval_ms":CLICK_INTERVAL_MS}
                app.close()
                report={"ok":True,"interface":"ok","templates":9,"input_structure_bytes":ctypes.sizeof(Input),
                        "version":"1.5","defaults":defaults}
        except Exception as error:
            report={"ok":False,"error":str(error)}
        if args.report:
            Path(args.report).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        else:
            print(json.dumps(report,ensure_ascii=False,indent=2))
        return 0 if report["ok"] else 1
    from .wincontrol import enable_physical_pixels
    enable_physical_pixels()
    configure_tk()
    import tkinter as tk
    root=tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__=="__main__":
    raise SystemExit(main())
