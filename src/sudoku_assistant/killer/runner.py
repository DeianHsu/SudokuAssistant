"""Read, solve, fill and advance through the three Killer Sudoku level groups."""
from dataclasses import dataclass
from pathlib import Path
import json
import threading
import time
from .solver import solve_unique
from .vision import (Recognizer, RecognitionError, MenuNotReady, canonical, board_visible,
                    level_menu_visible, inspect_level_menu,
                    cell_center, digit_point, DIFFICULTIES, LEVEL_GROUPS)
from .wincontrol import ForegroundLost
from ..settings import CLICK_INTERVAL


class Stopped(Exception):
    pass


class RestartReading(Exception):
    pass


@dataclass
class Settings:
    continuous: bool = True
    difficulty: str = "当前难度（手动选择）"
    limit: int = 0
    interval: float = CLICK_INTERVAL


class Control:
    def __init__(self):
        self.stop=threading.Event()
        self.paused=threading.Event()

    def toggle_pause(self):
        if self.paused.is_set():
            self.paused.clear()
        else:
            self.paused.set()


class Runner:
    completion_timeout=20
    menu_timeout=8
    menu_settle=.18
    reading_timeout=8

    def __init__(self,backend,control,settings,emit,data_dir,recognizer=None):
        self.backend,self.control,self.settings,self.emit=backend,control,settings,emit
        self.data_dir=Path(data_dir)
        if hasattr(self.backend,"set_stop_event"):
            self.backend.set_stop_event(self.control.stop)
        self.recognizer=recognizer or Recognizer()
        self.completed=0
        self.last_capture=None
        self.last_reading=None
        self.pending_completion=False
        self.pending_level=None
        self.backend.set_timing(settings.interval)

    def log(self,message):
        self.emit("log",message)

    def check(self):
        if self.control.stop.is_set():
            raise Stopped()
        if not self.control.paused.is_set():
            return
        while True:
            self.log("已暂停；点击“暂停 / 继续”后，5 秒内切回游戏。")
            while self.control.paused.is_set():
                if self.control.stop.wait(.04):
                    raise Stopped()
            self.log("已继续，5 秒后重新读取，请切回游戏。")
            deadline = time.monotonic()+5
            while time.monotonic() < deadline:
                if self.control.stop.wait(min(.04, max(0, deadline-time.monotonic()))):
                    raise Stopped()
                if self.control.paused.is_set():
                    break
            if not self.control.paused.is_set():
                raise RestartReading()

    def wait(self,seconds):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            self.check()
            self.control.stop.wait(min(.04,max(0,deadline-time.monotonic())))

    def capture(self):
        self.check()
        self.last_capture=self.backend.capture()
        return canonical(self.last_capture)

    def click(self,point,menu=False,final_input=False):
        self.backend.set_timing(max(.03,self.settings.interval) if menu else self.settings.interval)
        try:
            self.backend.click(point,self.check)
            if final_input:
                # Preserve completion state even if pause arrives during the release wait.
                self.pending_completion=True
            self.wait(max(.05,self.settings.interval) if menu else self.settings.interval)
        finally:
            self.backend.set_timing(self.settings.interval)

    def stable_menu(self,expected_group=None):
        deadline=time.monotonic()+self.menu_timeout
        previous=None
        last_error="选关界面尚未稳定。"
        while time.monotonic()<deadline:
            try:
                frame=self.capture()
                catalog=inspect_level_menu(frame,expected_group)
                signature=(catalog.group,catalog.remaining,catalog.completed)
                if signature==previous:
                    return frame,catalog
                previous=signature
            except MenuNotReady as error:
                previous=None
                last_error=str(error)
            self.wait(self.menu_settle)
        raise RecognitionError(f"未能确认完整选关界面，未跳过当前难度。{last_error}")

    def stable_reading(self):
        reading_started=time.perf_counter()
        # Remove the pointer highlight from filled cells before reading their glyphs.
        self.backend.move((1740,1000),self.check)
        self.wait(.30)
        previous=None
        previous_reading=None
        deadline=time.monotonic()+self.reading_timeout
        last_error="连续两次的数字、虚线框或菜单状态尚未一致。"
        while time.monotonic()<deadline:
            try:
                frame=self.capture()
                same_frame=getattr(self.recognizer,"same_frame",None)
                if previous_reading is not None and callable(same_frame) and same_frame(previous_reading.frame,frame):
                    second=previous_reading
                    break
                second=self.recognizer.read(frame)
                signature=(second.board,second.cages,second.chooser)
                if signature==previous:
                    break
                previous=signature
                previous_reading=second
            except RecognitionError as error:
                last_error=str(error)
                previous=None
                previous_reading=None
            self.wait(.20)
        else:
            raise RecognitionError(f"未能确认稳定棋盘，已停止点击。{last_error}")
        self.last_reading=second
        reading_seconds=time.perf_counter()-reading_started
        self.log("棋盘读取完成，正在求解并确认唯一解。")
        solving_started=time.perf_counter()
        solution=solve_unique(second.board,second.cages,check=self.check)
        solving_seconds=time.perf_counter()-solving_started
        self.emit("board",(second.board,solution,second.cages))
        scores=[s for n,s in zip(second.board,second.scores) if n]
        self.log(f"识别到 {sum(n>0 for n in second.board)} 个已有数字；{len(second.cages)} 个虚线框，总和 405；已确认唯一解。")
        self.log(f"本盘耗时：读取及稳定确认 {reading_seconds:.2f} 秒；求解及唯一解确认 {solving_seconds:.2f} 秒。")
        if scores:
            self.log(f"最低数字匹配度 {min(scores):.0%}。")
        return second,solution

    def fill_board(self):
        reading,solution=self.stable_reading()
        empty=[i for i,n in enumerate(reading.board) if not n]
        if not empty:
            # The last input may have completed the puzzle just before a pause.
            return self.wait_for_menu(count=True)
        if reading.chooser is not None:
            empty.remove(reading.chooser)
            empty.insert(0,reading.chooser)
        self.log(f"解已确定，按固定节拍连续填写 {len(empty)} 个空格。")
        for position,i in enumerate(empty):
            self.check()
            r,c=divmod(i,9)
            if position or reading.chooser!=i:
                self.click(cell_center(r,c))
            # The solution and menu positions are fixed; do not capture or
            # recognize popup animation frames while entering the digits.
            self.click(digit_point(r,c,solution[i]),final_input=position==len(empty)-1)
            self.emit("progress",(position+1,len(empty)))
        self.backend.move((1740,1000),self.check)
        self.log("填写结束，等待通关动画及返回选关。")
        return self.wait_for_menu(count=True)

    def confirm_completion(self,catalog):
        if self.pending_level is not None:
            group,level=self.pending_level
            if catalog.group!=group or level in catalog.remaining:
                raise RecognitionError("返回选关后，刚填写的关卡未被确认为已完成，已停止。")
        self.completed+=1
        self.pending_completion=False
        self.pending_level=None
        self.log(f"本次已确认通关 {self.completed} 盘。")

    def wait_for_menu(self,count=False):
        deadline=time.monotonic()+self.completion_timeout
        while time.monotonic()<deadline:
            frame=self.capture()
            if level_menu_visible(frame):
                frame,catalog=self.stable_menu()
                if count:
                    self.confirm_completion(catalog)
                return frame
            self.wait(.25)
        raise RecognitionError("没有等到通关后选关界面，已停止。若有漏填，请重新开始补齐空格。")

    def save_diagnostic(self):
        self.data_dir.mkdir(parents=True,exist_ok=True)
        if self.last_capture is not None:
            self.last_capture.save(self.data_dir/"last_game.png")
        if self.last_reading:
            reading=self.last_reading
            report={"board":reading.board,"cages":[{"total":c.total,"cells":c.cells} for c in reading.cages],
                    "scores":reading.scores,"chooser":reading.chooser}
            (self.data_dir/"last_reading.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")

    def run(self):
        difficulty=self.settings.difficulty
        groups=list(LEVEL_GROUPS) if difficulty=="依次完成三个难度" else [difficulty]
        index=0
        chosen=False
        try:
            if difficulty not in (*LEVEL_GROUPS,"当前难度（手动选择）","依次完成三个难度"):
                raise RecognitionError("难度设置有误。")
            if self.settings.limit<0:
                raise RecognitionError("最多通关次数不能为负数。")
            mode="连续通关" if self.settings.continuous else "仅完成一盘"
            target=f"最多 {self.settings.limit} 盘" if self.settings.limit else "完成所选难度"
            self.log(f"本次设置：{mode}；{target}。")
            self.log("5 秒后开始，请切回 杀手数独（Sudoku Killer）；切出游戏会暂停。")
            while True:
                try:
                    self.wait(5)
                    break
                except RestartReading:
                    self.log("已继续，5 秒后重新读取，请切回游戏。")
            while True:
                try:
                    frame=self.capture()
                    if self.pending_completion:
                        frame=self.wait_for_menu(count=True)
                    elif board_visible(frame):
                        if groups[index] in DIFFICULTIES and not chosen:
                            raise RecognitionError("自动选择难度请从选关界面开始；从棋盘开始请选择“当前难度”。")
                        frame=self.fill_board()
                    if self.completed and (not self.settings.continuous or
                                           self.settings.limit and self.completed>=self.settings.limit):
                        if not self.settings.continuous:
                            self.log("已完成本盘；未勾选连续通关，自动操作结束。")
                        else:
                            self.log(f"已完成 {self.completed} 盘，达到最多通关 {self.settings.limit} 盘的设置，自动操作结束。")
                        return
                    if not level_menu_visible(frame):
                        raise RecognitionError("当前画面不是完整棋盘或选关界面，请关闭弹窗。")
                    group=groups[index]
                    if group in DIFFICULTIES and not chosen:
                        self.click(DIFFICULTIES[group],menu=True)
                        self.wait(.6)
                        self.stable_menu(group)
                        chosen=True
                        continue
                    _,catalog=self.stable_menu(group if group in LEVEL_GROUPS else None)
                    if not catalog.remaining:
                        if index+1<len(groups):
                            index+=1
                            chosen=False
                            self.log(f"已确认{catalog.group}的 {catalog.total} 关全部完成，切换到{groups[index]}。")
                            continue
                        self.log(f"已确认{catalog.group}的 {catalog.total} 关全部完成，自动操作结束。")
                        return
                    level=catalog.remaining[0]
                    self.log(f"{catalog.group}尚有 {len(catalog.remaining)} 关，进入第 {level} 关。")
                    self.pending_level=(catalog.group,level)
                    self.click(catalog.centers[0],menu=True)
                    self.wait(.6)
                    deadline=time.monotonic()+8
                    while not board_visible(self.capture()):
                        if time.monotonic()>deadline:
                            raise RecognitionError("选关后没有出现完整棋盘，已停止。")
                        self.wait(.2)
                except ForegroundLost as error:
                    self.control.paused.set()
                    self.log(str(error))
                    try:
                        self.check()
                    except RestartReading:
                        continue
                except RestartReading:
                    self.log("继续前重新识别数字和虚线框。")
                    continue
        except Stopped:
            self.log(f"已停止。本次确认通关 {self.completed} 盘。")
        except Exception as error:
            try:
                self.save_diagnostic()
            except OSError:
                pass
            self.emit("error",str(error))
        finally:
            self.emit("done",self.completed)
