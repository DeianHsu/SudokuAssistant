from dataclasses import dataclass
from pathlib import Path
import threading
import time

import numpy as np

from .solver import solve_unique
from .vision import (Recognizer, RecognitionError, canonical, board_visible, level_menu_visible,
                    available_levels, cell_center, cell_box, digit_point, DIFFICULTIES)
from .vision import completion_visible, LEVEL_GROUPS, inspect_level_menu, MenuNotReady, ENDLESS_CHOICES, endless_menu_visible
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
    mode: str = "关卡模式"
    endless_difficulty: str = "简单"
    endless_limit: int = 10


class Control:
    def __init__(self):
        self.stop = threading.Event()
        self.paused = threading.Event()

    def toggle_pause(self):
        if self.paused.is_set():
            self.paused.clear()
        else:
            self.paused.set()


class Runner:
    completion_timeout = 15
    menu_timeout = 6
    menu_settle = 0.18

    def __init__(self, backend, control, settings, emit, data_dir, recognizer=None):
        self.backend, self.control, self.settings, self.emit = backend, control, settings, emit
        self.data_dir = Path(data_dir)
        if hasattr(self.backend, "set_stop_event"):
            self.backend.set_stop_event(self.control.stop)
        self.recognizer = recognizer or Recognizer()
        self.completed = 0
        self.last_capture = None
        self.starting_frame = None
        self.pending_endless_completion = False
        if hasattr(self.backend, "set_timing"):
            self.backend.set_timing(settings.interval)

    def log(self, message):
        self.emit("log", message)

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

    def wait(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.check()
            self.control.stop.wait(min(0.04, max(0, end - time.monotonic())))

    def capture(self):
        self.check()
        self.last_capture = self.backend.capture()
        return canonical(self.last_capture)

    def click(self, point, menu=False, final_endless_input=False):
        configurable=menu and hasattr(self.backend,"set_timing")
        if configurable:
            self.backend.set_timing(max(0.03,self.settings.interval))
        try:
            self.backend.click(point,self.check)
            if final_endless_input:
                self.pending_endless_completion=True
            self.wait(max(0.04,self.settings.interval) if menu else self.settings.interval)
        finally:
            if configurable:
                self.backend.set_timing(self.settings.interval)

    def stable_menu(self, expected_group=None):
        deadline=time.monotonic()+self.menu_timeout
        previous=None;last_error="选关界面尚未稳定。"
        while time.monotonic()<deadline:
            try:
                frame=self.capture()
                catalog=inspect_level_menu(frame,expected_group)
                signature=(catalog.group,catalog.remaining,catalog.completed)
                if signature==previous:
                    return frame,catalog
                previous=signature
            except MenuNotReady as error:
                previous=None;last_error=str(error)
            self.wait(self.menu_settle)
        raise RecognitionError(f"未能确认完整选关界面，已停止，未跳过当前难度。{last_error}")

    def stable_reading(self):
        first = self.recognizer.read(self.capture())
        self.wait(0.2)
        second = self.recognizer.read(self.capture())
        if first.board != second.board:
            raise RecognitionError("连续两次识别结果不同，请等界面稳定后重新开始。")
        solution = solve_unique(second.board, check=self.check)
        self.starting_frame = second.frame
        self.emit("board", (second.board, solution))
        count = sum(n > 0 for n in second.board)
        scores = [s for n, s in zip(second.board, second.scores) if n]
        self.log(f"识别到 {count} 个已有数字；最低匹配度 {min(scores):.0%}；已确认唯一解。")
        return second.board, solution

    def stable_endless_menu(self):
        deadline=time.monotonic()+self.menu_timeout
        seen=False
        while time.monotonic()<deadline:
            frame=self.capture()
            ready=endless_menu_visible(frame)
            if ready and seen:
                return frame
            seen=ready
            self.wait(self.menu_settle)
        raise RecognitionError("无尽选择界面尚未完整显示，已停止。")

    def confirm_completion(self):
        self.completed+=1
        self.pending_endless_completion=False
        if self.settings.mode=="无尽模式":
            target=str(self.settings.endless_limit) if self.settings.endless_limit else "不限"
            self.log(f"本次无尽已完成 {self.completed}/{target} 盘，已返回无尽菜单。")
        else:
            self.log(f"已通关 {self.completed} 盘，游戏已返回选关界面。")

    def fill_board(self):
        board, solution = self.stable_reading()
        empty = [i for i, n in enumerate(board) if not n]
        if not empty:
            raise RecognitionError("当前棋盘已填满。请回到选关界面再启动。")
        # At startup/resume, an empty cell may already have its chooser open.
        # Its center is the chooser's FIVE button, so do not click it twice.
        first_r, first_c = divmod(empty[0], 9)
        x0, y0, x1, y1 = cell_box(first_r, first_c)
        first_patch = np.asarray(self.starting_frame)[round(y0+5):round(y1-5),round(x0+5):round(x1-5)].astype(int)
        orange = (first_patch[:,:,0]-first_patch[:,:,1] > 15) & (first_patch[:,:,1]-first_patch[:,:,2] > 20)
        first_menu_open = np.mean(orange) > 0.45
        self.log(f"解已确定，按解连续填写 {len(empty)} 个空格。")
        for position, i in enumerate(empty):
            self.check()
            r, c = divmod(i, 9)
            if position or not first_menu_open:
                self.click(cell_center(r, c))
            self.click(digit_point(r,c,solution[i]),final_endless_input=(
                self.settings.mode=="无尽模式" and position==len(empty)-1))
            self.emit("progress", (position + 1, len(empty)))
        self.backend.move((1740, 1000), self.check)
        self.log("填写结束，等待游戏完成提示并返回选关界面。")
        return self.wait_for_menu(count=True)

    def wait_for_menu(self, count=False):
        deadline = time.monotonic() + self.completion_timeout
        while time.monotonic() < deadline:
            frame = self.capture()
            endless=self.settings.mode=="无尽模式"
            ready=endless_menu_visible(frame) if endless else level_menu_visible(frame)
            if ready:
                if endless:
                    frame=self.stable_endless_menu()
                else:
                    frame,_=self.stable_menu()
                if count:
                    self.confirm_completion()
                return frame
            self.wait(0.25)
        raise RecognitionError("填写结束后没有等到选关界面。已停止；若有漏填，可以重新开始补齐空格。")

    def save_diagnostic(self):
        if self.last_capture is not None:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.last_capture.save(self.data_dir / "last_game.png")

    def run_endless(self):
        if self.settings.endless_difficulty not in ENDLESS_CHOICES or self.settings.endless_limit<0:
            raise RecognitionError("无尽难度或次数设置有误。")
        target=str(self.settings.endless_limit) if self.settings.endless_limit else "不限"
        self.log(f"无尽模式：{self.settings.endless_difficulty}，本次目标 {target} 盘。")
        while True:
            try:
                self.check()
                if self.settings.endless_limit and self.completed>=self.settings.endless_limit:
                    self.log("已达到无尽次数，自动操作结束。")
                    return
                if self.pending_endless_completion:
                    self.wait_for_menu(count=True)
                    continue
                frame=self.capture()
                if completion_visible(frame):
                    frame=self.wait_for_menu(count=self.pending_endless_completion)
                if board_visible(frame):
                    self.fill_board()
                    continue
                if endless_menu_visible(frame):
                    self.stable_endless_menu()
                    if self.pending_endless_completion:
                        self.confirm_completion()
                        continue
                    if self.settings.endless_limit and self.completed>=self.settings.endless_limit:
                        continue
                    self.log(f"开始第 {self.completed+1} 盘无尽，难度：{self.settings.endless_difficulty}。")
                    self.click(ENDLESS_CHOICES[self.settings.endless_difficulty],menu=True)
                    self.wait(0.6)
                    deadline=time.monotonic()+6
                    while not board_visible(self.capture()):
                        if time.monotonic()>deadline:
                            raise RecognitionError("选择无尽难度后没有出现棋盘，已停止。")
                        self.wait(0.2)
                    continue
                if level_menu_visible(frame):
                    self.click(DIFFICULTIES["无限"],menu=True)
                    self.wait(0.6)
                    self.stable_endless_menu()
                    continue
                raise RecognitionError("请从无尽选择界面或无尽数独棋盘开始。")
            except ForegroundLost as error:
                self.control.paused.set()
                self.log(str(error))
                try:
                    self.check()
                except RestartReading:
                    continue
            except RestartReading:
                self.log("继续前重新读取无尽游戏画面。")
                continue

    def run(self):
        chosen_difficulty = False
        if self.settings.difficulty == "依次完成全部七组":
            difficulties = list(LEVEL_GROUPS)
        elif self.settings.difficulty == "依次完成三个难度":
            difficulties = ["简单", "中等", "困难"]
        else:
            difficulties = [self.settings.difficulty]
        difficulty_index = 0
        last_chosen = None
        awaiting_menu = False
        try:
            self.log("5 秒后开始，请切回 Sudoku Universe；切出游戏会暂停。")
            while True:
                try:
                    self.wait(5)
                    break
                except RestartReading:
                    self.log("已继续，5 秒后重新读取游戏画面，请切回游戏。")
            if self.settings.mode=="无尽模式":
                self.run_endless()
                return
            while True:
                try:
                    frame = self.capture()
                    if completion_visible(frame):
                        self.log("等待关卡完成动画结束。")
                        frame = self.wait_for_menu()
                    if board_visible(frame):
                        if difficulties[difficulty_index] in DIFFICULTIES and not chosen_difficulty:
                            raise RecognitionError("自动选择难度请从选关界面开始；从棋盘开始请选择“当前难度”。")
                        frame = self.fill_board()
                        awaiting_menu = True
                        if not self.settings.continuous or (self.settings.limit and self.completed >= self.settings.limit):
                            self.log("已达到本次目标，自动操作结束。")
                            return
                    if not level_menu_visible(frame):
                        raise RecognitionError("当前界面既不是棋盘也不是选关菜单，已停止。")
                    if difficulties[difficulty_index] in DIFFICULTIES and not chosen_difficulty:
                        self.click(DIFFICULTIES[difficulties[difficulty_index]],menu=True)
                        self.wait(0.6)
                        self.stable_menu(difficulties[difficulty_index])
                        chosen_difficulty = True
                        last_chosen = None
                        continue
                    expected=difficulties[difficulty_index] if difficulties[difficulty_index] in LEVEL_GROUPS else None
                    frame,catalog=self.stable_menu(expected)
                    buttons=list(catalog.centers)
                    if not buttons:
                        if difficulty_index + 1 < len(difficulties):
                            difficulty_index += 1
                            chosen_difficulty = False
                            awaiting_menu = False
                            last_chosen = None
                            self.log(f"已确认 {catalog.group} 的 {catalog.completed}/{catalog.total} 个关卡全部完成，切换到{difficulties[difficulty_index]}。")
                            continue
                        self.log(f"已确认 {catalog.group} 的 {catalog.completed}/{catalog.total} 个关卡全部完成，自动操作结束。")
                        return
                    if last_chosen is not None and awaiting_menu:
                        if any(abs(p[0] - last_chosen[0]) < 15 and abs(p[1] - last_chosen[1]) < 15 for p in buttons):
                            raise RecognitionError("刚完成的关卡仍显示为未完成，已停止以避免反复刷同一关。")
                        awaiting_menu = False
                    last_chosen = buttons[0]
                    self.log(f"{catalog.group} 尚有 {len(catalog.remaining)} 关，选择第 {catalog.remaining[0]} 关。")
                    self.click(last_chosen,menu=True)
                    self.wait(0.6)
                    deadline = time.monotonic() + 6
                    while not board_visible(self.capture()):
                        if time.monotonic() > deadline:
                            raise RecognitionError("选择关卡后没有出现棋盘，已停止。")
                        self.wait(0.2)
                except ForegroundLost as error:
                    self.control.paused.set()
                    self.log(str(error))
                    try:
                        self.check()
                    except RestartReading:
                        continue
                except RestartReading:
                    self.log("继续前重新读取游戏画面。")
                    continue
        except Stopped:
            self.log(f"已停止。本次确认通关 {self.completed} 盘。")
        except Exception as error:
            try:
                self.save_diagnostic()
            except OSError:
                pass
            self.emit("error", str(error))
        finally:
            self.emit("done", self.completed)
