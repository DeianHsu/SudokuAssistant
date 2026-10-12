"""Small Windows API adapter. Importing this file does not click or capture."""
import ctypes
from ctypes import wintypes
import os
import time

from PIL import ImageGrab
from ..settings import CLICK_INTERVAL


class ForegroundLost(RuntimeError):
    pass


def enable_physical_pixels():
    try:
        api = ctypes.WinDLL("user32", use_last_error=True)
        api.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        api.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except (AttributeError, OSError):
        ctypes.windll.user32.SetProcessDPIAware()


class MouseInput(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class KeyboardInput(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class InputUnion(ctypes.Union):
    _fields_ = [("mi", MouseInput), ("ki", KeyboardInput)]


class Input(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", InputUnion)]


class WindowsGame:
    def __init__(self):
        self.hold_time = CLICK_INTERVAL
        self.api = ctypes.WinDLL("user32", use_last_error=True)
        self.api.GetForegroundWindow.restype = wintypes.HWND
        self.api.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        self.api.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        self.api.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        self.api.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
        self.api.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(Input), ctypes.c_int]
        self.api.SendInput.restype = wintypes.UINT
        self.api.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
        self.api.GetAsyncKeyState.argtypes = [ctypes.c_int]
        self.api.GetAsyncKeyState.restype = ctypes.c_short
        self.hwnd, self.rect = None, None

    def foreground(self):
        hwnd = self.api.GetForegroundWindow()
        title = ctypes.create_unicode_buffer(512)
        self.api.GetWindowTextW(hwnd, title, len(title))
        process_id = wintypes.DWORD()
        self.api.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
        if process_id.value == os.getpid():
            return None
        name = "".join(ch for ch in title.value.lower() if ch.isalnum())
        return hwnd if "sudokujigsaw" in name else None

    def key_down(self, code):
        return bool(self.api.GetAsyncKeyState(code) & 0x8000)

    def client_rect(self, hwnd):
        rect, point = wintypes.RECT(), wintypes.POINT()
        if not self.api.GetClientRect(hwnd, ctypes.byref(rect)) or not self.api.ClientToScreen(hwnd, ctypes.byref(point)):
            raise ForegroundLost("无法取得游戏窗口位置。")
        return point.x, point.y, point.x + rect.right, point.y + rect.bottom

    def capture(self):
        hwnd = self.foreground()
        if not hwnd:
            raise ForegroundLost("游戏不在前台，已暂停。切回游戏后点击“暂停 / 继续” 继续。")
        rect = self.client_rect(hwnd)
        if rect[2] - rect[0] < 800 or rect[3] - rect[1] < 450:
            raise ForegroundLost("游戏窗口太小或已经最小化。")
        self.hwnd, self.rect = hwnd, rect
        image = ImageGrab.grab(bbox=rect, all_screens=True).convert("RGB")
        self.guard()
        return image

    def guard(self):
        if not self.hwnd or self.api.GetForegroundWindow() != self.hwnd:
            raise ForegroundLost("游戏不在前台，已暂停。切回游戏后点击“暂停 / 继续” 继续。")
        if self.client_rect(self.hwnd) != self.rect:
            raise ForegroundLost("窗口位置或大小发生变化，已暂停。点击“暂停 / 继续” 重新识别。")
        point = wintypes.POINT()
        self.api.GetCursorPos(ctypes.byref(point))
        vx, vy = self.api.GetSystemMetrics(76), self.api.GetSystemMetrics(77)
        if point.x <= vx + 2 and point.y <= vy + 2:
            raise RuntimeError("鼠标进入屏幕左上角，已紧急停止。")

    def send(self, *events):
        array = (Input * len(events))(*events)
        if self.api.SendInput(len(events), array, ctypes.sizeof(Input)) != len(events):
            raise RuntimeError("Windows 未接受鼠标输入。请确保游戏和助手均以普通权限运行。")

    def set_timing(self, seconds):
        if not 0.001 <= seconds <= 1.0:
            raise ValueError("点击节拍需在 1–1000 毫秒之间。")
        self.hold_time = seconds

    def movement(self, point):
        left, top, right, bottom = self.rect
        x = left + round(point[0] / 2048 * (right - left))
        y = top + round(point[1] / 1152 * (bottom - top))
        if not left < x < right or not top < y < bottom:
            raise RuntimeError("计算出的点击位置超出游戏窗口，已停止。")
        vx, vy = self.api.GetSystemMetrics(76), self.api.GetSystemMetrics(77)
        vw, vh = self.api.GetSystemMetrics(78), self.api.GetSystemMetrics(79)
        return Input(type=0, mi=MouseInput(dx=round((x - vx) * 65535 / (vw - 1)),
                  dy=round((y - vy) * 65535 / (vh - 1)), dwFlags=0x8000 | 0x4000 | 0x0001))

    def move(self, point, check):
        check()
        self.guard()
        self.send(self.movement(point))

    def set_stop_event(self, event):
        self.stop_event = event

    def click(self, point, check):
        check()
        self.guard()
        press = self.movement(point)
        press.mi.dwFlags |= 0x0002
        try:
            # Move and press in one input event. Leave one 60 Hz frame for the
            # game to observe each edge; the runner waits after release.
            self.send(press)
            stop_event = getattr(self, "stop_event", None)
            if stop_event is None:
                time.sleep(self.hold_time)
            elif stop_event.wait(self.hold_time):
                check()
        finally:
            self.send(Input(type=0, mi=MouseInput(dwFlags=0x0004)))
