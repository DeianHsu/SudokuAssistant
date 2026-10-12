"""Coordinate assistant selection and cancellation without touching the mouse."""
from dataclasses import dataclass
import threading


@dataclass
class Endpoint:
    page: object
    start: object
    pause: object
    stop: object


class Coordinator:
    def __init__(self, selected="universe"):
        self.selected = selected
        self.endpoints = {}
        self.closing = False
        self.last_reason = ""
        self.lock = threading.RLock()

    def register(self, key, page, start, pause, stop):
        with self.lock:
            self.endpoints[key] = Endpoint(page, start, pause, stop)

    def active_key(self):
        with self.lock:
            for key, endpoint in self.endpoints.items():
                control = endpoint.page.control
                if endpoint.page.active() and (control is None or not control.stop.is_set()):
                    return key
        return None

    def stop_all(self):
        with self.lock:
            for endpoint in self.endpoints.values():
                endpoint.stop()

    def select(self, key):
        with self.lock:
            if key not in self.endpoints:
                raise ValueError("助手选择无效。")
            if key == self.selected:
                return False
            self.stop_all()
            self.selected = key
            return True

    def start(self, key=None):
        with self.lock:
            self.last_reason = ""
            key = self.selected if key is None else key
            if self.closing or key != self.selected:
                return False
            endpoint = self.endpoints[key]
            if endpoint.page.active():
                self.last_reason = "当前任务仍在运行或结束中；暂停后请点击“暂停 / 继续”。"
                return False
            if self.active_key() is not None:
                self.last_reason = "另一助手仍在运行，请先停止当前任务。"
                return False
            previous_control = endpoint.page.control
            endpoint.start()
            return endpoint.page.control is not previous_control

    def pause(self):
        with self.lock:
            if self.closing:
                return False
            key = self.active_key()
            if key is None:
                return False
            self.endpoints[key].pause()
            return True

    def close(self):
        with self.lock:
            self.closing = True
            self.stop_all()
