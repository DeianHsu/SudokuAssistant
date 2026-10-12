"""Integration regressions use fake mouse transports and never click a game."""
import ctypes
import importlib
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from sudoku_assistant.coordinator import Coordinator
from sudoku_assistant.main import ASSISTANTS, AssistantWindow, configure_tk
from sudoku_assistant.universe.runner import Control


ROOT = Path(__file__).resolve().parents[1]
KEYS = tuple(ASSISTANTS)


class FakePage:
    def __init__(self, control_factory=Control):
        self.control = None
        self.control_factory = control_factory
        self.running = False
        self.starts = 0

    def active(self):
        return self.running

    def start(self):
        self.control = self.control_factory()
        self.running = True
        self.starts += 1

    def pause(self):
        self.control.toggle_pause()

    def stop(self):
        if self.control:
            self.control.stop.set()


def fake_coordinator(selected="universe", control_factory=Control):
    coordinator = Coordinator(selected)
    pages = {key: FakePage(control_factory) for key in KEYS}
    for key, page in pages.items():
        coordinator.register(key, page, page.start, page.pause, page.stop)
    return coordinator, pages


class CoordinatorTests(unittest.TestCase):
    def test_switching_running_or_paused_assistant_stops_old_task(self):
        for paused in (False, True):
            with self.subTest(paused=paused):
                coordinator, pages = fake_coordinator()
                self.assertTrue(coordinator.start())
                if paused:
                    coordinator.pause()
                old_control = pages["universe"].control
                coordinator.select("killer")
                self.assertTrue(old_control.stop.is_set())
                self.assertIsNone(coordinator.active_key())
                self.assertTrue(coordinator.start())
                self.assertEqual(pages["killer"].starts, 1)
                self.assertFalse(pages["killer"].control.stop.is_set())

    def test_hidden_assistant_cannot_start_and_same_selection_keeps_running(self):
        coordinator, pages = fake_coordinator()
        self.assertFalse(coordinator.start("jigsaw"))
        self.assertTrue(coordinator.start())
        self.assertFalse(coordinator.select("universe"))
        self.assertFalse(pages["universe"].control.stop.is_set())
        self.assertFalse(coordinator.start())
        self.assertEqual(pages["universe"].starts, 1)

    def test_stop_and_close_cancel_every_task_and_close_blocks_restart(self):
        coordinator, pages = fake_coordinator()
        coordinator.start()
        coordinator.close()
        self.assertTrue(pages["universe"].control.stop.is_set())
        pages["universe"].running = False
        self.assertFalse(coordinator.start())

    def test_pause_uses_running_assistant_and_stopped_task_cannot_resume(self):
        coordinator, pages = fake_coordinator("jigsaw")
        coordinator.start()
        self.assertTrue(coordinator.pause())
        self.assertTrue(pages["jigsaw"].control.paused.is_set())
        self.assertTrue(coordinator.pause())
        self.assertFalse(pages["jigsaw"].control.paused.is_set())
        coordinator.stop_all()
        self.assertFalse(coordinator.pause())

    def test_switch_during_mouse_press_releases_immediately_for_all_games(self):
        for key in KEYS:
            with self.subTest(assistant=key):
                runner_module = importlib.import_module(f"sudoku_assistant.{key}.runner")
                windows = importlib.import_module(f"sudoku_assistant.{key}.wincontrol")

                class FakeTransport(windows.WindowsGame):
                    def __init__(self):
                        self.events = []
                        self.hold_time = 1

                    def guard(self):
                        pass

                    def movement(self, point):
                        return windows.Input(type=0, mi=windows.MouseInput(dwFlags=0xC001))

                    def send(self, *events):
                        self.events.extend(int(event.mi.dwFlags) for event in events)

                coordinator, pages = fake_coordinator(key, runner_module.Control)
                coordinator.start()
                control = pages[key].control
                transport = FakeTransport()
                runner = runner_module.Runner(transport, control, runner_module.Settings(interval=1),
                                              lambda *_: None, ROOT/"tests", recognizer=object())
                next_key = next(other for other in KEYS if other != key)

                def switch_during_hold(seconds):
                    self.assertEqual(seconds, 1)
                    coordinator.select(next_key)
                    return control.stop.is_set()

                with patch.object(control.stop, "wait", side_effect=switch_during_hold):
                    with self.assertRaises(runner_module.Stopped):
                        runner.click((700, 400))
                self.assertEqual(len(transport.events), 2)
                self.assertTrue(transport.events[0] & 0x0002)
                self.assertEqual(transport.events[-1], 0x0004)


class ResumeTests(unittest.TestCase):
    def test_button_resume_gives_five_seconds_before_reading_all_games(self):
        for key in KEYS:
            with self.subTest(assistant=key):
                module = importlib.import_module(f"sudoku_assistant.{key}.runner")

                class Backend:
                    def set_timing(self, value):
                        pass

                control = module.Control()
                control.paused.set()
                runner = module.Runner(Backend(), control, module.Settings(), lambda *_: None,
                                       ROOT/"tests", recognizer=object())
                clock = [0.]

                def wait(seconds):
                    clock[0] += seconds
                    control.paused.clear()
                    return False

                with patch.object(control.stop, "wait", side_effect=wait), \
                     patch.object(module.time, "monotonic", side_effect=lambda: clock[0]):
                    with self.assertRaises(module.RestartReading):
                        runner.check()
                self.assertGreaterEqual(clock[0], 5)

    def test_universe_search_can_be_cancelled_before_more_work(self):
        from sudoku_assistant.universe.solver import solve_unique

        class Cancelled(Exception):
            pass

        def cancelled():
            raise Cancelled()

        with self.assertRaises(Cancelled):
            solve_unique([0]*81, check=cancelled)

    def test_universe_does_not_mistake_own_window_for_game(self):
        import os
        from ctypes import wintypes
        from sudoku_assistant.universe.wincontrol import WindowsGame

        class API:
            def GetForegroundWindow(self):
                return 123

            def GetWindowTextW(self, hwnd, title, size):
                title.value = "SudokuUniverseAssistant"

            def GetWindowThreadProcessId(self, hwnd, pointer):
                ctypes.cast(pointer, ctypes.POINTER(wintypes.DWORD))[0] = os.getpid()

        backend = WindowsGame.__new__(WindowsGame)
        backend.api = API()
        self.assertIsNone(backend.foreground())


class WindowTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        configure_tk()
        self.folder = tempfile.TemporaryDirectory(dir=ROOT/"tests")
        self.assertEqual(Path(self.folder.name).resolve().parent, (ROOT/"tests").resolve())
        self.root = tk.Tk()
        self.root.withdraw()
        self.patches = []

        class FakeRunner:
            def __init__(self, backend, control, settings, emit, data_dir):
                self.control, self.emit = control, emit
                backend.started_settings = settings

            def run(self):
                self.control.stop.wait(5)
                self.emit("done", 0)

        for key in KEYS:
            module = importlib.import_module(f"sudoku_assistant.{key}.runner")
            patched = patch.object(module, "Runner", FakeRunner)
            patched.start()
            self.patches.append(patched)
        self.app = AssistantWindow(self.root, data_root=Path(self.folder.name)/"data")
        self.root.update()

    def tearDown(self):
        self.app.close()
        for patched in reversed(self.patches):
            patched.stop()
        self.folder.cleanup()

    def test_start_button_uses_selected_game_and_dropdown_cancels_previous(self):
        universe = self.app.pages["universe"]
        universe.start_button.invoke()
        self.assertTrue(universe.active())
        self.assertFalse(self.app.pages["killer"].active())
        old_control = universe.control
        self.app.choice.set(self.app.labels["killer"])
        self.assertTrue(old_control.stop.is_set())
        self.assertEqual(self.app.host.pack_slaves(), [self.app.frames["killer"]])
        self.app.pages["killer"].start_button.invoke()
        self.assertTrue(self.app.pages["killer"].active())
        self.assertFalse(self.app.pages["killer"].control.stop.is_set())

    def test_settings_and_endless_controls_survive_round_trip(self):
        universe = self.app.pages["universe"]
        universe.mode.set("无尽模式")
        universe.endless_limit.set("25")
        universe.change_mode()
        self.app.choice.set(self.app.labels["jigsaw"])
        self.app.pages["jigsaw"].limit.set("2")
        self.app.choice.set(self.app.labels["universe"])
        self.assertEqual(universe.mode.get(), "无尽模式")
        self.assertEqual(universe.endless_limit.get(), "25")
        self.assertTrue(universe.endless_difficulty_widget.instate(["readonly"]))
        self.assertEqual(self.app.pages["jigsaw"].limit.get(), "2")
        directories = [page.data_dir for page in self.app.pages.values()]
        self.assertEqual(len(set(directories)), 3)
        self.assertTrue(all(directory.parent == Path(self.folder.name)/"data" for directory in directories))

    def test_interfaces_keep_buttons_and_have_no_shortcut_prompts(self):
        for key in KEYS:
            self.app.choice.set(self.app.labels[key])
            texts = []
            pending = [self.app.frames[key]]
            while pending:
                widget = pending.pop()
                pending.extend(widget.winfo_children())
                if "text" in widget.keys():
                    texts.append(str(widget.cget("text")))
            self.assertIn("开始", texts)
            self.assertIn("暂停 / 继续", texts)
            self.assertIn("停止", texts)
            self.assertFalse(any(shortcut in text for shortcut in ("F8", "F9", "F10") for text in texts))
            self.assertFalse(hasattr(self.app.pages[key], "hotkeys"))

    def test_game_names_have_no_versions_and_start_uses_fixed_twenty_ms(self):
        for key in KEYS:
            with self.subTest(assistant=key):
                self.app.choice.set(self.app.labels[key])
                page = self.app.pages[key]
                texts = []
                pending = [self.app.frames[key]]
                while pending:
                    widget = pending.pop()
                    pending.extend(widget.winfo_children())
                    if "text" in widget.keys():
                        texts.append(str(widget.cget("text")))
                self.assertIn(self.app.labels[key], texts)
                self.assertFalse(any(value in text for value in ("1.2", "1.5", "极速版", "点击节拍") for text in texts))
                self.assertFalse(hasattr(page, "interval"))
                self.assertFalse(hasattr(page, "interval_widget"))
                page.start_button.invoke()
                self.assertEqual(page.backend.started_settings.interval, .020)
                self.assertEqual(page.backend.hold_time, .020)
                self.app.coordinator.stop_all()
                page.worker.join(timeout=1)
        self.assertEqual({values["interval_ms"] for values in self.app.describe()["assistants"].values()}, {20})


if __name__ == "__main__":
    unittest.main()
