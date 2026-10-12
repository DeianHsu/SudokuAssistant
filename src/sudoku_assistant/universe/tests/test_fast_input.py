"""Test accelerated input with a fake transport; never sends system input."""
import unittest
from unittest.mock import patch

from sudoku_assistant.universe.wincontrol import WindowsGame, Input, MouseInput, ForegroundLost


class FakeTransport(WindowsGame):
    def __init__(self):
        self.hold_time=0.017
        self.events=[]
        self.fail_guard=False

    def guard(self):
        if self.fail_guard:
            raise ForegroundLost("not foreground")

    def movement(self, point):
        return Input(type=0,mi=MouseInput(dx=100,dy=200,dwFlags=0xC001))

    def send(self,*events):
        self.events.extend(int(e.mi.dwFlags) for e in events)


class FastInputTests(unittest.TestCase):
    def test_move_and_press_are_combined_and_extra_settle_is_removed(self):
        backend=FakeTransport()
        with patch("sudoku_assistant.universe.wincontrol.time.sleep") as sleep:
            backend.click((1000,500),lambda:None)
        self.assertEqual(backend.events,[0xC003,0x0004])
        sleep.assert_called_once_with(0.017)

    def test_custom_cadence_controls_press_duration(self):
        backend=FakeTransport()
        backend.set_timing(0.001)
        with patch("sudoku_assistant.universe.wincontrol.time.sleep") as sleep:
            backend.click((1000,500),lambda:None)
        sleep.assert_called_once_with(0.001)

    def test_mouse_is_released_even_if_wait_is_interrupted(self):
        backend=FakeTransport()
        with patch("sudoku_assistant.universe.wincontrol.time.sleep",side_effect=RuntimeError("interrupted")):
            with self.assertRaises(RuntimeError):
                backend.click((1000,500),lambda:None)
        self.assertEqual(backend.events[-1],0x0004)

    def test_foreground_guard_still_prevents_input(self):
        backend=FakeTransport();backend.fail_guard=True
        with self.assertRaises(ForegroundLost):
            backend.click((1000,500),lambda:None)
        self.assertEqual(backend.events,[])

    def test_stop_check_prevents_input(self):
        backend=FakeTransport()
        def stop():raise RuntimeError("stopped")
        with self.assertRaises(RuntimeError):backend.click((1000,500),stop)
        self.assertEqual(backend.events,[])


if __name__=="__main__":unittest.main()
