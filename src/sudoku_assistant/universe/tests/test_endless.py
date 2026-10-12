"""Endless-menu and exact-session-count tests. No real mouse input."""
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
from sudoku_assistant.universe.tests.test_assistant import SimulatedGame,fixture,EXPECTED,FastRunner
from sudoku_assistant.universe.vision import ENDLESS_CHOICES,DIFFICULTIES,endless_menu_visible,inspect_level_menu,MenuNotReady
from sudoku_assistant.universe.runner import Control,Settings,RestartReading


class EndlessGame(SimulatedGame):
    def __init__(self,finite_start=False):
        super().__init__()
        self.endless_menu=fixture("endless")
        self.finite_start=finite_start
        self.started=[]
        self.done=0
        self.tab_clicks=0

    def capture(self):
        if self.phase=="menu":
            return (self.menu if self.finite_start else self.endless_menu).copy()
        if self.phase=="animation":
            self.animation_frames+=1
            if self.animation_frames<=4:
                return self.complete.copy()
            self.phase="menu";self.done+=1
            return self.endless_menu.copy()
        return self.screen.copy()

    def click(self,point,check):
        if self.phase!="menu":
            return super().click(point,check)
        check();self.clicked.append(point)
        if point==DIFFICULTIES["无限"]:
            self.finite_start=False;self.tab_clicks+=1
            return
        self.assert_endless()
        name=next(name for name,p in ENDLESS_CHOICES.items() if p==point)
        self.started.append(name)
        self.board=EXPECTED.copy();self.screen=self.original.copy()
        for i,n in enumerate(self.board):
            if not n:self.paint_value(*divmod(i,9),0)
        self.phase="board";self.selected=None;self.animation_frames=0

    def assert_endless(self):
        if self.finite_start:raise AssertionError("Finite level must not be entered in endless mode")


class EndlessTests(unittest.TestCase):
    def test_actual_menu_has_three_ready_choices(self):
        frame=fixture("endless")
        self.assertTrue(endless_menu_visible(frame))
        self.assertEqual(ENDLESS_CHOICES,{"简单":(864,577),"中等":(1024,577),"困难":(1184,577)})
        with self.assertRaises(MenuNotReady):inspect_level_menu(frame)

    def test_finite_menus_are_not_endless_menus(self):
        for name in ("easy","medium","hard","master","pro","pro2","pro3"):
            self.assertFalse(endless_menu_visible(fixture(name)))

    def run_game(self,difficulty="简单",limit=2,finite_start=False,control=None,emit=None,runner_type=FastRunner):
        game=EndlessGame(finite_start=finite_start);control=control or Control();events=[]
        def forward(k,v):
            events.append((k,v))
            if emit:emit(k,v,runner,control)
        runner=runner_type(game,control,Settings(continuous=False,limit=1,mode="无尽模式",
            endless_difficulty=difficulty,endless_limit=limit),forward,ROOT/"test-data")
        runner.run()
        return game,runner,events

    def test_each_difficulty_repeats_and_stops_at_exact_count(self):
        for difficulty in ENDLESS_CHOICES:
            with self.subTest(difficulty=difficulty):
                game,runner,events=self.run_game(difficulty,2)
                self.assertEqual(game.started,[difficulty,difficulty])
                self.assertEqual(game.done,2)
                self.assertEqual(runner.completed,2)
                self.assertEqual([v for k,v in events if k=="error"],[])

    def test_can_enter_infinite_tab_from_finite_menu(self):
        game,runner,events=self.run_game("中等",1,finite_start=True)
        self.assertEqual(game.tab_clicks,1)
        self.assertEqual(game.started,["中等"])
        self.assertEqual(runner.completed,1)
        self.assertEqual([v for k,v in events if k=="error"],[])

    def test_zero_count_runs_until_stop(self):
        def emit(k,v,runner,control):
            if k=="log" and runner.completed>=3:control.stop.set()
        game,runner,events=self.run_game(limit=0,emit=emit)
        self.assertEqual(game.done,3)
        self.assertEqual(len(game.started),3)
        self.assertEqual(runner.completed,3)
        self.assertEqual([v for k,v in events if k=="error"],[])

    def test_resume_during_completion_counts_once_and_stops(self):
        class InterruptedCompletion(FastRunner):
            interrupted=False
            def wait_for_menu(self,count=False):
                if count and not self.interrupted:
                    self.interrupted=True
                    raise RestartReading()
                return super().wait_for_menu(count)
        game,runner,events=self.run_game(limit=1,runner_type=InterruptedCompletion)
        self.assertEqual(game.done,1)
        self.assertEqual(runner.completed,1)
        self.assertEqual(len(game.started),1)
        self.assertEqual([v for k,v in events if k=="error"],[])

    def test_stop_before_start_sends_no_input(self):
        control=Control();control.stop.set()
        game,runner,events=self.run_game(control=control)
        self.assertEqual(game.clicked,[])
        self.assertEqual(runner.completed,0)


if __name__=="__main__":unittest.main()
