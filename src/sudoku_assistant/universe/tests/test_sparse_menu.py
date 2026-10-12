"""Regressions for sparse unfinished tiles and menu-loading transitions."""
from pathlib import Path
import unittest
from PIL import ImageDraw,Image

ROOT=Path(__file__).resolve().parents[1]
from sudoku_assistant.universe.vision import inspect_level_menu,available_levels,MenuNotReady,LEVEL_GROUPS,DIFFICULTIES
from sudoku_assistant.universe.runner import Runner,Control,Settings
from sudoku_assistant.universe.tests.test_assistant import fixture

LAYOUTS={
    "简单":("easy",1,30,(423,576,730)),
    "中等":("medium",31,30,(423,576,730)),
    "困难":("hard",61,40,(346,499,653,807)),
    "Master Ninja":("master",101,50,(269,423,576,730,883)),
    "Pro Pack":("pro",151,25,(423,576,730)),
    "Pro Pack 2":("pro2",176,25,(423,576,730)),
    "Pro Pack 3":("pro3",201,25,(423,576,730)),
}


def completed_page(group):
    name,first,count,rows=LAYOUTS[group]
    frame=fixture(name);draw=ImageDraw.Draw(frame)
    for i in range(count):
        row,col=divmod(i,10);x,y=690+123*col,rows[row]
        draw.rounded_rectangle((x-46,y-46,x+46,y+46),radius=10,fill=(105,140,160))
    return frame


class QuickRunner(Runner):
    menu_timeout=0.5
    def wait(self,seconds):self.check()


class SparseMenuTests(unittest.TestCase):
    def test_all_six_actual_failure_screenshots_exact_level_numbers(self):
        expected={
            "left_easy":("简单",[10,20,30]),
            "left_medium":("中等",[40,*range(42,61)]),
            "left_hard":("困难",[70,80,*range(88,101)]),
            "left_master":("Master Ninja",list(range(104,151))),
            "left_pro":("Pro Pack",[160,170,175]),
            "left_pro2":("Pro Pack 2",[185,*range(194,201)]),
        }
        for name,(group,remaining) in expected.items():
            with self.subTest(name=name):
                result=inspect_level_menu(fixture(name))
                self.assertEqual(result.group,group)
                self.assertEqual(list(result.remaining),remaining)
                self.assertEqual(result.completed+len(result.remaining),result.total)

    def test_every_one_of_225_slots_is_found_when_it_is_the_only_white_tile(self):
        for group,(_,first,count,rows) in LAYOUTS.items():
            base=completed_page(group)
            for i in range(count):
                with self.subTest(group=group,level=first+i):
                    frame=base.copy();row,col=divmod(i,10);x,y=690+123*col,rows[row]
                    ImageDraw.Draw(frame).rounded_rectangle((x-46,y-46,x+46,y+46),radius=10,fill=(250,250,250))
                    result=inspect_level_menu(frame)
                    self.assertEqual(result.remaining,(first+i,))
                    self.assertEqual(result.completed,count-1)
                    self.assertEqual(available_levels(frame),[(float(x),float(y))])

    def test_all_seven_fully_completed_pages_require_actual_dark_tiles(self):
        for group in LEVEL_GROUPS:
            with self.subTest(group=group):
                catalog=inspect_level_menu(completed_page(group))
                self.assertEqual(catalog.remaining,())
                self.assertEqual(catalog.completed,catalog.total)

    def test_blank_and_fading_tiles_are_not_treated_as_completed(self):
        original=fixture("easy")
        blank=original.copy()
        ImageDraw.Draw(blank).rectangle((570,190,1950,1040),fill=(105,140,160))
        fading=Image.blend(blank,original,0.3)
        for frame in (blank,fading):
            with self.assertRaises(MenuNotReady):inspect_level_menu(frame)

    def test_waits_for_loaded_menu_instead_of_skipping_group(self):
        ready=fixture("left_easy");blank=ready.copy()
        ImageDraw.Draw(blank).rectangle((570,190,1950,1040),fill=(105,140,160))
        class Delayed:
            captures=0
            def capture(self):
                self.captures+=1
                return (blank if self.captures<=3 else ready).copy()
            def click(self,*args):raise AssertionError("No clicks during loading")
        backend=Delayed();runner=QuickRunner(backend,Control(),Settings(),lambda *args:None,ROOT/"test-data")
        _,catalog=runner.stable_menu("简单")
        self.assertEqual(catalog.remaining,(10,20,30))
        self.assertGreaterEqual(backend.captures,5)

    def test_wrong_group_never_passes_expected_group_check(self):
        class Wrong:
            def capture(self):return fixture("left_medium")
        runner=QuickRunner(Wrong(),Control(),Settings(),lambda *args:None,ROOT/"test-data")
        with self.assertRaisesRegex(ValueError,"未跳过"):
            runner.stable_menu("困难")


if __name__=="__main__":unittest.main()
