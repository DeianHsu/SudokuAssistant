"""Meaningful recognition, solving and simulated input regressions. No live clicks."""
from collections import Counter
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
from sudoku_assistant.jigsaw.vision import (Recognizer,canonical,read_regions,board_visible,RecognitionError,
                    inspect_level_menu,MenuNotReady,cell_box,cell_center,digit_point,
                    LEVEL_LAYOUTS,chooser_open,glyph_mask)
from sudoku_assistant.jigsaw.solver import solve_unique,validate_regions,PuzzleError
from sudoku_assistant.jigsaw.runner import Runner,Control,Settings,RestartReading,Stopped
from sudoku_assistant.jigsaw.wincontrol import WindowsGame,Input,MouseInput,ForegroundLost


def fixture(name):
    with Image.open(ROOT/"tests"/"fixtures"/(name+".png")) as image:
        return canonical(image)


def completed_page(group,name):
    frame=fixture(name)
    first,count,rows=LEVEL_LAYOUTS[group]
    draw=ImageDraw.Draw(frame)
    for i in range(count):
        row,col=divmod(i,10)
        x,y=690+123*col,rows[row]
        draw.rounded_rectangle((x-46,y-46,x+46,y+46),radius=10,fill=(29,37,38))
    return frame


def faint_completed_tile(frame,source,level,group,fill=(53,53,53)):
    first,_,rows=LEVEL_LAYOUTS[group]
    row,col=divmod(level-first,10)
    x,y=690+123*col,rows[row]
    mask=glyph_mask(source.crop((x-44,y-31,x+44,y+31)))
    ImageDraw.Draw(frame).rounded_rectangle((x-46,y-46,x+46,y+46),radius=10,fill=fill)
    a=np.asarray(frame).copy()
    patch=a[y-31:y+31,x-44:x+44]
    patch[mask]=(220,220,220)
    return Image.fromarray(a)


class ScreenshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recognizer=Recognizer()
        cls.expected=json.loads((ROOT/"tests"/"fixtures"/"boards.json").read_text(encoding="utf-8"))

    def test_all_four_actual_puzzles_exact_digits_and_valid_solutions(self):
        for name,rows in self.expected.items():
            with self.subTest(name=name):
                reading=self.recognizer.read(fixture(name))
                expected=[int(n) for row in rows for n in row]
                self.assertEqual(reading.board,expected)
                self.assertEqual(Counter(reading.regions),Counter({n:9 for n in range(9)}))
                solution=solve_unique(expected,reading.regions)
                for r in range(9):
                    self.assertEqual(set(solution[r*9:r*9+9]),set(range(1,10)))
                    self.assertEqual(set(solution[r::9]),set(range(1,10)))
                    self.assertEqual({solution[i] for i in range(81) if reading.regions[i]==r},set(range(1,10)))
                self.assertTrue(all(not n or solution[i]==n for i,n in enumerate(expected)))
                # Actual supplied jigsaws must not be accidentally solved using 3x3 boxes.
                regular=[(r//3)*3+c//3 for r in range(9) for c in range(9)]
                self.assertNotEqual(reading.regions,regular)
                self.assertEqual(reading.chooser,{"pink-chooser":39,"purple-chooser":1}.get(name))

    def test_all_channels_rotated_preserve_all_palace_boundaries(self):
        for name in self.expected:
            frame=fixture(name)
            regions=read_regions(frame)
            for order in ((1,2,0),(2,0,1),(2,1,0)):
                with self.subTest(name=name,channels=order):
                    altered=Image.fromarray(np.asarray(frame)[:,:,order])
                    self.assertEqual(read_regions(altered),regions)

    def test_white_grid_with_erased_palace_lines_is_rejected(self):
        frame=fixture("black")
        draw=ImageDraw.Draw(frame)
        draw.rectangle((640,192,1408,960),fill=(250,250,250))
        self.assertTrue(board_visible(frame))
        with self.assertRaises(RecognitionError):
            read_regions(frame)

    def test_actual_chooser_all_nine_clicks_and_uniform_grid(self):
        frame=fixture("pink-chooser")
        self.assertTrue(chooser_open(frame,4,3))
        self.assertFalse(chooser_open(frame,0,0))
        for n in range(1,10):
            x,y=digit_point(4,3,n)
            x0,y0,x1,y1=cell_box(4,3)
            self.assertTrue(x0<x<x1 and y0<y<y1)
        for a,b in zip(digit_point(4,3,5),cell_center(4,3)):
            self.assertAlmostEqual(a,b)

    def test_1280_1920_and_2560_frames_read_consistently(self):
        frame=fixture("blue")
        original=self.recognizer.read(frame)
        for size in ((1280,720),(1920,1080),(2560,1440)):
            with self.subTest(size=size):
                reading=self.recognizer.read(frame.resize(size,Image.Resampling.LANCZOS))
                self.assertEqual(reading.board,original.board)
                self.assertEqual(reading.regions,original.regions)

    def test_non_widescreen_is_rejected(self):
        with self.assertRaises(RecognitionError):
            canonical(Image.new("RGB",(1200,900)))

    def test_empty_hover_circle_is_empty_and_white_hover_digit_is_preserved(self):
        frame=fixture("purple-circle")
        self.assertEqual(self.recognizer.read_cell(frame,2,3)[0],0)
        # Use the real game glyph and real badge radius for white-on-circle values.
        source=fixture("black")
        x0,y0,x1,y1=cell_box(1,2)
        digit_mask=glyph_mask(source.crop((round(x0+14),round(y0+10),round(x1-14),round(y1-10))))
        for color in ((140,60,165),(45,125,110),(105,105,105)):
            with self.subTest(color=color):
                h,w=digit_mask.shape
                image=Image.new("RGB",(w,h),(245,235,245))
                ImageDraw.Draw(image).ellipse((1,(h-54)/2,w-2,(h+54)/2),fill=color)
                a=np.asarray(image).copy()
                a[digit_mask]=(250,250,250)
                cleaned=glyph_mask(Image.fromarray(a))
                self.assertTrue(np.array_equal(cleaned,digit_mask))
                self.assertGreater(float(cleaned.sum()),0)


class MenuTests(unittest.TestCase):
    def test_actual_completed_first_tile_yields_level_two_next(self):
        catalog=inspect_level_menu(fixture("easy-after-one"))
        self.assertEqual(catalog.remaining,tuple(range(2,31)))
        self.assertEqual(catalog.completed,1)
        self.assertEqual(catalog.centers[0],(813.,423.))

    def test_all_100_faint_completed_labels_are_read_including_two_and_three_digits(self):
        for name,group in (("easy","简单"),("medium","中等"),("hard","困难")):
            source=fixture(name)
            frame=source.copy()
            ImageDraw.Draw(frame).rectangle((570,200,1900,950),fill=(85,85,85))
            first,count,_=LEVEL_LAYOUTS[group]
            for level in range(first,first+count):
                frame=faint_completed_tile(frame,source,level,group,fill=(81,81,81))
            catalog=inspect_level_menu(frame)
            self.assertEqual(catalog.remaining,())
            self.assertEqual(catalog.completed,count)

    def test_faint_tile_without_label_or_with_wrong_label_is_not_completed(self):
        frame=fixture("easy-after-one")
        ImageDraw.Draw(frame).rectangle((646,392,734,454),fill=(53,53,53))
        with self.assertRaises(MenuNotReady):
            inspect_level_menu(frame)
        frame=fixture("easy-after-one")
        label=frame.crop((646,392,734,454))
        ImageDraw.Draw(frame).rounded_rectangle((813-46,423-46,813+46,423+46),radius=10,fill=(53,53,53))
        frame.paste(label,(813-44,423-31))
        with self.assertRaises(MenuNotReady):
            inspect_level_menu(frame)

    def test_exact_numbers_in_three_supplied_menus(self):
        for name,group in (("easy","简单"),("medium","中等"),("hard","困难")):
            catalog=inspect_level_menu(fixture(name))
            first,count,_=LEVEL_LAYOUTS[group]
            self.assertEqual(catalog.group,group)
            self.assertEqual(catalog.remaining,tuple(range(first,first+count)))

    def test_every_one_of_100_slots_as_the_only_remaining_white_button(self):
        for group,name in (("简单","easy"),("中等","medium"),("困难","hard")):
            base=completed_page(group,name)
            first,count,rows=LEVEL_LAYOUTS[group]
            self.assertEqual(inspect_level_menu(base).completed,count)
            for i in range(count):
                with self.subTest(level=first+i):
                    frame=base.copy()
                    row,col=divmod(i,10)
                    x,y=690+123*col,rows[row]
                    ImageDraw.Draw(frame).rounded_rectangle((x-46,y-46,x+46,y+46),radius=10,fill=(250,250,250))
                    catalog=inspect_level_menu(frame)
                    self.assertEqual(catalog.remaining,(first+i,))
                    self.assertEqual(catalog.completed,count-1)

    def test_missing_buttons_and_wrong_group_are_never_completed(self):
        frame=fixture("easy")
        with self.assertRaises(MenuNotReady):
            inspect_level_menu(frame,"困难")
        ImageDraw.Draw(frame).rectangle((570,190,1950,1040),fill=(65,65,65))
        with self.assertRaises(MenuNotReady):
            inspect_level_menu(frame)


class SolverTests(unittest.TestCase):
    def test_actual_level_100_with_fifteen_clues_is_unique_under_5000_nodes(self):
        reading=Recognizer().read(fixture("pastel-sparse"))
        self.assertEqual(sum(n>0 for n in reading.board),15)
        stats={}
        solution=solve_unique(reading.board,reading.regions,node_limit=5000,stats=stats)
        self.assertEqual(stats["solutions"],1)
        self.assertLess(stats["nodes"],5000)
        self.assertEqual(solve_unique(solution,reading.regions),solution)

    def test_nonconflicting_wrong_clue_is_unsatisfiable(self):
        reading=Recognizer().read(fixture("black"))
        board=list(reading.board)
        board[0]=2
        with self.assertRaisesRegex(PuzzleError,"无解"):
            solve_unique(board,reading.regions)

    def test_search_limit_still_stops_excessive_work(self):
        reading=Recognizer().read(fixture("pastel-sparse"))
        with self.assertRaisesRegex(PuzzleError,"上限"):
            solve_unique(reading.board,reading.regions,node_limit=1)

    def test_multiple_solutions_and_conflicting_digits_are_rejected(self):
        reading=Recognizer().read(fixture("teal"))
        with self.assertRaisesRegex(PuzzleError,"多个解"):
            solve_unique([0]*81,reading.regions)
        board=list(reading.board)
        board[0]=board[1]
        with self.assertRaisesRegex(PuzzleError,"冲突"):
            solve_unique(board,reading.regions)

    def test_nonconnected_and_wrong_sized_regions_are_rejected(self):
        for regions in ([0]*81,[i%9 for i in range(81)]):
            if regions[0]==regions[1]:
                with self.assertRaises(PuzzleError):
                    validate_regions(regions)
        reading=Recognizer().read(fixture("teal"))
        regions=list(reading.regions)
        regions[0],regions[80]=regions[80],regions[0]
        with self.assertRaises(PuzzleError):
            validate_regions(regions)

    def test_stop_callback_can_interrupt_solving(self):
        reading=Recognizer().read(fixture("black"))
        def stop():
            raise Stopped()
        with self.assertRaises(Stopped):
            solve_unique(reading.board,reading.regions,check=stop)


class FakeBackend:
    def __init__(self,frame):
        self.frame=frame
        self.clicks=[]
    def set_timing(self,value):
        self.timing=value
    def capture(self):
        return self.frame.copy()
    def click(self,point,check):
        check()
        self.clicks.append(point)
    def move(self,point,check):
        check()


def show_chooser(frame,r,c):
    x0,y0,x1,y1=cell_box(4,3)
    menu=fixture("pink-chooser").crop((round(x0),round(y0),round(x1),round(y1)))
    target_x,target_y,_,_=cell_box(r,c)
    frame.paste(menu,(round(target_x),round(target_y)))


class MenuBackend(FakeBackend):
    def __init__(self,frame):
        super().__init__(frame)
        self.opened=True
    def click(self,point,check):
        super().click(point,check)
        if self.opened:
            self.opened=False
        else:
            from sudoku_assistant.jigsaw.vision import STEP
            x,y=point
            r,c=int((y-192)/STEP),int((x-640)/STEP)
            show_chooser(self.frame,r,c)
            self.opened=True


class QuickRunner(Runner):
    menu_timeout=.2
    def wait(self,seconds):
        self.check()
    def wait_for_menu(self,count=False):
        self.waited_for_completion=count
        return fixture("easy")


class RunnerTests(unittest.TestCase):
    def make_runner(self,name):
        backend=FakeBackend(fixture(name))
        return QuickRunner(backend,Control(),Settings(),lambda *args:None,ROOT/"test-data"),backend

    def test_open_chooser_at_any_cell_is_filled_first_without_reopening(self):
        backend=MenuBackend(fixture("pink-chooser"))
        runner=QuickRunner(backend,Control(),Settings(),lambda *args:None,ROOT/"test-data")
        reading=runner.recognizer.read(backend.frame)
        solution=solve_unique(reading.board,reading.regions)
        runner.fill_board()
        empty=sum(n==0 for n in reading.board)
        self.assertEqual(backend.clicks[0],digit_point(4,3,solution[39]))
        self.assertEqual(len(backend.clicks),empty*2-1)
        self.assertTrue(runner.pending_completion)
        self.assertTrue(runner.waited_for_completion)

    def test_fill_completes_without_capturing_or_probing_any_menu(self):
        runner,backend=self.make_runner("black")
        capture=backend.capture
        captures=[]
        def initial_capture_only():
            if backend.clicks:
                raise AssertionError("求解后填写途中不得截图。")
            captures.append(True)
            return capture()
        with patch.object(backend,"capture",side_effect=initial_capture_only):
            runner.fill_board()
        self.assertEqual(len(captures),2)
        self.assertEqual(len(backend.clicks),57*2)
        self.assertEqual(backend.clicks[:2],[cell_center(0,0),digit_point(0,0,3)])
        self.assertTrue(runner.pending_completion)
        self.assertTrue(runner.waited_for_completion)

    def test_unmarked_level_is_not_counted_as_completed(self):
        runner,_=self.make_runner("black")
        runner.pending_level=("简单",1)
        with self.assertRaises(RecognitionError):
            runner.confirm_completion(inspect_level_menu(fixture("easy")))
        self.assertEqual(runner.completed,0)
        runner.confirm_completion(inspect_level_menu(completed_page("简单","easy")))
        self.assertEqual(runner.completed,1)

    def test_final_input_remains_pending_if_pause_occurs_after_release(self):
        runner,_=self.make_runner("black")
        with patch.object(runner,"wait",side_effect=RestartReading()):
            with self.assertRaises(RestartReading):
                runner.click((1000,500),final_input=True)
        self.assertTrue(runner.pending_completion)

    def test_stop_prevents_any_input(self):
        runner,backend=self.make_runner("black")
        runner.control.stop.set()
        with self.assertRaises(Stopped):
            runner.click((1000,500))
        self.assertEqual(backend.clicks,[])


class DirectFillTests(unittest.TestCase):
    def test_purple_failure_screen_continues_without_reopening_or_animation_capture(self):
        class NoMidFillCapture(FakeBackend):
            captures=0
            def capture(backend):
                if backend.clicks:
                    raise AssertionError("菜单动画不能触发中途截图。")
                backend.captures+=1
                return super().capture()
            def click(backend,point,check):
                super().click(point,check)
                # During filling the game may still show an opening animation.
                backend.frame=opening
        opening=fixture("purple-probe")
        self.assertFalse(chooser_open(opening,0,1))
        backend=NoMidFillCapture(fixture("purple-chooser"))
        runner=QuickRunner(backend,Control(),Settings(),lambda *args:None,ROOT/"test-data")
        waits=[]
        with patch.object(runner,"wait",side_effect=lambda seconds:(runner.check(),waits.append(seconds))):
            runner.fill_board()
        self.assertEqual(backend.captures,2)
        self.assertEqual(backend.clicks[:2],[digit_point(0,1,7),cell_center(0,2)])
        self.assertEqual(len(backend.clicks),53*2-1)
        self.assertEqual(waits,[.2,.2]+[.020]*(53*2-1))
        self.assertTrue(runner.pending_completion)
        self.assertTrue(runner.waited_for_completion)

    def test_pointer_moves_off_grid_before_any_reading_capture(self):
        class MoveFirstBackend(FakeBackend):
            moved=False
            def move(backend,point,check):
                check()
                self.assertEqual(point,(1740,1000))
                backend.moved=True
            def capture(backend):
                self.assertTrue(backend.moved)
                return super().capture()
        backend=MoveFirstBackend(fixture("purple-circle"))
        runner=QuickRunner(backend,Control(),Settings(),lambda *args:None,ROOT/"test-data")
        reading,solution=runner.stable_reading()
        self.assertEqual(sum(n>0 for n in reading.board),30)
        self.assertEqual(len(solution),81)


class FakeTransport(WindowsGame):
    def __init__(self):
        self.hold_time=.02
        self.events=[]
        self.fail_guard=False
    def guard(self):
        if self.fail_guard:
            raise ForegroundLost("not foreground")
    def movement(self,point):
        return Input(type=0,mi=MouseInput(dx=100,dy=200,dwFlags=0xC001))
    def send(self,*events):
        self.events.extend(int(e.mi.dwFlags) for e in events)


class InputTests(unittest.TestCase):
    def test_press_and_release_and_interrupted_hold(self):
        backend=FakeTransport()
        with patch("sudoku_assistant.jigsaw.wincontrol.time.sleep"):
            backend.click((1000,500),lambda:None)
        self.assertEqual(backend.events,[0xC003,0x0004])
        with patch("sudoku_assistant.jigsaw.wincontrol.time.sleep",side_effect=RuntimeError("interrupted")):
            with self.assertRaises(RuntimeError):
                backend.click((1000,500),lambda:None)
        self.assertEqual(backend.events[-1],0x0004)

    def test_foreground_loss_blocks_input(self):
        backend=FakeTransport()
        backend.fail_guard=True
        with self.assertRaises(ForegroundLost):
            backend.click((1000,500),lambda:None)
        self.assertEqual(backend.events,[])


class SimulatedGame(FakeBackend):
    """Stateful game transport for full menu -> puzzle -> completion workflows."""
    names={"简单":"easy","中等":"medium","困难":"hard"}

    def __init__(self):
        super().__init__(fixture("easy"))
        self.group="简单"
        self.finished=set()
        self.mode="menu"
        self.entered=[]
        self.inputs=0
        self.chooser=False
        self.rebuild_menu()

    def rebuild_menu(self):
        self.frame=completed_page(self.group,self.names[self.group])
        if self.group not in self.finished:
            _,count,rows=LEVEL_LAYOUTS[self.group]
            row,col=divmod(count-1,10)
            x,y=690+123*col,rows[row]
            ImageDraw.Draw(self.frame).rounded_rectangle((x-46,y-46,x+46,y+46),radius=10,fill=(250,250,250))

    def click(self,point,check):
        from sudoku_assistant.jigsaw.vision import DIFFICULTIES
        super().click(point,check)
        if self.mode=="menu":
            for group,center in DIFFICULTIES.items():
                if point==center:
                    self.group=group
                    self.rebuild_menu()
                    return
            catalog=inspect_level_menu(self.frame)
            assert point==catalog.centers[0]
            self.entered.append(catalog.remaining[0])
            self.mode="board"
            self.frame=fixture("black")
            self.inputs=0
            self.chooser=False
            return
        if not self.chooser:
            self.chooser=True
            from sudoku_assistant.jigsaw.vision import STEP
            x,y=point
            show_chooser(self.frame,int((y-192)/STEP),int((x-640)/STEP))
        else:
            self.inputs+=1
            self.chooser=False
            if self.inputs==57:
                self.finished.add(self.group)
                self.mode="menu"
                self.rebuild_menu()


class InstantRunner(Runner):
    def wait(self,seconds):
        self.check()


class WorkflowTests(unittest.TestCase):
    def test_actual_after_level_one_menu_automatically_enters_two_then_three(self):
        class AfterOneGame(SimulatedGame):
            def rebuild_menu(backend):
                backend.frame=fixture("easy-after-one")
                for level in backend.entered:
                    backend.frame=faint_completed_tile(backend.frame,fixture("easy"),level,"简单")
        backend=AfterOneGame()
        events=[]
        runner=InstantRunner(backend,Control(),Settings(limit=2),lambda *args:events.append(args),ROOT/"test-data")
        runner.run()
        self.assertFalse([event for event in events if event[0]=="error"],events)
        self.assertEqual(backend.entered,[2,3])
        self.assertEqual(runner.completed,2)

    def run_simulation(self,limit):
        backend=SimulatedGame()
        events=[]
        runner=InstantRunner(backend,Control(),Settings(difficulty="依次完成三个难度",limit=limit),
                             lambda *args:events.append(args),ROOT/"test-data")
        runner.run()
        self.assertFalse([event for event in events if event[0]=="error"],events)
        return runner,backend

    def test_all_three_groups_advance_only_after_completion(self):
        runner,backend=self.run_simulation(0)
        self.assertEqual(runner.completed,3)
        self.assertEqual(backend.entered,[30,60,100])
        self.assertEqual(backend.finished,{"简单","中等","困难"})

    def test_limit_one_stops_before_entering_second_group(self):
        runner,backend=self.run_simulation(1)
        self.assertEqual(runner.completed,1)
        self.assertEqual(backend.entered,[30])


if __name__=="__main__":
    unittest.main()
