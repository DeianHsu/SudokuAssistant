import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT=Path(__file__).resolve().parents[1]
from sudoku_assistant.killer.prepare_assets import LABELS
from sudoku_assistant.killer.solver import (Cage, PuzzleError, HOUSES, solve_unique, validate_cages,
                    validate_solution, cage_support, sum_support)
from sudoku_assistant.killer.vision import (Recognizer, Reading, RecognitionError, MenuNotReady, canonical,
                    inspect_level_menu, LEVEL_LAYOUTS, STEP, cell_box, cell_center,
                    digit_point, chooser_open, components, empty_hover_marker, badge_geometry)
from sudoku_assistant.killer.runner import Runner, Control, Settings, RestartReading, Stopped

FIXTURES=ROOT/"tests"/"fixtures"
EXPECTED=json.loads((FIXTURES/"boards.json").read_text(encoding="utf-8"))
EXPECTED["hard-62"]=json.loads((FIXTURES/"hard-62.json").read_text(encoding="utf-8"))
THEME_COLORS=((255,160,94),(230,50,65),(215,88,171),(60,75,215),
              (35,150,175),(50,145,70),(205,180,25),(70,70,70))


def cages_for(name):
    return tuple(Cage(c["total"],tuple(c["cells"])) for c in EXPECTED[name]["cages"])


def menu_frame(name, remaining=None):
    frame=canonical(Image.open(FIXTURES/(name+".png")))
    if remaining is not None:
        group={"easy":"简单","medium":"中等","hard":"困难"}[name]
        first,count,rows=LEVEL_LAYOUTS[group]
        draw=ImageDraw.Draw(frame)
        for i in range(count):
            if first+i not in remaining:
                r,c=divmod(i,10)
                x,y=690+123*c,rows[r]
                draw.rounded_rectangle((x-45,y-45,x+45,y+45),radius=12,fill=(10,15,20))
    return frame


class ScreenshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recognizer=Recognizer()

    def test_both_boards_at_reference_and_scaled_resolutions(self):
        for name in ("board1","board2"):
            image=Image.open(FIXTURES/(name+".png"))
            for size in ((2560,1440),(2048,1152),(1920,1080)):
                with self.subTest(name=name,size=size):
                    reading=self.recognizer.read(image.resize(size,Image.Resampling.LANCZOS))
                    self.assertEqual(reading.board,[0]*81)
                    self.assertEqual(reading.chooser,EXPECTED[name]["chooser"])
                    self.assertEqual(reading.cages,cages_for(name))
                    self.assertEqual(sum(c.total for c in reading.cages),405)
                    # Independent annotations of every visible orange number.
                    actual={(r+1,c+1):n for r in range(9) for c in range(9)
                            if (n:=self.recognizer.read_total(reading.frame,r,c)) is not None}
                    self.assertEqual(actual,{(r,c):n for r,c,n in LABELS[name]})

    def test_three_level_menus_and_sparse_remaining_buttons(self):
        for name,group,first,count in (("easy","简单",1,30),("medium","中等",31,30),("hard","困难",61,40)):
            with self.subTest(group=group):
                catalog=inspect_level_menu(menu_frame(name),group)
                self.assertEqual(catalog.remaining,tuple(range(first,first+count)))
                selected=set(range(first+9,first+count,10))
                catalog=inspect_level_menu(menu_frame(name,selected),group)
                self.assertEqual(catalog.remaining,tuple(sorted(selected)))
                self.assertEqual(catalog.completed,count-len(selected))
                catalog=inspect_level_menu(menu_frame(name,set()),group)
                self.assertEqual(catalog.remaining,())
                self.assertEqual(catalog.completed,count)
        with self.assertRaises(MenuNotReady):
            inspect_level_menu(menu_frame("easy"),"困难")

    def test_pastel_easy13_badges_and_unique_solution_at_scaled_resolutions(self):
        expected_totals={(r,c):n for r,c,n in LABELS["easy-13"]}
        with Image.open(FIXTURES/"easy-13.png") as image:
            # 2304x1296 is not used to generate recognition templates.
            for size in ((2560,1440),(2048,1152),(1920,1080),(2304,1296)):
                with self.subTest(size=size):
                    reading=self.recognizer.read(image.resize(size,Image.Resampling.LANCZOS))
                    actual={(r+1,c+1):n for r in range(9) for c in range(9)
                            if (n:=self.recognizer.read_total(reading.frame,r,c)) is not None}
                    self.assertEqual(actual,expected_totals)
                    self.assertEqual(actual[1,4],13)
                    self.assertEqual(reading.board,[0]*81)
                    self.assertIsNone(reading.chooser)
                    self.assertEqual(len(reading.cages),33)
                    self.assertEqual(sum(c.total for c in reading.cages),405)
                    solution=solve_unique(reading.board,reading.cages)
                    validate_solution(solution,reading.cages)

    def test_actual_completed_first_tile_selects_second_level(self):
        image=Image.open(FIXTURES/"easy-after-one.png")
        for size in ((2560,1440),(2048,1152),(1920,1080)):
            with self.subTest(size=size):
                catalog=inspect_level_menu(canonical(image.resize(size,Image.Resampling.LANCZOS)))
                self.assertEqual(catalog.remaining,tuple(range(2,31)))
                self.assertEqual(catalog.completed,1)
                self.assertEqual(catalog.centers[0],(813.,423.))

    def test_all_100_faint_completed_level_numbers_are_recognized(self):
        for name,group in (("easy","简单"),("medium","中等"),("hard","困难")):
            source=menu_frame(name)
            frame=source.copy()
            ImageDraw.Draw(frame).rectangle((570,200,1900,950),fill=(85,85,85))
            a=np.asarray(frame).copy()
            first,count,rows=LEVEL_LAYOUTS[group]
            for i in range(count):
                row,col=divmod(i,10)
                x,y=690+123*col,rows[row]
                label=np.asarray(source.crop((x-44,y-31,x+44,y+31))).astype(np.int16)
                mask=(label.mean(axis=2)<180)&(np.ptp(label,axis=2)<45)
                patch=a[y-31:y+31,x-44:x+44]
                patch[mask]=(220,220,220)
            catalog=inspect_level_menu(Image.fromarray(a))
            self.assertEqual(catalog.remaining,())
            self.assertEqual(catalog.completed,count)

    def test_missing_or_wrong_completed_label_is_not_accepted(self):
        source=canonical(Image.open(FIXTURES/"easy-after-one.png"))
        frame=source.copy()
        ImageDraw.Draw(frame).rectangle((627,376,754,470),fill=(53,53,53))
        with self.assertRaises(MenuNotReady):
            inspect_level_menu(frame)
        frame=source.copy()
        label=frame.crop((646,392,734,454))
        ImageDraw.Draw(frame).rectangle((750,376,877,470),fill=(53,53,53))
        frame.paste(label,(769,392))
        with self.assertRaises(MenuNotReady):
            inspect_level_menu(frame)

    def test_missing_total_and_wrong_aspect_are_rejected(self):
        image=canonical(Image.open(FIXTURES/"board1.png"))
        ImageDraw.Draw(image).rectangle((634,184,668,220),fill=(250,250,250))
        with self.assertRaises(RecognitionError):
            self.recognizer.read(image)
        with self.assertRaises(RecognitionError):
            canonical(Image.new("RGB",(800,600)))

    def test_actual_popups_have_nine_positions(self):
        for name in ("board1","board2"):
            frame=canonical(Image.open(FIXTURES/(name+".png")))
            r,c=divmod(EXPECTED[name]["chooser"],9)
            self.assertTrue(chooser_open(frame,r,c))
            self.assertEqual(self.recognizer.read_cell(frame,r,c),(0,1.0,True))
            for n in range(1,10):
                x,y=digit_point(r,c,n)
                left,top,right,bottom=cell_box(r,c)
                self.assertTrue(left<x<right and top<y<bottom)

    def test_open_menu_hiding_cage_total_can_be_read_and_solved(self):
        image=Image.open(FIXTURES/"board3-open-menu.png")
        for size in ((2560,1440),(2048,1152),(1920,1080)):
            with self.subTest(size=size):
                reading=self.recognizer.read(image.resize(size,Image.Resampling.LANCZOS))
                self.assertEqual(reading.chooser,0)
                self.assertEqual(reading.board,[0]*81)
                self.assertEqual(reading.cages,cages_for("board3"))
                self.assertEqual(solve_unique(reading.board,reading.cages),EXPECTED["board3"]["solution"])

    def test_open_menu_does_not_allow_a_second_missing_total(self):
        frame=canonical(Image.open(FIXTURES/"board3-open-menu.png"))
        cx,cy,radius=badge_geometry(0,2)
        ImageDraw.Draw(frame).ellipse((cx-radius-2,cy-radius-2,cx+radius+2,cy+radius+2),fill=(250,250,250))
        with self.assertRaises(RecognitionError):
            self.recognizer.read(frame)

    def test_actual_hard62_all_totals_and_unique_solution_at_three_sizes(self):
        image=Image.open(FIXTURES/"hard-62.png")
        expected={(r-1,c-1):total for r,c,total in EXPECTED["hard-62"]["labels"]}
        for size in ((2560,1440),(2048,1152),(1920,1080)):
            with self.subTest(size=size):
                reading=self.recognizer.read(image.resize(size,Image.Resampling.LANCZOS))
                totals={(r,c):total for r in range(9) for c in range(9)
                        if (total:=self.recognizer.read_total(reading.frame,r,c)) is not None}
                self.assertEqual(totals,expected)
                self.assertEqual(reading.board,[0]*81)
                self.assertIsNone(reading.chooser)
                self.assertEqual(reading.cages,cages_for("hard-62"))
                self.assertEqual(sum(c.total for c in reading.cages),405)
                self.assertEqual(solve_unique(reading.board,reading.cages),EXPECTED["hard-62"]["solution"])

    def test_hard62_blurred_total_is_rejected_instead_of_guessed(self):
        frame=canonical(Image.open(FIXTURES/"hard-62.png"))
        self.assertEqual(self.recognizer.read_total(frame,2,6),13)
        with self.assertRaises(RecognitionError):
            self.recognizer.read_total(frame.filter(ImageFilter.GaussianBlur(2)),2,6)

    def test_real_hover_dot_is_an_empty_cell_at_all_tested_sizes(self):
        for name in ("board3-hover-dot","board3-clean","board3-red","board3-pink-cyan"):
            image=Image.open(FIXTURES/(name+".png"))
            for size in ((2560,1440),(2048,1152),(1920,1080)):
                with self.subTest(name=name,size=size):
                    reading=self.recognizer.read(image.resize(size,Image.Resampling.LANCZOS))
                    self.assertEqual(reading.board,[0]*81)
                    self.assertIsNone(reading.chooser)
                    self.assertEqual(reading.cages,cages_for("board3"))
                    self.assertEqual(len(reading.cages),35)

    def test_hover_filter_keeps_all_existing_digit_templates(self):
        for filename in ("digits.npz","player-digits.npz","given-digits.npz"):
            with np.load(ROOT/"assets"/filename,allow_pickle=False) as data:
                for n in data.files:
                    samples=data[n]
                    if samples.ndim==2:
                        samples=samples[None,:,:]
                    for sample in samples:
                        mask=sample>.45
                        colored=np.full((*mask.shape,3),250,dtype=np.uint8)
                        colored[mask]=(255,160,94)
                        crop=Image.fromarray(colored)
                        big=[b for b in components(mask) if b[3]-b[1]>=22]
                        self.assertTrue(big)
                        for blob in big:
                            self.assertFalse(empty_hover_marker(crop,blob),(filename,n,blob))

    def test_empty_hover_dots_across_sizes_colors_and_fade_states(self):
        base=canonical(Image.open(FIXTURES/"board3-clean.png"))
        cx,cy=cell_center(2,0)
        for color in THEME_COLORS:
            for radius in (8,10,12,16,20,24,27):
                for opacity in (.4,1.):
                    with self.subTest(color=color,radius=radius,opacity=opacity):
                        frame=base.copy()
                        tint=tuple(round(opacity*q+(1-opacity)*250) for q in color)
                        ImageDraw.Draw(frame).ellipse((cx-radius,cy-radius,cx+radius,cy+radius),fill=tint)
                        self.assertEqual(self.recognizer.read_cell(frame,2,0)[0],0)

    def test_real_digit_shapes_survive_every_theme_color(self):
        base=canonical(Image.open(FIXTURES/"board3-clean.png"))
        cx,cy=cell_center(2,0)
        x,y=round(cx-16),round(cy-20)
        with np.load(ROOT/"assets"/"digits.npz",allow_pickle=False) as templates:
            for color in THEME_COLORS:
                for n in range(1,10):
                    with self.subTest(color=color,n=n):
                        a=np.asarray(base).copy()
                        sample=templates[str(n)][0]
                        pixels=np.round(250+(np.asarray(color)-250)*sample[:,:,None]).astype(np.uint8)
                        a[y:y+40,x:x+32]=pixels
                        self.assertEqual(self.recognizer.read_cell(Image.fromarray(a),2,0)[0],n)

    def test_badge_glyphs_are_independent_of_the_theme_hue(self):
        base=canonical(Image.open(FIXTURES/"board3-clean.png"))
        labels=((0,0,18),(0,2,7),(0,4,16),(0,6,5),(0,8,12),
                (1,4,3),(2,3,9),(2,4,8),(2,6,10),(3,1,14))
        for color in THEME_COLORS:
            for r,c,total in labels:
                with self.subTest(color=color,r=r,c=c,total=total):
                    a=np.asarray(base).copy()
                    cx,cy,radius=badge_geometry(r,c)
                    left,top=cx-20,cy-20
                    patch=a[top:top+41,left:left+41].astype(np.float64)
                    yy,xx=np.indices((41,41))
                    distance=(xx-20)**2+(yy-20)**2
                    ink=(np.ptp(patch,axis=2)>45)|(patch.mean(axis=2)<190)
                    old_color=np.median(patch[(distance<14**2)&ink],axis=0)
                    channels=255-old_color>40
                    opacity=((patch[:,:,channels]-old_color[channels])/(255-old_color[channels])).mean(axis=2)
                    opacity=np.clip(opacity,0,1)
                    recolored=np.round(np.asarray(color)+(255-np.asarray(color))*opacity[:,:,None]).astype(np.uint8)
                    disk=distance<radius**2
                    target=a[top:top+41,left:left+41]
                    target[disk]=recolored[disk]
                    self.assertEqual(self.recognizer.read_total(Image.fromarray(a),r,c),total)

    def test_existing_game_font_is_read_after_partial_input(self):
        # Real filled-number screenshots from the reference Universe assistant.
        reference=ROOT.parent/"universe"/"tests"/"fixtures"
        if not reference.exists():
            self.skipTest("外部参考截图未随独立源码分发")
        for name,count in (("after-two",40),("after-eleven",49),("bold-digits",9)):
            frame=canonical(Image.open(reference/(name+".png")))
            values=[self.recognizer.read_cell(frame,r,c)[0] for r in range(9) for c in range(9)]
            self.assertEqual(sum(n>0 for n in values),count)
            if name=="bold-digits":
                self.assertEqual(values[:9],list(range(1,10)))
            elif name=="after-eleven":
                labels=json.loads((reference/"player-labels.json").read_text())
                for r,c,n in labels:
                    self.assertEqual(values[(r-1)*9+c-1],n)
        frame=canonical(Image.open(reference/"hovered-six.png"))
        self.assertEqual(self.recognizer.read_cell(frame,0,0)[0],6)


class SolverTests(unittest.TestCase):
    def test_unique_solutions_and_resume_with_given_digits(self):
        for name in ("board1","board2","board3"):
            cages=cages_for(name)
            solution=solve_unique([0]*81,cages)
            self.assertEqual(solution,EXPECTED[name]["solution"])
            validate_solution(solution,cages)
            for house in HOUSES:
                self.assertEqual(sum(solution[i] for i in house),45)
            board=[n if i%3==0 else 0 for i,n in enumerate(solution)]
            self.assertEqual(solve_unique(board,cages),solution)

    def test_bad_coverage_totals_and_cage_repeats_are_rejected(self):
        cages=cages_for("board1")
        with self.assertRaises(PuzzleError):
            validate_cages(cages[:-1])
        with self.assertRaises(PuzzleError):
            validate_cages((Cage(cages[0].total+1,cages[0].cells),)+cages[1:])
        board=[0]*81
        board[10]=board[27]=5  # Same cage, different rows, columns and 3x3 boxes.
        with self.assertRaisesRegex(PuzzleError,"冲突"):
            solve_unique(board,cages)

    def test_unsatisfiable_and_ambiguous_puzzles_stop(self):
        board=[0]*81
        board[0]=2
        with self.assertRaisesRegex(PuzzleError,"无解"):
            solve_unique(board,cages_for("board1"))
        row_cages=tuple(Cage(45,tuple(range(r*9,r*9+9))) for r in range(9))
        with self.assertRaisesRegex(PuzzleError,"多个解"):
            solve_unique([0]*81,row_cages)

    def test_cancellation_and_limits(self):
        def stopped():
            raise Stopped()
        with self.assertRaises(Stopped):
            solve_unique([0]*81,cages_for("board1"),check=stopped)
        with self.assertRaisesRegex(PuzzleError,"上限"):
            solve_unique([0]*81,cages_for("board1"),timeout=0)

    def test_support_matches_small_exhaustive_example(self):
        from itertools import product
        domains=(sum(1<<n for n in (1,2,3)),sum(1<<n for n in (2,3,4)),sum(1<<n for n in (1,3,5)))
        for distinct,function in ((True,cage_support),(False,sum_support)):
            for total in range(4,13):
                supported=[0]*3
                for values in product((1,2,3),(2,3,4),(1,3,5)):
                    if sum(values)==total and (not distinct or len(set(values))==3):
                        for i,n in enumerate(values):
                            supported[i]|=1<<n
                self.assertEqual(function(domains,total),tuple(supported))


class FakeGame:
    """Game protocol simulator: verifies every click against the known solution."""
    def __init__(self,partial=False,puzzle="board1",frame_name=None):
        self.puzzle=puzzle
        filename="board3-hover-dot.png" if puzzle=="board3" else "hard-62.png" if puzzle=="hard-62" else "board1.png"
        if frame_name:
            filename=frame_name+".png"
        self.frame=canonical(Image.open(FIXTURES/filename))
        self.board=[n if partial and i%3==0 else 0 for i,n in enumerate(EXPECTED[puzzle]["solution"])]
        self.popup=0 if frame_name=="board3-open-menu" else None if puzzle in ("board3","hard-62") else 4
        self.menu=True
        self.inputs=[]
        self.finished=False

    def set_timing(self,value):
        self.timing=value

    def capture(self):
        return menu_frame("easy",set(range(2,31)) if self.finished else None) if self.menu else self.frame

    def click(self,point,check):
        check()
        if self.menu:
            assert point==(690.,423.)
            self.menu=False
            return
        if self.popup is None:
            nearest=min(range(81),key=lambda i:sum((a-b)**2 for a,b in zip(point,cell_center(*divmod(i,9)))))
            assert not self.board[nearest]
            self.popup=nearest
        else:
            r,c=divmod(self.popup,9)
            n=min(range(1,10),key=lambda n:sum((a-b)**2 for a,b in zip(point,digit_point(r,c,n))))
            assert n==EXPECTED[self.puzzle]["solution"][self.popup]
            self.board[self.popup]=n
            self.inputs.append((self.popup,n))
            self.popup=None
            if all(self.board):
                self.menu=self.finished=True

    def move(self,point,check):
        check()

    def read(self,image):
        return Reading(self.board.copy(),cages_for(self.puzzle),[1.]*81,self.frame,self.popup)


class InstantRunner(Runner):
    def wait(self,seconds):
        self.check()


class RunnerTests(unittest.TestCase):
    def run_two_level_simulation(self,settings,stop_before_third=False):
        control=Control()
        class TwoLevelGame(FakeGame):
            def __init__(self):
                super().__init__(puzzle="board3",frame_name="board3-clean")
                self.finished_count=0
                self.chosen=[]
                self.attempted_third=False

            def capture(self):
                if not self.menu:
                    return self.frame
                if self.finished_count==1:
                    return canonical(Image.open(FIXTURES/"easy-after-one.png"))
                return menu_frame("easy",set(range(self.finished_count+1,31)))

            def click(self,point,check):
                check()
                if self.menu:
                    level=self.finished_count+1
                    assert point==(690.+123*(level-1),423.)
                    if level==3 and stop_before_third:
                        self.attempted_third=True
                        control.stop.set()
                        check()
                    assert level<=2
                    self.chosen.append(level)
                    self.board=[0]*81
                    self.popup=None
                    self.menu=False
                    return
                super().click(point,check)
                if self.menu:
                    self.finished_count+=1

        game=TwoLevelGame()
        events=[]
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as folder:
            runner=InstantRunner(game,control,settings,lambda k,v:events.append((k,v)),folder,recognizer=game)
            runner.run()
        self.assertFalse([v for k,v in events if k=="error"],events)
        return game,runner,events

    def test_continuous_play_fills_second_level_after_real_first_completion(self):
        game,runner,events=self.run_two_level_simulation(Settings(limit=2))
        self.assertEqual(game.chosen,[1,2])
        self.assertEqual(game.inputs,list(enumerate(EXPECTED["board3"]["solution"]))*2)
        self.assertEqual(runner.completed,2)
        self.assertEqual(events[-1],("done",2))

    def test_default_settings_continue_beyond_one_level(self):
        game,runner,events=self.run_two_level_simulation(Settings(),stop_before_third=True)
        self.assertEqual(game.chosen,[1,2])
        self.assertTrue(game.attempted_third)
        self.assertEqual(runner.completed,2)
        self.assertEqual(events[-1],("done",2))

    def test_explicit_one_level_limit_or_disabled_continuous_stops(self):
        for settings in (Settings(limit=1),Settings(continuous=False)):
            with self.subTest(settings=settings):
                game,runner,events=self.run_two_level_simulation(settings)
                self.assertEqual(game.chosen,[1])
                self.assertEqual(runner.completed,1)
                self.assertEqual(events[-1],("done",1))

    def run_simulation(self,partial=False,pause_after_final=False,pause_midway=False,puzzle="board1",real_initial_reading=False,frame_name=None):
        game=FakeGame(partial,puzzle,frame_name)
        events=[]
        recognizer=game
        if real_initial_reading:
            actual=Recognizer()
            class InitialScreenshotReader:
                def read(self,image):
                    return actual.read(image) if not game.inputs else game.read(image)
            recognizer=InitialScreenshotReader()
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as folder:
            assert Path(folder).resolve().parent==(ROOT/"tests").resolve()
            runner=InstantRunner(game,Control(),Settings(limit=1),lambda k,v:events.append((k,v)),folder,recognizer=recognizer)
            paused=False
            def wait(seconds):
                nonlocal paused
                if not paused and ((pause_after_final and runner.pending_completion) or
                                   (pause_midway and len(game.inputs)>=10)):
                    paused=True
                    raise RestartReading()
                runner.check()
            runner.wait=wait
            runner.run()
            self.assertFalse([v for k,v in events if k=="error"],events)
            self.assertEqual(runner.completed,1)
            self.assertEqual(len(game.inputs),54 if partial else 81)
            self.assertEqual(len({i for i,n in game.inputs}),len(game.inputs))
            self.assertEqual(game.board,EXPECTED[puzzle]["solution"])
            self.assertEqual(events[-1],("done",1))
            if real_initial_reading:
                self.assertEqual(game.inputs[0],(0,EXPECTED[puzzle]["solution"][0]))

    def test_full_input_and_automatic_completion(self):
        self.run_simulation()

    def test_only_remaining_cells_are_filled(self):
        self.run_simulation(partial=True)

    def test_resume_after_some_digits(self):
        self.run_simulation(pause_midway=True)

    def test_final_click_pause_preserves_completion_count(self):
        self.run_simulation(pause_after_final=True)

    def test_failed_real_game_capture_reaches_filling(self):
        self.run_simulation(puzzle="board3",real_initial_reading=True)

    def test_actual_hard62_capture_reaches_filling_and_completion(self):
        self.run_simulation(puzzle="hard-62",real_initial_reading=True)

    def test_red_and_pink_theme_captures_reach_filling(self):
        for name in ("board3-red","board3-pink-cyan"):
            with self.subTest(name=name):
                self.run_simulation(puzzle="board3",real_initial_reading=True,frame_name=name)

    def test_restart_from_real_open_menu_fills_correct_digit_first(self):
        self.run_simulation(puzzle="board3",real_initial_reading=True,frame_name="board3-open-menu")

    def test_filling_all_cells_never_captures_between_clicks(self):
        class DirectClickGame(FakeGame):
            filling_started=False

            def click(self,point,check):
                self.filling_started=True
                super().click(point,check)

            def capture(self):
                if self.filling_started and not self.menu:
                    raise AssertionError("填写途中不应截图或识别数字菜单")
                return super().capture()

        game=DirectClickGame(puzzle="board3",frame_name="board3-clean")
        game.menu=False
        events=[]
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as folder:
            runner=InstantRunner(game,Control(),Settings(),lambda k,v:events.append((k,v)),
                                 folder,recognizer=Recognizer())
            runner.fill_board()
        self.assertEqual(game.inputs,list(enumerate(EXPECTED["board3"]["solution"])))
        self.assertEqual(game.board,EXPECTED["board3"]["solution"])
        self.assertEqual(runner.completed,1)
        self.assertEqual([v for k,v in events if k=="progress"],[(i,81) for i in range(1,82)])

    def test_transient_recognition_error_retries_before_filling(self):
        game=FakeGame()
        game.menu=False
        class SettlingReader:
            calls=0
            def read(self,image):
                self.calls+=1
                if self.calls==1:
                    raise RecognitionError("悬停动画未消退")
                return game.read(image)
        recognizer=SettlingReader()
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as folder:
            assert Path(folder).resolve().parent==(ROOT/"tests").resolve()
            runner=InstantRunner(game,Control(),Settings(),lambda k,v:None,folder,recognizer=recognizer)
            reading,solution=runner.stable_reading()
        self.assertEqual(recognizer.calls,3)
        self.assertEqual(solution,EXPECTED["board1"]["solution"])
        self.assertEqual(game.inputs,[])

    def test_does_not_count_level_that_is_still_unfinished(self):
        game=FakeGame()
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as folder:
            assert Path(folder).resolve().parent==(ROOT/"tests").resolve()
            runner=InstantRunner(game,Control(),Settings(),lambda k,v:None,folder,recognizer=game)
            runner.pending_level=("简单",1)
            with self.assertRaises(RecognitionError):
                runner.confirm_completion(inspect_level_menu(menu_frame("easy")))
            self.assertEqual(runner.completed,0)

    def test_switches_all_three_difficulties_only_when_complete(self):
        from sudoku_assistant.killer.vision import DIFFICULTIES
        class CompleteGame:
            group="简单"
            visited=[]
            def set_timing(self,value):
                pass
            def capture(self):
                return menu_frame({"简单":"easy","中等":"medium","困难":"hard"}[self.group],set())
            def click(self,point,check):
                check()
                self.group=next(g for g,p in DIFFICULTIES.items() if p==point)
                self.visited.append(self.group)
        game=CompleteGame()
        events=[]
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as folder:
            assert Path(folder).resolve().parent==(ROOT/"tests").resolve()
            runner=InstantRunner(game,Control(),Settings(difficulty="依次完成三个难度",limit=0),
                                 lambda k,v:events.append((k,v)),folder)
            runner.run()
        self.assertEqual(game.visited,["简单","中等","困难"])
        self.assertFalse([v for k,v in events if k=="error"])


if __name__=="__main__":
    unittest.main()
