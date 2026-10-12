"""Offline regression and simulated-game checks. Never injects Windows input."""
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
from sudoku_assistant.universe.solver import solve_unique, PuzzleError
from sudoku_assistant.universe.vision import (Recognizer, RecognitionError, canonical, board_visible, level_menu_visible,
                    available_levels, completion_visible, cell_box, glyph_mask)
from sudoku_assistant.universe.vision import DIFFICULTIES, LEVEL_GROUPS, similarity, inspect_level_menu, MenuNotReady
from sudoku_assistant.universe.runner import Runner, Control, Settings, Stopped, RestartReading
from sudoku_assistant.universe.wincontrol import ForegroundLost, WindowsGame, Input, MouseInput

FIXTURES = Path(__file__).resolve().parent / "fixtures"
EXPECTED = [int(n) for n in (
    "872095001" "053008600" "400710208" "940000103" "020006084"
    "608000000" "000007000" "304800500" "715329806")]


def fixture(name):
    with Image.open(FIXTURES / (name + ".png")) as image:
        return canonical(image)


class SimulatedGame:
    """Real screenshot surfaces, independent hit testing, and visible orange values."""
    def __init__(self, missed=False, no_chooser=False):
        self.menu = fixture("easy")
        self.original = fixture("board")
        self.complete = fixture("complete")
        self.phase = "menu"
        self.board = EXPECTED.copy()
        self.solution = solve_unique(EXPECTED)
        self.clicked = []
        self.selected = None
        self.level = None
        self.animation_frames = 0
        self.missed, self.no_chooser = missed, no_chooser
        self.screen = self.original.copy()
        # Use real glyph pixels, rather than generating the recognizer's templates.
        self.glyphs = {}
        for i, n in enumerate(EXPECTED):
            if n and n not in self.glyphs:
                r, c = divmod(i, 9)
                x0, y0, x1, y1 = self.bounds(r, c)
                crop = self.original.crop((x0 + 13, y0 + 10, x1 - 13, y1 - 10))
                mask = glyph_mask(crop)
                yy, xx = np.nonzero(mask)
                self.glyphs[n] = mask[yy.min():yy.max() + 1, xx.min():xx.max() + 1]

    @staticmethod
    def bounds(r, c):
        # Independent reference hit regions, including the eight-pixel separators.
        starts = (640, 724, 807, 899, 983, 1066, 1157, 1241, 1324)
        ys = (192, 276, 359, 451, 535, 618, 709, 793, 876)
        return starts[c], ys[r], starts[c] + 83, ys[r] + 83

    def paint_value(self, r, c, n):
        x0, y0, x1, y1 = self.bounds(r, c)
        ImageDraw.Draw(self.screen).rectangle((x0 + 4, y0 + 4, x1 - 4, y1 - 4), fill=(250, 250, 250))
        if n:
            mask = self.glyphs[n]
            h, w = mask.shape
            rgb = np.full((h, w, 3), 250, dtype=np.uint8)
            rgb[mask] = (255, 172, 103)
            self.screen.paste(Image.fromarray(rgb), (x0 + (83 - w) // 2, y0 + (83 - h) // 2))

    def capture(self):
        if self.phase == "menu":
            return self.menu.copy()
        if self.phase == "animation":
            self.animation_frames += 1
            if self.animation_frames <= 4:
                return self.complete.copy()
            self.phase = "menu"
            x, y = self.level
            ImageDraw.Draw(self.menu).rectangle((x - 52, y - 54, x + 52, y + 54), fill=(105, 140, 160))
            return self.menu.copy()
        return self.screen.copy()

    def move(self, point, check):
        check()

    def click(self, point, check):
        check()
        x, y = map(float, point)
        self.clicked.append((x, y))
        if self.phase == "menu":
            # Known reference easy-page centers, tested independently of the locator.
            centers = [(690 + 123 * c, 423 + 154 * r) for r in range(3) for c in range(10)]
            candidate = min(centers, key=lambda p: abs(p[0] - x) + abs(p[1] - y))
            assert abs(candidate[0] - x) < 12 and abs(candidate[1] - y) < 12
            assert candidate not in centers[:3], "Completed level was selected"
            self.level = candidate
            self.board = EXPECTED.copy()
            self.screen = self.original.copy()
            for i, n in enumerate(self.board):
                if not n:
                    self.paint_value(*divmod(i, 9), 0)
            self.phase = "board"
            self.selected = None
            self.animation_frames = 0
            return
        assert self.phase == "board", "Clicked during the completion animation"
        if self.selected is None:
            targets = [(r, c, self.bounds(r, c)) for r in range(9) for c in range(9)]
            hits = [(r, c) for r, c, (x0, y0, x1, y1) in targets if x0 < x < x1 and y0 < y < y1]
            assert len(hits) == 1
            r, c = hits[0]
            assert not self.board[r * 9 + c], "A given was clicked"
            self.selected = (r, c)
            if not self.no_chooser:
                x0, y0, x1, y1 = self.bounds(r, c)
                ImageDraw.Draw(self.screen).rectangle((x0 + 3, y0 + 3, x1 - 3, y1 - 3), fill=(255, 172, 103))
            return
        r, c = self.selected
        x0, y0, x1, y1 = self.bounds(r, c)
        assert x0 < x < x1 and y0 < y < y1
        dc, dr = int((x - x0) / 83 * 3), int((y - y0) / 83 * 3)
        n = dr * 3 + dc + 1
        assert n == self.solution[r * 9 + c], "Wrong inline number was clicked"
        self.selected = None
        self.paint_value(r, c, 0 if self.missed else n)
        if not self.missed:
            self.board[r * 9 + c] = n
        if self.board == self.solution:
            self.phase = "animation"


class FastRunner(Runner):
    completion_timeout = 0.5

    def wait(self, seconds):
        self.check()


LATEST = [int(n) for n in (
    "643791528" "512348976" "897562100" "720000010" "430000800"
    "085130067" "009007382" "004800601" "200003040")]


class AssistantTests(unittest.TestCase):
    def setUp(self):
        self.recognizer = Recognizer()

    def test_board_exactly_matches_manual_transcription(self):
        result = self.recognizer.read(fixture("board"))
        self.assertEqual(result.board, EXPECTED)
        self.assertEqual(sum(n > 0 for n in result.board), 38)

    def test_scaled_fullscreen_images(self):
        image = fixture("board")
        for size in ((2560, 1440), (1920, 1080), (1600, 900)):
            with self.subTest(size=size):
                self.assertEqual(self.recognizer.read(image.resize(size, Image.Resampling.LANCZOS)).board, EXPECTED)

    def test_three_difficulty_pages(self):
        for name, count, first_y in (("easy", 27, 423), ("medium", 30, 423), ("hard", 40, 346)):
            with self.subTest(name=name):
                frame = fixture(name)
                self.assertTrue(level_menu_visible(frame))
                self.assertFalse(board_visible(frame))
                buttons = available_levels(frame)
                self.assertEqual(len(buttons), count)
                self.assertAlmostEqual(buttons[0][1], first_y, delta=3)
        self.assertAlmostEqual(available_levels(fixture("easy"))[0][0], 1059, delta=3)

    def test_four_additional_level_group_pages(self):
        for name,count,first,last in (
            ("master",50,(690,269),(1796,883)),
            ("pro",25,(690,423),(1182,730)),
            ("pro2",25,(690,423),(1182,730)),
            ("pro3",25,(690,423),(1182,730))):
            with self.subTest(name=name):
                frame=fixture(name)
                self.assertTrue(level_menu_visible(frame))
                buttons=available_levels(frame)
                self.assertEqual(len(buttons),count)
                self.assertAlmostEqual(buttons[0][0],first[0],delta=4)
                self.assertAlmostEqual(buttons[0][1],first[1],delta=4)
                self.assertAlmostEqual(buttons[-1][0],last[0],delta=4)
                self.assertAlmostEqual(buttons[-1][1],last[1],delta=4)

    def test_all_seven_groups_are_available_in_correct_order(self):
        self.assertEqual(LEVEL_GROUPS,("简单","中等","困难","Master Ninja","Pro Pack","Pro Pack 2","Pro Pack 3"))
        for name in LEVEL_GROUPS:
            self.assertIn(name,DIFFICULTIES)

    def test_sweep_seven_completed_groups_switches_every_group(self):
        class CompletedGroups:
            def __init__(self):
                self.current="easy"
                self.clicked=[]
            def capture(self):
                frame=fixture(self.current)
                count=50 if self.current=="master" else 40 if self.current=="hard" else 25 if self.current.startswith("pro") else 30
                rows=(269,423,576,730,883) if self.current=="master" else (346,499,653,807) if self.current=="hard" else (423,576,730)
                draw=ImageDraw.Draw(frame)
                for i in range(count):
                    row,col=divmod(i,10);x,y=690+123*col,rows[row]
                    draw.rounded_rectangle((x-46,y-46,x+46,y+46),radius=10,fill=(105,140,160))
                return frame
            def click(self,point,check):
                check()
                names={DIFFICULTIES[n]:n for n in LEVEL_GROUPS}
                name=names[point]
                self.clicked.append(name)
                self.current={"简单":"easy","中等":"medium","困难":"hard","Master Ninja":"master",
                    "Pro Pack":"pro","Pro Pack 2":"pro2","Pro Pack 3":"pro3"}[name]
        game=CompletedGroups();events=[]
        runner=FastRunner(game,Control(),Settings(difficulty="依次完成全部七组"),
            lambda k,v:events.append((k,v)),ROOT/"test-data",self.recognizer)
        runner.run()
        self.assertEqual(game.clicked,list(LEVEL_GROUPS))
        self.assertEqual([v for k,v in events if k=="error"],[])

    def test_optimized_similarity_preserves_original_scores(self):
        rng=np.random.default_rng(51)
        for _ in range(4):
            a=rng.random((40,32),dtype=np.float32)
            b=rng.random((40,32),dtype=np.float32)
            padded=np.pad(b,((2,2),(2,2)))
            original=max(float(2*np.minimum(a,padded[y:y+40,x:x+32]).sum()/
                max(1,a.sum()+padded[y:y+40,x:x+32].sum())) for y in range(5) for x in range(5))
            self.assertAlmostEqual(similarity(a,b),original,places=6)

    def test_complete_message_is_distinct_from_playing(self):
        self.assertTrue(completion_visible(fixture("complete")))
        for name in ("easy", "medium", "hard", "board"):
            self.assertFalse(completion_visible(fixture(name)))

    def test_orange_entered_values_are_read(self):
        game = SimulatedGame()
        for i, n in enumerate(EXPECTED):
            if n:
                game.paint_value(*divmod(i, 9), n)
        # Remove the initial nine-number chooser before reading a simulated surface.
        game.paint_value(0, 6, 0)
        self.assertEqual(self.recognizer.read(game.screen).board, EXPECTED)

    def test_actual_hovered_white_digit_inside_orange_circle(self):
        frame = fixture("hovered-six")
        expected = [int(n) for n in (
            "640701508" "500348070" "090062100" "720000010" "430000800"
            "085130067" "009007382" "004800601" "200003040")]
        self.assertEqual(self.recognizer.read(frame).board, expected)
        self.assertEqual(self.recognizer.read_cell(frame, 0, 0)[0], 6)

    def test_same_real_digit_after_mouse_leaves(self):
        hovered = self.recognizer.read(fixture("hovered-six"))
        unhovered = self.recognizer.read(fixture("unhovered-six"))
        self.assertEqual(hovered.board, unhovered.board)
        self.assertEqual(solve_unique(hovered.board), solve_unique(unhovered.board))

    def test_actual_second_typed_digit_three(self):
        frame = fixture("after-two")
        reading = self.recognizer.read(frame)
        self.assertEqual(reading.board[0], 6)
        self.assertEqual(reading.board[2], 3)
        self.assertEqual(sum(n > 0 for n in reading.board), 40)
        self.assertEqual(solve_unique(reading.board)[2], 3)

    def test_latest_actual_partial_board_including_five(self):
        frame = fixture("after-eleven")
        reading = self.recognizer.read(frame)
        self.assertEqual(reading.board, LATEST)
        self.assertEqual(sum(n > 0 for n in reading.board), 49)
        self.assertEqual(self.recognizer.read_cell(frame, 2, 3)[0], 5)

    def test_independently_rendered_bold_digits_one_to_nine(self):
        frame = fixture("bold-digits")
        for n in range(1, 10):
            with self.subTest(number=n):
                self.assertEqual(self.recognizer.read_cell(frame, 0, n - 1)[0], n)

    def test_original_video_before_hover_and_after_second_input(self):
        before = self.recognizer.read(fixture("video-before"))
        hover = self.recognizer.read(fixture("video-hover"))
        after = self.recognizer.read(fixture("video-after-two"))
        self.assertEqual(sum(n > 0 for n in before.board), 38)
        self.assertEqual(hover.board[0], 6)
        self.assertEqual(after.board[0], 6)
        self.assertEqual(after.board[2], 3)
        self.assertEqual(sum(n > 0 for n in after.board), 40)
        self.assertEqual(solve_unique(before.board), solve_unique(after.board))

    def test_solution_obeys_every_sudoku_constraint(self):
        answer = solve_unique(EXPECTED)
        for r in range(9):
            self.assertEqual(set(answer[r * 9:r * 9 + 9]), set(range(1, 10)))
            self.assertEqual({answer[r + c * 9] for c in range(9)}, set(range(1, 10)))
        for br in range(0, 9, 3):
            for bc in range(0, 9, 3):
                self.assertEqual({answer[(br + r) * 9 + bc + c] for r in range(3) for c in range(3)}, set(range(1, 10)))

    def test_conflicting_values_stop_solver(self):
        wrong = EXPECTED.copy()
        wrong[3] = 8
        with self.assertRaises(PuzzleError):
            solve_unique(wrong)

    def test_ambiguous_puzzle_is_rejected(self):
        with self.assertRaises(PuzzleError):
            solve_unique([0] * 81)

    def test_wrong_aspect_ratio_is_rejected(self):
        with self.assertRaises(RecognitionError):
            self.recognizer.read(Image.new("RGB", (1280, 1024)))

    def run_simulation(self, game, count=2):
        events = []
        runner = FastRunner(game, Control(), Settings(limit=count), lambda k, v: events.append((k, v)), ROOT / "test-data", self.recognizer)
        runner.run()
        return runner, events

    def test_two_consecutive_levels_with_animation_and_orange_verification(self):
        game = SimulatedGame()
        runner, events = self.run_simulation(game)
        self.assertEqual([v for k, v in events if k == "error"], [])
        self.assertEqual(runner.completed, 2)
        self.assertEqual(len(game.clicked), 2 * (1 + 2 * (81 - 38)))
        self.assertEqual([v for k, v in events if k == "done"], [2])
        self.assertNotEqual(game.clicked[0], game.clicked[87])

    def test_no_digit_recognition_after_first_value_is_entered(self):
        game = SimulatedGame()
        class InitialOnlyRecognizer(Recognizer):
            def read_cell(self, frame, r, c):
                if game.board[3] != 0:
                    raise AssertionError("Digits must not be re-read after filling starts")
                return super().read_cell(frame, r, c)
        events = []
        runner = FastRunner(game, Control(), Settings(limit=1),
            lambda k, v: events.append((k, v)), ROOT / "test-data", InitialOnlyRecognizer())
        runner.run()
        self.assertEqual(runner.completed, 1)
        self.assertEqual([v for k, v in events if k == "error"], [])

    def test_resume_with_first_empty_cell_chooser_already_open(self):
        game = SimulatedGame()
        game.phase = "board"
        game.level = (1059,423)
        game.selected = (0,3)
        x0,y0,x1,y1 = game.bounds(0,3)
        ImageDraw.Draw(game.screen).rectangle((x0+3,y0+3,x1-3,y1-3), fill=(255,172,103))
        runner, events = self.run_simulation(game,count=1)
        self.assertEqual(runner.completed,1)
        self.assertEqual(len(game.clicked),85)
        self.assertEqual([v for k,v in events if k=="error"],[])

    def test_two_levels_using_real_player_font_for_three_and_six(self):
        game = SimulatedGame()
        actual = fixture("after-two")
        for n, c in ((6, 0), (3, 2)):
            x0, y0, x1, y1 = game.bounds(0, c)
            crop = actual.crop((x0 + 13, y0 + 10, x1 - 13, y1 - 10))
            mask = glyph_mask(crop)
            yy, xx = np.nonzero(mask)
            game.glyphs[n] = mask[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
        runner, events = self.run_simulation(game)
        self.assertEqual(runner.completed, 2)
        self.assertEqual([v for k, v in events if k == "error"], [])

    def test_resuming_during_countdown_does_not_emit_empty_error(self):
        class CountdownRunner(FastRunner):
            first_countdown = True
            def wait(self, seconds):
                if self.first_countdown:
                    self.first_countdown = False
                    raise RestartReading()
                super().wait(seconds)
        events = []
        runner = CountdownRunner(SimulatedGame(), Control(), Settings(limit=1),
            lambda k, v: events.append((k, v)), ROOT / "test-data", self.recognizer)
        runner.run()
        self.assertEqual(runner.completed, 1)
        self.assertEqual([v for k, v in events if k == "error"], [])

    def test_whole_board_using_independent_heavy_font_for_all_nine_digits(self):
        game = SimulatedGame()
        actual = fixture("bold-digits")
        for n in range(1, 10):
            x0, y0, x1, y1 = game.bounds(0, n - 1)
            mask = glyph_mask(actual.crop((x0 + 13, y0 + 10, x1 - 13, y1 - 10)))
            yy, xx = np.nonzero(mask)
            game.glyphs[n] = mask[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
        runner, events = self.run_simulation(game, count=1)
        self.assertEqual(runner.completed, 1)
        self.assertEqual([v for k, v in events if k == "error"], [])

    def test_continue_latest_actual_board_through_all_remaining_32_cells(self):
        game = SimulatedGame()
        actual = fixture("after-eleven")
        game.original = actual.copy()
        game.screen = actual.copy()
        game.board = LATEST.copy()
        game.solution = solve_unique(LATEST)
        game.phase = "board"
        game.level = (1059, 423)
        for row, col, n in ((0,0,6),(0,2,3),(0,4,9),(0,7,2),(1,1,1),(2,0,8),(2,2,7),(2,3,5)):
            mask = glyph_mask(actual.crop(tuple(round(v) for v in (
                game.bounds(row,col)[0]+13,game.bounds(row,col)[1]+10,
                game.bounds(row,col)[2]-13,game.bounds(row,col)[3]-10))))
            yy, xx = np.nonzero(mask)
            game.glyphs[n] = mask[yy.min():yy.max()+1,xx.min():xx.max()+1]
        # Four has not been typed in the real screenshot; use an independent
        # heavy-face sample for the simulated output.
        bold = fixture("bold-digits")
        x0,y0,x1,y1=game.bounds(0,3)
        mask=glyph_mask(bold.crop((x0+13,y0+10,x1-13,y1-10)))
        yy,xx=np.nonzero(mask)
        game.glyphs[4]=mask[yy.min():yy.max()+1,xx.min():xx.max()+1]
        runner, events = self.run_simulation(game,count=1)
        self.assertEqual(runner.completed,1)
        self.assertEqual(game.board,game.solution)
        self.assertEqual(len(game.clicked),64)
        self.assertEqual([v for k,v in events if k=="error"],[])

    def test_missed_input_is_detected_by_missing_completion(self):
        runner, events = self.run_simulation(SimulatedGame(missed=True))
        self.assertEqual(runner.completed, 0)
        self.assertEqual(len([v for k, v in events if k == "error"]), 1)

    def test_stop_prevents_any_click(self):
        game, control, events = SimulatedGame(), Control(), []
        control.stop.set()
        runner = FastRunner(game, control, Settings(), lambda k, v: events.append((k, v)), ROOT / "test-data")
        runner.run()
        self.assertEqual(game.clicked, [])
        self.assertEqual([v for k, v in events if k == "done"], [0])

    def test_foreground_loss_pauses_without_clicking(self):
        class MissingGame:
            def capture(self):
                raise ForegroundLost("lost foreground")
            def click(self, *args):
                raise AssertionError("Must never click")
        control, events = Control(), []
        def emit(k, v):
            events.append((k, v))
            if k == "log" and v == "lost foreground":
                control.stop.set()
        runner = FastRunner(MissingGame(), control, Settings(), emit, ROOT / "test-data")
        runner.run()
        self.assertTrue(control.paused.is_set())
        self.assertEqual([v for k, v in events if k == "done"], [0])


if __name__ == "__main__":
    unittest.main()
