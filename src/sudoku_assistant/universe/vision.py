"""Image-only recognition for the supplied 16:9 Sudoku Universe layout."""
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache
import sys

import numpy as np
from PIL import Image, ImageFilter

SIZE = (2048, 1152)
BOARD = (640, 192, 1408, 960)
DIFFICULTIES = {
    "简单": (277, 332), "中等": (277, 418), "困难": (277, 505), "无限": (277, 590),
    "Master Ninja": (277, 714), "Pro Pack": (277, 808),
    "Pro Pack 2": (277, 903), "Pro Pack 3": (277, 999),
}
LEVEL_GROUPS = ("简单", "中等", "困难", "Master Ninja", "Pro Pack", "Pro Pack 2", "Pro Pack 3")
LEVEL_LAYOUTS = {
    "简单": (1,30,(423,576,730)),
    "中等": (31,30,(423,576,730)),
    "困难": (61,40,(346,499,653,807)),
    "Master Ninja": (101,50,(269,423,576,730,883)),
    "Pro Pack": (151,25,(423,576,730)),
    "Pro Pack 2": (176,25,(423,576,730)),
    "Pro Pack 3": (201,25,(423,576,730)),
}
ENDLESS_CHOICES = {"简单": (864,577), "中等": (1024,577), "困难": (1184,577)}


def resource_dir():
    return Path(__file__).resolve().parent / "assets"


def orange_mask(a):
    a = np.asarray(a).astype(int)
    return (a[:, :, 0] - a[:, :, 1] > 20) & (a[:, :, 1] - a[:, :, 2] > 20)


@lru_cache(maxsize=1)
def completion_template():
    with np.load(resource_dir() / "completion.npz", allow_pickle=False) as data:
        return data["mask"].copy()


def completion_visible(frame):
    sample = orange_mask(np.asarray(frame)[535:615, 830:1260])
    template = completion_template()
    overlap = np.logical_and(sample, template).sum()
    return overlap / max(1, template.sum()) > 0.78 and overlap / max(1, sample.sum()) > 0.70


class RecognitionError(ValueError):
    pass


class MenuNotReady(RecognitionError):
    pass


@dataclass(frozen=True)
class LevelMenu:
    group: str
    remaining: tuple
    centers: tuple
    completed: int
    total: int


@dataclass
class Reading:
    board: list
    scores: list
    frame: Image.Image


def canonical(image):
    w, h = image.size
    if abs(w / h - 16 / 9) > 0.035:
        raise RecognitionError("目前支持 16:9 游戏画面，请使用 2560×1440 全屏。")
    return image.convert("RGB").resize(SIZE, Image.Resampling.LANCZOS)


def cell_box(r, c):
    # The blue separators between 3×3 blocks are wider than ordinary grid lines.
    gap, block = 8, (768 - 16) / 3
    step = block / 3
    x = 640 + (c // 3) * (block + gap) + (c % 3) * step
    y = 192 + (r // 3) * (block + gap) + (r % 3) * step
    return x, y, x + step, y + step


def cell_center(r, c):
    x0, y0, x1, y1 = cell_box(r, c)
    return (x0 + x1) / 2, (y0 + y1) / 2


def digit_point(r, c, n):
    x0, y0, x1, y1 = cell_box(r, c)
    dr, dc = divmod(n - 1, 3)
    return x0 + (dc + 0.5) / 3 * (x1 - x0), y0 + (dr + 0.5) / 3 * (y1 - y0)


def board_visible(frame):
    a = np.asarray(frame)
    hits = 0
    for r in range(9):
        for c in range(9):
            x0, y0, _, _ = cell_box(r, c)
            p = a[int(y0 + 12):int(y0 + 18), int(x0 + 12):int(x0 + 18)].astype(int)
            if np.mean((p.min(axis=2) > 208) & (np.ptp(p, axis=2) < 55)) > 0.75:
                hits += 1
    # Also require the blue cross-block separators, to reject menus and dialogs.
    gaps = a[300:850, 894:899].astype(int)
    blue = (gaps[:, :, 2] - gaps[:, :, 0] > 10) & (gaps[:, :, 1] > 130)
    return hits >= 72 and np.mean(blue) > 0.55


def level_menu_visible(frame):
    a = np.asarray(frame)
    hits = 0
    for x, y in (DIFFICULTIES[name] for name in ("简单", "中等", "困难", "无限")):
        patch = a[y - 12:y + 12, x - 105:x - 80].astype(int)
        bright = (patch[:, :, 0] > 210) & (patch[:, :, 1] > 210) & (patch[:, :, 2] > 125)
        if patch.size and np.mean(bright) > 0.85:
            hits += 1
    return hits == 4 and not board_visible(frame)


def intervals(flags, bridge=0):
    flags = np.array(flags, dtype=bool).copy()
    if bridge:
        for start, end in intervals(~flags):
            if start > 0 and end < len(flags) and end - start <= bridge:
                flags[start:end] = True
    edges = np.diff(np.r_[False, flags, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def selected_level_group(frame):
    a=np.asarray(frame)
    selected=[]
    for name in LEVEL_GROUPS:
        _,y=DIFFICULTIES[name]
        patch=a[y-8:y+8,173:186].astype(int)
        yellow=(patch[:,:,0]>215)&(patch[:,:,1]>215)&(np.minimum(patch[:,:,0],patch[:,:,1])-patch[:,:,2]>25)
        if yellow.mean()>0.75:
            selected.append(name)
    if len(selected)!=1:
        raise MenuNotReady("选关菜单的当前难度尚未显示清楚。")
    return selected[0]


def endless_menu_visible(frame):
    if not level_menu_visible(frame):
        return False
    a=np.asarray(frame)
    _,y=DIFFICULTIES["无限"]
    patch=a[y-8:y+8,173:186].astype(int)
    yellow=(patch[:,:,0]>215)&(patch[:,:,1]>215)&(np.minimum(patch[:,:,0],patch[:,:,1])-patch[:,:,2]>25)
    if yellow.mean()<=0.75:
        return False
    for x,y in ENDLESS_CHOICES.values():
        for dx,dy in ((-28,-30),(28,-30),(-28,30),(28,30)):
            color=np.median(a[y+dy-2:y+dy+3,x+dx-2:x+dx+3].reshape(-1,3),axis=0)
            if color.min()<=224 or np.ptp(color)>=25:
                return False
    return True


def inspect_level_menu(frame, expected_group=None):
    """Inspect every known slot, including lone white tiles in sparse rows.

    A completed tile must be darker than its surrounding background. Missing
    or fading tiles are unknown, never interpreted as a completed group.
    """
    if not level_menu_visible(frame):
        raise MenuNotReady("选关菜单尚未加载完成。")
    group=selected_level_group(frame)
    if expected_group is not None and group!=expected_group:
        raise MenuNotReady(f"当前仍显示{group}，正在等待{expected_group}。")
    first,count,rows=LEVEL_LAYOUTS[group]
    a=np.asarray(frame)
    remaining=[];centers=[];completed=0;unknown=[]
    offsets=((-25,-30),(0,-30),(25,-30),(-25,30),(0,30),(25,30))
    for i in range(count):
        row,col=divmod(i,10);x,y=690+123*col,rows[row]
        samples=np.stack([np.median(a[y+dy-2:y+dy+3,x+dx-2:x+dx+3].reshape(-1,3),axis=0)
                          for dx,dy in offsets])
        color=np.median(samples,axis=0)
        outside=np.concatenate((a[y-2:y+3,x-62:x-57].reshape(-1,3),a[y-2:y+3,x+58:x+63].reshape(-1,3)))
        background=np.median(outside,axis=0)
        white=(samples.min(axis=1)>224)&(np.ptp(samples,axis=1)<25)
        if white.sum()>=4:
            remaining.append(first+i);centers.append((float(x),float(y)))
        elif background.mean()-color.mean()>10 and np.abs(background-color).mean()>12:
            completed+=1
        else:
            unknown.append(first+i)
    if unknown:
        raise MenuNotReady(f"关卡按钮仍在加载，尚未确认 {len(unknown)} 个格子。")
    return LevelMenu(group,tuple(remaining),tuple(centers),completed,count)


def available_levels(frame):
    return list(inspect_level_menu(frame).centers)


def glyph_mask(image):
    a = np.asarray(image.convert("RGB")).astype(int)
    dark = (a.mean(axis=2) < 180) & (np.ptp(a, axis=2) < 45)
    orange = orange_mask(a)
    if np.mean(orange) > 0.20:
        # Hovered values are WHITE on an ORANGE CIRCLE, whose outer corners are
        # still the cell background. The border color alone cannot identify it.
        # Keep enclosed white glyphs, excluding any white background connected
        # to the crop boundary. This also handles the inline nine-number menu.
        white = (a.min(axis=2) > 210) & (np.ptp(a, axis=2) < 35)
        enclosed = white.copy()
        h, w = white.shape
        stack = [(int(y), int(x)) for y, x in zip(*np.nonzero(white))
                 if y == 0 or y == h - 1 or x == 0 or x == w - 1]
        for y, x in stack:
            enclosed[y, x] = False
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and enclosed[yy, xx]:
                        enclosed[yy, xx] = False
                        stack.append((yy, xx))
        dark |= enclosed
    else:
        # Values entered by the player are orange; givens are gray.
        dark |= orange
    return dark


def components(mask):
    remaining = mask.copy()
    out = []
    h, w = remaining.shape
    for y, x in zip(*np.nonzero(mask)):
        if not remaining[y, x]:
            continue
        stack, pixels = [(int(y), int(x))], []
        remaining[y, x] = False
        while stack:
            py, px = stack.pop()
            pixels.append((py, px))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = py + dy, px + dx
                    if 0 <= ny < h and 0 <= nx < w and remaining[ny, nx]:
                        remaining[ny, nx] = False
                        stack.append((ny, nx))
        yy, xx = zip(*pixels)
        out.append((min(xx), min(yy), max(xx) + 1, max(yy) + 1, len(pixels)))
    return out


def normalize(mask):
    yy, xx = np.nonzero(mask)
    if not len(xx):
        raise RecognitionError("数字模板为空。")
    tight = mask[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
    h, w = tight.shape
    scale = min(28 / w, 36 / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = Image.fromarray(tight.astype(np.uint8) * 255).resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = np.zeros((40, 32), dtype=np.float32)
    x, y = (32 - nw) // 2, (40 - nh) // 2
    canvas[y:y + nh, x:x + nw] = np.asarray(resized) / 255
    return canvas


def similarity(a, b):
    padded = np.pad(b, ((2, 2), (2, 2)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, (40, 32))
    numerator = 2 * np.minimum(a, windows).sum(axis=(-2, -1))
    denominator = np.maximum(1, a.sum() + windows.sum(axis=(-2, -1)))
    return float((numerator / denominator).max())


class Recognizer:
    def __init__(self, assets=None):
        source = Path(assets) if assets else resource_dir() / "digits.npz"
        with np.load(source, allow_pickle=False) as data:
            self.templates = {n: data[str(n)].copy() for n in range(1, 10)}
        # The game's player-entered values use a heavier face than the givens
        # and right-hand counters. Match both weights for all nine digits.
        self.template_variants = {}
        for n, template in self.templates.items():
            binary = Image.fromarray((template > 0.45).astype(np.uint8) * 255)
            variants = [template]
            for width in (3, 5):
                heavier = binary.filter(ImageFilter.MaxFilter(width))
                variants.append(normalize(np.asarray(heavier) > 127))
            self.template_variants[n] = variants

        # Prefer actual player glyphs over approximations of the counter font.
        players = source.parent / "player-digits.npz"
        if players.exists():
            self.load_player_samples(players)

    def load_player_samples(self, path):
        with np.load(path, allow_pickle=False) as data:
            for n in range(1, 10):
                if str(n) not in data:
                    continue
                samples = data[str(n)]
                if samples.ndim == 2:
                    samples = samples[None, :, :]
                if samples.ndim != 3 or samples.shape[1:] != (40, 32) or not np.isfinite(samples).all():
                    raise RecognitionError("保存的数字字形数据格式错误。")
                self.template_variants[n].extend(sample.copy() for sample in samples[:8])

    def cell_sample(self, frame, r, c):
        """Return a full-sized glyph, or None for an empty cell / chooser."""
        x0, y0, x1, y1 = cell_box(r, c)
        crop = frame.crop((round(x0 + 13), round(y0 + 10), round(x1 - 13), round(y1 - 10)))
        mask = glyph_mask(crop)
        blobs = [b for b in components(mask) if b[4] >= 6]
        big = [b for b in blobs if b[3] - b[1] >= 22]
        if not big:
            # Small pencil marks and the inline chooser are not givens.
            if blobs and max(b[3] - b[1] for b in blobs) >= 20 and len(blobs) <= 2:
                raise RecognitionError(f"第 {r + 1} 行第 {c + 1} 列的数字尺寸异常。")
            return None
        if len(big) != 1:
            raise RecognitionError(f"第 {r + 1} 行第 {c + 1} 列存在多个大字符。")
        x, y, xx, yy, _ = big[0]
        return normalize(mask[y:yy, x:xx])

    def candidates(self, sample):
        return sorted(((max(similarity(sample, template) for template in variants), n)
                       for n, variants in self.template_variants.items()), reverse=True)

    def read_cell(self, frame, r, c):
        sample = self.cell_sample(frame, r, c)
        if sample is None:
            return 0, 1.0
        ranked = self.candidates(sample)
        (score, number), (second, second_number) = ranked[:2]
        if score < 0.70 or score - second < 0.055:
            raise RecognitionError(f"第 {r + 1} 行第 {c + 1} 列识别不确定"
                                   f"（候选 {number}：{score:.0%}，{second_number}：{second:.0%}），已停止点击。")
        return number, score

    def read(self, image):
        frame = canonical(image)
        if not board_visible(frame):
            raise RecognitionError("未识别到完整棋盘；请关闭弹窗并进入数独关卡。")
        board, scores = [], []
        for r in range(9):
            for c in range(9):
                n, confidence = self.read_cell(frame, r, c)
                board.append(n)
                scores.append(confidence)
        if sum(n > 0 for n in board) < 17:
            raise RecognitionError("识别出的有效数字不足 17 个，请检查画面或截图。")
        return Reading(board, scores, frame)
