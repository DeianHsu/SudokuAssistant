"""Screenshot recognition for Sudoku Jigsaw, with color-independent line widths."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageFilter
from .solver import validate_regions, PuzzleError

SIZE = (2048, 1152)
BOARD = (640, 192, 1408, 960)
STEP = 768 / 9
DIFFICULTIES = {"简单": (277, 470), "中等": (277, 576), "困难": (277, 683)}
LEVEL_GROUPS = tuple(DIFFICULTIES)
LEVEL_LAYOUTS = {"简单": (1,30,(423,576,730)), "中等": (31,30,(423,576,730)),
                 "困难": (61,40,(346,499,653,807))}


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
    regions: list
    scores: list
    frame: Image.Image
    chooser: int | None = None


def resource_dir():
    return Path(__file__).resolve().parent / "assets"


def canonical(image):
    w, h = image.size
    if not w or not h or abs(w/h - 16/9) > 0.025:
        raise RecognitionError("需要完整的 16:9 游戏画面，建议使用 2560×1440 全屏。")
    return image.convert("RGB").resize(SIZE, Image.Resampling.LANCZOS)


def cell_box(r, c):
    return 640+c*STEP, 192+r*STEP, 640+(c+1)*STEP, 192+(r+1)*STEP


def cell_center(r, c):
    return 640+(c+.5)*STEP, 192+(r+.5)*STEP


def digit_point(r, c, n):
    if not 1 <= n <= 9:
        raise ValueError("数字必须在 1–9 之间。")
    dr, dc = divmod(n-1, 3)
    x, y, _, _ = cell_box(r, c)
    return x+(dc+.5)*STEP/3, y+(dr+.5)*STEP/3


def board_visible(frame):
    a = np.asarray(frame).astype(np.int16)
    hits = 0
    for r in range(9):
        for c in range(9):
            x,y,_,_ = cell_box(r,c)
            p = a[round(y+11):round(y+19), round(x+11):round(x+19)]
            # Highlighted rows/columns are pastel; accept any hue.
            hits += int(np.mean(p.mean(axis=2) > 195) > .8)
    # A puzzle must have all nine circular number counters, not just a white square.
    counters = 0
    for n in range(9):
        y=round(236+n*STEP)
        counters += int(np.mean(a[y-5:y+5,1496:1504].min(axis=2)>220)>.8)
    return hits >= 76 and counters == 9


def level_menu_visible(frame):
    a=np.asarray(frame).astype(np.int16)
    for x,y in DIFFICULTIES.values():
        patch=a[y-10:y+10,x-95:x-80]
        if np.mean((patch[:,:,0]>210)&(patch[:,:,1]>210)&(patch[:,:,2]>125)) < .85:
            return False
    return not board_visible(frame)


def selected_level_group(frame):
    a=np.asarray(frame).astype(np.int16)
    selected=[]
    for name,(_,y) in DIFFICULTIES.items():
        p=a[y-8:y+8,173:186]
        if np.mean((p[:,:,0]>215)&(p[:,:,1]>215)&(np.minimum(p[:,:,0],p[:,:,1])-p[:,:,2]>25))>.75:
            selected.append(name)
    if len(selected)!=1:
        raise MenuNotReady("当前难度尚未显示清楚。")
    return selected[0]


def inspect_level_menu(frame, expected_group=None):
    if not level_menu_visible(frame):
        raise MenuNotReady("选关菜单尚未加载完成。")
    group=selected_level_group(frame)
    if expected_group and group!=expected_group:
        raise MenuNotReady(f"当前显示{group}，正在等待{expected_group}。")
    first,count,rows=LEVEL_LAYOUTS[group]
    a=np.asarray(frame)
    remaining,centers,unknown=[],[],[]
    completed=0
    for i in range(count):
        row,col=divmod(i,10)
        x,y=690+123*col,rows[row]
        samples=np.stack([np.median(a[y+dy-2:y+dy+3,x+dx-2:x+dx+3].reshape(-1,3),axis=0)
                          for dx,dy in ((-25,-30),(0,-30),(25,-30),(-25,30),(0,30),(25,30))])
        color=np.median(samples,axis=0)
        outside=np.concatenate((a[y-2:y+3,x-62:x-57].reshape(-1,3),
                                a[y-2:y+3,x+58:x+63].reshape(-1,3)))
        background=np.median(outside,axis=0)
        white=(samples.min(axis=1)>224)&(np.ptp(samples,axis=1)<25)
        if white.sum()>=4:
            remaining.append(first+i)
            centers.append((float(x),float(y)))
        elif (background.mean()-color.mean()>10 and np.abs(background-color).mean()>12) or (
                color.mean()<160 and completed_label_matches(frame,x,y,first+i)):
            completed+=1
        else:
            unknown.append(first+i)
    if unknown:
        raise MenuNotReady(f"尚未确认 {len(unknown)} 个关卡按钮，等待完整菜单。")
    return LevelMenu(group,tuple(remaining),tuple(centers),completed,count)


def components(mask):
    remaining=mask.copy()
    out=[]
    h,w=remaining.shape
    for y,x in zip(*np.nonzero(mask)):
        if not remaining[y,x]:
            continue
        stack,pixels=[(int(y),int(x))],[]
        remaining[y,x]=False
        while stack:
            py,px=stack.pop()
            pixels.append((py,px))
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    yy,xx=py+dy,px+dx
                    if 0<=yy<h and 0<=xx<w and remaining[yy,xx]:
                        remaining[yy,xx]=False
                        stack.append((yy,xx))
        yy,xx=zip(*pixels)
        out.append((min(xx),min(yy),max(xx)+1,max(yy)+1,len(pixels)))
    return out


def normalize(mask):
    yy,xx=np.nonzero(mask)
    if not len(xx):
        raise RecognitionError("数字模板为空。")
    tight=mask[yy.min():yy.max()+1,xx.min():xx.max()+1]
    h,w=tight.shape
    scale=min(28/w,36/h)
    nw,nh=max(1,round(w*scale)),max(1,round(h*scale))
    resized=Image.fromarray(tight.astype(np.uint8)*255).resize((nw,nh),Image.Resampling.LANCZOS)
    canvas=np.zeros((40,32),dtype=np.float32)
    x,y=(32-nw)//2,(40-nh)//2
    canvas[y:y+nh,x:x+nw]=np.asarray(resized)/255
    return canvas


def similarity(a,b):
    windows=np.lib.stride_tricks.sliding_window_view(np.pad(b,((2,2),(2,2))),(40,32))
    return float((2*np.minimum(a,windows).sum(axis=(-2,-1))/
                  np.maximum(1,a.sum()+windows.sum(axis=(-2,-1)))).max())


@lru_cache(maxsize=1)
def menu_templates():
    with np.load(resource_dir()/"menu-digits.npz",allow_pickle=False) as data:
        return {str(n):data[str(n)].copy() for n in range(10)}


def completed_label_matches(frame,x,y,level):
    """Confirm a faint completed tile by its exact white level number."""
    a=np.asarray(frame.crop((x-44,y-31,x+44,y+31))).astype(np.int16)
    bg=np.median(np.concatenate((a[:4].reshape(-1,3),a[-4:].reshape(-1,3))),axis=0)
    bright=(a.min(axis=2)>180)&(np.ptp(a,axis=2)<45)&(a.mean(axis=2)-bg.mean()>75)
    blobs=sorted((b for b in components(bright) if b[4]>=25 and 24<=b[3]-b[1]<=48),key=lambda b:b[0])
    label=str(level)
    if len(blobs)!=len(label):
        return False
    templates=menu_templates()
    for (x0,y0,x1,y1,_),digit in zip(blobs,label):
        sample=normalize(bright[y0:y1,x0:x1])
        ranked=sorted((similarity(sample,template),n) for n,template in templates.items())
        (score,number),(second,_)=ranked[-1],ranked[-2]
        if number!=digit or score<.75 or score-second<.045:
            return False
    return True


def glyph_mask(image):
    a=np.asarray(image.convert("RGB")).astype(np.int16)
    h,w=a.shape[:2]
    # Fit the crop background from its outside ring, regardless of theme hue.
    ring=np.concatenate((a[:3].reshape(-1,3),a[-3:].reshape(-1,3),
                         a[:,:3].reshape(-1,3),a[:,-3:].reshape(-1,3)))
    bg=np.median(ring,axis=0)
    contrast=np.max(np.abs(a-bg),axis=2)
    mask=(contrast>32)&(np.abs(a.mean(axis=2)-bg.mean())>18)
    if mask.sum()<350:
        return mask
    # A hovered cell has a solid circular badge. Its fill is not a numeral;
    # any enclosed white glyph remains a real given/player-entered value.
    circles=[]
    for x0,y0,x1,y1,pixels in components(mask):
        width,height=x1-x0,y1-y0
        if (min(width,height)>=24 and min(width,height)/max(width,height)>.9
                and .55<pixels/(width*height)<.88):
            circles.append((x0,y0,x1,y1))
    if not circles:
        return mask
    white=(a.min(axis=2)>205)&(np.ptp(a,axis=2)<45)
    enclosed=np.zeros_like(mask)
    for x0,y0,x1,y1,_ in components(white):
        if x0>0 and y0>0 and x1<w and y1<h:
            enclosed[y0:y1,x0:x1]|=white[y0:y1,x0:x1]
    for x0,y0,x1,y1 in circles:
        mask[y0:y1,x0:x1]=enclosed[y0:y1,x0:x1]
    return mask


def chooser_open(frame,r,c):
    x,y,xx,yy=cell_box(r,c)
    crop=frame.crop((round(x+7),round(y+7),round(xx-7),round(yy-7)))
    blobs=[b for b in components(glyph_mask(crop)) if b[4]>=5 and 10<=b[3]-b[1]<=23]
    occupied=set()
    for x0,y0,x1,y1,_ in blobs:
        cx,cy=(x0+x1)/2+7,(y0+y1)/2+7
        dc,dr=min(2,int(cx/(STEP/3))),min(2,int(cy/(STEP/3)))
        occupied.add((dr,dc))
    return len(occupied)>=7


def edge_width(frame,r,c,vertical):
    a=np.asarray(frame)
    if vertical:
        axis=round(640+(c+1)*STEP)
        start,end=round(192+(r+.25)*STEP),round(192+(r+.75)*STEP)
        profile=np.median(a[start:end,axis-13:axis+14],axis=0)
    else:
        axis=round(192+(r+1)*STEP)
        start,end=round(640+(c+.25)*STEP),round(640+(c+.75)*STEP)
        profile=np.median(a[axis-13:axis+14,start:end],axis=1)
    left,right=np.median(profile[:4],axis=0),np.median(profile[-4:],axis=0)
    baseline=np.vstack((np.tile(left,(13,1)),np.tile(right,(14,1))))
    difference=np.max(np.abs(profile-baseline),axis=1)
    # Thin ordinary grid lines and wide palace boundaries have different widths.
    marked=difference>10
    spans=[]
    start=None
    for i,flag in enumerate(np.r_[marked,False]):
        if flag and start is None:
            start=i
        elif not flag and start is not None:
            if start<=15 and i>=11:
                spans.append(i-start)
            start=None
    return max(spans,default=0)


def read_regions(frame):
    edges=[]
    for r in range(9):
        for c in range(9):
            i=r*9+c
            if c<8:
                edges.append((i,i+1,edge_width(frame,r,c,True)))
            if r<8:
                edges.append((i,i+9,edge_width(frame,r,c,False)))
    # Try several thickness cutoffs; accept only one structurally valid partition.
    candidates={}
    for threshold in (3,4,5,6):
        neighbors=[[] for _ in range(81)]
        for a,b,width in edges:
            if width<threshold:
                neighbors[a].append(b)
                neighbors[b].append(a)
        regions=[-1]*81
        region=0
        for start in range(81):
            if regions[start]>=0:
                continue
            stack=[start]
            regions[start]=region
            while stack:
                i=stack.pop()
                for j in neighbors[i]:
                    if regions[j]<0:
                        regions[j]=region
                        stack.append(j)
            region+=1
        try:
            validate_regions(regions)
        except PuzzleError:
            continue
        candidates[tuple(regions)]=regions
    if len(candidates)!=1:
        raise RecognitionError("粗线边界无法唯一还原为 9 个各含 9 格的连通异形宫，已停止点击。")
    return next(iter(candidates.values()))


class Recognizer:
    def __init__(self,assets=None):
        source=Path(assets) if assets else resource_dir()/"digits.npz"
        with np.load(source,allow_pickle=False) as data:
            self.variants={n:list(data[str(n)]) if data[str(n)].ndim==3 else [data[str(n)]]
                           for n in range(1,10)}
        for n,variants in self.variants.items():
            binary=Image.fromarray((variants[0]>.45).astype(np.uint8)*255)
            variants.extend(normalize(np.asarray(binary.filter(ImageFilter.MaxFilter(k)))>127) for k in (3,5))
        players=source.parent/"player-digits.npz"
        if players.exists():
            with np.load(players,allow_pickle=False) as data:
                for n in range(1,10):
                    if str(n) in data:
                        sample=data[str(n)]
                        self.variants[n].extend(list(sample) if sample.ndim==3 else [sample])

    def read_cell(self,frame,r,c):
        if chooser_open(frame,r,c):
            return 0,1.0,True
        x,y,xx,yy=cell_box(r,c)
        mask=glyph_mask(frame.crop((round(x+14),round(y+10),round(xx-14),round(yy-10))))
        blobs=[b for b in components(mask) if b[4]>=6]
        big=[b for b in blobs if b[3]-b[1]>=22]
        if not big:
            if blobs and max(b[3]-b[1] for b in blobs)>=19 and len(blobs)<=2:
                raise RecognitionError(f"第 {r+1} 行第 {c+1} 列的数字尺寸异常。")
            return 0,1.0,False
        if len(big)!=1:
            raise RecognitionError(f"第 {r+1} 行第 {c+1} 列存在多个大字符。")
        x0,y0,x1,y1,_=big[0]
        sample=normalize(mask[y0:y1,x0:x1])
        ranked=sorted((max(similarity(sample,t) for t in variants),n) for n,variants in self.variants.items())
        (score,n),(second,other)=ranked[-1],ranked[-2]
        if score<.70 or score-second<.055:
            raise RecognitionError(f"第 {r+1} 行第 {c+1} 列识别不确定（{n}：{score:.0%}，{other}：{second:.0%}）。")
        return n,score,False

    def read(self,image):
        frame=canonical(image)
        if not board_visible(frame):
            raise RecognitionError("未识别到完整异形棋盘；请关闭弹窗并进入关卡。")
        regions=read_regions(frame)
        board,scores,opened=[],[],[]
        for r in range(9):
            for c in range(9):
                n,score,chooser=self.read_cell(frame,r,c)
                board.append(n)
                scores.append(score)
                if chooser:
                    opened.append(r*9+c)
        if len(opened)>1:
            raise RecognitionError("识别出多个数字菜单，请关闭弹窗后重试。")
        # Jigsaw puzzles can have fewer than the classic Sudoku minimum of 17.
        return Reading(board,regions,scores,frame,opened[0] if opened else None)
