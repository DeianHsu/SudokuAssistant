"""Screenshot recognition for Killer Sudoku, including inset dashed cage walls."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageFilter
from .solver import Cage, validate_cages, PuzzleError

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
    cages: tuple
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
            p = a[round(y+63):round(y+71), round(x+63):round(x+71)]
            # Highlighted rows/columns are pastel; accept any hue.
            hits += int(np.mean(p.mean(axis=2) > 195) > .8)
    # A puzzle must have all nine circular number counters, not just a white square.
    counters = 0
    for n in range(9):
        y=round(236+n*STEP)
        counters += int(np.mean(a[y-5:y+5,1496:1504].min(axis=2)>220)>.8)
    return hits >= 76 and counters == 9


def level_menu_visible(frame):
    a=np.asarray(frame,dtype=np.int16)
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
    """Confirm a completed slot by its exact light level number on the background."""
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
    orange=(a[:,:,0]-a[:,:,1]>20)&(a[:,:,1]-a[:,:,2]>20)
    dark=(a.mean(axis=2)<180)&(np.ptp(a,axis=2)<45)
    dark_background=np.mean(dark)>.20
    if np.mean(np.ptp(a,axis=2)>55)>.20 or dark_background:
        # Both the nine-number popup and hovered values have white glyphs on color.
        white=(a.min(axis=2)>210)&(np.ptp(a,axis=2)<35)
        enclosed=white.copy()
        stack=[(int(y),int(x)) for y,x in zip(*np.nonzero(white))
               if y==0 or y==h-1 or x==0 or x==w-1]
        for y,x in stack:
            enclosed[y,x]=False
        while stack:
            y,x=stack.pop()
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    yy,xx=y+dy,x+dx
                    if 0<=yy<h and 0<=xx<w and enclosed[yy,xx]:
                        enclosed[yy,xx]=False
                        stack.append((yy,xx))
        return enclosed if dark_background else dark|enclosed
    return dark|orange|(mask&(np.ptp(a,axis=2)>45))


def chooser_open(frame,r,c):
    x,y,xx,yy=cell_box(r,c)
    crop=frame.crop((round(x+7),round(y+7),round(xx-7),round(yy-7)))
    a=np.asarray(crop).astype(np.int16)
    ring=np.concatenate((a[:3].reshape(-1,3),a[-3:].reshape(-1,3),
                         a[:,:3].reshape(-1,3),a[:,-3:].reshape(-1,3)))
    bg=np.median(ring,axis=0)
    mask=(np.max(np.abs(a-bg),axis=2)>32)&(np.abs(a.mean(axis=2)-bg.mean())>18)
    blobs=[b for b in components(mask) if b[4]>=5 and 10<=b[3]-b[1]<=23]
    occupied=set()
    for x0,y0,x1,y1,_ in blobs:
        cx,cy=(x0+x1)/2+7,(y0+y1)/2+7
        dc,dr=min(2,int(cx/(STEP/3))),min(2,int(cy/(STEP/3)))
        occupied.add((dr,dc))
    return len(occupied)>=7


def badge_geometry(r,c):
    """Fixed game layout: corner badges are offset around the wide 3x3 gutters."""
    x,y,_,_=cell_box(r,c)
    dx=15 if c>0 and c%3==0 else 10.5
    dy=15 if r>0 and r%3==0 else 10.5
    return round(x+dx),round(y+dy),16


def badge_mask(frame,r,c):
    x,y,_,_=cell_box(r,c)
    left,top=round(x-10),round(y-10)
    crop=frame.crop((left,top,round(x+35),round(y+35)))
    a=np.asarray(crop).astype(np.int16)
    cx,cy,_=badge_geometry(r,c)
    yy,xx=np.indices(a.shape[:2])
    distance=(xx+left-cx)**2+(yy+top-cy)**2
    # Badge hues change with the theme. Detect the filled disk at its known
    # corner position, then read only its white numeral inside the disk.
    core=distance<14**2
    ink=(np.ptp(a,axis=2)>45)|(a.mean(axis=2)<190)
    if np.mean(ink[core])<.55:
        return None
    color=np.median(a[core&ink],axis=0)
    ring=(distance>18**2)&(distance<22**2)&(xx+left>x+5)&(yy+top>y+5)
    if np.mean(np.max(np.abs(a[ring]-color),axis=1)<25)>.18:
        return None
    disk=distance<13.5**2
    channels=255-color>40
    if not channels.any():
        raise RecognitionError("总和角标与文字的对比度不足。")
    # Recover the glyph's white-opacity rather than applying an absolute RGB
    # threshold: the same anti-aliased stroke must survive every badge hue.
    opacity=((a[:,:,channels]-color[channels])/(255-color[channels])).mean(axis=2)
    return disk&(opacity>.75)


def badge_glyphs(mask):
    return sorted((b for b in components(mask) if b[4]>=8 and b[3]-b[1]>=9),key=lambda b:b[0])


def empty_hover_marker(crop,blob):
    """A solid round hover dot contains no numeral; never send it to digit OCR."""
    x0,y0,x1,y1,count=blob
    w,h=x1-x0,y1-y0
    if not 8<=h<=min(72,crop.height) or abs(w-h)>3 or not .72<=count/(w*h)<=.88:
        return False
    a=np.asarray(crop,dtype=np.int16)[y0:y1,x0:x1]
    yy,xx=np.indices((h,w))
    interior=((xx-(w-1)/2)/(w*.40))**2+((yy-(h-1)/2)/(h*.40))**2<1
    pixels=a[interior]
    color=np.median(pixels,axis=0)
    # White or dark numeral strokes break the uniform interior of a real value.
    return np.count_nonzero(np.max(np.abs(pixels-color),axis=1)>28)<=2


def cage_edge_score(frame,r,c,vertical,chooser=None):
    """Sample both inset lanes; ignore the ordinary grid and the open number popup."""
    a=np.asarray(frame,dtype=np.int16)
    scores=[]
    for side in (-1,1):
        index=r*9+c+(1 if vertical and side==1 else 9 if not vertical and side==1 else 0)
        if index==chooser:
            continue
        x,y,_,_=cell_box(r,c)
        if vertical:
            axis=x+STEP
            inset=8 if c%3==2 else 6
            start,end=round(y+.37*STEP),round(y+.82*STEP)
            lane=a[start:end,round(axis+side*inset)-2:round(axis+side*inset)+3]
            ref=a[start:end,round(axis+side*17)-2:round(axis+side*17)+3]
            baseline=np.median(ref,axis=1)[:,None,:]
        else:
            axis=y+STEP
            inset=8 if r%3==2 else 6
            start,end=round(x+.37*STEP),round(x+.82*STEP)
            lane=a[round(axis+side*inset)-2:round(axis+side*inset)+3,start:end]
            ref=a[round(axis+side*17)-2:round(axis+side*17)+3,start:end]
            baseline=np.median(ref,axis=0)[None,:,:]
        contrast=np.max(baseline-lane,axis=2)
        marked=(contrast>28)&(lane.mean(axis=2)<baseline.mean(axis=2)-8)
        # Dashes occupy roughly half of this lane; max across lane width preserves them.
        profile=marked.any(axis=1 if vertical else 0)
        scores.append(float(profile.mean()))
    return max(scores,default=0)


def cage_partition(frame,chooser,threshold):
    frame=np.asarray(frame,dtype=np.int16)
    neighbors=[[] for _ in range(81)]
    for r in range(9):
        for c in range(9):
            i=r*9+c
            for j,vertical in ((i+1,True),(i+9,False)):
                if vertical and c==8 or not vertical and r==8:
                    continue
                if cage_edge_score(frame,r,c,vertical,chooser)<threshold:
                    neighbors[i].append(j)
                    neighbors[j].append(i)
    groups=[]
    seen=set()
    for start in range(81):
        if start in seen:
            continue
        group,pending=[],[start]
        seen.add(start)
        while pending:
            i=pending.pop()
            group.append(i)
            for j in neighbors[i]:
                if j not in seen:
                    seen.add(j)
                    pending.append(j)
        groups.append(tuple(sorted(group)))
    return groups


class Recognizer:
    def __init__(self,assets=None):
        source=Path(assets) if assets else resource_dir()/"digits.npz"
        with np.load(source,allow_pickle=False) as data:
            self.variants={n:list(data[str(n)]) if data[str(n)].ndim==3 else [data[str(n)]]
                           for n in range(1,10)}
        for n,variants in self.variants.items():
            binary=Image.fromarray((variants[0]>.45).astype(np.uint8)*255)
            variants.extend(normalize(np.asarray(binary.filter(ImageFilter.MaxFilter(k)))>127) for k in (3,5))
        for filename in ("player-digits.npz","given-digits.npz"):
            extra=source.parent/filename
            if extra.exists():
                with np.load(extra,allow_pickle=False) as data:
                    for n in range(1,10):
                        if str(n) in data:
                            sample=data[str(n)]
                            self.variants[n].extend(list(sample) if sample.ndim==3 else [sample])
        with np.load(source.parent/"sums.npz",allow_pickle=False) as data:
            self.sum_variants={n:list(data[str(n)]) for n in range(10)}

    def read_total(self,frame,r,c):
        mask=badge_mask(frame,r,c)
        if mask is None:
            return None
        glyphs=badge_glyphs(mask)
        if not 1<=len(glyphs)<=2:
            raise RecognitionError(f"第 {r+1} 行第 {c+1} 列的总和角标未显示清楚。")
        text=""
        for x0,y0,x1,y1,_ in glyphs:
            sample=normalize(mask[y0:y1,x0:x1])
            ranked=sorted((max(similarity(sample,t) for t in variants),n)
                          for n,variants in self.sum_variants.items())
            (score,n),(second,other)=ranked[-1],ranked[-2]
            if score<.72 or score-second<.035:
                raise RecognitionError(f"第 {r+1} 行第 {c+1} 列总和识别不确定（{n}：{score:.0%}，{other}：{second:.0%}）。")
            text+=str(n)
        if text.startswith("0") or not 1<=int(text)<=45:
            raise RecognitionError("总和角标不在 1–45 范围内。")
        return int(text)

    def read_cages(self,frame,chooser=None):
        totals={i:n for i in range(81) if (n:=self.read_total(frame,*divmod(i,9))) is not None}
        if not totals:
            raise RecognitionError("没有识别到总和角标，请确认打开的是杀手数独。")
        hidden_total=405-sum(totals.values())
        candidates={}
        for threshold in (.12,.18,.24,.30):
            groups=cage_partition(frame,chooser,threshold)
            cages=[]
            for cells in groups:
                anchors=[i for i in cells if i in totals]
                # An open chooser can hide its own cage's top-left badge.
                # Only that cage may recover its total from the board sum.
                if not anchors and chooser==cells[0] and 1<=hidden_total<=45:
                    cages.append(Cage(hidden_total,cells))
                    continue
                if len(anchors)!=1:
                    break
                cages.append(Cage(totals[anchors[0]],cells))
            else:
                try:
                    validate_cages(cages)
                except PuzzleError:
                    continue
                candidates[tuple(cages)]=tuple(cages)
        if len(candidates)!=1:
            raise RecognitionError("虚线边界无法唯一还原为每框一个总和、完整覆盖 81 格且总和为 405 的分区，已停止点击。")
        return next(iter(candidates.values()))

    def read_cell(self,frame,r,c):
        if chooser_open(frame,r,c):
            return 0,1.0,True
        x,y,xx,yy=cell_box(r,c)
        left,top=round(x+14),round(y+10)
        crop=frame.crop((left,top,round(xx-14),round(yy-10)))
        mask=glyph_mask(crop)
        if badge_mask(frame,r,c) is not None:
            cx,cy,radius=badge_geometry(r,c)
            yy0,xx0=np.indices(mask.shape)
            mask &= (xx0+left-cx)**2+(yy0+top-cy)**2>(radius+1.5)**2
        blobs=[b for b in components(mask) if b[4]>=6 and not empty_hover_marker(crop,b)]
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
            raise RecognitionError("未识别到完整杀手数独棋盘；请关闭弹窗并进入关卡。")
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
        chooser=opened[0] if opened else None
        cages=self.read_cages(frame,chooser)
        return Reading(board,cages,scores,frame,chooser)
