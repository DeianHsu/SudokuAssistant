"""Generate game-font templates from independently labelled reference screenshots."""
from pathlib import Path
import numpy as np
from PIL import Image
from .vision import canonical, badge_mask, badge_glyphs, normalize, components, glyph_mask, STEP

# (row, column, orange total), with row and column numbered from one.
LABELS = {
    "board1": [(1,1,6),(1,2,15),(1,4,14),(1,7,7),(1,9,14),
               (2,3,8),(2,5,11),(2,6,13),(2,8,17),
               (3,1,17),(3,3,11),(3,6,16),
               (4,2,26),(4,3,10),(4,6,9),(5,1,8),(5,5,14),(5,6,28),(5,9,4),
               (6,2,18),(6,7,16),(7,1,29),(7,3,13),(7,5,5),(7,6,18),(8,8,24),(9,3,34)],
    "board2": [(1,1,20),(1,4,21),(1,5,14),(1,7,30),(2,2,20),(2,3,11),(2,9,13),
               (3,1,4),(3,4,8),(3,6,22),(4,3,13),(4,8,10),
               (5,1,12),(5,3,30),(6,1,17),(6,3,16),(6,5,11),(6,8,17),(6,9,15),
               (7,5,9),(7,7,10),(8,4,12),(8,5,23),(9,1,28),(9,7,19)],
    "hard-62": [(1,1,18),(1,2,21),(1,5,14),(1,8,19),(2,3,12),(2,5,14),
                (3,3,20),(3,5,18),(3,7,13),(3,8,22),(5,1,8),(5,3,36),(5,8,17),
                (6,1,23),(6,3,13),(6,6,15),(7,4,15),(7,8,22),
                (8,1,25),(8,3,22),(8,6,12),(9,3,10),(9,7,16)],
    "easy-13": [(1,1,10),(1,3,7),(1,4,13),(1,5,20),(1,7,10),
                (2,1,13),(2,6,3),(2,7,17),(2,8,16),(2,9,10),
                (3,1,14),(3,3,10),(4,1,17),(4,2,12),(4,4,16),(4,8,7),
                (5,4,18),(5,7,18),(5,9,12),(6,1,9),(6,3,24),(6,4,11),
                (7,1,11),(7,2,11),(7,4,10),(7,5,10),(7,8,8),
                (8,6,7),(8,7,13),(8,8,5),(9,1,6),(9,4,20),(9,8,17)]
}


def generate_menu_templates(root):
    with Image.open(root/"tests"/"fixtures"/"easy.png") as image:
        menu=canonical(image)
    data={}
    for n in range(1,11):
        x,y=690+123*(n-1),423
        mask=glyph_mask(menu.crop((x-44,y-31,x+44,y+31)))
        blobs=sorted((b for b in components(mask) if b[3]-b[1]>=24),key=lambda b:b[0])
        if len(blobs)!=len(str(n)):
            raise ValueError(f"关卡 {n} 的数字模板不完整。")
        x0,y0,x1,y1,_=blobs[-1]
        data[str(n if n<10 else 0)]=normalize(mask[y0:y1,x0:x1])
    np.savez_compressed(root/"assets"/"menu-digits.npz",**data)


def generate_sum_templates(root):
    sums={str(n):[] for n in range(10)}
    for name,labels in LABELS.items():
        with Image.open(root/"tests"/"fixtures"/(name+".png")) as image:
            # Small badge glyphs rasterize differently at each resolution.
            # Preserve real variants instead of lowering recognition confidence.
            for size in ((2560,1440),(2048,1152),(1920,1080)):
                frame=canonical(image.resize(size,Image.Resampling.LANCZOS))
                for r,c,total in labels:
                    mask=badge_mask(frame,r-1,c-1)
                    if mask is None:
                        raise ValueError(f"{name} {size} {r},{c} 总和角标缺失")
                    glyphs=badge_glyphs(mask)
                    if len(glyphs)!=len(str(total)):
                        raise ValueError(f"{name} {size} {r},{c}: 字符数量错误 {glyphs}")
                    for n,(x0,y0,x1,y1,_) in zip(str(total),glyphs):
                        sums[n].append(normalize(mask[y0:y1,x0:x1]))
    if any(not values for values in sums.values()):
        raise ValueError("缺失总和角标字体模板")
    unique={n:np.stack(list({sample.tobytes():sample for sample in values}.values()))
            for n,values in sums.items()}
    np.savez_compressed(root/"assets"/"sums.npz",**unique)


def main():
    root=Path(__file__).resolve().parent
    generate_sum_templates(root)
    digits={str(n):[] for n in range(1,10)}
    for name in ("board1","board2"):
        frame=canonical(Image.open(root/"tests"/"fixtures"/(name+".png")))
        for n in range(1,10):
            y=236+(n-1)*STEP
            a=np.asarray(frame.crop((1502,round(y-23),1538,round(y+23)))).astype(np.int16)
            mask=(a.mean(axis=2)<180)&(np.ptp(a,axis=2)<35)
            blobs=[b for b in components(mask) if b[3]-b[1]>=20]
            if len(blobs)!=1:
                raise ValueError(f"数字 {n} 模板不完整")
            x0,y0,x1,y1,_=blobs[0]
            digits[str(n)].append(normalize(mask[y0:y1,x0:x1]))
    if any(not values for values in digits.values()):
        raise ValueError("缺失棋盘数字字体模板")
    np.savez_compressed(root/"assets"/"digits.npz",**{n:np.stack(v) for n,v in digits.items()})
    generate_menu_templates(root)
    print("已生成 0–9 总和角标模板、1–9 棋盘数字模板和 0–9 关卡数字模板。")


if __name__=="__main__":
    main()
