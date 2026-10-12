"""Build reproducible digit templates from the supplied screenshots."""
from pathlib import Path
import numpy as np
from PIL import Image
from .vision import canonical, components, normalize, glyph_mask, STEP


def main():
    root=Path(__file__).resolve().parent
    data={str(n):[] for n in range(1,10)}
    for name in ("black","blue","teal","pink-chooser"):
        with Image.open(root/"tests"/"fixtures"/(name+".png")) as image:
            frame=canonical(image)
        for n in range(1,10):
            y=236+(n-1)*STEP
            crop=frame.crop((1502,round(y-23),1538,round(y+23)))
            a=np.asarray(crop).astype(np.int16)
            # Counters have small colored badges near their upper-right edge.
            # Extract just the neutral gray numeral, excluding those badges.
            mask=(a.mean(axis=2)<180)&(np.ptp(a,axis=2)<35)
            blobs=[b for b in components(mask) if b[3]-b[1]>=20]
            if len(blobs)!=1:
                raise ValueError(f"{name} 的数字 {n} 模板不完整。")
            x0,y0,x1,y1,_=blobs[0]
            data[str(n)].append(normalize(mask[y0:y1,x0:x1]))
    target=root/"assets"
    target.mkdir(exist_ok=True)
    np.savez_compressed(target/"digits.npz",**{n:np.stack(samples) for n,samples in data.items()})
    with Image.open(root/"tests"/"fixtures"/"easy.png") as image:
        menu=canonical(image)
    menu_data={}
    for n in range(1,11):
        x,y=690+123*(n-1),423
        mask=glyph_mask(menu.crop((x-44,y-31,x+44,y+31)))
        blobs=sorted((b for b in components(mask) if b[3]-b[1]>=24),key=lambda b:b[0])
        if len(blobs)!=(1 if n<10 else 2):
            raise ValueError(f"关卡 {n} 的数字模板不完整。")
        x0,y0,x1,y1,_=blobs[-1]
        menu_data[str(n if n<10 else 0)]=normalize(mask[y0:y1,x0:x1])
    np.savez_compressed(target/"menu-digits.npz",**menu_data)
    print("已生成九种数字的四组实际游戏字体模板。")
    print("已生成关卡数字 0–9 的实际菜单字体模板。")


if __name__=="__main__":
    main()
