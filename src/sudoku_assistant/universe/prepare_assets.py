"""Developer utility: derive game-font templates from the supplied screenshot."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from .vision import canonical, glyph_mask, normalize, orange_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("screenshot")
    parser.add_argument("--completion")
    args = parser.parse_args()
    with Image.open(args.screenshot) as image:
        frame = canonical(image)
    data = {}
    for n in range(1, 10):
        y = 236 + (n - 1) * 85.333333
        crop = frame.crop((1502, round(y - 23), 1538, round(y + 23)))
        data[str(n)] = normalize(glyph_mask(crop))
    target = Path(__file__).resolve().parent / "assets"
    target.mkdir(exist_ok=True)
    np.savez_compressed(target / "digits.npz", **data)
    if args.completion:
        with Image.open(args.completion) as image:
            completed = canonical(image)
        np.savez_compressed(target / "completion.npz", mask=orange_mask(np.asarray(completed)[535:615, 830:1260]))
    print("已生成九个游戏字体模板")


if __name__ == "__main__":
    main()
