"""Build actual player-font samples from manually verified screenshot labels."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .vision import Recognizer, canonical


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("screenshot")
    parser.add_argument("labels", help="JSON 数组，每项为 [从 1 开始的行, 列, 数字]")
    args = parser.parse_args()
    frame = canonical(Image.open(args.screenshot))
    samples = {n: [] for n in range(1, 10)}
    recognizer = Recognizer()
    for row, col, n in json.loads(Path(args.labels).read_text(encoding="utf-8")):
        sample = recognizer.cell_sample(frame, row - 1, col - 1)
        if sample is None:
            raise ValueError(f"标签 {row}, {col} 未找到数字")
        samples[n].append(sample)
    target = Path(__file__).resolve().parent / "assets" / "player-digits.npz"
    np.savez_compressed(target, **{str(n): np.stack(values) for n, values in samples.items() if values})
    print("Actual player-font templates saved")


if __name__ == "__main__":
    main()
