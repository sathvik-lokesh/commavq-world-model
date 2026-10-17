"""Same start, four controls: one clip's dream under real/right/left/stop pose, at 0, 0.5, 1 and 2 s.

Reads the mp4s written by dreams.py (top half = real future, bottom half = dream) and writes
runs/pose/steer.png. Frame 14 is the last real context frame; 15.. are imagined at 20 fps.
"""
import argparse

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="clip0_seg2188_983.mp4")
    ap.add_argument("--out", default="runs/pose/steer.png")
    a = ap.parse_args()
    cols = {"last real frame": 14, "+0.5 s": 24, "+1 s": 34, "+2 s": 54}
    rows = ["real", "right", "left", "stop"]
    fig, ax = plt.subplots(len(rows) + 1, len(cols), figsize=(2.6 * len(cols), 1.45 * (len(rows) + 1)))
    vids = {st: iio.imread(f"runs/pose/dreams_{st}/{a.clip}") for st in rows}
    for j, (title, f) in enumerate(cols.items()):
        ax[0, j].imshow(vids["real"][f, :128])                      # top half = what really happened
        ax[0, j].set_title(title)
        for i, st in enumerate(rows, 1):
            ax[i, j].imshow(vids[st][f, 128:])                      # bottom half = dream
    for i, name in enumerate(["what happened"] + [f"dream: {st}" for st in rows]):
        ax[i, 0].set_ylabel(name)
    for x in ax.flat:
        x.set_xticks([]); x.set_yticks([])
    fig.tight_layout()
    fig.savefig(a.out, dpi=120)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
