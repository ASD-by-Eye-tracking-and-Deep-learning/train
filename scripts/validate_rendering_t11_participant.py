"""T11 — geometry validation of the rendering against the real images, using
the per-row ``Participant`` column of the public raw export (Figshare
20113592, Cilia et al. 2022; all 59 participants identifiable — the earlier
"only 3/59" note came from the Drive copy). Run from train/:

    uv run python scripts/validate_rendering_t11_participant.py

Question: is each real scanpath image drawn from its own child's gaze, in the
coordinate convention rendering.py assumes? Colour is ignored here — this
checks geometry only (where the lines are), which must hold before any
colour/kinematics comparison means anything.

For every (participant, trial) gaze segment S and every real image I:
directed chamfer d(I -> S) = mean distance from I's line pixels to the
nearest pixel of S's rasterised path. An image drawn from S's samples lies on
S's path, so d is small for the right child. Per image, each participant gets
the minimum d over their segments; the image "matches" when its own child
has the smallest value (top-1). Chance level is 1 / #participants compared.

All four variants (right/left eye x vertical flip on/off) are reported — not
only the best one — so the convention is chosen by a visible margin, not by
picking whichever number looks best.
"""
import glob
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt as edt

RAW_DIR = "../../data/mahmoud-dataset/Eye-tracking Output"
IMG_DIR = "../../data/mahmoud-dataset/Images"
OUT = "artifacts/rendering_t11/participant_geometry.md"
SCREEN_W, SCREEN_H = 1280, 1024
W, H = 320, 240  # half the 640x480 image size, for speed
FG_THRESH = 30


def real_masks() -> dict[str, tuple[int, np.ndarray]]:
    out = {}
    for path in glob.glob(os.path.join(IMG_DIR, "*", "*.png")):
        name = os.path.basename(path)
        pid = int(name.rsplit("_", 1)[1].split(".")[0])
        a = np.asarray(Image.open(path).convert("RGB").resize((W, H), Image.NEAREST))
        m = a.max(axis=2) > FG_THRESH
        if m.sum() >= 20:
            out[name] = (pid, m)
    return out


def segments(eye: str) -> list[tuple[int, str, np.ndarray, np.ndarray]]:
    xc, yc = f"Point of Regard {eye} X [px]", f"Point of Regard {eye} Y [px]"
    segs = []
    for f in glob.glob(os.path.join(RAW_DIR, "*.csv")):
        header = pd.read_csv(f, nrows=0).columns
        if xc not in header or yc not in header:  # 8.csv and 13.csv have no left-eye columns
            continue
        df = pd.read_csv(f, usecols=["Trial", "Participant", xc, yc], low_memory=False)
        # Columns mix numbers and "-" placeholders (blinks/separators), so
        # they load as object dtype.
        for c in ("Participant", xc, yc):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["pid"] = df["Participant"]
        df = df.dropna(subset=["pid", xc, yc])
        df = df[(df[xc] > 0) & (df[yc] > 0) & (df[xc] < SCREEN_W) & (df[yc] < SCREEN_H)]
        for (pid, trial), g in df.groupby(["pid", "Trial"], sort=False):
            if len(g) >= 50:
                segs.append((int(pid), f"{os.path.basename(f)}/{trial}", g[xc].to_numpy(), g[yc].to_numpy()))
    return segs


# Line pixels of the 547 real images always sit inside x in [80, 576],
# y in [58, 427] (5th/95th percentiles of per-image bounding boxes) — exactly
# Matplotlib's default axes box on a 640x480 figure (subplot left=0.125,
# right=0.9, bottom=0.11, top=0.88), with the axes hidden but not removed.
# Paths often do not fill that box, so the axis limits are fixed (the screen),
# not autoscaled. Screen coordinates therefore map into this box, not onto
# the whole image.
AX_X0, AX_X1 = 0.125 * 640, 0.9 * 640
AX_Y0, AX_Y1 = (1 - 0.88) * 480, (1 - 0.11) * 480  # pixel rows, top-down


def raster(x: np.ndarray, y: np.ndarray, flip: bool) -> np.ndarray:
    scale = W / 640
    px = (AX_X0 + x / SCREEN_W * (AX_X1 - AX_X0)) * scale
    fy = y / SCREEN_H  # screen y, 0 at top
    if flip:
        fy = 1 - fy
    py = (AX_Y0 + fy * (AX_Y1 - AX_Y0)) * scale
    im = Image.new("1", (W, H), 0)
    ImageDraw.Draw(im).line(list(zip(px.tolist(), py.tolist())), fill=1, width=1)
    return np.asarray(im, dtype=bool)


def evaluate(masks, segs, flip):
    # best[image][participant] = min directed chamfer over that child's segments
    best = defaultdict(dict)
    best_seg = {}
    names = list(masks)
    for pid, seg_id, x, y in segs:
        dt = edt(~raster(x, y, flip))
        for n in names:
            d = float(dt[masks[n][1]].mean())
            if d < best[n].get(pid, np.inf):
                best[n][pid] = d
                if pid == masks[n][0]:
                    best_seg[n] = (seg_id, d)
    top1, ranks, n_cand = 0, [], []
    for n in names:
        own = masks[n][0]
        if own not in best[n]:
            continue
        order = sorted(best[n], key=best[n].get)
        r = order.index(own) + 1
        ranks.append(r)
        n_cand.append(len(order))
        top1 += r == 1
    return {
        "images": len(ranks),
        "top1": top1 / max(len(ranks), 1),
        "median_rank": float(np.median(ranks)) if ranks else float("nan"),
        "chance_top1": float(np.mean([1 / c for c in n_cand])) if n_cand else float("nan"),
        "best_seg": best_seg,
    }


def main():
    masks = real_masks()
    print(f"{len(masks)} real images with >=20 line pixels")
    rows, results = [], {}
    for eye in ("Right", "Left"):
        segs = segments(eye)
        print(f"{eye} eye: {len(segs)} (participant, trial) segments from {len({s[0] for s in segs})} participants", flush=True)
        for flip in (False, True):
            r = evaluate(masks, segs, flip)
            results[(eye, flip)] = r
            rows.append(
                f"| {eye} | {'yes' if flip else 'no'} | {r['images']} | {r['top1']:.3f} | {r['chance_top1']:.3f} | {r['median_rank']:.1f} |"
            )
            print(rows[-1], flush=True)

    best_key = max(results, key=lambda k: results[k]["top1"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(
            "# T11 — participant-level geometry check (directed chamfer, image -> gaze path)\n\n"
            "Each real image is compared with every child's gaze segments; 'top-1' = its own child is the "
            "closest. Images whose own child has no usable segment are excluded.\n\n"
            "| Eye | Vertical flip | Images | Top-1 own-child | Chance top-1 | Median rank of own child |\n"
            "|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n\n"
            f"Best variant: eye={best_key[0]}, flip={'yes' if best_key[1] else 'no'}.\n\n"
            "## Closest own-child segment per image (best variant)\n\n"
            "| Image | Segment (file/trial) | Directed chamfer (px at 320x240) |\n|---|---|---|\n"
            + "\n".join(
                f"| {n} | {s} | {d:.2f} |" for n, (s, d) in sorted(results[best_key]["best_seg"].items())
            )
            + "\n"
        )
    print(f"\nOK: written {OUT}")


if __name__ == "__main__":
    main()
