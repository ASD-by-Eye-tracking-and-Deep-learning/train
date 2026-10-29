"""T11 step 2 — window + colour validation, after the geometry check
(validate_rendering_t11_participant.py) fixed the coordinate convention.
Run from train/:

    uv run python scripts/validate_rendering_t11_colour.py [--n 60]

For the images whose own child's best segment is geometrically closest
(smallest directed chamfer in the geometry report), slide a
MAX_POINTS-sample window (step 5) over that child's best (file, trial)
segment, pick the window whose rasterised path best matches the image (symmetric
chamfer), render it with rendering.render_original for each time unit, and
compare with the real image:
- line-mask IoU after 1-px dilation (does the window reproduce the drawing?)
- Pearson r of R, G, B between rendered and real on pixels that are line in
  both (does the colour encoding — kinematics, cap, time unit — match?).
"""
import argparse
import os
import re
import tempfile

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, distance_transform_edt as edt

from asd_train.rendering import MAX_POINTS, load_trial, render_original

GEOM_REPORT = "artifacts/rendering_t11/participant_geometry.md"
RAW_DIR = "../../data/mahmoud-dataset/Eye-tracking Output"
IMG_DIR = "../../data/mahmoud-dataset/Images"
OUT = "artifacts/rendering_t11/colour_validation.md"
FG = 30
AX = (80.0, 576.0, 57.6, 427.2)  # Matplotlib default axes box on 640x480, pixel rows top-down

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=60, help="images to check (lowest geometric distance first)")
parser.add_argument("--eye", default="Right")
args = parser.parse_args()


def img_path(name):
    for sub in ("ASD", "non-ASD"):
        p = os.path.join(IMG_DIR, sub, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


def raster_samples(samples):
    from PIL import ImageDraw

    x0, x1, y0, y1 = AX
    pts = [(x0 + s.x / 1280 * (x1 - x0), y1 - s.y / 1024 * (y1 - y0)) for s in samples]
    im = Image.new("1", (640, 480), 0)
    ImageDraw.Draw(im).line(pts, fill=1, width=1)
    return np.asarray(im, dtype=bool)


rows = re.findall(r"^\| (\S+\.png) \| (\S+?)/(\S+) \| ([\d.]+) \|$", open(GEOM_REPORT, encoding="utf-8").read(), re.M)
rows = sorted(rows, key=lambda r: float(r[3]))[: args.n]

results = []
tmp = tempfile.mkdtemp()
for name, csv_name, trial, _ in rows:
    pid = int(name.rsplit("_", 1)[1].split(".")[0])
    real = np.asarray(Image.open(img_path(name)).convert("RGB"), dtype=float) / 255
    real_m = real.max(axis=2) > FG / 255
    real_dt = edt(~real_m)
    full = load_trial(os.path.join(RAW_DIR, csv_name), trial, eye=args.eye, participant=pid, start=0)
    # load_trial truncates to MAX_POINTS; reload the whole segment for window search
    import asd_train.rendering as R

    saved = R.MAX_POINTS
    R.MAX_POINTS = 10**9
    seg = load_trial(os.path.join(RAW_DIR, csv_name), trial, eye=args.eye, participant=pid)
    R.MAX_POINTS = saved
    best = (np.inf, 0)
    for start in range(0, max(len(seg) - MAX_POINTS, 0) + 1, 5):
        r = raster_samples(seg[start : start + MAX_POINTS])
        if not r.any():
            continue
        d = (real_dt[r].mean() + edt(~r)[real_m].mean()) / 2
        best = min(best, (d, start))
    d, start = best
    window = seg[start : start + MAX_POINTS]
    entry = {"image": name, "segment": f"{csv_name}/{trial}", "start": start, "chamfer": d}
    for unit in ("sample", "ms"):
        out = os.path.join(tmp, f"{name}_{unit}.png")
        render_original(window, out, time_unit=unit)
        ren = np.asarray(Image.open(out).convert("RGB").resize((640, 480)), dtype=float) / 255
        ren_m = ren.max(axis=2) > FG / 255
        inter = binary_dilation(ren_m) & real_m
        union = ren_m | real_m
        entry[f"iou_{unit}"] = float((binary_dilation(ren_m) & real_m).sum() / max(union.sum(), 1))
        both = ren_m & real_m
        for c, ch in enumerate("RGB"):
            a, b = ren[..., c][both], real[..., c][both]
            entry[f"r{ch}_{unit}"] = float(np.corrcoef(a, b)[0, 1]) if both.sum() > 10 and a.std() > 0 and b.std() > 0 else float("nan")
    results.append(entry)
    print(entry, flush=True)


def med(k):
    v = np.array([e[k] for e in results], dtype=float)
    return float(np.nanmedian(v))


os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(
        f"# T11 — window + colour validation ({len(results)} images, eye={args.eye})\n\n"
        "Images: smallest own-child directed chamfer in the geometry report. "
        f"Window: best {MAX_POINTS}-sample window (step 5) in that child's segment.\n\n"
        "| Time unit | median IoU (dilated) | median r(R=velocity) | median r(G=accel) | median r(B=jerk) |\n"
        "|---|---|---|---|---|\n"
        + "\n".join(
            f"| {u} | {med('iou_' + u):.3f} | {med('rR_' + u):.3f} | {med('rG_' + u):.3f} | {med('rB_' + u):.3f} |"
            for u in ("sample", "ms")
        )
        + f"\n\nMedian symmetric chamfer of chosen window: {med('chamfer'):.2f} px (640x480).\n\n"
        "| Image | Segment | Start | Chamfer | IoU sample | rR sample | rR ms |\n|---|---|---|---|---|---|---|\n"
        + "\n".join(
            f"| {e['image']} | {e['segment']} | {e['start']} | {e['chamfer']:.2f} | {e['iou_sample']:.3f} | {e['rR_sample']:.3f} | {e['rR_ms']:.3f} |"
            for e in results
        )
        + "\n"
    )
print(open(OUT, encoding="utf-8").read()[:1500])
