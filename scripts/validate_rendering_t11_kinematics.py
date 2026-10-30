"""T11 step 3 — which acceleration/jerk definition reproduces the G/B
channels of the real images? Run from train/ after validate_rendering_t11_colour.py:

    uv run python scripts/validate_rendering_t11_kinematics.py

Uses the windows recovered in artifacts/rendering_t11/colour_validation.md.
For every line segment of a window, reads the real image's colour at the
segment midpoint (skipping black pixels, i.e. covered/thin lines) and
correlates it with the colour each candidate formula predicts for that
segment. Candidates (per-sample differencing, dt = 1, as step 2 established):

- scalar: v = |p(t+1) - p(t)|, a = v(t+1) - v(t), j = a(t+1) - a(t)  (rendering.py today)
- vector: vel = p(t+1) - p(t), a = |vel(t+1) - vel(t)|, j = |acc(t+1) - acc(t)|
and, for each, the derivative aligned to the segment start (lag 0) or as
rendering.py does (acceleration index i-1, jerk i-2). Values are clipped at
the fixed cap (1/4 screen diagonal) and divided by it, as in the papers.
Spearman is used: the cap and Matplotlib anti-aliasing make the mapping
monotone but not linear.
"""
import os
import re

import numpy as np
from PIL import Image
from scipy import stats

import asd_train.rendering as R

RAW_DIR = "../../data/mahmoud-dataset/Eye-tracking Output"
IMG_DIR = "../../data/mahmoud-dataset/Images"
REPORT = "artifacts/rendering_t11/colour_validation.md"
OUT = "artifacts/rendering_t11/kinematics_variants.md"
AX = (80.0, 576.0, 57.6, 427.2)
CAP = R.CAP


def img_path(name):
    for sub in ("ASD", "non-ASD"):
        p = os.path.join(IMG_DIR, sub, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


def kinematics(x, y, kind):
    dx, dy = np.diff(x), np.diff(y)
    v = np.hypot(dx, dy)
    if kind == "scalar":
        a = np.diff(v)
        j = np.diff(a)
    else:
        ax_, ay_ = np.diff(dx), np.diff(dy)
        a = np.hypot(ax_, ay_)
        j = np.hypot(np.diff(ax_), np.diff(ay_))
    return v, a, j


def per_segment(v, a, j, lag):
    n = len(v)
    out = np.zeros((n, 3))
    for i in range(n):
        ia, ij = (i, i) if lag == 0 else (i - 1, i - 2)
        out[i, 0] = abs(v[i])
        out[i, 1] = abs(a[ia]) if 0 <= ia < len(a) else 0.0
        out[i, 2] = abs(j[ij]) if 0 <= ij < len(j) else 0.0
    return np.minimum(out / CAP, 1.0)


rows = re.findall(r"^\| (\S+\.png) \| (\S+?)/(\S+) \| (\d+) \|", open(REPORT, encoding="utf-8").read(), re.M)
variants = [(k, lag) for k in ("scalar", "vector") for lag in (0, "shift")]
pred = {v: [] for v in variants}
real_cols = []
saved = R.MAX_POINTS
for name, csv_name, trial, start in rows:
    pid = int(name.rsplit("_", 1)[1].split(".")[0])
    R.MAX_POINTS = 10**9
    seg = R.load_trial(os.path.join(RAW_DIR, csv_name), trial, participant=pid)
    R.MAX_POINTS = saved
    w = seg[int(start) : int(start) + saved]
    x = np.array([s.x for s in w])
    y = np.array([s.y for s in w])
    img = np.asarray(Image.open(img_path(name)).convert("RGB"), dtype=float) / 255
    x0, x1, y0, y1 = AX
    mx = x0 + (x[:-1] + x[1:]) / 2 / 1280 * (x1 - x0)
    my = y1 - (y[:-1] + y[1:]) / 2 / 1024 * (y1 - y0)
    xi, yi = np.clip(mx.round().astype(int), 0, 639), np.clip(my.round().astype(int), 0, 479)
    col = img[yi, xi]
    keep = col.max(axis=1) > 30 / 255
    real_cols.append(col[keep])
    for k, lag in variants:
        pred[(k, lag)].append(per_segment(*kinematics(x, y, k), lag)[keep])

real = np.concatenate(real_cols)
lines = [
    f"# T11 — acceleration/jerk definition ({len(rows)} images, {len(real)} segment midpoints)\n",
    "Spearman correlation between predicted and real channel value at segment midpoints.\n",
    "| Kinematics | Alignment | R (velocity) | G (acceleration) | B (jerk) |",
    "|---|---|---|---|---|",
]
for k, lag in variants:
    p = np.concatenate(pred[(k, lag)])
    rho = [stats.spearmanr(p[:, c], real[:, c]).statistic for c in range(3)]
    lines.append(f"| {k} | {'a[i], j[i]' if lag == 0 else 'a[i-1], j[i-2]'} | {rho[0]:.3f} | {rho[1]:.3f} | {rho[2]:.3f} |")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
