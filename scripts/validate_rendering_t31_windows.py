"""T31 geometry-only window search: a real image uses 200 consecutive
coordinates from *somewhere* in a trial (not necessarily the first 200 --
load_trial()'s samples[:200] was wrong for matching). Slide every 200-sample
window over every trial, rasterise the polyline (no colour) at 160x120 under
several geometry variants, and score symmetric chamfer against each real
image's stroke mask. A match must be both low in absolute terms and clearly
separated from the runner-up; otherwise report no match.
Run from train/: uv run python scripts/validate_rendering_t31_windows.py
"""
import glob, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt as edt

sys.path.insert(0, "scripts")
from validate_rendering_t31 import PARTICIPANTS, RAW_DIR

W, H, S = 160, 120, 4  # low-res raster, 640/160
STRIDE, N = 10, 200


def real_mask(p):
    a = np.asarray(Image.open(p).convert("RGB").resize((640, 480)))
    m = a.max(axis=2) > 8
    return m.reshape(H, S, W, S).any(axis=(1, 3))


def raster(xs, ys):
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    d.line(list(zip(xs, ys)), fill=255, width=1)
    return np.asarray(im) > 0


def chamfer(r, rdt, m, mdt):
    if r.sum() < 5 or m.sum() < 5:
        return 1e9
    return (mdt[r].mean() + rdt[m].mean()) / 2  # in low-res px (x4 = full px)


def variants(x, y, eye_note):
    sx, sy = W / 1280, H / 1024
    yield "noflip", x * sx, y * sy
    yield "flipY", x * sx, (1024 - y) * sy


def main():
    for pid, cfg in PARTICIPANTS.items():
        reals = {os.path.basename(p): real_mask(p) for p in sorted(glob.glob(os.path.join(cfg["images_dir"], cfg["pattern"])))}
        rdts = {n: edt(~m) for n, m in reals.items()}
        rows = []
        for fid in cfg["files"]:
            df = pd.read_csv(os.path.join(RAW_DIR, f"{fid}.csv"), low_memory=False)
            for eye in ("Right", "Left"):
                xc, yc = f"Point of Regard {eye} X [px]", f"Point of Regard {eye} Y [px]"
                if xc not in df.columns:
                    continue
                for trial, g in df.groupby("Trial"):
                    g = g[[xc, yc]].apply(pd.to_numeric, errors="coerce").dropna()
                    X, Y = g[xc].values, g[yc].values
                    for s in range(0, max(len(X) - N, 0) + 1, STRIDE):
                        for vn, vx, vy in variants(X[s:s+N], Y[s:s+N], eye):
                            r = raster(vx, vy)
                            rd = edt(~r) if r.sum() >= 5 else None
                            if rd is None: continue
                            for n, m in reals.items():
                                rows.append((chamfer(r, rd, m, rdts[n]) * S, fid, trial, eye, s, vn, n))
        rows.sort(key=lambda t: t[0])
        print(f"\n=== participant {pid}: {len(rows)} comparisons")
        for r in rows[:5]: print("  %.1f px" % r[0], r[1:])
        # best per image, to see if any image is clearly matched
        seen = {}
        for r in rows:
            seen.setdefault(r[6], r)
        print("  best per image:", {k: round(v[0], 1) for k, v in seen.items()})
        print("  variant counts in top-30:", pd.Series([r[5] + "/" + r[3] for r in rows[:30]]).value_counts().to_dict())

main()
