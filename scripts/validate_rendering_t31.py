"""T31 -- validate rendering.py against real images by brute-force trial-to-
image matching: render every trial available for a participant, compare each
rendered output against every one of that participant's real images via
per-pixel MSE, and report the best match found (or honestly report none is
close). No trial-to-image mapping exists in any metadata (checked directly --
see plan.csv T31/record.md Decision #20), so this empirical search *is* the
mapping method.

Run from train/: `uv run python scripts/validate_rendering_t31.py`
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, "src")
from asd_train.rendering import load_trial, render_original

RAW_DIR = "../../data/mahmoud-dataset/Eye-tracking Output"
IMAGES_ASD = "../../data/mahmoud-dataset/Images/ASD"
IMAGES_NONASD = "../../data/mahmoud-dataset/Images/non-ASD"
OUT_DIR = "/tmp/t31_renders"

PARTICIPANTS = {
    11: {"files": ["13"], "images_dir": IMAGES_ASD, "pattern": f"*_11.png"},
    20: {"files": ["14"], "images_dir": IMAGES_ASD, "pattern": f"*_20.png"},
    50: {
        "files": ["1", "4", "6", "10", "15", "16", "22", "24"],
        "images_dir": IMAGES_NONASD,
        "pattern": f"*_50.png",
    },
}


def trials_in_file(csv_path: str) -> list[str]:
    df = pd.read_csv(csv_path, usecols=["Trial"], low_memory=False)
    return sorted(df["Trial"].dropna().unique().tolist())


def load_real_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((640, 480))
    return np.asarray(img, dtype=np.float64)


def compare(rendered_path: str, real_arrays: dict[str, np.ndarray]) -> tuple[str, float]:
    rendered = np.asarray(Image.open(rendered_path).convert("RGB").resize((640, 480)), dtype=np.float64)
    best_name, best_mse = None, float("inf")
    for name, real in real_arrays.items():
        mse = float(np.mean((rendered - real) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_name = name
    return best_name, best_mse


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for pid, cfg in PARTICIPANTS.items():
        print(f"\n{'=' * 20} participant {pid} {'=' * 20}")
        image_paths = sorted(glob.glob(os.path.join(cfg["images_dir"], cfg["pattern"])))
        real_arrays = {os.path.basename(p): load_real_image(p) for p in image_paths}
        print(f"{len(image_paths)} real images: {list(real_arrays.keys())}")

        overall_best = (None, None, float("inf"))  # (trial_label, matched_image, mse)
        for file_id in cfg["files"]:
            csv_path = os.path.join(RAW_DIR, f"{file_id}.csv")
            for trial in trials_in_file(csv_path):
                try:
                    samples = load_trial(csv_path, trial)
                    if len(samples) < 10:
                        continue
                    out_path = os.path.join(OUT_DIR, f"p{pid}_{file_id}_{trial}.png")
                    render_original(samples, out_path, time_unit="sample")
                    matched_image, mse = compare(out_path, real_arrays)
                    if mse < overall_best[2]:
                        overall_best = (f"{file_id}.csv/{trial}", matched_image, mse)
                except Exception as e:
                    print(f"  {file_id}.csv/{trial}: FAILED ({e})")
        label, image, mse = overall_best
        print(f"BEST for participant {pid}: {label} vs {image}  MSE={mse:.1f}")


if __name__ == "__main__":
    main()
