"""T31 -- reimplementation of the ORIGINAL ML4Autism scanpath-rendering
algorithm (Carette et al. 2018 IEEE ICDIM Sec III.D; Carette/Elbattah 2019
HEALTHINF Sec 3.4), confirmed by reading both papers directly -- no public
source code exists anywhere for this (T30, ``record.md`` Decision #20).

Deliberately separate from ``app/api/diagnose.py`` (the team's own deploy-time
webcam approximation) -- this module targets the SMI raw-export format the
547 *training* images were actually rendered from, and is meant to be
validated against those real images (T31) before informing any change to the
deployed app (T32).

Exact spec, quoted from the papers (not guessed):
- "A line is drawn for each transition from position [x(t), y(t)] to
  [x(t+1), y(t+1)]."
- RGB = (velocity, acceleration, jerk) at instant t. No smoothing of any
  kind is mentioned in either paper -- the EMA step in ``app/api/diagnose.py``
  is this team's own addition for noisier webcam input, confirmed not part
  of the original algorithm.
- "All values were capped to one-quarter of the diagonal length of the
  screen" -- a FIXED cap, not per-trial min-max (unlike the current app
  code). The recording screen was 1280x1024, so
  cap = sqrt(1280**2 + 1024**2) / 4.
- "The images constructed were vertically mirrored as the y origin was
  located at the bottom of the screen."
- Exactly 200 consecutive coordinates per image.
- Matplotlib, black background, 640x480 output.

Open question this module's validation (T31) is meant to resolve empirically,
not assume: whether velocity/acceleration/jerk are computed against the raw
per-sample SMI timestamps (this module's default) or per-sample-index with
unit time step -- the papers don't specify units precisely enough to know
which the original implementation used.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Recording screen resolution both original papers report (1280x1024),
# NOT the 640x480 output image size -- confirmed via direct quote, see
# module docstring and record.md Decision #20.
RECORDING_SCREEN_W = 1280
RECORDING_SCREEN_H = 1024
CAP = math.sqrt(RECORDING_SCREEN_W**2 + RECORDING_SCREEN_H**2) / 4  # ~409.9

OUTPUT_SIZE = (640, 480)  # (width, height), matches the real dataset images
MAX_POINTS = 200


@dataclass
class GazeSample:
    x: float
    y: float
    t_ms: float


def load_trial(csv_path: str, trial: str, eye: str = "Right") -> list[GazeSample]:
    """Load one trial's raw gaze samples from an SMI export CSV
    (``data/mahmoud-dataset/Eye-tracking Output/*.csv``), filtered to a
    single ``Trial`` column value. Drops rows with missing/non-numeric POR
    coordinates (SMI marks blinks/track-loss as ``-`` or blank)."""
    df = pd.read_csv(csv_path)
    df = df[df["Trial"] == trial]
    x_col, y_col = f"Point of Regard {eye} X [px]", f"Point of Regard {eye} Y [px]"
    df = df[[x_col, y_col, "RecordingTime [ms]"]].apply(pd.to_numeric, errors="coerce")
    df = df.dropna()
    samples = [
        GazeSample(x=row[x_col], y=row[y_col], t_ms=row["RecordingTime [ms]"])
        for _, row in df.iterrows()
    ]
    return samples[:MAX_POINTS]


def compute_kinematics(
    samples: list[GazeSample], time_unit: str = "sample"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Velocity/acceleration/jerk via finite differences on raw (unsmoothed)
    samples -- no EMA, matching the original papers (see module docstring).
    Returns arrays aligned to segments (length len(samples)-1 for velocity,
    shorter for acceleration/jerk since each derivative drops one point).

    ``time_unit``: neither paper specifies the exact time unit used for the
    derivatives precisely enough to know for certain (T31 open question, see
    module docstring) -- tested empirically against real images:
    - "ms": divide by real ``RecordingTime [ms]`` deltas each derivative
      step. Produces acceleration/jerk many orders of magnitude smaller than
      velocity (compounding division by small ms deltas), which does not
      match the real images' observed per-channel intensities (R/G/B maxima
      all in a comparable 0-255 range) -- tested and rejected.
    - "sample" (default): dt=1 (raw per-sample-index differencing, no time
      normalization) -- since SMI's sampling rate is approximately constant,
      this is a constant-factor rescaling of true time-based derivatives.
      Produces velocity/acceleration/jerk magnitudes in the same
      hundreds-of-pixels order as the fixed cap (``CAP`` ~409.9), consistent
      with a single shared cap making sense across all three channels.
    """
    n = len(samples)
    velocity = np.zeros(n - 1)
    for i in range(n - 1):
        dx = samples[i + 1].x - samples[i].x
        dy = samples[i + 1].y - samples[i].y
        if time_unit == "ms":
            dt = max(samples[i + 1].t_ms - samples[i].t_ms, 1e-6)
        else:
            dt = 1.0
        velocity[i] = math.hypot(dx, dy) / dt

    acceleration = np.zeros(max(n - 2, 0))
    for i in range(len(acceleration)):
        if time_unit == "ms":
            dt = max(samples[i + 2].t_ms - samples[i + 1].t_ms, 1e-6)
        else:
            dt = 1.0
        acceleration[i] = (velocity[i + 1] - velocity[i]) / dt

    jerk = np.zeros(max(n - 3, 0))
    for i in range(len(jerk)):
        if time_unit == "ms":
            dt = max(samples[i + 3].t_ms - samples[i + 2].t_ms, 1e-6)
        else:
            dt = 1.0
        jerk[i] = (acceleration[i + 1] - acceleration[i]) / dt

    return velocity, acceleration, jerk


def render_original(samples: list[GazeSample], output_path: str, time_unit: str = "sample") -> None:
    """Render one trial's raw gaze samples using the confirmed original
    spec: line per transition, RGB=(velocity,acceleration,jerk) capped at a
    FIXED 1/4-screen-diagonal bound, vertically mirrored, black background,
    Matplotlib, 640x480 output. See ``compute_kinematics`` for ``time_unit``."""
    velocity, acceleration, jerk = compute_kinematics(samples, time_unit=time_unit)

    fig_w, fig_h = OUTPUT_SIZE[0] / 100, OUTPUT_SIZE[1] / 100
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    ax.set_xlim(0, RECORDING_SCREEN_W)
    ax.set_ylim(0, RECORDING_SCREEN_H)
    ax.invert_yaxis()  # vertical mirror: SMI y-origin at screen bottom
    ax.axis("off")

    n_segments = len(velocity)
    for i in range(n_segments):
        v = velocity[i]
        a = acceleration[i - 1] if 0 <= i - 1 < len(acceleration) else 0.0
        j = jerk[i - 2] if 0 <= i - 2 < len(jerk) else 0.0

        r = min(abs(v) / CAP, 1.0)
        g = min(abs(a) / CAP, 1.0)
        b = min(abs(j) / CAP, 1.0)

        p1, p2 = samples[i], samples[i + 1]
        ax.plot([p1.x, p2.x], [p1.y, p2.y], color=(r, g, b), linewidth=1)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(output_path, facecolor="black")
    plt.close(fig)
