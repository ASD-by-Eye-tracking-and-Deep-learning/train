# train

Training code for the ASD eye-tracking scanpath classifiers. One `src/`
package, one notebook (`train_all.ipynb`) that can run every model/config
explored for the paper.

## Layout

```
train/
├── src/asd_train/
│   ├── layers/          # GhostModule, CBAM
│   ├── models/           # GhostVariantModel (12 PyTorch configs), InceptionV3 hypermodel (TF)
│   ├── data.py            # dataset loading (PyTorch + TensorFlow paths)
│   ├── augmentation.py     # FMMix1 (two variants — see docstring)
│   ├── phase_schedules.py  # per-backbone LR/freeze schedules
│   ├── train_torch.py      # generic training loop for the 12 Ghost/CBAM configs
│   ├── train_tf.py         # InceptionV3 (Keras-Tuner) training pipeline
│   └── configs.py          # GhostRunConfig/InceptionV3Config + PRESETS
├── train_all.ipynb         # pick a config (or loop all of them), train, evaluate
├── sync_to_drive.sh         # one-way mirror: this repo -> Drive `code/` (for Colab)
└── pyproject.toml           # uv-managed deps
```

## Where this code came from

This replaces 5 original per-notebook Colab experiments (`inceptionv3_original_97.ipynb`,
and 4 PyTorch notebooks each training 3 GhostNet variants with/without CBAM).
Every hyperparameter in `configs.py` was read directly out of those notebooks,
not assumed uniform — several real per-config differences were found and are
preserved (see `configs.py` docstring and inline comments): patience, max_norm,
GhostModule kernel size, whether `pos_weight` is used inside the FMMix1 loss,
the LR scheduler (`cosine` for most, `none` for ghost+cbam/mobilenetv4,
`ReduceLROnPlateau` for one specific config), and one structural difference
(ghost-only ConvNeXtV2 PA2 uses a single GhostBottleneck stage instead of two).

**PA1/PA2/PA3 are not repeated trials** — they're 3 different ways of inserting
GhostModule into the backbone:
- **PA1 (head)**: replace the 1×1 head conv with GhostModule.
- **PA2 (bottleneck)**: stack GhostBottleneck blocks after the backbone.
- **PA3 (adapter)**: GhostModule as a residual adapter on the backbone output.

CBAM, when enabled, is inserted at a different point per variant (verified
from the original code, not assumed uniform): after the backbone for PA1,
*inside* each GhostBottleneck for PA2, after the adapter for PA3.

## Accuracy per config (held-out 64-image test set)

| Config | Backbone | Variant | CBAM | Test accuracy |
|---|---|---|---|---|
| `inceptionv3` | InceptionV3 | — | — | **97%** (reported in paper) |
| `ghost_mobilenetv4_pa1` | MobileNetV4-small | head | no | 78.12% |
| `ghost_mobilenetv4_pa2` | MobileNetV4-small | bottleneck | no | **81.25%** |
| `ghost_mobilenetv4_pa3` | MobileNetV4-small | adapter | no | 78.12% |
| `ghost_convnexttiny_pa1` | ConvNeXtV2-tiny | head | no | **82.81%** |
| `ghost_convnexttiny_pa2` | ConvNeXtV2-tiny | bottleneck | no | 78.12% |
| `ghost_convnexttiny_pa3` | ConvNeXtV2-tiny | adapter | no | 78.12% |
| `ghostcbam_mobilenetv4_pa1` | MobileNetV4-small | head | yes | 79.69% |
| `ghostcbam_mobilenetv4_pa2` | MobileNetV4-small | bottleneck | yes | **82.81%** |
| `ghostcbam_mobilenetv4_pa3` | MobileNetV4-small | adapter | yes | 78.12% |
| `ghostcbam_convnexttiny_pa1` | ConvNeXtV2-tiny | head | yes | **78.12%** (tied w/ pa3) |
| `ghostcbam_convnexttiny_pa2` | ConvNeXtV2-tiny | bottleneck | yes | 57.81% |
| `ghostcbam_convnexttiny_pa3` | ConvNeXtV2-tiny | adapter | yes | 78.12% |

Bold = the best-accuracy run per notebook — these are the ones kept as weight
files (see below). `BEST_PRESETS` in `configs.py` names them programmatically.

**Note on `app/`'s deployed model:** the paper (§4.6) says the web app uses
"the optimized InceptionV3 model", but the actual deployed file
(`app/core/customCNN.h5`) is a small custom CNN, not InceptionV3 — an
unresolved discrepancy, see the meta-repo's `record.md`.

## Model weights

Trained weights (`.h5`/`.pth`) are not committed here — some exceed GitHub's
100MB limit. They're hosted on Hugging Face:

https://huggingface.co/Jason-42195/asd-eye-tracking-models

`train_all.ipynb` saves new runs into a local `artifacts/` (gitignored).

## Running

```bash
uv sync                      # installs torch, tensorflow, timm, etc.
uv run jupyter lab train_all.ipynb
```

Locally, the dataset is expected at `../data/mahmoud-dataset/` — i.e. `data/`
next to this `train/` checkout, inside the `ASD/` meta-repo, not inside
`train/` itself. It's gitignored there too (too large for git); see
`ASD/.agents/record.md` for how to populate it (a local copy pulled via
`rclone copy` from Drive, not the slow FUSE mount).

Or on Colab: run `sync_to_drive.sh` once to mirror this repo into
`Duc n Huyen/ASD/ML4Autism/train/code/`, then open `train_all.ipynb` from there — Drive
also holds the dataset (`ML4Autism/train/datasets/mahmoud-dataset/`) that the notebook
mounts and reads from.

## Keeping Drive in sync

Local git is the source of truth. After changing anything under `src/` or
`train_all.ipynb`:

```bash
./sync_to_drive.sh
```

This one-way rsyncs code (never weights, never the dataset) into
`Duc n Huyen/ASD/ML4Autism/train/code/`. Drive is not edited directly and nothing syncs
back from it — if you need to change training code, do it here and re-run
the script.

Drive's `train/` also holds `artifacts/` (old model-weight files from every
historical experiment round, for reference — no code/notebooks) and
`datasets/mahmoud-dataset/` (the dataset itself, too large for git).
