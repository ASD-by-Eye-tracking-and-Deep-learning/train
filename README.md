# train

Notebooks used to train the ASD eye-tracking scanpath classifiers. Mirrors the
`final/` folder on Google Drive (`Duc n Huyen/ASD/train/final/`), where the
notebooks are actually edited (Colab). Run the sync workflow described below to
pull updates from Drive into this repo.

## Contents

- `inceptionv3/inceptionv3_original_97.ipynb` — InceptionV3 transfer-learning
  model (97% reported accuracy), the model discussed in the submitted paper.
  **Note:** the model file actually deployed in
  [app](https://github.com/ASD-by-Eye-tracking-and-Deep-learning/app)/core/customCNN.h5
  is a *different*, simpler custom CNN — not InceptionV3. See open question in
  the meta-repo's `record.md`.
- `ghost/` — GhostNet-based experiments (MobileNetV4-small, ConvNeXt-tiny-v2).
- `ghost+cbam/` — Same architectures with CBAM attention added.
- `cbam_module.py`, `ghost_module.py` — shared building blocks imported by the
  `ghost/` and `ghost+cbam/` notebooks.

Each `ghost`/`ghost+cbam` notebook trains 3 independent runs (PA1/PA2/PA3) on the
held-out 64-image test set; only the best-accuracy run's weights are kept here
(see table below) — the other runs' weights live only on Drive
(`train/final/` there keeps all 3 for reference).

| Model | Best run | Test accuracy |
|---|---|---|
| `ghost/mobilenetv4_small` | PA2 | 81.25% |
| `ghost/convnexttiny_v2` | PA1 | 82.81% |
| `ghost+cbam/mobilenetv4_small` | PA2 | 82.81% |
| `ghost+cbam/convnexttiny_v2` | PA1 (tied with PA3) | 78.12% |

## Model weights

Trained weights (`.h5` / `.pth`) are not committed here (see `.gitignore`) — some
exceed GitHub's 100MB file limit. They are hosted on Hugging Face:

https://huggingface.co/Jason-42195/asd-eye-tracking-models

Folder layout on Hugging Face mirrors this repo (`inceptionv3/`, `ghost/`,
`ghost+cbam/`).

## Keeping this repo in sync with Google Drive

The actual editing happens in Colab notebooks on Google Drive
(`Duc n Huyen/ASD/train/final/`), mounted locally via rclone at
`~/GoogleDrive/Duc n Huyen/ASD/train/final/`. After finishing a training round
there:

1. Copy the updated notebook(s) and shared `.py` modules from
   `~/GoogleDrive/Duc n Huyen/ASD/train/final/<arch>/` into the matching folder
   here.
2. Copy only the best-accuracy `.pth`/`.h5` into this repo's matching folder as
   `best.pth`/`best.h5` (check the notebook's printed test accuracy per run to
   pick the best one — see table above for the current picks).
3. `git add`, commit, push.
4. `hf upload Jason-42195/asd-eye-tracking-models <folder> <folder>` to publish
   the updated weight file(s).

Earlier, less-complete experiment rounds are archived on Drive under
`train/experiments/` (not mirrored here).
