# train

Notebooks used to train the ASD eye-tracking scanpath classifiers.

## Contents

- `inceptionv3_original_97.ipynb` — InceptionV3 transfer-learning model (97% reported
  accuracy), the model used in the submitted paper and currently deployed in the
  [app](https://github.com/ASD-by-Eye-tracking-and-Deep-learning/app) prototype.
- `ghost/` — GhostNet-based experiments (MobileNetV4-small, ConvNeXt-tiny-v2).
- `ghost+cbam/` — Same architectures with CBAM attention added.

## Model weights

Trained weights (`.h5` / `.pth`) are not committed here (see `.gitignore`) — some
exceed GitHub's 100MB file limit. They are hosted on Hugging Face:

https://huggingface.co/Jason-42195/asd-eye-tracking-models

Folder layout on Hugging Face mirrors this repo (`models/`, `ghost/`, `ghost+cbam/`).
