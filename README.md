# train

Notebooks used to train the ASD eye-tracking scanpath classifiers.

## Contents

- `inceptionv3_original_97.ipynb` — InceptionV3 transfer-learning model (97% reported
  accuracy), the model discussed in the submitted paper.
  **Note:** the model file actually deployed in
  [app](https://github.com/ASD-by-Eye-tracking-and-Deep-learning/app)/core/customCNN.h5
  (renamed from `best.h5` 2026-09-27) is a *different*, simpler custom CNN (3×
  Conv2D+MaxPool → Dense(256) → Dropout → Dense(1, sigmoid), 256×256×3 input) —
  not InceptionV3. Confirmed by inspecting the Keras `model_config` directly. The
  relationship between this notebook and the deployed model is unclear and needs
  checking.
- `ghost/` — GhostNet-based experiments (MobileNetV4-small, ConvNeXt-tiny-v2).
- `ghost+cbam/` — Same architectures with CBAM attention added.

## Model weights

Trained weights (`.h5` / `.pth`) are not committed here (see `.gitignore`) — some
exceed GitHub's 100MB file limit. They are hosted on Hugging Face:

https://huggingface.co/Jason-42195/asd-eye-tracking-models

Folder layout on Hugging Face mirrors this repo (`models/`, `ghost/`, `ghost+cbam/`).
