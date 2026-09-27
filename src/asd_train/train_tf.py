"""InceptionV3 training pipeline (TensorFlow/Keras-Tuner) — the accuracy-focused
track reported at 97% test accuracy in the paper. Mirrors
``inceptionv3_original_97.ipynb`` exactly, including its slightly unusual
FMMix1 bridge (the torch implementation is called from inside a TF graph via
``tf.numpy_function``).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

from asd_train.augmentation import FMMixArgs, fmmix1_multimode
from asd_train.configs import InceptionV3Config
from asd_train.data import load_tf_dataset
from asd_train.models.inceptionv3_tf import build_inceptionv3


@dataclass
class TFRunResult:
    config_name: str
    classification_report: str
    confusion_matrix: "np.ndarray"
    best_model_path: str


def _augment_pre_fmmix1_tf(images, labels):
    import tensorflow as tf

    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_brightness(images, max_delta=0.1)
    images = tf.image.random_contrast(images, lower=0.8, upper=1.2)
    return images, labels


def _apply_fmmix1_tf(images, labels, fmmix_args: FMMixArgs):
    import tensorflow as tf
    import torch

    images = tf.cast(images, tf.float32)

    def _wrapper(images_np, labels_np):
        images_torch = torch.from_numpy(images_np)
        labels_torch = torch.from_numpy(labels_np.astype(np.int64))
        images_aug, _ = fmmix1_multimode(fmmix_args, images_torch, labels_torch)
        return images_aug.numpy(), labels_np.astype(np.int64)

    augmented_images, new_labels = tf.numpy_function(
        func=_wrapper, inp=[images, labels], Tout=[tf.float32, tf.int64]
    )
    augmented_images.set_shape(images.shape)
    new_labels.set_shape(labels.shape)
    return augmented_images, new_labels


def run_inceptionv3_experiment(cfg: InceptionV3Config, dataset_root: str, output_dir: str) -> TFRunResult:
    import tensorflow as tf
    import keras_tuner as kt
    from sklearn.metrics import classification_report, confusion_matrix
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall

    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    dataset_path = os.path.join(dataset_root, os.path.basename(cfg.dataset_path.rstrip("/")))
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, f"{cfg.name}.h5")

    train, val, test = load_tf_dataset(
        dataset_path, train_batches=cfg.train_batches, val_batches=cfg.val_batches, test_batches=cfg.test_batches
    )

    fmmix_args = FMMixArgs(alpha=cfg.fmmix_alpha, mask_area_mode=1)
    aug_train = train.map(_augment_pre_fmmix1_tf).map(lambda x, y: _apply_fmmix1_tf(x, y, fmmix_args))

    tuner = kt.RandomSearch(
        build_inceptionv3,
        objective="val_accuracy",
        max_trials=cfg.tuner_max_trials,
        directory=os.path.join(output_dir, "hypertune_inception"),
        project_name="fmmix1_dynamic",
    )

    tuner.search(
        aug_train,
        validation_data=val,
        epochs=cfg.tuner_epochs,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        ],
    )
    best_hp = tuner.get_best_hyperparameters(1)[0]

    best_model = tuner.hypermodel.build(best_hp)
    best_model.fit(
        aug_train,
        validation_data=val,
        epochs=cfg.final_epochs,
        steps_per_epoch=math.ceil(547 / 32),
    )
    best_model.save(best_model_path)

    precision_m, recall_m, accuracy_m = Precision(), Recall(), BinaryAccuracy()
    y_all, yhat_all = [], []
    for images, labels in test.as_numpy_iterator():
        yhat = best_model.predict(images)
        y_all += list(labels)
        yhat_all += list(yhat.round())
        precision_m.update_state(labels, yhat)
        recall_m.update_state(labels, yhat)
        accuracy_m.update_state(labels, yhat)

    y_all = np.array(y_all).reshape(-1, 1)
    yhat_all = np.array(yhat_all).reshape(-1, 1)
    report = classification_report(y_all, yhat_all)
    conf_mat = confusion_matrix(y_all, yhat_all)

    print(f"[{cfg.name}] Test Accuracy:  {accuracy_m.result().numpy():.4f}")
    print(f"[{cfg.name}] Test Precision: {precision_m.result().numpy():.4f}")
    print(f"[{cfg.name}] Test Recall:    {recall_m.result().numpy():.4f}")
    print("\n" + report)

    return TFRunResult(
        config_name=cfg.name,
        classification_report=report,
        confusion_matrix=conf_mat,
        best_model_path=best_model_path,
    )
