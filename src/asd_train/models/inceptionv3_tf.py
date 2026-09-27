"""Keras-Tuner hypermodel for the accuracy-focused InceptionV3 track
(TensorFlow) — the model reported at 97% test accuracy in the paper.
"""
from __future__ import annotations


def build_inceptionv3(hp):
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import InceptionV3
    from tensorflow.keras.optimizers import Adam

    base_model = InceptionV3(include_top=False, input_shape=(256, 256, 3), weights="imagenet")

    unfreeze_from = hp.Choice("unfreeze_from", ["block5a_expand", "block6a_expand", "mixed7"])
    set_trainable = False
    for layer in base_model.layers:
        if unfreeze_from in layer.name:
            set_trainable = True
        layer.trainable = set_trainable

    inputs = tf.keras.Input(shape=(256, 256, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    for i in range(hp.Int("num_dense_layers", 1, 3)):
        x = layers.Dense(
            units=hp.Int(f"units_{i}", min_value=64, max_value=512, step=64),
            activation=hp.Choice(f"activation_{i}", ["relu", "tanh"]),
        )(x)
        x = layers.Dropout(rate=hp.Float(f"dropout_{i}", min_value=0.2, max_value=0.5, step=0.1))(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=Adam(hp.Choice("lr", [1e-3, 1e-4, 1e-5])),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
