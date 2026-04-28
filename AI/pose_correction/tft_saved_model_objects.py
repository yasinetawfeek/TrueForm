"""
Definitions required to load TFT pose-correction .keras files saved from
`TFT_pose_correction.ipynb`.

The notebook wires `Lambda(normalized_time, ...)` and custom layers; Keras 3
deserialization looks up `normalized_time` and the layer classes by name.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class GatedResidualNetwork(layers.Layer):
    """TFT-style GRN: ELU MLP with sigmoid gate and skip (Lim et al., 2020)."""

    def __init__(self, width: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.width = int(width)
        self.dropout = float(dropout)
        self.fc1 = layers.Dense(self.width, activation="elu")
        self.fc2 = layers.Dense(self.width)
        self.gate = layers.Dense(self.width, activation="sigmoid")
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.drop = layers.Dropout(self.dropout)
        self.proj = None

    def build(self, input_shape):
        last = int(input_shape[-1])
        if last != self.width:
            self.proj = layers.Dense(self.width)
        super().build(input_shape)

    def call(self, inputs, training=False):
        skip = self.proj(inputs) if self.proj is not None else inputs
        h = self.fc1(inputs)
        h = self.fc2(h)
        h = self.drop(h, training=training)
        g = self.gate(inputs)
        return self.norm(skip + g * h)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"width": self.width, "dropout": self.dropout})
        return cfg


class TemporalFusionBlock(layers.Layer):
    """Self-attention + pointwise FFN with residuals and layer norms."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dff: int,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.dff = int(dff)
        self.dropout = float(dropout)
        key_dim = max(1, self.d_model // self.num_heads)
        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=key_dim,
            dropout=self.dropout,
        )
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = keras.Sequential(
            [
                layers.Dense(self.dff, activation="gelu"),
                layers.Dropout(self.dropout),
                layers.Dense(self.d_model),
                layers.Dropout(self.dropout),
            ]
        )

    def build(self, input_shape):
        """Keras 3: parent must build MHA or load_model can mark us built while attention is not."""
        super().build(input_shape)
        # MultiHeadAttention.build(query_shape, value_shape, key_shape=None)
        self.mha.build(input_shape, input_shape, input_shape)
        # Do not call self.ffn.build() here: Sequential may prepend InputLayer and break checkpoints.

    def call(self, x, training=False):
        attn = self.mha(x, x, training=training)
        x = self.norm1(x + attn)
        f = self.ffn(x, training=training)
        return self.norm2(x + f)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "dff": self.dff,
                "dropout": self.dropout,
            }
        )
        return cfg


def normalized_time(x):
    """Matches `build_tft_pose_model` in TFT_pose_correction.ipynb."""
    b = tf.shape(x)[0]
    tlen = tf.shape(x)[1]
    u = tf.linspace(0.0, 1.0, tlen)
    u = tf.reshape(u, [1, tlen, 1])
    return tf.tile(u, [b, 1, 1])


TFT_CUSTOM_OBJECTS: dict[str, type | object] = {
    "GatedResidualNetwork": GatedResidualNetwork,
    "TemporalFusionBlock": TemporalFusionBlock,
    "normalized_time": normalized_time,
}

__all__ = [
    "GatedResidualNetwork",
    "TemporalFusionBlock",
    "normalized_time",
    "TFT_CUSTOM_OBJECTS",
]
