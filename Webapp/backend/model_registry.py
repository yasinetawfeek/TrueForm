from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from tensorflow import keras
from tensorflow.keras import layers

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKOUT_MODELS_DIR = REPO_ROOT / "AI" / "workout_classifier" / "models"
POSE_MODELS_DIR = REPO_ROOT / "AI" / "pose_correction" / "models"

# Reuse custom objects used when pose-correction models were exported.
POSE_CORRECTION_PY_DIR = REPO_ROOT / "AI" / "pose_correction"
if str(POSE_CORRECTION_PY_DIR) not in sys.path:
    sys.path.insert(0, str(POSE_CORRECTION_PY_DIR))

from lstm_saved_model_objects import LSTM_CUSTOM_OBJECTS  # noqa: E402
from tft_saved_model_objects import TFT_CUSTOM_OBJECTS  # noqa: E402


class PositionalEncoding(layers.Layer):
    def __init__(self, max_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model

    def build(self, input_shape):
        # Match the original training notebook serialization.
        self.pos_encoding = self.add_weight(
            name="pos_encoding",
            shape=(self.max_len, self.d_model),
            initializer="uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:seq_len, :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"max_len": self.max_len, "d_model": self.d_model})
        return cfg


class TransformerBlock(layers.Layer):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dff: int,
        rate: float = 0.1,
        dropout_rate: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if dropout_rate is not None:
            rate = float(dropout_rate)
        key_dim = max(1, d_model // num_heads)
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=rate,
        )
        self.ffn = keras.Sequential(
            [
                layers.Dense(dff, activation="gelu"),
                layers.Dense(d_model),
                layers.Dropout(rate),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = float(rate)

    def call(self, x, training=False):
        attn_output = self.mha(x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "dff": self.dff,
                # Keep backward compatibility with how the original model was serialized.
                "dropout_rate": self.dropout_rate,
            }
        )
        return cfg


TRANSFORMER_CUSTOM_OBJECTS = {
    "PositionalEncoding": PositionalEncoding,
    "TransformerBlock": TransformerBlock,
}


@dataclass
class WorkoutModelBundle:
    model: keras.Model
    class_names: List[str]
    mean: np.ndarray
    std: np.ndarray


@dataclass
class PoseModelBundle:
    model: Any
    model_type: str


WORKOUT_MODEL_FILES = {
    "bilstm": {
        "model": WORKOUT_MODELS_DIR / "bilstm_workout_classifier.keras",
        "class_names": WORKOUT_MODELS_DIR / "class_names_bilstm.json",
        "norm": WORKOUT_MODELS_DIR / "normalization_params_bilstm.json",
    },
    "gru": {
        "model": WORKOUT_MODELS_DIR / "gru_workout_classifier.keras",
        "class_names": WORKOUT_MODELS_DIR / "class_names_gru.json",
        "norm": WORKOUT_MODELS_DIR / "normalization_params_gru.json",
    },
    "transformer": {
        "model": WORKOUT_MODELS_DIR / "transformer_workout_classifier.keras",
        "class_names": WORKOUT_MODELS_DIR / "class_names_transformer.json",
        "norm": WORKOUT_MODELS_DIR / "normalization_params_transformer.json",
    },
}

POSE_MODEL_FILES = {
    "lstm_embedding": {
        "model": POSE_MODELS_DIR / "lstm_embedding_pose_correction_best.keras",
        "type": "lstm",
    },
    "tcn_film": {
        "model": POSE_MODELS_DIR / "tcn_film_pose_correction_best.keras",
        "type": "tcn",
    },
    "tft": {
        "model": POSE_MODELS_DIR / "tft_pose_correction_best.keras",
        "type": "tft",
    },
    "desd_pth": {
        "model": POSE_MODELS_DIR / "DESD_best_model.pth",
        "type": "desd_pth",
    },
}


class EnhancedPoseModel(nn.Module):
    """
    DESD PyTorch pose correction architecture:
    input: 37 (workout id + 12 joints * xyz)
    output: 36 (12 joints * xyz corrections)
    """

    def __init__(self, input_dim: int = 37, hidden_dim: int = 512, output_dim: int = 36):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x) * 0.1


class ModelRegistry:
    def __init__(self):
        self._workout_cache: Dict[str, WorkoutModelBundle] = {}
        self._pose_cache: Dict[str, PoseModelBundle] = {}
        self._torch_device = torch.device("cpu")
        self.pose_class_names = self._load_pose_class_names()
        self.pose_norm = self._load_pose_norm_stats()

    def _load_pose_class_names(self) -> List[str]:
        meta_path = REPO_ROOT / "Data" / "output_displacement" / "training_data_displacement_metadata.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return metadata["class_names"]

    def _load_pose_norm_stats(self) -> Dict[str, np.ndarray]:
        stats_path = POSE_MODELS_DIR / "tcn_film_preprocess_stats.npz"
        if stats_path.exists():
            stats = np.load(stats_path, allow_pickle=True)
            return {
                "x_mean": np.asarray(stats["X_mean"], dtype=np.float32),
                "x_std": np.asarray(stats["X_std"], dtype=np.float32),
                "y_mean": np.asarray(stats["y_mean"], dtype=np.float32).reshape(-1),
                "y_std": np.asarray(stats["y_std"], dtype=np.float32).reshape(-1),
            }
        # Safe fallback if stats file is missing.
        return {
            "x_mean": np.zeros((1, 1, 24), dtype=np.float32),
            "x_std": np.ones((1, 1, 24), dtype=np.float32),
            "y_mean": np.zeros((24,), dtype=np.float32),
            "y_std": np.ones((24,), dtype=np.float32),
        }

    def get_workout_model(self, key: str) -> WorkoutModelBundle:
        if key in self._workout_cache:
            return self._workout_cache[key]
        if key not in WORKOUT_MODEL_FILES:
            raise KeyError(f"Unknown workout model key: {key}")

        files = WORKOUT_MODEL_FILES[key]
        custom_objects = TRANSFORMER_CUSTOM_OBJECTS if key == "transformer" else None
        load_kwargs = {"compile": False, "custom_objects": custom_objects}
        if custom_objects and "safe_mode" in keras.models.load_model.__code__.co_varnames:
            load_kwargs["safe_mode"] = False
        model = keras.models.load_model(files["model"], **load_kwargs)

        with open(files["class_names"], "r", encoding="utf-8") as f:
            class_names = json.load(f)
        with open(files["norm"], "r", encoding="utf-8") as f:
            norm = json.load(f)
        mean = np.asarray(norm["mean"], dtype=np.float32).reshape(-1)
        std = np.asarray(norm["std"], dtype=np.float32).reshape(-1)
        std = np.where(std < 1e-8, 1.0, std)

        bundle = WorkoutModelBundle(model=model, class_names=class_names, mean=mean, std=std)
        self._workout_cache[key] = bundle
        return bundle

    def get_pose_model(self, key: str) -> PoseModelBundle:
        if key in self._pose_cache:
            return self._pose_cache[key]
        if key not in POSE_MODEL_FILES:
            raise KeyError(f"Unknown pose model key: {key}")

        files = POSE_MODEL_FILES[key]
        if files["type"] == "desd_pth":
            state_dict = torch.load(files["model"], map_location=self._torch_device)
            model = EnhancedPoseModel(input_dim=37, hidden_dim=512, output_dim=36).to(
                self._torch_device
            )
            model.load_state_dict(state_dict)
            model.eval()
            bundle = PoseModelBundle(model=model, model_type=files["type"])
            self._pose_cache[key] = bundle
            return bundle

        custom_objects = {**TFT_CUSTOM_OBJECTS, **LSTM_CUSTOM_OBJECTS}
        load_kw = {"compile": False, "custom_objects": custom_objects}
        if "safe_mode" in keras.models.load_model.__code__.co_varnames:
            load_kw["safe_mode"] = False

        model = keras.models.load_model(files["model"], **load_kw)
        bundle = PoseModelBundle(model=model, model_type=files["type"])
        self._pose_cache[key] = bundle
        return bundle
