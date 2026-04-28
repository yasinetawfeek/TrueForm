from __future__ import annotations

import os
import time
from typing import Any, Dict

import numpy as np
import torch
from flask import Flask
from flask_socketio import SocketIO, emit

from model_registry import ModelRegistry
from preprocess import (
    build_correction_dict,
    class_name_to_pose_class_id,
    prepare_sequence,
)

app = Flask(__name__)
# Let python-socketio select a compatible async mode for the runtime env.
socketio = SocketIO(app, cors_allowed_origins="*")
registry = ModelRegistry()

DEFAULT_WORKOUT_MODEL = "bilstm"
DEFAULT_POSE_MODEL = "tcn_film"
MAX_PACKET_AGE_MS = 1200
BACKEND_HOST = os.getenv("AI_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("AI_PORT", "8001"))


def _preload_default_models() -> None:
    """
    Warm model cache at startup to avoid long first-response delay on websocket frames.
    """
    registry.get_workout_model(DEFAULT_WORKOUT_MODEL)
    registry.get_pose_model(DEFAULT_POSE_MODEL)


def _unpack_prediction(raw: Any) -> np.ndarray:
    out = raw
    if isinstance(out, dict):
        if "disp" in out:
            out = out["disp"]
        elif len(out) == 1:
            out = next(iter(out.values()))
        else:
            raise ValueError(f"Unexpected multi-output prediction: {list(out.keys())}")
    if isinstance(out, (list, tuple)):
        out = out[0]
    arr = np.asarray(out, dtype=np.float32).reshape(-1)
    return arr


def _predict_workout(workout_key: str, sequence_xyz: np.ndarray) -> tuple[int, str, float]:
    bundle = registry.get_workout_model(workout_key)
    x = (sequence_xyz - bundle.mean) / bundle.std
    x = np.expand_dims(x, axis=0)
    probs = np.asarray(bundle.model(x, training=False))[0]
    idx = int(np.argmax(probs))
    confidence = float(probs[idx])
    return idx, bundle.class_names[idx], confidence


def _predict_pose_correction(pose_key: str, sequence_xy: np.ndarray, workout_name: str) -> np.ndarray:
    bundle = registry.get_pose_model(pose_key)
    pose_class_id = class_name_to_pose_class_id(registry.pose_class_names, workout_name)
    if bundle.model_type == "desd_pth":
        # DESD model expects a single 37-dim vector:
        # [workout_class_id, latest_frame_xyz(36)]
        raise ValueError("DESD model requires xyz sequence input")

    seq = np.expand_dims(sequence_xy, axis=0)  # (1, 15, 24)
    x_mean = registry.pose_norm["x_mean"]
    x_std = registry.pose_norm["x_std"]
    seq_norm = (seq - x_mean) / x_std

    if bundle.model_type in {"lstm", "tcn"}:
        model_input = {
            "pose": seq_norm.astype(np.float32),
            "class_id": np.asarray([[pose_class_id]], dtype=np.int32),
        }
    else:
        n_classes = len(registry.pose_class_names)
        class_onehot = np.zeros((1, seq.shape[1], n_classes), dtype=np.float32)
        class_onehot[:, :, pose_class_id] = 1.0
        model_input = np.concatenate([seq_norm.astype(np.float32), class_onehot], axis=-1)

    raw_out = bundle.model(model_input, training=False)
    pred_std = _unpack_prediction(raw_out)
    y_mean = registry.pose_norm["y_mean"]
    y_std = registry.pose_norm["y_std"]
    return pred_std * y_std + y_mean


@socketio.on("connect")
def on_connect():
    emit(
        "connected",
        {
            "client_id": request_sid(),
            "available_models": {
                "workout": ["bilstm", "gru", "transformer"],
                "pose_correction": ["lstm_embedding", "tcn_film", "tft", "desd_pth"],
            },
        },
    )


@socketio.on("connect", namespace="/classifier")
def on_classifier_connect():
    emit("connected", {"client_id": request_sid(), "namespace": "classifier"})


@socketio.on("connect", namespace="/correction")
def on_correction_connect():
    emit("connected", {"client_id": request_sid(), "namespace": "correction"})


@socketio.on("disconnect", namespace="/classifier")
def on_classifier_disconnect():
    return None


@socketio.on("disconnect", namespace="/correction")
def on_correction_disconnect():
    return None


def request_sid() -> str:
    from flask import request

    return str(request.sid)


@socketio.on("classify_sequence", namespace="/classifier")
def on_classify_sequence(data):
    try:
        sent_ts = int(data.get("timestamp", 0) or 0)
        if sent_ts and (int(time.time() * 1000) - sent_ts > MAX_PACKET_AGE_MS):
            emit(
                "workout_prediction",
                {"dropped": True, "reason": "stale_packet"},
                namespace="/classifier",
            )
            return

        sequence_xyz = prepare_sequence(data.get("sequence_xyz", []), feature_dim=36, target_len=15)
        workout_model_key = data.get("selected_workout_model", DEFAULT_WORKOUT_MODEL)
        workout_idx, workout_name, confidence = _predict_workout(workout_model_key, sequence_xyz)
        emit(
            "workout_prediction",
            {
                "timestamp": int(time.time() * 1000),
                "source_timestamp": sent_ts,
                "predicted_workout_type": workout_idx,
                "predicted_workout_name": workout_name,
                "confidence": confidence,
                "selected_workout_model": workout_model_key,
            },
            namespace="/classifier",
        )
    except Exception as exc:
        emit("error", {"message": f"Classifier inference error: {exc}"}, namespace="/classifier")


@socketio.on("correct_sequence", namespace="/correction")
def on_correct_sequence(data):
    try:
        sent_ts = int(data.get("timestamp", 0) or 0)
        if sent_ts and (int(time.time() * 1000) - sent_ts > MAX_PACKET_AGE_MS):
            emit(
                "pose_corrections",
                {"corrections": {}, "dropped": True, "reason": "stale_packet"},
                namespace="/correction",
            )
            return

        sequence_xy = prepare_sequence(data.get("sequence_xy", []), feature_dim=24, target_len=15)
        pose_model_key = data.get("selected_pose_model", DEFAULT_POSE_MODEL)
        workout_name = data.get("workout_name")
        if not workout_name:
            emit("pose_corrections", {"corrections": {}}, namespace="/correction")
            return

        if pose_model_key == "desd_pth":
            sequence_xyz = prepare_sequence(data.get("sequence_xyz", []), feature_dim=36, target_len=15)
            pose_class_id = class_name_to_pose_class_id(registry.pose_class_names, workout_name)
            latest_xyz = sequence_xyz[-1]
            input_vec = np.concatenate(
                [np.asarray([float(pose_class_id)], dtype=np.float32), latest_xyz.astype(np.float32)],
                axis=0,
            )
            input_tensor = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                raw = registry.get_pose_model("desd_pth").model(input_tensor)
            correction_vec = np.asarray(raw.squeeze(0).cpu().numpy(), dtype=np.float32).reshape(-1)
        else:
            correction_vec = _predict_pose_correction(pose_model_key, sequence_xy, workout_name)
        emit(
            "pose_corrections",
            {
                "timestamp": int(time.time() * 1000),
                "source_timestamp": sent_ts,
                "selected_pose_model": pose_model_key,
                "predicted_workout_name": workout_name,
                "corrections": build_correction_dict(correction_vec),
                "predicted_muscle_group": 0,
            },
            namespace="/correction",
        )
    except Exception as exc:
        emit("error", {"message": f"Pose correction inference error: {exc}"}, namespace="/correction")


@app.route("/")
def health():
    return "TrueForm Webapp realtime socket server is running.", 200


if __name__ == "__main__":
    _preload_default_models()
    socketio.run(
        app,
        host=BACKEND_HOST,
        port=BACKEND_PORT,
        debug=False,
        use_reloader=False,
        allow_unsafe_werkzeug=True,
    )
