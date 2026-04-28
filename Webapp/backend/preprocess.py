from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

# Keep landmark ordering aligned with TrueForm training pipeline.
BODY_KEYPOINTS_INDICES: List[int] = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
WORKOUT_SEQUENCE_LEN = 15


def flatten_landmarks(landmarks: List[Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert MediaPipe landmarks to flat vectors for model inference.

    Returns:
        - xyz_flat: shape (36,) for workout classifiers
        - xy_flat: shape (24,) for pose-correction models
    """
    xyz_flat: List[float] = []
    xy_flat: List[float] = []

    for idx in BODY_KEYPOINTS_INDICES:
        lm = landmarks[idx] if idx < len(landmarks) else {}
        x = float(lm.get("x", 0.0))
        y = float(lm.get("y", 0.0))
        z = float(lm.get("z", 0.0))
        xyz_flat.extend([x, y, z])
        xy_flat.extend([x, y])

    return np.asarray(xyz_flat, dtype=np.float32), np.asarray(xy_flat, dtype=np.float32)


@dataclass
class ClientState:
    workout_sequence: Deque[np.ndarray] = field(
        default_factory=lambda: deque(maxlen=WORKOUT_SEQUENCE_LEN)
    )
    pose_sequence_xy: Deque[np.ndarray] = field(
        default_factory=lambda: deque(maxlen=WORKOUT_SEQUENCE_LEN)
    )
    last_workout_name: Optional[str] = None
    workout_history: Deque[str] = field(default_factory=lambda: deque(maxlen=5))


def update_client_sequence(
    state: ClientState, xyz_flat: np.ndarray, xy_flat: np.ndarray
) -> None:
    state.workout_sequence.append(xyz_flat)
    state.pose_sequence_xy.append(xy_flat)


def has_full_sequence(state: ClientState) -> bool:
    return (
        len(state.workout_sequence) >= WORKOUT_SEQUENCE_LEN
        and len(state.pose_sequence_xy) >= WORKOUT_SEQUENCE_LEN
    )


def sequence_to_array(sequence: Deque[np.ndarray]) -> np.ndarray:
    return np.asarray(list(sequence), dtype=np.float32)


def smooth_workout_prediction(state: ClientState, predicted_name: str) -> str:
    state.workout_history.append(predicted_name)
    counts: Dict[str, int] = {}
    for name in state.workout_history:
        counts[name] = counts.get(name, 0) + 1
    stable = max(counts.items(), key=lambda item: item[1])[0]
    state.last_workout_name = stable
    return stable


def build_correction_dict(correction: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Convert correction vectors into socket payload format.
    Supports:
      - 24 dims: 12 joints x (x,y)
      - 36 dims: 12 joints x (x,y,z)
    """
    output: Dict[str, Dict[str, float]] = {}
    vec = np.asarray(correction, dtype=np.float32).reshape(-1)
    if vec.size not in (24, 36):
        raise ValueError(f"Expected correction size 24 or 36, got {vec.size}")

    stride = 3 if vec.size == 36 else 2
    for i, landmark_idx in enumerate(BODY_KEYPOINTS_INDICES):
        j = i * stride
        output[str(landmark_idx)] = {
            "x": float(vec[j]),
            "y": float(vec[j + 1]),
            "z": float(vec[j + 2]) if stride == 3 else 0.0,
        }
    return output


def class_name_to_pose_class_id(class_names: List[str], workout_name: str) -> int:
    """
    Map workout class label string to class index used by pose-correction training metadata.
    """
    normalized = workout_name.strip().lower()
    # Remove optional UI suffix such as "(warming up x/15)".
    if "(" in normalized:
        normalized = normalized.split("(", 1)[0].strip()
    for idx, class_name in enumerate(class_names):
        if class_name.strip().lower() == normalized:
            return idx
    # Fallback keeps server resilient if labels differ slightly.
    return 0


def prepare_sequence(raw_sequence: List[List[float]], feature_dim: int, target_len: int = 15) -> np.ndarray:
    """
    Convert incoming sequence payload to (target_len, feature_dim) float32.
    Pads with the last frame when shorter; keeps the latest frames when longer.
    """
    arr = np.asarray(raw_sequence, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != feature_dim:
        raise ValueError(f"Expected sequence shape (N, {feature_dim}), got {arr.shape}")

    if arr.shape[0] == 0:
        raise ValueError("Empty sequence payload")
    if arr.shape[0] < target_len:
        pad_n = target_len - arr.shape[0]
        pad = np.repeat(arr[-1:, :], pad_n, axis=0)
        arr = np.concatenate([arr, pad], axis=0)
    elif arr.shape[0] > target_len:
        arr = arr[-target_len:, :]
    return arr
