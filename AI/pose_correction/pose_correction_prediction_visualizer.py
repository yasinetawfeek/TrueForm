#!/usr/bin/env python3
"""
Visualize pose-correction model predictions on the displacement NPZ.

Loads a Keras model trained by `LSTM_embedding_pose_correction.ipynb`,
`TFT_pose_correction.ipynb`, or `TCN_FiLM_pose_correction.ipynb`, applies the same preprocessing as those notebooks
(z-score X on the full dataset, standardize y; optional test-split filter),
runs predict, and draws the skeleton at a chosen frame with **two** arrow sets
in the x–y plane: ground-truth final displacement vs model prediction.

Layout and drawing mirror `Data/displacement_dataset_visualizer.py`.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "Data"
_POSE_CORRECTION_DIR = Path(__file__).resolve().parent
if str(_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_DIR))
if str(_POSE_CORRECTION_DIR) not in sys.path:
    sys.path.insert(0, str(_POSE_CORRECTION_DIR))

from sklearn.model_selection import train_test_split  # noqa: E402
from tensorflow import keras  # noqa: E402

from lstm_saved_model_objects import LSTM_CUSTOM_OBJECTS  # noqa: E402
from tft_saved_model_objects import TFT_CUSTOM_OBJECTS  # noqa: E402
from video_pose_extractor_displacement import LANDMARK_NAMES  # noqa: E402

# Match Data/displacement_dataset_visualizer.py
POSE_CONNECTIONS = [
    (0, 1),
    (0, 2),
    (2, 4),
    (1, 3),
    (3, 5),
    (0, 6),
    (1, 7),
    (6, 7),
    (6, 8),
    (8, 10),
    (7, 9),
    (9, 11),
]


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _unpack_regression_output(raw: Any, out_dim: int) -> np.ndarray:
    """Turn model output into a 1d vector of standardized targets (per-dim z-scores, not geometric z)."""
    out = raw
    if isinstance(out, dict):
        if "disp" in out:
            out = out["disp"]
        elif len(out) == 1:
            out = next(iter(out.values()))
        else:
            raise ValueError(
                f"Model returned multiple outputs {list(out.keys())}; "
                "expected one regression head or an output named 'disp'."
            )
    if isinstance(out, (list, tuple)):
        if len(out) != 1:
            raise ValueError(f"Expected a single model output tensor, got {len(out)}")
        out = out[0]
    arr = np.asarray(out, dtype=np.float64)
    arr = np.squeeze(arr)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    elif arr.ndim == 0:
        raise ValueError("Model output is scalar; expected a displacement vector.")
    if int(arr.shape[0]) != int(out_dim):
        raise ValueError(
            f"Model output length {arr.shape[0]} != dataset y dim {out_dim} "
            f"(raw output type={type(raw).__name__})"
        )
    return arr


def _print_debug_weight_digest(model: keras.Model) -> None:
    """Log max|weight| for a few tensors to spot unloaded / zero checkpoints."""
    print("[pose_correction_prediction_visualizer] weight digest (max |.| per variable, key layers):")

    def _arr(w: Any) -> np.ndarray:
        if hasattr(w, "numpy") and callable(getattr(w, "numpy")):
            return np.asarray(w.numpy(), dtype=np.float64)
        return np.asarray(w, dtype=np.float64)

    keys = ("disp", "head_1", "head_2", "temporal_proj", "lstm_1", "lstm_2")
    shown = 0
    for w in model.weights:
        path = str(getattr(w, "path", None) or getattr(w, "name", "") or "")
        if not any(k in path for k in keys):
            continue
        arr = _arr(w)
        print(f"  {path}: max|.| = {float(np.max(np.abs(arr))):.6g}")
        shown += 1
    if shown == 0:
        print("  (no name match; first 10 tensors:)")
        for w in list(model.weights)[:10]:
            path = str(getattr(w, "path", None) or getattr(w, "name", "") or repr(w))
            arr = _arr(w)
            print(f"  {path}: max|.| = {float(np.max(np.abs(arr))):.6g}")


def _tensorish_to_numpy(obj: Any) -> Any:
    """Recursively convert Keras/TF tensors in predict/__call__ outputs to NumPy."""
    if isinstance(obj, dict):
        return {k: _tensorish_to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_tensorish_to_numpy(v) for v in obj)
    if hasattr(obj, "numpy") and callable(getattr(obj, "numpy")):
        return obj.numpy()
    return obj


def _infer_model_kind(model: keras.Model) -> str:
    inputs = model.inputs
    if len(inputs) == 2:
        names = [getattr(i, "name", "") or "" for i in inputs]
        if "pose" in names and "class_id" in names:
            # Both LSTM and TCN-FiLM models have (pose, class_id) inputs.
            # Disambiguate by model/layer naming convention.
            mname = (getattr(model, "name", "") or "").lower()
            if "tcn" in mname:
                return "tcn"
            try:
                layer_names = [getattr(l, "name", "") or "" for l in model.layers]
            except Exception:
                layer_names = []
            if any(
                n.startswith("tcn_conv_")
                or n.startswith("film_dense_")
                or n.startswith("film_gamma_")
                or n.startswith("film_beta_")
                for n in layer_names
            ):
                return "tcn"
            return "lstm"
        dtypes = [getattr(i, "dtype", None) for i in inputs]
        if any(d is not None and "int" in str(d) for d in dtypes):
            return "lstm"
    if len(inputs) == 1:
        return "tft"
    raise ValueError(
        f"Unsupported model inputs ({len(inputs)}). "
        "Use --model-type lstm, tcn, or tft, or a saved pose-correction model."
    )


class PoseCorrectionPredictionVisualizer:
    """NPZ + Keras pose correction: skeleton + GT vs predicted displacement arrows."""

    def __init__(
        self,
        npz_path: str | Path,
        model_path: str | Path,
        metadata_path: Optional[str | Path] = None,
        model_type: str = "auto",
        test_split_only: bool = False,
        test_size: float = 0.2,
        random_state: int = 42,
        debug_predict: bool = False,
    ):
        self.npz_path = Path(npz_path)
        if not self.npz_path.is_file():
            raise FileNotFoundError(f"NPZ not found: {self.npz_path}")

        meta_file = (
            Path(metadata_path)
            if metadata_path
            else self.npz_path.with_name("training_data_displacement_metadata.json")
        )
        if not meta_file.is_file():
            raise FileNotFoundError(f"Metadata JSON not found: {meta_file}")
        self.metadata = _load_json(meta_file)

        bundle = np.load(self.npz_path, allow_pickle=True)
        self.X_raw = bundle["X"].astype(np.float32)
        self.y_raw = bundle["y"].astype(np.float32)
        self.y_class = bundle["y_class"].astype(np.int32)
        self.class_names = bundle["class_names"]

        self.N = int(self.X_raw.shape[0])
        self.T = int(self.metadata["sequence_length"])
        self.pose_flat_dim = int(self.metadata["X_pose_flat_dim"])
        self.class_dim = int(self.metadata["X_class_dim"])
        self.n_coords = int(self.metadata.get("n_coords", 3))
        self.n_landmarks = int(self.metadata["n_landmarks"])
        self.pose_feats_per_step = self.n_landmarks * self.n_coords

        if self.y_raw.shape[1] != self.n_landmarks * self.n_coords:
            raise ValueError(
                f"y dim {self.y_raw.shape[1]} != n_landmarks * n_coords "
                f"({self.n_landmarks * self.n_coords})"
            )

        X_pose_flat = self.X_raw[:, : self.pose_flat_dim]
        self.X_pose_seq = np.nan_to_num(
            X_pose_flat.reshape(self.N, self.T, self.pose_feats_per_step),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        y_clean = np.nan_to_num(self.y_raw, nan=0.0, posinf=0.0, neginf=0.0)

        X_class_feat = self.X_raw[:, self.pose_flat_dim : self.pose_flat_dim + self.class_dim]
        self.X_class_seq = np.repeat(X_class_feat[:, None, :], self.T, axis=1)

        # Full-dataset stats (same as notebooks — before train/test split).
        self.X_pose_mean = np.mean(self.X_pose_seq, axis=(0, 1), keepdims=True)
        self.X_pose_std = np.std(self.X_pose_seq, axis=(0, 1), keepdims=True) + 1e-8
        self.X_pose_norm = (self.X_pose_seq - self.X_pose_mean) / self.X_pose_std

        self.X_seq = np.concatenate([self.X_pose_seq, self.X_class_seq], axis=2)
        self.X_seq_mean = np.mean(self.X_seq, axis=(0, 1), keepdims=True)
        self.X_seq_std = np.std(self.X_seq, axis=(0, 1), keepdims=True) + 1e-8
        self.X_seq_norm = (self.X_seq - self.X_seq_mean) / self.X_seq_std

        self.y_mean = np.mean(y_clean, axis=0, keepdims=True)
        self.y_std = np.std(y_clean, axis=0, keepdims=True) + 1e-8
        self.y_standardized = (y_clean - self.y_mean) / self.y_std

        self._debug_predict = bool(debug_predict)
        self._debug_predict_done = False

        self._indices = np.arange(self.N, dtype=np.int64)
        if test_split_only:
            _train_idx, test_idx = train_test_split(
                self._indices,
                test_size=test_size,
                random_state=random_state,
                stratify=self.y_class,
            )
            self._indices = np.sort(test_idx.astype(np.int64))
            print(f"Using test split only: {len(self._indices)} / {self.N} samples.")

        # TFT checkpoints use Lambda + custom layers; Keras 3 needs safe_mode=False and
        # custom_objects (see tft_saved_model_objects). Do not use bare except TypeError:
        # deserialization errors are also TypeError and must not drop safe_mode=False.
        mp = Path(model_path)
        load_kw: Dict[str, Any] = {
            "compile": False,
            "custom_objects": {**dict(TFT_CUSTOM_OBJECTS), **dict(LSTM_CUSTOM_OBJECTS)},
        }
        if "safe_mode" in inspect.signature(keras.models.load_model).parameters:
            load_kw["safe_mode"] = False
        self.model = keras.models.load_model(mp, **load_kw)
        mt = (model_type or "auto").lower()
        if mt == "auto":
            self.model_kind = _infer_model_kind(self.model)
        elif mt in ("lstm", "tcn", "tft"):
            self.model_kind = mt
        else:
            raise ValueError("model_type must be auto, lstm, tcn, or tft")

        if self._debug_predict:
            _print_debug_weight_digest(self.model)

        self._y_pred_cache: Dict[int, np.ndarray] = {}
        self.n_samples = int(len(self._indices))
        self.current_pos = 0
        self.current_frame = self.T - 1

    def _global_index(self) -> int:
        return int(self._indices[self.current_pos])

    def _predict_one(self, global_idx: int) -> np.ndarray:
        if global_idx in self._y_pred_cache:
            return self._y_pred_cache[global_idx]

        out_dim = int(self.y_raw.shape[1])
        pose_n = self.X_pose_norm[global_idx : global_idx + 1].astype(np.float32, copy=False)
        yc = np.asarray([[int(self.y_class[global_idx])]], dtype=np.int32)

        if self.model_kind in ("lstm", "tcn"):
            feed: Dict[str, Any] = {"pose": pose_n, "class_id": yc}
            raw_out = _tensorish_to_numpy(self.model.predict(feed, verbose=0))
        else:
            # Match TFT_pose_correction.ipynb: single tensor, no dict (avoids Keras 3
            # "structure of inputs doesn't match" warnings and __call__ quirks).
            x_n = self.X_seq_norm[global_idx : global_idx + 1].astype(np.float32, copy=False)
            raw_out = _tensorish_to_numpy(self.model.predict(x_n, verbose=0))
        y_std = _unpack_regression_output(raw_out, out_dim)
        ym = np.asarray(self.y_mean, dtype=np.float64).reshape(-1)
        ys = np.asarray(self.y_std, dtype=np.float64).reshape(-1)
        y_hat = y_std * ys + ym

        if self._debug_predict and not self._debug_predict_done:
            self._debug_predict_done = True
            yt = np.asarray(self.y_raw[global_idx], dtype=np.float64).reshape(-1)
            baseline = ym.copy()
            mae_baseline = float(np.mean(np.abs(yt - baseline)))
            mae_pred = float(np.mean(np.abs(yt - y_hat)))
            print(
                "[pose_correction_prediction_visualizer] first prediction debug:\n"
                f"  model_kind={self.model_kind!r}  "
                f"{'predict(dict)' if self.model_kind in ('lstm', 'tcn') else 'predict(ndarray)'}\n"
                f"  standardized head out: min={y_std.min():.6f} max={y_std.max():.6f} "
                f"mean(abs)={np.mean(np.abs(y_std)):.6f}\n"
                f"  (Each of the {out_dim} outputs is (Δx/Δy per joint when n_coords=2); "
                f"\"standardized\" = (y-μ)/σ per dim over the dataset—not the pose z axis.)\n"
                f"  → head ≈ 0 in standardized units ⇒ predicts ~dataset mean per dim; "
                f"denorm ≈ y_mean (mean(abs(y_mean))={float(np.mean(np.abs(ym))):.8f}).\n"
                f"  denorm y_hat:          min={y_hat.min():.8f} max={y_hat.max():.8f} "
                f"mean(abs)={np.mean(np.abs(y_hat)):.8f}\n"
                f"  GT y (npz):            min={yt.min():.8f} max={yt.max():.8f} "
                f"mean(abs)={np.mean(np.abs(yt)):.8f}\n"
                f"  mean abs err (pred):   {mae_pred:.8f}\n"
                f"  mean abs err (z=0 baseline, y_hat=y_mean): {mae_baseline:.8f}\n"
                f"  (If these two MAEs match, the net is effectively outputting ~0 in standardized units "
                f"(≈constant y_mean)—not a matplotlib issue; check checkpoint, weight load, or evaluate() in the notebook.)"
            )
            yt_nc = yt.reshape(self.n_landmarks, self.n_coords)
            yh_nc = y_hat.reshape(self.n_landmarks, self.n_coords)
            ax_labels = ("x", "y", "z")[: self.n_coords]
            t_hdr = " ".join(f"t{a:>8}" for a in ax_labels)
            p_hdr = " ".join(f"p{a:>8}" for a in ax_labels)
            print("  per-joint displacement (denormalized, same units as npz y / plot arrows):")
            print(f"    {'#':>3}  {'name':<16}  {t_hdr}  {p_hdr}  {'mae':>8}")
            for j in range(self.n_landmarks):
                nm = (
                    LANDMARK_NAMES[j].replace("_", " ")
                    if j < len(LANDMARK_NAMES)
                    else f"joint_{j}"
                )[:16]
                tcols = " ".join(f"{float(yt_nc[j, c]):>9.6f}" for c in range(self.n_coords))
                pcols = " ".join(f"{float(yh_nc[j, c]):>9.6f}" for c in range(self.n_coords))
                mae_j = float(np.mean(np.abs(yt_nc[j] - yh_nc[j])))
                print(f"    {j:3d}  {nm:<16}  {tcols}  {pcols}  {mae_j:>8.6f}")

        self._y_pred_cache[global_idx] = y_hat
        return y_hat

    def pose_and_vectors(
        self, position: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Raw pose (T, 12, 3), true correction (12, 3), pred correction (12, 3)."""
        g = int(self._indices[int(position)])
        row = self.X_raw[g]
        pose_nc = row[: self.pose_flat_dim].reshape(self.T, self.n_landmarks, self.n_coords)
        y_true_flat = self.y_raw[g]
        y_true_nc = y_true_flat.reshape(self.n_landmarks, self.n_coords)

        y_pred_flat = self._predict_one(g)
        y_pred_nc = y_pred_flat.reshape(self.n_landmarks, self.n_coords)

        if self.n_coords == 2:
            pose = np.zeros((self.T, self.n_landmarks, 3), dtype=np.float64)
            pose[:, :, :2] = pose_nc.astype(np.float64, copy=False)
            yt = np.zeros((self.n_landmarks, 3), dtype=np.float64)
            yt[:, :2] = y_true_nc.astype(np.float64, copy=False)
            yp = np.zeros((self.n_landmarks, 3), dtype=np.float64)
            yp[:, :2] = y_pred_nc.astype(np.float64, copy=False)
            return pose, yt, yp

        return (
            pose_nc.astype(np.float64, copy=False),
            y_true_nc.astype(np.float64, copy=False),
            y_pred_nc.astype(np.float64, copy=False),
        )

    def class_label(self, position: int) -> str:
        g = int(self._indices[int(position)])
        ci = int(self.y_class[g])
        names = self.class_names.tolist()
        if 0 <= ci < len(names):
            return str(names[ci])
        return f"class_{ci}"

    def _y_class_at_view_position(self, view_pos: int) -> int:
        g = int(self._indices[int(view_pos)])
        return int(self.y_class[g])

    def _jump_to_adjacent_workout_type(self, direction: int) -> None:
        """Move current_pos to the nearest sample (along the viewer list) whose y_class differs (+1 forward, -1 back)."""
        if self.n_samples <= 1 or direction not in (-1, 1):
            return
        pos = int(self.current_pos)
        c0 = self._y_class_at_view_position(pos)
        if direction > 0:
            for step in range(pos + 1, self.n_samples):
                if self._y_class_at_view_position(step) != c0:
                    self.current_pos = step
                    self.current_frame = min(self.current_frame, self.T - 1)
                    return
        else:
            for step in range(pos - 1, -1, -1):
                if self._y_class_at_view_position(step) != c0:
                    self.current_pos = step
                    self.current_frame = min(self.current_frame, self.T - 1)
                    return

    def draw_frame(self, ax, position: int, frame_idx: int) -> None:
        pose, y_true, y_pred = self.pose_and_vectors(position)
        f = int(np.clip(frame_idx, 0, self.T - 1))

        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x (normalized)")
        ax.set_ylabel("y (normalized)")

        landmarks = pose[f]
        xs = landmarks[:, 0]
        ys = landmarks[:, 1]

        for a, b in POSE_CONNECTIONS:
            ax.plot([xs[a], xs[b]], [ys[a], ys[b]], "b-", linewidth=2, alpha=0.75, zorder=3)

        circle_r = 0.012
        for j in range(self.n_landmarks):
            ax.add_patch(
                Circle(
                    (xs[j], ys[j]),
                    circle_r,
                    facecolor=plt.cm.viridis(0.2 + 0.6 * j / max(self.n_landmarks - 1, 1)),
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=5,
                )
            )

        eps = 1e-9
        color_true = "#e63946"
        color_pred = "#2a9d8f"

        def _arrows(yvec: np.ndarray, color: str, z0: int) -> None:
            for j in range(self.n_landmarks):
                corr = yvec[j]
                dx, dy = float(corr[0]), float(corr[1])
                if float(np.linalg.norm(corr[:2])) <= eps:
                    continue
                if abs(dx) < eps and abs(dy) < eps:
                    continue
                x0, y0 = float(xs[j]), float(ys[j])
                arr = FancyArrowPatch(
                    (x0, y0),
                    (x0 + dx, y0 + dy),
                    arrowstyle="-|>",
                    mutation_scale=11,
                    linewidth=1.6,
                    color=color,
                    zorder=z0,
                    alpha=0.88,
                )
                ax.add_patch(arr)

        _arrows(y_true, color_true, z0=6)
        _arrows(y_pred, color_pred, z0=7)

        g = self._global_index()
        err = float(np.mean(np.abs(y_true[:, :2] - y_pred[:, :2])))
        ax.set_title(
            f"NPZ idx {g} | view {position + 1}/{self.n_samples} | "
            f"frame {f + 1}/{self.T} | {self.class_label(position)}\n"
            f"Red: GT displacement  |  Teal: predicted  |  mean |err| xy: {err:.5f}",
            fontsize=11,
        )

    def show_static(self, position: int, frame_idx: int) -> None:
        fig, ax = plt.subplots(figsize=(9, 9))
        self.draw_frame(ax, position, frame_idx)
        plt.tight_layout()
        plt.show()

    def animate_sample(self, position: int, interval_ms: int = 120) -> animation.Animation:
        fig, ax = plt.subplots(figsize=(9, 9))

        def step(t: int):
            self.draw_frame(ax, position, t)

        anim = animation.FuncAnimation(fig, step, frames=self.T, interval=interval_ms, repeat=True)
        plt.tight_layout()
        plt.show()
        return anim

    def interactive_viewer(self) -> None:
        fig, ax = plt.subplots(figsize=(9, 9))

        def refresh():
            self.draw_frame(ax, self.current_pos, self.current_frame)
            extra = (
                f"Model: {self.model_kind} | Samples in view: {self.n_samples} | "
                f"n_coords={self.n_coords}\n"
                "←/→ / a/d frame  ↑/↓ / w/s sample  shift+↑/↓ / shift+w/s next/prev workout type  "
                "m animate  q quit"
            )
            fig.suptitle(extra, fontsize=10, y=0.02)
            fig.canvas.draw_idle()

        def on_key(event):
            k = (event.key or "").lower()
            mods = getattr(event, "modifiers", None)
            shift_held = k.startswith("shift+") or (
                isinstance(mods, (set, frozenset, list, tuple))
                and any(str(m).lower() == "shift" for m in mods)
            )
            if shift_held:
                base = k.split("+")[-1] if "+" in k else k
                if base in ("up", "w"):
                    self._jump_to_adjacent_workout_type(+1)
                    refresh()
                    return
                if base in ("down", "s"):
                    self._jump_to_adjacent_workout_type(-1)
                    refresh()
                    return
            if event.key in ("right", "d"):
                self.current_frame = min(self.current_frame + 1, self.T - 1)
            elif event.key in ("left", "a"):
                self.current_frame = max(self.current_frame - 1, 0)
            elif event.key == "up" or event.key == "w":
                self.current_pos = min(self.current_pos + 1, self.n_samples - 1)
            elif event.key == "down" or event.key == "s":
                self.current_pos = max(self.current_pos - 1, 0)
            elif event.key == "m":
                plt.close(fig)
                self.animate_sample(self.current_pos)
                return
            elif event.key == "q":
                plt.close(fig)
                return
            refresh()

        fig.canvas.mpl_connect("key_press_event", on_key)
        refresh()
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08)
        plt.show()


def main() -> int:
    default_npz = _REPO_ROOT / "Data/output_displacement/training_data_displacement.npz"
    default_model = Path(__file__).resolve().parent / "models" / "lstm_embedding_pose_correction_best.keras"

    parser = argparse.ArgumentParser(
        description="Visualize pose-correction Keras predictions (GT vs pred arrows, x–y)."
    )
    parser.add_argument(
        "npz",
        type=str,
        nargs="?",
        default=str(default_npz),
        help="Path to training_data_displacement.npz",
    )
    parser.add_argument(
        "model",
        type=str,
        nargs="?",
        default=str(default_model),
        help="Path to .keras model (LSTM embedding or TFT)",
    )
    parser.add_argument(
        "-m",
        "--metadata",
        type=str,
        default=None,
        help="Path to training_data_displacement_metadata.json",
    )
    parser.add_argument(
        "--model-type",
        choices=("auto", "lstm", "tcn", "tft"),
        default="auto",
        help="How to call model.predict (default: infer from inputs)",
    )
    parser.add_argument(
        "--test-split-only",
        action="store_true",
        help="Restrict to sklearn test split (20 percent, stratified), same random_state=42 as notebooks",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Index within the viewer list (all NPZ rows, or test split if --test-split-only)",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=-1,
        help="Frame index (default: -1 = last)",
    )
    parser.add_argument(
        "--mode",
        choices=("static", "animate", "interactive"),
        default="interactive",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=120,
        help="Animation interval ms",
    )
    parser.add_argument(
        "--debug-predict",
        action="store_true",
        help="Print standardized-target vs denormalized prediction stats on the first forward pass",
    )

    args = parser.parse_args()
    vis = PoseCorrectionPredictionVisualizer(
        args.npz,
        args.model,
        metadata_path=args.metadata,
        model_type=args.model_type,
        test_split_only=args.test_split_only,
        debug_predict=args.debug_predict,
    )
    frame = args.frame
    if frame < 0:
        frame = vis.T - 1

    if args.mode == "static":
        vis.show_static(args.sample, frame)
    elif args.mode == "animate":
        vis.animate_sample(args.sample, interval_ms=args.interval)
    else:
        vis.current_pos = min(max(args.sample, 0), vis.n_samples - 1)
        vis.current_frame = frame
        vis.interactive_viewer()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
