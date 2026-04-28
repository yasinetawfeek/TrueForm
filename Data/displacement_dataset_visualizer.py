#!/usr/bin/env python3
"""
Visualize displacement training data (training_data_displacement.npz).

Shows the displaced pose from X. For each joint where the correction vector y
is non-zero, draws an arrow from the joint position toward the corrected
position (displaced + y) in the x–y plane.

With training_data_displacement_debug.json present, per-frame arrow length
follows the same linear ramp as the dataset generator. Without it, arrows
are drawn only on the last frame (full correction).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent
if str(_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_DIR))
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

from video_pose_extractor_displacement import LANDMARK_NAMES, NUM_LANDMARKS

# Match Data/pose_visualizer.py
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


def _load_debug_list(path: Path) -> Optional[List[Dict[str, Any]]]:
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return None
    return data


def _ramp_alpha(frame_idx: int, start: int, displacement_time: int) -> float:
    """Same schedule as video_pose_extractor_displacement._displacement_profile."""
    if frame_idx < start:
        return 0.0
    if frame_idx >= start + displacement_time:
        return 1.0
    ramp = np.linspace(0.0, 1.0, int(displacement_time), dtype=np.float64)
    return float(ramp[int(frame_idx - start)])


class DisplacementDatasetVisualizer:
    def __init__(
        self,
        npz_path: str | Path,
        metadata_path: Optional[str | Path] = None,
        debug_path: Optional[str | Path] = None,
    ):
        self.npz_path = Path(npz_path)
        if not self.npz_path.is_file():
            raise FileNotFoundError(f"NPZ not found: {self.npz_path}")

        bundle = np.load(self.npz_path, allow_pickle=True)
        self.X = bundle["X"]
        self.y = bundle["y"]
        self.y_class = bundle["y_class"]
        self.class_names = bundle["class_names"]

        meta_file = (
            Path(metadata_path)
            if metadata_path
            else self.npz_path.with_name("training_data_displacement_metadata.json")
        )
        if meta_file.is_file():
            self.metadata = _load_json(meta_file)
        else:
            self.metadata = {}

        row = int(self.X.shape[1])
        ydim = int(self.y.shape[1])

        if "n_coords" in bundle.files:
            self.n_coords = int(bundle["n_coords"])
        elif self.metadata.get("n_coords") is not None:
            self.n_coords = int(self.metadata["n_coords"])
        elif ydim % NUM_LANDMARKS == 0 and ydim // NUM_LANDMARKS in (2, 3):
            self.n_coords = ydim // NUM_LANDMARKS
        else:
            self.n_coords = 3

        if "sequence_length" in bundle.files:
            self.T = int(bundle["sequence_length"])
        else:
            self.T = int(self.metadata.get("sequence_length", 0))

        self.pose_flat_dim = int(self.metadata.get("X_pose_flat_dim", 0))
        self.class_dim = int(self.metadata.get("X_class_dim", 0))

        if self.pose_flat_dim <= 0 and self.T > 0:
            self.pose_flat_dim = self.T * NUM_LANDMARKS * self.n_coords
        if self.class_dim <= 0:
            self.class_dim = row - self.pose_flat_dim
        if self.T <= 0 and self.pose_flat_dim > 0 and self.n_coords > 0:
            self.T = self.pose_flat_dim // (NUM_LANDMARKS * self.n_coords)

        # Older NPZ (no n_coords / sequence_length) or mismatched metadata
        if (
            self.pose_flat_dim <= 0
            or self.T <= 0
            or self.pose_flat_dim + self.class_dim != row
            or self.T * NUM_LANDMARKS * self.n_coords != self.pose_flat_dim
        ):
            inferred_nc = (
                ydim // NUM_LANDMARKS
                if ydim % NUM_LANDMARKS == 0 and ydim // NUM_LANDMARKS in (2, 3)
                else self.n_coords
            )
            for nc in (inferred_nc, 3, 2):
                denom = NUM_LANDMARKS * nc
                whole = (row // denom) * denom
                cd = row - whole
                if whole > 0 and 1 <= cd <= row - 1 and whole % denom == 0:
                    self.n_coords = nc
                    self.pose_flat_dim = int(whole)
                    self.class_dim = int(cd)
                    self.T = int(whole // denom)
                    break

        if self.T * NUM_LANDMARKS * self.n_coords != self.pose_flat_dim:
            raise ValueError(
                f"Inconsistent pose layout: T={self.T}, n_coords={self.n_coords}, "
                f"pose_flat_dim={self.pose_flat_dim}"
            )
        if self.pose_flat_dim + self.class_dim != row:
            raise ValueError(
                f"X row length mismatch: pose_flat_dim={self.pose_flat_dim}, "
                f"class_dim={self.class_dim}, X.shape[1]={row}"
            )

        dbg = (
            Path(debug_path)
            if debug_path
            else self.npz_path.with_name("training_data_displacement_debug.json")
        )
        self.debug_entries = _load_debug_list(dbg)
        if self.debug_entries is not None and len(self.debug_entries) != len(self.X):
            print(
                f"Warning: debug JSON length {len(self.debug_entries)} != "
                f"NPZ samples {len(self.X)}; per-frame ramp disabled."
            )
            self.debug_entries = None

        self.n_samples = int(self.X.shape[0])
        self.current_idx = 0
        self.current_frame = 0

    def pose_and_y(self, sample_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Displaced pose (T, 12, 3) and correction y (12, 3); z is 0 when n_coords==2."""
        row = self.X[int(sample_idx)]
        pose_nc = row[: self.pose_flat_dim].reshape(self.T, NUM_LANDMARKS, self.n_coords)
        y_nc = self.y[int(sample_idx)].reshape(NUM_LANDMARKS, self.n_coords)
        if self.n_coords == 2:
            pose = np.zeros((self.T, NUM_LANDMARKS, 3), dtype=np.float64)
            pose[:, :, :2] = pose_nc.astype(np.float64, copy=False)
            yvec = np.zeros((NUM_LANDMARKS, 3), dtype=np.float64)
            yvec[:, :2] = y_nc.astype(np.float64, copy=False)
            return pose, yvec
        return pose_nc.astype(np.float64), y_nc.astype(np.float64)

    def class_label(self, sample_idx: int) -> str:
        ci = int(self.y_class[int(sample_idx)])
        names = self.class_names.tolist()
        if 0 <= ci < len(names):
            return str(names[ci])
        return f"class_{ci}"

    def _correction_scale(
        self, sample_idx: int, frame_idx: int, joint_idx: int, y_j: np.ndarray
    ) -> np.ndarray:
        """Correction vector (dx,dy,dz) for this frame (may be zero)."""
        if self.debug_entries is None:
            if frame_idx == self.T - 1:
                return y_j.copy()
            return np.zeros(3, dtype=np.float64)

        entry = self.debug_entries[int(sample_idx)]
        for jm in entry.get("displaced_joints", []):
            if int(jm.get("joint_index", -1)) != joint_idx:
                continue
            a = _ramp_alpha(
                frame_idx,
                int(jm["start_frame_index"]),
                int(jm["displacement_time"]),
            )
            return (a * y_j).astype(np.float64)
        return np.zeros(3, dtype=np.float64)

    def draw_frame(self, ax, sample_idx: int, frame_idx: int) -> None:
        pose, y_full = self.pose_and_y(sample_idx)
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
        for j in range(NUM_LANDMARKS):
            ax.add_patch(
                Circle(
                    (xs[j], ys[j]),
                    circle_r,
                    facecolor=plt.cm.viridis(0.2 + 0.6 * j / max(NUM_LANDMARKS - 1, 1)),
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=5,
                )
            )

        eps = 1e-9
        arrow_color = "#e63946"
        for j in range(NUM_LANDMARKS):
            corr = self._correction_scale(sample_idx, f, j, y_full[j])
            if float(np.linalg.norm(corr)) <= eps:
                continue
            # Arrow in x–y plane toward corrected pose
            dx, dy = float(corr[0]), float(corr[1])
            if abs(dx) < eps and abs(dy) < eps:
                continue
            x0, y0 = float(xs[j]), float(ys[j])
            arr = FancyArrowPatch(
                (x0, y0),
                (x0 + dx, y0 + dy),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.8,
                color=arrow_color,
                zorder=6,
                alpha=0.9,
            )
            ax.add_patch(arr)
            ax.text(
                x0 + dx * 0.5,
                y0 + dy * 0.5 - 0.03,
                LANDMARK_NAMES[j].replace("_", " ")[:14],
                fontsize=7,
                color=arrow_color,
                ha="center",
                zorder=7,
            )

        mode = "ramp (debug)" if self.debug_entries else "last-frame only"
        ax.set_title(
            f"Sample {sample_idx} | frame {f + 1}/{self.T} | {self.class_label(sample_idx)}\n"
            f"Arrows: correction in x–y ({mode})",
            fontsize=11,
        )

    def show_static(self, sample_idx: int, frame_idx: int) -> None:
        fig, ax = plt.subplots(figsize=(9, 9))
        self.draw_frame(ax, sample_idx, frame_idx)
        plt.tight_layout()
        plt.show()

    def animate_sample(self, sample_idx: int, interval_ms: int = 120) -> animation.Animation:
        fig, ax = plt.subplots(figsize=(9, 9))

        def step(t: int):
            self.draw_frame(ax, sample_idx, t)

        anim = animation.FuncAnimation(
            fig, step, frames=self.T, interval=interval_ms, repeat=True
        )
        plt.tight_layout()
        plt.show()
        return anim

    def interactive_viewer(self) -> None:
        fig, ax = plt.subplots(figsize=(9, 9))

        def refresh():
            self.draw_frame(ax, self.current_idx, self.current_frame)
            extra = (
                f"Samples: {self.n_samples} | n_coords={self.n_coords} | "
                f"Debug: {'yes' if self.debug_entries else 'no'}\n"
                "←/→ / a/d frame  ↑/↓ / w/s sample  m animate  q quit"
            )
            fig.suptitle(extra, fontsize=10, y=0.02)
            fig.canvas.draw_idle()

        def on_key(event):
            if event.key in ("right", "d"):
                self.current_frame = min(self.current_frame + 1, self.T - 1)
            elif event.key in ("left", "a"):
                self.current_frame = max(self.current_frame - 1, 0)
            elif event.key == "up" or event.key == "w":
                self.current_idx = min(self.current_idx + 1, self.n_samples - 1)
                self.current_frame = min(self.current_frame, self.T - 1)
            elif event.key == "down" or event.key == "s":
                self.current_idx = max(self.current_idx - 1, 0)
            elif event.key == "m":
                plt.close(fig)
                self.animate_sample(self.current_idx)
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
    parser = argparse.ArgumentParser(
        description="Visualize displacement dataset with correction arrows (x–y plane)."
    )
    parser.add_argument(
        "npz",
        type=str,
        help="Path to training_data_displacement.npz",
    )
    parser.add_argument(
        "-m",
        "--metadata",
        type=str,
        default=None,
        help="Path to training_data_displacement_metadata.json (optional)",
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str,
        default=None,
        help="Path to training_data_displacement_debug.json (optional, enables per-frame ramp)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Sample index (default: 0)",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=-1,
        help="Frame index (default: -1 = last frame)",
    )
    parser.add_argument(
        "--mode",
        choices=("static", "animate", "interactive"),
        default="interactive",
        help="View mode (default: interactive)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=120,
        help="Animation interval ms (default: 120)",
    )

    args = parser.parse_args()
    vis = DisplacementDatasetVisualizer(
        args.npz,
        metadata_path=args.metadata,
        debug_path=args.debug,
    )
    frame = args.frame
    if frame < 0:
        frame = vis.T - 1

    if args.mode == "static":
        vis.show_static(args.sample, frame)
    elif args.mode == "animate":
        vis.animate_sample(args.sample, interval_ms=args.interval)
    else:
        vis.current_idx = args.sample
        vis.current_frame = frame
        vis.interactive_viewer()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
