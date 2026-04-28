#!/usr/bin/env python3
"""
Video Pose Extractor + Displacement Dataset Generator

This script mirrors `video_pose_extractor.py` for pose extraction and sequence creation,
but generates a supervised dataset where:

- X: flattened pose sequence (T * 12 * C) + workout class feature(s), C=3 or C=2 if --drop-z
- y: per-sequence negative final displacement (12 * C)

Displacement generation (per sequence):
- Randomly choose N joints to displace
- For each chosen joint: sample (dx, dy, dz), a displacement_time (2..T-1), and a start frame
- Apply a uniform (linear) ramp over displacement_time frames, then keep the final offset
- y is the negative of the final displacement for each joint; non-selected joints remain 0

With --drop-z, z is omitted from saved X and y (C=2: T*12*2 + class, y length 24); depth is not stored.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence as TypingSequence, Tuple

import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Keep the landmark ordering identical to `video_pose_extractor.py` (12 core body landmarks).
# We duplicate these constants here so this script can still run `--help` (or generate
# datasets from existing extracted JSON) even if MediaPipe/TensorFlow deps aren't available.
LANDMARK_NAMES = [
    "left_shoulder",   # 0  (was 11)
    "right_shoulder",  # 1  (was 12)
    "left_elbow",      # 2  (was 13)
    "right_elbow",     # 3  (was 14)
    "left_wrist",      # 4  (was 15)
    "right_wrist",     # 5  (was 16)
    "left_hip",        # 6  (was 23)
    "right_hip",       # 7  (was 24)
    "left_knee",       # 8  (was 25)
    "right_knee",      # 9  (was 26)
    "left_ankle",      # 10 (was 27)
    "right_ankle",     # 11 (was 28)
]
NUM_LANDMARKS = 12


def extract_pose_features_from_sequence(sequence: Dict, *, drop_z: bool = False) -> np.ndarray:
    """
    Extract pose features from a saved sequence dict into (T, 12, 3) float32.
    This mirrors `VideoPoseExtractor.extract_pose_features` but avoids importing MediaPipe.

    If drop_z is True, every landmark z coordinate is set to 0 (array shape stays (T, 12, 3)).
    """
    frames = sequence.get("frames", [])
    num_frames = len(frames)
    pose_array = np.zeros((num_frames, NUM_LANDMARKS, 3), dtype=np.float32)

    for frame_idx, frame in enumerate(frames):
        pose = frame.get("pose") or {}
        if pose.get("detected") and pose.get("landmarks"):
            for landmark_idx, landmark in enumerate(pose["landmarks"]):
                if landmark_idx >= NUM_LANDMARKS:
                    break
                pose_array[frame_idx, landmark_idx, 0] = float(landmark.get("x", 0.0))
                pose_array[frame_idx, landmark_idx, 1] = float(landmark.get("y", 0.0))
                z = 0.0 if drop_z else float(landmark.get("z", 0.0))
                pose_array[frame_idx, landmark_idx, 2] = z

    return pose_array


def _displacement_profile(
    *,
    num_frames: int,
    rng: np.random.Generator,
    displaced_joint_indices: np.ndarray,
    max_abs_displacement_xyz: Tuple[float, float, float],
    displacement_time_min: int,
    displacement_time_max: int,
    drop_z: bool = False,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Returns:
      - applied_disp: (T, 12, 3) float32 displacement applied to the pose features
      - metadata list describing per-joint displacement params
    """
    T = int(num_frames)
    if T < 3:
        raise ValueError(f"Sequence must have at least 3 frames, got {T}")

    disp = np.zeros((T, NUM_LANDMARKS, 3), dtype=np.float32)
    meta: List[Dict] = []

    # Clamp time bounds to valid range [2, T-1]
    t_min = max(2, int(displacement_time_min))
    t_max = min(T - 1, int(displacement_time_max))
    if t_min > t_max:
        t_min, t_max = 2, max(2, T - 1)

    max_dx, max_dy, max_dz = (float(max_abs_displacement_xyz[0]),
                              float(max_abs_displacement_xyz[1]),
                              float(max_abs_displacement_xyz[2]))

    for j in displaced_joint_indices.tolist():
        # Sample final displacement vector.
        dx = float(rng.uniform(-max_dx, max_dx))
        dy = float(rng.uniform(-max_dy, max_dy))
        dz = 0.0 if drop_z else float(rng.uniform(-max_dz, max_dz))
        disp_vec = np.array([dx, dy, dz], dtype=np.float32)

        displacement_time = int(rng.integers(t_min, t_max + 1))
        start = int(rng.integers(0, T - displacement_time + 1))
        end_exclusive = start + displacement_time

        # Linear ramp from 0 -> disp_vec over `displacement_time` frames.
        # At frame start: 0. At frame end_exclusive-1: disp_vec.
        if displacement_time == 1:
            ramp = np.ones((1,), dtype=np.float32)
        else:
            ramp = np.linspace(0.0, 1.0, displacement_time, dtype=np.float32)

        disp[start:end_exclusive, j, :] = ramp[:, None] * disp_vec[None, :]
        if end_exclusive < T:
            disp[end_exclusive:, j, :] = disp_vec[None, :]

        meta.append(
            {
                "joint_index": int(j),
                "joint_name": LANDMARK_NAMES[int(j)] if 0 <= int(j) < len(LANDMARK_NAMES) else str(j),
                "dx": dx,
                "dy": dy,
                "dz": dz,
                "displacement_time": int(displacement_time),
                "start_frame_index": int(start),
            }
        )

    return disp, meta


def _encode_class_feature(
    *,
    class_index: int,
    n_classes: int,
    encoding: str,
) -> np.ndarray:
    if encoding == "int":
        return np.array([float(class_index)], dtype=np.float32)
    if encoding == "onehot":
        v = np.zeros((n_classes,), dtype=np.float32)
        v[int(class_index)] = 1.0
        return v
    raise ValueError(f"Unknown class encoding: {encoding}")


def _parse_int_range(value: str) -> Tuple[int, int]:
    """
    Parse either:
      - "k" -> (k, k)
      - "a-b" -> (a, b)
    """
    s = str(value).strip()
    if not s:
        raise ValueError("empty range")
    if "-" in s:
        parts = [p.strip() for p in s.split("-", 1)]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"invalid range: {value}")
        lo = int(parts[0])
        hi = int(parts[1])
    else:
        lo = hi = int(s)
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def save_displacement_training_data_npz(
    *,
    sequences: List[Dict],
    workout_classes: List[str],
    output_dir: Path,
    displaced_joints_n_range: Tuple[int, int],
    max_abs_displacement_xyz: Tuple[float, float, float],
    displacement_time_min: int,
    displacement_time_max: int,
    class_encoding: str,
    random_seed: Optional[int],
    save_debug_metadata: bool,
    drop_z: bool = False,
) -> str:
    if not sequences:
        raise ValueError("No sequences to save")

    # Label encoding (pure Python/NumPy; avoids scikit-learn dependency)
    y_raw = np.array([seq.get("workout_class", "") for seq in sequences], dtype=object)
    if np.any(y_raw == ""):
        missing = [seq.get("sequence_number", "unknown") for seq in sequences if not seq.get("workout_class")]
        raise ValueError(f"Some sequences are missing workout_class: {missing[:10]}")

    class_names = sorted({str(x) for x in y_raw.tolist()})
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    y_class_idx = np.array([class_to_idx[str(x)] for x in y_raw.tolist()], dtype=np.int64)
    n_classes = int(len(class_names))

    # Generate X (displaced pose + class feature) and y (negative final displacement per joint)
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    displacement_debug: List[Dict] = []

    base_rng = np.random.default_rng(random_seed)

    for i, seq in enumerate(sequences):
        features = extract_pose_features_from_sequence(seq, drop_z=drop_z)  # (T, 12, 3)
        T = int(features.shape[0])

        # Sequence-specific RNG for reproducibility while still varying.
        seq_number = int(seq.get("sequence_number", i))
        seq_rng = np.random.default_rng(base_rng.integers(0, np.iinfo(np.int64).max) ^ seq_number)

        lo_n, hi_n = displaced_joints_n_range
        lo_n = max(0, min(NUM_LANDMARKS, int(lo_n)))
        hi_n = max(0, min(NUM_LANDMARKS, int(hi_n)))
        if lo_n > hi_n:
            lo_n, hi_n = hi_n, lo_n

        # Randomize how many joints are displaced for this sequence.
        # rng.integers high is exclusive; +1 makes it inclusive.
        n = int(seq_rng.integers(lo_n, hi_n + 1)) if hi_n > lo_n else int(lo_n)
        if n == 0:
            displaced_joint_indices = np.array([], dtype=np.int64)
        else:
            displaced_joint_indices = seq_rng.choice(NUM_LANDMARKS, size=n, replace=False)

        applied_disp, meta = _displacement_profile(
            num_frames=T,
            rng=seq_rng,
            displaced_joint_indices=displaced_joint_indices,
            max_abs_displacement_xyz=max_abs_displacement_xyz,
            displacement_time_min=displacement_time_min,
            displacement_time_max=displacement_time_max,
            drop_z=drop_z,
        )

        displaced_features = (features + applied_disp).astype(np.float32, copy=False)
        # Final displacement applied to each joint at the end of the sequence.
        # y is the negative of that final displacement (shape: 12x3 internally).
        final_disp = applied_disp[-1, :, :] if T > 0 else np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
        y_final = (-final_disp).astype(np.float32, copy=False)

        if drop_z:
            x_flat = displaced_features[:, :, :2].reshape(-1)  # (T*12*2,)
            y_flat = y_final[:, :2].reshape(-1)  # (12*2,)
        else:
            x_flat = displaced_features.reshape(-1)  # (T*12*3,)
            y_flat = y_final.reshape(-1)  # (12*3,)

        class_feat = _encode_class_feature(
            class_index=int(y_class_idx[i]),
            n_classes=n_classes,
            encoding=class_encoding,
        )

        X_list.append(np.concatenate([x_flat, class_feat], axis=0))
        y_list.append(y_flat)

        if save_debug_metadata:
            displacement_debug.append(
                {
                    "sequence_number": int(seq.get("sequence_number", i)),
                    "video_name": seq.get("video_name"),
                    "workout_class": str(seq.get("workout_class")),
                    "class_index": int(y_class_idx[i]),
                    "displaced_joints_n": int(n),
                    "displaced_joints": meta,
                }
            )

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.stack(y_list, axis=0).astype(np.float32)

    y_onehot = np.eye(n_classes, dtype=np.float32)[y_class_idx]

    T0 = int(extract_pose_features_from_sequence(sequences[0], drop_z=drop_z).shape[0])
    n_coords_eff = 2 if drop_z else 3

    npz_path = output_dir / "training_data_displacement.npz"
    np.savez_compressed(
        npz_path,
        X=X,
        y=y,
        y_class=y_class_idx.astype(np.int64),
        y_class_onehot=y_onehot,
        y_class_raw=y_raw.astype(object),
        class_names=np.array(class_names, dtype=object),
        n_landmarks=np.int64(NUM_LANDMARKS),
        n_coords=np.int64(n_coords_eff),
        sequence_length=np.int64(T0),
    )

    # Metadata JSON
    unique_classes, counts = np.unique(y_raw, return_counts=True)
    class_distribution = {str(cls): int(count) for cls, count in zip(unique_classes, counts)}

    class_dim = 1 if class_encoding == "int" else n_classes

    metadata = {
        "dataset": "displacement",
        "n_samples": int(X.shape[0]),
        "sequence_length": int(T0),
        "n_landmarks": int(NUM_LANDMARKS),
        "n_coords": int(n_coords_eff),
        "X_shape": [int(d) for d in X.shape],
        "y_shape": [int(d) for d in y.shape],
        "X_pose_flat_dim": int(T0 * NUM_LANDMARKS * n_coords_eff),
        "X_class_dim": int(class_dim),
        "y_final_disp_flat_dim": int(NUM_LANDMARKS * n_coords_eff),
        "class_encoding": str(class_encoding),
        "class_names": [str(x) for x in class_names],
        "class_distribution": class_distribution,
        "displacement": {
            "displaced_joints_n_range": [int(displaced_joints_n_range[0]), int(displaced_joints_n_range[1])],
            "max_abs_displacement_xyz": [float(v) for v in max_abs_displacement_xyz],
            "displacement_time_min": int(displacement_time_min),
            "displacement_time_max": int(displacement_time_max),
            "ramp": "linear_then_hold",
            "random_seed": None if random_seed is None else int(random_seed),
            "drop_z": bool(drop_z),
        },
    }

    metadata_path = output_dir / "training_data_displacement_metadata.json"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    if save_debug_metadata:
        debug_path = output_dir / "training_data_displacement_debug.json"
        with open(debug_path, "w") as f:
            json.dump(displacement_debug, f, indent=2)

    return str(npz_path)


def process_workout_directory_displacement(
    *,
    extractor,
    videos_dir: str,
    output_dir: Optional[str],
    save_frames: bool,
    video_extensions: Tuple[str, ...],
    num_processes: int,
    displaced_joints_n_range: Tuple[int, int],
    max_abs_displacement_xyz: Tuple[float, float, float],
    displacement_time_min: int,
    displacement_time_max: int,
    class_encoding: str,
    random_seed: Optional[int],
    save_debug_metadata: bool,
    drop_z: bool = False,
) -> Dict:
    """
    Mirrors `VideoPoseExtractor.process_workout_directory`, but emits displacement training data.
    """
    videos_dir_path = Path(videos_dir)
    if not videos_dir_path.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir_path}")

    out_dir = (videos_dir_path.parent / "output") if output_dir is None else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {out_dir.absolute()}")

    ignore_file = videos_dir_path / ".ignore_videos.txt"
    ignored_videos = set()
    if ignore_file.exists():
        with open(ignore_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    ignored_videos.add(line)
    if ignored_videos:
        print(f"Loaded {len(ignored_videos)} videos to ignore from .ignore_videos.txt")

    workout_dirs = [d for d in videos_dir_path.iterdir() if d.is_dir() and "disabled" not in d.name]
    if not workout_dirs:
        raise ValueError(f"No workout type subdirectories found in {videos_dir_path}")

    print(f"\n{'='*60}")
    print(f"Processing workout videos from: {videos_dir_path}")
    print(f"Found {len(workout_dirs)} workout types")
    print(f"{'='*60}\n")

    # Count videos for overall progress.
    total_videos = 0
    for workout_dir in sorted(workout_dirs):
        video_files: List[Path] = []
        for ext in video_extensions:
            video_files.extend(workout_dir.glob(f"*{ext}"))
        total_videos += len(video_files)

    overall_pbar = tqdm(
        total=total_videos,
        desc="Overall Progress",
        position=0,
        leave=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} videos [{elapsed}<{remaining}, {rate_fmt}]",
    )

    all_sequences: List[Dict] = []
    class_counts: Dict[str, int] = {}
    workout_classes: List[str] = []
    workout_data_paths: Dict[str, str] = {}
    workout_npz_paths: Dict[str, str] = {}

    for workout_dir in sorted(workout_dirs):
        workout_class = workout_dir.name
        workout_classes.append(workout_class)
        class_counts[workout_class] = 0

        safe_workout_name = workout_class.replace(" ", "_").replace("/", "_")
        workout_output_dir = out_dir / safe_workout_name
        workout_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'─'*60}")
        print(f"Processing workout class: {workout_class}")
        print(f"Directory: {workout_dir}")
        print(f"Output directory: {workout_output_dir}")
        print(f"{'─'*60}")

        video_files: List[Path] = []
        for ext in video_extensions:
            video_files.extend(workout_dir.glob(f"*{ext}"))

        original_count = len(video_files)
        video_files = [vf for vf in video_files if f"{workout_class}/{vf.name}" not in ignored_videos]
        ignored_count = original_count - len(video_files)
        if ignored_count > 0:
            print(f"  Skipping {ignored_count} ignored video(s)")
        if not video_files:
            print(f"  No video files found in {workout_dir} (after filtering ignored videos)")
            continue
        print(f"  Found {len(video_files)} video files")

        workout_sequences: List[Dict] = []

        if num_processes == 0:
            effective_processes = max(1, cpu_count() - 1)
        else:
            effective_processes = max(1, min(num_processes, len(video_files)))

        if effective_processes > 1 and len(video_files) > 1:
            print(f"  Using {effective_processes} parallel processes")
            # Local import so `--help` can run without MediaPipe deps.
            from video_pose_extractor import process_single_video_worker  # type: ignore
            worker_args = [
                (
                    video_file,
                    workout_output_dir,
                    save_frames,
                    workout_class,
                    int(extractor.target_fps),
                    extractor.sequence_duration,
                    extractor.normalize_pose,
                    extractor.augment_data,
                    extractor.video_speed_augment,
                    extractor.video_speed_factors,
                    extractor.sequence_start_stride,
                    extractor.verbose_pool_workers,
                )
                for video_file in video_files
            ]

            pool = Pool(processes=effective_processes)
            interrupted = False
            results: List[Tuple[str, List[Dict], Optional[str]]] = []
            try:
                it = pool.imap_unordered(process_single_video_worker, worker_args, chunksize=1)
                for result in tqdm(
                    it,
                    total=len(video_files),
                    desc=f"  {workout_class[:30]:<30}",
                    position=1,
                    leave=False,
                ):
                    results.append(result)
                    overall_pbar.update(1)
            except KeyboardInterrupt:
                interrupted = True
                print("\nInterrupted — terminating worker pool...", flush=True)
                pool.terminate()
                pool.join()
                raise
            finally:
                if not interrupted:
                    pool.close()
                    pool.join()

            for video_path, sequences, error in results:
                if error:
                    print(f"\n  Error processing {Path(video_path).name}: {error}")
                    continue
                workout_sequences.extend(sequences)
                all_sequences.extend(sequences)
                class_counts[workout_class] += len(sequences)
        else:
            for video_file in tqdm(video_files, desc=f"  {workout_class[:30]:<30}", position=1, leave=False):
                try:
                    sequences = extractor.process_video(
                        str(video_file),
                        output_dir=str(workout_output_dir),
                        save_frames=save_frames,
                        workout_class=workout_class,
                    )
                    workout_sequences.extend(sequences)
                    all_sequences.extend(sequences)
                    class_counts[workout_class] += len(sequences)
                    overall_pbar.update(1)
                except Exception as e:
                    print(f"\n  Error processing {video_file.name}: {e}")
                    overall_pbar.update(1)
                    continue

        # Save workout-specific JSON (same structure as base)
        if workout_sequences:
            eff_stride = extractor.sequence_start_stride if extractor.sequence_start_stride is not None else extractor.frames_per_sequence
            workout_data = {
                "metadata": {
                    "workout_class": workout_class,
                    "total_sequences": len(workout_sequences),
                    "frames_per_sequence": extractor.frames_per_sequence,
                    "sequence_duration": extractor.sequence_duration,
                    "target_fps": extractor.target_fps,
                    "sequence_start_stride": int(eff_stride),
                },
                "sequences": workout_sequences,
            }

            workout_json_path = workout_output_dir / "training_data.json"
            with open(workout_json_path, "w") as f:
                json.dump(workout_data, f, indent=2)
            workout_data_paths[workout_class] = str(workout_json_path)

            # Save workout-specific displacement dataset
            try:
                workout_npz_path = save_displacement_training_data_npz(
                    sequences=workout_sequences,
                    workout_classes=[workout_class],
                    output_dir=workout_output_dir,
                    displaced_joints_n_range=displaced_joints_n_range,
                    max_abs_displacement_xyz=max_abs_displacement_xyz,
                    displacement_time_min=displacement_time_min,
                    displacement_time_max=displacement_time_max,
                    class_encoding=class_encoding,
                    random_seed=random_seed,
                    save_debug_metadata=save_debug_metadata,
                    drop_z=drop_z,
                )
                workout_npz_paths[workout_class] = workout_npz_path
                print(f"  Saved {workout_class} displacement dataset: {len(workout_sequences)} sequences")
            except Exception as e:
                print(f"  Warning: Failed to save {workout_class} displacement dataset: {e}")

    # Consolidated JSON (same structure as base)
    eff_stride = extractor.sequence_start_stride if extractor.sequence_start_stride is not None else extractor.frames_per_sequence
    training_data = {
        "metadata": {
            "total_sequences": len(all_sequences),
            "total_workout_classes": len(workout_classes),
            "workout_classes": sorted(workout_classes),
            "class_distribution": class_counts,
            "frames_per_sequence": extractor.frames_per_sequence,
            "sequence_duration": extractor.sequence_duration,
            "target_fps": extractor.target_fps,
            "sequence_start_stride": int(eff_stride),
        },
        "sequences": all_sequences,
    }

    training_data_path = out_dir / "training_data.json"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(training_data_path, "w") as f:
            json.dump(training_data, f, indent=2)

    # Consolidated displacement dataset
    training_data_npz_path = None
    if all_sequences:
        try:
            training_data_npz_path = save_displacement_training_data_npz(
                sequences=all_sequences,
                workout_classes=workout_classes,
                output_dir=out_dir,
                displaced_joints_n_range=displaced_joints_n_range,
                max_abs_displacement_xyz=max_abs_displacement_xyz,
                displacement_time_min=displacement_time_min,
                displacement_time_max=displacement_time_max,
                class_encoding=class_encoding,
                random_seed=random_seed,
                save_debug_metadata=save_debug_metadata,
                drop_z=drop_z,
            )
        except Exception as e:
            print(f"\nWarning: Failed to save consolidated displacement dataset: {e}")

    overall_pbar.close()

    print(f"\n{'='*60}")
    print("Batch Processing Complete!")
    print(f"{'='*60}")
    print(f"Total sequences extracted: {len(all_sequences)}")
    print(f"Workout classes: {len(workout_classes)}")
    print("\nClass distribution:")
    for wc, count in sorted(class_counts.items()):
        print(f"  {wc}: {count} sequences")
    print("\nTraining data saved:")
    print(f"  Consolidated JSON: {training_data_path}")
    if training_data_npz_path:
        print(f"  Consolidated displacement NumPy: {training_data_npz_path}")
    print("\n  Workout-specific files organized in subdirectories:")
    for wc in sorted(workout_classes):
        if wc in workout_data_paths:
            print(f"    {wc}/")
    print(f"{'='*60}\n")

    return {
        "total_sequences": len(all_sequences),
        "workout_classes": workout_classes,
        "class_counts": class_counts,
        "all_sequences": all_sequences,
        "training_data_path": str(training_data_path),
        "training_data_displacement_npz_path": training_data_npz_path,
        "workout_data_paths": workout_data_paths,
        "workout_displacement_npz_paths": workout_npz_paths,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract pose sequences (same as video_pose_extractor.py) and also build a "
            "displacement regression dataset: X = pose sequence + class feature, y = "
            "negative displacement field."
        )
    )

    parser.add_argument("input_path", type=str, help="Path to input video file or workout directory")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Output directory for extracted pose data. If not specified: defaults to "
            '"output/" for single video, or "Videos/../output" for batch processing.'
        ),
    )

    # Keep parity with base extractor args
    parser.add_argument("--fps", type=int, default=15, help="Target fps for sequences (default: 15)")
    parser.add_argument("--duration", type=float, default=1.0, help="Sequence duration seconds (default: 1.0)")
    parser.add_argument("--save-frames", action="store_true", default=False, help="Save frame images (default: False)")
    parser.add_argument("--no-save-frames", dest="save_frames", action="store_false", help="Do not save frame images")
    parser.add_argument("--batch", action="store_true", help="Process directory of workout videos (auto-detected)")
    parser.add_argument("--workout-class", type=str, default=None, help="Workout class label for single video (optional)")
    parser.add_argument("--num-processes", "-p", type=int, default=1, metavar="N", help="Parallel processes for batch (default: 1, 0=auto)")
    parser.add_argument("--normalize-pose", action="store_true", default=False, help="Normalize landmarks to hip center")
    parser.add_argument("--augment-data", action="store_true", default=False, help="Mirror augmentation (same as base)")
    parser.add_argument("--video-speed-augment", action="store_true", default=False, help="Time-stretch augmentation passes")
    parser.add_argument(
        "--video-speed-factors",
        type=str,
        default="0.75,1.25",
        metavar="LIST",
        help='Comma-separated speed factors for --video-speed-augment (default: "0.75,1.25")',
    )
    parser.add_argument(
        "--sequence-start-stride",
        type=int,
        default=None,
        metavar="N",
        help="Stride between sequence starts; <frames-per-sequence yields overlap (default: none)",
    )
    parser.add_argument("--verbose-pool-workers", action="store_true", default=False, help="Verbose worker logging in batch multiprocessing")

    # Displacement dataset args
    parser.add_argument(
        "--displaced-joints-n",
        type=str,
        default="3",
        metavar="N|MIN-MAX",
        help=(
            "How many of the 12 joints to randomly displace per sequence. "
            'Accepts "N" or "MIN-MAX" (inclusive). Default: "3". Use "0" to disable.'
        ),
    )
    parser.add_argument(
        "--max-abs-displacement-xyz",
        type=str,
        default="0.02,0.02,0.02",
        metavar="X,Y,Z",
        help="Max absolute displacement for each axis as comma-separated floats (default: 0.02,0.02,0.02).",
    )
    parser.add_argument(
        "--displacement-time-min",
        type=int,
        default=2,
        metavar="T",
        help="Minimum displacement_time (frames) (default: 2). Must be >=2 and < sequence length.",
    )
    parser.add_argument(
        "--displacement-time-max",
        type=int,
        default=14,
        metavar="T",
        help="Maximum displacement_time (frames) (default: 14). Must be >=2 and < sequence length.",
    )
    parser.add_argument(
        "--class-encoding",
        type=str,
        choices=("int", "onehot"),
        default="int",
        help='How to include workout class in X (default: "int").',
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for displacement generation (default: None).",
    )
    parser.add_argument(
        "--save-displacement-debug-metadata",
        action="store_true",
        default=False,
        help="Save per-sequence displacement params to JSON for auditing (default: False).",
    )
    parser.add_argument(
        "--drop-z",
        action="store_true",
        default=False,
        help=(
            "Zero depth (z) for all landmarks in the displacement dataset X and disable z "
            "offsets in synthetic displacement (dz=0); y z-components stay 0. "
            "Shape remains T×12×3 per sequence (default: False)."
        ),
    )

    args = parser.parse_args()

    speed_factors_tuple: Optional[Tuple[float, ...]] = None
    if args.video_speed_augment:
        raw_parts = [p.strip() for p in args.video_speed_factors.split(",") if p.strip()]
        if not raw_parts:
            print("Error: --video-speed-factors must list at least one number when using --video-speed-augment")
            return 1
        try:
            speed_factors_tuple = tuple(float(p) for p in raw_parts)
        except ValueError:
            print("Error: --video-speed-factors must be comma-separated floats")
            return 1

    try:
        displaced_joints_n_range = _parse_int_range(args.displaced_joints_n)
    except Exception:
        print('Error: --displaced-joints-n must be like "3" or "2-5"')
        return 1

    try:
        max_abs_parts = [float(x.strip()) for x in str(args.max_abs_displacement_xyz).split(",")]
        if len(max_abs_parts) != 3:
            raise ValueError
        max_abs_disp = (float(max_abs_parts[0]), float(max_abs_parts[1]), float(max_abs_parts[2]))
    except Exception:
        print('Error: --max-abs-displacement-xyz must be like "0.02,0.02,0.02"')
        return 1

    input_path = Path(args.input_path)
    is_batch = args.batch or (input_path.is_dir() and not input_path.suffix)

    # Local import so `--help` can run without MediaPipe/TensorFlow deps.
    from video_pose_extractor import VideoPoseExtractor  # type: ignore

    extractor = VideoPoseExtractor(
        fps=args.fps,
        sequence_duration=args.duration,
        normalize_pose=args.normalize_pose,
        augment_data=args.augment_data,
        video_speed_augment=args.video_speed_augment,
        video_speed_factors=speed_factors_tuple,
        sequence_start_stride=args.sequence_start_stride,
        verbose_pool_workers=args.verbose_pool_workers,
    )

    try:
        if is_batch:
            if not input_path.is_dir():
                print(f"Error: {input_path} is not a directory")
                return 1
            result = process_workout_directory_displacement(
                extractor=extractor,
                videos_dir=str(input_path),
                output_dir=args.output,
                save_frames=args.save_frames,
                video_extensions=(".mp4", ".mov", ".MOV", ".avi", ".mkv"),
                num_processes=args.num_processes,
                displaced_joints_n_range=displaced_joints_n_range,
                max_abs_displacement_xyz=max_abs_disp,
                displacement_time_min=args.displacement_time_min,
                displacement_time_max=args.displacement_time_max,
                class_encoding=args.class_encoding,
                random_seed=args.random_seed,
                save_debug_metadata=args.save_displacement_debug_metadata,
                drop_z=args.drop_z,
            )
            print(
                f"\nSuccessfully processed {result['total_sequences']} sequences from "
                f"{len(result['workout_classes'])} workout classes!"
            )
        else:
            if not input_path.is_file():
                print(f"Error: {input_path} is not a file")
                return 1

            # Single-video mode: extract sequences and still save them as JSON as the base extractor does.
            output_dir = args.output or "output"
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            print(f"Output directory: {output_path.absolute()}")

            sequences = extractor.process_video(
                str(input_path),
                output_dir,
                save_frames=args.save_frames,
                workout_class=args.workout_class,
            )
            print(f"\nSuccessfully extracted {len(sequences)} sequences!")

            # If workout class is present, write a displacement dataset for this single video.
            if sequences and sequences[0].get("workout_class"):
                npz_path = save_displacement_training_data_npz(
                    sequences=sequences,
                    workout_classes=[str(sequences[0].get("workout_class"))],
                    output_dir=output_path,
                    displaced_joints_n_range=displaced_joints_n_range,
                    max_abs_displacement_xyz=max_abs_disp,
                    displacement_time_min=args.displacement_time_min,
                    displacement_time_max=args.displacement_time_max,
                    class_encoding=args.class_encoding,
                    random_seed=args.random_seed,
                    save_debug_metadata=args.save_displacement_debug_metadata,
                    drop_z=args.drop_z,
                )
                print(f"Displacement dataset saved: {npz_path}")
            else:
                print("Note: No workout_class found for sequences; displacement dataset not saved in single-video mode.")
    except Exception as e:
        print(f"Error processing: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

