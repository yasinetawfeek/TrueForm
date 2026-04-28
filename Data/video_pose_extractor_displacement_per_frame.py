#!/usr/bin/env python3
"""
Per-frame pose extraction + displacement dataset (no temporal windows).

Unlike ``video_pose_extractor_displacement.py``, this does **not** resample the
video to a target FPS, duplicate/skip frames, sliding windows, or speed
augmentation. Each decoded frame becomes one training row.

- X: flattened pose for that frame (12 * C coordinates, C=2 or 3) + workout
     class as int (1 float) or one-hot (n_classes floats).
- y: unchanged vs the displacement dataset — negative of the applied per-joint
     displacement on that frame (shape 12*C after flattening). Rows can skip
     displacement with ``--displacement-probability`` < 1 (clean pose, y = 0).

JSON output is **grouped by video**: one object per video with a ``frames``
array (not ``sequences`` of multi-frame clips).

``--help`` avoids importing MediaPipe by keeping heavy imports inside ``main()``
and worker entrypoints.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
import zlib
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Same 12 landmarks as ``video_pose_extractor.py`` / ``video_pose_extractor_displacement.py``.
LANDMARK_NAMES = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
NUM_LANDMARKS = 12


def frame_pose_to_array(frame: Dict, *, drop_z: bool = False) -> np.ndarray:
    """Single-frame pose -> (12, 3) float32; missing pose -> zeros."""
    out = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
    pose = frame.get("pose") or {}
    if pose.get("detected") and pose.get("landmarks"):
        for j, lm in enumerate(pose["landmarks"]):
            if j >= NUM_LANDMARKS:
                break
            out[j, 0] = float(lm.get("x", 0.0))
            out[j, 1] = float(lm.get("y", 0.0))
            out[j, 2] = 0.0 if drop_z else float(lm.get("z", 0.0))
    return out


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
    s = str(value).strip()
    if not s:
        raise ValueError("empty range")
    if "-" in s:
        parts = [p.strip() for p in s.split("-", 1)]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"invalid range: {value}")
        lo, hi = int(parts[0]), int(parts[1])
    else:
        lo = hi = int(s)
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def apply_instant_displacement(
    pose: np.ndarray,
    *,
    rng: np.random.Generator,
    displaced_joints_n_range: Tuple[int, int],
    max_abs_displacement_xyz: Tuple[float, float, float],
    drop_z: bool,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    One-frame displacement: selected joints receive a constant random offset.

    Returns:
        displaced_pose (12, 3), final_disp (12, 3), metadata list per displaced joint.
    """
    lo_n, hi_n = displaced_joints_n_range
    lo_n = max(0, min(NUM_LANDMARKS, int(lo_n)))
    hi_n = max(0, min(NUM_LANDMARKS, int(hi_n)))
    if lo_n > hi_n:
        lo_n, hi_n = hi_n, lo_n

    n = int(rng.integers(lo_n, hi_n + 1)) if hi_n > lo_n else int(lo_n)
    if n == 0:
        final = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
        return pose.astype(np.float32, copy=False), final, []

    joints = rng.choice(NUM_LANDMARKS, size=n, replace=False)
    max_dx, max_dy, max_dz = (
        float(max_abs_displacement_xyz[0]),
        float(max_abs_displacement_xyz[1]),
        float(max_abs_displacement_xyz[2]),
    )

    final = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
    meta: List[Dict] = []

    for j in joints.tolist():
        dx = float(rng.uniform(-max_dx, max_dx))
        dy = float(rng.uniform(-max_dy, max_dy))
        dz = 0.0 if drop_z else float(rng.uniform(-max_dz, max_dz))
        final[int(j), 0] = dx
        final[int(j), 1] = dy
        final[int(j), 2] = dz
        meta.append(
            {
                "joint_index": int(j),
                "joint_name": LANDMARK_NAMES[int(j)] if 0 <= int(j) < len(LANDMARK_NAMES) else str(j),
                "dx": dx,
                "dy": dy,
                "dz": dz,
            }
        )

    displaced = (pose.astype(np.float32, copy=True) + final).astype(np.float32, copy=False)
    return displaced, final, meta


def _stable_frame_seed(global_seed: Optional[int], video_key: str, frame_index: int) -> int:
    """Deterministic 32-bit seed per (video path string, frame index, global seed)."""
    part = zlib.crc32(video_key.encode("utf-8", errors="replace")) & 0xFFFFFFFF
    gs = int(global_seed or 0) & 0xFFFFFFFF
    x = (part ^ gs ^ (int(frame_index) * 0x9E3779B9)) & 0xFFFFFFFF
    return int(x) if x != 0 else 1


def extract_frames_from_video_capture(
    extractor,
    cap,
    video_path: Path,
    *,
    output_dir: Optional[Path],
    save_frames: bool,
    workout_class: Optional[str],
) -> Tuple[List[Dict], float]:
    """
    Read every frame from ``cap`` in order; run pose on each. No FPS conversion.

    Returns:
        frames list, native video FPS (float).
    """
    import cv2

    actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frames: List[Dict] = []
    read_idx = -1
    mp_timestamp_ms = 0

    stem = video_path.stem
    frames_dir: Optional[Path] = None
    if output_dir is not None and save_frames:
        frames_dir = Path(output_dir) / f"{stem}_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(
        total=max(total_hint, 0) or None,
        desc=f"  frames {stem[:24]:<24}",
        leave=False,
        disable=os.environ.get("VIDEO_POSE_EXTRACT_QUIET", "").lower() in ("1", "true", "yes"),
    )

    while cap.isOpened():
        ret, bgr = cap.read()
        if not ret:
            break
        read_idx += 1

        pose_data = extractor.extract_pose_landmarks(bgr, mp_timestamp_ms)
        mp_timestamp_ms += 33

        ts = read_idx / max(actual_fps, 1e-6)
        frame_image_path = None
        if frames_dir is not None:
            fname = f"frame_{read_idx:06d}.jpg"
            fpath = frames_dir / fname
            cv2.imwrite(str(fpath), bgr)
            frame_image_path = str(Path(f"{stem}_frames") / fname)

        row = {
            "frame_index": read_idx,
            "timestamp": ts,
            "pose": pose_data,
            "frame_image": frame_image_path,
        }
        if workout_class:
            row["workout_class"] = workout_class

        frames.append(row)
        if pbar.total:
            pbar.update(1)

    pbar.close()
    return frames, actual_fps


def build_video_record(
    *,
    video_path: Path,
    video_file: str,
    workout_class: str,
    video_fps: float,
    frames: List[Dict],
) -> Dict:
    detected = sum(
        1
        for f in frames
        if (f.get("pose") or {}).get("detected")
    )
    rec = {
        "metadata": {
            "video_file": video_file,
            "video_name": video_path.stem,
            "workout_class": workout_class,
            "video_fps": float(video_fps),
            "total_frames": len(frames),
            "frames_with_pose": int(detected),
        },
        "frames": frames,
    }
    return rec


def save_video_json(path: Path, record: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(path, "w") as f:
            json.dump(record, f, indent=2)


def save_displacement_npz_from_video_records(
    *,
    video_records: List[Dict],
    class_names: List[str],
    output_dir: Path,
    displaced_joints_n_range: Tuple[int, int],
    max_abs_displacement_xyz: Tuple[float, float, float],
    class_encoding: str,
    random_seed: Optional[int],
    save_debug_metadata: bool,
    drop_z: bool,
    skip_undetected: bool,
    displacement_probability: float,
) -> str:
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    n_classes = len(class_names)
    n_coords_eff = 2 if drop_z else 3
    pose_flat = NUM_LANDMARKS * n_coords_eff
    class_dim = 1 if class_encoding == "int" else n_classes

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    y_class_idx_list: List[int] = []
    debug_rows: List[Dict] = []

    global_row = 0
    y_raw_list: List[str] = []

    for vrec in video_records:
        meta = vrec["metadata"]
        wc = str(meta["workout_class"])
        if wc not in class_to_idx:
            raise ValueError(f"Unknown workout_class in video record: {wc}")
        cidx = int(class_to_idx[wc])
        vname = str(meta.get("video_name", ""))
        vfile = str(meta.get("video_file", ""))

        for local_i, frame in enumerate(vrec["frames"]):
            if skip_undetected and not (frame.get("pose") or {}).get("detected"):
                continue

            pose = frame_pose_to_array(frame, drop_z=drop_z)
            fi = int(frame.get("frame_index", local_i))
            row_rng = np.random.default_rng(_stable_frame_seed(random_seed, vfile, fi))

            if float(displacement_probability) >= 1.0:
                do_displace = True
            elif float(displacement_probability) <= 0.0:
                do_displace = False
            else:
                do_displace = bool(row_rng.random() < float(displacement_probability))

            if do_displace:
                displaced, final_disp, meta_j = apply_instant_displacement(
                    pose,
                    rng=row_rng,
                    displaced_joints_n_range=displaced_joints_n_range,
                    max_abs_displacement_xyz=max_abs_displacement_xyz,
                    drop_z=drop_z,
                )
            else:
                displaced = pose.astype(np.float32, copy=False)
                final_disp = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
                meta_j = []

            if drop_z:
                x_pose = displaced[:, :2].reshape(-1)
                y_flat = (-final_disp[:, :2]).reshape(-1)
            else:
                x_pose = displaced.reshape(-1)
                y_flat = (-final_disp).reshape(-1)

            class_feat = _encode_class_feature(
                class_index=cidx,
                n_classes=n_classes,
                encoding=class_encoding,
            )
            X_list.append(np.concatenate([x_pose, class_feat], axis=0))
            y_list.append(y_flat)
            y_class_idx_list.append(cidx)
            y_raw_list.append(wc)

            if save_debug_metadata:
                debug_rows.append(
                    {
                        "row_index": int(global_row),
                        "video_name": vname,
                        "video_file": vfile,
                        "frame_index": fi,
                        "workout_class": wc,
                        "class_index": cidx,
                        "displacement_applied": bool(do_displace),
                        "displaced_joints": meta_j,
                    }
                )
            global_row += 1

    if not X_list:
        raise ValueError("No frame rows to save (empty after filters)")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.stack(y_list, axis=0).astype(np.float32)
    y_class_idx = np.array(y_class_idx_list, dtype=np.int64)
    y_onehot = np.eye(n_classes, dtype=np.float32)[y_class_idx]

    npz_path = output_dir / "training_data_displacement_per_frame.npz"
    y_raw_arr = np.array(y_raw_list, dtype=object)
    np.savez_compressed(
        npz_path,
        X=X,
        y=y,
        y_class=y_class_idx,
        y_class_onehot=y_onehot,
        y_class_raw=y_raw_arr,
        class_names=np.array(class_names, dtype=object),
        n_landmarks=np.int64(NUM_LANDMARKS),
        n_coords=np.int64(n_coords_eff),
        sequence_length=np.int64(1),
    )

    meta_out = {
        "dataset": "displacement_per_frame",
        "n_samples": int(X.shape[0]),
        "n_videos": int(len(video_records)),
        "sequence_length": 1,
        "n_landmarks": NUM_LANDMARKS,
        "n_coords": n_coords_eff,
        "X_shape": [int(d) for d in X.shape],
        "y_shape": [int(d) for d in y.shape],
        "X_pose_flat_dim": int(pose_flat),
        "X_class_dim": int(class_dim),
        "y_final_disp_flat_dim": int(pose_flat),
        "class_encoding": class_encoding,
        "class_names": class_names,
        "skip_undetected": bool(skip_undetected),
        "displacement": {
            "displaced_joints_n_range": [int(displaced_joints_n_range[0]), int(displaced_joints_n_range[1])],
            "max_abs_displacement_xyz": [float(v) for v in max_abs_displacement_xyz],
            "instant_per_frame": True,
            "probability": float(displacement_probability),
            "random_seed": None if random_seed is None else int(random_seed),
            "drop_z": bool(drop_z),
        },
    }
    with open(output_dir / "training_data_displacement_per_frame_metadata.json", "w") as f:
        json.dump(meta_out, f, indent=2)

    if save_debug_metadata and debug_rows:
        with open(output_dir / "training_data_displacement_per_frame_debug.json", "w") as f:
            json.dump(debug_rows, f, indent=2)

    return str(npz_path)


def process_one_video_file(
    *,
    video_path: Path,
    output_dir: Path,
    save_frames: bool,
    workout_class: str,
    normalize_pose: bool,
) -> Dict:
    """Extract frame rows for a single video; returns video JSON record."""
    from video_pose_extractor import VideoPoseExtractor

    extractor = VideoPoseExtractor(
        fps=15,
        sequence_duration=1.0,
        normalize_pose=normalize_pose,
        augment_data=False,
        video_speed_augment=False,
        sequence_start_stride=None,
    )
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        frames, vfps = extract_frames_from_video_capture(
            extractor,
            cap,
            video_path,
            output_dir=output_dir,
            save_frames=save_frames,
            workout_class=workout_class,
        )
        cap.release()

        return build_video_record(
            video_path=video_path,
            video_file=str(video_path),
            workout_class=workout_class,
            video_fps=vfps,
            frames=frames,
        )
    finally:
        extractor.close()


def process_single_video_worker(args_tuple):
    (
        video_path,
        output_dir,
        save_frames,
        workout_class,
        normalize_pose,
        verbose_pool_workers,
    ) = args_tuple

    if verbose_pool_workers:
        os.environ.pop("VIDEO_POSE_EXTRACT_QUIET", None)
    else:
        os.environ["VIDEO_POSE_EXTRACT_QUIET"] = "1"

    try:
        vp = Path(video_path)
        out = Path(output_dir)
        rec = process_one_video_file(
            video_path=vp,
            output_dir=out,
            save_frames=save_frames,
            workout_class=workout_class,
            normalize_pose=normalize_pose,
        )
        json_path = out / f"{vp.stem}_pose_frames.json"
        save_video_json(json_path, rec)
        return (str(video_path), rec, None)
    except Exception as e:
        return (str(video_path), None, str(e))


def process_workout_directory(
    *,
    videos_dir: Path,
    output_dir: Optional[Path],
    save_frames: bool,
    video_extensions: Tuple[str, ...],
    num_processes: int,
    normalize_pose: bool,
    displaced_joints_n_range: Tuple[int, int],
    max_abs_displacement_xyz: Tuple[float, float, float],
    class_encoding: str,
    random_seed: Optional[int],
    save_debug_metadata: bool,
    drop_z: bool,
    skip_undetected: bool,
    displacement_probability: float,
    verbose_pool_workers: bool,
) -> Dict:
    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")

    out_dir = (videos_dir.parent / "output_per_frame") if output_dir is None else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ignore_file = videos_dir / ".ignore_videos.txt"
    ignored: set = set()
    if ignore_file.exists():
        with open(ignore_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    ignored.add(line)
    if ignored:
        print(f"Loaded {len(ignored)} videos to ignore from .ignore_videos.txt")

    workout_dirs = [d for d in videos_dir.iterdir() if d.is_dir() and "disabled" not in d.name.lower()]
    if not workout_dirs:
        raise ValueError(f"No workout subdirectories in {videos_dir}")

    all_video_records: List[Dict] = []
    class_counts: Dict[str, int] = {}
    workout_classes: List[str] = []
    workout_json_paths: Dict[str, str] = {}
    workout_npz_paths: Dict[str, str] = {}

    total_videos = 0
    for wd in sorted(workout_dirs):
        for ext in video_extensions:
            total_videos += len(list(wd.glob(f"*{ext}")))

    overall_pbar = tqdm(
        total=total_videos,
        desc="Videos",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )

    for workout_dir in sorted(workout_dirs):
        workout_class = workout_dir.name
        workout_classes.append(workout_class)
        class_counts[workout_class] = 0

        safe_name = workout_class.replace(" ", "_").replace("/", "_")
        w_out = out_dir / safe_name
        w_out.mkdir(parents=True, exist_ok=True)

        video_files: List[Path] = []
        for ext in video_extensions:
            video_files.extend(workout_dir.glob(f"*{ext}"))

        before = len(video_files)
        video_files = [vf for vf in video_files if f"{workout_class}/{vf.name}" not in ignored]
        if before - len(video_files):
            print(f"  [{workout_class}] skipped {before - len(video_files)} ignored")

        if not video_files:
            print(f"  [{workout_class}] no videos")
            continue

        records: List[Dict] = []

        if num_processes == 0:
            nproc = max(1, cpu_count() - 1)
        else:
            nproc = max(1, min(num_processes, len(video_files)))

        if nproc > 1 and len(video_files) > 1:
            worker_args = [
                (
                    str(vf),
                    str(w_out),
                    save_frames,
                    workout_class,
                    normalize_pose,
                    verbose_pool_workers,
                )
                for vf in video_files
            ]
            pool = Pool(processes=nproc)
            interrupted = False
            try:
                it = pool.imap_unordered(process_single_video_worker, worker_args, chunksize=1)
                for path_s, rec, err in tqdm(
                    it,
                    total=len(video_files),
                    desc=f"  {workout_class[:28]}",
                    leave=False,
                ):
                    overall_pbar.update(1)
                    if err:
                        print(f"\n  Error {Path(path_s).name}: {err}")
                        continue
                    assert rec is not None
                    records.append(rec)
                    all_video_records.append(rec)
                    class_counts[workout_class] += int(rec["metadata"]["total_frames"])
            except KeyboardInterrupt:
                interrupted = True
                pool.terminate()
                pool.join()
                raise
            finally:
                if not interrupted:
                    pool.close()
                    pool.join()
        else:
            for vf in tqdm(video_files, desc=f"  {workout_class[:28]}", leave=False):
                try:
                    rec = process_one_video_file(
                        video_path=vf,
                        output_dir=w_out,
                        save_frames=save_frames,
                        workout_class=workout_class,
                        normalize_pose=normalize_pose,
                    )
                    save_video_json(w_out / f"{vf.stem}_pose_frames.json", rec)
                    records.append(rec)
                    all_video_records.append(rec)
                    class_counts[workout_class] += int(rec["metadata"]["total_frames"])
                except Exception as e:
                    print(f"\n  Error {vf.name}: {e}")
                overall_pbar.update(1)

        if records:
            consolidated_workout = {
                "metadata": {
                    "workout_class": workout_class,
                    "n_videos": len(records),
                    "total_frame_rows": sum(int(r["metadata"]["total_frames"]) for r in records),
                    "index_note": "Full frame arrays live in each video's *_pose_frames.json beside this file.",
                },
                "videos": [
                    {
                        **rec["metadata"],
                        "pose_frames_json": f"{rec['metadata']['video_name']}_pose_frames.json",
                    }
                    for rec in records
                ],
            }
            jp = w_out / "training_data.json"
            with open(jp, "w") as f:
                json.dump(consolidated_workout, f, indent=2)
            workout_json_paths[workout_class] = str(jp)

            try:
                npz = save_displacement_npz_from_video_records(
                    video_records=records,
                    class_names=[workout_class],
                    output_dir=w_out,
                    displaced_joints_n_range=displaced_joints_n_range,
                    max_abs_displacement_xyz=max_abs_displacement_xyz,
                    class_encoding=class_encoding,
                    random_seed=random_seed,
                    save_debug_metadata=save_debug_metadata,
                    drop_z=drop_z,
                    skip_undetected=skip_undetected,
                    displacement_probability=displacement_probability,
                )
                workout_npz_paths[workout_class] = npz
                print(f"  Saved {workout_class}: {len(records)} videos -> {npz}")
            except Exception as e:
                print(f"  Warning: NPZ failed for {workout_class}: {e}")

    overall_pbar.close()

    class_names_sorted = sorted(set(workout_classes))
    total_rows = sum(int(r["metadata"]["total_frames"]) for r in all_video_records)
    root_meta = {
        "total_videos": len(all_video_records),
        "total_frame_rows": total_rows,
        "total_workout_classes": len(class_names_sorted),
        "workout_classes": class_names_sorted,
        "class_distribution_frame_rows": class_counts,
        "index_note": (
            "Per-video pose data is in <workout_class_dir>/<video_stem>_pose_frames.json "
            "and workout summaries in <workout_class_dir>/training_data.json."
        ),
    }
    consolidated = {
        "metadata": root_meta,
        "videos": [
            {
                **rec["metadata"],
                "workout_output_subdir": rec["metadata"]["workout_class"].replace(" ", "_").replace("/", "_"),
                "pose_frames_json": f"{rec['metadata']['video_name']}_pose_frames.json",
            }
            for rec in all_video_records
        ],
    }
    training_path = out_dir / "training_data.json"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(training_path, "w") as f:
            json.dump(consolidated, f, indent=2)

    global_npz = None
    if all_video_records:
        try:
            global_npz = save_displacement_npz_from_video_records(
                video_records=all_video_records,
                class_names=class_names_sorted,
                output_dir=out_dir,
                displaced_joints_n_range=displaced_joints_n_range,
                max_abs_displacement_xyz=max_abs_displacement_xyz,
                class_encoding=class_encoding,
                random_seed=random_seed,
                save_debug_metadata=save_debug_metadata,
                drop_z=drop_z,
                skip_undetected=skip_undetected,
                displacement_probability=displacement_probability,
            )
        except Exception as e:
            print(f"Warning: consolidated NPZ failed: {e}")

    print(f"\nConsolidated JSON: {training_path}")
    if global_npz:
        print(f"Consolidated NPZ: {global_npz}")

    return {
        "total_videos": len(all_video_records),
        "total_frame_rows": root_meta["total_frame_rows"],
        "workout_classes": workout_classes,
        "class_counts": class_counts,
        "training_data_path": str(training_path),
        "training_npz_path": global_npz,
        "workout_json_paths": workout_json_paths,
        "workout_npz_paths": workout_npz_paths,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract per-video frame poses (native frame stream) and build a displacement "
            "dataset with one row per frame."
        )
    )
    parser.add_argument("input_path", type=str, help="Video file or workout directory")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help='Output root (default: "<VideosParent>/output_per_frame")',
    )
    parser.add_argument("--save-frames", action="store_true", default=False)
    parser.add_argument("--batch", action="store_true", help="Treat input as workout directory")
    parser.add_argument("--workout-class", type=str, default=None, help="Label for single video")
    parser.add_argument("--num-processes", "-p", type=int, default=1, metavar="N", help="0 = auto")
    parser.add_argument("--normalize-pose", action="store_true", default=False)
    parser.add_argument(
        "--verbose-pool-workers",
        action="store_true",
        default=False,
    )

    parser.add_argument("--displaced-joints-n", type=str, default="3", metavar="N|MIN-MAX")
    parser.add_argument(
        "--max-abs-displacement-xyz",
        type=str,
        default="0.02,0.02,0.02",
        metavar="X,Y,Z",
    )
    parser.add_argument("--class-encoding", choices=("int", "onehot"), default="int")
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--save-displacement-debug-metadata", action="store_true", default=False)
    parser.add_argument("--drop-z", action="store_true", default=False)
    parser.add_argument(
        "--skip-undetected-frames",
        action="store_true",
        default=False,
        help="Exclude frames with no pose detection from the NPZ (JSON still lists all frames).",
    )
    parser.add_argument(
        "--displacement-probability",
        type=float,
        default=1.0,
        metavar="P",
        help=(
            "Per frame row, probability of applying a random displacement (0–1). "
            "If a row is skipped, X uses the clean pose and y is all zeros. Default: 1.0."
        ),
    )

    args = parser.parse_args()

    try:
        displaced_joints_n_range = _parse_int_range(args.displaced_joints_n)
    except Exception:
        print('Error: --displaced-joints-n must be like "3" or "2-5"')
        return 1

    try:
        parts = [float(x.strip()) for x in str(args.max_abs_displacement_xyz).split(",")]
        if len(parts) != 3:
            raise ValueError
        max_abs = (parts[0], parts[1], parts[2])
    except Exception:
        print('Error: --max-abs-displacement-xyz must be like "0.02,0.02,0.02"')
        return 1

    p_disp = float(args.displacement_probability)
    if p_disp < 0.0 or p_disp > 1.0 or not np.isfinite(p_disp):
        print("Error: --displacement-probability must be a finite number in [0, 1]")
        return 1

    inp = Path(args.input_path)
    is_batch = args.batch or (inp.is_dir() and not inp.suffix)

    try:
        if is_batch:
            if not inp.is_dir():
                print(f"Error: not a directory: {inp}")
                return 1
            process_workout_directory(
                videos_dir=inp,
                output_dir=Path(args.output) if args.output else None,
                save_frames=args.save_frames,
                video_extensions=(".mp4", ".mov", ".MOV", ".avi", ".mkv"),
                num_processes=args.num_processes,
                normalize_pose=args.normalize_pose,
                displaced_joints_n_range=displaced_joints_n_range,
                max_abs_displacement_xyz=max_abs,
                class_encoding=args.class_encoding,
                random_seed=args.random_seed,
                save_debug_metadata=args.save_displacement_debug_metadata,
                drop_z=args.drop_z,
                skip_undetected=args.skip_undetected_frames,
                displacement_probability=p_disp,
                verbose_pool_workers=args.verbose_pool_workers,
            )
        else:
            if not inp.is_file():
                print(f"Error: not a file: {inp}")
                return 1
            wc = args.workout_class
            if not wc:
                print("Error: single-video mode requires --workout-class")
                return 1
            out = Path(args.output or "output_per_frame")
            out.mkdir(parents=True, exist_ok=True)
            rec = process_one_video_file(
                video_path=inp,
                output_dir=out,
                save_frames=args.save_frames,
                workout_class=wc,
                normalize_pose=args.normalize_pose,
            )
            save_video_json(out / f"{inp.stem}_pose_frames.json", rec)
            npz_path = save_displacement_npz_from_video_records(
                video_records=[rec],
                class_names=[wc],
                output_dir=out,
                displaced_joints_n_range=displaced_joints_n_range,
                max_abs_displacement_xyz=max_abs,
                class_encoding=args.class_encoding,
                random_seed=args.random_seed,
                save_debug_metadata=args.save_displacement_debug_metadata,
                drop_z=args.drop_z,
                skip_undetected=args.skip_undetected_frames,
                displacement_probability=p_disp,
            )
            print(f"Saved JSON: {out / (inp.stem + '_pose_frames.json')}")
            print(f"Saved NPZ: {npz_path}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
