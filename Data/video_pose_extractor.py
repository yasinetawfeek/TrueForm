#!/usr/bin/env python3
"""
Video Pose Extractor using MediaPipe

This script processes video files and extracts pose landmarks for each frame,
organizing them into 0.5-second sequences (15 frames at 30fps).

Evolution: Now supports batch processing of videos organized by workout type,
adding class labels for training workout classification and pose optimization models.
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import os
import sys
import logging
import warnings

# Suppress ALL warnings first, before importing other libraries
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Suppress MediaPipe warnings and logs
os.environ['GLOG_minloglevel'] = '3'  # Only show FATAL (suppress INFO, WARNING, ERROR)
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings (if used by MediaPipe)

# Suppress NumPy warnings
np.seterr(all='ignore')

# Suppress Python logging - configure before importing libraries that use logging
logging.basicConfig(level=logging.CRITICAL, format='')
logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger('absl').setLevel(logging.CRITICAL)
logging.getLogger('mediapipe').setLevel(logging.CRITICAL)
logging.getLogger('google').setLevel(logging.CRITICAL)
logging.getLogger('sklearn').setLevel(logging.CRITICAL)
logging.getLogger('numpy').setLevel(logging.CRITICAL)
logging.getLogger('cv2').setLevel(logging.CRITICAL)

# Disable all loggers
for logger_name in ['absl', 'mediapipe', 'google', 'sklearn', 'numpy', 'cv2', 'PIL', 'Pillow']:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.CRITICAL)
    logger.disabled = True
    logger.propagate = False

# Import libraries after warning suppression is set up
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool, cpu_count
from functools import partial

# Suppress tqdm output if needed (but keep progress bars, just suppress warnings)
tqdm.pandas = lambda: None  # Prevent pandas integration warnings



# MediaPipe landmark indices we keep (12 body landmarks).
# Face (0-10), hands (17-22), and feet (29-32) are removed.
# Wrists (15-16) and ankles (27-28) are kept.
KEPT_LANDMARK_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# Human-readable names for the 12 kept landmarks (in order after filtering)
LANDMARK_NAMES = [
    'left_shoulder',   # 0  (was 11)
    'right_shoulder',  # 1  (was 12)
    'left_elbow',      # 2  (was 13)
    'right_elbow',     # 3  (was 14)
    'left_wrist',      # 4  (was 15)
    'right_wrist',     # 5  (was 16)
    'left_hip',        # 6  (was 23)
    'right_hip',       # 7  (was 24)
    'left_knee',       # 8  (was 25)
    'right_knee',      # 9  (was 26)
    'left_ankle',      # 10 (was 27)
    'right_ankle',     # 11 (was 28)
]

NUM_LANDMARKS = len(KEPT_LANDMARK_INDICES)  # 12


class VideoPoseExtractor:
    """Extracts pose landmarks from video files using MediaPipe.
    
    Always produces sequences with exactly 15 frames representing 0.5 seconds,
    regardless of the input video's frame rate. For videos with different FPS:
    - FPS > 30: Skips frames to maintain 15 frames per 0.5 seconds
    - FPS < 30: Duplicates frames to maintain 15 frames per 0.5 seconds
    
    Only the 12 core body landmarks are kept (shoulders, elbows, wrists,
    hips, knees, ankles). Face, hand, and foot landmarks are discarded.
    """
    
    def __init__(self, fps: int = 15, sequence_duration: float = 1.0, 
                 normalize_pose: bool = False, augment_data: bool = False):
        """
        Initialize the pose extractor.
        
        Args:
            fps: Target frames per second for sequences (default: 15, always produces 15 frames per 1.0s)
            sequence_duration: Duration of each sequence in seconds (default: 1.0)
            normalize_pose: If True, normalize landmarks relative to hip center (default: False)
            augment_data: If True, apply data augmentation (mirror, etc.) (default: False)
        """
        self.target_fps = fps  # Always target 15 fps for sequences (15 frames per 1.0s)
        self.sequence_duration = sequence_duration
        self.frames_per_sequence = int(fps * sequence_duration)  # Always 15 frames per sequence
        self.normalize_pose = normalize_pose
        self.augment_data = augment_data
        
        # Initialize MediaPipe Pose (suppress stderr during initialization)
        self.mp_pose = mp.solutions.pose
        
        # Redirect stderr to suppress MediaPipe initialization messages
        old_stderr = sys.stderr
        old_stdout = sys.stdout
        devnull = open(os.devnull, 'w')
        sys.stderr = devnull
        sys.stdout = devnull
        try:
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
        finally:
            devnull.close()
            sys.stderr = old_stderr
            sys.stdout = old_stdout
        
        self.mp_drawing = mp.solutions.drawing_utils
    
    def normalize_landmarks_to_hip_center(self, landmarks: List[Dict]) -> List[Dict]:
        """
        Normalize all pose landmarks relative to the center point of the hips.
        The hip center is calculated as the midpoint between left hip (landmark 23) 
        and right hip (landmark 24).
        
        Args:
            landmarks: List of landmark dictionaries with 'x', 'y', 'z' keys
            
        Returns:
            List of normalized landmark dictionaries
        """
        if len(landmarks) < 33:
            # Not enough landmarks, return as-is
            return landmarks
        
        # MediaPipe pose landmark indices:
        # Left hip: 23, Right hip: 24
        LEFT_HIP_IDX = 23
        RIGHT_HIP_IDX = 24
        
        left_hip = landmarks[LEFT_HIP_IDX]
        right_hip = landmarks[RIGHT_HIP_IDX]
        
        # Calculate hip center (midpoint of left and right hip)
        hip_center = {
            'x': (left_hip['x'] + right_hip['x']) / 2.0,
            'y': (left_hip['y'] + right_hip['y']) / 2.0,
            'z': (left_hip['z'] + right_hip['z']) / 2.0
        }
        
        # Normalize all landmarks by subtracting hip center
        normalized_landmarks = []
        for landmark in landmarks:
            normalized_landmarks.append({
                'x': landmark['x'] - hip_center['x'],
                'y': landmark['y'] - hip_center['y'],
                'z': landmark['z'] - hip_center['z']
            })
        
        return normalized_landmarks
    
    def mirror_landmarks(self, landmarks: List[Dict], visibility: List[float] = None, 
                        presence: List[float] = None) -> Tuple[List[Dict], List[float], List[float]]:
        """
        Mirror/flip landmarks horizontally (swap left and right).
        
        Landmark mapping for the 12 kept landmarks:
        0: L shoulder  <-> 1: R shoulder
        2: L elbow     <-> 3: R elbow
        4: L wrist     <-> 5: R wrist
        6: L hip       <-> 7: R hip
        8: L knee      <-> 9: R knee
        10: L ankle    <-> 11: R ankle
        
        Args:
            landmarks: List of landmark dictionaries with 'x', 'y', 'z' keys
            visibility: Optional list of visibility scores
            presence: Optional list of presence scores
            
        Returns:
            Tuple of (mirrored_landmarks, mirrored_visibility, mirrored_presence)
        """
        if len(landmarks) != NUM_LANDMARKS:
            raise ValueError(f"Expected {NUM_LANDMARKS} landmarks, got {len(landmarks)}")
        
        # Create mirrored landmarks by swapping left/right pairs
        mirrored_landmarks = landmarks.copy()
        mirrored_visibility = visibility.copy() if visibility else None
        mirrored_presence = presence.copy() if presence else None
        
        # Swap left/right pairs and negate x coordinate (mirror horizontally)
        swap_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
        
        for left_idx, right_idx in swap_pairs:
            # Swap landmarks
            left_landmark = mirrored_landmarks[left_idx].copy()
            right_landmark = mirrored_landmarks[right_idx].copy()
            
            # Mirror x coordinate (negate it)
            mirrored_landmarks[left_idx] = {
                'x': -right_landmark['x'],  # Negate x for mirroring
                'y': right_landmark['y'],
                'z': right_landmark['z']
            }
            mirrored_landmarks[right_idx] = {
                'x': -left_landmark['x'],  # Negate x for mirroring
                'y': left_landmark['y'],
                'z': left_landmark['z']
            }
            
            # Swap visibility and presence if provided
            if mirrored_visibility is not None:
                mirrored_visibility[left_idx], mirrored_visibility[right_idx] = \
                    mirrored_visibility[right_idx], mirrored_visibility[left_idx]
            if mirrored_presence is not None:
                mirrored_presence[left_idx], mirrored_presence[right_idx] = \
                    mirrored_presence[right_idx], mirrored_presence[left_idx]
        
        return mirrored_landmarks, mirrored_visibility, mirrored_presence
    
    def _create_mirrored_sequence(self, sequence: Dict, workout_class: str = None) -> Dict:
        """
        Create a mirrored version of a sequence by flipping all landmarks horizontally.
        
        Args:
            sequence: Original sequence dictionary
            workout_class: Workout class label (optional)
            
        Returns:
            Mirrored sequence dictionary
        """
        mirrored_frames = []
        
        for frame in sequence['frames']:
            pose = frame['pose']
            mirrored_frame = frame.copy()
            
            if pose['detected'] and pose['landmarks']:
                # Mirror the landmarks
                mirrored_landmarks, mirrored_visibility, mirrored_presence = \
                    self.mirror_landmarks(
                        pose['landmarks'],
                        pose.get('visibility'),
                        pose.get('presence')
                    )
                
                mirrored_frame['pose'] = {
                    'landmarks': mirrored_landmarks,
                    'visibility': mirrored_visibility,
                    'presence': mirrored_presence,
                    'detected': True
                }
            else:
                # Keep as-is if no pose detected
                mirrored_frame['pose'] = pose.copy()
            
            mirrored_frames.append(mirrored_frame)
        
        # Create mirrored sequence
        mirrored_sequence = {
            'sequence_number': sequence['sequence_number'],
            'start_frame': sequence['start_frame'],
            'end_frame': sequence['end_frame'],
            'start_time': sequence['start_time'],
            'end_time': sequence['end_time'],
            'frames': mirrored_frames,
            'video_fps': sequence['video_fps'],
            'frames_per_sequence': sequence['frames_per_sequence'],
            'video_file': sequence['video_file'],
            'video_name': sequence['video_name'],
            'augmented': True,  # Mark as augmented
            'augmentation_type': 'mirror'
        }
        
        if workout_class:
            mirrored_sequence['workout_class'] = workout_class
        
        return mirrored_sequence
    
    def extract_pose_landmarks(self, frame: np.ndarray) -> Dict:
        """
        Extract pose landmarks from a single frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dictionary containing pose landmarks and visibility scores
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame (suppress all output during processing)
        old_stderr = sys.stderr
        old_stdout = sys.stdout
        devnull = open(os.devnull, 'w')
        sys.stderr = devnull
        sys.stdout = devnull
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = self.pose.process(rgb_frame)
        finally:
            devnull.close()
            sys.stderr = old_stderr
            sys.stdout = old_stdout
        
        # Extract landmarks
        landmarks_data = {
            'landmarks': [],
            'visibility': [],
            'presence': [],
            'detected': False
        }
        
        if results.pose_landmarks:
            landmarks_data['detected'] = True
            
            all_landmarks = []
            all_visibility = []
            all_presence = []
            for landmark in results.pose_landmarks.landmark:
                all_landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z
                })
                all_visibility.append(landmark.visibility)
                all_presence.append(landmark.presence)
            
            # Normalize all 33 landmarks relative to hip center if flag is enabled
            if self.normalize_pose:
                all_landmarks = self.normalize_landmarks_to_hip_center(all_landmarks)
            
            # Filter to keep only the 12 core body landmarks
            landmarks_data['landmarks'] = [all_landmarks[i] for i in KEPT_LANDMARK_INDICES]
            landmarks_data['visibility'] = [all_visibility[i] for i in KEPT_LANDMARK_INDICES]
            landmarks_data['presence'] = [all_presence[i] for i in KEPT_LANDMARK_INDICES]
        
        return landmarks_data
    
    def process_video(self, video_path: str, output_dir: str = None, 
                     save_frames: bool = True, workout_class: str = None) -> List[Dict]:
        """
        Process a video file and extract pose landmarks for each frame.
        Always produces sequences with exactly 15 frames representing 0.5 seconds.
        
        Args:
            video_path: Path to the input video file
            output_dir: Directory to save output files (optional)
            save_frames: Whether to save frame images (default: True)
            workout_class: Workout class label to add to sequences (optional)
            
        Returns:
            List of sequences, where each sequence contains exactly 15 frames with pose data
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Check if video is in ignore list
        # Find the Videos directory (parent of workout directories)
        current_dir = video_path.parent
        videos_dir = current_dir
        # If we're in a workout subdirectory, Videos is the parent
        if current_dir.parent.name == 'Data' or 'Videos' in str(current_dir.parent):
            videos_dir = current_dir.parent if current_dir.name != 'Videos' else current_dir
        
        ignore_file = videos_dir / '.ignore_videos.txt'
        if ignore_file.exists():
            with open(ignore_file, 'r') as f:
                ignored_videos = {line.strip() for line in f if line.strip() and not line.strip().startswith('#')}
            
            # Check if this video should be ignored
            # Format: workout_type/video_name.mp4
            if workout_class:
                video_key = f"{workout_class}/{video_path.name}"
            else:
                # Try to infer workout class from directory structure
                workout_dir = video_path.parent
                if workout_dir != videos_dir:
                    video_key = f"{workout_dir.name}/{video_path.name}"
                else:
                    video_key = video_path.name
            
            if video_key in ignored_videos:
                print(f"Video {video_path.name} is in ignore list, skipping...")
                return []
        
        # Open video file
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Processing video: {video_path.name}")
        print(f"FPS: {actual_fps:.2f}, Total frames: {total_frames}, Resolution: {width}x{height}")
        if workout_class:
            print(f"Workout class: {workout_class}")
        
        # Target: 15 frames per 0.5 seconds = 30 frames per second
        target_fps = self.target_fps
        frames_per_sequence = self.frames_per_sequence  # Always 15 frames per sequence
        
        # Calculate frame sampling strategy
        # We need exactly 15 frames to represent 0.5 seconds
        # Frames available from video for 0.5 seconds = actual_fps * 0.5
        frames_available = actual_fps * self.sequence_duration
        
        # Calculate sampling ratio
        if actual_fps >= target_fps:
            # Skip frames: take every Nth frame
            # For 0.5 seconds: we have (actual_fps * 0.5) frames, need 15
            frame_skip_ratio = frames_available / frames_per_sequence
            frame_skip = max(1, int(round(frame_skip_ratio)))
            frame_duplication = 1
            print(f"FPS >= {self.target_fps} ({actual_fps:.2f}): Skipping frames (taking every {frame_skip} frame)")
        else:
            # Duplicate frames: repeat frames to reach 15 frames
            # For 0.5 seconds: we have (actual_fps * 0.5) frames, need 15
            duplication_ratio = frames_per_sequence / frames_available
            frame_skip = 1
            frame_duplication = max(1, int(round(duplication_ratio)))
            print(f"FPS < 30 ({actual_fps:.2f}): Duplicating frames "
                  f"(each frame repeated {frame_duplication} times to get 15 frames from {frames_available:.1f} frames)")
        
        sequences = []
        current_sequence = []
        frame_count = 0
        processed_frame_count = 0  # Count of frames added to sequences
        frames_in_sequence_from_video = 0  # Count of video frames processed for current sequence
        
        # For fractional duplication ratios, use a pattern
        if actual_fps < target_fps and frames_available > 0:
            duplication_ratio = frames_per_sequence / frames_available
            # Calculate how many frames need extra duplication
            # e.g., for 24fps: 15/12 = 1.25, so 3 out of 12 frames need 2x duplication
            frames_needing_extra = int(round((duplication_ratio - int(duplication_ratio)) * frames_available))
        else:
            frames_needing_extra = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Determine if we should process this frame
            should_process = (frame_count % frame_skip == 0)
            
            if should_process:
                # Extract pose landmarks
                pose_data = self.extract_pose_landmarks(frame)
                
                # Calculate timestamp based on original video frame
                original_timestamp = frame_count / actual_fps
                
                # Determine duplication for this frame
                # For fractional ratios, some frames get extra duplication
                if actual_fps < target_fps:
                    # Check if this frame needs extra duplication
                    frames_in_current_sequence = frames_in_sequence_from_video % int(frames_available)
                    if frames_needing_extra > 0 and frames_in_current_sequence < frames_needing_extra:
                        current_duplication = frame_duplication + 1
                    else:
                        current_duplication = frame_duplication
                else:
                    current_duplication = frame_duplication
                
                # Save frame image if output directory is specified and save_frames is True
                frame_image_path = None
                if output_dir and save_frames:
                    # Create frames directory for this sequence
                    frames_dir = Path(output_dir) / (video_path.stem + '_frames')
                    frames_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save frame image
                    frame_filename = f"frame_{frame_count:06d}.jpg"
                    frame_image_path_full = frames_dir / frame_filename
                    cv2.imwrite(str(frame_image_path_full), frame)
                    
                    # Store relative path for portability
                    frame_image_path = str(Path(video_path.stem + '_frames') / frame_filename)
                
                # Add frame data to sequence (with duplication if needed)
                for dup_idx in range(current_duplication):
                    frame_data = {
                        'frame_number': frame_count,
                        'original_frame_number': frame_count,
                        'timestamp': original_timestamp,
                        'sequence_frame_index': processed_frame_count + dup_idx,
                        'pose': pose_data,
                        'frame_image': str(frame_image_path) if frame_image_path else None
                    }
                    current_sequence.append(frame_data)
                
                processed_frame_count += current_duplication
                frames_in_sequence_from_video += 1
                
                # When we have a complete sequence (exactly 15 frames), save it
                if len(current_sequence) >= frames_per_sequence:
                    # Take exactly 15 frames
                    sequence_frames = current_sequence[:frames_per_sequence]
                    
                    sequence_data = {
                        'sequence_number': len(sequences),
                        'start_frame': sequence_frames[0]['original_frame_number'],
                        'end_frame': sequence_frames[-1]['original_frame_number'],
                        'start_time': sequence_frames[0]['timestamp'],
                        'end_time': sequence_frames[-1]['timestamp'],
                        'frames': sequence_frames,
                        'video_fps': actual_fps,
                        'frames_per_sequence': len(sequence_frames),
                        'video_file': str(video_path),
                        'video_name': video_path.stem
                    }
                    
                    # Add workout class label if provided
                    if workout_class:
                        sequence_data['workout_class'] = workout_class
                    
                    sequences.append(sequence_data)
                    
                    # Apply data augmentation if enabled
                    if self.augment_data:
                        # Create mirrored version of the sequence
                        mirrored_sequence = self._create_mirrored_sequence(sequence_data, workout_class)
                        sequences.append(mirrored_sequence)
                    
                    # Start new sequence (non-overlapping)
                    current_sequence = []
                    processed_frame_count = 0
                    frames_in_sequence_from_video = 0
            
            frame_count += 1
            
            # Progress indicator
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} video frames, "
                      f"created {len(sequences)} sequences...")
        
        # Handle remaining frames if any (only if we have at least some frames)
        if current_sequence:
            # Pad with last frame if needed to reach 15 frames, or use as-is if close
            if len(current_sequence) >= frames_per_sequence // 2:
                # Pad to 15 frames with last frame
                while len(current_sequence) < frames_per_sequence:
                    last_frame = current_sequence[-1].copy()
                    last_frame['sequence_frame_index'] = len(current_sequence)
                    current_sequence.append(last_frame)
                
                sequence_frames = current_sequence[:frames_per_sequence]
                sequence_data = {
                    'sequence_number': len(sequences),
                    'start_frame': sequence_frames[0]['original_frame_number'],
                    'end_frame': sequence_frames[-1]['original_frame_number'],
                    'start_time': sequence_frames[0]['timestamp'],
                    'end_time': sequence_frames[-1]['timestamp'],
                    'frames': sequence_frames,
                    'video_fps': actual_fps,
                    'frames_per_sequence': len(sequence_frames),
                    'video_file': str(video_path),
                    'video_name': video_path.stem
                }
                
                # Add workout class label if provided
                if workout_class:
                    sequence_data['workout_class'] = workout_class
                
                sequences.append(sequence_data)
                
                # Apply data augmentation if enabled
                if self.augment_data:
                    # Create mirrored version of the sequence
                    mirrored_sequence = self._create_mirrored_sequence(sequence_data, workout_class)
                    sequences.append(mirrored_sequence)
        
        cap.release()
        
        print(f"\nExtracted {len(sequences)} sequences from video")
        print(f"Total video frames processed: {frame_count}")
        print(f"Each sequence contains exactly {frames_per_sequence} frames ({self.sequence_duration} seconds)")
        
        # Save output if output directory is specified
        if output_dir:
            self.save_output(sequences, video_path, output_dir, actual_fps)
        
        return sequences
    
    def save_output(self, sequences: List[Dict], video_path: Path, output_dir: str, actual_fps: float):
        """
        Save extracted sequences to JSON file.
        
        Args:
            sequences: List of sequence data
            video_path: Original video file path
            output_dir: Directory to save output
            actual_fps: Actual FPS of the video
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output filename
        output_filename = video_path.stem + '_pose_sequences.json'
        output_path = output_dir / output_filename
        
        # Prepare output data
        output_data = {
            'video_file': str(video_path),
            'video_fps': actual_fps,
            'target_fps': 30.0,
            'sequence_duration': self.sequence_duration,
            'frames_per_sequence': self.frames_per_sequence,  # Always 15 frames
            'total_sequences': len(sequences),
            'sequences': sequences
        }
        
        # Add workout class if present in sequences
        if sequences and 'workout_class' in sequences[0]:
            output_data['workout_class'] = sequences[0]['workout_class']
        
        # Save to JSON
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
        
        print(f"Output saved to: {output_path}")
        
        # Also save individual sequence files
        sequences_dir = output_dir / (video_path.stem + '_sequences')
        sequences_dir.mkdir(exist_ok=True)
        
        for sequence in sequences:
            seq_filename = f"sequence_{sequence['sequence_number']:04d}.json"
            seq_path = sequences_dir / seq_filename
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with open(seq_path, 'w') as f:
                    json.dump(sequence, f, indent=2)
        
        print(f"Individual sequence files saved to: {sequences_dir}")
    
    def extract_pose_features(self, sequence: Dict) -> np.ndarray:
        """
        Extract pose features from a sequence into a numpy array.
        
        Args:
            sequence: Sequence dictionary with frames containing pose data
            
        Returns:
            Array of shape (15, 12, 3) - (frames, landmarks, xyz)
        """
        frames = sequence['frames']
        num_frames = len(frames)
        
        # Initialize array: (frames, landmarks, xyz)
        pose_array = np.zeros((num_frames, NUM_LANDMARKS, 3), dtype=np.float32)
        
        for frame_idx, frame in enumerate(frames):
            pose = frame['pose']
            if pose['detected']:
                for landmark_idx, landmark in enumerate(pose['landmarks']):
                    pose_array[frame_idx, landmark_idx, 0] = landmark['x']
                    pose_array[frame_idx, landmark_idx, 1] = landmark['y']
                    pose_array[frame_idx, landmark_idx, 2] = landmark['z']
        
        return pose_array
    
    def save_training_data_npz(self, sequences: List[Dict], workout_classes: List[str], 
                               output_dir: Path) -> str:
        """
        Convert sequences to NumPy format and save as .npz file.
        
        Args:
            sequences: List of sequence dictionaries
            workout_classes: List of workout class names
            output_dir: Directory to save the .npz file
            
        Returns:
            Path to saved .npz file
        """
        if not sequences:
            raise ValueError("No sequences to save")
        
        # Extract features and labels
        X_list = []
        y_raw = []
        
        for seq in sequences:
            # Extract pose features
            features = self.extract_pose_features(seq)
            X_list.append(features)
            
            # Get workout class label
            workout_class = seq.get('workout_class', '')
            if not workout_class:
                raise ValueError(f"Sequence missing workout_class: {seq.get('sequence_number', 'unknown')}")
            y_raw.append(workout_class)
        
        # Convert to numpy arrays
        X = np.array(X_list, dtype=np.float32)  # Shape: (n_sequences, 15, 12, 3)
        y_raw = np.array(y_raw)
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_raw)
        y_onehot = np.eye(len(label_encoder.classes_), dtype=np.float32)[y_encoded]
        
        # Save as .npz
        npz_path = output_dir / 'training_data.npz'
        np.savez_compressed(
            npz_path,
            X=X,
            y=y_encoded,
            y_onehot=y_onehot,
            y_raw=y_raw,
            class_names=label_encoder.classes_
        )
        
        # Save metadata separately for easy loading
        # Convert NumPy types to native Python types for JSON serialization
        unique_classes, counts = np.unique(y_raw, return_counts=True)
        class_distribution = {str(cls): int(count) for cls, count in zip(unique_classes, counts)}
        
        metadata = {
            'class_names': [str(name) for name in label_encoder.classes_.tolist()],
            'n_samples': int(len(X)),
            'n_classes': int(len(label_encoder.classes_)),
            'sequence_length': int(X.shape[1]),
            'n_landmarks': int(X.shape[2]),
            'n_coords': int(X.shape[3]),
            'feature_shape': [int(dim) for dim in X.shape[1:]],  # (15, 12, 3)
            'class_distribution': class_distribution
        }
        
        metadata_path = output_dir / 'training_data_metadata.json'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"NumPy training data saved to: {npz_path}")
        print(f"Metadata saved to: {metadata_path}")
        
        return str(npz_path)
    
    def process_workout_directory(self, videos_dir: str, output_dir: str = None, 
                                  save_frames: bool = False, 
                                  video_extensions: Tuple[str, ...] = ('.mp4', '.mov', '.MOV', '.avi', '.mkv'),
                                  num_processes: int = 1) -> Dict:
        """
        Process all videos in a directory structure organized by workout type.
        Each subdirectory represents a workout class, and all videos within are labeled with that class.
        
        Args:
            videos_dir: Root directory containing workout type subdirectories
            output_dir: Directory to save output files (default: 'output' in videos_dir parent)
            save_frames: Whether to save frame images (default: False for batch processing)
            video_extensions: Tuple of video file extensions to process
            
        Returns:
            Dictionary containing:
                - 'total_sequences': Total number of sequences extracted
                - 'workout_classes': List of workout classes found
                - 'class_counts': Dictionary mapping workout class to sequence count
                - 'all_sequences': List of all sequences with class labels
                - 'training_data_path': Path to consolidated training data file (JSON)
                - 'training_data_npz_path': Path to NumPy training data file (.npz)
        """
        videos_dir = Path(videos_dir)
        if not videos_dir.exists():
            raise FileNotFoundError(f"Videos directory not found: {videos_dir}")
        
        if output_dir is None:
            output_dir = videos_dir.parent / 'output'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Output directory: {output_dir.absolute()}")
        
        # Load ignored videos list
        ignore_file = videos_dir / '.ignore_videos.txt'
        ignored_videos = set()
        if ignore_file.exists():
            with open(ignore_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        ignored_videos.add(line)
        
        if ignored_videos:
            print(f"Loaded {len(ignored_videos)} videos to ignore from .ignore_videos.txt")
        
        # Find all workout type directories
        workout_dirs = [d for d in videos_dir.iterdir() if d.is_dir() and 'disabled' not in d.name]
        
        if not workout_dirs:
            raise ValueError(f"No workout type subdirectories found in {videos_dir}")
        
        print(f"\n{'='*60}")
        print(f"Processing workout videos from: {videos_dir}")
        print(f"Found {len(workout_dirs)} workout types")
        print(f"{'='*60}\n")
        
        # First pass: count total videos for overall progress bar
        total_videos = 0
        workout_video_counts = {}
        for workout_dir in sorted(workout_dirs):
            video_files = []
            for ext in video_extensions:
                video_files.extend(workout_dir.glob(f'*{ext}'))
            workout_video_counts[workout_dir.name] = len(video_files)
            total_videos += len(video_files)
        
        # Create persistent overall progress bar at the bottom (position 0)
        overall_pbar = tqdm(
            total=total_videos,
            desc="Overall Progress",
            position=0,
            leave=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} videos [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        all_sequences = []
        class_counts = {}
        workout_classes = []
        workout_data_paths = {}
        workout_npz_paths = {}
        
        # Process each workout type directory
        for workout_dir in sorted(workout_dirs):
            workout_class = workout_dir.name
            workout_classes.append(workout_class)
            class_counts[workout_class] = 0
            
            # Create workout-specific output directory
            # Sanitize workout class name for directory name
            safe_workout_name = workout_class.replace(' ', '_').replace('/', '_')
            workout_output_dir = output_dir / safe_workout_name
            workout_output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n{'─'*60}")
            print(f"Processing workout class: {workout_class}")
            print(f"Directory: {workout_dir}")
            print(f"Output directory: {workout_output_dir}")
            print(f"{'─'*60}")
            
            # Find all video files in this directory
            video_files = []
            for ext in video_extensions:
                video_files.extend(workout_dir.glob(f'*{ext}'))
            
            # Filter out ignored videos
            original_count = len(video_files)
            video_files = [
                vf for vf in video_files 
                if f"{workout_class}/{vf.name}" not in ignored_videos
            ]
            ignored_count = original_count - len(video_files)
            if ignored_count > 0:
                print(f"  Skipping {ignored_count} ignored video(s)")
            
            if not video_files:
                print(f"  No video files found in {workout_dir} (after filtering ignored videos)")
                continue
            
            print(f"  Found {len(video_files)} video files")
            
            # Collect sequences for this workout type
            workout_sequences = []
            
            # Determine number of processes to use
            if num_processes == 0:
                effective_processes = max(1, cpu_count() - 1)  # Leave one CPU free
            else:
                effective_processes = max(1, min(num_processes, len(video_files)))
            
            if effective_processes > 1 and len(video_files) > 1:
                # Use multiprocessing
                print(f"  Using {effective_processes} parallel processes")
                
                # Prepare arguments for worker function
                worker_args = [
                    (video_file, workout_output_dir, save_frames, workout_class, 
                     int(self.target_fps), self.sequence_duration,
                     self.normalize_pose, self.augment_data)
                    for video_file in video_files
                ]
                
                # Process videos in parallel
                with Pool(processes=effective_processes) as pool:
                    # Use position 1 for workout-specific progress bar (above overall bar)
                    results = []
                    for result in tqdm(
                        pool.imap(process_single_video_worker, worker_args),
                        total=len(video_files),
                        desc=f"  {workout_class[:30]:<30}",
                        position=1,
                        leave=False
                    ):
                        results.append(result)
                        # Update overall progress bar
                        overall_pbar.update(1)
                
                # Collect results
                for video_path, sequences, error in results:
                    if error:
                        print(f"\n  Error processing {Path(video_path).name}: {error}")
                        # Progress already updated in the loop above
                        continue
                    
                    workout_sequences.extend(sequences)
                    all_sequences.extend(sequences)
                    class_counts[workout_class] += len(sequences)
            else:
                # Use sequential processing (single process)
                # Use position 1 for workout-specific progress bar (above overall bar)
                for video_file in tqdm(video_files, desc=f"  {workout_class[:30]:<30}", position=1, leave=False):
                    try:
                        sequences = self.process_video(
                            str(video_file),
                            output_dir=str(workout_output_dir),
                            save_frames=save_frames,
                            workout_class=workout_class
                        )
                        
                        # Add sequences to collections
                        workout_sequences.extend(sequences)
                        all_sequences.extend(sequences)
                        class_counts[workout_class] += len(sequences)
                        
                        # Update overall progress bar
                        overall_pbar.update(1)
                        
                    except Exception as e:
                        print(f"\n  Error processing {video_file.name}: {e}")
                        # Update overall progress bar even on error
                        overall_pbar.update(1)
                        continue
            
            # Save workout-specific training data
            if workout_sequences:
                # Save workout-specific JSON
                workout_data = {
                    'metadata': {
                        'workout_class': workout_class,
                        'total_sequences': len(workout_sequences),
                        'frames_per_sequence': self.frames_per_sequence,
                        'sequence_duration': self.sequence_duration,
                        'target_fps': self.target_fps
                    },
                    'sequences': workout_sequences
                }
                
                workout_json_path = workout_output_dir / 'training_data.json'
                with open(workout_json_path, 'w') as f:
                    json.dump(workout_data, f, indent=2)
                workout_data_paths[workout_class] = str(workout_json_path)
                
                # Save workout-specific NumPy .npz format
                try:
                    workout_npz_path = self.save_training_data_npz(
                        workout_sequences,
                        [workout_class],
                        workout_output_dir
                    )
                    workout_npz_paths[workout_class] = workout_npz_path
                    print(f"  Saved {workout_class} training data: {len(workout_sequences)} sequences")
                except Exception as e:
                    print(f"  Warning: Failed to save {workout_class} NumPy format: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Save consolidated training data
        training_data = {
            'metadata': {
                'total_sequences': len(all_sequences),
                'total_workout_classes': len(workout_classes),
                'workout_classes': sorted(workout_classes),
                'class_distribution': class_counts,
                'frames_per_sequence': self.frames_per_sequence,
                'sequence_duration': self.sequence_duration,
                'target_fps': self.target_fps
            },
            'sequences': all_sequences
        }
        
        training_data_path = output_dir / 'training_data.json'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(training_data_path, 'w') as f:
                json.dump(training_data, f, indent=2)
        
        # Save training data as NumPy .npz format
        training_data_npz_path = None
        if all_sequences:
            try:
                training_data_npz_path = self.save_training_data_npz(
                    all_sequences,
                    workout_classes,
                    output_dir
                )
            except Exception as e:
                print(f"\nWarning: Failed to save NumPy format: {e}")
                import traceback
                traceback.print_exc()
        
        # Close overall progress bar
        overall_pbar.close()
        
        print(f"\n{'='*60}")
        print(f"Batch Processing Complete!")
        print(f"{'='*60}")
        print(f"Total sequences extracted: {len(all_sequences)}")
        print(f"Workout classes: {len(workout_classes)}")
        print(f"\nClass distribution:")
        for workout_class, count in sorted(class_counts.items()):
            print(f"  {workout_class}: {count} sequences")
        print(f"\nTraining data saved:")
        print(f"  Consolidated JSON: {training_data_path}")
        if training_data_npz_path:
            print(f"  Consolidated NumPy: {training_data_npz_path}")
        print(f"\n  Workout-specific files organized in subdirectories:")
        for workout_class in sorted(workout_classes):
            if workout_class in workout_data_paths:
                print(f"    {workout_class}/")
        print(f"{'='*60}\n")
        
        return {
            'total_sequences': len(all_sequences),
            'workout_classes': workout_classes,
            'class_counts': class_counts,
            'all_sequences': all_sequences,
            'training_data_path': str(training_data_path),
            'training_data_npz_path': training_data_npz_path,
            'workout_data_paths': workout_data_paths,
            'workout_npz_paths': workout_npz_paths
        }


def process_single_video_worker(args_tuple):
    """
    Worker function for multiprocessing. Processes a single video file.
    
    Args:
        args_tuple: Tuple of (video_path, output_dir, save_frames, workout_class, fps, sequence_duration, normalize_pose, augment_data)
    
    Returns:
        Tuple of (video_path, sequences, error) where error is None if successful
    """
    video_path, output_dir, save_frames, workout_class, fps, sequence_duration, normalize_pose, augment_data = args_tuple
    
    try:
        # Create a new extractor instance for this process
        # (MediaPipe can't be shared across processes)
        extractor = VideoPoseExtractor(
            fps=fps, 
            sequence_duration=sequence_duration,
            normalize_pose=normalize_pose,
            augment_data=augment_data
        )
        
        # Process the video
        sequences = extractor.process_video(
            str(video_path),
            output_dir=str(output_dir),
            save_frames=save_frames,
            workout_class=workout_class
        )
        
        return (str(video_path), sequences, None)
    except Exception as e:
        return (str(video_path), [], str(e))


def main():
    """Main function to run the video pose extractor."""
    parser = argparse.ArgumentParser(
        description='Extract pose landmarks from video files using MediaPipe. '
                    'Supports single video processing or batch processing of workout directories.'
    )
    parser.add_argument(
        'input_path',
        type=str,
        help='Path to input video file or directory containing workout type subdirectories'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        metavar='DIR',
        help='Output directory for extracted pose data. If not specified: '
             'defaults to "output/" for single video, or "Videos/../output" for batch processing. '
             'You can specify any directory path (will be created if it doesn\'t exist).'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=15,
        help='Expected frames per second (default: 30)'
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=1.0,
        help='Sequence duration in seconds (default: 0.5)'
    )
    parser.add_argument(
        '--save-frames',
        action='store_true',
        default=False,
        help='Save frame images for visualization (default: False for batch, True for single video)'
    )
    parser.add_argument(
        '--no-save-frames',
        dest='save_frames',
        action='store_false',
        help='Do not save frame images'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process directory of workout videos (auto-detected if input is directory)'
    )
    parser.add_argument(
        '--workout-class',
        type=str,
        default=None,
        help='Workout class label for single video (optional)'
    )
    parser.add_argument(
        '--num-processes', '-p',
        type=int,
        default=1,
        metavar='N',
        help='Number of parallel processes to use for batch processing (default: 1). '
             'Use 0 to auto-detect based on CPU count (will use CPU_count - 1). '
             'Only applies to batch processing mode.'
    )
    parser.add_argument(
        '--normalize-pose',
        action='store_true',
        default=False,
        help='Normalize pose landmarks relative to hip center (default: False)'
    )
    parser.add_argument(
        '--augment-data',
        action='store_true',
        default=False,
        help='Apply data augmentation (mirror sequences) (default: False)'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    
    # Determine if batch processing
    is_batch = args.batch or (input_path.is_dir() and not input_path.suffix)
    
    # Create extractor
    extractor = VideoPoseExtractor(
        fps=args.fps, 
        sequence_duration=args.duration,
        normalize_pose=args.normalize_pose,
        augment_data=args.augment_data
    )
    
    try:
        if is_batch:
            # Batch processing mode
            if not input_path.is_dir():
                print(f"Error: {input_path} is not a directory")
                return 1
            
            result = extractor.process_workout_directory(
                str(input_path),
                args.output,
                save_frames=args.save_frames,
                num_processes=args.num_processes
            )
            print(f"\nSuccessfully processed {result['total_sequences']} sequences from {len(result['workout_classes'])} workout classes!")
        else:
            # Single video processing mode
            if not input_path.is_file():
                print(f"Error: {input_path} is not a file")
                return 1
            
            output_dir = args.output or 'output'
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            print(f"Output directory: {output_path.absolute()}")
            
            sequences = extractor.process_video(
                str(input_path),
                output_dir,
                save_frames=args.save_frames,
                workout_class=args.workout_class
            )
            print(f"\nSuccessfully extracted {len(sequences)} sequences!")
            
    except Exception as e:
        print(f"Error processing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
