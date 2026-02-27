#!/usr/bin/env python3
"""
Pose Sequence Visualizer with Workout Class Predictions

Visualizes pose landmarks extracted from video sequences and displays
workout class predictions from a trained classifier model.
"""

import json
import argparse
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import sys
import os
import cv2

# Add parent directory to path to import VideoPoseExtractor
sys.path.insert(0, str(Path(__file__).parent.parent / 'Data'))
from video_pose_extractor import VideoPoseExtractor


# Skeleton connections for the 12 kept body landmarks.
# Landmark index mapping (after filtering):
#   0: L shoulder, 1: R shoulder, 2: L elbow, 3: R elbow,
#   4: L wrist,    5: R wrist,    6: L hip,   7: R hip,
#   8: L knee,     9: R knee,    10: L ankle, 11: R ankle
POSE_CONNECTIONS = [
    # Shoulders
    (0, 1),
    # Left arm
    (0, 2), (2, 4),
    # Right arm
    (1, 3), (3, 5),
    # Torso
    (0, 6), (1, 7), (6, 7),
    # Left leg
    (6, 8), (8, 10),
    # Right leg
    (7, 9), (9, 11),
]

# Simplified connections (identical to full set with 12 landmarks)
SIMPLIFIED_CONNECTIONS = POSE_CONNECTIONS

# Number of landmarks after filtering
NUM_LANDMARKS = 12


class PoseVisualizerWithPredictions:
    """Visualizes pose landmarks from video sequences with workout class predictions."""
    
    def __init__(self, video_path: str, model_path: str, use_simplified: bool = True):
        """
        Initialize the visualizer.
        
        Args:
            video_path: Path to video file
            model_path: Path to trained classifier model (.pkl file)
            use_simplified: Use simplified skeleton connections
        """
        self.video_path = Path(video_path)
        self.model_path = Path(model_path)
        self.use_simplified = use_simplified
        self.connections = SIMPLIFIED_CONNECTIONS if use_simplified else POSE_CONNECTIONS
        
        # Validate paths
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Load model and class names
        self.model, self.class_names = self._load_model()
        
        # Extract sequences from video
        print("Extracting pose sequences from video...")
        self.extractor = VideoPoseExtractor()
        self.sequences = self.extractor.process_video(str(self.video_path), save_frames=False)
        
        # Open video file for frame extraction
        self.video_cap = cv2.VideoCapture(str(self.video_path))
        if not self.video_cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        
        # Get video properties
        self.video_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        
        # Make predictions for each sequence
        print("Making predictions for sequences...")
        self.predictions = self._predict_sequences()
        
        # Convert sequences to visualization format
        self.data = self._prepare_visualization_data()
        
        self.current_sequence_idx = 0
        self.current_frame_idx = 0
        self.animation_running = False
    
    def _load_model(self) -> Tuple:
        """Load the trained model and class names."""
        # Load model
        with open(self.model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Try to load class names from model directory
        model_dir = self.model_path.parent
        class_names_path = model_dir / 'class_names.json'
        
        if not class_names_path.exists():
            # Try alternative names
            for alt_name in ['class_names_rf.json', 'class_names.json']:
                alt_path = model_dir / alt_name
                if alt_path.exists():
                    class_names_path = alt_path
                    break
        
        if class_names_path.exists():
            import json
            with open(class_names_path, 'r') as f:
                class_names = json.load(f)
        else:
            # Try to get from model metadata
            metadata_path = model_dir / 'model_metadata.json'
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    class_names = metadata.get('class_names', [])
            else:
                raise FileNotFoundError(f"Could not find class names file in {model_dir}")
        
        print(f"Loaded model: {self.model_path.name}")
        print(f"Number of classes: {len(class_names)}")
        return model, np.array(class_names)
    
    def _extract_sequence_features(self, sequence: Dict) -> np.ndarray:
        """Extract features from a sequence for prediction."""
        frames = sequence['frames']
        features = []
        
        for frame in frames:
            pose = frame['pose']
            if pose['detected'] and pose['landmarks']:
                # Extract x, y, z coordinates for all landmarks
                frame_features = []
                for landmark in pose['landmarks']:
                    frame_features.extend([landmark['x'], landmark['y'], landmark['z']])
                features.append(frame_features)
            else:
                # If pose not detected, use zeros
                features.append([0.0] * (NUM_LANDMARKS * 3))
        
        # Ensure we have exactly 15 frames
        while len(features) < 15:
            features.append([0.0] * (NUM_LANDMARKS * 3))
        features = features[:15]
        
        # Flatten to (15, 12, 3) -> (540,)
        features_array = np.array(features).reshape(-1)
        return features_array
    
    def _predict_sequences(self) -> List[Dict]:
        """Make predictions for all sequences."""
        predictions = []
        
        for seq_idx, sequence in enumerate(self.sequences):
            # Extract features
            features = self._extract_sequence_features(sequence)
            features = features.reshape(1, -1)
            
            # Handle NaN and infinite values
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Make prediction
            probabilities = self.model.predict_proba(features)[0]
            pred_class_idx = np.argmax(probabilities)
            pred_class = self.class_names[pred_class_idx]
            confidence = probabilities[pred_class_idx]
            
            # Get top 3 predictions
            top3_indices = np.argsort(probabilities)[-3:][::-1]
            top3_predictions = [
                {
                    'class': self.class_names[idx],
                    'probability': float(probabilities[idx])
                }
                for idx in top3_indices
            ]
            
            predictions.append({
                'sequence_number': seq_idx,
                'predicted_class': pred_class,
                'confidence': float(confidence),
                'probabilities': probabilities.tolist(),
                'top3': top3_predictions
            })
        
        return predictions
    
    def _prepare_visualization_data(self) -> Dict:
        """Prepare data in visualization format."""
        return {
            'sequences': self.sequences,
            'predictions': self.predictions,
            'fps': 15,
            'frames_per_sequence': 15
        }
    
    def get_current_sequence(self) -> Dict:
        """Get the current sequence."""
        return self.data['sequences'][self.current_sequence_idx]
    
    def get_current_prediction(self) -> Dict:
        """Get the prediction for the current sequence."""
        return self.data['predictions'][self.current_sequence_idx]
    
    def get_current_frame(self) -> Dict:
        """Get the current frame from current sequence."""
        sequence = self.get_current_sequence()
        return sequence['frames'][self.current_frame_idx]
    
    def draw_pose(self, ax, landmarks: List[Dict], detected: bool = True, 
                  image_width: int = None, image_height: int = None):
        """
        Draw pose landmarks and connections on the axes.
        
        Args:
            ax: Matplotlib axes
            landmarks: List of landmark dictionaries with x, y, z
            detected: Whether pose was detected
            image_width: Width of the image in pixels (for coordinate conversion)
            image_height: Height of the image in pixels (for coordinate conversion)
        """
        if not detected or not landmarks:
            if image_width and image_height:
                ax.text(image_width/2, image_height/2, 'No pose detected', 
                       ha='center', va='center', fontsize=16, color='red')
            else:
                ax.text(0.5, 0.5, 'No pose detected', 
                       ha='center', va='center', fontsize=16, color='red')
            return
        
        # Extract x, y coordinates (normalized 0-1)
        x_coords = [lm['x'] for lm in landmarks]
        y_coords = [lm['y'] for lm in landmarks]
        z_coords = [lm['z'] for lm in landmarks]
        
        # Store original coordinates for hover display
        original_coords = [(lm['x'], lm['y'], lm['z']) for lm in landmarks]
        
        # Convert normalized coordinates to pixel coordinates if image dimensions provided
        if image_width and image_height:
            x_coords = [x * image_width for x in x_coords]
            y_coords = [y * image_height for y in y_coords]
            circle_radius = max(3, min(image_width, image_height) / 100)
            text_offset = max(5, min(image_width, image_height) / 50)
        else:
            # Use normalized coordinates (0-1)
            circle_radius = 0.01
            text_offset = 0.02
        
        # Draw connections
        for start_idx, end_idx in self.connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                line_width = 3 if image_width else 2
                ax.plot(
                    [x_coords[start_idx], x_coords[end_idx]],
                    [y_coords[start_idx], y_coords[end_idx]],
                    'b-', linewidth=line_width, alpha=0.7, zorder=5
                )
        
        # Store landmark patches and their data for hover detection
        if not hasattr(ax, '_landmark_patches'):
            ax._landmark_patches = []
            ax._landmark_data = []
        
        # Clear previous patches
        ax._landmark_patches.clear()
        ax._landmark_data.clear()
        
        # Draw landmarks
        for i, (x, y, z) in enumerate(zip(x_coords, y_coords, z_coords)):
            # Color based on depth (z coordinate)
            color = plt.cm.RdYlGn(0.5 + z * 0.5)  # Red for closer, green for farther
            circle = Circle((x, y), circle_radius, color=color, zorder=10, ec='white', linewidth=1)
            ax.add_patch(circle)
            
            # Store patch and original coordinates for hover
            ax._landmark_patches.append((circle, circle_radius))
            ax._landmark_data.append({
                'index': i,
                'original_coords': original_coords[i],
                'display_coords': (x, y, z)
            })
            
            # Label key points: shoulders (0,1), wrists (4,5), hips (6,7), ankles (10,11)
            if i in [0, 1, 4, 5, 6, 7, 10, 11]:
                ax.text(x, y - text_offset, str(i), fontsize=8, ha='center', 
                       color='white', weight='bold', zorder=11)
    
    def _get_frame_image(self, frame_number: int) -> Optional[np.ndarray]:
        """Extract frame image from video file."""
        try:
            # Set video to the specific frame
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
            ret, frame = self.video_cap.read()
            
            if ret and frame is not None:
                # Convert BGR to RGB for matplotlib
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return frame_rgb
        except Exception as e:
            print(f"Warning: Could not load frame {frame_number}: {e}")
        return None
    
    def __del__(self):
        """Cleanup: close video capture."""
        if hasattr(self, 'video_cap'):
            self.video_cap.release()
    
    def _setup_hover_handler(self, fig, ax):
        """Set up hover event handler to show coordinates on landmark hover."""
        # Create annotation for hover tooltip
        annot = ax.annotate('', xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                           bbox=dict(boxstyle='round', fc='w', alpha=0.8, edgecolor='black'),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        annot.set_visible(False)
        
        def update_annot(ind, patch_data):
            """Update annotation with landmark coordinates."""
            x, y = patch_data['display_coords'][0], patch_data['display_coords'][1]
            orig_x, orig_y, orig_z = patch_data['original_coords']
            annot.xy = (x, y)
            text = f"Landmark {patch_data['index']}\n"
            text += f"x: {orig_x:.4f}\n"
            text += f"y: {orig_y:.4f}\n"
            text += f"z: {orig_z:.4f}"
            annot.set_text(text)
            annot.get_bbox_patch().set_facecolor('lightyellow')
            annot.get_bbox_patch().set_alpha(0.9)
        
        def hover(event):
            """Handle mouse hover events."""
            if event.inaxes != ax:
                annot.set_visible(False)
                fig.canvas.draw_idle()
                return
            
            if not hasattr(ax, '_landmark_patches') or not ax._landmark_patches:
                annot.set_visible(False)
                fig.canvas.draw_idle()
                return
            
            # Check if mouse is over any landmark
            found = False
            for (circle, radius), data in zip(ax._landmark_patches, ax._landmark_data):
                center = circle.center
                dist = np.sqrt((event.xdata - center[0])**2 + (event.ydata - center[1])**2)
                
                if dist < radius * 2:  # Make hover area slightly larger than circle
                    update_annot(0, data)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    found = True
                    break
            
            if not found:
                annot.set_visible(False)
                fig.canvas.draw_idle()
        
        fig.canvas.mpl_connect("motion_notify_event", hover)
        return annot
    
    def create_frame_view(self, frame_data: Dict, ax, prediction: Dict = None):
        """Create a single frame visualization with prediction overlay."""
        ax.clear()
        
        # Try to load the actual frame image from video
        frame_number = frame_data.get('original_frame_number', frame_data.get('frame_number', 0))
        frame_image = self._get_frame_image(frame_number)
        
        image_width = None
        image_height = None
        
        if frame_image is not None:
            image_height, image_width = frame_image.shape[:2]
            ax.imshow(frame_image, aspect='auto', origin='upper')
            ax.set_xlim(0, image_width)
            ax.set_ylim(image_height, 0)  # Inverted y-axis (image coordinates)
            ax.set_aspect('equal')
        else:
            # Fallback to normalized coordinates if frame can't be loaded
            ax.set_xlim(0, 1)
            ax.set_ylim(1, 0)  # Inverted y-axis (image coordinates)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        
        # Title with prediction info
        title = f"Frame {frame_data['frame_number']} | Time: {frame_data['timestamp']:.2f}s"
        if prediction:
            title += f"\nPredicted: {prediction['predicted_class']} ({prediction['confidence']:.2%})"
        ax.set_title(title, fontsize=12)
        
        pose = frame_data['pose']
        self.draw_pose(ax, pose['landmarks'], pose['detected'], image_width, image_height)
    
    def show_sequence(self, sequence_idx: Optional[int] = None, 
                     frame_idx: Optional[int] = None):
        """Display a specific sequence and frame with prediction."""
        if sequence_idx is not None:
            self.current_sequence_idx = sequence_idx
        if frame_idx is not None:
            self.current_frame_idx = frame_idx
        
        sequence = self.get_current_sequence()
        if self.current_frame_idx >= len(sequence['frames']):
            self.current_frame_idx = len(sequence['frames']) - 1
        
        frame = self.get_current_frame()
        prediction = self.get_current_prediction()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        self.create_frame_view(frame, ax, prediction)
        
        # Set up hover handler
        self._setup_hover_handler(fig, ax)
        
        # Add sequence info and prediction details
        info_text = (
            f"Sequence {sequence['sequence_number']} | "
            f"Frame {self.current_frame_idx + 1}/{len(sequence['frames'])}\n"
            f"Time: {sequence['start_time']:.2f}s - {sequence['end_time']:.2f}s"
        )
        
        # Add prediction box
        prediction_text = f"\n\nPrediction:\n"
        prediction_text += f"Class: {prediction['predicted_class']}\n"
        prediction_text += f"Confidence: {prediction['confidence']:.2%}\n"
        prediction_text += f"\nTop 3 Predictions:\n"
        for i, pred in enumerate(prediction['top3'], 1):
            prediction_text += f"{i}. {pred['class']}: {pred['probability']:.2%}\n"
        
        fig.suptitle(info_text, fontsize=14, fontweight='bold')
        
        # Add text box with predictions
        fig.text(0.02, 0.02, prediction_text, fontsize=10, 
                verticalalignment='bottom', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def animate_sequence(self, sequence_idx: Optional[int] = None, 
                        interval: int = 100):
        """
        Animate through frames in a sequence with prediction.
        
        Args:
            sequence_idx: Sequence to animate (None for current)
            interval: Animation interval in milliseconds
        """
        if sequence_idx is not None:
            self.current_sequence_idx = sequence_idx
        
        sequence = self.get_current_sequence()
        prediction = self.get_current_prediction()
        num_frames = len(sequence['frames'])
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Set up hover handler
        self._setup_hover_handler(fig, ax)
        
        def animate(frame_idx):
            ax.clear()
            frame = sequence['frames'][frame_idx]
            self.create_frame_view(frame, ax, prediction)
            
            info_text = (
                f"Sequence {sequence['sequence_number']} | "
                f"Frame {frame_idx + 1}/{num_frames}\n"
                f"Time: {sequence['start_time']:.2f}s - {sequence['end_time']:.2f}s"
            )
            fig.suptitle(info_text, fontsize=14, fontweight='bold')
            
            # Add prediction box
            prediction_text = f"\n\nPrediction:\n"
            prediction_text += f"Class: {prediction['predicted_class']}\n"
            prediction_text += f"Confidence: {prediction['confidence']:.2%}\n"
            prediction_text += f"\nTop 3 Predictions:\n"
            for i, pred in enumerate(prediction['top3'], 1):
                prediction_text += f"{i}. {pred['class']}: {pred['probability']:.2%}\n"
            
            fig.text(0.02, 0.02, prediction_text, fontsize=10, 
                    verticalalignment='bottom', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.8))
        
        anim = animation.FuncAnimation(
            fig, animate, frames=num_frames,
            interval=interval, repeat=True
        )
        
        plt.tight_layout()
        plt.show()
        return anim
    
    def interactive_viewer(self):
        """Create an interactive viewer with keyboard controls and predictions."""
        sequence = self.get_current_sequence()
        prediction = self.get_current_prediction()
        num_frames = len(sequence['frames'])
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Set up hover handler
        self._setup_hover_handler(fig, ax)
        
        def update_display():
            ax.clear()
            frame = self.get_current_frame()
            current_prediction = self.get_current_prediction()
            self.create_frame_view(frame, ax, current_prediction)
            
            info_text = (
                f"Sequence {self.current_sequence_idx + 1}/{len(self.data['sequences'])} | "
                f"Frame {self.current_frame_idx + 1}/{num_frames}\n"
                f"Time: {sequence['start_time']:.2f}s - {sequence['end_time']:.2f}s\n"
                f"Controls: ←/→ (frame), ↑/↓ (sequence), Space (animate), Q (quit)"
            )
            fig.suptitle(info_text, fontsize=12, fontweight='bold')
            
            # Add prediction box
            prediction_text = f"\n\nPrediction:\n"
            prediction_text += f"Class: {current_prediction['predicted_class']}\n"
            prediction_text += f"Confidence: {current_prediction['confidence']:.2%}\n"
            prediction_text += f"\nTop 3 Predictions:\n"
            for i, pred in enumerate(current_prediction['top3'], 1):
                prediction_text += f"{i}. {pred['class']}: {pred['probability']:.2%}\n"
            
            fig.text(0.02, 0.02, prediction_text, fontsize=10, 
                    verticalalignment='bottom', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.8))
            
            fig.canvas.draw()
        
        def on_key(event):
            nonlocal sequence, prediction, num_frames
            
            if event.key == 'right' or event.key == 'd':
                # Next frame
                if self.current_frame_idx < num_frames - 1:
                    self.current_frame_idx += 1
                update_display()
            
            elif event.key == 'left' or event.key == 'a':
                # Previous frame
                if self.current_frame_idx > 0:
                    self.current_frame_idx -= 1
                update_display()
            
            elif event.key == 'up' or event.key == 'w':
                # Next sequence
                if self.current_sequence_idx < len(self.data['sequences']) - 1:
                    self.current_sequence_idx += 1
                    sequence = self.get_current_sequence()
                    prediction = self.get_current_prediction()
                    num_frames = len(sequence['frames'])
                    self.current_frame_idx = 0
                update_display()
            
            elif event.key == 'down' or event.key == 's':
                # Previous sequence
                if self.current_sequence_idx > 0:
                    self.current_sequence_idx -= 1
                    sequence = self.get_current_sequence()
                    prediction = self.get_current_prediction()
                    num_frames = len(sequence['frames'])
                    self.current_frame_idx = 0
                update_display()
            
            elif event.key == ' ':
                # Toggle animation
                if not self.animation_running:
                    self.animate_sequence(self.current_sequence_idx)
            
            elif event.key == 'q':
                plt.close(fig)
        
        fig.canvas.mpl_connect('key_press_event', on_key)
        update_display()
        plt.tight_layout()
        plt.show()


def main():
    """Main function for the visualizer."""
    parser = argparse.ArgumentParser(
        description='Visualize pose landmarks from video with workout class predictions'
    )
    parser.add_argument(
        'video_path',
        type=str,
        help='Path to video file'
    )
    parser.add_argument(
        'model_path',
        type=str,
        help='Path to trained classifier model (.pkl file)'
    )
    parser.add_argument(
        '-m', '--mode',
        type=str,
        choices=['frame', 'animate', 'interactive'],
        default='interactive',
        help='Visualization mode (default: interactive)'
    )
    parser.add_argument(
        '-s', '--sequence',
        type=int,
        default=0,
        help='Sequence index to display (default: 0)'
    )
    parser.add_argument(
        '-f', '--frame',
        type=int,
        default=0,
        help='Frame index to display (default: 0)'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=100,
        help='Animation interval in milliseconds (default: 100)'
    )
    parser.add_argument(
        '--simplified',
        action='store_true',
        default=True,
        help='Use simplified skeleton connections (default: True)'
    )
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = PoseVisualizerWithPredictions(
        args.video_path, 
        args.model_path,
        use_simplified=args.simplified
    )
    
    # Run visualization based on mode
    if args.mode == 'frame':
        visualizer.show_sequence(args.sequence, args.frame)
    elif args.mode == 'animate':
        visualizer.animate_sequence(args.sequence, args.interval)
    elif args.mode == 'interactive':
        visualizer.current_sequence_idx = args.sequence
        visualizer.current_frame_idx = args.frame
        visualizer.interactive_viewer()


if __name__ == '__main__':
    main()
