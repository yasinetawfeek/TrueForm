#!/usr/bin/env python3
"""
Pose Sequence Visualizer

Visualizes pose landmarks extracted from video sequences.
Supports frame-by-frame viewing, animation, and sequence navigation.
"""

import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image


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


class PoseVisualizer:
    """Visualizes pose landmarks from extracted sequences."""
    
    def __init__(self, data_path: str, use_simplified: bool = True):
        """
        Initialize the visualizer.
        
        Args:
            data_path: Path to JSON file (either main file or sequence file)
            use_simplified: Use simplified skeleton connections
        """
        self.data_path = Path(data_path)
        self.use_simplified = use_simplified
        self.connections = SIMPLIFIED_CONNECTIONS if use_simplified else POSE_CONNECTIONS
        
        # Load data
        self.data = self._load_data()
        self.current_sequence_idx = 0
        self.current_frame_idx = 0
        self.animation_running = False
        
    def _load_data(self) -> Dict:
        """Load pose data from JSON file."""
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        # If it's a sequence file, wrap it in the expected format
        if 'sequences' not in data:
            # Single sequence file
            return {
                'sequences': [data],
                'fps': 30,
                'frames_per_sequence': len(data.get('frames', []))
            }
        
        return data
    
    def get_current_sequence(self) -> Dict:
        """Get the current sequence."""
        return self.data['sequences'][self.current_sequence_idx]
    
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
            # Scale circle size based on image size
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
    
    def create_frame_view(self, frame_data: Dict, ax):
        """Create a single frame visualization with image overlay if available."""
        ax.clear()
        
        # Try to load frame image if available
        frame_image_path = frame_data.get('frame_image')
        image = None
        image_width = None
        image_height = None
        
        if frame_image_path:
            frame_path = Path(frame_image_path)
            # If path is relative, try relative to data file location
            if not frame_path.is_absolute():
                # Try multiple possible locations
                possible_paths = [
                    self.data_path.parent.parent / frame_path,  # output/video_frames/frame_xxx.jpg
                    self.data_path.parent / frame_path,  # sequences/video_frames/frame_xxx.jpg
                    Path(frame_image_path)  # Try as-is
                ]
                frame_path = None
                for path in possible_paths:
                    if path.exists():
                        frame_path = path
                        break
                if frame_path is None:
                    # Try absolute path from main data file location
                    if 'sequences' in str(self.data_path):
                        # We're viewing a sequence file, go up to output dir
                        base_dir = self.data_path.parent.parent
                    else:
                        # We're viewing main file, use parent
                        base_dir = self.data_path.parent
                    frame_path = base_dir / frame_image_path
            
            if frame_path and frame_path.exists():
                try:
                    image = Image.open(frame_path)
                    image_width, image_height = image.size
                    ax.imshow(image, aspect='auto', origin='upper')
                    ax.set_xlim(0, image_width)
                    ax.set_ylim(image_height, 0)  # Inverted y-axis (image coordinates)
                    ax.set_aspect('equal')
                except Exception as e:
                    print(f"Warning: Could not load image {frame_path}: {e}")
                    image = None
        
        # If no image, use normalized coordinates
        if image is None:
            ax.set_xlim(0, 1)
            ax.set_ylim(1, 0)  # Inverted y-axis (image coordinates)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        
        ax.set_title(
            f"Frame {frame_data['frame_number']} | "
            f"Time: {frame_data['timestamp']:.2f}s",
            fontsize=12
        )
        
        pose = frame_data['pose']
        self.draw_pose(ax, pose['landmarks'], pose['detected'], 
                      image_width, image_height)
    
    def show_sequence(self, sequence_idx: Optional[int] = None, 
                     frame_idx: Optional[int] = None):
        """Display a specific sequence and frame."""
        if sequence_idx is not None:
            self.current_sequence_idx = sequence_idx
        if frame_idx is not None:
            self.current_frame_idx = frame_idx
        
        sequence = self.get_current_sequence()
        if self.current_frame_idx >= len(sequence['frames']):
            self.current_frame_idx = len(sequence['frames']) - 1
        
        frame = self.get_current_frame()
        
        fig, ax = plt.subplots(figsize=(10, 10))
        self.create_frame_view(frame, ax)
        
        # Set up hover handler
        self._setup_hover_handler(fig, ax)
        
        # Add sequence info
        info_text = (
            f"Sequence {sequence['sequence_number']} | "
            f"Frame {self.current_frame_idx + 1}/{len(sequence['frames'])}\n"
            f"Time: {sequence['start_time']:.2f}s - {sequence['end_time']:.2f}s"
        )
        fig.suptitle(info_text, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def animate_sequence(self, sequence_idx: Optional[int] = None, 
                        interval: int = 100):
        """
        Animate through frames in a sequence.
        
        Args:
            sequence_idx: Sequence to animate (None for current)
            interval: Animation interval in milliseconds
        """
        if sequence_idx is not None:
            self.current_sequence_idx = sequence_idx
        
        sequence = self.get_current_sequence()
        num_frames = len(sequence['frames'])
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Set up hover handler
        self._setup_hover_handler(fig, ax)
        
        def animate(frame_idx):
            ax.clear()
            frame = sequence['frames'][frame_idx]
            self.create_frame_view(frame, ax)
            
            info_text = (
                f"Sequence {sequence['sequence_number']} | "
                f"Frame {frame_idx + 1}/{num_frames}\n"
                f"Time: {sequence['start_time']:.2f}s - {sequence['end_time']:.2f}s"
            )
            fig.suptitle(info_text, fontsize=14, fontweight='bold')
        
        anim = animation.FuncAnimation(
            fig, animate, frames=num_frames,
            interval=interval, repeat=True
        )
        
        plt.tight_layout()
        plt.show()
        return anim
    
    def show_all_sequences_grid(self, max_sequences: int = 9):
        """Display multiple sequences in a grid layout."""
        sequences = self.data['sequences'][:max_sequences]
        n_sequences = len(sequences)
        
        # Calculate grid size
        cols = int(np.ceil(np.sqrt(n_sequences)))
        rows = int(np.ceil(n_sequences / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        if n_sequences == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, sequence in enumerate(sequences):
            ax = axes[idx]
            # Show first frame of each sequence
            if sequence['frames']:
                frame = sequence['frames'][0]
                self.create_frame_view(frame, ax)
                ax.set_title(
                    f"Seq {sequence['sequence_number']}\n"
                    f"Frames: {len(sequence['frames'])}",
                    fontsize=10
                )
        
        # Hide unused subplots
        for idx in range(n_sequences, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def interactive_viewer(self):
        """Create an interactive viewer with keyboard controls."""
        sequence = self.get_current_sequence()
        num_frames = len(sequence['frames'])
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Set up hover handler
        self._setup_hover_handler(fig, ax)
        
        def update_display():
            ax.clear()
            frame = self.get_current_frame()
            self.create_frame_view(frame, ax)
            
            info_text = (
                f"Sequence {self.current_sequence_idx + 1}/{len(self.data['sequences'])} | "
                f"Frame {self.current_frame_idx + 1}/{num_frames}\n"
                f"Time: {sequence['start_time']:.2f}s - {sequence['end_time']:.2f}s\n"
                f"Controls: ←/→ (frame), ↑/↓ (sequence), Space (animate), Q (quit)"
            )
            fig.suptitle(info_text, fontsize=12, fontweight='bold')
            fig.canvas.draw()
        
        def on_key(event):
            nonlocal sequence, num_frames
            
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
                    num_frames = len(sequence['frames'])
                    self.current_frame_idx = 0
                update_display()
            
            elif event.key == 'down' or event.key == 's':
                # Previous sequence
                if self.current_sequence_idx > 0:
                    self.current_sequence_idx -= 1
                    sequence = self.get_current_sequence()
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
        description='Visualize pose landmarks from extracted sequences'
    )
    parser.add_argument(
        'data_path',
        type=str,
        help='Path to JSON file (main file or sequence file)'
    )
    parser.add_argument(
        '-m', '--mode',
        type=str,
        choices=['frame', 'animate', 'grid', 'interactive'],
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
    visualizer = PoseVisualizer(args.data_path, use_simplified=args.simplified)
    
    # Run visualization based on mode
    if args.mode == 'frame':
        visualizer.show_sequence(args.sequence, args.frame)
    elif args.mode == 'animate':
        visualizer.animate_sequence(args.sequence, args.interval)
    elif args.mode == 'grid':
        visualizer.show_all_sequences_grid()
    elif args.mode == 'interactive':
        visualizer.current_sequence_idx = args.sequence
        visualizer.current_frame_idx = args.frame
        visualizer.interactive_viewer()


if __name__ == '__main__':
    main()
