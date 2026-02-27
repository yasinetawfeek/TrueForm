#!/usr/bin/env python3
"""
Real-time Workout Classifier Web App

Flask backend for real-time pose detection and workout classification using webcam.
"""

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import json
from collections import deque
import sys
import time
from pathlib import Path

# Add parent directory to path to import VideoPoseExtractor
sys.path.insert(0, str(Path(__file__).parent.parent / 'Data'))
from video_pose_extractor import VideoPoseExtractor, KEPT_LANDMARK_INDICES, NUM_LANDMARKS

# Import model loader
from model_loader import ModelLoader

app = Flask(__name__)
CORS(app)

# Global variables
pose_extractor = None
model_loader = None
current_model_name = None
current_model = None
current_class_names = None
current_norm_params = None
sequence_buffer = deque(maxlen=15)  # Buffer for 15 frames (1 second at 15fps)
last_frame_time = None
target_fps = 15  # Target 15 frames per second for sequences
frame_interval = 1.0 / target_fps  # Time between frames (1/15 seconds)
sequence_step_size = 15  # Number of frames to advance before creating new sequence (15 = no overlap, 1 = max overlap)
frames_since_last_prediction = 0  # Counter for step size
latest_prediction = None  # Store latest prediction for API access

# Skeleton connections for visualization
POSE_CONNECTIONS = [
    (0, 1),   # Shoulders
    (0, 2), (2, 4),   # Left arm
    (1, 3), (3, 5),   # Right arm
    (0, 6), (1, 7), (6, 7),  # Torso
    (6, 8), (8, 10),  # Left leg
    (7, 9), (9, 11),  # Right leg
]


def init_pose_extractor():
    """Initialize MediaPipe pose extractor."""
    global pose_extractor
    if pose_extractor is None:
        pose_extractor = VideoPoseExtractor(fps=15, sequence_duration=1.0)
    return pose_extractor


def init_model_loader():
    """Initialize model loader."""
    global model_loader
    if model_loader is None:
        model_loader = ModelLoader()
    return model_loader


def normalize_landmarks_to_hip_center(landmarks):
    """Normalize landmarks relative to hip center."""
    if len(landmarks) < 33:
        return landmarks
    
    LEFT_HIP_IDX = 23
    RIGHT_HIP_IDX = 24
    
    left_hip = landmarks[LEFT_HIP_IDX]
    right_hip = landmarks[RIGHT_HIP_IDX]
    
    hip_center = {
        'x': (left_hip['x'] + right_hip['x']) / 2.0,
        'y': (left_hip['y'] + right_hip['y']) / 2.0,
        'z': (left_hip['z'] + right_hip['z']) / 2.0
    }
    
    normalized = []
    for landmark in landmarks:
        normalized.append({
            'x': landmark['x'] - hip_center['x'],
            'y': landmark['y'] - hip_center['y'],
            'z': landmark['z'] - hip_center['z']
        })
    
    return normalized


def extract_pose_from_frame(frame):
    """Extract pose landmarks from a frame."""
    extractor = init_pose_extractor()
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame
    results = extractor.pose.process(rgb_frame)
    
    landmarks_data = {
        'landmarks': [],
        'detected': False
    }
    
    if results.pose_landmarks:
        landmarks_data['detected'] = True
        
        all_landmarks = []
        for landmark in results.pose_landmarks.landmark:
            all_landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z
            })
        
        # Normalize relative to hip center
        all_landmarks = normalize_landmarks_to_hip_center(all_landmarks)
        
        # Filter to keep only the 12 core body landmarks
        landmarks_data['landmarks'] = [all_landmarks[i] for i in KEPT_LANDMARK_INDICES]
    
    return landmarks_data, results.pose_landmarks


def draw_skeleton(frame, landmarks, mp_landmarks=None):
    """Draw skeleton on frame."""
    # Use MediaPipe drawing if available (more accurate)
    if mp_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        mp_drawing.draw_landmarks(
            frame, mp_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )
        return frame
    
    # Fallback: draw using normalized landmarks
    if not landmarks or len(landmarks) < 12:
        return frame
    
    h, w = frame.shape[:2]
    
    # Convert normalized coordinates (centered at origin) to pixel coordinates
    # Find the range of coordinates to scale appropriately
    x_coords = [lm['x'] for lm in landmarks]
    y_coords = [lm['y'] for lm in landmarks]
    
    if not x_coords or not y_coords:
        return frame
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Scale to fit in frame with some padding
    x_range = x_max - x_min if x_max != x_min else 0.1
    y_range = y_max - y_min if y_max != y_min else 0.1
    
    scale = min(w * 0.8 / x_range, h * 0.8 / y_range) if x_range > 0 and y_range > 0 else 1
    
    points = []
    for lm in landmarks:
        x = int((lm['x'] - x_min) * scale + w * 0.1)
        y = int((lm['y'] - y_min) * scale + h * 0.1)
        points.append((x, y))
    
    # Draw connections
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)
    
    # Draw landmarks
    for i, (x, y) in enumerate(points):
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
    
    return frame


def make_prediction(sequence_array):
    """Make prediction on a sequence."""
    global current_model, current_class_names, current_norm_params, model_loader, current_model_name
    
    if current_model is None or current_class_names is None:
        return None
    
    try:
        prediction = model_loader.predict(current_model_name, sequence_array)
        return prediction
    except Exception as e:
        print(f"Prediction error: {e}")
        return None


def generate_frames():
    """Generate video frames from webcam with proper 15fps sampling for sequences."""
    global sequence_buffer, current_model_name, last_frame_time, frame_interval
    global sequence_step_size, frames_since_last_prediction
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize timing
    last_frame_time = time.time()
    frames_since_last_prediction = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = time.time()
        elapsed = current_time - last_frame_time
        
        # Extract pose from every frame for display (smooth video)
        landmarks_data, mp_landmarks = extract_pose_from_frame(frame)
        
        # Only add to sequence buffer at 15fps (every ~0.067 seconds)
        # This ensures we create sequences with exactly 15 frames per second
        if elapsed >= frame_interval:
            # Add to sequence buffer
            if landmarks_data['detected']:
                sequence_buffer.append(landmarks_data['landmarks'])
            else:
                sequence_buffer.append([{'x': 0, 'y': 0, 'z': 0}] * NUM_LANDMARKS)
            
            last_frame_time = current_time
            frames_since_last_prediction += 1
            
            # Make prediction if we have a full sequence (15 frames) AND
            # we've advanced by the step size (controls overlap)
            if len(sequence_buffer) == 15 and current_model_name and frames_since_last_prediction >= sequence_step_size:
                # Convert buffer to numpy array
                sequence_array = np.zeros((15, NUM_LANDMARKS, 3), dtype=np.float32)
                for i, landmarks in enumerate(sequence_buffer):
                    for j, lm in enumerate(landmarks):
                        sequence_array[i, j, 0] = lm['x']
                        sequence_array[i, j, 1] = lm['y']
                        sequence_array[i, j, 2] = lm['z']
                
                # Make prediction
                prediction = make_prediction(sequence_array)
                if prediction:
                    global latest_prediction
                    latest_prediction = prediction  # Store for API access
                
                # Reset counter for next sequence
                frames_since_last_prediction = 0
        
        # Draw skeleton on every frame (for smooth display)
        frame = draw_skeleton(frame.copy(), landmarks_data['landmarks'], mp_landmarks)
        
        # Draw prediction on frame if available
        prediction_text = ""
        if latest_prediction and current_model_name:
            pred_class = latest_prediction['predicted_class']
            confidence = latest_prediction['confidence']
            prediction_text = f"{pred_class}: {confidence:.1%}"
            
            # Draw prediction on frame
            cv2.putText(frame, prediction_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show buffer status and step size
        overlap = 15 - sequence_step_size
        buffer_status = f"Buffer: {len(sequence_buffer)}/15 | Overlap: {overlap} frames"
        cv2.putText(frame, buffer_status, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()


@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of available models."""
    loader = init_model_loader()
    available = loader.get_available_models()
    return jsonify(available)


@app.route('/api/model/select', methods=['POST'])
def select_model():
    """Select a model to use for predictions."""
    global current_model, current_class_names, current_norm_params, current_model_name, model_loader
    
    data = request.json
    model_name = data.get('model_name')
    
    if not model_name:
        return jsonify({'error': 'Model name required'}), 400
    
    try:
        loader = init_model_loader()
        model, class_names, metadata, norm_params = loader.load_model(model_name)
        
        current_model = model
        current_class_names = class_names
        current_norm_params = norm_params
        current_model_name = model_name
        
        return jsonify({
            'success': True,
            'model_name': model_name,
            'class_names': class_names.tolist(),
            'metadata': metadata
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """Make a prediction on a sequence."""
    global current_model_name
    
    if not current_model_name:
        return jsonify({'error': 'No model selected'}), 400
    
    data = request.json
    sequence = np.array(data.get('sequence'))
    
    if sequence is None:
        return jsonify({'error': 'Sequence required'}), 400
    
    try:
        loader = init_model_loader()
        prediction = loader.predict(current_model_name, sequence)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/prediction/latest', methods=['GET'])
def get_latest_prediction():
    """Get the latest prediction from the video stream."""
    global latest_prediction
    
    if latest_prediction is None:
        return jsonify({'prediction': None})
    
    return jsonify(latest_prediction)


@app.route('/api/sequence/step_size', methods=['POST'])
def set_sequence_step_size():
    """Set the step size for sequence overlap (0-14 frames overlap)."""
    global sequence_step_size, frames_since_last_prediction
    
    data = request.json
    overlap = data.get('overlap', 0)
    
    # Validate overlap (0-14)
    overlap = max(0, min(14, int(overlap)))
    
    # Convert overlap to step size
    # overlap = 0 means step_size = 15 (no overlap, new sequence every 15 frames)
    # overlap = 14 means step_size = 1 (max overlap, new sequence every frame)
    sequence_step_size = 15 - overlap
    
    # Reset counter so new setting takes effect immediately
    frames_since_last_prediction = sequence_step_size
    
    return jsonify({
        'success': True,
        'overlap': overlap,
        'step_size': sequence_step_size
    })


@app.route('/api/sequence/step_size', methods=['GET'])
def get_sequence_step_size():
    """Get the current step size setting."""
    global sequence_step_size
    overlap = 15 - sequence_step_size
    return jsonify({
        'overlap': overlap,
        'step_size': sequence_step_size
    })


if __name__ == '__main__':
    # Initialize components
    init_pose_extractor()
    init_model_loader()
    
    print("Starting Flask app...")
    print("Open http://localhost:5002 in your browser")
    
    app.run(debug=True, host='0.0.0.0', port=5002, threaded=True)
