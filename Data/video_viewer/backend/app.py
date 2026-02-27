#!/usr/bin/env python3
"""
Flask backend server for video viewer webapp.
Serves videos and provides API endpoints to list videos.
"""

from flask import Flask, send_file, jsonify, send_from_directory, request, Response
from flask_cors import CORS
from pathlib import Path
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Path to Videos directory
VIDEOS_DIR = Path(__file__).parent.parent.parent / 'Videos'
IGNORE_FILE = VIDEOS_DIR / '.ignore_videos.txt'

def load_ignored_videos():
    """Load the list of ignored videos from the ignore file."""
    ignored = set()
    if IGNORE_FILE.exists():
        with open(IGNORE_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    ignored.add(line)
    return ignored

def save_ignored_videos(ignored_set):
    """Save the list of ignored videos to the ignore file."""
    IGNORE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(IGNORE_FILE, 'w') as f:
        f.write("# Ignore list for videos\n")
        f.write("# One video path per line, format: workout_type/video_name.mp4\n")
        f.write("# Example:\n")
        f.write("# barbell biceps curl/barbell biceps curl_1.mp4\n\n")
        for video_path in sorted(ignored_set):
            f.write(f"{video_path}\n")

@app.route('/api/videos', methods=['GET'])
def list_videos():
    """List all videos organized by workout type."""
    videos_data = {}
    ignored_videos = load_ignored_videos()
    
    if not VIDEOS_DIR.exists():
        return jsonify({'error': 'Videos directory not found'}), 404
    
    # Supported video extensions
    video_extensions = {'.mp4', '.mov', '.MOV', '.avi', '.mkv'}
    
    # Iterate through workout type directories
    for workout_dir in sorted(VIDEOS_DIR.iterdir()):
        if workout_dir.is_dir() and 'disabled' not in workout_dir.name:
            workout_name = workout_dir.name
            videos = []
            
            # Find all video files in this directory
            for video_file in sorted(workout_dir.iterdir()):
                if video_file.is_file() and video_file.suffix.lower() in video_extensions:
                    full_path = f"{workout_name}/{video_file.name}"
                    is_ignored = full_path in ignored_videos
                    
                    videos.append({
                        'name': video_file.name,
                        'path': f"/api/video/{workout_name}/{video_file.name}",
                        'workout_type': workout_name,
                        'full_path': full_path,
                        'ignored': is_ignored
                    })
            
            if videos:
                videos_data[workout_name] = videos
    
    # Flatten all videos into a single list for easy navigation
    all_videos = []
    for workout_name, videos in videos_data.items():
        for video in videos:
            all_videos.append(video)
    
    return jsonify({
        'by_workout': videos_data,
        'all_videos': all_videos,
        'total_count': len(all_videos)
    })

@app.route('/api/video/<workout_type>/<filename>', methods=['GET'])
def serve_video(workout_type, filename):
    """Serve a video file with range request support for streaming."""
    video_path = VIDEOS_DIR / workout_type / filename
    
    if not video_path.exists():
        return jsonify({'error': 'Video not found'}), 404
    
    # Support range requests for video streaming
    range_header = request.headers.get('Range', None)
    if not range_header:
        return send_file(str(video_path))
    
    import os
    file_size = os.path.getsize(video_path)
    byte1 = 0
    byte2 = file_size - 1
    
    range_match = range_header.replace('bytes=', '').split('-')
    if range_match[0]:
        byte1 = int(range_match[0])
    if range_match[1]:
        byte2 = int(range_match[1])
    
    length = byte2 - byte1 + 1
    
    def generate():
        with open(video_path, 'rb') as f:
            f.seek(byte1)
            data = f.read(length)
            yield data
    
    rv = Response(generate(), 206, mimetype='video/mp4', direct_passthrough=True)
    rv.headers.add('Content-Range', f'bytes {byte1}-{byte2}/{file_size}')
    rv.headers.add('Accept-Ranges', 'bytes')
    rv.headers.add('Content-Length', str(length))
    
    return rv

@app.route('/api/video-info/<workout_type>/<filename>', methods=['GET'])
def get_video_info(workout_type, filename):
    """Get information about a video file."""
    video_path = VIDEOS_DIR / workout_type / filename
    
    if not video_path.exists():
        return jsonify({'error': 'Video not found'}), 404
    
    stat = video_path.stat()
    full_path = f"{workout_type}/{filename}"
    ignored_videos = load_ignored_videos()
    
    return jsonify({
        'name': filename,
        'workout_type': workout_type,
        'size': stat.st_size,
        'path': f"/api/video/{workout_type}/{filename}",
        'full_path': full_path,
        'ignored': full_path in ignored_videos
    })

@app.route('/api/video/toggle-ignore', methods=['POST'])
def toggle_ignore_video():
    """Toggle ignore status for a video."""
    data = request.get_json()
    if not data or 'full_path' not in data:
        return jsonify({'error': 'full_path is required'}), 400
    
    full_path = data['full_path']
    ignored_videos = load_ignored_videos()
    
    if full_path in ignored_videos:
        ignored_videos.remove(full_path)
        is_ignored = False
    else:
        ignored_videos.add(full_path)
        is_ignored = True
    
    save_ignored_videos(ignored_videos)
    
    return jsonify({
        'full_path': full_path,
        'ignored': is_ignored
    })

if __name__ == '__main__':
    print(f"Starting video viewer backend...")
    print(f"Videos directory: {VIDEOS_DIR.absolute()}")
    print(f"Videos directory exists: {VIDEOS_DIR.exists()}")
    app.run(debug=True, port=5003, host='0.0.0.0')
