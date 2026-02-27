# Video Pose Viewer

A React web application for viewing videos with MediaPipe pose skeleton overlay. Navigate through all videos in the Videos directory and see real-time pose detection.

## Features

- Browse all videos from the Videos directory organized by workout type
- Real-time MediaPipe pose skeleton overlay on videos
- Video navigation (next/previous)
- Video list sidebar with workout type grouping
- Play/pause controls and seek bar

## Setup

### Prerequisites

- Python 3.7+ with pip
- Node.js 14+ with npm
- Videos directory at `Data/Videos/` (should already exist)

### Quick Start (Recommended)

Use the provided startup script:

```bash
cd Data/video_viewer
./start.sh
```

This will start both the backend and frontend servers automatically.

### Manual Setup

#### Backend Setup

1. Navigate to the backend directory:
```bash
cd Data/video_viewer/backend
```

2. (Optional) Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Start the Flask server:
```bash
python app.py
```

The backend will run on `http://localhost:5000`

#### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd Data/video_viewer/frontend
```

2. Install Node.js dependencies:
```bash
npm install
```

3. Start the React development server:
```bash
npm start
```

The frontend will run on `http://localhost:3000` and should automatically open in your browser.

## Usage

1. Make sure both the backend and frontend servers are running
2. Open your browser and navigate to `http://localhost:3000`
3. The app will automatically load all videos from the `Data/Videos/` directory
4. Click on any video in the sidebar to view it
5. Use the play/pause button to control playback
6. Use the Next/Previous buttons to navigate between videos
7. The MediaPipe pose skeleton will be overlaid on the video in real-time

## Structure

```
video_viewer/
├── backend/
│   ├── app.py              # Flask server
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── App.js          # Main app component
│   │   ├── App.css
│   │   ├── index.js
│   │   ├── index.css
│   │   └── components/
│   │       ├── VideoPlayer.js    # Video player with pose overlay
│   │       ├── VideoPlayer.css
│   │       ├── VideoList.js      # Video list sidebar
│   │       └── VideoList.css
│   └── package.json
└── README.md
```

## Notes

- The backend serves videos from `Data/Videos/` directory
- MediaPipe Pose uses the web version from CDN
- Videos are organized by workout type (subdirectories in Videos/)
- The app automatically skips directories with "disabled" in the name
