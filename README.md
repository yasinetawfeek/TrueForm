# 🏋️ TrueForm

<div align="center">

**An intelligent workout pose analysis system powered by MediaPipe and Machine Learning**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)](https://mediapipe.dev/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![React](https://img.shields.io/badge/React-18+-61dafb.svg)](https://reactjs.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Components](#-components)
- [Supported Workouts](#-supported-workouts)
- [Technology Stack](#-technology-stack)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

TrueForm is a comprehensive workout pose analysis pipeline that extracts human pose landmarks from exercise videos, trains machine learning classifiers to recognize workout types, and provides interactive visualization tools for pose inspection. The system achieves **95.0% accuracy** in classifying 22 different workout exercises using XGBoost.

### Key Capabilities

- 🎥 **Video Pose Extraction**: Extract 12 core body landmarks from workout videos using MediaPipe
- 🤖 **ML Classification**: Train and deploy models (XGBoost, Random Forest, Transformer, BiLSTM, GRU) to classify workout types
- 📊 **Interactive Visualization**: Visualize pose sequences with skeleton overlays and real-time predictions
- 🌐 **Web Applications**: 
  - Real-time webcam workout classifier
  - Video viewer with pose skeleton overlay
- 📈 **Data Pipeline**: Batch processing with parallel support for large video datasets

---

## ✨ Features

### Core Features

- ✅ **Pose Landmark Extraction**: 12 key body points (shoulders, elbows, wrists, hips, knees, ankles)
- ✅ **Position-Invariant Normalization**: All poses normalized relative to hip center
- ✅ **Sequence-Based Processing**: Organizes frames into configurable sequences (default: 15 frames)
- ✅ **Batch Processing**: Parallel processing support for multiple videos
- ✅ **Multiple ML Models**: Support for XGBoost, Random Forest, Transformer, BiLSTM, and GRU
- ✅ **Real-Time Classification**: Webcam-based live workout recognition
- ✅ **Interactive Visualizations**: Navigate, animate, and inspect pose sequences
- ✅ **Web Interface**: Modern React frontend with Flask backend

### Model Performance

| Model | Accuracy | Notes |
|-------|----------|-------|
| **XGBoost** | **95.0%** | Best performing model |
| Random Forest | 92.3% | 200 trees, max_depth=20 |
| Transformer | - | Deep learning model |
| BiLSTM | - | Sequential model |
| GRU | - | Sequential model |

---

## 📁 Project Structure

```
TrueForm/
├── 📂 Data/                          # Data processing pipeline
│   ├── video_pose_extractor.py      # Pose extraction from videos
│   ├── pose_visualizer.py           # Interactive pose visualizer
│   ├── requirements.txt             # Python dependencies
│   ├── Videos/                      # Raw workout videos (organized by type)
│   ├── output/                      # Extracted pose data
│   └── video_viewer/                # Web video viewer application
│       ├── backend/                 # Flask backend
│       └── frontend/                # React frontend
│
├── 📂 AI/                            # Machine learning models
│   ├── pose_visualizer_with_predictions.py
│   └── workout_classifier/          # Model training notebooks
│       ├── XGboost_workout_classifier.ipynb
│       ├── RandomForest_workout_classifier.ipynb
│       ├── Transformer_workout_classifier.ipynb
│       ├── BiLSTM_workout_classifier.ipynb
│       ├── GRU_workout_classifier.ipynb
│       └── models/                 # Trained model files
│
└── 📂 Testing/                       # Real-time webcam classifier
    ├── app.py                       # Flask web application
    ├── model_loader.py              # Model loading utilities
    ├── templates/                   # HTML templates
    └── static/                      # CSS styles
```

---

## 🚀 Installation

### Prerequisites

- **Python 3.8+** with pip
- **Node.js 14+** with npm (for web applications)
- **Webcam** (for real-time classification)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/TrueForm.git
cd TrueForm
```

### Step 2: Install Python Dependencies

#### Data Processing Pipeline

```bash
cd Data
pip install -r requirements.txt
```

**Dependencies:** `opencv-python`, `mediapipe`, `numpy`, `matplotlib`, `Pillow`, `tqdm`

#### Video Viewer Backend

```bash
cd Data/video_viewer/backend
pip install -r requirements.txt
```

#### Real-Time Classifier

```bash
cd Testing
pip install -r requirements.txt
```

**Additional ML Dependencies:** `scikit-learn`, `xgboost`, `pandas`, `seaborn`, `tensorflow`

### Step 3: Install Frontend Dependencies (Optional)

For the video viewer web application:

```bash
cd Data/video_viewer/frontend
npm install
```

---

## 🏃 Quick Start

### 1. Extract Poses from Videos

Process a single video:

```bash
cd Data
python video_pose_extractor.py path/to/video.mp4 -o output/
```

Process all videos in a directory:

```bash
python video_pose_extractor.py Videos/ -o output/ -p 4
```

### 2. Train a Classifier

Open and run the Jupyter notebook:

```bash
cd AI/workout_classifier
jupyter notebook XGboost_workout_classifier.ipynb
```

### 3. Real-Time Classification

Start the webcam classifier:

```bash
cd Testing
python app.py
```

Open `http://localhost:5000` in your browser.

### 4. Video Viewer

Use the startup script:

```bash
cd Data/video_viewer
./start.sh
```

Or manually start backend and frontend:

```bash
# Terminal 1: Backend
cd Data/video_viewer/backend
python app.py

# Terminal 2: Frontend
cd Data/video_viewer/frontend
npm start
```

---

## 🧩 Components

### 1. Video Pose Extractor

Extracts pose landmarks from videos using MediaPipe Pose.

**Key Features:**
- Extracts 12 body landmarks (x, y, z) per frame
- Normalizes positions relative to hip center
- Organizes frames into sequences (default: 15 frames × 1.0s)
- Supports single video or batch processing
- Parallel processing with multiprocessing

**Usage:**

```bash
# Single video
python video_pose_extractor.py video.mp4 -o output/ --workout-class "squat"

# Batch processing
python video_pose_extractor.py Videos/ -o output/ -p 4
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output DIR` | `output/` | Output directory |
| `--fps N` | `15` | Target frames per second |
| `--duration N` | `1.0` | Sequence duration in seconds |
| `-p, --num-processes N` | `1` | Parallel workers (0 = auto-detect) |
| `--workout-class NAME` | `None` | Attach class label |

### 2. Pose Visualizer

Interactive tool to visualize extracted pose sequences.

**Usage:**

```bash
python pose_visualizer.py output/training_data.json
```

**Controls:**
- `←` / `→` or `A` / `D`: Previous / next frame
- `↑` / `↓` or `W` / `S`: Previous / next sequence
- `Space`: Start animation
- `Q`: Quit
- **Hover**: See landmark coordinates

### 3. Workout Classifiers

Train ML models to classify workout types from pose sequences.

**Available Models:**
- **XGBoost** (95.0% accuracy) - Recommended
- **Random Forest** (92.3% accuracy)
- **Transformer** (Deep learning)
- **BiLSTM** (Sequential model)
- **GRU** (Sequential model)

**Dataset:** 16,239 sequences across 22 workout classes

### 4. Real-Time Webcam Classifier

Web application for real-time workout classification using your webcam.

**Features:**
- Live pose detection with MediaPipe
- Multiple model support
- Top-3 predictions with confidence scores
- Skeleton visualization overlay

### 5. Video Viewer

Web application to browse and view videos with pose skeleton overlay.

**Features:**
- Browse videos organized by workout type
- Real-time MediaPipe pose detection
- Video navigation controls
- Play/pause and seek functionality

---

## 🏋️ Supported Workouts

The system currently supports **22 workout exercises**:

| Exercise | Exercise | Exercise |
|----------|----------|----------|
| Barbell Biceps Curl | Bench Press | Chest Fly Machine |
| Deadlift | Decline Bench Press | Hammer Curl |
| Hip Thrust | Incline Bench Press | Lat Pulldown |
| Lateral Raise | Leg Extension | Leg Raises |
| Plank | Pull Up | Push-up |
| Romanian Deadlift | Russian Twist | Shoulder Press |
| Squat | T Bar Row | Tricep Dips |
| Tricep Pushdown | | |

---

## 🛠 Technology Stack

### Backend
- **Python 3.8+**
- **MediaPipe** - Pose detection
- **OpenCV** - Video processing
- **Flask** - Web framework
- **NumPy** - Numerical computing
- **scikit-learn** - Machine learning
- **XGBoost** - Gradient boosting
- **TensorFlow/Keras** - Deep learning

### Frontend
- **React** - UI framework
- **MediaPipe** (Web) - Client-side pose detection
- **CSS3** - Styling

### Data Science
- **Jupyter Notebooks** - Model development
- **Pandas** - Data manipulation
- **Matplotlib** - Visualization
- **Seaborn** - Statistical visualization

---

## 📚 Documentation

Detailed documentation is available in each component's directory:

- **[Data Pipeline Documentation](Data/README.md)** - Pose extraction and visualization
- **[Video Viewer Documentation](Data/video_viewer/README.md)** - Web video viewer setup
- **[Real-Time Classifier Documentation](Testing/README.md)** - Webcam classifier setup

### Data Format

Each pose sequence contains:
- **12 landmarks** (shoulders, elbows, wrists, hips, knees, ankles)
- **3D coordinates** (x, y, z) normalized to hip center
- **15 frames** per sequence (configurable)
- **Visibility and presence scores** for each landmark

### Model Input Format

- **Shape**: `(N, 15, 12, 3)` → Flattened to `(N, 540)`
- **Normalization**: Relative to hip center (position-invariant)
- **Labels**: 22 workout classes (integer-encoded or one-hot)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines for Python code
- Add docstrings to functions and classes
- Include tests for new features
- Update documentation as needed

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **MediaPipe** by Google for pose detection capabilities
- **OpenCV** community for video processing tools
- All contributors and users of this project

---

## 📧 Contact

For questions, issues, or contributions, please open an issue on GitHub.

---

<div align="center">

**Made with ❤️ for fitness enthusiasts and developers**

⭐ Star this repo if you find it helpful!

</div>
