# TrueForm

A workout pose analysis pipeline that extracts human pose landmarks from exercise videos using MediaPipe, trains machine learning classifiers to recognise workout types, and provides interactive visualisation tools for pose inspection.

## Project Structure

```
TrueForm/
├── Data/
│   ├── video_pose_extractor.py      # Pose extraction from videos (single & batch)
│   ├── pose_visualizer.py           # Interactive pose sequence visualiser
│   ├── requirements.txt             # Python dependencies for data pipeline
│   ├── Videos/                      # Raw workout videos organised by exercise type
│   │   ├── barbell biceps curl/
│   │   ├── bench press/
│   │   ├── squat/
│   │   └── ... (22 workout types)
│   └── output/                      # Extracted pose data
│       ├── training_data.npz        # Consolidated NumPy training data
│       ├── training_data.json       # Consolidated JSON training data
│       ├── training_data_metadata.json
│       └── <workout_type>/          # Per-class pose data & sequences
│
├── AI/
│   ├── pose_visualizer_with_predictions.py   # Visualiser with live model predictions
│   └── workout_classifier/
│       ├── XGboost_workout_classifier.ipynb   # XGBoost classifier notebook
│       ├── RandomForest_workout_classifier.ipynb  # Random Forest classifier notebook
│       └── models/
│           ├── xgboost_workout_classifier.pkl     # Trained XGBoost model (95.0% accuracy)
│           ├── randomforest_workout_classifier.pkl # Trained Random Forest model (92.3% accuracy)
│           ├── class_names.json
│           ├── class_names_rf.json
│           ├── model_metadata.json
│           └── model_metadata_rf.json
```

## Supported Workout Classes (22)

barbell biceps curl, bench press, chest fly machine, deadlift, decline bench press, hammer curl, hip thrust, incline bench press, lat pulldown, lateral raise, leg extension, leg raises, plank, pull up, push-up, romanian deadlift, russian twist, shoulder press, squat, t bar row, tricep dips, tricep pushdown

## Installation

```bash
cd Data
pip install -r requirements.txt
```

**Dependencies:** opencv-python, mediapipe, numpy, matplotlib, Pillow, tqdm

For the AI notebooks you will also need: scikit-learn, xgboost, pandas, seaborn, pickle.

## Data Pipeline

### 1. Video Pose Extractor (`Data/video_pose_extractor.py`)

Processes video files using MediaPipe Pose to extract body landmarks per frame, organised into fixed-length sequences. Only the 12 core body landmarks are kept (face, hand, and foot points are discarded; wrists and ankles are kept).

**Key features:**
- Extracts 12 body pose landmarks (x, y, z) per frame — shoulders, elbows, wrists, hips, knees, ankles
- Normalises all landmark positions relative to the hip centre (midpoint of left hip and right hip) — making poses position-invariant
- Organises frames into sequences of configurable length (default: 15 frames × 1.0s)
- Handles varying video FPS by skipping or duplicating frames to hit the target rate
- Supports single video or batch processing of an entire workout directory
- Parallel processing with multiprocessing support
- Outputs both JSON (per-sequence and consolidated) and NumPy `.npz` formats

#### Single Video

```bash
python video_pose_extractor.py path/to/video.mp4
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output DIR` | `output/` | Output directory |
| `--fps N` | `15` | Target frames per second |
| `--duration N` | `1.0` | Sequence duration in seconds |
| `--save-frames` | `False` | Save frame images as JPGs |
| `--workout-class NAME` | `None` | Attach a class label to sequences |

#### Batch Processing (Workout Directory)

Pass a directory where each subdirectory is a workout type containing video files:

```bash
python video_pose_extractor.py Videos/ -o output/
```

The directory structure should look like:

```
Videos/
├── squat/
│   ├── squat_1.mp4
│   └── squat_2.mp4
├── bench press/
│   ├── bench press_1.mp4
│   └── ...
└── ...
```

**Batch-specific options:**

| Flag | Default | Description |
|------|---------|-------------|
| `-p, --num-processes N` | `1` | Parallel workers (0 = auto-detect CPU count - 1) |
| `--batch` | auto | Force batch mode (auto-detected when input is a directory) |

**Batch output:**
- `output/training_data.npz` — consolidated NumPy array of shape `(N, 15, 12, 3)`
- `output/training_data.json` — consolidated JSON with all sequences and labels
- `output/training_data_metadata.json` — dataset statistics and class distribution
- `output/<workout_type>/` — per-class JSON + `.npz` files and individual sequence JSONs

### Pose Normalisation

All landmark coordinates are normalised relative to the **hip centre** — the midpoint between left hip and right hip. Normalisation is performed on the full 33 MediaPipe landmarks first (using original indices 23 and 24), then the result is filtered down to the 12 kept landmarks. After normalisation, the hip centre sits at `(0, 0, 0)` and all other landmarks are expressed as offsets from it. This makes the pose data position-invariant so the model focuses on body shape rather than where the person appears in the frame.

### Output Data Format

Each **sequence** contains:

| Field | Description |
|-------|-------------|
| `sequence_number` | Sequence index |
| `start_frame` / `end_frame` | Frame range in original video |
| `start_time` / `end_time` | Time range in seconds |
| `frames` | Array of frame data (see below) |
| `workout_class` | Class label (batch mode) |

Each **frame** contains:

| Field | Description |
|-------|-------------|
| `frame_number` | Frame index in the video |
| `timestamp` | Time in seconds from video start |
| `pose.landmarks` | 12 landmarks, each with `x`, `y`, `z` (normalised to hip centre) |
| `pose.visibility` | Visibility score per landmark |
| `pose.presence` | Presence score per landmark |
| `pose.detected` | Boolean — whether a pose was detected |

The **NumPy `.npz`** file contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `X` | `(N, 15, 12, 3)` | Pose feature arrays |
| `y` | `(N,)` | Integer-encoded labels |
| `y_onehot` | `(N, 22)` | One-hot encoded labels |
| `y_raw` | `(N,)` | String class names |
| `class_names` | `(22,)` | Ordered class name list |

### Kept Landmarks (12)

From MediaPipe's 33 pose landmarks, only the 12 core body points are kept. Face (0–10), hands (17–22), and feet (29–32) are discarded. Wrists and ankles are retained.

| New Index | Name | Original MediaPipe Index |
|-----------|------|--------------------------|
| 0 | Left shoulder | 11 |
| 1 | Right shoulder | 12 |
| 2 | Left elbow | 13 |
| 3 | Right elbow | 14 |
| 4 | Left wrist | 15 |
| 5 | Right wrist | 16 |
| 6 | Left hip | 23 |
| 7 | Right hip | 24 |
| 8 | Left knee | 25 |
| 9 | Right knee | 26 |
| 10 | Left ankle | 27 |
| 11 | Right ankle | 28 |

---

## Visualisation

### 2. Pose Visualiser (`Data/pose_visualizer.py`)

Visualises extracted pose sequences from JSON files. Supports interactive navigation, animation, and hover-to-inspect landmark coordinates.

```bash
cd Data
python pose_visualizer.py path/to/pose_sequences.json
```

You can point it at a consolidated file or an individual sequence file:

```bash
# Consolidated file (navigate across sequences)
python pose_visualizer.py output/training_data.json

# Individual sequence
python pose_visualizer.py "output/barbell_biceps_curl/barbell biceps curl_1_sequences/sequence_0000.json"
```

**Modes:**

| Mode | Flag | Description |
|------|------|-------------|
| Interactive | `-m interactive` | Navigate with keyboard (default) |
| Single frame | `-m frame -s 0 -f 5` | Show sequence 0, frame 5 |
| Animation | `-m animate -s 0 --interval 100` | Animate a sequence |
| Grid | `-m grid` | Show first frame of multiple sequences in a grid |

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `-s, --sequence N` | `0` | Sequence index |
| `-f, --frame N` | `0` | Frame index |
| `--interval MS` | `100` | Animation interval in milliseconds |
| `--simplified` | `True` | Use simplified skeleton connections |

**Interactive Controls:**

| Key | Action |
|-----|--------|
| `←` / `→` or `A` / `D` | Previous / next frame |
| `↑` / `↓` or `W` / `S` | Previous / next sequence |
| `Space` | Start animation |
| `Q` | Quit |
| **Hover over a landmark** | Shows tooltip with landmark index and (x, y, z) coordinates |

**Visualisation features:**
- Skeleton drawn with connections between landmarks
- Landmarks colour-coded by depth (z-coordinate): red = closer, green = farther
- Key landmarks labelled (shoulders, wrists, hips, ankles)
- Frame images overlaid when available (if `--save-frames` was used during extraction)

### 3. Pose Visualiser with Predictions (`AI/pose_visualizer_with_predictions.py`)

Processes a video file in real time, extracts pose sequences, runs a trained classifier, and displays the predicted workout class with confidence scores overlaid on each frame.

```bash
cd AI
python pose_visualizer_with_predictions.py path/to/video.mp4 path/to/model.pkl
```

**Example:**

```bash
python pose_visualizer_with_predictions.py \
  "../Data/Videos/squat/squat_1.mp4" \
  "workout_classifier/models/xgboost_workout_classifier.pkl"
```

**Modes:** Same as the basic visualiser (`-m interactive`, `-m frame`, `-m animate`).

**Additional features over the basic visualiser:**
- Displays predicted workout class and confidence percentage
- Shows top-3 predictions with probabilities
- Extracts actual video frames as background images
- Hover over landmarks to see (x, y, z) coordinates

---

## AI / Model Training

### 4. Workout Classifiers (`AI/workout_classifier/`)

Two Jupyter notebooks that train workout classifiers on the extracted pose data:

| Notebook | Model | Test Accuracy |
|----------|-------|---------------|
| `XGboost_workout_classifier.ipynb` | XGBoost (Gradient Boosted Trees) | **95.0%** |
| `RandomForest_workout_classifier.ipynb` | Random Forest (200 trees, max_depth=20) | **92.3%** |

Both notebooks:
1. Load `training_data.npz` from `Data/output/`
2. Flatten sequences from `(15, 12, 3)` to `540`-dimensional feature vectors
3. Split into train/test sets
4. Train and evaluate the classifier
5. Save the trained model as `.pkl` and class names as `.json` to `models/`

**Dataset:** 16,239 sequences across 22 workout classes.

### Trained Models

Saved in `AI/workout_classifier/models/`:

| File | Description |
|------|-------------|
| `xgboost_workout_classifier.pkl` | Trained XGBoost model |
| `randomforest_workout_classifier.pkl` | Trained Random Forest model |
| `class_names.json` | Class names (XGBoost) |
| `class_names_rf.json` | Class names (Random Forest) |
| `model_metadata.json` | XGBoost metadata & accuracy |
| `model_metadata_rf.json` | Random Forest metadata & hyperparameters |

---

## End-to-End Workflow

```
1. Collect workout videos → Data/Videos/<workout_type>/

2. Extract poses:
   python Data/video_pose_extractor.py Data/Videos/ -o Data/output/ -p 4

3. Train a classifier:
   Run AI/workout_classifier/XGboost_workout_classifier.ipynb

4. Visualise & predict:
   python AI/pose_visualizer_with_predictions.py video.mp4 AI/workout_classifier/models/xgboost_workout_classifier.pkl
```
