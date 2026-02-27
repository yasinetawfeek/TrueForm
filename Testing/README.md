# Real-time Workout Classifier Web App

A web application that uses your webcam and MediaPipe framework to detect poses in real-time and classify workout exercises using trained machine learning models.

## Features

- 🎥 **Real-time webcam feed** with pose detection using MediaPipe
- 🤖 **Multiple model support**: Choose from RandomForest, XGBoost, Transformer, BiLSTM, or GRU models
- 📊 **Live predictions** with confidence scores and top-3 predictions
- 🦴 **Skeleton visualization** overlaid on the video feed
- 🎨 **Modern web interface** with intuitive controls

## Requirements

- Python 3.8+
- Webcam access
- Trained model files in `AI/workout_classifier/models/`

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure you have trained models in the `AI/workout_classifier/models/` directory. The app will automatically detect available models.

## Usage

1. Start the Flask server:

```bash
python app.py
```

2. Open your web browser and navigate to:

```
http://localhost:5000
```

3. Allow camera access when prompted by your browser.

4. Select a model from the dropdown menu and click "Load Model".

5. Start performing a workout exercise in front of your webcam to see real-time predictions!

## Available Models

The app automatically detects and loads the following model types if available:

- **RandomForest** (`randomforest_workout_classifier.pkl`)
- **XGBoost** (`xgboost_workout_classifier.pkl`)
- **Transformer** (`transformer_workout_classifier.keras`)
- **BiLSTM** (`bilstm_workout_classifier.keras`)
- **GRU** (`gru_workout_classifier.keras`)

## How It Works

1. **Pose Detection**: MediaPipe processes each webcam frame to extract 12 key body landmarks (shoulders, elbows, wrists, hips, knees, ankles).

2. **Sequence Building**: The app buffers 15 frames (representing 0.5 seconds at 30fps) to create a pose sequence.

3. **Normalization**: Landmarks are normalized relative to the hip center for position-invariant classification.

4. **Prediction**: The selected model processes the sequence and returns:
   - Predicted workout class
   - Confidence score
   - Top 3 predictions with probabilities

5. **Visualization**: The skeleton is overlaid on the video feed, and predictions are displayed in real-time.

## Project Structure

```
Testing/
├── app.py                 # Flask backend application
├── model_loader.py        # Model loading and prediction utilities
├── templates/
│   └── index.html        # Frontend HTML template
├── static/
│   └── style.css         # CSS styling
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## API Endpoints

- `GET /` - Main web interface
- `GET /video_feed` - Video stream endpoint
- `GET /api/models` - List available models
- `POST /api/model/select` - Select and load a model
- `POST /api/predict` - Make a prediction on a sequence

## Troubleshooting

### Camera not working
- Make sure you've granted camera permissions in your browser
- Check that no other application is using the webcam
- Try a different browser (Chrome, Firefox, Edge)

### Models not loading
- Verify that model files exist in `AI/workout_classifier/models/`
- Check that class names and metadata JSON files are present
- Ensure TensorFlow/Keras models are compatible with your TensorFlow version

### Performance issues
- Lower the webcam resolution in `app.py` (currently set to 640x480)
- Close other applications using the webcam
- Use a lighter model (RandomForest or XGBoost) instead of deep learning models

## Notes

- The app requires 15 frames (0.5 seconds) of pose data before making predictions
- Predictions update in real-time as you move
- The skeleton overlay helps visualize detected poses
- All pose landmarks are normalized relative to the hip center for better accuracy

## License

Part of the TrueForm project.
