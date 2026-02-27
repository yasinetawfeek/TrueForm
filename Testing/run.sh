#!/bin/bash
# Quick start script for the Real-time Workout Classifier Web App

echo "Starting Real-time Workout Classifier Web App..."
echo ""
echo "Make sure you have:"
echo "  1. Installed dependencies: pip install -r requirements.txt"
echo "  2. Trained models in AI/workout_classifier/models/"
echo "  3. Webcam access enabled"
echo ""
echo "Opening http://localhost:5000 in your browser..."
echo "Press Ctrl+C to stop the server"
echo ""

python app.py
