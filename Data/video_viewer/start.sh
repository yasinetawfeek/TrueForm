#!/bin/bash

# Start script for Video Pose Viewer
# This script starts both the backend and frontend servers

echo "Starting Video Pose Viewer..."
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed"
    exit 1
fi

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "Error: node is not installed"
    exit 1
fi

# Start backend in background
echo "Starting backend server..."
cd backend
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
pip install -q -r requirements.txt
python app.py &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 2

# Start frontend
echo "Starting frontend server..."
cd frontend
if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install
fi

PORT=3003 npm start &
FRONTEND_PID=$!
cd ..

echo ""
echo "Backend running on http://localhost:5003 (PID: $BACKEND_PID)"
echo "Frontend running on http://localhost:3003 (PID: $FRONTEND_PID)"
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for user interrupt
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT TERM
wait
