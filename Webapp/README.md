# TrueForm Webapp Realtime Replica

This folder contains a DesD-style realtime app replica with:

- Frontend MediaPipe camera + skeleton overlay + correction arrows
- Flask-SocketIO backend
- Runtime-selectable workout classifier and pose-correction model pairs
- Static side muscle overlays (present in UI, no activation/highlight logic)

## Structure

- `backend/main.py`: socket server and inference pipeline
- `backend/model_registry.py`: model loading and caching
- `backend/preprocess.py`: landmark flattening, buffers, and payload formatting
- `frontend/src/pages/WorkoutPage.jsx`: main UI replica

## Backend setup

```bash
cd Webapp
conda run -n true_form pip install -r backend/requirements.txt
npm run dev:backend
```

Server runs on `http://localhost:8001` by default.

If `8001` is busy, pick another port:

```bash
cd Webapp
AI_PORT=8002 npm run dev:backend
```

## Frontend setup

```bash
cd Webapp/frontend
npm install
npm run dev
```

Frontend runs on `http://localhost:5173` and connects to `http://localhost:8001` by default.

To override backend URL:

```bash
cd Webapp/frontend
VITE_AI_URL=http://localhost:8002 npm run dev
```

## Realtime flow

1. Browser captures camera frames and runs MediaPipe Pose in the client.
2. Client emits `pose_data` over Socket.IO with landmarks + selected model pair.
3. Backend predicts workout class from the selected workout model (`bilstm`, `gru`, `transformer`).
4. Backend maps predicted class to pose class id and runs selected correction model (`lstm_embedding`, `tcn_film`, `tft`).
5. Backend returns `pose_corrections`, and frontend overlays correction arrows.

## Notes

- Muscle overlays are intentionally static (uncolored and non-predictive).
- Use the `true_form` conda environment for backend commands.
- If you see TensorFlow / h5py binary errors, reinstall backend deps inside `true_form`.
