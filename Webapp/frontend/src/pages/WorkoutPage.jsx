import { useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Camera } from "@mediapipe/camera_utils";
import * as poseDetection from "@mediapipe/pose";
import { POSE_CONNECTIONS } from "@mediapipe/pose";
import io from "socket.io-client";
import Model from "react-body-highlighter";
import { ModelType } from "react-body-highlighter";
import { AI_URL } from "../config";

const WORKOUT_MODEL_OPTIONS = ["bilstm", "gru", "transformer"];
const POSE_MODEL_OPTIONS = ["lstm_embedding", "tcn_film", "tft", "desd_pth"];
const JOINT_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
const BODY_CONNECTIONS = POSE_CONNECTIONS.filter(
  ([start, end]) => JOINT_INDICES.includes(start) && JOINT_INDICES.includes(end)
);
const WINDOW_SIZE = 15;
const TARGET_FEED_FPS = 30;
const TARGET_FEED_FRAME_MS = 1000 / TARGET_FEED_FPS;
// Mirror DesD reference arrow tuning.
const SMOOTHING_FACTOR = 0.7;
const CORRECTION_THRESHOLD = 0.01;
const CORRECTION_FADE = 0.8;
const CORRECTION_CLEANUP_THRESHOLD = 0.005;
const DRAW_MIN_VECTOR_PX = 2;
const ARROW_MIN_VECTOR_PX = 5;
const ARROW_EXTENSION_FACTOR = 1.5;
const ARROW_LINE_WIDTH = 6;
const WORKOUT_LABEL_OPTIONS = [
  "barbell biceps curl",
  "bench press",
  "chest fly machine",
  "deadlift",
  "hip thrust",
  "incline bench press",
  "lat pulldown",
  "lateral raise",
  "leg extension",
  "leg raises",
  "push-up",
  "russian twist",
  "shoulder press",
  "squat",
  "t bar row",
  "tricep Pushdown",
  "tricep dips",
];

function flattenFrame(landmarks) {
  const xyz = [];
  const xy = [];
  for (const idx of JOINT_INDICES) {
    const lm = landmarks[idx] || {};
    const x = Number(lm.x || 0);
    const y = Number(lm.y || 0);
    const z = Number(lm.z || 0);
    xyz.push(x, y, z);
    xy.push(x, y);
  }
  return { xyz, xy };
}

function drawSkeleton(ctx, landmarks, width, height) {
  ctx.strokeStyle = "#6d28d9";
  ctx.lineWidth = 3;
  for (const [start, end] of BODY_CONNECTIONS) {
    if (!landmarks[start] || !landmarks[end]) continue;
    ctx.beginPath();
    ctx.moveTo(landmarks[start].x * width, landmarks[start].y * height);
    ctx.lineTo(landmarks[end].x * width, landmarks[end].y * height);
    ctx.stroke();
  }

  for (const idx of JOINT_INDICES) {
    const lm = landmarks[idx];
    if (!lm) continue;
    ctx.beginPath();
    ctx.arc(lm.x * width, lm.y * height, 5, 0, Math.PI * 2);
    ctx.fillStyle = "#ffffff";
    ctx.fill();
  }
}

function getJointName(index) {
  switch (index) {
    case 11: return "Left Shoulder";
    case 12: return "Right Shoulder";
    case 13: return "Left Elbow";
    case 14: return "Right Elbow";
    case 15: return "Left Wrist";
    case 16: return "Right Wrist";
    case 23: return "Left Hip";
    case 24: return "Right Hip";
    case 25: return "Left Knee";
    case 26: return "Right Knee";
    case 27: return "Left Ankle";
    case 28: return "Right Ankle";
    default: return "";
  }
}

function getArrowColor(distance) {
  if (distance < 0.05) return "#eab308";
  if (distance < 0.1) return "#f97316";
  return "#ef4444";
}

function drawArrow(ctx, fromX, fromY, toX, toY, color, lineWidth) {
  fromX = Math.round(fromX);
  fromY = Math.round(fromY);
  toX = Math.round(toX);
  toY = Math.round(toY);

  const headLength = 15;
  const dx = toX - fromX;
  const dy = toY - fromY;
  const magnitude = Math.sqrt(dx * dx + dy * dy);
  if (magnitude < ARROW_MIN_VECTOR_PX) return;
  const angle = Math.atan2(dy, dx);

  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(fromX, fromY);
  ctx.lineTo(toX, toY);
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(toX, toY);
  ctx.lineTo(
    Math.round(toX - headLength * Math.cos(angle - Math.PI / 6)),
    Math.round(toY - headLength * Math.sin(angle - Math.PI / 6))
  );
  ctx.lineTo(
    Math.round(toX - headLength * Math.cos(angle + Math.PI / 6)),
    Math.round(toY - headLength * Math.sin(angle + Math.PI / 6))
  );
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
  ctx.restore();
}

function smoothCorrections(newCorrections, prevCorrections) {
  if (!newCorrections || Object.keys(newCorrections).length === 0) return prevCorrections || {};
  if (!prevCorrections || Object.keys(prevCorrections).length === 0) return newCorrections;

  const smoothed = {};
  Object.keys(newCorrections).forEach((key) => {
    if (prevCorrections[key]) {
      smoothed[key] = {
        x: SMOOTHING_FACTOR * prevCorrections[key].x + (1 - SMOOTHING_FACTOR) * newCorrections[key].x,
        y: SMOOTHING_FACTOR * prevCorrections[key].y + (1 - SMOOTHING_FACTOR) * newCorrections[key].y,
      };
    } else {
      smoothed[key] = newCorrections[key];
    }
  });

  Object.keys(prevCorrections).forEach((key) => {
    if (!newCorrections[key]) {
      smoothed[key] = {
        x: prevCorrections[key].x * CORRECTION_FADE,
        y: prevCorrections[key].y * CORRECTION_FADE,
      };
      if (
        Math.abs(smoothed[key].x) < CORRECTION_CLEANUP_THRESHOLD &&
        Math.abs(smoothed[key].y) < CORRECTION_CLEANUP_THRESHOLD
      ) {
        delete smoothed[key];
      }
    }
  });
  return smoothed;
}

function findJointsToCorrect(landmarks, corrections) {
  if (!landmarks || !corrections || Object.keys(corrections).length === 0) return [];
  const correctJoints = [];
  Object.keys(corrections).forEach((indexStr) => {
    const i = parseInt(indexStr, 10);
    const correction = corrections[indexStr];
    if (!JOINT_INDICES.includes(i)) return;
    if (i >= 0 && i < landmarks.length && landmarks[i] && correction && typeof correction === "object") {
      const corrX = correction.x ?? 0;
      const corrY = correction.y ?? 0;
      if (Math.abs(corrX) > CORRECTION_THRESHOLD || Math.abs(corrY) > CORRECTION_THRESHOLD) {
        const jointName = getJointName(i);
        if (jointName) {
          correctJoints.push({ name: jointName, index: i, correction: { x: corrX, y: corrY } });
        }
      }
    }
  });
  return correctJoints;
}

function drawCorrectionArrows(ctx, landmarks, corrections, jointsToCorrect, width, height) {
  if (!landmarks || !corrections || jointsToCorrect.length === 0) return;
  ctx.globalCompositeOperation = "source-over";

  jointsToCorrect.forEach((joint) => {
    const i = joint.index;
    if (!landmarks[i] || landmarks[i].x == null || landmarks[i].y == null) return;

    const originalX = landmarks[i].x * width;
    const originalY = landmarks[i].y * height;
    const correction = corrections[String(i)];
    if (!correction || correction.x == null || correction.y == null) return;

    const vectorX = correction.x * width;
    const vectorY = correction.y * height;
    const magnitude = Math.sqrt(vectorX * vectorX + vectorY * vectorY);
    if (magnitude < DRAW_MIN_VECTOR_PX) return;

    const extendedTargetX = originalX + vectorX * ARROW_EXTENSION_FACTOR;
    const extendedTargetY = originalY + vectorY * ARROW_EXTENSION_FACTOR;
    const correctionMagnitude = Math.sqrt(correction.x * correction.x + correction.y * correction.y);
    const arrowColor = getArrowColor(correctionMagnitude);

    drawArrow(ctx, originalX, originalY, extendedTargetX, extendedTargetY, arrowColor, ARROW_LINE_WIDTH);
  });
  ctx.globalCompositeOperation = "source-over";
}

export default function WorkoutPage() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const cameraStageRef = useRef(null);
  const poseRef = useRef(null);
  const cameraRef = useRef(null);
  const classifierSocketRef = useRef(null);
  const correctionSocketRef = useRef(null);
  const latestLandmarksRef = useRef(null);
  const latestCorrectionsRef = useRef({});
  const xyzWindowRef = useRef([]);
  const xyWindowRef = useRef([]);
  const classifierXyzWindowRef = useRef([]);
  const classifierFrameToggleRef = useRef(true);
  const classifierBatchStartMsRef = useRef(null);
  const latestWorkoutRef = useRef("");
  const lastClassifierEmitRef = useRef(0);
  const lastCorrectionEmitRef = useRef(0);
  const classifierInFlightRef = useRef(false);
  const correctionInFlightRef = useRef(false);
  const classifierTimeoutRef = useRef(null);
  const correctionTimeoutRef = useRef(null);
  const previousCorrectionsRef = useRef({});
  const lastFrameTsRef = useRef(0);
  const fpsEmaRef = useRef(0);
  const lastFpsUiUpdateRef = useRef(0);
  const nextPoseSendAtMsRef = useRef(0);

  const [connectionStatus, setConnectionStatus] = useState("connecting");
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [predictedWorkout, setPredictedWorkout] = useState("Collecting 15 frames...");
  const [detectedWorkout, setDetectedWorkout] = useState("N/A");
  const [currentWorkout, setCurrentWorkout] = useState("N/A");
  const [predictionConfidence, setPredictionConfidence] = useState(0);
  const [manualWorkoutOverride, setManualWorkoutOverride] = useState("");
  const [bufferedFrames, setBufferedFrames] = useState(0);
  const [lastBufferFillMs, setLastBufferFillMs] = useState(null);
  const [feedFps, setFeedFps] = useState(0);
  const [workoutModel, setWorkoutModel] = useState("bilstm");
  const [poseModel, setPoseModel] = useState("tcn_film");
  const [showMuscles, setShowMuscles] = useState(true);

  const staticMuscleData = useMemo(() => [{ name: "inactive", muscles: [] }], []);

  useEffect(() => {
    const classifierSocket = io(`${AI_URL}/classifier`, {
      transports: ["polling", "websocket"],
      reconnection: true,
      reconnectionDelay: 1000,
    });
    const correctionSocket = io(`${AI_URL}/correction`, {
      transports: ["polling", "websocket"],
      reconnection: true,
      reconnectionDelay: 1000,
    });
    classifierSocketRef.current = classifierSocket;
    correctionSocketRef.current = correctionSocket;

    const updateConnectionStatus = () => {
      const cls = classifierSocketRef.current?.connected;
      const cor = correctionSocketRef.current?.connected;
      if (cls && cor) setConnectionStatus("connected");
      else if (cls || cor) setConnectionStatus("partial");
      else setConnectionStatus("disconnected");
    };

    classifierSocket.on("connect", updateConnectionStatus);
    correctionSocket.on("connect", updateConnectionStatus);
    classifierSocket.on("disconnect", updateConnectionStatus);
    correctionSocket.on("disconnect", updateConnectionStatus);

    classifierSocket.on("workout_prediction", (payload) => {
      classifierInFlightRef.current = false;
      if (classifierTimeoutRef.current) {
        clearTimeout(classifierTimeoutRef.current);
        classifierTimeoutRef.current = null;
      }
      if (payload?.predicted_workout_name) {
        const detected = payload.predicted_workout_name;
        const confidence = Number(payload?.confidence || 0);
        setDetectedWorkout(detected);
        setPredictionConfidence(confidence);
        setPredictedWorkout(detected);
        // Keep "current selected workout" sticky unless confidence is weak.
        if (!manualWorkoutOverride && (confidence >= 0.45 || latestWorkoutRef.current === "")) {
          latestWorkoutRef.current = detected;
          setCurrentWorkout(detected);
        }
      }
    });
    correctionSocket.on("pose_corrections", (payload) => {
      correctionInFlightRef.current = false;
      if (correctionTimeoutRef.current) {
        clearTimeout(correctionTimeoutRef.current);
        correctionTimeoutRef.current = null;
      }
      const smoothed = smoothCorrections(payload.corrections || {}, previousCorrectionsRef.current);
      previousCorrectionsRef.current = smoothed;
      latestCorrectionsRef.current = smoothed;
    });
    classifierSocket.on("error", (payload) => {
      classifierInFlightRef.current = false;
      setConnectionStatus(`error: ${payload?.message || "unknown"}`);
    });
    correctionSocket.on("error", (payload) => {
      correctionInFlightRef.current = false;
      setConnectionStatus(`error: ${payload?.message || "unknown"}`);
    });

    return () => {
      if (classifierTimeoutRef.current) clearTimeout(classifierTimeoutRef.current);
      if (correctionTimeoutRef.current) clearTimeout(correctionTimeoutRef.current);
      classifierSocket.disconnect();
      correctionSocket.disconnect();
      classifierSocketRef.current = null;
      correctionSocketRef.current = null;
    };
  }, [manualWorkoutOverride]);

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(document.fullscreenElement === cameraStageRef.current);
    };
    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, []);

  useEffect(() => {
    let active = true;
    let initTimeout;

    const onResults = (results) => {
      if (!active || !canvasRef.current || !webcamRef.current) return;

      const now = Date.now();
      if (lastFrameTsRef.current > 0) {
        const deltaMs = now - lastFrameTsRef.current;
        if (deltaMs > 0) {
          const instantFps = 1000 / deltaMs;
          fpsEmaRef.current = fpsEmaRef.current > 0 ? fpsEmaRef.current * 0.8 + instantFps * 0.2 : instantFps;
          if (now - lastFpsUiUpdateRef.current >= 250) {
            setFeedFps(fpsEmaRef.current);
            lastFpsUiUpdateRef.current = now;
          }
        }
      }
      lastFrameTsRef.current = now;

      const width = webcamRef.current.videoWidth || 1280;
      const height = webcamRef.current.videoHeight || 720;
      const canvas = canvasRef.current;
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, width, height);

      if (results.poseLandmarks) {
        latestLandmarksRef.current = results.poseLandmarks;
        const { xyz, xy } = flattenFrame(results.poseLandmarks);
        xyzWindowRef.current = [...xyzWindowRef.current.slice(-(WINDOW_SIZE - 1)), xyz];
        xyWindowRef.current = [...xyWindowRef.current.slice(-(WINDOW_SIZE - 1)), xy];

        // Sample workout-classifier input at ~15fps by buffering every other frame.
        // This is a discrete batch buffer: no overlap between consecutive batches.
        if (classifierFrameToggleRef.current) {
          if (classifierXyzWindowRef.current.length < WINDOW_SIZE) {
            if (classifierXyzWindowRef.current.length === 0) {
              classifierBatchStartMsRef.current = now;
            }
            classifierXyzWindowRef.current = [...classifierXyzWindowRef.current, xyz];
          }
        }
        classifierFrameToggleRef.current = !classifierFrameToggleRef.current;
        setBufferedFrames(classifierXyzWindowRef.current.length);

        if (
          classifierXyzWindowRef.current.length === WINDOW_SIZE &&
          classifierSocketRef.current?.connected &&
          !classifierInFlightRef.current &&
          now - lastClassifierEmitRef.current >= 80
        ) {
          const sequenceForClassification = [...classifierXyzWindowRef.current];
          lastClassifierEmitRef.current = now;
          classifierInFlightRef.current = true;
          classifierSocketRef.current.emit("classify_sequence", {
            timestamp: now,
            selected_workout_model: workoutModel,
            sequence_xyz: sequenceForClassification,
          });
          if (classifierBatchStartMsRef.current !== null) {
            setLastBufferFillMs(now - classifierBatchStartMsRef.current);
          }
          classifierXyzWindowRef.current = [];
          classifierBatchStartMsRef.current = null;
          setBufferedFrames(0);
          classifierTimeoutRef.current = setTimeout(() => {
            classifierInFlightRef.current = false;
            classifierTimeoutRef.current = null;
          }, 1500);
        }

        if (
          xyWindowRef.current.length >= WINDOW_SIZE &&
          correctionSocketRef.current?.connected &&
          latestWorkoutRef.current &&
          !correctionInFlightRef.current &&
          now - lastCorrectionEmitRef.current >= 80
        ) {
          lastCorrectionEmitRef.current = now;
          correctionInFlightRef.current = true;
          correctionSocketRef.current.emit("correct_sequence", {
            timestamp: now,
            selected_pose_model: poseModel,
            workout_name: manualWorkoutOverride || latestWorkoutRef.current,
            sequence_xy: xyWindowRef.current,
            sequence_xyz: xyzWindowRef.current,
          });
          correctionTimeoutRef.current = setTimeout(() => {
            correctionInFlightRef.current = false;
            correctionTimeoutRef.current = null;
          }, 1500);
        }

        drawSkeleton(ctx, results.poseLandmarks, width, height);
        const currentCorrections = latestCorrectionsRef.current;
        if (currentCorrections && Object.keys(currentCorrections).length > 0) {
          const jointsToCorrect = findJointsToCorrect(results.poseLandmarks, currentCorrections);
          if (jointsToCorrect.length > 0) {
            drawCorrectionArrows(
              ctx,
              results.poseLandmarks,
              currentCorrections,
              jointsToCorrect,
              width,
              height
            );
          }
        }
      } else {
        latestLandmarksRef.current = null;
        xyzWindowRef.current = [];
        xyWindowRef.current = [];
        classifierXyzWindowRef.current = [];
        classifierFrameToggleRef.current = true;
        classifierBatchStartMsRef.current = null;
        lastFrameTsRef.current = 0;
        fpsEmaRef.current = 0;
        lastFpsUiUpdateRef.current = 0;
        nextPoseSendAtMsRef.current = 0;
        previousCorrectionsRef.current = {};
        latestCorrectionsRef.current = {};
        setBufferedFrames(0);
        setFeedFps(0);
      }
    };

    const start = async () => {
      const pose = new poseDetection.Pose({
        locateFile: (file) =>
          `https://cdn.jsdelivr.net/npm/@mediapipe/pose@${poseDetection.VERSION}/${file}`,
      });
      poseRef.current = pose;
      pose.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        enableSegmentation: false,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      pose.onResults(onResults);
      await pose.initialize();

      const camera = new Camera(webcamRef.current, {
        onFrame: async () => {
          if (poseRef.current && webcamRef.current) {
            const now = performance.now();
            if (!nextPoseSendAtMsRef.current) {
              nextPoseSendAtMsRef.current = now;
            }
            if (now + 0.5 < nextPoseSendAtMsRef.current) return;
            nextPoseSendAtMsRef.current += TARGET_FEED_FRAME_MS;
            if (now - nextPoseSendAtMsRef.current > TARGET_FEED_FRAME_MS * 2) {
              nextPoseSendAtMsRef.current = now + TARGET_FEED_FRAME_MS;
            }
            await poseRef.current.send({ image: webcamRef.current });
          }
        },
        width: 1280,
        height: 720,
        frameRate: TARGET_FEED_FPS,
      });
      cameraRef.current = camera;
      await camera.start();
    };

    initTimeout = setTimeout(() => {
      start().catch((err) => {
        setConnectionStatus(`camera error: ${err.message}`);
      });
    }, 80);

    return () => {
      active = false;
      clearTimeout(initTimeout);
      if (cameraRef.current) cameraRef.current = null;
      if (poseRef.current) {
        poseRef.current.close();
        poseRef.current = null;
      }
    };
  }, []);

  const toggleFullscreen = async () => {
    const el = cameraStageRef.current;
    if (!el) return;
    try {
      if (document.fullscreenElement === el) {
        await document.exitFullscreen();
      } else if (el.requestFullscreen) {
        await el.requestFullscreen();
      }
    } catch (err) {
      setConnectionStatus(`fullscreen error`);
    }
  };

  return (
    <div className="page-shell">
      <div className="top-bar">
        <div className="brand">TrueForm Testing Replica</div>
        <div className={`status-pill status-${connectionStatus.startsWith("connected") ? "ok" : "warn"}`}>
          {connectionStatus}
        </div>
      </div>

      <div className="content-grid">
        <motion.aside className="left-panel" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <h3>Realtime Models</h3>
          <label>
            Workout Classifier
            <select value={workoutModel} onChange={(e) => setWorkoutModel(e.target.value)}>
              {WORKOUT_MODEL_OPTIONS.map((opt) => (
                <option key={opt} value={opt}>
                  {opt}
                </option>
              ))}
            </select>
          </label>
          <label>
            Pose Correction
            <select value={poseModel} onChange={(e) => setPoseModel(e.target.value)}>
              {POSE_MODEL_OPTIONS.map((opt) => (
                <option key={opt} value={opt}>
                  {opt}
                </option>
              ))}
            </select>
          </label>
          <button className="toggle-button" onClick={() => setShowMuscles((v) => !v)}>
            {showMuscles ? "Hide" : "Show"} Muscle Overlays
          </button>
          <div className="prediction">
            <span>Detected workout</span>
            <strong>{predictedWorkout}</strong>
            <span>Sequence buffer: {bufferedFrames}/15</span>
          </div>
        </motion.aside>

        <div className="camera-stage-stack">
          <section ref={cameraStageRef} className="camera-stage">
            <video ref={webcamRef} className="camera-video" autoPlay playsInline muted />
            <canvas ref={canvasRef} className="overlay-canvas" />

            <button
              onClick={toggleFullscreen}
              className="fullscreen-control"
              title={isFullscreen ? "Exit Fullscreen" : "Enter Fullscreen"}
            >
              {isFullscreen ? "🗗" : "⛶"}
            </button>

            <div className="override-control">
              <label htmlFor="workout-override">Workout override</label>
              <select
                id="workout-override"
                value={manualWorkoutOverride}
                onChange={(e) => {
                  const value = e.target.value;
                  setManualWorkoutOverride(value);
                  if (value) {
                    latestWorkoutRef.current = value;
                    setCurrentWorkout(value);
                  } else if (detectedWorkout && detectedWorkout !== "N/A") {
                    latestWorkoutRef.current = detectedWorkout;
                    setCurrentWorkout(detectedWorkout);
                  }
                }}
              >
                <option value="">Auto (use detected)</option>
                {WORKOUT_LABEL_OPTIONS.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </div>

            <div className="feed-info-panel">
              <div className="feed-info-row feed-info-top-row">
                <div className="feed-info-inline">
                  <span className="feed-info-label">Current:</span>
                  <span className="feed-info-value">{currentWorkout}</span>
                </div>
                <div className="feed-info-inline">
                  <span className="feed-info-label">Buffer:</span>
                  <span className="feed-info-value">{bufferedFrames}/15</span>
                </div>
              </div>
              <div className="feed-info-row">
                <span className="feed-info-label">Detected:</span>
                <span className={`feed-pill ${detectedWorkout !== currentWorkout ? "feed-pill-active" : ""}`}>
                  {detectedWorkout}
                </span>
                <div className="feed-confidence-wrap">
                  <div className="feed-confidence-track">
                    <div
                      className={`feed-confidence-fill ${predictionConfidence >= 0.6 ? "high" : ""}`}
                      style={{ width: `${Math.max(0, Math.min(100, predictionConfidence * 100))}%` }}
                    />
                  </div>
                  <span className={`feed-confidence-value ${predictionConfidence >= 0.6 ? "high" : ""}`}>
                    {Math.round(predictionConfidence * 100)}%
                  </span>
                </div>
              </div>
              <div className="feed-info-row feed-info-small">
                <span>
                  Models: <strong>{workoutModel}</strong> + <strong>{poseModel}</strong>
                </span>
                {manualWorkoutOverride ? (
                  <span className="feed-override-badge">override on</span>
                ) : (
                  <span className="feed-override-badge auto">auto</span>
                )}
                <span className={`feed-conn ${connectionStatus.startsWith("connected") ? "ok" : "warn"}`}>
                  {connectionStatus}
                </span>
              </div>
            </div>

            <AnimatePresence>
              {showMuscles && (
                <>
                  <motion.div
                    className="muscle-overlay muscle-left"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 0.8, x: 0 }}
                    exit={{ opacity: 0 }}
                  >
                    <Model
                      data={staticMuscleData}
                      type={ModelType.ANTERIOR}
                      highlightedColors={["#ffffff"]}
                      onClick={() => {}}
                    />
                  </motion.div>
                  <motion.div
                    className="muscle-overlay muscle-right"
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 0.8, x: 0 }}
                    exit={{ opacity: 0 }}
                  >
                    <Model
                      data={staticMuscleData}
                      type={ModelType.POSTERIOR}
                      highlightedColors={["#ffffff"]}
                      onClick={() => {}}
                    />
                  </motion.div>
                </>
              )}
            </AnimatePresence>
          </section>

          <div className="buffer-strip" aria-label="Workout classifier frame buffer">
            {Array.from({ length: WINDOW_SIZE }).map((_, idx) => (
              <span
                key={idx}
                className={`buffer-slot ${idx < bufferedFrames ? "filled" : ""}`}
                aria-hidden="true"
              />
            ))}
          </div>
          <div className="buffer-timer">
            Last fill time: {lastBufferFillMs === null ? "--" : `${(lastBufferFillMs / 1000).toFixed(2)}s`} | Feed FPS:{" "}
            {feedFps > 0 ? feedFps.toFixed(1) : "--"}
          </div>
        </div>
      </div>
    </div>
  );
}
