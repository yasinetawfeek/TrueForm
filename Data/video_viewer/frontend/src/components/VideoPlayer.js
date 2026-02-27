import React, { useRef, useEffect, useState } from 'react';
import { Pose } from '@mediapipe/pose';
import { drawConnectors, drawLandmarks } from '@mediapipe/drawing_utils';
import { POSE_CONNECTIONS } from '@mediapipe/pose';
import './VideoPlayer.css';

const VideoPlayer = ({ videoPath, videoName, workoutType, onNext, onPrevious, hasNext, hasPrevious }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const poseRef = useRef(null);
  const animationFrameRef = useRef(null);
  const renderFrameRef = useRef(null);
  const lastPoseResultsRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    if (!videoPath) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    setIsLoading(true);
    setError(null);
    lastPoseResultsRef.current = null;

    let pose = null;
    let isInitialized = false;
    let isProcessing = false;

    // Initialize MediaPipe Pose
    const initPose = async () => {
      try {
        console.log('Initializing MediaPipe Pose...');
        
        pose = new Pose({
          locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
          }
        });

        pose.setOptions({
          modelComplexity: 1,
          smoothLandmarks: true,
          enableSegmentation: false,
          smoothSegmentation: false,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5
        });

        // Set up results callback
        pose.onResults((results) => {
          lastPoseResultsRef.current = results;
          if (results.poseLandmarks) {
            console.log(`Pose detected: ${results.poseLandmarks.length} landmarks`);
          }
          // Trigger a redraw
          drawFrame();
        });

        poseRef.current = pose;
        isInitialized = true;
        console.log('MediaPipe Pose initialized successfully');
      } catch (err) {
        console.error('Error initializing MediaPipe:', err);
        setError('Failed to initialize pose detection');
      }
    };

    // Draw function that renders video + skeleton
    const drawFrame = () => {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      if (!canvas || !video) return;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Only draw if video has data
      if (video.readyState >= video.HAVE_CURRENT_DATA && canvas.width > 0 && canvas.height > 0) {
        ctx.save();
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw video frame
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Draw pose skeleton if available
        // MediaPipe drawing utilities automatically handle coordinate conversion
        if (lastPoseResultsRef.current && lastPoseResultsRef.current.poseLandmarks) {
          try {
            // Draw connections (green lines)
            drawConnectors(ctx, lastPoseResultsRef.current.poseLandmarks, POSE_CONNECTIONS, {
              color: '#00FF00',
              lineWidth: 3
            });
            
            // Draw landmarks (red dots)
            drawLandmarks(ctx, lastPoseResultsRef.current.poseLandmarks, {
              color: '#FF0000',
              lineWidth: 2,
              radius: 4
            });
          } catch (err) {
            console.error('Error drawing pose:', err);
          }
        }

        ctx.restore();
      }
    };

    // Continuous render loop
    const renderLoop = () => {
      drawFrame();
      renderFrameRef.current = requestAnimationFrame(renderLoop);
    };

    // Process video frames with MediaPipe
    const processFrame = () => {
      if (!video || !pose || !isInitialized || video.paused || video.ended) {
        if (!video.paused && !video.ended) {
          animationFrameRef.current = requestAnimationFrame(processFrame);
        }
        return;
      }

      if (video.readyState >= video.HAVE_CURRENT_DATA) {
        try {
          pose.send({ image: video });
        } catch (err) {
          console.error('Error sending frame to MediaPipe:', err);
        }
      }
      
      animationFrameRef.current = requestAnimationFrame(processFrame);
    };

    // Set up video element
    video.src = `http://localhost:5003${videoPath}`;
    video.autoplay = true;
    video.muted = true;
    video.playsInline = true;
    
    const handleLoadedMetadata = () => {
      if (canvas && video) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        setDuration(video.duration);
        console.log(`Video loaded: ${video.videoWidth}x${video.videoHeight}`);
        
        // Start render loop
        renderLoop();
        setIsLoading(false);
      }
    };

    const handleCanPlay = () => {
      if (video.paused) {
        video.play().catch(err => {
          console.log('Autoplay prevented:', err);
        });
      }
    };

    const handlePlay = () => {
      setIsPlaying(true);
      if (isInitialized && !isProcessing) {
        isProcessing = true;
        processFrame();
      }
    };

    const handlePause = () => {
      setIsPlaying(false);
      isProcessing = false;
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };

    const handleTimeUpdate = () => {
      setCurrentTime(video.currentTime);
    };

    const handleError = (e) => {
      const errorMsg = video.error ? `Video error: ${video.error.code} - ${video.error.message}` : 'Failed to load video';
      setError(errorMsg);
      setIsLoading(false);
      console.error('Video error:', e, video.error);
    };

    // Initialize MediaPipe first
    initPose().then(() => {
      // Set up event listeners
      video.addEventListener('loadedmetadata', handleLoadedMetadata);
      video.addEventListener('canplay', handleCanPlay);
      video.addEventListener('play', handlePlay);
      video.addEventListener('pause', handlePause);
      video.addEventListener('timeupdate', handleTimeUpdate);
      video.addEventListener('error', handleError);
      video.addEventListener('ended', handlePause);
    });

    // Cleanup
    return () => {
      video.removeEventListener('loadedmetadata', handleLoadedMetadata);
      video.removeEventListener('canplay', handleCanPlay);
      video.removeEventListener('play', handlePlay);
      video.removeEventListener('pause', handlePause);
      video.removeEventListener('timeupdate', handleTimeUpdate);
      video.removeEventListener('error', handleError);
      video.removeEventListener('ended', handlePause);
      
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (renderFrameRef.current) {
        cancelAnimationFrame(renderFrameRef.current);
      }
      if (poseRef.current) {
        try {
          poseRef.current.close();
        } catch (err) {
          console.error('Error closing MediaPipe:', err);
        }
      }
      
      video.pause();
      video.src = '';
      lastPoseResultsRef.current = null;
      isProcessing = false;
    };
  }, [videoPath]);

  const handlePlayPause = () => {
    const video = videoRef.current;
    if (video) {
      if (video.paused) {
        video.play();
      } else {
        video.pause();
      }
    }
  };

  const handleSeek = (e) => {
    const video = videoRef.current;
    if (video) {
      const rect = e.currentTarget.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const percent = x / rect.width;
      video.currentTime = percent * video.duration;
    }
  };

  if (error) {
    return (
      <div className="video-player error">
        <p>{error}</p>
      </div>
    );
  }

  return (
    <div className="video-player">
      <div className="video-info">
        <h2>{videoName || 'Loading...'}</h2>
        {workoutType && <span className="workout-type">{workoutType}</span>}
      </div>

      <div className="video-container">
        <video
          ref={videoRef}
          className="video-element"
          playsInline
          muted
          autoPlay
        />
        <canvas
          ref={canvasRef}
          className="canvas-overlay"
        />
        {isLoading && (
          <div className="loading-overlay">
            <div className="spinner"></div>
            <p>Loading video...</p>
          </div>
        )}
      </div>

      <div className="video-controls">
        <button
          onClick={handlePlayPause}
          className="control-button play-pause"
          disabled={isLoading}
        >
          {isPlaying ? '⏸' : '▶'}
        </button>
        
        <div className="seek-bar-container" onClick={handleSeek}>
          <div className="seek-bar">
            <div
              className="seek-bar-fill"
              style={{
                width: duration > 0 ? `${(currentTime / duration) * 100}%` : '0%'
              }}
            />
          </div>
        </div>

        <div className="navigation-buttons">
          <button
            onClick={onPrevious}
            className="control-button nav-button"
            disabled={!hasPrevious || isLoading}
          >
            ← Previous
          </button>
          <button
            onClick={onNext}
            className="control-button nav-button"
            disabled={!hasNext || isLoading}
          >
            Next →
          </button>
        </div>
      </div>
    </div>
  );
};

export default VideoPlayer;
