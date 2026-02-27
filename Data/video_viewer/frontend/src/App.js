import React, { useState, useEffect } from 'react';
import './App.css';
import VideoPlayer from './components/VideoPlayer';
import VideoList from './components/VideoList';

const API_BASE_URL = 'http://localhost:5003/api';

function App() {
  const [videos, setVideos] = useState([]);
  const [currentVideoIndex, setCurrentVideoIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchVideos();
  }, []);

  const fetchVideos = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch(`${API_BASE_URL}/videos`);
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `Failed to fetch videos: ${response.status} ${response.statusText}`);
      }
      const data = await response.json();
      setVideos(data.all_videos || []);
      setError(null);
    } catch (err) {
      const errorMessage = err.message || 'Failed to connect to backend server. Make sure the backend is running on port 5003.';
      setError(errorMessage);
      console.error('Error fetching videos:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleNextVideo = () => {
    if (currentVideoIndex < videos.length - 1) {
      setCurrentVideoIndex(currentVideoIndex + 1);
    }
  };

  const handlePreviousVideo = () => {
    if (currentVideoIndex > 0) {
      setCurrentVideoIndex(currentVideoIndex - 1);
    }
  };

  const handleVideoSelect = (index) => {
    setCurrentVideoIndex(index);
  };

  const handleToggleIgnore = async (fullPath) => {
    try {
      const response = await fetch(`${API_BASE_URL}/video/toggle-ignore`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ full_path: fullPath }),
      });

      if (!response.ok) {
        throw new Error('Failed to toggle ignore status');
      }

      const data = await response.json();
      
      // Update the video's ignored status in the local state
      setVideos(prevVideos =>
        prevVideos.map(video =>
          video.full_path === fullPath
            ? { ...video, ignored: data.ignored }
            : video
        )
      );
    } catch (err) {
      console.error('Error toggling ignore status:', err);
      alert('Failed to update ignore status. Please try again.');
    }
  };

  if (loading) {
    return (
      <div className="app">
        <div className="loading">Loading videos...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="app">
        <div className="error">
          <h2>Error loading videos</h2>
          <p>{error}</p>
          <button onClick={fetchVideos}>Retry</button>
        </div>
      </div>
    );
  }

  if (videos.length === 0) {
    return (
      <div className="app">
        <div className="error">No videos found in the Videos directory.</div>
      </div>
    );
  }

  const currentVideo = videos[currentVideoIndex];

  return (
    <div className="app">
      <header className="app-header">
        <h1>Video Pose Viewer</h1>
        <div className="video-counter">
          Video {currentVideoIndex + 1} of {videos.length}
        </div>
      </header>
      
      <div className="app-content">
        <div className="video-section">
          <VideoPlayer
            videoPath={currentVideo?.path}
            videoName={currentVideo?.name}
            workoutType={currentVideo?.workout_type}
            onNext={handleNextVideo}
            onPrevious={handlePreviousVideo}
            hasNext={currentVideoIndex < videos.length - 1}
            hasPrevious={currentVideoIndex > 0}
          />
        </div>
        
        <div className="sidebar">
          <VideoList
            videos={videos}
            currentIndex={currentVideoIndex}
            onVideoSelect={handleVideoSelect}
            onToggleIgnore={handleToggleIgnore}
          />
        </div>
      </div>
    </div>
  );
}

export default App;
