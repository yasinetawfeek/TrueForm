import React from 'react';
import './VideoList.css';

const VideoList = ({ videos, currentIndex, onVideoSelect, onToggleIgnore }) => {
  // Group videos by workout type
  const groupedVideos = videos.reduce((acc, video, index) => {
    const workoutType = video.workout_type || 'Other';
    if (!acc[workoutType]) {
      acc[workoutType] = [];
    }
    acc[workoutType].push({ ...video, index });
    return acc;
  }, {});

  return (
    <div className="video-list">
      <h3>Videos ({videos.length})</h3>
      <div className="video-list-content">
        {Object.entries(groupedVideos).map(([workoutType, workoutVideos]) => (
          <div key={workoutType} className="workout-group">
            <div className="workout-header">
              <h4>{workoutType}</h4>
              <span className="video-count">{workoutVideos.length}</span>
            </div>
            <div className="workout-videos">
              {workoutVideos.map((video) => (
                <div
                  key={video.index}
                  className={`video-item ${video.index === currentIndex ? 'active' : ''} ${video.ignored ? 'ignored' : ''}`}
                >
                  <div 
                    className="video-item-content"
                    onClick={() => onVideoSelect(video.index)}
                  >
                    <div className="video-item-name">{video.name}</div>
                    {video.index === currentIndex && (
                      <div className="active-indicator">▶</div>
                    )}
                  </div>
                  <button
                    className={`ignore-toggle ${video.ignored ? 'ignored' : ''}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      if (onToggleIgnore && video.full_path) {
                        onToggleIgnore(video.full_path);
                      }
                    }}
                    title={video.ignored ? 'Include in processing (click to unblock)' : 'Ignore in processing (click to block)'}
                  >
                    {video.ignored ? '✕' : '○'}
                  </button>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default VideoList;
