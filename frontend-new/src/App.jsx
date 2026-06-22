import { useState, useRef, useEffect } from 'react';
import './App.css';

function App() {
  const [videoFile, setVideoFile] = useState(null);
  const [videoPreview, setVideoPreview] = useState('');
  const [dragActive, setDragActive] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const fileInputRef = useRef(null);

  // Clean up object URL to prevent memory leaks
  useEffect(() => {
    return () => {
      if (videoPreview) {
        URL.revokeObjectURL(videoPreview);
      }
    };
  }, [videoPreview]);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      processFile(file);
    }
  };

  const processFile = (file) => {
    if (file.type !== 'video/mp4') {
      alert('Please upload an MP4 video file.');
      return;
    }
    
    // Revoke previous URL if any
    if (videoPreview) {
      URL.revokeObjectURL(videoPreview);
    }

    setVideoFile(file);
    setVideoPreview(URL.createObjectURL(file));
    setResult(null); // Clear previous results
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const file = e.dataTransfer.files?.[0];
    if (file) {
      processFile(file);
    }
  };

  const handleRemove = () => {
    if (videoPreview) {
      URL.revokeObjectURL(videoPreview);
    }
    setVideoFile(null);
    setVideoPreview('');
    setResult(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handlePredict = () => {
    if (!videoFile) return;
    setIsLoading(true);
    setResult(null);

    setTimeout(() => {
      setResult({
        word: "PLACE",
        confidence: 0.82
      });
      setIsLoading(false);
    }, 1000);
  };

  const formatBytes = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="app-container">
      <header className="brand-header">
        <h1 className="brand-title">LipSense</h1>
        <p className="brand-subtitle">Deep learning lip-reading prediction interface</p>
      </header>

      <main className="glass-card">
        {/* File Upload Zone */}
        {!videoFile ? (
          <div 
            className={`upload-zone ${dragActive ? 'drag-active' : ''}`}
            onDragEnter={handleDrag}
            onDragOver={handleDrag}
            onDragLeave={handleDrag}
            onDrop={handleDrop}
          >
            <svg className="upload-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 16.5V9.75m0 0l3 3m-3-3l-3 3M6.75 19.5a4.5 4.5 0 01-1.41-8.775 5.25 5.25 0 0110.233-2.33 3 3 0 013.758 3.848A3.752 3.752 0 0118 19.5H6.75z" />
            </svg>
            <div className="upload-text">
              <span className="upload-title">Drag & drop your video here</span>
              <span className="upload-desc">or click to browse your files (MP4 format)</span>
            </div>
            <input 
              ref={fileInputRef}
              type="file" 
              className="file-input" 
              accept="video/mp4" 
              onChange={handleFileChange}
            />
          </div>
        ) : (
          /* Video Preview Section */
          <div className="video-preview-container">
            <div className="video-header">
              <div className="video-info">
                {/* Movie/Film icon */}
                <svg className="success-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 20.25h12A2.25 2.25 0 0 0 20.25 18V6A2.25 2.25 0 0 0 18 3.75H6A2.25 2.25 0 0 0 3.75 6v12A2.25 2.25 0 0 0 6 20.25Z" />
                  <path strokeLinecap="round" strokeLinejoin="round" d="m9 14.25 6-3.75-6-3.75v7.5Z" />
                </svg>
                <span className="video-name" title={videoFile.name}>{videoFile.name}</span>
                <span className="video-size">({formatBytes(videoFile.size)})</span>
              </div>
              <button className="btn-remove" onClick={handleRemove} title="Remove video">
                {/* Trash Icon */}
                <svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0" />
                </svg>
              </button>
            </div>
            
            <div className="video-element-wrapper">
              <video className="preview-video" src={videoPreview} controls />
            </div>
          </div>
        )}

        {/* Predict Action Button */}
        <button 
          className="btn-predict" 
          onClick={handlePredict} 
          disabled={!videoFile || isLoading}
        >
          {isLoading ? (
            <>
              <div className="spinner"></div>
              Analyzing Video...
            </>
          ) : (
            'Run LipSense Prediction'
          )}
        </button>

        {/* Loading Indicator */}
        {isLoading && (
          <div className="loader-container">
            <span className="loading-text">Processing frames...</span>
          </div>
        )}

        {/* Results Area */}
        {result && !isLoading && (
          <div className="result-card">
            <div className="result-header">
              <svg className="success-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Prediction Complete
            </div>
            <div className="result-body">
              <div className="prediction-item">
                <span className="label">Predicted Sentence/Word:</span>
                <span className="prediction-value">{result.word}</span>
              </div>
              
              <div className="confidence-section">
                <div className="confidence-header">
                  <span className="label">Model Confidence:</span>
                  <span className="confidence-val">{(result.confidence * 100).toFixed(0)}%</span>
                </div>
                <div className="confidence-bar-bg">
                  <div 
                    className="confidence-bar-fill" 
                    style={{ width: `${result.confidence * 100}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>


    </div>
  );
}

export default App;
