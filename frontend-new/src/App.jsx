import { useState, useRef, useEffect } from 'react';
import './App.css';
import LipPreview from './LipPreview';

function App() {
  const [videoFile, setVideoFile] = useState(null);
  const [videoPreview, setVideoPreview] = useState('');
  const [dragActive, setDragActive] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [validationError, setValidationError] = useState('');
  const fileInputRef = useRef(null);
  const videoRef = useRef(null);

  // Custom cursor states
  const [cursorPos, setCursorPos] = useState({ x: 0, y: 0 });
  const [showCustomCursor, setShowCustomCursor] = useState(false);
  const [isHovered, setIsHovered] = useState(false);

  // Clean up object URL to prevent memory leaks
  useEffect(() => {
    return () => {
      if (videoPreview) {
        URL.revokeObjectURL(videoPreview);
      }
    };
  }, [videoPreview]);

  // Set up custom cursor listeners for desktop only (pointer: fine) and no reduced motion
  useEffect(() => {
    const isDesktop = window.matchMedia('(pointer: fine)').matches && window.innerWidth > 768;
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    if (isDesktop && !prefersReducedMotion) {
      setShowCustomCursor(true);

      const handleMouseMove = (e) => {
        setCursorPos({ x: e.clientX, y: e.clientY });
        const target = e.target;
        if (target && target.closest('button, a, input, [role="button"], .upload-zone, .btn-remove')) {
          setIsHovered(true);
        } else {
          setIsHovered(false);
        }
      };

      window.addEventListener('mousemove', handleMouseMove);
      return () => window.removeEventListener('mousemove', handleMouseMove);
    }
  }, []);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      processFile(file);
    }
  };

  const processFile = (file) => {
    const validExtensions = ['mp4', 'mov', 'webm'];
    const fileExtension = file.name.split('.').pop().toLowerCase();

    // Check both extension and mime type (with fallback for raw formats)
    const isValidExtension = validExtensions.includes(fileExtension);
    const isValidMime = file.type && (
      file.type.includes('mp4') ||
      file.type.includes('quicktime') ||
      file.type.includes('webm')
    );

    if (!isValidExtension && !isValidMime) {
      setValidationError('Invalid format. Please upload a video clip.');
      setVideoFile(null);
      setVideoPreview('');
      setResult(null);
      return;
    }

    setValidationError('');
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
    setValidationError('');
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
    <div className={`app-container ${showCustomCursor ? 'custom-cursor-active' : ''}`}>
      {/* Custom Cursor */}
      {showCustomCursor && (
        <>
          <div
            className={`custom-cursor-dot ${isHovered ? 'hovered' : ''}`}
            style={{ left: `${cursorPos.x}px`, top: `${cursorPos.y}px` }}
          />
          <div
            className={`custom-cursor-glow ${isHovered ? 'hovered' : ''}`}
            style={{ left: `${cursorPos.x}px`, top: `${cursorPos.y}px` }}
          />
        </>
      )}

      {/* Ambient background blob/wave */}
      <div className="ambient-bg-pulse" />

      <header className="brand-header">
        <h1 className="brand-title">LipSense</h1>
        <p className="brand-subtitle">Visual speech recognition & predictive interface</p>
      </header>

      <div className="content-layout">
        {/* Short description sidebar */}
        <aside className="description-sidebar">
          <div className="sidebar-badge">Model Specs</div>
          <h2 className="sidebar-title">AI-Powered Lip Reading</h2>
          <p className="sidebar-desc">
            LipSense translates spoken words directly from visual lip movements.
            By analyzing spatio-temporal features across silent video frames, our neural network models speech patterns with high accuracy.
          </p>
          <p className="sidebar-desc-secondary">
            Provide a short video clip of a person speaking directly to the camera to generate predictions.
          </p>

          <div className="sidebar-divider" />

          <div className="sidebar-specs">
            <div className="spec-item">
              <span className="spec-label">Target Phrase</span>
              <span className="spec-val">Word-Level</span>
            </div>
          </div>
        </aside>

        {/* Main interactive glass card */}
        <main className="glass-card">
          {/* Inline Validation Error */}
          {validationError && (
            <div className="validation-error-box">
              <svg className="error-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
              <span>{validationError}</span>
            </div>
          )}

          {/* File Upload Zone */}
          {!videoFile ? (
            <div
              className={`upload-zone ${dragActive ? 'drag-active' : ''} ${validationError ? 'has-error' : ''}`}
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
                <span className="upload-desc">or click to browse your files</span>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                className="file-input"
                accept="video/mp4,video/quicktime,video/webm"
                onChange={handleFileChange}
              />
            </div>
          ) : (
            /* Video Preview Section */
            <div className="video-preview-container">
              <div className="video-header">
                <div className="video-info">
                  <svg className="video-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M6 20.25h12A2.25 2.25 0 0 0 20.25 18V6A2.25 2.25 0 0 0 18 3.75H6A2.25 2.25 0 0 0 3.75 6v12A2.25 2.25 0 0 0 6 20.25Z" />
                    <path strokeLinecap="round" strokeLinejoin="round" d="m9 14.25 6-3.75-6-3.75v7.5Z" />
                  </svg>
                  <span className="video-name" title={videoFile.name}>{videoFile.name}</span>
                  <span className="video-size">({formatBytes(videoFile.size)})</span>
                </div>
                <button className="btn-remove" onClick={handleRemove} title="Remove video">
                  <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              <div className="video-with-lip">
                <div className="video-element-wrapper animate-preview">
                  <video ref={videoRef} className="preview-video" src={videoPreview} controls />
                </div>

                {/* Lip region preview — runs independently, does not affect Predict */}
                <LipPreview videoRef={videoRef} videoSrc={videoPreview} />
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
                Analyzing Frames...
              </>
            ) : (
              'Run LipSense Prediction'
            )}
          </button>

          {/* Loading Indicator */}
          {isLoading && (
            <div className="loader-container">
              <span className="loading-text">Extracting spatio-temporal features...</span>
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
                  <span className="label">Predicted Word:</span>
                  <span className="prediction-value">{result.word}</span>
                </div>

                <div className="confidence-section">
                  <div className="confidence-header">
                    <span className="label">Confidence Score:</span>
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
    </div>
  );
}

export default App;
