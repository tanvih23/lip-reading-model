import { useEffect, useRef, useState, useCallback } from 'react';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

// ─── Mouth landmark indices (MediaPipe 478-point face mesh) ─────────────────
// Outer lip contour — 20 points forming a tight ring around the mouth.
const MOUTH_OUTER = [
  61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
  291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
];

// Padding ratio around the crop (relative to mouth bounding box)
const PAD_X = 0.55;
const PAD_Y = 0.75;

// Detection throttle: ~10 FPS
const DETECT_INTERVAL_MS = 100;

// How long (ms) detection must fail continuously before showing "no-face"
const NO_FACE_DEBOUNCE_MS = 1000;

// ─── Module-level singleton — model loaded once, reused forever ──────────────
let faceLandmarkerPromise = null;

async function getFaceLandmarker() {
  if (!faceLandmarkerPromise) {
    faceLandmarkerPromise = (async () => {
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
      );
      return FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
          delegate: 'GPU',
        },
        outputFaceBlendshapes: false,
        runningMode: 'VIDEO',
        numFaces: 1,
      });
    })();
  }
  return faceLandmarkerPromise;
}

// ─── Utility: crop the lip region from a video frame onto a canvas ───────────
function drawLipCrop(videoEl, landmarker, outCanvas, timestampMs) {
  const result = landmarker.detectForVideo(videoEl, timestampMs);

  if (!result?.faceLandmarks?.length) return false;

  const landmarks = result.faceLandmarks[0];
  const W = videoEl.videoWidth;
  const H = videoEl.videoHeight;

  const mouthPts = MOUTH_OUTER.map((i) => ({
    x: landmarks[i].x * W,
    y: landmarks[i].y * H,
  }));

  const xs = mouthPts.map((p) => p.x);
  const ys = mouthPts.map((p) => p.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  const bboxW = maxX - minX;
  const bboxH = maxY - minY;

  const cropX = Math.max(0, Math.round(minX - bboxW * PAD_X));
  const cropY = Math.max(0, Math.round(minY - bboxH * PAD_Y));
  const cropW = Math.min(W - cropX, Math.round(bboxW * (1 + 2 * PAD_X)));
  const cropH = Math.min(H - cropY, Math.round(bboxH * (1 + 2 * PAD_Y)));

  if (cropW <= 0 || cropH <= 0) return false;

  if (outCanvas.width !== cropW || outCanvas.height !== cropH) {
    outCanvas.width = cropW;
    outCanvas.height = cropH;
  }

  const ctx = outCanvas.getContext('2d');
  ctx.drawImage(videoEl, cropX, cropY, cropW, cropH, 0, 0, cropW, cropH);
  return true;
}

// ─── LipPreview ──────────────────────────────────────────────────────────────
export default function LipPreview({ videoRef, videoSrc }) {
  const canvasRef = useRef(null);

  const [status, setStatus] = useState('loading');

  const landmarkerRef = useRef(null);
  const intervalRef = useRef(null);
  const noFaceTimerRef = useRef(null);
  const lastStatusRef = useRef('loading');

  // Tracks the highest timestamp we've ever sent to the model, so we can
  // always send a strictly larger one — even after the video replays from
  // the start, which would otherwise send a smaller number than before.
  const lastTsRef = useRef(0);

  const updateStatus = useCallback((next) => {
    if (lastStatusRef.current !== next) {
      lastStatusRef.current = next;
      setStatus(next);
    }
  }, []);

  // Always-increasing timestamp source, independent of video playback time.
  const getNextTimestamp = useCallback(() => {
    const next = Math.max(lastTsRef.current + 1, Math.round(performance.now()));
    lastTsRef.current = next;
    return next;
  }, []);

  const stopLoop = useCallback(() => {
    if (intervalRef.current !== null) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (noFaceTimerRef.current !== null) {
      clearTimeout(noFaceTimerRef.current);
      noFaceTimerRef.current = null;
    }
  }, []);

  const runDetection = useCallback(() => {
    const videoEl = videoRef?.current;
    const canvas = canvasRef.current;
    const landmarker = landmarkerRef.current;

    if (
      !videoEl ||
      !canvas ||
      !landmarker ||
      videoEl.readyState < 2 ||
      videoEl.videoWidth === 0
    ) return;

    const timestampMs = getNextTimestamp();

    try {
      const detected = drawLipCrop(videoEl, landmarker, canvas, timestampMs);

      if (detected) {
        if (noFaceTimerRef.current !== null) {
          clearTimeout(noFaceTimerRef.current);
          noFaceTimerRef.current = null;
        }
        updateStatus('ok');
      } else {
        if (noFaceTimerRef.current === null && lastStatusRef.current !== 'loading') {
          noFaceTimerRef.current = setTimeout(() => {
            noFaceTimerRef.current = null;
            updateStatus('no-face');
          }, NO_FACE_DEBOUNCE_MS);
        }
      }
    } catch (err) {
      if (!String(err).includes('timestamp')) {
        console.warn('[LipPreview] frame detection error:', err);
      }
    }
  }, [videoRef, updateStatus, getNextTimestamp]);

  const startLoop = useCallback(() => {
    if (intervalRef.current !== null) return;
    intervalRef.current = setInterval(runDetection, DETECT_INTERVAL_MS);
  }, [runDetection]);

  useEffect(() => {
    if (!videoSrc) return;

    let cancelled = false;
    updateStatus('loading');
    lastStatusRef.current = 'loading';

    (async () => {
      try {
        const landmarker = await getFaceLandmarker();
        if (cancelled) return;

        landmarkerRef.current = landmarker;

        const videoEl = videoRef?.current;
        if (!videoEl) return;

        const doInitialSnapshot = () => {
          if (cancelled) return;
          const canvas = canvasRef.current;
          if (!canvas || !landmarkerRef.current || videoEl.videoWidth === 0) return;

          try {
            const ts = getNextTimestamp();
            drawLipCrop(videoEl, landmarkerRef.current, canvas, ts);
            if (canvas.width > 0) updateStatus('ok');
          } catch (_) {
            // Silently ignore; the live loop will catch the next good frame.
          }
        };

        if (videoEl.readyState >= 2) {
          doInitialSnapshot();
        } else {
          videoEl.addEventListener('loadeddata', doInitialSnapshot, { once: true });
        }

        const onPlay = () => { if (!cancelled) startLoop(); };
        const onPause = () => stopLoop();
        const onEnded = () => stopLoop();

        videoEl.addEventListener('play', onPlay);
        videoEl.addEventListener('pause', onPause);
        videoEl.addEventListener('ended', onEnded);

        if (!videoEl.paused && !videoEl.ended) startLoop();

        return () => {
          videoEl.removeEventListener('play', onPlay);
          videoEl.removeEventListener('pause', onPause);
          videoEl.removeEventListener('ended', onEnded);
        };
      } catch (err) {
        if (!cancelled) {
          console.error('[LipPreview] model load error:', err);
          updateStatus('error');
        }
      }
    })();

    return () => {
      cancelled = true;
      stopLoop();
      landmarkerRef.current = null;
    };
  }, [videoSrc, videoRef, startLoop, stopLoop, updateStatus, getNextTimestamp]);

  return (
    <div className="lip-preview-card">
      <div className="lip-preview-header">
        <svg
          className="lip-preview-icon"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={1.75}
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M7 10c1.5 2.5 8.5 2.5 10 0M7 10c0-1.5 10-1.5 10 0M7 14c1.5 2.5 8.5 2.5 10 0"
          />
        </svg>
        <span className="lip-preview-label">Lip Region</span>
        {status === 'ok' && (
          <span className="lip-preview-badge">Live</span>
        )}
      </div>

      <div className="lip-preview-body">
        {status === 'loading' && (
          <div className="lip-preview-state">
            <div className="lip-preview-spinner" />
            <span className="lip-preview-state-text">Loading face model…</span>
          </div>
        )}

        {status === 'no-face' && (
          <div className="lip-preview-state lip-preview-state--warn">
            <svg
              className="lip-preview-warn-icon"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
            <span className="lip-preview-state-text">
              Couldn't detect a face — try a clearer angle
            </span>
          </div>
        )}

        {status === 'error' && (
          <div className="lip-preview-state lip-preview-state--warn">
            <svg
              className="lip-preview-warn-icon"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
            <span className="lip-preview-state-text">
              Couldn't load face model — check your connection
            </span>
          </div>
        )}

        <canvas
          ref={canvasRef}
          className="lip-preview-canvas"
          style={{ display: status === 'ok' ? 'block' : 'none' }}
        />
      </div>
    </div>
  );
}
