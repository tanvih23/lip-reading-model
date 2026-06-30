/*
 * App.jsx — LipReader Frontend
 * TAM-VIT AI/ML Club
 *
 * Retro pixel UI for the lip reading model.
 * Calls POST /predict on the FastAPI backend (backend/main.py)
 * with a video file, receives { predicted_word, confidence, all_scores }
 * and displays the result.
 *
 * Components:
 *   Deco         — floating pixel decorations (hearts, diamonds, stars)
 *   Win          — reusable retro window chrome
 *   PixelBar     — animated pixel progress bar
 *   WordBar      — single confidence score bar for one word
 *   TypedText    — types out text letter by letter with blinking cursor
 *   StartScreen  — boot/start screen shown before the app loads
 *   App          — main application
 */

import { useMemo, useState, useEffect,useRef} from 'react';
import LipPreview from './LipPreview';


// Backend URL — set VITE_API_URL in a .env file to override
const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

// ── pixel decoration positions ─────────────────────────────────
// These are the little hearts, diamonds and stars that float
// around the edges of the page. Pure CSS, no images.
const DECO_SHAPES = [
  { cls: 'deco-heart',    style: { top: '8%',  left: '2%'  } },
  { cls: 'deco-diamond',  style: { top: '6%',  right: '3%' } },
  { cls: 'deco-heart',    style: { top: '32%', left: '1%'  } },
  { cls: 'deco-star',     style: { top: '55%', right: '2%' } },
  { cls: 'deco-diamond',  style: { top: '72%', left: '2%'  } },
  { cls: 'deco-star',     style: { top: '14%', right: '8%' } },
  { cls: 'deco-dot-trio', style: { top: '42%', right: '1%' } },
  { cls: 'deco-dot-trio', style: { top: '18%', left: '8%'  } },
];

function Deco() {
  return (
    <>
      {DECO_SHAPES.map((d, i) => (
        <div key={i} className={`deco ${d.cls}`} style={d.style} />
      ))}
    </>
  );
}

// ── retro window chrome ────────────────────────────────────────
// Wraps any content in a pixel-style window with a title bar.
// accent=true gives it a pink title bar instead of blue
// (used for the output window to make it stand out).
function Win({ title, accent = false, children, className = '' }) {
  return (
    <div className={`win ${accent ? 'win-accent' : ''} ${className}`}>
      <div className="win-tb">
        <div className="win-dots">
          <span /><span /><span />
        </div>
        <span className="win-title">{title}</span>
        <button className="win-x">X</button>
      </div>
      <div className="win-body">{children}</div>
    </div>
  );
}

// ── pixel progress bar ─────────────────────────────────────────
// When animated=true: cycles through blocks like a loading bar.
// When animated=false: fills to `value` percent (0-100).
// color='pink' or color='blue' controls the fill color.
function PixelBar({ value = 0, animated = false, color = 'pink' }) {
  const [tick, setTick] = useState(0);
  

  useEffect(() => {
    if (!animated) return;
    const id = setInterval(() => setTick(t => (t + 1) % 18), 80);
    return () => clearInterval(id);
  }, [animated]);

  const blocks = 18;
  const filled = animated
    ? tick
    : Math.round((value / 100) * blocks);

  return (
    <div className="pbar">
      {Array.from({ length: blocks }).map((_, i) => (
        <div
          key={i}
          className={`pbar-b ${i < filled ? `pbar-${color}` : ''}`}
        />
      ))}
    </div>
  );
}

// ── word confidence bar ────────────────────────────────────────
// One row in the "all scores" section of the result panel.
// isTop=true highlights the predicted word in pink.
function WordBar({ word, score, isTop }) {
  const pct = Math.round(score * 100);
  return (
    <div className={`wbar ${isTop ? 'wbar-top' : ''}`}>
      <span className="wbar-lbl">{word.toUpperCase()}</span>
      <div className="wbar-track">
        <div className="wbar-fill" style={{ width: `${pct}%` }} />
      </div>
      <span className="wbar-pct">{pct}%</span>
    </div>
  );
}

// ── typed text animation ───────────────────────────────────────
// Types out `text` letter by letter at 60ms per character.
// After finishing, shows a blinking underscore cursor.
// Used on the predicted word in the result panel.
function TypedText({ text, className = '' }) {
  const [shown, setShown] = useState('');
  const [done, setDone]   = useState(false);
  const [cur, setCur]     = useState(true);

  // restart animation every time the text changes
  useEffect(() => {
    setShown(''); setDone(false);
    let i = 0;
    const id = setInterval(() => {
      setShown(text.slice(0, i + 1));
      i++;
      if (i >= text.length) { clearInterval(id); setDone(true); }
    }, 60);
    return () => clearInterval(id);
  }, [text]);

  // blink cursor after typing finishes
  useEffect(() => {
    if (!done) return;
    const id = setInterval(() => setCur(c => !c), 500);
    return () => clearInterval(id);
  }, [done]);

  return (
    <span className={className}>
      {shown}
      <span className="typing-cur" style={{ opacity: cur ? 1 : 0 }}>_</span>
    </span>
  );
}

// ── start screen ───────────────────────────────────────────────
// Shown before the main app. Animates a progress bar loading,
// then reveals PLAY / MENU / EXIT buttons. Clicking anywhere
// (or clicking PLAY) calls onStart() to enter the main app.
function StartScreen({ onStart }) {
  const [blink, setBlink]       = useState(true);
  const [progress, setProgress] = useState(0);
  const [ready, setReady]       = useState(false);

  // blink "press any key" text
  useEffect(() => {
    const b = setInterval(() => setBlink(v => !v), 530);
    return () => clearInterval(b);
  }, []);

  // animate the loading bar filling up
  useEffect(() => {
    if (progress < 100) {
      const t = setTimeout(
        () => setProgress(p => Math.min(p + 5, 100)),
        60
      );
      return () => clearTimeout(t);
    } else {
      // short pause after 100% before showing buttons
      const t = setTimeout(() => setReady(true), 200);
      return () => clearTimeout(t);
    }
  }, [progress]);

  return (
    <div className="start" onClick={ready ? onStart : undefined}>
      <Deco />
      <div className="start-inner">
        {/* main card */}
        <div className="start-card">
          <div className="start-card-tb">
            <div className="win-dots"><span /><span /><span /></div>
            <span className="start-card-url">lipreader.exe / start</span>
            <button className="win-x">X</button>
          </div>
          <div className="start-card-body">
            <p className="start-tag">-- TAM-VIT AI/ML CLUB --</p>
            <h1 className="start-heading">
              <span className="start-h-line1">LIP</span>
              <span className="start-h-line2">READER</span>
            </h1>
            <p className="start-sub">
              CNN + LSTM &nbsp;|&nbsp; GRID CORPUS &nbsp;|&nbsp; 12 WORDS
            </p>
            <div className="start-bar-wrap">
              <PixelBar value={progress} color="blue" />
              <p className="start-bar-lbl">
                {progress < 100 ? `LOADING ${progress}%` : 'SYSTEM READY'}
              </p>
            </div>
            {ready && (
              <div className="start-btns">
                <button className="btn-play" onClick={onStart}>PLAY</button>
              </div>
            )}
            <p className={`start-press ${blink ? 'op1' : 'op0'}`}>
              {ready ? '-- PRESS ANY KEY TO START --' : 'INITIALIZING...'}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── main app ───────────────────────────────────────────────────
export default function App() {
  const [started, setStarted]           = useState(false);
  const [file, setFile]                 = useState(null);
  const [previewUrl, setPreviewUrl]     = useState('');
  const [status, setStatus]             = useState('READY');
  const [result, setResult]             = useState(null);
  const [error, setError]               = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [dragging, setDragging]         = useState(false);
  const [clock, setClock]               = useState('');
  const [scanY, setScanY]               = useState(0);
  const videoRef = useRef(null);

  const acceptedTypes = useMemo(() => '.mp4,.mpeg,.mpg,.mov,.webm', []);

  // live clock in taskbar
  useEffect(() => {
    const tick = () => setClock(
      new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    );
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

  // sweeping shine effect position on the header logo
  useEffect(() => {
    const id = setInterval(() => setScanY(y => (y + 1) % 110), 20);
    return () => clearInterval(id);
  }, []);

  // handle a new file being selected or dropped
  const handleFile = (f) => {
    setError('');
    setResult(null);
    setFile(f);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    if (f) {
      setPreviewUrl(URL.createObjectURL(f));
      setStatus('FILE LOADED');
    } else {
      setPreviewUrl('');
      setStatus('READY');
    }
  };

  // send video to FastAPI and display result
  const onSubmit = async (e) => {
    e.preventDefault();
    if (!file) { setError('NO FILE SELECTED'); return; }

    setIsSubmitting(true);
    setStatus('ANALYZING...');
    setError('');
    setResult(null);

    try {
      const fd = new FormData();
      fd.append('video', file);  // must match FastAPI parameter name

      const res  = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        body: fd,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'PREDICTION FAILED');

      // handle both response shapes:
      //   new: { predicted_word, confidence, all_scores }  (inference.py)
      //   old: { prediction }  (fallback)
      setResult(
        data.predicted_word !== undefined
          ? data
          : { predicted_word: data.prediction || 'UNKNOWN', confidence: 1, all_scores: {} }
      );
      setStatus('COMPLETE!');
    } catch (err) {
      setError(err.message || 'UNEXPECTED ERROR');
      setStatus('ERROR');
    } finally {
      setIsSubmitting(false);
    }
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) handleFile(f);
  };

  // sort all word scores highest first for the bar chart
  const sortedScores = result?.all_scores
    ? Object.entries(result.all_scores).sort((a, b) => b[1] - a[1])
    : [];

  // show start screen first
  if (!started) return <StartScreen onStart={() => setStarted(true)} />;

  return (
    <div className="app">
      <Deco />

      {/* ── header bar ── */}
      <header className="app-hdr">
        <div className="app-hdr-inner">
          <div className="hdr-logo-wrap">
            {/* sweeping shine effect via CSS custom property */}
            <span className="hdr-logo" style={{ '--scan-y': `${scanY}%` }}>
              LIPREADER.EXE
            </span>
          </div>
          <div className="hdr-center"></div>
          <div className="hdr-right">
            <span className="hdr-clock">{clock}</span>
          </div>
        </div>
        {/* pink / yellow / blue pixel stripe under the header */}
        <div className="hdr-border-deco" />
      </header>

      {/* ── two-column layout ── */}
      <div className="layout">

        {/* LEFT: upload + preview */}
        <div className="col-l">

          {/* upload window */}
          <Win title="UPLOAD_CLIP.EXE" className="upload-win">
            <form onSubmit={onSubmit}>

              {/* drag-and-drop zone */}
              <label
                className={`dropzone ${dragging ? 'dz-over' : ''} ${file ? 'dz-has' : ''}`}
                onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
              >
                <input
                  type="file"
                  accept={acceptedTypes}
                  style={{ display: 'none' }}
                  onChange={(e) => handleFile(e.target.files?.[0] || null)}
                />
                <div className="dz-inner">
                  <div className={`dz-icon ${file ? 'dz-icon-loaded' : ''}`}>
                    {file ? '[VIDEO]' : '[  +  ]'}
                  </div>
                  {file ? (
                    <>
                      <p className="dz-fname">{file.name}</p>
                      <p className="dz-hint">click to replace</p>
                    </>
                  ) : (
                    <>
                      <p className="dz-main">DROP VIDEO FILE HERE</p>
                      <p className="dz-hint">MP4 / MOV / WEBM / MPG</p>
                    </>
                  )}
                </div>
              </label>

              {/* submit button — shimmer effect on hover via CSS */}
              <button
                className="run-btn"
                type="submit"
                disabled={isSubmitting}
              >
                <span className="run-btn-inner">
                  {isSubmitting ? '[ ANALYZING... ]' : '[ RUN INFERENCE ]'}
                </span>
              </button>

              {/* loading bar while request is in flight */}
              {isSubmitting && (
                <div className="mt8">
                  <PixelBar animated color="pink" />
                </div>
              )}

              {/* error message */}
              {error && (
                <div className="error-box">
                  <span className="err-tag">[ERR]</span> {error}
                </div>
              )}
            </form>
          </Win>

          {/* video preview window */}
          <Win title="PREVIEW.MP4">
          {previewUrl ? (
          <>
            <video
              key={previewUrl}
              ref={videoRef}
              className="preview-vid"
              controls
              preload="auto"
              src={previewUrl}
              onLoadedData={() => console.log('[video] loadeddata fired, readyState:', videoRef.current?.readyState)}
              onError={(e) => console.error('[video] error event:', e.target.error)}
            />
          <LipPreview videoRef={videoRef} videoSrc={previewUrl} />
          </>
          ) : (
            <div className="preview-empty">
            <p className="pe-icon">[  ]</p>
            <p className="pe-txt">NO FILE LOADED</p>
            <p className="pe-sub">video preview appears here</p>
          </div>
          )}
        </Win>

        </div>

        {/* RIGHT: result + sysinfo */}
        <div className="col-r">

          {/* result window — accent blue border */}
          <Win title="OUTPUT.TXT" accent>
            {result ? (
              <>
                {/* big predicted word display */}
                <div className="result-hero">
                  <p className="result-label">// DETECTED WORD</p>
                  <p className="result-word">
                    {/* types out the word letter by letter */}
                    <TypedText
                      text={result.predicted_word.toUpperCase()}
                      className="result-typed"
                    />
                  </p>
                  <p className="result-conf">
                    {Math.round(result.confidence * 100)}% CONFIDENCE
                  </p>
                  <PixelBar
                    value={Math.round(result.confidence * 100)}
                    color="blue"
                  />
                </div>

                {/* all 12 word scores as horizontal bars */}
                {sortedScores.length > 0 && (
                  <div className="scores">
                    <p className="scores-lbl">// ALL SCORES</p>
                    {sortedScores.map(([w, s]) => (
                      <WordBar
                        key={w}
                        word={w}
                        score={s}
                        isTop={w === result.predicted_word}
                      />
                    ))}
                  </div>
                )}
              </>
            ) : (
              /* placeholder before first inference */
              <div className="result-empty">
                <p className="re-icon">[?]</p>
                <p className="re-title">AWAITING INPUT</p>
                <p className="re-sub">upload a clip and run inference</p>
              </div>
            )}
          </Win>

          {/* system info window */}
          <Win title="SYSINFO.EXE">
            <table className="sys-tbl">
              <tbody>
                <tr><td>MODEL</td><td>CNN + LSTM</td></tr>
                <tr><td>VOCAB</td><td>12 WORDS</td></tr>
                <tr><td>DATASET</td><td>GRID CORPUS</td></tr>
                <tr><td>FRAMES</td><td>20 / CLIP</td></tr>
                <tr><td>INPUT</td><td>96 x 96 px</td></tr>
                <tr><td>EPOCHS</td><td>100</td></tr>
                <tr><td>ACCURACY</td><td className="sys-acc">99.9%  *</td></tr>
              </tbody>
            </table>
          </Win>

        </div>
      </div>

      {/* ── taskbar ── */}
      <div className="taskbar">
        <button className="tb-start">[ START ]</button>
        <div className="tb-mid">
          <span className="tb-item tb-on">LIPREADER.EXE</span>
          <span className="tb-item">OUTPUT.TXT</span>
          <span className="tb-item">PREVIEW.MP4</span>
        </div>
        <span className="tb-clock">{clock}</span>
      </div>
    </div>
  );
}