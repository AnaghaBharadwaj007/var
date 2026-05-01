'use strict';

/* ============================================================
   AR Hand Vision — script.js
   ============================================================
   Flow:
     1. init() → resize canvas, wire events
     2. initHands() → configure MediaPipe Hands model
     3. startCamera() → getUserMedia (front cam) → video feed
     4. processFrames() → send each video frame to MediaPipe
     5. onHandResults() → store latest landmarks, hide loading
     6. drawLoop() → runs every rAF; draws skeleton + AR boxes
   ============================================================ */

// ── Configuration constants ────────────────────────────────────
const MEDIAPIPE_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/hands/';
const LERP = 0.18;   // box position smoothing (0 = static, 1 = instant)
const SPEECH_TTL = 3200;   // ms before speech box starts fading

// ── Gesture lookup (finger count → display text + RGB accent) ──
const GESTURES = {
  0: { text: 'Hello', r: 0, g: 212, b: 255 },   // cyan
  1: { text: 'One', r: 0, g: 212, b: 255 },   // cyan
  2: { text: 'Peace', r: 168, g: 85, b: 247 },   // purple
  3: { text: 'Three', r: 168, g: 85, b: 247 },   // purple
  4: { text: 'Four', r: 168, g: 85, b: 247 },   // purple
  5: { text: 'Stop', r: 255, g: 0, b: 110 },   // hot-pink
};

// ── Standard MediaPipe hand skeleton connections ───────────────
const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],          // thumb
  [0, 5], [5, 6], [6, 7], [7, 8],          // index
  [5, 9], [9, 10], [10, 11], [11, 12],     // middle
  [9, 13], [13, 14], [14, 15], [15, 16],   // ring
  [0, 17], [17, 18], [18, 19], [19, 20],   // pinky
  [13, 17],                                 // lower palm
];

// ── DOM references ─────────────────────────────────────────────
const videoEl = document.getElementById('video');
const canvasEl = document.getElementById('canvas');
const ctx = canvasEl.getContext('2d');
const loadingEl = document.getElementById('loading');
const loadingTxt = document.getElementById('loading-text');
const statusDot = document.getElementById('status-dot');
const statusTxt = document.getElementById('status-text');
const micBtn = document.getElementById('mic-btn');
const cameraBtn = document.getElementById('camera-btn');
const handHint = document.getElementById('hand-hint');

// ── Runtime state ──────────────────────────────────────────────
let latestHands = [];         // [{ lm: Landmark[21], handedness: 'Left'|'Right'|string }, ...]
let modelReady = false;  // true after first MediaPipe result
let frameRunning = false;  // prevents queuing concurrent sends
let currentFacingMode = 'user';  // 'user' for front, 'environment' for back
let isMirrored = true;          // true for front camera (mirrored), false for back

// Per-hand visual smoothing + motion info (keyed by handedness label when available)
const handViz = new Map(); // key -> { boxX, boxY, lastPalmX, lastPalmT }

// Speech overlay state
let speechText = '';
let speechOpacity = 0;
let speechFadeId = null;
let isListening = false;
let wantListening = false; // user-intent toggle (button on/off)
let lastSpeechMs = 0;     // last time we updated transcript (for fade timing)

// ══════════════════════════════════════════════════════════════
//  1. MEDIAPIPE HANDS INIT
// ══════════════════════════════════════════════════════════════
let hands;

function initHands() {
  if (typeof Hands === 'undefined') {
    setLoadingText('⚠ MediaPipe failed to load. Check network connection.');
    return;
  }

  hands = new Hands({
    // Tell MediaPipe where to find its WASM/BIN files
    locateFile: (file) => `${MEDIAPIPE_CDN}${file}`,
  });

  hands.setOptions({
    selfieMode: false,  // we mirror coordinates ourselves
    maxNumHands: 2,
    modelComplexity: 0,      // 0 = Lite (fast on mobile), 1 = Full
    minDetectionConfidence: 0.70,
    minTrackingConfidence: 0.50,
  });

  hands.onResults(onHandResults);
}

// ── MediaPipe result callback ──────────────────────────────────
function onHandResults(results) {
  // First successful result → model is loaded, hide loading screen
  if (!modelReady) {
    modelReady = true;
    hideLoading();
    setStatus('LIVE', 'live');
  }

  const lms = results.multiHandLandmarks ?? [];
  const hds = results.multiHandedness ?? [];
  latestHands = lms.map((lm, i) => ({
    lm,
    handedness: hds[i]?.label ?? `Hand${i + 1}`,
  }));
}

// ══════════════════════════════════════════════════════════════
//  2. CAMERA SETUP  (getUserMedia, no Camera-utils dependency)
// ══════════════════════════════════════════════════════════════
async function startCamera(facingMode = 'user') {
  setLoadingText('Accessing camera…');

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: facingMode },
        width: { ideal: 640 },
        height: { ideal: 480 },
      },
      audio: false,
    });

    videoEl.srcObject = stream;

    // Set mirroring based on camera
    isMirrored = (facingMode === 'user');
    videoEl.style.transform = isMirrored ? 'scaleX(-1)' : 'none';

    // Wait until video metadata is ready, then start playing
    await new Promise((resolve, reject) => {
      videoEl.onloadedmetadata = () => videoEl.play().then(resolve).catch(reject);
    });

    setLoadingText('Loading hand detection model…');
    processFrames();              // begin feeding frames to MediaPipe

  } catch (err) {
    console.error('Camera error:', err);
    setLoadingText('⚠ Camera denied. Please allow camera access and reload.');
  }
}

// Switch between front and back camera
async function switchCamera() {
  // Stop current stream
  if (videoEl.srcObject) {
    const stream = videoEl.srcObject;
    stream.getTracks().forEach(track => track.stop());
    videoEl.srcObject = null;
  }

  // Toggle facing mode
  currentFacingMode = (currentFacingMode === 'user') ? 'environment' : 'user';

  // Restart camera with new mode
  await startCamera(currentFacingMode);
}

// Continuously sends video frames to MediaPipe (skips if still processing)
async function processFrames() {
  // Only process when a frame is not already in-flight and video has data
  if (!frameRunning && hands && videoEl.readyState >= 2) {
    frameRunning = true;
    try {
      await hands.send({ image: videoEl });
    } catch (e) {
      // Silently ignore transient send errors (e.g. model still initialising)
    } finally {
      frameRunning = false;
    }
  }
  requestAnimationFrame(processFrames);
}

// ══════════════════════════════════════════════════════════════
//  3. GESTURE DETECTION
// ══════════════════════════════════════════════════════════════
/**
 * Detect simple finger states using landmarks.
 *
 * For the 4 main fingers: \"up\" when tip is above PIP (smaller Y).
 * For the thumb: \"up\" when tip is further outward on X (depends on handedness).
 *
 * This is intentionally simple demo logic (not production-accurate).
 */
function getFingerStates(lm, handednessLabel) {
  const Y_MARGIN = 0.015; // helps reduce flicker from tiny landmark jitters
  const X_MARGIN = 0.010;

  const index = lm[8].y < lm[6].y - Y_MARGIN;
  const middle = lm[12].y < lm[10].y - Y_MARGIN;
  const ring = lm[16].y < lm[14].y - Y_MARGIN;
  const pinky = lm[20].y < lm[18].y - Y_MARGIN;

  // Thumb uses X direction (works better than Y for many poses).
  // Right hand thumb tends to extend toward smaller X in image coords.
  // Left hand thumb tends to extend toward larger X in image coords.
  let thumb = false;
  if (handednessLabel === 'Right') thumb = lm[4].x < lm[3].x - X_MARGIN;
  else if (handednessLabel === 'Left') thumb = lm[4].x > lm[3].x + X_MARGIN;
  else {
    // Fallback: if handedness is missing, use a weak heuristic.
    thumb = Math.abs(lm[4].x - lm[3].x) > 0.05;
  }

  const count =
    (thumb ? 1 : 0) +
    (index ? 1 : 0) +
    (middle ? 1 : 0) +
    (ring ? 1 : 0) +
    (pinky ? 1 : 0);

  const mainCount =
    (index ? 1 : 0) +
    (middle ? 1 : 0) +
    (ring ? 1 : 0) +
    (pinky ? 1 : 0);

  return { thumb, index, middle, ring, pinky, count, mainCount };
}

/**
 * Classify a few \"AR demo\" gestures from finger states.
 * We prefer pattern-based matching (more stable than raw count).
 */
function classifyGesture(s, lm, ctxInfo = {}) {
  const allDown = !s.thumb && !s.index && !s.middle && !s.ring && !s.pinky;
  const onlyThumb = s.thumb && !s.index && !s.middle && !s.ring && !s.pinky;
  const onlyIndex = s.index && !s.thumb && !s.middle && !s.ring && !s.pinky;

  // Basic distance helper (normalised coord space)
  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

  // Required gesture (as requested)
  if (allDown) return { ...GESTURES[0], label: 'Fist', displayCount: 0 };

  // ── \"Deaf / sign-like\" easy gestures (simple pose checks) ──
  // I LOVE YOU (ASL): thumb + index + pinky up
  if (s.thumb && s.index && s.pinky && !s.middle && !s.ring) {
    return { text: 'Take care', r: 0, g: 255, b: 170, label: 'ILY' };
  }

  // CALL ME / SHAKA: thumb + pinky up
  if (s.thumb && s.pinky && !s.index && !s.middle && !s.ring) {
    return { text: 'Call Me', r: 255, g: 184, b: 0, label: 'Shaka' };
  }

  // OK: thumb tip close to index tip + (usually) 3 fingers up
  const okCircle = dist(lm[4], lm[8]) < 0.055;
  if (okCircle && s.middle && s.ring && s.pinky) {
    return { text: 'OK', r: 168, g: 85, b: 247, label: 'OK Sign' };
  }

  // THUMBS DOWN: thumb only, pointing down (thumb tip below wrist)
  if (onlyThumb && lm[4].y > lm[0].y + 0.06) {
    return { text: 'Thumbs Down', r: 255, g: 0, b: 110, label: 'Thumb' };
  }

  // Peace: index + middle up (thumb can be anything)
  if (s.index && s.middle && !s.ring && !s.pinky) return { ...GESTURES[2], label: 'Peace', displayCount: 2 };

  // Rock: index + pinky
  if (s.index && s.pinky && !s.middle && !s.ring) {
    return { text: 'Awesome', r: 255, g: 0, b: 110, label: 'Rock' };
  }

  // L shape: thumb + index
  if (s.thumb && s.index && !s.middle && !s.ring && !s.pinky) {
    return { text: 'L', r: 0, g: 255, b: 170, label: 'L' };
  }

  // One finger: Point (index up)
  if (onlyIndex) {
    return { text: 'Point', r: 0, g: 212, b: 255, label: 'Point' };
  }

  // Stop / Wave: open palm (4 main fingers up + thumb up)
  if (s.mainCount === 4 && s.thumb) {
    if (ctxInfo.isWaving) return { text: 'Hi', r: 0, g: 212, b: 255, label: 'Wave', displayCount: 5 };
    return { ...GESTURES[5], label: 'Stop', displayCount: 5 };
  }

  // Thumbs up (after thumbs-down, so down wins when pointing downward)
  if (onlyThumb) {
    return { text: 'Thumbs Up', r: 255, g: 184, b: 0, label: 'Thumb' };
  }

  // 3 and 4 fingers (without thumb) as extras
  if (s.index && s.middle && s.ring && !s.pinky) return { text: 'Three', r: 168, g: 85, b: 247, label: 'Three' };
  if (s.mainCount === 4 && !s.thumb) return { text: 'Four', r: 168, g: 85, b: 247, label: 'Four' };

  // Fallback to count-based labels so something shows for most poses
  if (GESTURES[s.count]) return { ...GESTURES[s.count], label: `${s.count} fingers` };
  return null;
}

// ══════════════════════════════════════════════════════════════
//  4. COORDINATE TRANSFORM
// ══════════════════════════════════════════════════════════════
/**
 * Convert a normalised MediaPipe landmark {x, y} to canvas pixel coords.
 *
 * Two corrections are applied:
 *   a) object-fit: cover   — landmarks must scale with the cropped video rect
 *   b) horizontal mirror   — video has CSS transform:scaleX(-1), so flip x
 */
function toCanvas(lm) {
  const vw = videoEl.videoWidth || 640;
  const vh = videoEl.videoHeight || 480;
  const cw = canvasEl.width;
  const ch = canvasEl.height;

  // Scale that fills both dimensions (object-fit: cover)
  const scale = Math.max(cw / vw, ch / vh);
  const ox = (cw - vw * scale) / 2;  // horizontal render offset
  const oy = (ch - vh * scale) / 2;  // vertical render offset

  // Mirror x if front camera: landmark at normalised 0 (raw-left) becomes the right edge
  const x = ox + vw * scale * (isMirrored ? (1 - lm.x) : lm.x);
  const y = oy + lm.y * vh * scale;

  return { x, y };
}

// ══════════════════════════════════════════════════════════════
//  5. CANVAS DRAWING HELPERS
// ══════════════════════════════════════════════════════════════

/** Trace a rounded-rectangle path without filling or stroking it. */
function roundRect(x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

/**
 * Draw the four L-shaped corner brackets that frame the hand bounding box.
 * This is the classic AR "target lock" visual.
 */
function drawCornerBrackets(x, y, w, h, r, g, b) {
  const arm = Math.min(w, h) * 0.22;
  const color = `rgb(${r},${g},${b})`;

  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.shadowBlur = 14;
  ctx.shadowColor = color;

  // Each corner: [startX, startY, cornerX, cornerY, endX, endY]
  const corners = [
    [x, y + arm, x, y, x + arm, y],  // top-left
    [x + w - arm, y, x + w, y, x + w, y + arm], // top-right
    [x, y + h - arm, x, y + h, x + arm, y + h],  // bottom-left
    [x + w - arm, y + h, x + w, y + h, x + w, y + h - arm],// bottom-right
  ];

  corners.forEach(([sx, sy, mx, my, ex, ey]) => {
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(mx, my);
    ctx.lineTo(ex, ey);
    ctx.stroke();
  });

  ctx.restore();
}

/**
 * Draw the hand skeleton: connections between landmarks + dots at each joint.
 * Finger-tip dots are larger and brighter for a "hot-spot" look.
 */
function drawSkeleton(lm, r, g, b) {
  const tipSet = new Set([4, 8, 12, 16, 20]);

  ctx.save();
  ctx.shadowBlur = 6;
  ctx.shadowColor = `rgba(${r},${g},${b},0.8)`;

  // Bone connections
  ctx.strokeStyle = `rgba(${r},${g},${b},0.42)`;
  ctx.lineWidth = 1.6;
  CONNECTIONS.forEach(([a, bIdx]) => {
    const pa = toCanvas(lm[a]);
    const pb = toCanvas(lm[bIdx]);
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.stroke();
  });

  // Joint dots
  lm.forEach((pt, i) => {
    const { x, y } = toCanvas(pt);
    const isTip = tipSet.has(i);
    ctx.beginPath();
    ctx.arc(x, y, isTip ? 5.5 : 2.8, 0, Math.PI * 2);
    ctx.fillStyle = isTip ? `rgb(${r},${g},${b})` : 'rgba(255,255,255,0.72)';
    ctx.shadowBlur = isTip ? 14 : 4;
    ctx.fill();
  });

  ctx.restore();
}

/**
 * Draw the AR floating gesture-text box above the palm centre.
 *
 * Layout:
 *   ┌─────────────────────────┐
 *   │  ◆ GESTURE DETECTED ◆  │  ← small coloured label
 *   │                         │
 *   │         Hello           │  ← large white text
 *   └─────────────────────────┘
 *               ╎               ← dashed connector to hand
 *
 * @param {number} cx     Smoothed palm-centre X (canvas px)
 * @param {number} cy     Smoothed palm-centre Y (canvas px)
 * @param {number} count  Finger count (shown as sub-label)
 * @param {object} g      Gesture entry {text, r, g, b}
 * @param {number} t      Time in seconds (drives pulsing border)
 */
function drawGestureBox(cx, cy, count, g, t) {
  const { text, r, g: gv, b } = g;
  const label = g.label ?? 'GESTURE DETECTED';

  ctx.save();

  // ── Measure box dimensions ──
  ctx.font = 'bold 28px "Courier New"';
  const textW = ctx.measureText(text).width;
  const boxW = Math.max(textW + 52, 172);
  const boxH = 86;
  const pad = 10;

  // Position: centred on palm, placed above it
  let bx = cx - boxW / 2;
  let by = cy - boxH - 30;

  // Clamp to canvas edges so the box never goes off-screen
  bx = Math.max(pad, Math.min(canvasEl.width - boxW - pad, bx));
  by = Math.max(pad, Math.min(canvasEl.height - boxH - pad, by));

  // Pulsing border (sine wave on alpha)
  const borderAlpha = 0.5 + Math.sin(t * 3.8) * 0.4;

  // ── Background fill ──
  ctx.shadowBlur = 26;
  ctx.shadowColor = `rgba(${r},${gv},${b},0.5)`;
  roundRect(bx, by, boxW, boxH, 14);
  ctx.fillStyle = 'rgba(0,0,0,0.80)';
  ctx.fill();

  // ── Coloured border ──
  roundRect(bx, by, boxW, boxH, 14);
  ctx.strokeStyle = `rgba(${r},${gv},${b},${borderAlpha})`;
  ctx.lineWidth = 1.6;
  ctx.stroke();
  ctx.shadowBlur = 0;

  // ── "◆ GESTURE DETECTED ◆" label ──
  ctx.fillStyle = `rgba(${r},${gv},${b},0.7)`;
  ctx.font = '9px "Courier New"';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillText(`◆ ${label.toUpperCase()} ◆`, bx + boxW / 2, by + 12);

  // ── Main gesture word ──
  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 28px "Courier New"';
  ctx.textBaseline = 'middle';
  ctx.shadowBlur = 14;
  ctx.shadowColor = `rgba(${r},${gv},${b},0.9)`;
  ctx.fillText(text, bx + boxW / 2, by + boxH * 0.64);
  ctx.shadowBlur = 0;

  // ── Finger-count sub-label ──
  ctx.fillStyle = `rgba(${r},${gv},${b},0.5)`;
  ctx.font = '9px "Courier New"';
  ctx.textBaseline = 'top';
  ctx.fillText(
    `[ ${count} finger${count !== 1 ? 's' : ''} ]`,
    bx + boxW / 2,
    by + boxH + 9,
  );

  // ── Dashed connector: box bottom → palm centre ──
  ctx.beginPath();
  ctx.moveTo(bx + boxW / 2, by + boxH);
  ctx.lineTo(cx, cy);
  ctx.strokeStyle = `rgba(${r},${gv},${b},0.32)`;
  ctx.lineWidth = 1;
  ctx.setLineDash([5, 5]);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.restore();
}

/**
 * Draw the speech-recognition subtitle box at the bottom of the screen.
 * Text is auto-wrapped. The box fades out via globalAlpha.
 *
 * @param {string} text     Recognised transcript
 * @param {number} opacity  0..1 (drives fade-out animation)
 */
function drawSpeechBox(text, opacity) {
  if (!text || opacity <= 0) return;

  const r = 168, gv = 85, b = 247;  // purple accent
  const cw = canvasEl.width;
  const ch = canvasEl.height;

  ctx.save();
  ctx.globalAlpha = opacity;

  // ── Word-wrap into lines ──
  const maxInnerW = Math.min(cw - 48, 420);
  const lineH = 23;

  ctx.font = '15px "Courier New"';
  const words = text.split(' ');
  const lines = [];
  let line = '';

  words.forEach((word) => {
    const test = line ? `${line} ${word}` : word;
    if (ctx.measureText(test).width > maxInnerW - 32) {
      if (line) lines.push(line);
      line = word;
    } else {
      line = test;
    }
  });
  if (line) lines.push(line);

  const boxW = maxInnerW;
  const boxH = 22 + lines.length * lineH + 18;  // header + lines + padding
  const bx = (cw - boxW) / 2;
  const by = ch - boxH - 78;                  // positioned above gesture guide

  // ── Background ──
  ctx.shadowBlur = 22;
  ctx.shadowColor = `rgba(${r},${gv},${b},0.5)`;
  roundRect(bx, by, boxW, boxH, 14);
  ctx.fillStyle = 'rgba(0,0,0,0.84)';
  ctx.fill();

  // ── Border ──
  roundRect(bx, by, boxW, boxH, 14);
  ctx.strokeStyle = `rgba(${r},${gv},${b},0.72)`;
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.shadowBlur = 0;

  // ── "◆ SPEECH RECOGNITION ◆" label ──
  ctx.fillStyle = `rgba(${r},${gv},${b},0.78)`;
  ctx.font = '9px "Courier New"';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillText('◆ SPEECH RECOGNITION ◆', bx + boxW / 2, by + 10);

  // ── Recognised text lines ──
  ctx.fillStyle = '#ffffff';
  ctx.font = '15px "Courier New"';
  ctx.textBaseline = 'top';
  lines.forEach((l, i) => {
    ctx.fillText(l, bx + boxW / 2, by + 28 + i * lineH);
  });

  ctx.restore();
}

// ══════════════════════════════════════════════════════════════
//  6. MAIN DRAW LOOP
// ══════════════════════════════════════════════════════════════
function drawLoop(timestamp) {
  requestAnimationFrame(drawLoop);

  const t = timestamp / 1000;  // seconds (used for animations)

  // Keep canvas sized to the window (cheap check each frame)
  if (canvasEl.width !== window.innerWidth || canvasEl.height !== window.innerHeight) {
    canvasEl.width = window.innerWidth;
    canvasEl.height = window.innerHeight;
  }

  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

  // ── Hands detected (up to 2) ─────────────────────────────────
  if (latestHands.length > 0) handHint.classList.add('hidden');
  else handHint.classList.remove('hidden');

  for (let i = 0; i < latestHands.length; i++) {
    const { lm, handedness } = latestHands[i];
    const key = handedness || `Hand${i + 1}`;
    const handLabel = (handedness === 'Left' || handedness === 'Right') ? handedness : `Hand ${i + 1}`;

    // Finger states (thumb depends on handedness, if available)
    const s = getFingerStates(lm, handedness);

    // Palm centre in canvas space (average of wrist + 4 knuckle bases)
    const palmIdx = [0, 5, 9, 13, 17];
    const pts = palmIdx.map((idx) => toCanvas(lm[idx]));
    const targetX = pts.reduce((sum, p) => sum + p.x, 0) / pts.length;
    const targetY = pts.reduce((sum, p) => sum + p.y, 0) / pts.length;

    // Per-hand smoothing + simple \"wave\" motion detection (x speed)
    const now = performance.now();
    let v = handViz.get(key);
    if (!v) {
      v = { boxX: targetX, boxY: targetY, lastPalmX: targetX, lastPalmT: now };
      handViz.set(key, v);
    }
    const dtMs = Math.max(1, now - v.lastPalmT);
    const vx = (targetX - v.lastPalmX) / dtMs; // px per ms
    v.lastPalmX = targetX;
    v.lastPalmT = now;

    v.boxX += (targetX - v.boxX) * LERP;
    v.boxY += (targetY - v.boxY) * LERP;

    const isWaving = (s.mainCount === 4 && s.thumb) && Math.abs(vx) > 0.35;

    const g = classifyGesture(s, lm, { isWaving });

    // Default accent: cyan; override with gesture colour if recognised
    const { r, g: gv, b } = g ?? { r: 0, g: 212, b: 255 };

    // Hand bounding box in canvas space
    const allPts = lm.map(toCanvas);
    const xs = allPts.map((p) => p.x);
    const ys = allPts.map((p) => p.y);
    const pad = 22;
    const bb = {
      x: Math.min(...xs) - pad,
      y: Math.min(...ys) - pad,
      w: Math.max(...xs) - Math.min(...xs) + pad * 2,
      h: Math.max(...ys) - Math.min(...ys) + pad * 2,
    };

    // ① Bone skeleton + joint dots
    drawSkeleton(lm, r, gv, b);

    // ② AR corner brackets around the hand bounding box
    drawCornerBrackets(bb.x, bb.y, bb.w, bb.h, r, gv, b);

    // ③ Floating gesture text box
    if (g) {
      const displayCount = (typeof g.displayCount === 'number') ? g.displayCount : s.count;
      drawGestureBox(
        v.boxX,
        v.boxY,
        displayCount,
        { ...g, label: `${handLabel} · ${g.label ?? 'Gesture'}` },
        t,
      );
    }
  }

  // ④ Speech subtitle box (drawn regardless of hand presence)
  drawSpeechBox(speechText, speechOpacity);
}

// ══════════════════════════════════════════════════════════════
//  7. SPEECH RECOGNITION
// ══════════════════════════════════════════════════════════════
let recognition = null;
let lastSpeechError = null;

function initSpeech() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;

  if (!SR) {
    // Speech not available (e.g. Firefox without flag) — disable button
    micBtn.disabled = true;
    micBtn.title = 'Speech recognition not supported on this browser';
    micBtn.style.opacity = '0.3';
    return;
  }

  recognition = new SR();
  recognition.continuous = true;   // keep listening until user turns mic off
  recognition.interimResults = true;   // show partial results live
  recognition.lang = 'en-US';

  recognition.onstart = () => {
    lastSpeechError = null;
    isListening = true;
    micBtn.classList.add('listening');
    setStatus('LISTENING', 'listening');
    setSpeechOverlay('Listening…');
  };

  recognition.onend = () => {
    isListening = false;
    micBtn.classList.remove('listening');
    if (wantListening && lastSpeechError !== 'network') {
      // Some Chrome builds stop recognition after a pause. If the user still
      // has the mic toggle ON, immediately restart to keep subtitles live.
      setTimeout(() => {
        if (!wantListening || isListening) return;
        try { recognition.start(); } catch { /* ignore double-start */ }
      }, 180);
      return;
    }

    setStatus('LIVE', 'live');

    // When the user turns the mic OFF, fade whatever was last captured.
    if (speechText && speechText !== 'Listening…') {
      clearTimeout(speechFadeId);
      const delay = Math.max(0, SPEECH_TTL - (performance.now() - lastSpeechMs));
      speechFadeId = setTimeout(startSpeechFade, delay);
    } else {
      speechText = '';
      speechOpacity = 0;
    }
  };

  recognition.onresult = (e) => {
    // Concatenate all result segments into one transcript
    const transcript = Array.from(e.results)
      .map((res) => res[0].transcript)
      .join('');

    speechText = transcript;
    speechOpacity = 1;
    lastSpeechMs = performance.now();

    clearTimeout(speechFadeId);

    // While the mic toggle is ON, never auto-fade — keep subtitles visible.
    // Fade is triggered when the user turns the mic OFF (see onend).
  };

  recognition.onerror = (e) => {
    // 'no-speech' is normal; others are worth logging
    if (e.error !== 'no-speech') console.warn('Speech error:', e.error);
    lastSpeechError = e.error;
    isListening = false;
    micBtn.classList.remove('listening');
    setStatus('LIVE', 'live');
    if (e.error === 'not-allowed' || e.error === 'service-not-allowed') {
      wantListening = false;
      showSpeechMessage('Microphone blocked. Allow mic permission, then try again.', 3000);
    } else if (e.error === 'network') {
      wantListening = false;
      showSpeechMessage('Speech service network error. Check internet + try again.', 3200);
    } else if (e.error !== 'no-speech') {
      showSpeechMessage(`Speech error: ${e.error}`, 2200);
    }
  };
}

/** Smoothly fade out the speech box over ~1.5 s using rAF. */
function startSpeechFade() {
  const step = () => {
    speechOpacity -= 0.018;
    if (speechOpacity > 0) {
      requestAnimationFrame(step);
    } else {
      speechOpacity = 0;
      speechText = '';
    }
  };
  requestAnimationFrame(step);
}

// Mic button toggles recognition on/off
micBtn.addEventListener('click', () => {
  if (!recognition) return;
  if (!navigator.onLine) {
    showSpeechMessage('No network connection detected. Speech recognition needs internet.', 3200);
    return;
  }
  if (!window.isSecureContext) {
    showSpeechMessage('Speech needs HTTPS (or localhost). Open via http://localhost:8080.', 3200);
    return;
  }
  try {
    wantListening = !wantListening;

    if (wantListening) {
      // Turn mic ON: start (or restart) recognition and keep overlay visible.
      clearTimeout(speechFadeId);
      setSpeechOverlay('Listening…');
      if (!isListening) recognition.start();
    } else {
      // Turn mic OFF: stop recognition; onend will schedule the fade-out.
      if (isListening) recognition.stop();
      micBtn.classList.remove('listening');
    }
  } catch (e) {
    showSpeechMessage('Could not start speech. Check mic permissions.', 2600);
  }
});

// Camera button switches between front and back camera
cameraBtn.addEventListener('click', async () => {
  try {
    await switchCamera();
  } catch (e) {
    console.error('Camera switch error:', e);
    showSpeechMessage('Could not switch camera. Check permissions.', 2600);
  }
});

// ══════════════════════════════════════════════════════════════
//  8. STATUS / LOADING HELPERS
// ══════════════════════════════════════════════════════════════
function setStatus(text, mode) {
  statusTxt.textContent = text;
  statusDot.className = 'pulse-dot';   // clear old mode classes
  if (mode) statusDot.classList.add(mode);
  statusTxt.style.color = mode === 'listening' ? '#ff006e' : '#00d4ff';
}

function setLoadingText(text) {
  loadingTxt.textContent = text;
}

function hideLoading() {
  loadingEl.classList.add('fade-out');
  setTimeout(() => { loadingEl.style.display = 'none'; }, 600);
}

function showSpeechMessage(text, ttlMs = SPEECH_TTL) {
  speechText = text;
  speechOpacity = 1;
  clearTimeout(speechFadeId);
  speechFadeId = setTimeout(startSpeechFade, ttlMs);
}

function setSpeechOverlay(text) {
  speechText = text;
  speechOpacity = 1;
  lastSpeechMs = performance.now();
  clearTimeout(speechFadeId);
}

// ══════════════════════════════════════════════════════════════
//  9. INIT
// ══════════════════════════════════════════════════════════════
function init() {
  // Size canvas to window immediately
  canvasEl.width = window.innerWidth;
  canvasEl.height = window.innerHeight;

  // Keep canvas in sync with window resizes (e.g. rotation)
  window.addEventListener('resize', () => {
    canvasEl.width = window.innerWidth;
    canvasEl.height = window.innerHeight;
  });

  initHands();    // configure MediaPipe
  initSpeech();   // configure Web Speech API
  startCamera(currentFacingMode);  // request getUserMedia + begin frame loop

  // Start rendering immediately so speech overlays appear even while the hand
  // model is still loading.
  requestAnimationFrame(drawLoop);
}

init();

// Register Service Worker for PWA
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/service-worker.js')
      .then(() => console.log('SW registered'))
      .catch(err => console.log('SW error', err));
  });
}
