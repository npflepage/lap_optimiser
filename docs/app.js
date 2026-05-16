/* ═══════════════════════════════════════════════════════════════════
   APEX v2 // application logic
   ─ loads data/index.json → data/<Track>.json
   ─ track canvas: speed-coloured racing line, zoomable (scroll +
     drag to pan), reset zoom on double-click or button
   ─ playback at near-real speed via arc-length / speed_ms
   ─ car glyph oriented by heading
   ─ G-G Frenet envelope (red ellipse) + live acceleration dot (blue)
   ─ full-width trace: SPEED | |ACCEL| | BOTH  (lat + lon)
   ═══════════════════════════════════════════════════════════════════ */
'use strict';

// ── Application state ──────────────────────────────────────────────
const S = {
  index: null, track: null, sol: null, resIx: 0,
  frame: 0, playing: false, lastT: 0, accT: 0,
  trace: 'speed',          // 'speed' | 'accel' | 'both'
  // zoom / pan state (in data-space)
  zoom: { scale: 1, ox: 0, oy: 0,   // current view offset in canvas px
          dragging: false, dragX: 0, dragY: 0,
          xmin:0,xmax:1,ymin:0,ymax:1 }, // data extents
};

const $ = id => document.getElementById(id);

// ── Colour maps ────────────────────────────────────────────────────
function speedColor(t) {
  const stops = [[0xd7,0x19,0x1c],[0xfd,0xae,0x61],[0xff,0xff,0xbf],
                 [0xa6,0xd9,0x6a],[0x1a,0x96,0x41]];
  t = Math.max(0, Math.min(1, t));
  const f = t*(stops.length-1), i = Math.floor(f), g = f-i;
  const a = stops[i], b = stops[Math.min(i+1, stops.length-1)];
  return `rgb(${a[0]+(b[0]-a[0])*g|0},${a[1]+(b[1]-a[1])*g|0},${a[2]+(b[2]-a[2])*g|0})`;
}

// ── Bootstrap ──────────────────────────────────────────────────────
async function boot() {
  S.index = await fetch('data/index.json').then(r => r.json());
  const sel = $('trackSelect');
  S.index.tracks.forEach(t => {
    const o = document.createElement('option');
    o.value = t.name;
    o.textContent = t.name.replace(/([A-Z])/g,' $1').trim().toUpperCase();
    sel.appendChild(o);
  });
  sel.addEventListener('change', () => loadTrack(sel.value));
  await loadTrack(S.index.tracks[0].name);
  wireControls();
  requestAnimationFrame(loop);
  window.addEventListener('resize', () => { drawTrack(); drawTrace(); });
}

async function loadTrack(name) {
  stop();
  S.track = await fetch(`data/${name}.json`).then(r => r.json());
  S.resIx = S.track.solutions.length - 1;
  buildResTicks();
  $('resSlider').max   = S.track.solutions.length - 1;
  $('resSlider').value = S.resIx;
  selectRes(S.resIx);
  $('trackTitle').textContent = name.replace(/([A-Z])/g,' $1').trim().toUpperCase();
  $('trackLen').textContent   = S.track.track_length.toLocaleString() + ' m';
  resetZoom();
}

function buildResTicks() {
  const box = $('resTicks'); box.innerHTML = '';
  S.track.resolutions.forEach(n => {
    const s = document.createElement('span'); s.textContent = n; box.appendChild(s);
  });
}

function selectRes(ix) {
  S.resIx = ix;
  S.sol   = S.track.solutions[ix];
  S.frame = 0;
  const N = S.sol.N;
  $('resValue').textContent    = `N = ${N}`;
  $('resBadge').textContent    = `N=${N}`;
  $('resLap').textContent      = S.sol.lap_time.toFixed(3) + ' s';
  $('frameTot').textContent    = S.sol.x.length - 1;
  $('progress').max            = S.sol.x.length - 1;
  $('lapBadgeTime').textContent = S.sol.lap_time.toFixed(3);
  [...$('resTicks').children].forEach((c, i) => c.classList.toggle('on', i === ix));

  $('lapTime').textContent = S.sol.lap_time.toFixed(3);
  const rec = S.track.f1_record;
  if (rec) {
    $('f1Record').textContent = rec.toFixed(3) + ' s';
    const d = S.sol.lap_time - rec;
    const de = $('deltaRecord');
    de.textContent = (d >= 0 ? '+' : '') + d.toFixed(3) + ' s';
    de.className   = 'rec-val ' + (d >= 0 ? 'slower' : 'faster');
    $('recordFill').style.width =
      Math.max(4, Math.min(100, rec/S.sol.lap_time*100)).toFixed(1) + '%';
  } else {
    $('f1Record').textContent   = '—';
    $('deltaRecord').textContent = 'N/A';
    $('deltaRecord').className  = 'rec-val';
    $('recordFill').style.width = '0%';
  }
  drawTrack(); drawTrace(); updateFrame(true);
}

// ── ZOOM / PAN ─────────────────────────────────────────────────────
function resetZoom() {
  S.zoom.scale = 1; S.zoom.ox = 0; S.zoom.oy = 0;
  showHint();
  drawTrack();
}

function showHint() {
  const h = $('zoomHint');
  h.classList.remove('hidden');
  clearTimeout(S._hintTimer);
  S._hintTimer = setTimeout(() => h.classList.add('hidden'), 2800);
}

function wireZoom() {
  const frame = $('trackFrame');

  // Scroll to zoom
  frame.addEventListener('wheel', e => {
    e.preventDefault();
    const cv  = $('trackCanvas');
    const rect = cv.getBoundingClientRect();
    const mx   = (e.clientX - rect.left) * devicePixelRatio;
    const my   = (e.clientY - rect.top)  * devicePixelRatio;
    const factor = e.deltaY < 0 ? 1.12 : 1/1.12;
    const newScale = Math.max(1, Math.min(20, S.zoom.scale * factor));
    // Zoom toward mouse position
    S.zoom.ox = mx - (mx - S.zoom.ox) * (newScale / S.zoom.scale);
    S.zoom.oy = my - (my - S.zoom.oy) * (newScale / S.zoom.scale);
    S.zoom.scale = newScale;
    if (S.zoom.scale <= 1) { S.zoom.ox = 0; S.zoom.oy = 0; }
    drawTrack();
  }, { passive: false });

  // Drag to pan
  frame.addEventListener('mousedown', e => {
    if (S.zoom.scale <= 1) return;
    S.zoom.dragging = true;
    S.zoom.dragX = e.clientX;
    S.zoom.dragY = e.clientY;
    frame.style.cursor = 'grabbing';
  });
  window.addEventListener('mousemove', e => {
    if (!S.zoom.dragging) return;
    const dx = (e.clientX - S.zoom.dragX) * devicePixelRatio;
    const dy = (e.clientY - S.zoom.dragY) * devicePixelRatio;
    S.zoom.ox += dx; S.zoom.oy += dy;
    S.zoom.dragX = e.clientX; S.zoom.dragY = e.clientY;
    drawTrack();
  });
  window.addEventListener('mouseup', () => {
    S.zoom.dragging = false;
    $('trackFrame').style.cursor = 'crosshair';
  });

  // Double-click to reset zoom
  frame.addEventListener('dblclick', resetZoom);

  // Touch pinch-to-zoom
  let lastDist = null;
  frame.addEventListener('touchstart', e => {
    if (e.touches.length === 2) {
      lastDist = Math.hypot(
        e.touches[0].clientX - e.touches[1].clientX,
        e.touches[0].clientY - e.touches[1].clientY);
    }
  }, { passive: true });
  frame.addEventListener('touchmove', e => {
    if (e.touches.length === 2 && lastDist) {
      const d = Math.hypot(
        e.touches[0].clientX - e.touches[1].clientX,
        e.touches[0].clientY - e.touches[1].clientY);
      const factor = d / lastDist;
      S.zoom.scale = Math.max(1, Math.min(20, S.zoom.scale * factor));
      if (S.zoom.scale <= 1) { S.zoom.ox = 0; S.zoom.oy = 0; }
      lastDist = d;
      drawTrack();
    }
  }, { passive: true });
  frame.addEventListener('touchend', () => { lastDist = null; });
}

// ── TRACK RENDERING ────────────────────────────────────────────────
function baseTransform(cv, xs, ys, pad) {
  const w = cv.width, h = cv.height;
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const sx = (w - 2*pad) / (xmax - xmin);
  const sy = (h - 2*pad) / (ymax - ymin);
  const sc = Math.min(sx, sy);
  const ox = (w - (xmax - xmin)*sc) / 2;
  const oy = (h - (ymax - ymin)*sc) / 2;
  // store data extents for later
  S.zoom.xmin = xmin; S.zoom.xmax = xmax;
  S.zoom.ymin = ymin; S.zoom.ymax = ymax;
  S.zoom.baseScale = sc; S.zoom.baseOx = ox; S.zoom.baseOy = oy;
  S.zoom.h = h;
  return {
    X: x => (x - xmin)*sc + ox,
    Y: y => h - ((y - ymin)*sc + oy),
    sc
  };
}

function applyZoom(ctx) {
  ctx.translate(S.zoom.ox, S.zoom.oy);
  ctx.scale(S.zoom.scale, S.zoom.scale);
}

function drawTrack() {
  if (!S.sol) return;
  const cv  = $('trackCanvas');
  const rect = cv.getBoundingClientRect();
  cv.width  = rect.width  * devicePixelRatio;
  cv.height = rect.height * devicePixelRatio;
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, cv.width, cv.height);

  const B   = S.track.bounds, sol = S.sol;
  const allX = [...B.xr, ...B.xl], allY = [...B.yr, ...B.yl];
  const pad  = 36 * devicePixelRatio;
  const T    = baseTransform(cv, allX, allY, pad);

  ctx.save();
  applyZoom(ctx);

  // Tarmac fill
  ctx.beginPath();
  ctx.moveTo(T.X(B.xr[0]), T.Y(B.yr[0]));
  for (let i = 1; i < B.xr.length; i++) ctx.lineTo(T.X(B.xr[i]), T.Y(B.yr[i]));
  for (let i = B.xl.length-1; i >= 0; i--) ctx.lineTo(T.X(B.xl[i]), T.Y(B.yl[i]));
  ctx.closePath();
  ctx.fillStyle = 'rgba(255,255,255,0.04)';
  ctx.fill();

  // Track bounds
  const drawPoly = (xs, ys, col, lw) => {
    ctx.beginPath();
    ctx.moveTo(T.X(xs[0]), T.Y(ys[0]));
    for (let i = 1; i < xs.length; i++) ctx.lineTo(T.X(xs[i]), T.Y(ys[i]));
    ctx.strokeStyle = col;
    ctx.lineWidth   = lw * devicePixelRatio / S.zoom.scale;
    ctx.stroke();
  };
  drawPoly(B.xr, B.yr, 'rgba(255,255,255,0.30)', 1.2);
  drawPoly(B.xl, B.yl, 'rgba(255,255,255,0.30)', 1.2);

  // Speed-coloured racing line
  const sp = sol.speed;
  const smin = Math.min(...sp), smax = Math.max(...sp);
  ctx.lineWidth = (3.2 * devicePixelRatio) / S.zoom.scale;
  ctx.lineCap   = 'round';
  for (let i = 0; i < sol.x.length - 1; i++) {
    ctx.beginPath();
    ctx.moveTo(T.X(sol.x[i]),   T.Y(sol.y[i]));
    ctx.lineTo(T.X(sol.x[i+1]), T.Y(sol.y[i+1]));
    ctx.strokeStyle = speedColor((sp[i] - smin) / (smax - smin || 1));
    ctx.stroke();
  }

  ctx.restore();
  S._T = T; // stash for car glyph
}

// ── CAR GLYPH ──────────────────────────────────────────────────────
function drawCar() {
  if (!S.sol || !S._T) return;
  const cv  = $('trackCanvas');
  const ctx = cv.getContext('2d');
  drawTrack();          // repaint base then overlay

  const T = S._T, sol = S.sol, i = S.frame;

  // Apply same zoom transform for car position
  ctx.save();
  applyZoom(ctx);

  const px  = T.X(sol.x[i]);
  const py  = T.Y(sol.y[i]);
  const ang = -sol.heading[i];
  const r   = 10 * devicePixelRatio;

  ctx.translate(px, py);
  ctx.rotate(ang);

  // White circle marker
  ctx.beginPath();
  ctx.arc(0, 0, r * 1.1, 0, 7);
  ctx.fillStyle   = 'rgba(255,255,255,0.15)';
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth   = 2 * devicePixelRatio;
  ctx.fill();
  ctx.stroke();

  // Red arrow pointing forward
  ctx.beginPath();
  ctx.moveTo(r * 1.5, 0);
  ctx.lineTo(-r * 0.8, r * 0.9);
  ctx.lineTo(-r * 0.2, 0);
  ctx.lineTo(-r * 0.8, -r * 0.9);
  ctx.closePath();
  ctx.fillStyle = '#e8001e';
  ctx.fill();

  ctx.restore();
}

// ── G-G FRENET ─────────────────────────────────────────────────────
function drawGG() {
  const cv  = $('ggCanvas');
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height, cx = W/2, cy = H/2;
  ctx.clearRect(0, 0, W, H);

  const G    = 9.81, GRID = 6;
  const R    = Math.min(W, H)/2 - 22;
  const px   = v => v / (GRID * G) * R;

  // Grid rings
  for (let g = 2; g <= GRID; g += 2) {
    ctx.beginPath(); ctx.arc(cx, cy, px(g*G), 0, 7);
    ctx.strokeStyle = g === GRID ? 'rgba(255,255,255,.25)' : 'rgba(255,255,255,.10)';
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = 'rgba(160,160,180,.55)';
    ctx.font = `10px "Major Mono Display",monospace`;
    ctx.fillText(g + 'g', cx + px(g*G) - 16, cy - 4);
  }
  // Axes
  ctx.strokeStyle = 'rgba(255,255,255,.18)';
  ctx.lineWidth   = 1;
  ctx.beginPath();
  ctx.moveTo(cx - R, cy); ctx.lineTo(cx + R, cy);
  ctx.moveTo(cx, cy - R); ctx.lineTo(cx, cy + R);
  ctx.stroke();

  // Axis labels
  ctx.fillStyle = 'rgba(160,160,180,.55)';
  ctx.font = '9px "Major Mono Display",monospace';
  ctx.fillText('LAT', cx + R + 3, cy + 4);
  ctx.fillText('M', cx + 3, cy - R - 4);

  const sol = S.sol, i = S.frame;
  const v   = Math.max(sol.speed_ms[i], 5);

  const GF = 7978.162499470494, DFF = 2.1259279125731587,
        POW = 745000, DR = 0.745, M = 750,
        GB  = 11254.767298703786, DBB = 2.2808897580274747;
  const front = (Math.min(GF + DFF*v*v, POW/Math.max(v, 1e-3))) / M - DR*v*v/M;
  const back  = (GB + DBB*v*v + DR*v*v) / M;
  const side  = 1.2 * (GB + DBB*v*v) / M;

  // Friction ellipse
  ctx.beginPath();
  for (let a = 0; a <= 360; a += 3) {
    const rad = a * Math.PI / 180;
    const lon = Math.cos(rad) >= 0 ? front : back;
    const X   = cx + px(Math.sin(rad) * side);
    const Y   = cy + px(-Math.cos(rad) * lon);
    a === 0 ? ctx.moveTo(X, Y) : ctx.lineTo(X, Y);
  }
  ctx.closePath();
  ctx.strokeStyle = 'rgba(232,0,30,.85)';
  ctx.lineWidth   = 2;
  ctx.stroke();
  ctx.fillStyle   = 'rgba(232,0,30,.06)';
  ctx.fill();

  // Car arrow (always up)
  ctx.save(); ctx.translate(cx, cy);
  ctx.beginPath();
  ctx.moveTo(0, -12); ctx.lineTo(7, 7); ctx.lineTo(0, 2); ctx.lineTo(-7, 7);
  ctx.closePath();
  ctx.fillStyle = 'rgba(255,255,255,.5)';
  ctx.fill();
  ctx.restore();

  // Current acceleration vector + dot
  const aL = sol.a_lon[i], aT = sol.a_lat[i];
  const dx  = cx + px(aT), dy = cy + px(-aL);
  ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(dx, dy);
  ctx.strokeStyle = 'rgba(21,101,255,.6)'; ctx.lineWidth = 1.5; ctx.stroke();
  ctx.beginPath(); ctx.arc(dx, dy, 7, 0, 7);
  ctx.fillStyle = '#1565ff';
  ctx.fill();
  ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
}

// ── FULL-WIDTH TRACE ───────────────────────────────────────────────
function drawTrace() {
  if (!S.sol) return;
  const cv   = $('traceCanvas');
  const wrap = cv.parentElement;
  cv.width   = wrap.clientWidth  * devicePixelRatio;
  cv.height  = wrap.clientHeight * devicePixelRatio;
  const ctx  = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  const padL = 52 * devicePixelRatio, padR = 20 * devicePixelRatio;
  const padT = 18 * devicePixelRatio, padB = 28 * devicePixelRatio;
  ctx.clearRect(0, 0, W, H);

  const sol  = S.sol;
  const n    = sol.speed.length;
  const Xp   = ii => padL + (ii / (n-1)) * (W - padL - padR);
  const mode = S.trace;

  // Draw background grid
  ctx.strokeStyle = 'rgba(255,255,255,.06)';
  ctx.lineWidth   = 1;
  for (let g = 0; g <= 4; g++) {
    const yy = padT + g/4 * (H - padT - padB);
    ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(W - padR, yy); ctx.stroke();
  }

  function drawSeries(data, col, yMin, yMax, dashed) {
    const Yp = v => H - padB - (v - yMin) / (yMax - yMin || 1) * (H - padT - padB);
    // Fill
    ctx.beginPath(); ctx.moveTo(Xp(0), Yp(data[0]));
    for (let i = 1; i < n; i++) ctx.lineTo(Xp(i), Yp(data[i]));
    ctx.lineTo(Xp(n-1), H - padB); ctx.lineTo(Xp(0), H - padB);
    ctx.closePath();
    const grd = ctx.createLinearGradient(0, padT, 0, H - padB);
    grd.addColorStop(0, col + '33'); grd.addColorStop(1, col + '05');
    ctx.fillStyle = grd; ctx.fill();
    // Line
    if (dashed) ctx.setLineDash([6 * devicePixelRatio, 4 * devicePixelRatio]);
    ctx.beginPath(); ctx.moveTo(Xp(0), Yp(data[0]));
    for (let i = 1; i < n; i++) ctx.lineTo(Xp(i), Yp(data[i]));
    ctx.strokeStyle = col; ctx.lineWidth = 1.8; ctx.stroke();
    ctx.setLineDash([]);
    return Yp;
  }

  let Yp_main;
  if (mode === 'speed') {
    const data = sol.speed;
    const mx = Math.max(...data), mn = 0;
    Yp_main = drawSeries(data, '#ffffff', mn, mx * 1.06);
    // Y axis label
    ctx.fillStyle = 'rgba(200,200,220,.5)';
    ctx.font      = `${9 * devicePixelRatio}px "Major Mono Display",monospace`;
    ctx.fillText('300', padL - 4 * devicePixelRatio, padT + 4 * devicePixelRatio);
    ctx.fillText('0',   padL - 4 * devicePixelRatio, H - padB);
  } else if (mode === 'accel') {
    const data = sol.a_mag;
    const mx = Math.max(...data);
    Yp_main = drawSeries(data, '#e8001e', 0, mx * 1.1);
    ctx.fillStyle = 'rgba(200,200,220,.5)';
    ctx.font      = `${9 * devicePixelRatio}px "Major Mono Display",monospace`;
    ctx.fillText(mx.toFixed(0), padL - 4 * devicePixelRatio, padT + 4 * devicePixelRatio);
  } else {
    // BOTH: speed (white, scale km/h) + lon (red, g) + lat (blue, g, dashed)
    const spd  = sol.speed;
    const smax = Math.max(...spd);
    drawSeries(spd, '#aaaaaa', 0, smax * 1.1);

    // Accel on a centred scale [-max, +max]
    const aLon = sol.a_lon, aLat = sol.a_lat;
    const amax = Math.max(Math.max(...aLon.map(Math.abs)),
                          Math.max(...aLat.map(Math.abs))) * 1.1;
    const Ycent = v => H - padB - (v + amax) / (2*amax) * (H - padT - padB);
    // Zero line
    ctx.strokeStyle = 'rgba(255,255,255,.15)'; ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, Ycent(0)); ctx.lineTo(W - padR, Ycent(0)); ctx.stroke();
    // lon
    ctx.beginPath(); ctx.moveTo(Xp(0), Ycent(aLon[0]));
    for (let i = 1; i < n; i++) ctx.lineTo(Xp(i), Ycent(aLon[i]));
    ctx.strokeStyle = '#e8001e'; ctx.lineWidth = 1.5; ctx.stroke();
    // lat
    ctx.setLineDash([6 * devicePixelRatio, 4 * devicePixelRatio]);
    ctx.beginPath(); ctx.moveTo(Xp(0), Ycent(aLat[0]));
    for (let i = 1; i < n; i++) ctx.lineTo(Xp(i), Ycent(aLat[i]));
    ctx.strokeStyle = '#4488ff'; ctx.lineWidth = 1.5; ctx.stroke();
    ctx.setLineDash([]);
    Yp_main = null; // No single series to track playhead on cleanly
  }

  // Corner markers along arc length (approximate)
  if (sol.s && sol.s.length) {
    const totalS = sol.s[n-1];
    const corners = [];
    for (let i = 1; i < n-1; i++) {
      if (Math.abs(sol.a_lat[i]) > 5 &&
          Math.abs(sol.a_lat[i]) > Math.abs(sol.a_lat[i-1]) &&
          Math.abs(sol.a_lat[i]) > Math.abs(sol.a_lat[i+1])) corners.push(i);
    }
    ctx.fillStyle = 'rgba(200,200,200,.18)';
    ctx.font = `${8 * devicePixelRatio}px "Major Mono Display",monospace`;
    let cnt = 1;
    corners.forEach(ci => {
      ctx.fillText('T' + cnt++, Xp(ci) + 2 * devicePixelRatio, padT - 2 * devicePixelRatio);
    });
  }

  // Playhead
  const i = S.frame;
  ctx.strokeStyle = 'rgba(255,255,255,.55)';
  ctx.lineWidth   = 1.5;
  ctx.beginPath();
  ctx.moveTo(Xp(i), padT);
  ctx.lineTo(Xp(i), H - padB);
  ctx.stroke();
  if (Yp_main) {
    ctx.beginPath();
    ctx.arc(Xp(i), Yp_main(
      mode === 'speed' ? sol.speed[i] : sol.a_mag[i]
    ), 5 * devicePixelRatio, 0, 7);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
  }

  // X axis distance ticks
  ctx.fillStyle = 'rgba(160,160,180,.5)';
  ctx.font = `${8 * devicePixelRatio}px "Major Mono Display",monospace`;
  if (sol.s && sol.s.length) {
    const totS = sol.s[n-1];
    const step = totS > 5000 ? 1000 : totS > 2000 ? 500 : 200;
    for (let s = 0; s <= totS; s += step) {
      // find nearest frame index
      let fi = 0, minD = Infinity;
      for (let k = 0; k < n; k++) {
        const d = Math.abs(sol.s[k] - s); if (d < minD) { minD = d; fi = k; }
      }
      ctx.fillText(s, Xp(fi) - 10 * devicePixelRatio, H - padB + 14 * devicePixelRatio);
    }
  }
}

// ── PER-FRAME HUD ──────────────────────────────────────────────────
function updateFrame(noredraw) {
  const sol = S.sol, i = S.frame, G = 9.81;
  $('curSpeed').textContent = sol.speed[i].toFixed(0);
  $('curLonG').textContent  = (sol.a_lon[i]/G).toFixed(2);
  $('curLatG').textContent  = (sol.a_lat[i]/G).toFixed(2);
  $('curAmag').textContent  = sol.a_mag[i].toFixed(1);
  $('frameIx').textContent  = i;
  $('progress').value       = i;
  drawCar(); drawGG(); drawTrace();
}

// ── PLAYBACK LOOP ──────────────────────────────────────────────────
function loop(ts) {
  if (S.playing && S.sol) {
    if (!S.lastT) S.lastT = ts;
    const dtMs = ts - S.lastT; S.lastT = ts;
    const sol  = S.sol, n = sol.x.length;
    const segLen = sol.s[n-1] / (n-1);
    const v      = Math.max(sol.speed_ms[S.frame], 5);
    S.accT += (dtMs / 1000) * v;
    while (S.accT >= segLen) {
      S.accT -= segLen; S.frame++;
      if (S.frame >= n-1) { S.frame = 0; S.accT = 0; }
    }
    updateFrame();
  } else { S.lastT = ts; }
  requestAnimationFrame(loop);
}
function play() {
  S.playing = true; S.lastT = 0;
  $('btnPlay').textContent = '⏸';
  $('btnPlay').classList.add('is-playing');
}
function stop() {
  S.playing = false;
  const b = $('btnPlay');
  if (b) { b.textContent = '▶'; b.classList.remove('is-playing'); }
}
function toggle() { S.playing ? stop() : play(); }

// ── WIRE CONTROLS ──────────────────────────────────────────────────
function wireControls() {
  wireZoom();

  $('btnPlay').onclick    = toggle;
  $('btnFirst').onclick   = () => { S.frame = 0; updateFrame(); };
  $('btnLast').onclick    = () => { S.frame = S.sol.x.length-1; updateFrame(); };
  $('btnPrev').onclick    = () => { stop(); S.frame = Math.max(0, S.frame-1); updateFrame(); };
  $('btnNext').onclick    = () => { stop(); S.frame = Math.min(S.sol.x.length-1, S.frame+1); updateFrame(); };
  $('progress').oninput   = e => { stop(); S.frame = +e.target.value; updateFrame(); };
  $('resSlider').oninput  = e => { stop(); selectRes(+e.target.value); };
  $('btnZoomReset').onclick = resetZoom;

  $('tgSpeed').onclick = () => {
    S.trace = 'speed';
    [$('tgSpeed'), $('tgAccel'), $('tgBoth')].forEach(b => b.classList.remove('active'));
    $('tgSpeed').classList.add('active');
    drawTrace();
  };
  $('tgAccel').onclick = () => {
    S.trace = 'accel';
    [$('tgSpeed'), $('tgAccel'), $('tgBoth')].forEach(b => b.classList.remove('active'));
    $('tgAccel').classList.add('active');
    drawTrace();
  };
  $('tgBoth').onclick = () => {
    S.trace = 'both';
    [$('tgSpeed'), $('tgAccel'), $('tgBoth')].forEach(b => b.classList.remove('active'));
    $('tgBoth').classList.add('active');
    drawTrace();
  };

  document.addEventListener('keydown', e => {
    if (e.code === 'Space')       { e.preventDefault(); toggle(); }
    if (e.code === 'ArrowRight')  { stop(); S.frame = Math.min(S.sol.x.length-1, S.frame+1); updateFrame(); }
    if (e.code === 'ArrowLeft')   { stop(); S.frame = Math.max(0, S.frame-1); updateFrame(); }
    if (e.code === 'Escape')      resetZoom();
  });
}

boot();
