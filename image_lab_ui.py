"""
image_lab_ui.py — Full web UI returned as inline HTML/CSS/JS from GET /.
"""

UI_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Arthur Image Lab</title>
<style>
  :root {
    --bg:     #0f1117;
    --panel:  #1a1d27;
    --border: #2d3147;
    --accent: #6c8ef7;
    --accent2: #a78bfa;
    --text:   #e2e6f0;
    --muted:  #7a8099;
    --ok:     #34d399;
    --warn:   #fbbf24;
    --err:    #f87171;
    --radius: 10px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; font-size: 14px; }

  /* ---- Layout ---- */
  .shell { display: grid; grid-template-columns: 300px 1fr; height: 100vh; overflow: hidden; }
  .sidebar { background: var(--panel); border-right: 1px solid var(--border);
             display: flex; flex-direction: column; overflow: hidden; }
  .main    { display: flex; flex-direction: column; overflow: hidden; }

  /* ---- Header ---- */
  .header { padding: 14px 20px; border-bottom: 1px solid var(--border);
            display: flex; align-items: center; gap: 12px; }
  .header h1 { font-size: 16px; font-weight: 700; letter-spacing: .5px; }
  .header h1 span { color: var(--accent); }
  .vram-bar-wrap { margin-left: auto; display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--muted); }
  .vram-bar { width: 100px; height: 8px; background: var(--border); border-radius: 4px; overflow: hidden; }
  .vram-fill { height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent2)); transition: width .4s; }

  /* ---- Engine tabs ---- */
  .engine-tabs { display: flex; border-bottom: 1px solid var(--border); }
  .engine-tab  { flex: 1; padding: 10px 4px; text-align: center; cursor: pointer;
                 font-size: 12px; font-weight: 600; color: var(--muted);
                 border-bottom: 2px solid transparent; transition: all .2s; }
  .engine-tab:hover  { color: var(--text); }
  .engine-tab.active { color: var(--accent); border-bottom-color: var(--accent); }
  .engine-tab .badge { font-size: 9px; display: block; margin-top: 2px;
                       color: var(--muted); font-weight: 400; }

  /* ---- Params ---- */
  .params-area { flex: 1; overflow-y: auto; padding: 16px; min-height: 0; }
  .param-group { margin-bottom: 14px; }
  .param-group label { display: block; font-size: 12px; color: var(--muted);
                        margin-bottom: 5px; font-weight: 500; }
  textarea, input[type=text], input[type=number], select {
    width: 100%; background: var(--bg); border: 1px solid var(--border);
    color: var(--text); border-radius: 6px; padding: 7px 10px; font-size: 13px;
    outline: none; font-family: inherit; transition: border .15s;
  }
  textarea:focus, input:focus, select:focus { border-color: var(--accent); }
  textarea { resize: vertical; min-height: 70px; }
  .range-row { display: flex; align-items: center; gap: 8px; }
  .range-row input[type=range] { flex: 1; accent-color: var(--accent); }
  .range-val { min-width: 36px; text-align: right; color: var(--accent); font-weight: 600; font-size: 13px; }
  .file-drop { border: 1.5px dashed var(--border); border-radius: 6px; padding: 14px;
               text-align: center; color: var(--muted); font-size: 12px; cursor: pointer; transition: border .15s; }
  .file-drop:hover { border-color: var(--accent); }
  .file-drop.has-file { border-color: var(--ok); color: var(--ok); }

  /* ---- Generate button ---- */
  .btn-generate { margin: 0 16px 16px; padding: 12px; background: var(--accent);
                  color: #fff; border: none; border-radius: var(--radius); font-size: 14px;
                  font-weight: 700; cursor: pointer; letter-spacing: .4px; transition: opacity .2s; }
  .btn-generate:hover   { opacity: .88; }
  .btn-generate:disabled { opacity: .4; cursor: not-allowed; }

  /* ---- Engine description ---- */
  .eng-desc { padding: 12px 16px; font-size: 11px; color: var(--muted);
              border-top: 1px solid var(--border); line-height: 1.5; }

  /* ---- Status bar ---- */
  .statusbar { padding: 6px 16px; font-size: 11px; color: var(--muted);
               border-bottom: 1px solid var(--border); display: flex; gap: 16px; align-items: center; }
  .status-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--muted); display: inline-block; margin-right: 4px; }
  .status-dot.ok   { background: var(--ok); }
  .status-dot.busy { background: var(--warn); animation: pulse 1s infinite; }
  .status-dot.err  { background: var(--err); }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }

  /* ---- Output pane ---- */
  .output-pane { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 16px; }
  .result-card { background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; }
  .result-card img  { width: 100%; display: block; border-radius: var(--radius) var(--radius) 0 0; }
  .result-card video { width: 100%; display: block; border-radius: var(--radius) var(--radius) 0 0; }
  .result-meta { padding: 10px 14px; font-size: 11px; color: var(--muted); display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  .result-meta strong { color: var(--text); }
  .btn-dl { margin-left: auto; padding: 4px 10px; background: var(--border); color: var(--text);
            border: none; border-radius: 5px; cursor: pointer; font-size: 11px; }
  .btn-dl:hover { background: var(--accent); color: #fff; }

  /* ---- Gallery pane ---- */
  .gallery-header { padding: 10px 20px; border-bottom: 1px solid var(--border);
                    display: flex; align-items: center; gap: 10px; }
  .gallery-header h2 { font-size: 13px; font-weight: 600; }
  .gallery-filter { margin-left: auto; }
  .gallery-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px,1fr));
                  gap: 10px; padding: 16px; }
  .gallery-thumb { position: relative; cursor: pointer; border-radius: 7px; overflow: hidden;
                   border: 1.5px solid var(--border); transition: border .15s; }
  .gallery-thumb:hover { border-color: var(--accent); }
  .gallery-thumb img, .gallery-thumb video { width: 100%; display: block; aspect-ratio: 1; object-fit: cover; }
  .gallery-thumb .g-label { position: absolute; bottom: 0; left: 0; right: 0;
                             background: rgba(0,0,0,.6); font-size: 9px; padding: 3px 5px;
                             color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .gallery-thumb .g-del { position: absolute; top: 4px; right: 4px; background: var(--err);
                           color: #fff; border: none; border-radius: 3px; font-size: 10px;
                           padding: 1px 5px; cursor: pointer; display: none; }
  .gallery-thumb:hover .g-del { display: block; }

  /* ---- API panel ---- */
  .api-panel { margin: 0 16px 14px; }
  .api-panel summary { font-size: 11px; color: var(--muted); cursor: pointer; user-select: none; }
  .api-panel pre { background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
                   padding: 10px; font-size: 10px; color: var(--ok); overflow-x: auto;
                   margin-top: 6px; white-space: pre-wrap; word-break: break-all; }
  .btn-copy { float: right; padding: 2px 7px; font-size: 10px; background: var(--border);
              color: var(--text); border: none; border-radius: 4px; cursor: pointer; margin-bottom: 4px; }

  /* ---- Quant reload warning ---- */
  .quant-warn { display: none; margin: 0 16px 10px; padding: 8px 12px;
                background: rgba(251,191,36,.12); border: 1px solid var(--warn);
                border-radius: 7px; font-size: 11px; color: var(--warn);
                gap: 6px; align-items: center; }

  /* ---- Spinner overlay ---- */
  .spinner-overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.55);
                      z-index: 999; align-items: center; justify-content: center; flex-direction: column; gap: 14px; }
  .spinner-overlay.show { display: flex; }
  .spinner { width: 46px; height: 46px; border: 4px solid var(--border);
             border-top-color: var(--accent); border-radius: 50%; animation: spin .8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .spinner-msg { color: var(--text); font-size: 14px; font-weight: 600; }

  /* ---- Tabs (Generate / Gallery) ---- */
  .view-tabs { display: flex; border-bottom: 1px solid var(--border); }
  .view-tab  { padding: 10px 20px; cursor: pointer; font-size: 13px; font-weight: 600;
               color: var(--muted); border-bottom: 2px solid transparent; }
  .view-tab.active { color: var(--accent); border-bottom-color: var(--accent); }

  .view-panel { display: none; flex: 1; flex-direction: column; overflow: hidden; }
  .view-panel.active { display: flex; }

  .empty-state { flex: 1; display: flex; align-items: center; justify-content: center;
                  color: var(--muted); font-size: 13px; flex-direction: column; gap: 8px; }
  .empty-state svg { opacity: .3; }

  /* ---- Log panel ---- */
  .log-panel { border-top: 1px solid var(--border); background: #10121a; flex-shrink: 0; }
  .log-header { display: flex; align-items: center; gap: 6px; padding: 6px 10px;
                cursor: pointer; user-select: none; font-size: 11px; color: var(--muted); }
  .log-header:hover { color: var(--text); }
  .log-header .log-badge { background: var(--err); color: #fff; border-radius: 8px;
                            padding: 1px 5px; font-size: 10px; display: none; }
  .log-header .log-badge.show { display: inline; }
  .log-header .btn-clr { margin-left: auto; font-size: 10px; background: none;
                          border: 1px solid var(--border); color: var(--muted);
                          border-radius: 4px; padding: 1px 6px; cursor: pointer; }
  .log-body { display: none; max-height: 220px; overflow-y: auto;
              padding: 4px 0; font-size: 11px; font-family: monospace; }
  .log-body.open { display: block; }
  .log-entry { padding: 2px 10px; border-bottom: 1px solid #1d2030; word-break: break-all;
               white-space: pre-wrap; }
  .log-entry.info  { color: #7cb8ff; }
  .log-entry.warn  { color: var(--warn); }
  .log-entry.error { color: var(--err); background: rgba(248,113,113,.07); }
  .log-entry.debug { color: #6b7280; }
  .log-entry .ts   { color: #4b5563; margin-right: 4px; }
  .log-entry .lvl  { font-weight: 700; margin-right: 5px; }
  .log-entry .detail { display: block; margin-top: 2px; color: #9ca3af;
                        font-size: 10px; padding-left: 8px; }
</style>
</head>
<body>

<!-- Spinner overlay -->
<div class="spinner-overlay" id="spinner">
  <div class="spinner"></div>
  <div class="spinner-msg" id="spinnerMsg">Generating…</div>
</div>

<div class="shell">
  <!-- ====== SIDEBAR ====== -->
  <aside class="sidebar">
    <div class="engine-tabs" id="engineTabs"></div>
    <div class="params-area" id="paramsArea"></div>

    <!-- API snippet -->
    <details class="api-panel" id="apiPanel">
      <summary>▸ curl API snippet</summary>
      <button class="btn-copy" onclick="copyCurl()">Copy</button>
      <pre id="curlSnippet"></pre>
    </details>

    <div class="quant-warn" id="quantWarnBar">
      ⚠️ Quantization change — model will reload (~60 s)
    </div>
    <button class="btn-generate" id="btnGenerate" onclick="doGenerate()">⚡ Generate</button>
    <div class="eng-desc" id="engDesc"></div>

    <!-- Live log panel -->
    <div class="log-panel" id="logPanel">
      <div class="log-header" onclick="toggleLog()">
        <span>▸ Log</span>
        <span class="log-badge" id="logBadge">0</span>
        <button class="btn-clr" onclick="event.stopPropagation();clearLog()">Clear</button>
      </div>
      <div class="log-body" id="logBody"></div>
    </div>
  </aside>

  <!-- ====== MAIN ====== -->
  <main class="main">
    <!-- Status bar -->
    <div class="statusbar">
      <span><span class="status-dot" id="statusDot"></span><span id="statusText">Connecting…</span></span>
      <span id="activeEngineLabel" style="color:var(--accent)"></span>
      <span id="vramText" style="margin-left:auto"></span>
      <div class="vram-bar-wrap">
        <div class="vram-bar"><div class="vram-fill" id="vramFill" style="width:0%"></div></div>
      </div>
    </div>

    <!-- View tabs -->
    <div class="view-tabs">
      <div class="view-tab active" onclick="switchView('generate')">Generate</div>
      <div class="view-tab"        onclick="switchView('gallery')">Gallery</div>
    </div>

    <!-- Generate view -->
    <div class="view-panel active" id="viewGenerate">
      <div class="output-pane" id="outputPane">
        <div class="empty-state">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/>
            <polyline points="21 15 16 10 5 21"/>
          </svg>
          <span>Your generated images and videos will appear here</span>
        </div>
      </div>
    </div>

    <!-- Gallery view -->
    <div class="view-panel" id="viewGallery">
      <div class="gallery-header">
        <h2>Gallery</h2>
        <select class="gallery-filter" id="galleryFilter" onchange="loadGallery()">
          <option value="">All engines</option>
          <option value="flux2">FLUX.2 [dev]</option>
          <option value="sd35">SD 3.5 Large</option>
          <option value="wan">Wan2.2</option>
        </select>
      </div>
      <div class="gallery-grid" id="galleryGrid"></div>
    </div>
  </main>
</div>

<script>
// ============================================================
// State
// ============================================================
const ENGINES_META = {};   // filled by /status
let currentEngine  = 'flux2';
let formValues     = {};   // param key → current value
let uploadedFile   = null; // reference image bytes
let lastStatus     = null; // last /status response

// ============================================================
// Boot
// ============================================================
(async function boot() {
  uiLog('info', 'Image Lab UI booting', {url: location.href, ua: navigator.userAgent.slice(0,60)});
  await refreshStatus();
  setInterval(refreshStatus, 4000);
  buildEngineTabs();
  selectEngine(currentEngine);
  loadGallery();
  uiLog('info', 'Boot complete');
})();

// ============================================================
// Status polling
// ============================================================
async function refreshStatus() {
  try {
    const s = await apiFetch('/status');
    lastStatus = s;
    // Update engine meta
    for (const e of s.engines) {
      ENGINES_META[e.key] = e;
    }
    // Status dot
    const dot  = document.getElementById('statusDot');
    const txt  = document.getElementById('statusText');
    if (s.generating || s.loading) {
      dot.className = 'status-dot busy';
      txt.textContent = s.generating ? 'Generating…' : 'Loading model…';
    } else {
      dot.className = 'status-dot ok';
      txt.textContent = 'Ready';
    }
    document.getElementById('activeEngineLabel').textContent =
      s.active_engine
        ? `Loaded: ${ENGINES_META[s.active_engine]?.label ?? s.active_engine}${
            s.active_quant ? ' · ' + s.active_quant : ''}`
        : '';
    updateQuantWarning();
    // VRAM
    const v = s.vram;
    if (v && v.available) {
      const pct = Math.round(v.reserved_gb / v.total_gb * 100);
      document.getElementById('vramFill').style.width = pct + '%';
      document.getElementById('vramText').textContent =
        `VRAM ${v.reserved_gb.toFixed(1)} / ${v.total_gb.toFixed(1)} GB`;
    }
    // Update tab availability dots
    for (const e of s.engines) {
      const el = document.getElementById('tab-' + e.key);
      if (el) {
        const badge = el.querySelector('.badge');
        badge.textContent = e.available ? '✓ available' : '✗ unavailable';
        badge.style.color  = e.available ? 'var(--ok)' : 'var(--err)';
      }
    }
  } catch(e) {
    document.getElementById('statusDot').className = 'status-dot err';
    document.getElementById('statusText').textContent = 'Server unreachable';
    uiLog('warn', 'Status poll failed', e.message);
  }
}

// ============================================================
// Engine tabs
// ============================================================
function buildEngineTabs() {
  const tabs = document.getElementById('engineTabs');
  const keys = ['flux2', 'flux2klein', 'sd35', 'wan'];
  const labels = { flux2: 'FLUX.2', flux2klein: 'FLUX.2 Klein', sd35: 'SD 3.5', wan: 'Wan2.2' };
  tabs.innerHTML = keys.map(k => `
    <div class="engine-tab" id="tab-${k}" onclick="selectEngine('${k}')">
      ${labels[k]}
      <span class="badge">…</span>
    </div>`).join('');
}

function selectEngine(key) {
  currentEngine = key;
  uploadedFile  = null;
  document.querySelectorAll('.engine-tab').forEach(t => t.classList.remove('active'));
  const tab = document.getElementById('tab-' + key);
  if (tab) tab.classList.add('active');
  renderParams(key);
  updateQuantWarning();
  updateCurl();
}

// ============================================================
// Parameter rendering
// ============================================================
function renderParams(key) {
  const meta = ENGINES_META[key];
  if (!meta) { return; }

  document.getElementById('engDesc').textContent = meta.description;
  const area = document.getElementById('paramsArea');
  area.innerHTML = '';
  formValues = {};

  for (const p of meta.params) {
    if (p.type === 'file') {
      area.appendChild(buildFileParam(p));
    } else {
      area.appendChild(buildParam(p));
    }
  }
}

function buildParam(p) {
  const wrap = document.createElement('div');
  wrap.className = 'param-group';
  const label = document.createElement('label');
  label.title = p.tooltip || '';
  label.textContent = p.label + (p.required ? ' *' : '');
  wrap.appendChild(label);

  let el;
  if (p.type === 'textarea') {
    el = document.createElement('textarea');
    el.placeholder = p.tooltip || '';
    el.value = formValues[p.name] ?? p.default ?? '';
  } else if (p.type === 'select') {
    el = document.createElement('select');
    for (const opt of (p.options || [])) {
      const o   = document.createElement('option');
      const val = (opt && typeof opt === 'object') ? opt.value : opt;
      const lbl = (opt && typeof opt === 'object') ? opt.label  : opt;
      o.value = val; o.textContent = lbl;
      if (val === (formValues[p.name] ?? p.default)) o.selected = true;
      el.appendChild(o);
    }
  } else if (p.type === 'int' || p.type === 'float') {
    // Use range + number input side by side when min/max defined
    if (p.min !== undefined && p.max !== undefined) {
      const row = document.createElement('div');
      row.className = 'range-row';
      const range = document.createElement('input');
      range.type = 'range';
      range.min = p.min; range.max = p.max; range.step = p.step ?? (p.type === 'float' ? 0.1 : 1);
      range.value = formValues[p.name] ?? p.default;
      const valSpan = document.createElement('span');
      valSpan.className = 'range-val';
      valSpan.textContent = range.value;
      range.oninput = () => { valSpan.textContent = range.value; formValues[p.name] = +range.value; updateCurl(); };
      row.appendChild(range); row.appendChild(valSpan);
      formValues[p.name] = +range.value;
      wrap.appendChild(row);
      return wrap;
    } else {
      el = document.createElement('input');
      el.type = 'number';
      el.step = p.step ?? (p.type === 'float' ? 0.1 : 1);
      el.value = formValues[p.name] ?? p.default ?? 0;
    }
  } else {
    el = document.createElement('input');
    el.type = 'text';
    el.value = formValues[p.name] ?? p.default ?? '';
  }

  el.oninput = el.onchange = () => {
    formValues[p.name] = el.value;
    updateCurl();
    if (p.name === 'quant') updateQuantWarning();
  };
  formValues[p.name] = el.value ?? el.options?.[el.selectedIndex]?.value ?? '';
  wrap.appendChild(el);
  return wrap;
}

function buildFileParam(p) {
  const wrap = document.createElement('div');
  wrap.className = 'param-group';
  const label = document.createElement('label');
  label.textContent = p.label;
  wrap.appendChild(label);
  const drop = document.createElement('div');
  drop.className = 'file-drop';
  drop.id = 'filedrop-' + p.name;
  drop.textContent = 'Click or drag to upload image';
  drop.onclick = () => { const inp = document.createElement('input'); inp.type='file'; inp.accept='image/*';
    inp.onchange = e => { uploadedFile = e.target.files[0]; drop.textContent = uploadedFile.name; drop.classList.add('has-file'); };
    inp.click(); };
  wrap.appendChild(drop);
  return wrap;
}

// ============================================================
// Quantization reload warning
// ============================================================
function updateQuantWarning() {
  const loadedEngine = lastStatus?.active_engine;
  const loadedQuant  = lastStatus?.active_quant ?? '';
  const selQuant     = formValues['quant'] ?? '';
  // Show warning only when the SAME engine is loaded but quant differs
  const engineMatch  = loadedEngine === currentEngine;
  const quantDiffers = engineMatch && selQuant && loadedQuant && selQuant !== loadedQuant;
  const warnEl = document.getElementById('quantWarnBar');
  if (warnEl) warnEl.style.display = quantDiffers ? 'flex' : 'none';
}

// ============================================================
// Live log panel
// ============================================================
let _logErrors = 0;
function uiLog(level, msg, detail) {
  const ts    = new Date().toTimeString().slice(0,8);
  const lvlMap = { info:'INFO', warn:'WARN', error:'ERROR', debug:'DBG' };
  const lvl   = lvlMap[level] || level.toUpperCase();

  // Console mirror
  const consoleFn = { info: console.info, warn: console.warn, error: console.error, debug: console.debug }[level] || console.log;
  consoleFn(`[${ts}] [${lvl}] ${msg}`, detail !== undefined ? detail : '');

  // DOM panel
  const body = document.getElementById('logBody');
  if (!body) return;
  const entry = document.createElement('div');
  entry.className = 'log-entry ' + level;
  entry.innerHTML =
    `<span class="ts">${ts}</span><span class="lvl">${lvl}</span>${escHtml(msg)}` +
    (detail !== undefined ? `<span class="detail">${escHtml(typeof detail === 'string' ? detail : JSON.stringify(detail, null, 2))}</span>` : '');
  body.appendChild(entry);
  body.scrollTop = body.scrollHeight;

  if (level === 'error') {
    _logErrors++;
    const badge = document.getElementById('logBadge');
    if (badge) { badge.textContent = _logErrors; badge.classList.add('show'); }
    // Auto-open on first error
    const lb = document.getElementById('logBody');
    if (lb && !lb.classList.contains('open')) lb.classList.add('open');
  }
}
function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function toggleLog() {
  document.getElementById('logBody')?.classList.toggle('open');
}
function clearLog() {
  const b = document.getElementById('logBody');
  if (b) b.innerHTML = '';
  _logErrors = 0;
  const badge = document.getElementById('logBadge');
  if (badge) { badge.textContent = '0'; badge.classList.remove('show'); }
}

// Extract a readable string from any FastAPI error shape
function extractErrorMsg(data, fallback) {
  if (!data) return fallback;
  if (typeof data.detail === 'string') return data.detail;
  if (Array.isArray(data.detail)) {
    // FastAPI 422 validation errors: [{loc, msg, type}]
    return data.detail.map(e => `${(e.loc||[]).join('.')}: ${e.msg}`).join(' | ');
  }
  if (data.detail) return JSON.stringify(data.detail);
  return fallback;
}

// ============================================================
// Generate
// ============================================================
async function doGenerate() {
  const prompt = (formValues['prompt'] ?? '').trim();
  if (!prompt) {
    alert('Please enter a prompt before generating.');
    return;
  }
  const btn = document.getElementById('btnGenerate');
  btn.disabled = true;
  const loadedQuant = lastStatus?.active_quant ?? '';
  const selQuant    = formValues['quant'] ?? '';
  const willReload  = lastStatus?.active_engine !== currentEngine ||
                      (selQuant && loadedQuant && selQuant !== loadedQuant);
  showSpinner(true, willReload ? 'Reloading model… (this may take ~60 s)' : 'Generating…');

  const fd = new FormData();
  const meta = ENGINES_META[currentEngine];
  const sentParams = {};
  for (const p of (meta?.params ?? [])) {
    if (p.type === 'file') continue;
    const v = formValues[p.name];
    if (v !== undefined && v !== '') { fd.append(p.name, v); sentParams[p.name] = v; }
  }
  if (uploadedFile) { fd.append('reference_image', uploadedFile); sentParams.reference_image = uploadedFile.name; }

  uiLog('info', `POST /generate/${currentEngine}`, sentParams);

  let resp, data;
  try {
    const t0 = Date.now();
    resp = await fetch('/generate/' + currentEngine, { method: 'POST', body: fd });
    const elapsed = Date.now() - t0;
    uiLog('debug', `Response ${resp.status} ${resp.statusText} in ${elapsed} ms`);
  } catch(fetchErr) {
    uiLog('error', 'Network error (fetch failed)', fetchErr.message);
    alert('Network error: ' + fetchErr.message);
    btn.disabled = false; showSpinner(false);
    return;
  }

  try {
    data = await resp.json();
    uiLog('debug', `Response body (${resp.status})`, data);
  } catch(jsonErr) {
    uiLog('error', `Non-JSON response (${resp.status} ${resp.statusText})`, jsonErr.message);
    alert(`Server returned non-JSON response (${resp.status} ${resp.statusText})`);
    btn.disabled = false; showSpinner(false);
    return;
  }

  if (!resp.ok) {
    const msg = extractErrorMsg(data, resp.statusText);
    uiLog('error', `Generation failed — HTTP ${resp.status}`, msg);
    uiLog('error', 'Full error payload', data);
    alert(`Generation failed (${resp.status}): ${msg}`);
    btn.disabled = false; showSpinner(false);
    return;
  }

  try {
    uiLog('info', `Success — ${(data.results||[]).length} result(s)`);
    renderResults(data.results);
    switchView('generate');
    loadGallery();
  } catch(renderErr) {
    uiLog('error', 'Render error', renderErr.message);
  } finally {
    btn.disabled = false;
    showSpinner(false);
  }
}

function renderResults(results) {
  const pane = document.getElementById('outputPane');
  // Remove empty state
  pane.querySelector('.empty-state')?.remove();

  for (const r of results) {
    const card = document.createElement('div');
    card.className = 'result-card';

    if (r.type === 'image') {
      const img = document.createElement('img');
      img.src = r.base64 ? `data:image/png;base64,${r.base64}` : r.url;
      img.alt = 'Generated image';
      card.appendChild(img);
    } else {
      const vid = document.createElement('video');
      vid.src = r.url; vid.controls = true; vid.loop = true;
      card.appendChild(vid);
    }

    const meta = document.createElement('div');
    meta.className = 'result-meta';
    const seed = r.params?.seed ?? '?';
    const steps = r.params?.num_inference_steps ?? r.params?.num_frames ?? '?';
    meta.innerHTML = `<strong>${ENGINES_META[currentEngine]?.label ?? currentEngine}</strong>
      &nbsp;seed <code>${seed}</code>&nbsp;|&nbsp;steps/frames <code>${steps}</code>
      &nbsp;|&nbsp;${r.width ? r.width+'×'+r.height : r.fps+'fps'}`;
    const dlBtn = document.createElement('button');
    dlBtn.className = 'btn-dl'; dlBtn.textContent = '⬇ Download';
    dlBtn.onclick = () => { const a = document.createElement('a'); a.href = r.url;
      a.download = r.filename ?? 'output'; a.click(); };
    meta.appendChild(dlBtn);
    card.appendChild(meta);

    pane.insertBefore(card, pane.firstChild);
  }
}

// ============================================================
// Gallery
// ============================================================
async function loadGallery() {
  const engine = document.getElementById('galleryFilter')?.value || '';
  try {
    const data = await apiFetch('/gallery?limit=60' + (engine ? '&engine='+engine : ''));
    renderGallery(data.entries);
  } catch(e) {}
}

function renderGallery(entries) {
  const grid = document.getElementById('galleryGrid');
  grid.innerHTML = '';
  if (!entries.length) {
    grid.innerHTML = '<div style="color:var(--muted);font-size:12px;padding:10px">No generations yet.</div>';
    return;
  }
  for (const e of entries) {
    const thumb = document.createElement('div');
    thumb.className = 'gallery-thumb';
    thumb.title = e.params?.prompt ?? '';
    if (e.type === 'image') {
      const img = document.createElement('img');
      img.src = e.url; img.loading = 'lazy';
      thumb.appendChild(img);
    } else {
      const vid = document.createElement('video');
      vid.src = e.url; vid.muted = true;
      vid.onmouseenter = () => vid.play();
      vid.onmouseleave = () => { vid.pause(); vid.currentTime=0; };
      thumb.appendChild(vid);
    }
    const lbl = document.createElement('div'); lbl.className = 'g-label';
    lbl.textContent = ENGINES_META[e.engine]?.label ?? e.engine;
    thumb.appendChild(lbl);
    const del = document.createElement('button'); del.className = 'g-del'; del.textContent = '✕';
    del.onclick = async ev => { ev.stopPropagation();
      if (!confirm('Delete this generation?')) return;
      await fetch('/gallery/'+e.id, {method:'DELETE'});
      loadGallery(); };
    thumb.appendChild(del);
    thumb.onclick = () => { window.open(e.url, '_blank'); };
    grid.appendChild(thumb);
  }
}

// ============================================================
// curl snippet
// ============================================================
function updateCurl() {
  const base = `http://${location.host}`;
  const fields = Object.entries(formValues)
    .map(([k,v]) => `  -F '${k}=${v}'`)
    .join(' \\\n');
  const snippet = `curl -X POST ${base}/generate/${currentEngine} \\\n${fields}`;
  const el = document.getElementById('curlSnippet');
  if (el) el.textContent = snippet;
}

function copyCurl() {
  const txt = document.getElementById('curlSnippet')?.textContent ?? '';
  navigator.clipboard.writeText(txt).catch(() => {});
}

// ============================================================
// Helpers
// ============================================================
async function apiFetch(path) {
  const t0 = Date.now();
  uiLog('debug', `GET ${path}`);
  const r = await fetch(path);
  uiLog('debug', `GET ${path} → ${r.status} in ${Date.now()-t0} ms`);
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    const msg = extractErrorMsg(body, r.statusText);
    uiLog('warn', `apiFetch ${path} failed (${r.status})`, msg);
    throw new Error(msg);
  }
  return r.json();
}

function showSpinner(show, msg='') {
  const el = document.getElementById('spinner');
  el.classList.toggle('show', show);
  if (msg) document.getElementById('spinnerMsg').textContent = msg;
}

function switchView(name) {
  document.querySelectorAll('.view-tab').forEach((t,i) => {
    t.classList.toggle('active', ['generate','gallery'][i] === name);
  });
  document.querySelectorAll('.view-panel').forEach(p => {
    p.classList.toggle('active', p.id === 'view' + name.charAt(0).toUpperCase() + name.slice(1));
  });
  if (name === 'gallery') loadGallery();
}
</script>
</body>
</html>
"""


def get_ui_html() -> str:
    return UI_HTML
