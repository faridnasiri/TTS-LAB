"""
image_lab_ui.py — Full web UI returned as inline HTML/CSS/JS from GET /.
"""

from image_lab_config import USE_COMFYUI

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
  /* ---- VRAM/system report strip (TTS-style) ---- */
  .reportbar { padding: 8px 16px 9px; border-bottom: 1px solid var(--border); background: var(--panel); }
  .reportbar .row { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 18px; }
  .reportbar .row + .row { margin-top: 8px; }
  .reportbar .spacer { margin-left: auto; }
  .reportbar .status-dot { vertical-align: 0; }
  .bar-item { min-width: 150px; }
  .bar-head { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; font-size: 10px; letter-spacing: .4px; color: var(--muted); font-weight: 600; text-transform: uppercase; margin-bottom: 3px; }
  .bar-head .bar-label { text-transform: none; letter-spacing: 0; font-weight: 500; font-variant-numeric: tabular-nums; }
  .bar-track { width: 190px; height: 7px; background: var(--border); border-radius: 4px; overflow: hidden; }
  .bar-track.wide { width: 250px; }
  .bar-fill { height: 100%; border-radius: 4px; transition: width .5s ease; }
  .bar-fill.ram  { background: linear-gradient(90deg, #5b8def, var(--accent2)); }
  .bar-fill.vram { background: linear-gradient(90deg, var(--accent), var(--accent2)); }
  .bar-fill.hot  { background: linear-gradient(90deg, #f59e0b, var(--err)); }
  #vramDetail { margin-top: 3px; font-size: 10px; color: var(--muted); max-width: 420px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .gpu-badge { font-size: 10px; padding: 3px 10px; border-radius: 20px; border: 1px solid var(--border); color: var(--muted); white-space: nowrap; max-width: 260px; overflow: hidden; text-overflow: ellipsis; }
  .gpu-badge.ok  { color: var(--ok); border-color: rgba(52,211,153,.35); background: rgba(52,211,153,.08); }
  .gpu-badge.off { color: var(--err); border-color: rgba(248,113,113,.35); background: rgba(248,113,113,.08); }
  .btn-action { padding: 4px 12px; font-size: 11px; font-weight: 600; color: var(--text);
                background: var(--bg); border: 1px solid var(--border); border-radius: 6px; cursor: pointer; transition: all .15s; white-space: nowrap; }
  .btn-action:hover { border-color: var(--accent); color: var(--accent); }
  .btn-action.danger:hover { border-color: var(--err); color: var(--err); }
  .btn-action:disabled { opacity: .4; cursor: not-allowed; }
  .loaded-wrap { display: flex; align-items: center; gap: 6px; min-width: 0; }
  .loaded-label { font-size: 11px; color: var(--muted); white-space: nowrap; }
  #loadedChips { display: flex; flex-wrap: wrap; gap: 5px; }
  .chip { display: inline-flex; align-items: center; gap: 5px; background: rgba(108,142,247,.12);
          border: 1px solid rgba(108,142,247,.35); color: var(--text); border-radius: 20px;
          padding: 2px 6px 2px 10px; font-size: 11px; }
  .chip-idle { color: var(--muted); border-color: var(--border); background: transparent; }
  .chip-x { background: none; border: none; color: var(--muted); cursor: pointer; font-size: 10px; padding: 0 3px; line-height: 1; border-radius: 50%; }
  .chip-x:hover { color: var(--err); }

  /* ---- Sidebar engine actions (preload / unload) ---- */
  .engine-actions { display: flex; gap: 8px; margin: 0 16px 10px; }
  .btn-load { flex: 1; padding: 8px; font-size: 12px; font-weight: 700; color: #fff;
              background: linear-gradient(90deg, var(--accent), var(--accent2)); border: none;
              border-radius: 8px; cursor: pointer; transition: opacity .2s; }
  .btn-load:hover { opacity: .88; }
  .btn-unload { flex: 1; padding: 8px; font-size: 12px; font-weight: 600; color: var(--text);
                background: var(--bg); border: 1px solid var(--border); border-radius: 8px; cursor: pointer; transition: all .15s; }
  .btn-unload:hover { border-color: var(--err); color: var(--err); }
  .engine-actions button:disabled { opacity: .4; cursor: not-allowed; }

  /* ---- Toast ---- */
  #toast { position: fixed; right: 18px; bottom: 18px; z-index: 3000; max-width: 420px;
           background: var(--panel); border: 1px solid var(--border); border-left: 3px solid var(--ok);
           color: var(--text); font-size: 12px; line-height: 1.5; padding: 9px 13px; border-radius: 8px;
           box-shadow: 0 8px 24px rgba(0,0,0,.45); opacity: 0; transform: translateY(8px);
           transition: opacity .25s, transform .25s; pointer-events: none; white-space: pre-line; }
  #toast.show { opacity: 1; transform: translateY(0); }
  #toast.warn { border-left-color: var(--warn); }
  #toast.err  { border-left-color: var(--err); }

  /* ---- Per-image generation stats ---- */
  .result-stats { padding: 11px 14px; border-top: 1px solid var(--border); background: var(--panel); }
  .rs-head { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 10px; }
  .rs-time { font-size: 10px; color: var(--muted); margin-left: auto; font-variant-numeric: tabular-nums; }
  .prompt-wrap { margin: 10px 0 4px; }
  .prompt-head { font-size: 9px; text-transform: uppercase; letter-spacing: .7px; color: var(--muted); margin-bottom: 4px; font-weight: 600; }
  .prompt-text { background: rgba(0,0,0,.28); border: 1px solid var(--border); border-radius: 6px;
                 padding: 8px 10px; font-size: 12px; color: #c9d2e6; line-height: 1.55;
                 white-space: pre-wrap; word-break: break-word; }
  .prompt-text.clamp { display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
  .show-more { background: none; border: none; color: var(--accent); cursor: pointer; font-size: 10px; padding: 4px 2px; }
  .show-more:hover { text-decoration: underline; }
  .stats-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(145px, 1fr)); gap: 7px 16px; margin-top: 8px; }
  .stat-cell .k { font-size: 9px; text-transform: uppercase; letter-spacing: .6px; color: var(--muted); margin-bottom: 1px; }
  .stat-cell .v { font-size: 12px; color: var(--text); font-variant-numeric: tabular-nums; word-break: break-word; }
  .stat-cell .v.code { font-family: 'Consolas', monospace; cursor: pointer; }
  .stat-cell .v.code:hover { color: var(--accent); }
  .stat-cell .v.neg { font-size: 11px; color: #a9b1c9; }
  .dur-pill { display: inline-block; background: rgba(52,211,153,.12); color: var(--ok);
              border: 1px solid rgba(52,211,153,.3); border-radius: 20px; padding: 1px 9px;
              font-size: 12px; font-weight: 600; font-variant-numeric: tabular-nums; }
  .dur-split { margin-left: 6px; font-size: 10px; color: var(--muted); font-weight: 400; }

  /* ---- Gallery detail modal ---- */
  .modal-overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.68); z-index: 2000;
                   align-items: center; justify-content: center; padding: 26px; }
  .modal-overlay.show { display: flex; }
  .modal { background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
           max-width: 880px; width: 100%; max-height: calc(100vh - 52px); overflow-y: auto; }
  .modal-media { background: #05060a; border-radius: 12px 12px 0 0; }
  .modal-media img, .modal-media video { width: 100%; max-height: 60vh; object-fit: contain; display: block; }
  .modal-close { position: sticky; top: 8px; float: right; margin: 8px 8px 0 0; background: rgba(0,0,0,.55);
                 border: 1px solid var(--border); color: var(--text); border-radius: 50%; width: 28px; height: 28px;
                 cursor: pointer; font-size: 13px; z-index: 5; }
  .modal-close:hover { color: var(--err); border-color: var(--err); }
  .modal-body { padding: 4px 18px 16px; }
  .modal-body .rs-time { float: none; }
  .modal-actions { display: flex; gap: 8px; margin-top: 12px; }

  /* ---- Engine tabs ---- */
  /* Engine tabs — 2-col grid so every engine stays reachable; a scrollbar
     appears automatically if the catalogue outgrows the height budget. */
  .engine-tabs { display: grid; grid-template-columns: 1fr 1fr; gap: 6px;
                 padding: 8px 10px 6px; border-bottom: 1px solid var(--border);
                 max-height: 248px; overflow-y: auto; }
  .engine-tab  { display: flex; flex-direction: column; align-items: center;
                 justify-content: center; min-height: 40px; padding: 5px 6px;
                 text-align: center; cursor: pointer; font-size: 11px; font-weight: 600;
                 color: var(--muted); border: 1px solid var(--border); border-radius: 8px;
                 transition: all .2s; line-height: 1.2; }
  .engine-tab:hover  { color: var(--text); border-color: var(--accent); }
  .engine-tab.active { color: var(--accent); border-color: var(--accent);
                       background: rgba(108,142,247,.08); }
  .engine-tab .badge { font-size: 9px; margin-top: 2px; color: var(--muted);
                       font-weight: 400; }

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
  .comfy-banner { display: none; padding: 10px 16px; background: rgba(79, 70, 229, 0.12); color: #dbeafe; border: 1px solid rgba(129, 140, 248, 0.25); margin: 8px 16px; border-radius: 8px; font-size: 12px; }
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

<div class="comfy-banner" id="comfyBanner">🚀 ComfyUI mode enabled — GPU-only execution is active.</div>

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
    <div class="engine-actions">
      <button class="btn-load" id="btnPreload" onclick="preloadEngine()" title="Load this engine into VRAM now (honors the selected quantization)">⬇ Preload</button>
      <button class="btn-unload" id="btnUnload" onclick="unloadEngine()" title="Unload the resident engine from VRAM">⏏ Unload</button>
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
    <!-- VRAM / system report strip -->
    <div class="reportbar">
      <div class="row">
        <span style="font-size:11px;color:var(--muted)"><span class="status-dot" id="statusDot"></span><span id="statusText">Connecting…</span></span>
        <span id="activeEngineLabel" style="color:var(--accent);font-size:11px;font-weight:600"></span>
        <span class="spacer"></span>
        <button id="btnEvict" class="btn-action danger" onclick="evictAllVRAM()"
                title="Unload the Image Lab model AND all TTS engines across the containers">Evict VRAM</button>
        <button id="btnRefresh" class="btn-action" onclick="refreshAvailability()"
                title="Re-probe engine availability">🔄 Refresh</button>
      </div>
      <div class="row">
        <div class="bar-item" id="ramItem">
          <div class="bar-head"><span>RAM</span><span class="bar-label" id="ramText">—</span></div>
          <div class="bar-track"><div class="bar-fill ram" id="ramFill" style="width:0%"></div></div>
        </div>
        <div class="bar-item">
          <div class="bar-head"><span>VRAM</span><span class="bar-label" id="vramText">—</span></div>
          <div class="bar-track wide"><div class="bar-fill vram" id="vramFill" style="width:0%"></div></div>
          <div id="vramDetail" title=""></div>
        </div>
        <span class="gpu-badge" id="gpuBadge">GPU —</span>
        <div class="loaded-wrap" style="margin-left:auto">
          <span class="loaded-label">🧠 In VRAM:</span>
          <span id="loadedChips"><span class="chip chip-idle">—</span></span>
        </div>
      </div>
    </div>

    <!-- View tabs -->
    <div class="view-tabs">
      <div class="view-tab active" onclick="switchView('generate')">Generate</div>
      <div class="view-tab"        onclick="switchView('gallery')">Gallery</div>
      <div class="view-tab"        onclick="switchView('logs')">Logs</div>
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
        <!-- Options are filled from /status at boot (see buildGalleryFilter) -->
        <select class="gallery-filter" id="galleryFilter" onchange="loadGallery()">
          <option value="">All engines</option>
        </select>
      </div>
      <div class="gallery-grid" id="galleryGrid"></div>
    </div>

    <!-- Logs view -->
    <div class="view-panel" id="viewLogs">
      <div class="gallery-header">
        <h2>Server Logs</h2>
        <button class="btn-copy" onclick="loadLogs()" style="padding:6px 14px;font-size:12px;">↻ Refresh</button>
      </div>
      <div id="logsViewer" style="background:#0d1117;color:#c9d1d9;font:12px/1.6 'JetBrains Mono','Consolas',monospace;padding:12px 16px;border-radius:6px;max-height:calc(100vh - 200px);overflow-y:auto;white-space:pre-wrap;word-break:break-all;"></div>
    </div>
  </main>
</div>

<!-- Gallery detail modal -->
<div class="modal-overlay" id="galleryModal" onclick="if (event.target === this) closeGalleryDetail()">
  <div class="modal">
    <button class="modal-close" onclick="closeGalleryDetail()" title="Close">✕</button>
    <div class="modal-media" id="galleryMedia"></div>
    <div class="modal-body">
      <div id="galleryStats"></div>
      <div class="modal-actions">
        <button class="btn-dl" id="galleryDl" style="margin-left:0">⬇ Download</button>
        <button class="btn-action danger" id="galleryDel" onclick="deleteFromDetail()">🗑 Delete</button>
      </div>
    </div>
  </div>
</div>

<div id="toast"></div>

<script>
// ============================================================
// State
// ============================================================
const ENGINES_META = {};   // filled by /status
let currentEngine  = 'flux2klein';
let formValues     = {};   // param key → current value
let uploadedFile   = null; // reference image bytes
let lastStatus     = null; // last /status response

// ============================================================
// Boot
// ============================================================
(async function boot() {
  uiLog('info', 'Image Lab UI booting', {url: location.href, ua: navigator.userAgent.slice(0,60)});
  const ok = await refreshStatus();      // full fetch — loads param schemas AND builds the engine tabs
  if (!ok || !lastStatus) {
    // Server unreachable at boot — show the known engines anyway; the first
    // successful poll rebuilds the list from /status.
    buildEngineTabs(DEFAULT_ENGINES);
    buildGalleryFilter(DEFAULT_ENGINES);
  }
  await selectEngine(currentEngine);
  loadGallery();
  // VRAM/system report auto-refresh — brief=1 keeps the polled payload light
  // (per-engine param schemas are only fetched at boot).
  setInterval(() => refreshStatus(true), 4000);
  uiLog('info', 'Boot complete');
})();

// ============================================================
// Status polling
// ============================================================
async function refreshStatus(brief) {
  try {
    const s = await apiFetch('/status' + (brief ? '?brief=1' : ''));
    lastStatus = s;
    // Update engine meta — a brief poll carries no params/description, so it
    // must never overwrite the full schemas captured at boot.
    for (const e of s.engines) {
      if (e.params) ENGINES_META[e.key] = e;
    }
    // Rebuild the tab bar / gallery filter when the server-side catalogue
    // changes (boot, or an engine added without a UI redeploy).
    syncEngineList(s.engines);
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
    renderVramReport(s);
    // Update tab availability badges
    for (const e of s.engines) {
      const el = document.getElementById('tab-' + e.key);
      if (el) {
        const badge = el.querySelector('.badge');
        badge.textContent = e.available ? '✓ available' : '✗ unavailable';
        badge.style.color  = e.available ? 'var(--ok)' : 'var(--err)';
      }
    }
    return true;
  } catch(e) {
    document.getElementById('statusDot').className = 'status-dot err';
    document.getElementById('statusText').textContent = 'Server unreachable';
    uiLog('warn', 'Status poll failed', e.message);
    return false;
  }
}

/* ---- VRAM / system report rendering ---- */

function fmtMem(mb) {
  if (mb === undefined || mb === null || isNaN(mb)) return '—';
  return mb >= 1024 ? (mb/1024).toFixed(1) + ' GiB' : Math.round(mb) + ' MiB';
}

function fmtTime(epoch) {
  if (!epoch) return '—';
  const d = new Date(epoch * 1000);
  const today = new Date();
  const sameDay = d.toDateString() === today.toDateString();
  return sameDay
    ? d.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'})
    : d.toLocaleString([], {month:'short', day:'numeric', hour:'2-digit', minute:'2-digit'});
}

function procLabel(p) {
  if (p.container) {
    let c = p.container;
    if (c.startsWith('tts-lab-')) c = c.slice('tts-lab-'.length);
    return c;
  }
  const n = (p.process || '').toLowerCase();
  // Bare-metal host process — the Image Lab service is the only host CUDA app
  return n.includes('python') ? 'Image Lab' : (p.process ? p.process.split(/[\\/]/).pop() : 'host');
}

function renderVramReport(s) {
  // RAM bar (hidden when host stats are unavailable)
  const ramItem = document.getElementById('ramItem');
  const sys = s.system;
  if (sys && sys.total > 0) {
    ramItem.style.display = '';
    const pct = Math.min(100, sys.used / sys.total * 100);
    document.getElementById('ramFill').style.width = pct + '%';
    document.getElementById('ramText').textContent =
      `${fmtMem(sys.used)} / ${fmtMem(sys.total)} (${pct.toFixed(0)}%)`;
  } else {
    ramItem.style.display = 'none';
  }

  // VRAM: device-wide nvidia-smi view first; torch-process view as fallback
  const g = s.gpu, v = s.vram;
  let usedMB = 0, totalMB = 0, name = '', procs = [];
  if (g && g.available && g.vram_total_mb > 0) {
    usedMB = g.vram_used_mb; totalMB = g.vram_total_mb; name = g.name; procs = g.processes || [];
  } else if (v && v.available) {
    usedMB = v.reserved_gb * 1024; totalMB = v.total_gb * 1024; name = v.device_name || '';
  }
  const pct = totalMB > 0 ? Math.min(100, usedMB / totalMB * 100) : 0;
  const fill = document.getElementById('vramFill');
  fill.style.width = pct + '%';
  fill.classList.toggle('hot', pct >= 88);
  document.getElementById('vramText').textContent =
    totalMB > 0 ? `${fmtMem(usedMB)} / ${fmtMem(totalMB)} (${pct.toFixed(0)}%)` : '—';

  const badge = document.getElementById('gpuBadge');
  if (badge) {
    if (totalMB > 0) {
      badge.className = 'gpu-badge ok';
      badge.textContent = `🟢 ${name || 'GPU'} · ${fmtMem(totalMB)}`;
      badge.title = name || '';
    } else {
      badge.className = 'gpu-badge off';
      badge.textContent = '🔴 GPU unavailable';
      badge.title = '';
    }
  }

  // WHO holds the VRAM — containers + host processes
  const det = document.getElementById('vramDetail');
  if (det) {
    if (procs.length) {
      det.textContent = 'GPU: ' + procs.map(p => `${procLabel(p)} ${fmtMem(p.mb)}`).join(' · ');
      det.title = det.textContent;
    } else if (totalMB > 0) {
      det.textContent = 'GPU: no CUDA compute processes';
      det.title = '';
    } else {
      det.textContent = '';
    }
  }

  // Resident-engine chip (single-resident model — one chip max)
  const chips = document.getElementById('loadedChips');
  const key = s.active_engine;
  if (key) {
    const label = (ENGINES_META[key]?.label) || key;
    chips.innerHTML = `<span class="chip" title="Resident in VRAM — ✕ unloads it">🧠 ${escHtml(label)}${
      s.active_quant ? ' · ' + escHtml(s.active_quant) : ''}` +
      `<button class="chip-x" onclick="evictEngine('${key}')" title="Unload from VRAM">✕</button></span>`;
  } else {
    chips.innerHTML = '<span class="chip chip-idle">nothing loaded</span>';
  }

  // Busy → management buttons disabled (enabled again on the next poll)
  const busy = s.loading || s.generating;
  ['btnEvict', 'btnRefresh', 'btnPreload', 'btnUnload'].forEach(id => {
    const b = document.getElementById(id);
    if (b) b.disabled = busy;
  });
}

// ============================================================
// VRAM management
// ============================================================
let _toastTimer = null;
function showToast(msg, kind) {
  const t = document.getElementById('toast');
  if (!t) return;
  t.textContent = msg;
  t.className = 'show' + (kind ? ' ' + kind : '');
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => { t.className = ''; }, kind === 'err' ? 6000 : 4200);
}

async function preloadEngine() {
  const label = ENGINES_META[currentEngine]?.label || currentEngine;
  const quant = formValues['quant'] ?? '';
  const btn = document.getElementById('btnPreload');
  if (btn) btn.disabled = true;
  uiLog('info', `POST /engines/${currentEngine}/load` + (quant ? ` quant=${quant}` : ''));
  try {
    const fd = new FormData();
    if (quant) fd.append('quant', quant);
    const r = await fetch('/engines/' + currentEngine + '/load', {method: 'POST', body: fd});
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(extractErrorMsg(data, r.statusText));
    uiLog('info', `Preloaded ${label}`, {quant});
    showToast('Loaded ' + label + (quant ? ' · ' + quant : ''));
    await refreshStatus();
  } catch(e) {
    uiLog('error', 'Preload failed', e.message);
    showToast('Load failed: ' + e.message, 'err');
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function unloadEngine() {
  const btn = document.getElementById('btnUnload');
  if (btn) btn.disabled = true;
  uiLog('info', 'POST /engines/unload');
  try {
    const r = await fetch('/engines/unload', {method: 'POST'});
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(extractErrorMsg(data, r.statusText));
    showToast('Unloaded from VRAM');
    await refreshStatus();
  } catch(e) {
    uiLog('error', 'Unload failed', e.message);
    showToast('Unload failed: ' + e.message, 'err');
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function evictEngine(key) {
  const label = ENGINES_META[key]?.label || key;
  if (!confirm(`Unload ${label} from VRAM?\n\nIt will reload lazily on the next generation.`)) return;
  uiLog('info', `POST /engines/${key}/evict`);
  try {
    const r = await fetch('/engines/' + key + '/evict', {method: 'POST'});
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(extractErrorMsg(data, r.statusText));
    if (data.evicted) showToast('Unloaded ' + label + ' from VRAM');
    else showToast(label + ': ' + (data.note || 'not evicted'), 'warn');
    await refreshStatus();
  } catch(e) {
    uiLog('error', 'Evict failed', e.message);
    showToast('Evict failed: ' + e.message, 'err');
  }
}

async function evictAllVRAM() {
  const btn = document.getElementById('btnEvict');
  if (!btn || btn.disabled) return;
  if (!confirm('Evict everything from VRAM?\n\nUnloads the Image Lab model AND every TTS engine across the TTS containers. Models reload lazily on their next request.')) return;
  btn.disabled = true;
  btn.textContent = 'Evicting…';
  uiLog('info', 'POST /evict-all');
  try {
    const r = await fetch('/evict-all', {method: 'POST'});
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(extractErrorMsg(data, r.statusText));
    const parts = [];
    const il = data.image_lab || {};
    if (il.unloaded) parts.push('Image Lab: unloaded' + (il.engine ? ' ' + (ENGINES_META[il.engine]?.label || il.engine) : ''));
    else parts.push('Image Lab: kept (busy)');
    const tts = data.tts || {};
    if (tts.evicted_count !== undefined) {
      parts.push(`TTS: ${tts.evicted_count} evicted` + (tts.freed_mb_total ? `, freed ${fmtMem(tts.freed_mb_total)}` : ''));
    } else if (tts.error) {
      parts.push('TTS: ' + tts.error);
    }
    const errs = (data.errors || []).length;
    showToast(parts.join('\n') || 'Eviction done', errs ? 'warn' : '');
    await refreshStatus();
    btn.textContent = 'Done';
    setTimeout(() => { btn.textContent = 'Evict VRAM'; }, 2000);
  } catch(e) {
    uiLog('error', 'Evict-all failed', e.message);
    showToast('Eviction failed: ' + e.message, 'err');
    btn.textContent = 'Evict VRAM';
  } finally {
    btn.disabled = false;
  }
}

async function refreshAvailability() {
  const btn = document.getElementById('btnRefresh');
  if (btn) btn.disabled = true;
  uiLog('info', 'POST /refresh');
  try {
    const r = await fetch('/refresh', {method: 'POST'});
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(extractErrorMsg(data, r.statusText));
    showToast('Availability refreshed');
    await refreshStatus();
  } catch(e) {
    uiLog('error', 'Refresh failed', e.message);
    showToast('Refresh failed: ' + e.message, 'err');
  } finally {
    if (btn) btn.disabled = false;
  }
}

// ============================================================
// Engine tabs — built from the /status catalogue, never hardcoded:
// adding an engine to image_lab_config.py must not require a UI edit.
// ============================================================
const DEFAULT_ENGINES = [
  {key: 'flux2klein',    label: 'FLUX.2 Klein',      available: null},
  {key: 'flux2klein9b',  label: 'Klein 9B-KV',       available: null},
  {key: 'ideogram4',     label: 'Ideogram 4',        available: null},
  {key: 'sana',          label: 'SANA 1.6B',         available: null},
  {key: 'boogu',         label: 'Boogu Turbo',       available: null},
  {key: 'zimage',        label: 'Z-Image Turbo',     available: null},
  {key: 'qwenimage',     label: 'Qwen-Image 2512',   available: null},
  {key: 'hidream',       label: 'HiDream O1',        available: null},
  {key: 'ernie',         label: 'ERNIE-Image',       available: null},
];

let engineSig = null;   // signature of the currently-rendered engine list

function syncEngineList(engines) {
  const sig = engines.map(e => e.key).join('|');
  if (sig === engineSig) return;
  engineSig = sig;
  const prev = currentEngine;
  if (!engines.some(e => e.key === prev)) {
    // Selected engine vanished from the catalogue — jump to the first
    // available one (or the first listed).
    currentEngine = (engines.find(e => e.available) || engines[0] || {}).key || currentEngine;
  }
  buildEngineTabs(engines);
  buildGalleryFilter(engines);
  if (currentEngine !== prev) selectEngine(currentEngine);
}

function escAttr(s) {
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function badgeMarkup(e) {
  // available: true / false from the server; null = not yet probed
  if (e.available === true)  return '<span class="badge" style="color:var(--ok)">✓ available</span>';
  if (e.available === false) return '<span class="badge" style="color:var(--err)">✗ unavailable</span>';
  return '<span class="badge">…</span>';
}

function buildEngineTabs(engines) {
  const tabs = document.getElementById('engineTabs');
  tabs.innerHTML = (engines || []).map(e => `
    <div class="engine-tab" id="tab-${e.key}" onclick="selectEngine('${e.key}')"
         title="${escAttr(e.label || e.key)}">
      ${escHtml(e.label || e.key)}
      ${badgeMarkup(e)}
    </div>`).join('');
  const tab = document.getElementById('tab-' + currentEngine);
  if (tab) tab.classList.add('active');
}

function buildGalleryFilter(engines) {
  const sel = document.getElementById('galleryFilter');
  if (!sel) return;
  const prev = sel.value;
  sel.innerHTML = '<option value="">All engines</option>' +
    (engines || []).map(e =>
      `<option value="${escAttr(e.key)}">${escHtml(e.label || e.key)}</option>`).join('');
  if (prev && engines && engines.some(e => e.key === prev)) sel.value = prev;
}

// Full /status fetch used when an engine is selected before its schema is
// known (e.g. an engine added server-side mid-session).
async function ensureMeta(key) {
  try {
    const s = await apiFetch('/status');
    for (const e of s.engines) {
      if (e.params) ENGINES_META[e.key] = e;
    }
    syncEngineList(s.engines);
    return !!ENGINES_META[key];
  } catch(e) {
    return false;
  }
}

async function selectEngine(key) {
  currentEngine = key;
  uploadedFile  = null;
  document.querySelectorAll('.engine-tab').forEach(t => t.classList.remove('active'));
  const tab = document.getElementById('tab-' + key);
  if (tab) tab.classList.add('active');
  if (!ENGINES_META[key] && !(await ensureMeta(key))) {
    showToast('Engine "' + key + '" is not known by this server', 'err');
    return;
  }
  renderParams(key);
  updateQuantWarning();
  updateCurl();
}

// ============================================================
// Parameter rendering
// ============================================================
function toggleMagicPrompt(visible) {
  const magicRatio = document.querySelector('[data-param="magic_prompt_aspect_ratio"]');
  if (magicRatio) {
    const group = magicRatio.closest('.param-group');
    if (group) group.style.display = visible ? '' : 'none';
  }
}

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
  // Initialise magic prompt visibility for ideogram4
  if (key === 'ideogram4') {
    const magicEnabled = formValues['use_magic_prompt'] === true;
    toggleMagicPrompt(magicEnabled);
  }
  // Apply the SANA variant defaults (steps for the selected checkpoint)
  if (key === 'sana') {
    applySanaVariant(formValues['quant'] || 'sprint-1.6b');
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
    el.name = p.name;
    el.dataset.param = p.name;
    el.placeholder = p.tooltip || '';
    el.value = formValues[p.name] ?? p.default ?? '';
  } else if (p.type === 'select') {
    el = document.createElement('select');
    el.name = p.name;
    el.dataset.param = p.name;
    for (const opt of (p.options || [])) {
      const o   = document.createElement('option');
      const val = (opt && typeof opt === 'object') ? opt.value : opt;
      const lbl = (opt && typeof opt === 'object') ? opt.label  : opt;
      o.value = val; o.textContent = lbl;
      if (val === (formValues[p.name] ?? p.default)) o.selected = true;
      el.appendChild(o);
    }
  } else if (p.type === 'checkbox') {
    el = document.createElement('input');
    el.type = 'checkbox';
    el.name = p.name;
    el.dataset.param = p.name;
    el.checked = formValues[p.name] ?? p.default ?? false;
    // Repurpose the outer label to be an inline checkbox label
    label.textContent = '';
    const labelSpan = document.createElement('span');
    labelSpan.textContent = p.label;
    label.prepend(el);
    label.appendChild(labelSpan);
    label.style.cssText = 'display:flex;align-items:center;gap:8px;cursor:pointer;font-size:12px;';
    el.onchange = () => {
      formValues[p.name] = el.checked;
      updateCurl();
      if (p.name === 'use_magic_prompt') toggleMagicPrompt(el.checked);
    };
    formValues[p.name] = el.checked;
    return wrap;
  } else if (p.type === 'int' || p.type === 'float') {
    // Use range + number input side by side when min/max defined
    if (p.min !== undefined && p.max !== undefined) {
      const row = document.createElement('div');
      row.className = 'range-row';
      const range = document.createElement('input');
      range.type = 'range';
      range.name = p.name;
      range.dataset.param = p.name;
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
      el.name = p.name;
      el.dataset.param = p.name;
      el.step = p.step ?? (p.type === 'float' ? 0.1 : 1);
      el.value = formValues[p.name] ?? p.default ?? 0;
    }
  } else {
    el = document.createElement('input');
    el.type = 'text';
    el.name = p.name;
    el.dataset.param = p.name;
    el.value = formValues[p.name] ?? p.default ?? '';
  }

  el.oninput = el.onchange = () => {
    formValues[p.name] = el.value;
    updateCurl();
    if (p.name === 'quant') updateQuantWarning();
    if (currentEngine === 'sana' && p.name === 'quant') applySanaVariant(el.value);
  };
  formValues[p.name] = el.value ?? el.options?.[el.selectedIndex]?.value ?? '';
  wrap.appendChild(el);
  return wrap;
}

// SANA variant preset — switch the step default with the checkpoint. Sprint is
// step-distilled (1-4, no CFG); SANA 1.5 wants ~20 steps at CFG 4.5. The
// variant itself travels in the `quant` field, so the existing reload warning
// banner fires on change (correct — switching checkpoint reloads the engine).
function applySanaVariant(variant) {
  const defaults = {
    'sprint-1.6b': {steps: 4,  guidance: 4.5},  // guidance ignored by Sprint
    '1.5-1.6b':    {steps: 20, guidance: 4.5},
  };
  const data = defaults[variant];
  if (!data) return;

  const stepEl = document.querySelector('[data-param="num_inference_steps"]');
  if (stepEl) {
    stepEl.value = data.steps;
    stepEl.dispatchEvent(new Event('input', { bubbles: true }));
  }
  // Guidance is intentionally left alone: it only matters for the 1.5 variant
  // and the 4.5 default is already correct for both presets.
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
  // Trim old log entries to prevent DOM bloat (>300 nodes)
  while (body.children.length > 300) {
    body.removeChild(body.firstChild);
  }
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
    if (p.type === 'file' || p.client_only) continue;
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

    const stats = document.createElement('div');
    stats.className = 'result-stats';
    stats.innerHTML = buildStatsHTML(r, {download: true});
    card.appendChild(stats);

    pane.insertBefore(card, pane.firstChild);
  }
}

// ============================================================
// Per-image stats (shared by output cards + gallery detail modal)
// ============================================================
const PARAM_LABELS = {
  seed: 'Seed', width: 'Width', height: 'Height', quant: 'Quant',
  num_inference_steps: 'Steps', guidance_scale: 'Guidance', num_images: 'Images',
  mode: 'Mode', num_frames: 'Frames', fps: 'FPS', resolution: 'Resolution',
  preset: 'Preset', mu: 'MU', std: 'STD', use_magic_prompt: 'Magic prompt',
  magic_prompt_aspect_ratio: 'Aspect', negative_prompt: 'Negative prompt',
};
const PARAM_ORDER = ['seed', 'width', 'height', 'quant', 'num_inference_steps',
  'guidance_scale', 'num_images', 'mode', 'num_frames', 'fps', 'resolution',
  'preset', 'mu', 'std', 'use_magic_prompt', 'magic_prompt_aspect_ratio',
  'negative_prompt'];

function buildStatsHTML(r, opts) {
  const meta = ENGINES_META[r.engine] || {};
  const label = meta.label || r.engine;
  const p = r.params || {};
  const st = r.stats || null;
  const h = [];
  const esc = escHtml;

  // Head: engine chip · quant chip · time · download
  h.push('<div class="rs-head">');
  h.push('<span class="chip">' + (r.type === 'image' ? '🖼 ' : '🎞 ') + esc(label) + '</span>');
  if (p.quant) h.push('<span class="chip chip-idle">' + esc(p.quant) + '</span>');
  h.push('<span class="rs-time">' +
    (st ? 'finished ' + fmtTime(st.finished_at) : 'generated ' + fmtTime(r.created_at)) +
    '</span>');
  if (opts && opts.download && r.url) {
    h.push(`<button class="btn-dl" style="margin-left:0" onclick="dlFile('${r.url}','${r.filename || 'output'}')">⬇ Download</button>`);
  }
  h.push('</div>');

  // The actual prompt sent to the model. Ideogram 4 records the magic-prompt
  // expanded caption separately, so it becomes the primary block with the raw
  // submitted prompt underneath when the two differ.
  const prompt  = (p.prompt || '').trim();
  const caption = (p.caption || '').trim();
  const eff     = caption && caption !== prompt ? caption : prompt;
  if (eff) {
    h.push('<div class="prompt-wrap">');
    h.push('<div class="prompt-head">' + (caption && caption !== prompt ? 'Prompt used (expanded)' : 'Prompt') + '</div>');
    const long = eff.length > 140;
    h.push(`<div class="prompt-text${long ? ' clamp' : ''}">${esc(eff)}</div>`);
    if (long) h.push('<button class="show-more" onclick="togglePrompt(this)">Show more</button>');
    h.push('</div>');
  }
  if (prompt && caption && caption !== prompt) {
    h.push('<div class="prompt-wrap">');
    h.push('<div class="prompt-head" style="color:#5d6478">Submitted prompt</div>');
    h.push(`<div class="prompt-text" style="background:none;font-size:11px;color:#8a91a8">${esc(prompt)}</div>`);
    h.push('</div>');
  }

  // Timing + parameter grid
  const cells = [];
  if (st && st.total_s !== undefined) {
    const gen = Math.max(st.total_s - (st.load_s || 0), 0);
    cells.push(`<div class="stat-cell"><div class="k">Started</div><div class="v">${fmtTime(st.started_at)}</div></div>`);
    cells.push(`<div class="stat-cell"><div class="k">Finished</div><div class="v">${fmtTime(st.finished_at)}</div></div>`);
    let split = '';
    if (st.load_s !== null && st.load_s !== undefined && st.load_s > 0 && gen > 0) {
      split = `<span class="dur-split">${st.load_s.toFixed(1)} s load · ${gen.toFixed(1)} s gen</span>`;
    }
    cells.push(`<div class="stat-cell"><div class="k">Duration</div><div class="v"><span class="dur-pill">⏱ ${st.total_s.toFixed(1)} s</span>${split}</div></div>`);
  } else if (r.created_at) {
    cells.push(`<div class="stat-cell"><div class="k">Generated</div><div class="v">${fmtTime(r.created_at)}</div></div>`);
    cells.push('<div class="stat-cell"><div class="k">Duration</div><div class="v">not recorded</div></div>');
  }
  for (const k of PARAM_ORDER) {
    const v = p[k];
    if (v === undefined || v === null || v === '') continue;
    const txt = typeof v === 'boolean' ? (v ? 'Yes' : 'No') : String(v);
    const lbl = PARAM_LABELS[k] || k;
    if (k === 'seed') {
      cells.push(`<div class="stat-cell"><div class="k">${lbl}</div><div class="v code" onclick="copyText('${esc(txt)}')" title="Click to copy">${esc(txt)}</div></div>`);
    } else if (k === 'negative_prompt') {
      const clipped = txt.length > 180 ? txt.slice(0, 180) + '…' : txt;
      cells.push(`<div class="stat-cell"><div class="k">${lbl}</div><div class="v neg">${esc(clipped)}</div></div>`);
    } else {
      cells.push(`<div class="stat-cell"><div class="k">${lbl}</div><div class="v">${esc(txt)}</div></div>`);
    }
  }
  if (cells.length) h.push(`<div class="stats-grid">${cells.join('')}</div>`);
  return h.join('');
}

function togglePrompt(btn) {
  const t = btn.previousElementSibling;
  if (t && t.classList) {
    const clamped = t.classList.toggle('clamp');
    btn.textContent = clamped ? 'Show more' : 'Show less';
  }
}

function dlFile(url, name) {
  const a = document.createElement('a');
  a.href = url; a.download = name || 'output';
  document.body.appendChild(a); a.click(); a.remove();
}

function copyText(txt) {
  navigator.clipboard.writeText(txt)
    .then(() => showToast('Copied: ' + txt))
    .catch(() => {});
}

// ============================================================
// Gallery detail modal
// ============================================================
let _detailEntry = null;

function openGalleryDetail(entry) {
  _detailEntry = entry;
  const media = document.getElementById('galleryMedia');
  media.innerHTML = '';
  if (entry.type === 'image') {
    const img = document.createElement('img');
    img.src = entry.url; img.alt = 'Generated image';
    media.appendChild(img);
  } else {
    const vid = document.createElement('video');
    vid.src = entry.url; vid.controls = true; vid.muted = true;
    media.appendChild(vid);
  }
  document.getElementById('galleryStats').innerHTML = buildStatsHTML(entry, {});
  const dl = document.getElementById('galleryDl');
  dl.onclick = () => dlFile(entry.url, entry.filename || 'output');
  document.getElementById('galleryModal').classList.add('show');
}

function closeGalleryDetail() {
  document.getElementById('galleryModal').classList.remove('show');
  _detailEntry = null;
}

async function deleteFromDetail() {
  const e = _detailEntry;
  if (!e) return;
  if (!confirm('Delete this generation?')) return;
  try {
    await fetch('/gallery/' + e.id, {method: 'DELETE'});
    closeGalleryDetail();
    loadGallery();
    showToast('Deleted');
  } catch(err) {
    showToast('Delete failed: ' + err.message, 'err');
  }
}

document.addEventListener('keydown', ev => {
  if (ev.key === 'Escape') closeGalleryDetail();
});

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
    thumb.onclick = () => { openGalleryDetail(e); };
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

const USE_COMFYUI = false;
if (USE_COMFYUI) {
  const banner = document.getElementById('comfyBanner');
  if (banner) banner.style.display = 'block';
}

function switchView(name) {
  document.querySelectorAll('.view-tab').forEach((t,i) => {
    t.classList.toggle('active', ['generate','gallery','logs'][i] === name);
  });
  document.querySelectorAll('.view-panel').forEach(p => {
    p.classList.toggle('active', p.id === 'view' + name.charAt(0).toUpperCase() + name.slice(1));
  });
  if (name === 'gallery') loadGallery();
  if (name === 'logs') loadLogs();
}

async function loadLogs() {
  const viewer = document.getElementById('logsViewer');
  if (!viewer) return;
  viewer.textContent = 'Loading logs…';
  try {
    const data = await apiFetch('/logs?n=200');
    if (!data.logs || !data.logs.length) {
      viewer.textContent = 'No log entries yet.';
      return;
    }
    const lines = data.logs.map(l =>
      `\x1b[90m${l.time}\x1b[0m \x1b[${l.level==='ERROR'?'31':l.level==='WARNING'?'33':l.level==='INFO'?'36':'37'}m[${l.level.padEnd(7)}]\x1b[0m \x1b[35m${l.logger}\x1b[0m — ${l.msg}`
    );
    // Strip ANSI for non-terminal display — use HTML spans instead
    const htmlLines = data.logs.map(l => {
      const lvlColor = l.level==='ERROR'?'#f85149':l.level==='WARNING'?'#d29922':l.level==='INFO'?'#58a6ff':'#8b949e';
      return `<span style="color:#484f58">${l.time}</span> <span style="color:${lvlColor}">[${l.level.padEnd(7)}]</span> <span style="color:#bc8cff">${l.logger}</span> — ${escHtml(l.msg)}`;
    });
    viewer.innerHTML = htmlLines.join('\n');
    // Auto-scroll to bottom
    viewer.scrollTop = viewer.scrollHeight;
  } catch (e) {
    viewer.textContent = 'Failed to load logs: ' + e.message;
  }
}

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}
</script>
</body>
</html>
"""


def get_ui_html() -> str:
    if USE_COMFYUI:
        return UI_HTML.replace('const USE_COMFYUI = false;', 'const USE_COMFYUI = true;')
    return UI_HTML
    return UI_HTML
