"""
lab_infra_ui.py — The Infrastructure dashboard page (HTML + CSS + JS).

Self-contained by design: no CDN, no build step, no external fonts — the page
must render on an isolated LAN. Served by lab_infra.py's ``GET /infra`` route.

Rendering model: one ``/infra/api/overview`` poll feeds every panel; the
topology SVG is only rebuilt when its node/edge signature changes so hover
state survives the 5 s refresh.
"""

INFRA_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Arthur Lab — Infrastructure</title>
<style>
  :root{
    --bg:#080a0f; --bg2:#0c0f16; --panel:#12151e; --panel2:#161a25; --panel3:#1c2130;
    --border:#242a3a; --border2:#333b52;
    --text:#e7eaf3; --muted:#8b93a8; --dim:#5b6379;
    --acc:#6366f1; --acc2:#22d3ee; --acc3:#a78bfa;
    --ok:#34d399; --warn:#fbbf24; --err:#f87171; --info:#60a5fa;
    --r:14px; --r2:10px;
    --mono:'JetBrains Mono',ui-monospace,'Cascadia Mono',Consolas,monospace;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  html,body{height:100%}
  body{background:radial-gradient(1200px 600px at 15% -10%,#151a2e 0%,var(--bg) 55%) no-repeat,var(--bg);
       color:var(--text);font:14px/1.45 'Segoe UI',system-ui,-apple-system,sans-serif;
       -webkit-font-smoothing:antialiased}
  a{color:var(--acc2);text-decoration:none}
  ::-webkit-scrollbar{width:10px;height:10px}
  ::-webkit-scrollbar-track{background:transparent}
  ::-webkit-scrollbar-thumb{background:#262d40;border-radius:6px;border:2px solid var(--bg)}
  ::-webkit-scrollbar-thumb:hover{background:#36405c}

  /* ── shell ───────────────────────────────────────────────────────── */
  .app{display:flex;flex-direction:column;min-height:100%}
  .topbar{position:sticky;top:0;z-index:40;display:flex;align-items:center;gap:18px;
          padding:12px 20px;background:rgba(10,12,18,.82);backdrop-filter:blur(14px);
          border-bottom:1px solid var(--border);flex-wrap:wrap}
  .brand{display:flex;align-items:center;gap:11px}
  .brand .glyph{width:34px;height:34px;border-radius:10px;display:grid;place-items:center;
                background:linear-gradient(135deg,var(--acc),var(--acc2));
                box-shadow:0 6px 20px -6px var(--acc)}
  .brand .glyph svg{width:19px;height:19px;stroke:#fff}
  .brand h1{font-size:15px;font-weight:700;letter-spacing:.2px}
  .brand .sub{font-size:11px;color:var(--dim)}
  .pills{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
  .pill{display:flex;align-items:center;gap:7px;background:var(--panel);border:1px solid var(--border);
        border-radius:999px;padding:5px 12px;font-size:11.5px;color:var(--muted);
        font-variant-numeric:tabular-nums;white-space:nowrap}
  .pill b{color:var(--text);font-weight:650}
  .pill.ok{border-color:rgba(52,211,153,.35)}
  .pill.bad{border-color:rgba(248,113,113,.45);color:#fca5a5}
  .spacer{flex:1}
  .btn{background:var(--panel);border:1px solid var(--border);color:var(--text);border-radius:9px;
       padding:6px 12px;font-size:12px;cursor:pointer;font-family:inherit;transition:.15s;white-space:nowrap}
  .btn:hover{background:var(--panel3);border-color:var(--border2);transform:translateY(-1px)}
  .btn:active{transform:none}
  .btn.ghost{background:transparent}
  .btn.accent{background:linear-gradient(135deg,var(--acc),#4f46e5);border-color:transparent;
              box-shadow:0 6px 18px -8px var(--acc)}
  .btn.danger{border-color:rgba(248,113,113,.4);color:#fca5a5}
  .btn.danger:hover{background:rgba(248,113,113,.12)}
  .btn.tiny{padding:4px 8px;font-size:11px;border-radius:7px}
  .btn[disabled]{opacity:.4;cursor:not-allowed;pointer-events:none}
  .auto{display:flex;align-items:center;gap:6px;font-size:11.5px;color:var(--muted)}
  .auto input{accent-color:var(--acc);width:14px;height:14px}

  /* ── tabs ────────────────────────────────────────────────────────── */
  .tabs{display:flex;gap:2px;padding:0 20px;border-bottom:1px solid var(--border);
        background:rgba(10,12,18,.55);position:sticky;top:59px;z-index:30;overflow-x:auto}
  .tabs button{background:none;border:none;border-bottom:2px solid transparent;color:var(--muted);
               font:inherit;font-size:12.5px;font-weight:600;padding:11px 15px;cursor:pointer;
               white-space:nowrap;display:flex;align-items:center;gap:7px}
  .tabs button:hover{color:var(--text)}
  .tabs button.active{color:var(--acc2);border-bottom-color:var(--acc2)}
  .tabs .n{background:var(--panel3);border-radius:999px;padding:0 7px;font-size:10.5px;
           color:var(--muted);font-variant-numeric:tabular-nums}

  /* ── views ───────────────────────────────────────────────────────── */
  main{padding:18px 20px 40px;flex:1}
  .view{display:none;animation:fade .18s ease}
  .view.active{display:block}
  @keyframes fade{from{opacity:0;transform:translateY(4px)}to{opacity:1}}
  .card{background:linear-gradient(180deg,var(--panel),var(--bg2));border:1px solid var(--border);
        border-radius:var(--r);padding:16px}
  .card h2{font-size:12px;font-weight:700;letter-spacing:.6px;text-transform:uppercase;
           color:var(--muted);margin-bottom:12px;display:flex;align-items:center;gap:8px}
  .grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}
  .muted{color:var(--muted)}
  .dim{color:var(--dim)}
  .mono{font-family:var(--mono);font-size:11.5px}
  .empty{padding:34px;text-align:center;color:var(--dim);font-size:13px}

  /* ── status dots ─────────────────────────────────────────────────── */
  .dot{width:9px;height:9px;border-radius:50%;display:inline-block;flex:none;background:var(--dim);
       box-shadow:0 0 0 0 rgba(0,0,0,0)}
  .dot.running{background:var(--ok);box-shadow:0 0 10px rgba(52,211,153,.75);animation:pulse 2.4s infinite}
  .dot.exited{background:#4b5563}
  .dot.created{background:var(--info)}
  .dot.paused{background:var(--acc3)}
  .dot.restarting{background:var(--info);animation:pulse 1s infinite}
  .dot.dead,.dot.removing{background:var(--err);box-shadow:0 0 10px rgba(248,113,113,.7)}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}

  .tag{display:inline-flex;align-items:center;gap:5px;font-size:10px;font-weight:650;
       border-radius:6px;padding:2px 7px;border:1px solid var(--border2);color:var(--muted);
       background:rgba(255,255,255,.03);letter-spacing:.3px;text-transform:uppercase}
  .tag.tts{color:#a5b4fc;border-color:rgba(99,102,241,.45);background:rgba(99,102,241,.12)}
  .tag.image{color:#c4b5fd;border-color:rgba(167,139,250,.45);background:rgba(167,139,250,.12)}
  .tag.infra{color:#67e8f9;border-color:rgba(34,211,238,.4);background:rgba(34,211,238,.1)}
  .tag.gpu{color:#fcd34d;border-color:rgba(251,191,36,.4);background:rgba(251,191,36,.1)}
  .tag.ok{color:#6ee7b7;border-color:rgba(52,211,153,.4);background:rgba(52,211,153,.1)}
  .tag.bad{color:#fca5a5;border-color:rgba(248,113,113,.4);background:rgba(248,113,113,.1)}
  .tag.warn{color:#fcd34d;border-color:rgba(251,191,36,.4);background:rgba(251,191,36,.1)}

  /* ── bars ────────────────────────────────────────────────────────── */
  .bar{height:5px;border-radius:99px;background:rgba(255,255,255,.07);overflow:hidden}
  .bar>i{display:block;height:100%;border-radius:99px;transition:width .4s ease;
         background:linear-gradient(90deg,var(--acc),var(--acc2))}
  .bar.warn>i{background:linear-gradient(90deg,#f59e0b,#fbbf24)}
  .bar.bad>i{background:linear-gradient(90deg,#dc2626,#f87171)}

  /* ── topology ────────────────────────────────────────────────────── */
  .topo-wrap{background:linear-gradient(180deg,#0e111a,#0a0c13);border:1px solid var(--border);
             border-radius:var(--r);padding:6px;overflow:auto;max-height:62vh}
  .topo-wrap svg{display:block;min-width:100%}
  .node-box{cursor:pointer}
  .node-box rect.body{transition:.18s}
  .node-box:hover rect.body{filter:brightness(1.28)}
  .edge{fill:none;stroke-width:1.6;transition:opacity .18s}
  .edge.on{stroke-dasharray:7 5;animation:flow 1s linear infinite}
  @keyframes flow{to{stroke-dashoffset:-24}}
  .edge.off{stroke:#33394d!important;opacity:.5}
  .edge.gpu{stroke-dasharray:3 4;opacity:.9;stroke-width:1.4}
  .dimmed{opacity:.14}
  .legend{display:flex;gap:16px;flex-wrap:wrap;margin:12px 0 0;font-size:11.5px;color:var(--muted)}
  .legend span{display:flex;align-items:center;gap:7px}
  .legend i{width:22px;height:0;border-top:2px solid var(--acc2);display:inline-block}
  .legend i.dash{border-top-style:dashed;border-color:var(--dim)}

  /* ── tables ──────────────────────────────────────────────────────── */
  table{width:100%;border-collapse:collapse;font-size:12.5px}
  th{text-align:left;font-size:10.5px;text-transform:uppercase;letter-spacing:.5px;color:var(--dim);
     font-weight:700;padding:7px 10px;border-bottom:1px solid var(--border);position:sticky;top:0;
     background:var(--bg2);z-index:1}
  td{padding:8px 10px;border-bottom:1px solid rgba(36,42,58,.6);vertical-align:middle}
  tbody tr:hover{background:rgba(99,102,241,.06)}
  .tbl-wrap{overflow:auto;max-height:66vh;border-radius:var(--r2)}
  .cell-ip{font-family:var(--mono);font-size:11.5px;color:#9fd8e8}
  .chips{display:flex;flex-wrap:wrap;gap:4px}
  .chip{font-size:10.5px;background:rgba(255,255,255,.045);border:1px solid var(--border);
        border-radius:6px;padding:1px 6px;color:var(--muted);font-family:var(--mono)}
  .chip.more{border-style:dashed}

  /* ── container cards ─────────────────────────────────────────────── */
  .toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:14px}
  .toolbar input[type=search],.toolbar select{background:var(--panel);border:1px solid var(--border);
      color:var(--text);border-radius:9px;padding:7px 11px;font:inherit;font-size:12.5px;outline:none}
  .toolbar input[type=search]{min-width:220px}
  .toolbar input:focus,.toolbar select:focus{border-color:var(--acc)}
  .cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(370px,1fr));gap:14px}
  .ccard{background:linear-gradient(180deg,var(--panel),var(--bg2));border:1px solid var(--border);
         border-radius:var(--r);padding:14px;display:flex;flex-direction:column;gap:11px;
         transition:.18s;position:relative;overflow:hidden}
  .ccard:before{content:'';position:absolute;inset:0 auto 0 0;width:3px;background:var(--dim)}
  .ccard.running:before{background:linear-gradient(180deg,var(--ok),#059669)}
  .ccard.exited:before{background:#3f4759}
  .ccard.dead:before,.ccard.restarting:before{background:linear-gradient(180deg,var(--err),#b91c1c)}
  .ccard:hover{border-color:var(--border2);transform:translateY(-2px);
               box-shadow:0 14px 32px -22px #000,0 0 0 1px rgba(99,102,241,.14)}
  .cc-head{display:flex;align-items:flex-start;gap:9px}
  .cc-name{font-size:13.5px;font-weight:700;word-break:break-all;line-height:1.25}
  .cc-img{font-family:var(--mono);font-size:11px;color:var(--muted);word-break:break-all}
  .cc-note{font-size:11px;color:var(--dim);font-style:italic}
  .cc-meta{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px 12px;font-size:11.5px}
  .cc-meta div{display:flex;gap:6px;align-items:baseline;min-width:0}
  .cc-meta .k{color:var(--dim);font-size:10.5px;text-transform:uppercase;letter-spacing:.4px;flex:none}
  .cc-meta .v{color:var(--text);font-variant-numeric:tabular-nums;overflow:hidden;
              text-overflow:ellipsis;white-space:nowrap}
  .cc-stats{display:flex;flex-direction:column;gap:7px}
  .stat-row{display:flex;align-items:center;gap:9px;font-size:11px}
  .stat-row .lbl{width:34px;color:var(--dim);font-weight:650;font-size:10px;letter-spacing:.4px}
  .stat-row .bar{flex:1}
  .stat-row .val{width:106px;text-align:right;color:var(--muted);font-variant-numeric:tabular-nums}
  .cc-acts{display:flex;gap:6px;flex-wrap:wrap;margin-top:auto;padding-top:4px;
           border-top:1px solid rgba(36,42,58,.7)}
  .cc-acts .btn{flex:0 0 auto}

  /* ── drawer ──────────────────────────────────────────────────────── */
  .scrim{position:fixed;inset:0;background:rgba(4,6,10,.6);backdrop-filter:blur(2px);z-index:60;
         opacity:0;pointer-events:none;transition:.2s}
  .scrim.open{opacity:1;pointer-events:auto}
  .drawer{position:fixed;top:0;right:0;height:100%;width:min(760px,94vw);z-index:61;
          background:var(--bg2);border-left:1px solid var(--border2);display:flex;flex-direction:column;
          transform:translateX(102%);transition:transform .24s cubic-bezier(.4,0,.2,1);
          box-shadow:-24px 0 60px -30px #000}
  .drawer.open{transform:none}
  .dr-head{padding:15px 18px;border-bottom:1px solid var(--border);display:flex;gap:12px;align-items:flex-start}
  .dr-head .x{margin-left:auto;background:none;border:none;color:var(--muted);font-size:19px;
              cursor:pointer;line-height:1;padding:2px 6px;border-radius:7px}
  .dr-head .x:hover{background:var(--panel3);color:var(--text)}
  .dr-tabs{display:flex;gap:2px;padding:0 14px;border-bottom:1px solid var(--border);overflow-x:auto}
  .dr-tabs button{background:none;border:none;border-bottom:2px solid transparent;color:var(--muted);
       font:inherit;font-size:12px;font-weight:600;padding:9px 12px;cursor:pointer;white-space:nowrap}
  .dr-tabs button.active{color:var(--acc2);border-bottom-color:var(--acc2)}
  .dr-body{flex:1;overflow:auto;padding:16px 18px}
  .kv{display:grid;grid-template-columns:150px 1fr;gap:7px 14px;font-size:12.5px}
  .kv .k{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.4px;padding-top:2px}
  .kv .v{word-break:break-word}
  .logs{background:#07090d;border:1px solid var(--border);border-radius:var(--r2);padding:11px;
        font-family:var(--mono);font-size:11.5px;line-height:1.55;white-space:pre-wrap;
        word-break:break-all;max-height:56vh;overflow:auto;color:#c7d0e0}
  .logs .lv-err{color:#fca5a5}
  .logs .lv-warn{color:#fcd34d}
  .sec{margin-bottom:20px}
  .sec>h3{font-size:11px;text-transform:uppercase;letter-spacing:.6px;color:var(--muted);
          margin-bottom:9px;font-weight:700}
  .sub-tbl{width:100%;font-size:12px}
  .secretval{filter:blur(5px);cursor:pointer;transition:.15s}
  .secretval.shown{filter:none}

  /* ── toasts ──────────────────────────────────────────────────────── */
  #toasts{position:fixed;bottom:18px;left:50%;transform:translateX(-50%);z-index:80;
          display:flex;flex-direction:column;gap:8px;align-items:center;pointer-events:none}
  .toast{background:var(--panel2);border:1px solid var(--border2);border-left:3px solid var(--acc);
         border-radius:var(--r2);padding:9px 15px;font-size:12.5px;box-shadow:0 12px 30px -12px #000;
         animation:rise .22s ease}
  .toast.ok{border-left-color:var(--ok)}
  .toast.err{border-left-color:var(--err);color:#fca5a5}
  @keyframes rise{from{opacity:0;transform:translateY(10px)}to{opacity:1}}

  .gpu-big{display:flex;gap:16px;align-items:center;flex-wrap:wrap}
  .gpu-big .num{font-size:22px;font-weight:750;font-variant-numeric:tabular-nums}
  .gpu-big .lbl{font-size:10.5px;color:var(--dim);text-transform:uppercase;letter-spacing:.5px}
  .stat-tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:12px}
  .tile{background:var(--panel);border:1px solid var(--border);border-radius:var(--r2);padding:11px 13px}
  .tile .v{font-size:17px;font-weight:700;font-variant-numeric:tabular-nums}
  .tile .l{font-size:10.5px;color:var(--dim);text-transform:uppercase;letter-spacing:.5px;margin-top:2px}
  .banner{padding:11px 16px;border-radius:var(--r2);border:1px solid rgba(248,113,113,.4);
          background:rgba(248,113,113,.09);color:#fca5a5;font-size:12.5px;margin-bottom:14px}
  .banner.warn{border-color:rgba(251,191,36,.35);background:rgba(251,191,36,.08);color:#fcd34d}
  .nowrap{white-space:nowrap}
</style>
</head>
<body>
<div class="app">
  <header class="topbar">
    <div class="brand">
      <span class="glyph"><svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round"
        stroke-linejoin="round"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3"
        width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14"
        y="14" width="7" height="7" rx="1.5"/></svg></span>
      <div>
        <h1>Infrastructure</h1>
        <div class="sub">containers · pipes · images · host</div>
      </div>
    </div>
    <div class="pills" id="pills"></div>
    <div class="spacer"></div>
    <label class="auto" title="Poll the daemon every 5 s">
      <input type="checkbox" id="auto" checked/> auto
    </label>
    <button class="btn" id="btnRefresh" title="Refresh now">↻ Refresh</button>
    <a class="btn ghost" id="btnSelf" target="_blank" rel="noopener"
       title="Open this dashboard in a browser tab">↗ Open</a>
  </header>

  <nav class="tabs" id="tabs">
    <button data-tab="topology" class="active">Topology</button>
    <button data-tab="containers">Containers <span class="n" id="nCont">–</span></button>
    <button data-tab="images">Images <span class="n" id="nImg">–</span></button>
    <button data-tab="networks">Networks</button>
    <button data-tab="volumes">Volumes</button>
    <button data-tab="host">Host &amp; GPU</button>
  </nav>

  <main>
    <div id="banners"></div>

    <div class="view active" id="v-topology">
      <div class="topo-wrap"><svg id="topo" xmlns="http://www.w3.org/2000/svg"></svg></div>
      <div class="legend">
        <span><i></i> live HTTP pipe (flowing = both ends up)</span>
        <span><i class="dash"></i> target stopped / unreachable</span>
        <span><span class="dot" style="background:#a78bfa"></span> GPU ↔ holder (shared card)</span>
        <span><span class="dot running"></span> running</span>
        <span><span class="dot exited"></span> stopped</span>
        <span><span class="dot dead"></span> dead / restarting</span>
        <span class="dim">click a node for detail</span>
      </div>
      <div class="card" style="margin-top:16px">
        <h2>Pipes <span class="dim" id="pipeCount"></span>
          <button class="btn tiny" id="btnPipes" style="margin-left:auto">Test now</button></h2>
        <div class="tbl-wrap"><table id="pipeTable"></table></div>
      </div>
    </div>

    <div class="view" id="v-containers">
      <div class="toolbar">
        <input type="search" id="q" placeholder="filter containers…"/>
        <select id="fLab">
          <option value="">all labs</option>
          <option value="tts">TTS Lab</option>
          <option value="image">Image Lab</option>
          <option value="infra">Infra</option>
          <option value="other">Other</option>
        </select>
        <label class="auto"><input type="checkbox" id="fRun"/> running only</label>
        <span class="spacer"></span>
        <span class="dim" id="contSummary"></span>
        <button class="btn" data-lab="tts" data-act="start" title="Start every TTS container">▶ TTS</button>
        <button class="btn danger" data-lab="tts" data-act="stop" title="Stop every TTS container">■ TTS</button>
        <button class="btn" data-lab="tts" data-act="restart" title="Restart every TTS container">⟳ TTS</button>
      </div>
      <div class="cards" id="grid"></div>
    </div>

    <div class="view" id="v-images">
      <div class="card" style="margin-bottom:14px"><h2>Docker disk usage</h2><div id="disk"></div></div>
      <div class="card">
        <h2>Images <span class="dim" id="imgCount"></span>
          <label class="auto" style="margin-left:auto"><input type="checkbox" id="fLabImg"/> lab images only</label></h2>
        <div class="tbl-wrap"><table id="imgTable"></table></div>
      </div>
    </div>

    <div class="view" id="v-networks">
      <div class="card"><h2>Networks</h2><div class="tbl-wrap"><table id="netTable"></table></div></div>
    </div>

    <div class="view" id="v-volumes">
      <div class="card"><h2>Volumes &amp; binds</h2><div class="tbl-wrap"><table id="volTable"></table></div></div>
    </div>

    <div class="view" id="v-host">
      <div class="grid2">
        <div class="card" id="gpuCard"><h2>GPU</h2><div id="gpuBody"></div></div>
        <div class="card"><h2>Docker daemon</h2><div id="dockBody"></div></div>
        <div class="card"><h2>Host memory &amp; CPU</h2><div id="hostBody"></div></div>
        <div class="card"><h2>Filesystems</h2>
          <!-- must stay a real <table>: injecting thead/tbody into a <div> makes
               the parser drop the table tags and the rows render as a text blob -->
          <div class="tbl-wrap"><table id="diskBody"></table></div>
        </div>
      </div>
      <div class="card" style="margin-top:14px">
        <h2>Bare-metal systemd units <span class="dim">(not containers)</span></h2>
        <div class="tbl-wrap"><table id="unitTable"></table></div>
      </div>
      <div class="card" style="margin-top:14px">
        <h2>VRAM holders
          <span class="dim">— the one resource both labs contend for</span></h2>
        <div class="tbl-wrap"><table id="gpuHolders"></table></div>
      </div>
      <div class="card" style="margin-top:14px">
        <h2>GPU processes <span class="dim">— raw pids, including host strays</span></h2>
        <div class="tbl-wrap"><table id="gpuProc"></table></div>
      </div>
    </div>
  </main>
</div>

<div class="scrim" id="scrim"></div>
<aside class="drawer" id="drawer">
  <div class="dr-head">
    <div style="min-width:0">
      <div class="cc-name" id="drTitle">–</div>
      <div class="cc-img" id="drSub">–</div>
    </div>
    <button class="x" id="drClose" title="Close (Esc)">✕</button>
  </div>
  <div class="dr-tabs" id="drTabs">
    <button data-dt="overview" class="active">Overview</button>
    <button data-dt="logs">Logs</button>
    <button data-dt="env">Env</button>
    <button data-dt="mounts">Mounts</button>
    <button data-dt="net">Networks</button>
    <button data-dt="ports">Ports</button>
  </div>
  <div class="dr-body" id="drBody"></div>
</aside>

<div id="toasts"></div>

<script>
"use strict";
const API = '/infra/api';
const S = { d:null, tab:'topology', dt:'overview', timer:null, logTimer:null,
            sel:null, sig:'', busy:false, disk:null, gpu:null };

/* ── utils ─────────────────────────────────────────────────────────── */
const $  = (id) => document.getElementById(id);
const esc = (s) => String(s === null || s === undefined ? '' : s)
  .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
  .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
const num = (n, d) => (n === null || n === undefined || isNaN(n)) ? (d === undefined ? '–' : d) : n;

function fmtMB(mb){
  if (mb === null || mb === undefined || isNaN(mb)) return '–';
  if (mb >= 1024) return (mb/1024).toFixed(mb >= 10240 ? 0 : 1) + ' GB';
  return Math.round(mb) + ' MB';
}
function fmtDur(s){
  if (!s || s < 0) return '–';
  const d = Math.floor(s/86400), h = Math.floor(s%86400/3600), m = Math.floor(s%3600/60);
  if (d) return d + 'd ' + h + 'h';
  if (h) return h + 'h ' + m + 'm';
  if (m) return m + 'm';
  return Math.floor(s) + 's';
}
function fmtAge(ts){
  if (!ts) return '–';
  const secs = (Date.now()/1000) - (typeof ts === 'number' ? ts : Date.parse(ts)/1000);
  if (isNaN(secs)) return '–';
  return fmtDur(secs);
}
function statusClass(c){
  if (c.paused) return 'paused';
  if (c.restarting) return 'restarting';
  return c.running ? 'running' : (c.status || 'exited');
}
function healthTag(c){
  if (!c.running) return c.oom_killed ? '<span class="tag bad">OOM killed</span>' : '';
  if (c.health === 'unhealthy') return '<span class="tag bad">unhealthy</span>' +
    (c.health_ignored ? '<span class="tag" title="disposable helper - its healthcheck can never pass">ignored</span>' : '');
  if (c.health === 'starting')  return '<span class="tag warn">health: starting</span>';
  if (c.health === 'healthy')   return '<span class="tag ok">healthy</span>';
  return '';
}
function labTag(lab){
  const map = {tts:'TTS Lab', image:'Image Lab', infra:'Infra', other:'other'};
  return '<span class="tag ' + esc(lab) + '">' + esc(map[lab] || lab) + '</span>';
}
async function jfetch(path, opts){
  const r = await fetch(path, opts);
  const t = await r.text();
  let j = null;
  try { j = JSON.parse(t); } catch(e) { j = {error:t.slice(0,300)}; }
  return {ok:r.ok, status:r.status, json:j};
}
function toast(msg, kind){
  const el = document.createElement('div');
  el.className = 'toast ' + (kind || '');
  el.textContent = msg;
  $('toasts').appendChild(el);
  setTimeout(() => el.remove(), 4200);
}

/* ── tabs ──────────────────────────────────────────────────────────── */
function setTab(name){
  S.tab = name;
  document.querySelectorAll('.tabs button').forEach(b =>
    b.classList.toggle('active', b.dataset.tab === name));
  document.querySelectorAll('.view').forEach(v =>
    v.classList.toggle('active', v.id === 'v-' + name));
  // Render the panel we just revealed — waiting for the next poll leaves an
  // empty tab for up to 5 s, which reads as "nothing here" rather than "not
  // drawn yet". Looked up lazily because these are declared further down.
  if (name === 'topology') { renderTopology(true); return; }
  const fn = {containers: renderContainers, images: renderImages,
              networks: renderNetworks, volumes: renderVolumes, host: renderHost}[name];
  if (fn && S.d) fn();
  if (name === 'images') loadDisk();
  if (name === 'host') loadOwnStatus();
}
document.addEventListener('click', (e) => {
  const t = e.target.closest('[data-tab]');
  if (t) setTab(t.dataset.tab);
});

/* ── polling ───────────────────────────────────────────────────────── */
async function refresh(showErr){
  if (S.busy) return;
  S.busy = true;
  try {
    const r = await jfetch(API + '/overview?probe=1');
    if (r.json && r.json.containers) { S.d = r.json; renderAll(); }
    else if (showErr) toast('overview failed: HTTP ' + r.status, 'err');
  } catch(e) {
    if (showErr) toast('overview failed: ' + e.message, 'err');
  } finally { S.busy = false; }
}
function schedule(){
  clearInterval(S.timer);
  if ($('auto').checked) S.timer = setInterval(() => refresh(false), 5000);
}
$('auto').addEventListener('change', schedule);
$('btnRefresh').addEventListener('click', () => refresh(true));
$('btnSelf').href = location.pathname;
$('btnPipes').addEventListener('click', testPipes);

/* ── render ────────────────────────────────────────────────────────── */
function renderAll(){
  const d = S.d; if (!d) return;
  renderPills(d);
  renderBanners(d);
  $('nCont').textContent = d.counts.running + '/' + d.counts.total;
  $('nImg').textContent  = d.counts.images;
  if (S.tab === 'topology') renderTopology(false);
  if (S.tab === 'containers') renderContainers();
  if (S.tab === 'images') renderImages();
  if (S.tab === 'networks') renderNetworks();
  if (S.tab === 'volumes') renderVolumes();
  if (S.tab === 'host') renderHost();
  if (S.sel) renderDrawer();
}

function renderPills(d){
  const av = d.docker || {};
  let h = '';
  if (av.ok) {
    h += '<span class="pill ok"><span class="dot running"></span>docker <b>' + esc(av.version) +
         '</b><span class="dim">api ' + esc(av.api) + '</span></span>';
  } else {
    h += '<span class="pill bad">⚠ docker unavailable</span>';
  }
  h += '<span class="pill"><span class="dot ' + (d.counts.unhealthy ? 'dead' : 'running') +
       '"></span><b>' + d.counts.running + '</b>/' + d.counts.total + ' running</span>';
  h += '<span class="pill">images <b>' + d.counts.images + '</b></span>';
  const g = (d.host && d.host.gpu) || null;
  const ram = d.host && d.host.ram;
  if (ram) {
    const pct = ram.total_mb ? Math.round(ram.used_mb/ram.total_mb*100) : 0;
    h += '<span class="pill">RAM <b>' + pct + '%</b> <span class="dim">' +
         ((ram.used_mb/1024).toFixed(1)) + '/' + ((ram.total_mb/1024).toFixed(0)) + ' GB</span></span>';
  }
  const nh = d.pipes ? d.pipes.filter(p => p.ok).length : 0;
  const np = d.pipes ? d.pipes.length : 0;
  if (np) h += '<span class="pill ' + (nh === np ? 'ok' : '') + '">pipes <b>' + nh + '</b>/' + np +
               ' up</span>';
  h += '<span class="pill dim">updated ' + new Date().toLocaleTimeString() + '</span>';
  $('pills').innerHTML = h;
}

function renderBanners(d){
  const out = [];
  if (d.docker && !d.docker.ok) {
    out.push('<div class="banner">Docker daemon unreachable at <span class="mono">' +
      esc(d.docker.socket || '') + '</span> — ' + esc(d.docker.error || '') +
      '<br/>The orchestrator container needs <span class="mono">-v /var/run/docker.sock:/var/run/docker.sock</span>' +
      ' (it is in docker-compose.yml); the bare-metal image lab needs the service user in the ' +
      '<span class="mono">docker</span> group.</div>');
  }
  if (d.tool && d.tool.readonly) {
    out.push('<div class="banner warn">Read-only mode — INFRA_READONLY=1, start/stop is disabled.</div>');
  }
  const bad = (d.containers || []).filter(c => c.running && c.health === 'unhealthy'
                                               && !c.health_ignored);
  if (bad.length) {
    out.push('<div class="banner warn">Unhealthy: ' +
      bad.map(c => esc(c.name)).join(', ') + '</div>');
  }
  $('banners').innerHTML = out.join('');
}

/* ── topology graph ────────────────────────────────────────────────── */
function topoModel(){
  const d = S.d;
  // rank 0 — host-level resources and bare-metal units
  const units = ((d.host && d.host.units) || []).map(u => ({
    id: 'unit:' + u.id, name: u.label, tier: 'unit', rank: 0,
    isUnit: true, unit: u, running: u.active === 'active',
    status: u.active === 'active' ? 'running' : 'exited',
    health: (u.active === 'active' && u.http && !u.http.ok) ? 'unhealthy' : '',
    lab: u.lab, image: u.unit, routes: [], ports: [{container: u.port}],
    note: (u.enabled ? 'enabled: ' + u.enabled : '')
  }));
  const edges = (d.edges || []).map(e => Object.assign({}, e));
  const g = d.gpu || {};
  if (g.available && (g.holders || []).length) {
    units.push({
      id: 'gpu', name: g.name || 'GPU', tier: 'unit', rank: 0, isGPU: true,
      running: true, status: 'running', health: '', lab: 'infra',
      image: 'shared accelerator', routes: [], ports: [],
      note: fmtMB(g.used_mb) + ' / ' + fmtMB(g.total_mb)
    });
    // the cross-lab coupling: every process parked on the card
    g.holders.forEach(h => edges.push({
      from: 'gpu', to: (h.node || h.name), engines: [], kind: 'gpu', gpu: true,
      label: fmtMB(h.mb), holder: h
    }));
  }
  const nodes = (d.containers || [])
    .filter(c => c.tier !== 'aux')
    .map(c => Object.assign({}, c, {id: c.name}));
  const aux = (d.containers || [])
    .filter(c => c.tier === 'aux')
    .map(c => Object.assign({}, c, {id: c.name}));
  return {nodes: units.concat(nodes, aux), edges: edges};
}
function renderTopology(force){
  if (!S.d) return;
  const model = topoModel();
  const nodes = model.nodes, edges = model.edges;
  const sig = JSON.stringify([
    nodes.map(n => [n.id, n.running, n.health, (n.routes || []).length, n.status]),
    edges.map(e => [e.from, e.to, (e.engines || []).length, e.label || '']),
    (S.d.pipes || []).map(p => [p.target, p.ok]),
    (S.d.gpu || {}).used_mb
  ]);
  if (!force && sig === S.sig) { renderPipeTable(); return; }
  S.sig = sig;

  const byId = {}; nodes.forEach(n => byId[n.id] = n);
  const cols = {};
  nodes.forEach(n => { (cols[n.rank] = cols[n.rank] || []).push(n); });
  const ranks = Object.keys(cols).map(Number).sort((a,b) => a-b);
  const NW = 252, NH = 82, GX = 96, GY = 26, PAD = 26;
  const maxRows = Math.max.apply(null, ranks.map(r => cols[r].length).concat([1]));
  const W = PAD*2 + ranks.length*NW + (ranks.length-1)*GX;
  const H = PAD*2 + maxRows*NH + (maxRows-1)*GY;

  const pos = {};
  ranks.forEach((r, ci) => {
    const col = cols[r];
    const totalH = col.length*NH + (col.length-1)*GY;
    const y0 = PAD + Math.max(0, (H - PAD*2 - totalH)/2);
    col.forEach((n, ri) => {
      pos[n.id] = { x: PAD + ci*(NW+GX), y: y0 + ri*(NH+GY), rank: r };
    });
  });

  const edgeColor = (lab) => lab === 'image' ? '#a78bfa' : (lab === 'infra' ? '#22d3ee' : '#6366f1');
  const statusOf = (id) => { const n = byId[id]; return n ? n.running : false; };
  let svg = '';
  svg += '<defs>';
  svg += '<linearGradient id="gOK" x1="0" y1="0" x2="0" y2="1">' +
         '<stop offset="0" stop-color="#161a25"/><stop offset="1" stop-color="#0f1219"/></linearGradient>';
  svg += '<linearGradient id="gUp" x1="0" y1="0" x2="0" y2="1">' +
         '<stop offset="0" stop-color="#14261f"/><stop offset="1" stop-color="#0f1219"/></linearGradient>';
  svg += '<filter id="soft" x="-30%" y="-30%" width="160%" height="160%">' +
         '<feDropShadow dx="0" dy="6" stdDeviation="8" flood-color="#000" flood-opacity="0.55"/></filter>';
  svg += '</defs>';

  edges.forEach((e, i) => {
    const a = pos[e.from], b = pos[e.to];
    if (!a || !b) return;
    const x1 = a.x + NW, y1 = a.y + NH/2, x2 = b.x, y2 = b.y + NH/2;
    const c = Math.max(40, (x2-x1)*0.5);
    const path = 'M' + x1 + ',' + y1 + ' C' + (x1+c) + ',' + y1 + ' ' + (x2-c) + ',' + y2 + ' ' + x2 + ',' + y2;
    const up = e.gpu ? true : (statusOf(e.from) && statusOf(e.to));
    const tgt = byId[e.to] || {};
    const col = e.gpu ? '#a78bfa' : (up ? edgeColor(tgt.lab) : '#33394d');
    const cls = 'edge' + (e.gpu ? ' gpu' : (up ? ' on' : ' off'));
    svg += '<path class="' + cls + '" data-a="' + esc(e.from) + '" data-b="' +
           esc(e.to) + '" d="' + path + '" stroke="' + col + '" marker-end="url(#arw)"/>';
    const mx = (x1+x2)/2, my = (y1+y2)/2;
    const n = (e.engines || []).length;
    const svcs = (e.services || []).length;
    const label = e.label || (n > 1 ? n + ' engines'
      : (e.engines[0] || (svcs ? (svcs > 1 ? svcs + ' services' : e.services[0]) : 'route')));
    svg += '<g class="edge-lbl" data-a="' + esc(e.from) + '" data-b="' + esc(e.to) + '">' +
           '<rect x="' + (mx - 34) + '" y="' + (my - 22) + '" rx="6" width="68" height="17" ' +
           'fill="#0d1017" stroke="' + col + '" stroke-opacity=".5"/>' +
           '<text x="' + mx + '" y="' + (my - 10) + '" text-anchor="middle" font-size="9.5" ' +
           'font-family="ui-monospace,monospace" fill="' + (up ? '#c9d3e8' : '#6b7387') + '">' +
           esc(String(label).slice(0,13)) + '</text></g>';
  });
  svg += '<defs><marker id="arw" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6"' +
         ' markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#5b6480"/>' +
         '</marker></defs>';

  nodes.forEach((n) => {
    const p = pos[n.id]; if (!p) return;
    const st = statusClass(n);
    const accent = n.isGPU ? '#a78bfa'
      : (n.lab === 'image' ? '#a78bfa' : (n.lab === 'infra' ? '#22d3ee' : '#6366f1'));
    const dotCol = n.isGPU ? '#a78bfa'
      : (st === 'running' ? '#34d399' : (st === 'dead' || st === 'restarting' ? '#f87171' : '#4b5563'));
    const fill = n.isGPU ? '#180f28' : (n.running ? 'url(#gUp)' : 'url(#gOK)');
    svg += '<g class="node-box" data-id="' + esc(n.id) + '">';
    svg += '<rect class="body" x="' + p.x + '" y="' + p.y + '" width="' + NW + '" height="' + NH +
           '" rx="13" fill="' + fill + '" stroke="' + (n.running ? accent : '#2b3244') +
           '" stroke-opacity="' + (n.running ? '.65' : '.9') + '"' +
           (n.isGPU ? ' stroke-dasharray="6 4"' : '') + ' filter="url(#soft)"/>';
    svg += '<rect x="' + p.x + '" y="' + (p.y+16) + '" width="3" height="' + (NH-32) + '" rx="2" fill="' +
           (n.running ? dotCol : '#3f4759') + '"/>';
    svg += '<circle cx="' + (p.x+18) + '" cy="' + (p.y+19) + '" r="4.5" fill="' + dotCol + '"/>';
    svg += '<text x="' + (p.x+30) + '" y="' + (p.y+23) + '" font-size="12.5" font-weight="700" fill="#e7eaf3">' +
           esc(String(n.name).slice(0, 26)) + '</text>';
    const sub = n.isUnit ? (n.unit.unit) : (n.stack || n.image || '');
    svg += '<text x="' + (p.x+16) + '" y="' + (p.y+42) + '" font-size="10" fill="#8b93a8" ' +
           'font-family="ui-monospace,monospace">' + esc(String(sub).slice(0, 34)) + '</text>';
    const badges = (n.lab || '').toUpperCase();
    svg += '<text x="' + (p.x+16) + '" y="' + (p.y+63) + '" font-size="9.5" fill="' +
           accent + '" font-weight="650" letter-spacing=".6">' + esc(badges) + '</text>';
    const routes = (n.routes || []).length;
    const info = n.isUnit ? ('port ' + (n.ports[0] ? n.ports[0].container : '?'))
                          : (routes ? routes + ' engines' : (n.ip || ''));
    svg += '<text x="' + (p.x+NW-14) + '" y="' + (p.y+63) + '" text-anchor="end" font-size="9.5" ' +
           'fill="#8b93a8" font-family="ui-monospace,monospace">' + esc(String(info).slice(0,22)) + '</text>';
    if (n.gpu || n.isGPU) {
      svg += '<text x="' + (p.x+NW-14) + '" y="' + (p.y+20) + '" text-anchor="end" font-size="9.5" ' +
             'fill="#fcd34d" font-weight="650">' + (n.isGPU ? 'SHARED' : 'GPU') + '</text>';
    }
    svg += '</g>';
  });

  const el = $('topo');
  el.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  el.setAttribute('width', W);
  el.setAttribute('height', H);
  el.innerHTML = svg;

  el.querySelectorAll('.node-box').forEach(g => {
    g.addEventListener('click', () => { const n = byId[g.dataset.id]; if (n && !n.isUnit) openDrawer(n.name); });
    g.addEventListener('mouseenter', () => highlight(g.dataset.id, true));
    g.addEventListener('mouseleave', () => highlight(null, false));
  });
  renderPipeTable();
}
function highlight(id, on){
  const el = $('topo');
  if (!el) return;
  // Dim everything that is not the hovered node or one of its pipes.
  el.querySelectorAll('.edge, .edge-lbl').forEach(x => {
    const related = !on || x.dataset.a === id || x.dataset.b === id;
    x.classList.toggle('dimmed', on && !related);
  });
  el.querySelectorAll('.node-box').forEach(x => {
    const related = !on || x.dataset.id === id ||
      !!el.querySelector('.edge[data-a="' + x.dataset.id + '"]:not(.dimmed),' +
                        '.edge[data-b="' + x.dataset.id + '"]:not(.dimmed)');
    x.classList.toggle('dimmed', on && !related);
  });
}

function renderPipeTable(){
  const p = (S.d && S.d.pipes) || [];
  $('pipeCount').textContent = p.length ? '(' + p.filter(x => x.ok).length + '/' + p.length + ' up)' : '';
  if (!p.length) { $('pipeTable').innerHTML = ''; return; }
  let h = '<thead><tr><th>From</th><th></th><th>To</th><th>Address</th><th>Kind</th>' +
          '<th>Engines routed there</th><th>Health</th><th>Latency</th></tr></thead><tbody>';
  p.forEach(x => {
    const addr = x.ip ? (x.ip + ':' + x.port + x.path) : '–';
    const eng = (x.engines || []).length;
    h += '<tr><td class="mono">' + esc(x.target === null ? '' : '—') + '</td>';
    h += '<td>' + (x.ok ? '<span class="dot running"></span>' : '<span class="dot dead"></span>') + '</td>';
    h += '<td><b>' + esc(x.target) + '</b></td>';
    h += '<td class="cell-ip">' + esc(addr) + '</td>';
    h += '<td>' + (x.kind === 'service' ? '<span class="tag">service</span>' : '<span class="tag tts">engine</span>') + '</td>';
    h += '<td><div class="chips">' + (x.engines || []).slice(0, 9).map(e => '<span class="chip">' + esc(e) + '</span>').join('') +
         (eng > 9 ? '<span class="chip more">+' + (eng-9) + '</span>' : '') +
         (x.services || []).map(s => '<span class="chip" title="proxied service, not an engine">' +
           esc(s) + '</span>').join('') + '</div></td>';
    let st = '<span class="tag ok">up</span>';
    if (!x.ok) st = '<span class="tag bad">' + esc(x.reason || ('HTTP ' + (x.code || '?'))) + '</span>';
    else if (x.health && x.health.current_engine) st += ' <span class="tag">' + esc(x.health.current_engine) + '</span>';
    h += '<td>' + st + '</td>';
    h += '<td class="mono">' + (x.ms === null || x.ms === undefined ? '–' : x.ms + ' ms') + '</td></tr>';
  });
  $('pipeTable').innerHTML = h + '</tbody>';
}
async function testPipes(){
  $('btnPipes').textContent = 'testing…';
  try {
    const r = await jfetch(API + '/pipes');
    if (r.json && r.json.pipes) { S.d.pipes = r.json.pipes; renderPipeTable(); refreshPillsOnly(); }
  } catch(e) { toast('pipe probe failed', 'err'); }
  $('btnPipes').textContent = 'Test now';
}
function refreshPillsOnly(){
  if (!S.d) return;
  const np = S.d.pipes.length, nh = S.d.pipes.filter(p => p.ok).length;
  const el = $('pills').querySelector('.pill:last-child');
  if (el) el.textContent = 'updated ' + new Date().toLocaleTimeString();
  toast('pipes: ' + nh + '/' + np + ' up', nh === np ? 'ok' : 'err');
}

/* ── containers ────────────────────────────────────────────────────── */
function visibleContainers(){
  const q = $('q').value.trim().toLowerCase();
  const lab = $('fLab').value;
  const run = $('fRun').checked;
  return (S.d.containers || []).filter(c => {
    if (run && !c.running) return false;
    if (lab && c.lab !== lab) return false;
    if (q) {
      const hay = (c.name + ' ' + c.image + ' ' + c.stack + ' ' + (c.routes||[]).join(' ') +
                   ' ' + (c.ip||'')).toLowerCase();
      if (hay.indexOf(q) < 0) return false;
    }
    return true;
  });
}
function renderContainers(){
  const list = visibleContainers();
  const all = S.d.containers || [];
  $('contSummary').textContent = list.length + ' of ' + all.length + ' containers';
  if (!list.length) { $('grid').innerHTML = '<div class="empty">No containers match.</div>'; return; }
  $('grid').innerHTML = list.map(cardHTML).join('');
}
function cardHTML(c){
  const st = statusClass(c);
  const stt = c.stats || {};
  const cpu = num(stt.cpu_pct, 0), mem = num(stt.mem_mb, 0), lim = num(stt.mem_limit_mb, 0);
  const memPct = lim ? Math.round(mem/lim*100) : 0;
  const ports = (c.ports || []).map(p => p.host ? (p.host + '→' + p.container) : p.container).join(', ');
  const routes = c.routes || [];
  const ip = c.ip || '–';
  const live = c.live || {};
  let engineRows = '';
  if (routes.length) {
    engineRows = '<div class="cc-meta" style="grid-template-columns:1fr"><div style="align-items:flex-start">' +
      '<span class="k">routed</span><span class="chips" style="margin-left:6px">' +
      routes.slice(0,10).map(e => '<span class="chip">' + esc(e) + '</span>').join('') +
      (routes.length > 10 ? '<span class="chip more">+' + (routes.length-10) + '</span>' : '') +
      '</span></div></div>';
  }
  let liveRow = '';
  if (live.current_engine) {
    liveRow = '<div style="font-size:11.5px"><span class="tag ok">in VRAM: ' +
      esc(live.current_engine) + '</span>' + (live.gpu && live.gpu.vram_used
        ? ' <span class="tag">' + fmtMB(live.gpu.vram_used) + '</span>' : '') + '</div>';
  } else if (live.loaded && live.loaded.length) {
    liveRow = '<div class="chips">' + live.loaded.map(e => '<span class="chip">' + esc(e) + '</span>').join('') + '</div>';
  }
  const acts = [];
  if (c.running) {
    acts.push(btn(c, 'stop', '■ Stop', 'danger'));
    acts.push(btn(c, 'restart', '⟳ Restart', ''));
  } else {
    acts.push(btn(c, 'start', '▶ Start', 'accent'));
  }
  acts.push('<button class="btn tiny" data-open="' + esc(c.name) + '">⛭ Detail</button>');
  acts.push('<button class="btn tiny" data-logs="' + esc(c.name) + '">≡ Logs</button>');
  return '<div class="ccard ' + esc(st) + '">' +
    '<div class="cc-head"><span class="dot ' + esc(st) + '" style="margin-top:5px"></span>' +
      '<div style="min-width:0;flex:1">' +
        '<div class="cc-name">' + esc(c.name) + '</div>' +
        '<div class="cc-img">' + esc(c.image) + '</div>' +
      '</div>' + labTag(c.lab) +
      (c.gpu ? '<span class="tag gpu">GPU</span>' : '') +
    '</div>' +
    (c.stack_note ? '<div class="cc-note">' + esc(c.stack_note) + '</div>' : '') +
    '<div class="cc-meta">' +
      '<div><span class="k">state</span><span class="v">' + esc(c.status) + '</span></div>' +
      '<div><span class="k">' + (c.running ? 'uptime' : 'down') + '</span><span class="v">' +
        (c.running ? fmtDur(c.uptime_s) : fmtDur(c.exited_s)) + '</span></div>' +
      '<div><span class="k">ip</span><span class="v cell-ip">' + esc(ip) + '</span></div>' +
      '<div><span class="k">ports</span><span class="v mono">' + esc(ports || '–') + '</span></div>' +
      '<div><span class="k">restarts</span><span class="v">' + num(c.restarts, 0) +
        (c.restart_policy ? ' · ' + esc(c.restart_policy) : '') + '</span></div>' +
      '<div><span class="k">mem</span><span class="v">' + fmtMB(mem) + '</span></div>' +
    '</div>' + engineRows + liveRow +
    '<div class="cc-stats">' +
      statRow('CPU', cpu, 100, cpu.toFixed(1) + ' %') +
      statRow('MEM', memPct, 100, fmtMB(mem) + (lim ? ' / ' + fmtMB(lim) : '')) +
    '</div>' +
    '<div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap">' + healthTag(c) +
      (c.oom_killed ? '<span class="tag bad">oom</span>' : '') +
      (c.protected ? '<span class="tag warn">protected</span>' : '') +
      '<span class="dim mono" style="margin-left:auto">' + esc(c.id) + '</span>' +
    '</div>' +
    '<div class="cc-acts">' + acts.join('') + '</div>' +
  '</div>';
}
function btn(c, act, label, cls){
  const off = (c.protected && act !== 'start') ? ' disabled' : '';
  return '<button class="btn tiny ' + cls + '" data-act="' + act + '" data-name="' + esc(c.name) +
         '"' + off + '>' + label + '</button>';
}
function statRow(label, val, max, text){
  const pct = Math.max(0, Math.min(100, (val/max)*100));
  const cls = pct > 88 ? 'bar bad' : (pct > 70 ? 'bar warn' : 'bar');
  return '<div class="stat-row"><span class="lbl">' + label + '</span>' +
    '<span class="' + cls + '"><i style="width:' + pct.toFixed(1) + '%"></i></span>' +
    '<span class="val">' + esc(text) + '</span></div>';
}
$('q').addEventListener('input', renderContainers);
$('fLab').addEventListener('change', renderContainers);
$('fRun').addEventListener('change', renderContainers);

document.addEventListener('click', (e) => {
  const a = e.target.closest('[data-act][data-name]');
  if (a) { doAction(a.dataset.name, a.dataset.act); return; }
  const l = e.target.closest('[data-lab][data-act]');
  if (l) { doLabAction(l.dataset.lab, l.dataset.act); return; }
  const o = e.target.closest('[data-open]');
  if (o) { openDrawer(o.dataset.open); return; }
  const g = e.target.closest('[data-logs]');
  if (g) { openDrawer(g.dataset.logs, 'logs'); return; }
});

async function doAction(name, act, force){
  const danger = (act === 'stop' || act === 'restart' || act === 'kill');
  if (danger && !confirm(act.toUpperCase() + ' ' + name + '?')) return;
  const r = await jfetch(API + '/containers/' + encodeURIComponent(name) + '/action/' + act +
                         (force ? '?force=1' : ''), {method:'POST'});
  const j = r.json || {};
  if (j.ok) toast(act + ' ' + name + ' ✓', 'ok');
  else toast(act + ' ' + name + ' ✗ ' + (j.error || ('HTTP ' + r.status)), 'err');
  setTimeout(() => refresh(false), 900);
}
async function doLabAction(lab, act){
  if (!confirm(act.toUpperCase() + ' all ' + lab.toUpperCase() + ' containers?')) return;
  const r = await jfetch(API + '/lab/' + lab + '/action/' + act, {method:'POST'});
  const j = r.json || {};
  const n = Object.keys(j.results || {}).length;
  toast(act + ' ' + lab + ': ' + n + ' containers ' + (j.ok ? '✓' : '(some failed)'), j.ok ? 'ok' : 'err');
  setTimeout(() => refresh(false), 1200);
}

/* ── images / networks / volumes ───────────────────────────────────── */
function renderImages(){
  const all = S.d.images || [];
  const only = $('fLabImg').checked;
  const list = only ? all.filter(i => i.lab) : all;
  $('imgCount').textContent = '(' + list.length + ')';
  if (!list.length) { $('imgTable').innerHTML = '<tbody><tr><td class="empty">none</td></tr></tbody>'; return; }
  let h = '<thead><tr><th>Repository:Tag</th><th>ID</th><th>Size</th><th>Created</th>' +
          '<th>Used by</th></tr></thead><tbody>';
  list.forEach(i => {
    h += '<tr><td><b>' + i.tags.map(t => esc(t)).join('<br/>') + '</b>' +
         (i.dangling ? ' <span class="tag warn">dangling</span>' : '') + '</td>' +
         '<td class="mono">' + esc(i.id) + '</td>' +
         '<td class="nowrap">' + fmtMB(i.size_mb) + '</td>' +
         '<td class="nowrap dim">' + fmtAge(i.created) + ' ago</td>' +
         '<td><div class="chips">' + (i.used_by.length
            ? i.used_by.map(u => '<span class="chip">' + esc(u) + '</span>').join('')
            : '<span class="dim">—</span>') + '</div></td></tr>';
  });
  $('imgTable').innerHTML = h + '</tbody>';
}
$('fLabImg').addEventListener('change', renderImages);

async function loadDisk(){
  if (S.disk) { renderDisk(S.disk); return; }
  const r = await jfetch(API + '/disk');
  S.disk = r.json || {};
  renderDisk(S.disk);
}
function renderDisk(d){
  if (!d || d.error) { $('disk').innerHTML = '<div class="dim">' + esc((d && d.error) || 'unavailable') + '</div>'; return; }
  const tiles = [
    ['Image layers', fmtMB(d.layers_mb)],
    ['Volumes', d.volumes_mb >= 0 ? fmtMB(d.volumes_mb) : 'n/a'],
    ['Containers (rw)', fmtMB(d.containers_mb)],
    ['Build cache', fmtMB(d.build_mb)],
    ['Images', d.counts.images], ['Volumes', d.counts.volumes],
  ];
  $('disk').innerHTML = '<div class="stat-tiles">' + tiles.map(t =>
    '<div class="tile"><div class="v">' + esc(t[1]) + '</div><div class="l">' + esc(t[0]) + '</div></div>'
  ).join('') + '</div>';
}

function renderNetworks(){
  const list = S.d.networks || [];
  if (!list.length) { $('netTable').innerHTML = ''; return; }
  let h = '<thead><tr><th>Name</th><th>Driver</th><th>Subnet</th><th>Gateway</th>' +
          '<th>Attached containers</th></tr></thead><tbody>';
  list.forEach(n => {
    h += '<tr><td><b>' + esc(n.name) + '</b>' + (n.internal ? ' <span class="tag">internal</span>' : '') +
         '</td><td>' + esc(n.driver) + '</td>' +
         '<td class="cell-ip">' + esc(n.subnets.join(', ') || '–') + '</td>' +
         '<td class="cell-ip">' + esc(n.gateway || '–') + '</td>' +
         '<td><div class="chips">' + (n.attached.length
            ? n.attached.map(c => '<span class="chip">' + esc(c) + '</span>').join('')
            : '<span class="dim">—</span>') + '</div></td></tr>';
  });
  $('netTable').innerHTML = h + '</tbody>';
}
function renderVolumes(){
  const list = S.d.volumes || [];
  // also surface every bind-mount (the labs bind /opt/* heavily)
  const binds = [];
  (S.d.containers || []).forEach(c => (c.mounts || []).forEach(m => {
    if (m.type === 'bind') binds.push({container:c.name, src:m.source, dest:m.dest, rw:m.rw});
  }));
  let h = '';
  if (list.length) {
    h += '<thead><tr><th>Volume</th><th>Driver</th><th>Mountpoint</th><th>Created</th>' +
         '<th>Used by</th></tr></thead><tbody>';
    list.forEach(v => {
      h += '<tr><td><b>' + esc(v.name) + '</b></td><td>' + esc(v.driver) + '</td>' +
           '<td class="mono dim">' + esc(v.mountpoint) + '</td>' +
           '<td class="dim nowrap">' + esc(String(v.created || '').slice(0,10)) + '</td>' +
           '<td><div class="chips">' + (v.used_by.length
              ? v.used_by.map(u => '<span class="chip">' + esc(u) + '</span>').join('')
              : '<span class="dim">—</span>') + '</div></td></tr>';
    });
    h += '</tbody>';
  }
  if (binds.length) {
    h += '<thead><tr><th colspan="5" style="color:var(--dim)">bind mounts from the host</th></tr>' +
         '<tr><th>Source</th><th>→</th><th>Destination</th><th>Mode</th><th>Container</th></tr></thead><tbody>';
    binds.sort((a,b) => a.src.localeCompare(b.src)).forEach(b => {
      h += '<tr><td class="mono">' + esc(b.src) + '</td><td class="dim">→</td>' +
           '<td class="mono">' + esc(b.dest) + '</td>' +
           '<td>' + (b.rw ? '<span class="tag warn">rw</span>' : '<span class="tag">ro</span>') + '</td>' +
           '<td>' + esc(b.container) + '</td></tr>';
    });
    h += '</tbody>';
  }
  $('volTable').innerHTML = h || '<tbody><tr><td class="empty">none</td></tr></tbody>';
}

/* ── host & GPU ────────────────────────────────────────────────────── */
async function loadOwnStatus(){
  try {
    const r = await jfetch('/status');
    const j = r.json || {};
    S.gpu = j.gpu || null;
    S.ownDevice = j.device || '';
    renderHost();
  } catch(e) { /* endpoint may differ per lab — non-fatal */ }
}
function renderHost(){
  const d = S.d;
  if (!d) return;
  // Normalised view (containers + bare-metal services) is preferred; the
  // process's own /status is the fallback when no container reports GPU.
  const norm = d.gpu && d.gpu.available ? d.gpu : null;
  const raw  = S.gpu || null;
  const gname = norm ? (norm.name || 'GPU') : (raw ? (raw.name || raw.device_name || 'GPU') : '—');
  const tot  = norm ? norm.total_mb : (raw ? (raw.vram_total_mb || raw.vram_total || 0) : 0);
  const used = norm ? norm.used_mb  : (raw ? (raw.vram_used_mb  || raw.vram_used  || 0) : 0);
  const free = norm ? norm.free_mb  : (raw ? (raw.vram_free_mb  || raw.vram_free  || Math.max(0, tot-used)) : 0);
  const holders = (norm && norm.holders) || [];
  const procs = (raw && raw.processes) || [];

  if (!tot) {
    $('gpuBody').innerHTML = '<div class="dim">' + esc(
      (raw && (raw.mode || raw.error)) || 'No GPU telemetry from this process — the orchestrator ' +
      'has no CUDA. Start an engine container and its /health will report the device here.') + '</div>';
  } else {
    const pct = tot ? Math.round(used/tot*100) : 0;
    const fleet = holders.reduce((a,h) => a + (h.mb || 0), 0);
    $('gpuBody').innerHTML =
      '<div class="gpu-big">' +
        '<div><div class="num">' + pct + '%</div><div class="lbl">allocated</div></div>' +
        '<div style="flex:1;min-width:180px">' +
          '<div style="font-size:12.5px;font-weight:650;margin-bottom:6px">' + esc(gname) + '</div>' +
          '<div class="bar ' + (pct > 88 ? 'bad' : pct > 70 ? 'warn' : '') + '" style="height:7px">' +
          '<i style="width:' + pct + '%"></i></div>' +
          '<div class="dim" style="font-size:11.5px;margin-top:6px">' + fmtMB(used) + ' used · ' +
          fmtMB(free) + ' free · ' + fmtMB(tot) + ' total' +
          (raw && raw.source ? ' <span class="dim">(' + esc(raw.source) + ')</span>' : '') + '</div>' +
        '</div>' +
        '<div><div class="num">' + holders.length + '</div><div class="lbl">holders</div></div>' +
        '<div><div class="num">' + fmtMB(fleet) + '</div><div class="lbl">lab fleet</div></div>' +
      '</div>';
  }

  let hh = '<thead><tr><th>Holder</th><th>Kind</th><th>Resident</th><th>Its share</th>' +
           '<th style="width:150px">of card</th></tr></thead><tbody>';
  if (!holders.length) hh += '<tr><td colspan="5" class="empty">nobody is holding VRAM</td></tr>';
  holders.forEach(h => {
    const pct = tot ? Math.max(0, Math.min(100, (h.mb || 0)/tot*100)) : 0;
    hh += '<tr><td><b>' + esc(h.name) + '</b>' + (h.note ? ' <span class="dim">' + esc(h.note) + '</span>' : '') +
          '</td><td>' + (h.kind === 'container'
            ? '<span class="tag tts">container</span>'
            : '<span class="tag image">bare-metal</span>') + '</td>' +
          '<td>' + (h.running ? '<span class="dot running"></span>' : '<span class="dot exited"></span>') +
          ' ' + (h.running ? 'up' : 'down') + '</td>' +
          '<td class="nowrap"><b>' + (h.mb ? fmtMB(h.mb) : '<span class="dim">device-wide only</span>') +
          '</b></td>' +
          '<td><div class="bar"><i style="width:' + pct + '%"></i></div></td></tr>';
  });
  $('gpuHolders').innerHTML = hh + '</tbody>';

  let ph = '<thead><tr><th>PID</th><th>Container</th><th>Process</th><th>VRAM</th></tr></thead><tbody>';
  if (!procs.length) ph += '<tr><td colspan="4" class="empty">no per-process breakdown available</td></tr>';
  procs.slice().sort((a,b) => (b.mb||0)-(a.mb||0)).forEach(p => {
    ph += '<tr><td class="mono">' + esc(p.pid) + '</td>' +
          '<td>' + (p.container ? '<span class="tag tts">' + esc(p.container) + '</span>' :
                    '<span class="tag warn">host / bare-metal</span>') + '</td>' +
          '<td class="mono dim">' + esc(String(p.process || p.process_name || '').slice(0,52)) + '</td>' +
          '<td class="nowrap"><b>' + fmtMB(p.mb) + '</b></td></tr>';
  });
  $('gpuProc').innerHTML = ph + '</tbody>';

  const dk = d.host.docker || {};
  if (dk.error) $('dockBody').innerHTML = '<div class="banner">' + esc(dk.error) + '</div>';
  else $('dockBody').innerHTML = '<div class="stat-tiles">' + [
    ['Server', dk.server], ['Storage driver', dk.storage], ['Cgroup', 'v' + dk.cgroup],
    ['Host', dk.name], ['OS', (dk.os || '').slice(0,28)], ['Kernel', dk.kernel],
    ['Containers', dk.containers + ' (' + dk.running + ' up)'], ['Images', dk.images],
    ['CPUs', dk.ncpu], ['Runtimes', (dk.runtimes || []).join(', ')],
  ].map(t => '<div class="tile"><div class="v" style="font-size:13px">' + esc(num(t[1], '–')) +
    '</div><div class="l">' + esc(t[0]) + '</div></div>').join('') + '</div>' +
    (dk.warnings && dk.warnings.length
      ? '<div class="banner warn" style="margin-top:12px">' + dk.warnings.map(esc).join('<br/>') + '</div>'
      : '');

  const ram = d.host.ram, cpu = d.host.cpu, load = d.host.load;
  let hb = '';
  if (ram) {
    const pct = ram.total_mb ? Math.round(ram.used_mb/ram.total_mb*100) : 0;
    hb += '<div class="gpu-big" style="margin-bottom:14px"><div><div class="num">' + pct +
          '%</div><div class="lbl">RAM</div></div>' +
          '<div style="flex:1;min-width:170px"><div class="bar ' + (pct > 90 ? 'bad' : pct > 75 ? 'warn' : '') +
          '" style="height:7px"><i style="width:' + pct + '%"></i></div>' +
          '<div class="dim" style="font-size:11.5px;margin-top:6px">' + fmtMB(ram.used_mb) + ' used · ' +
          fmtMB(ram.free_mb) + ' free · ' + fmtMB(ram.total_mb) + ' total</div></div></div>';
  }
  hb += '<div class="stat-tiles">' + [
    ['CPU threads', cpu],
    ['Load 1m', load ? load[0].toFixed(2) : '–'],
    ['Load 5m', load ? load[1].toFixed(2) : '–'],
    ['Load 15m', load ? load[2].toFixed(2) : '–'],
  ].map(t => '<div class="tile"><div class="v">' + esc(num(t[1], '–')) + '</div><div class="l">' +
    esc(t[0]) + '</div></div>').join('') + '</div>';
  $('hostBody').innerHTML = hb;

  let dh = '<thead><tr><th>Path</th><th>Used</th><th>Free</th><th>Total</th><th style="width:130px">Fill</th></tr></thead><tbody>';
  (d.host.disks || []).forEach(x => {
    dh += '<tr><td class="mono">' + esc(x.path) + '</td><td>' + x.used_gb + ' GB</td>' +
          '<td>' + x.free_gb + ' GB</td><td class="dim">' + x.total_gb + ' GB</td>' +
          '<td><div class="bar ' + (x.pct > 90 ? 'bad' : x.pct > 75 ? 'warn' : '') + '">' +
          '<i style="width:' + x.pct + '%"></i></div></td></tr>';
  });
  $('diskBody').innerHTML = (d.host.disks || []).length ? dh + '</tbody>' :
    '<div class="dim">no mounts visible</div>';

  let uh = '<thead><tr><th>Unit</th><th>Node</th><th>Port</th><th>Active</th><th>Enabled</th>' +
           '<th>HTTP probe</th></tr></thead><tbody>';
  if (d.host.systemd === false) {
    uh += '<tr><td colspan="6" class="dim">systemctl is not visible from inside this ' +
          'container, so bare-metal unit state cannot be read here — open the Image Lab ' +
          'dashboard (:8002/infra) for real active/enabled/probe values.</td></tr>';
  }
  ((d.host && d.host.units) || []).forEach(u => {
    let hp = '<span class="dim">–</span>';
    if (u.note) hp = '<span class="dim">' + esc(u.note) + '</span>';
    else if (u.http) hp = u.http.ok
      ? '<span class="tag ok">HTTP ' + esc(u.http.code) + '</span> <span class="dim mono">' +
        esc(num(u.http.ms, '?')) + ' ms</span>'
      : '<span class="tag bad">' + esc(u.http.reason || ('HTTP ' + num(u.http.code, '?'))) + '</span>';
    uh += '<tr><td class="mono">' + esc(u.unit) + '</td><td>' + esc(u.label) + '</td>' +
          '<td class="mono">' + esc(u.port) + '</td>' +
          '<td>' + (u.active === 'active' ? '<span class="tag ok">active</span>' :
                    '<span class="tag ' + (u.retired ? '' : 'bad') + '">' + esc(num(u.active, 'n/a')) + '</span>') + '</td>' +
          '<td class="dim">' + esc(num(u.enabled, '–')) + '</td>' +
          '<td>' + hp + '</td></tr>';
  });
  $('unitTable').innerHTML = uh + '</tbody>';
}

/* ── drawer ────────────────────────────────────────────────────────── */
async function openDrawer(name, dt){
  S.sel = name; S.dt = dt || S.dt || 'overview'; S.detail = null;
  $('drawer').classList.add('open'); $('scrim').classList.add('open');
  $('drTitle').textContent = name;
  const c = (S.d.containers || []).find(x => x.name === name);
  $('drSub').textContent = c ? c.image : '';
  setDrawerTab(S.dt);
  const r = await jfetch(API + '/containers/' + encodeURIComponent(name));
  S.detail = r.json || {};
  renderDrawer();
}
function closeDrawer(){
  $('drawer').classList.remove('open'); $('scrim').classList.remove('open');
  clearInterval(S.logTimer); S.logTimer = null; S.sel = null; S.detail = null;
}
$('drClose').addEventListener('click', closeDrawer);
$('scrim').addEventListener('click', closeDrawer);
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeDrawer(); });
$('drTabs').addEventListener('click', (e) => {
  const b = e.target.closest('[data-dt]'); if (!b) return;
  setDrawerTab(b.dataset.dt);
});
function setDrawerTab(dt){
  S.dt = dt;
  document.querySelectorAll('#drTabs button').forEach(b =>
    b.classList.toggle('active', b.dataset.dt === dt));
  clearInterval(S.logTimer); S.logTimer = null;
  renderDrawer();
  if (dt === 'logs') { loadLogs(); S.logTimer = setInterval(loadLogs, 3000); }
}
function renderDrawer(){
  const c = S.detail || {}; if (!S.sel) return;
  const live = ((S.d.containers || []).find(x => x.name === S.sel) || {});
  const body = $('drBody');
  const acts = ['start','stop','restart','pause','unpause'].map(a =>
    '<button class="btn tiny' + ((a==='stop')?' danger':'') + '" data-act="' + a +
    '" data-name="' + esc(S.sel) + '">' + a + '</button>').join('');
  const head = '<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:16px">' + acts +
    '<span class="spacer" style="flex:1"></span>' +
    '<button class="btn tiny" data-act="restart" data-name="' + esc(S.sel) + '">⟳ reload engine</button></div>';

  if (S.dt === 'overview') {
    const st = statusClass(live);
    body.innerHTML = head +
      '<div class="sec"><h3>State</h3><div class="kv">' +
        kv('status', '<span class="dot ' + esc(st) + '"></span> ' + esc(live.status)) +
        kv('health', esc((c.raw_state && c.raw_state.Health && c.raw_state.Health.Status) || live.health || 'none')) +
        kv('uptime', live.running ? fmtDur(live.uptime_s) : '–') +
        kv('restarts', num(live.restarts, 0) + ' · policy ' + esc(live.restart_policy || '–')) +
        kv('exit code', num(live.exit_code, '–') + (live.oom_killed ? ' (OOM killed)' : '')) +
        kv('pid', num(live.pid, '–')) +
        kv('created', esc(String(live.created || '').slice(0,19).replace('T',' '))) +
        kv('container id', '<span class="mono">' + esc(c.id || live.id || '') + '</span>') +
        kv('stack', esc(live.stack_note || live.stack || '–')) +
      '</div></div>' +
      '<div class="sec"><h3>Runtime stats</h3><div class="kv">' +
        kv('cpu', (num(live.stats && live.stats.cpu_pct, 0)).toFixed(1) + ' %') +
        kv('memory', fmtMB(live.stats && live.stats.mem_mb) +
           (live.stats && live.stats.mem_limit_mb ? ' / ' + fmtMB(live.stats.mem_limit_mb) : '')) +
        kv('gpu', live.gpu ? '<span class="tag gpu">requested</span>' : 'none') +
      '</div></div>' +
      '<div class="sec"><h3>Command</h3><div class="mono dim">' +
        esc(live.entrypoint || '') + ' ' + esc(live.cmd || '') + '</div></div>';
    return;
  }
  if (S.dt === 'logs') {
    body.innerHTML = head +
      '<div style="display:flex;gap:8px;align-items:center;margin-bottom:10px">' +
        '<input class="mono" id="logGrep" placeholder="grep…" style="flex:1;background:var(--panel);' +
        'border:1px solid var(--border);color:var(--text);border-radius:8px;padding:6px 10px"/>' +
        '<label class="auto"><input type="checkbox" id="logFollow" checked/> follow</label>' +
        '<button class="btn tiny" id="logReload">↻</button></div>' +
      '<div class="logs" id="logBox">loading…</div>';
    $('logReload').addEventListener('click', loadLogs);
    loadLogs();
    return;
  }
  if (S.dt === 'env') {
    const env = c.env || {};
    const keys = Object.keys(env).sort();
    $('drBody').innerHTML = head + '<div class="sec"><h3>' + keys.length + ' variables</h3>' +
      '<table class="sub-tbl"><tbody>' + keys.map(k => {
        const secret = /TOKEN|SECRET|PASS|KEY|CREDENTIAL|API/i.test(k);
        return '<tr><td class="mono dim" style="width:240px">' + esc(k) + '</td><td class="mono">' +
          (secret ? '<span class="secretval" onclick="this.classList.toggle(\'shown\')">' +
                    esc(env[k]) + '</span>' : esc(env[k])) + '</td></tr>';
      }).join('') + '</tbody></table></div>';
    return;
  }
  if (S.dt === 'mounts') {
    const m = c.mounts || [];
    $('drBody').innerHTML = head + '<div class="sec"><h3>' + m.length + ' mounts</h3>' +
      '<table class="sub-tbl"><thead><tr><th>Source</th><th>Destination</th><th>Mode</th></tr></thead><tbody>' +
      (m.length ? m.map(x => '<tr><td class="mono">' + esc(x.source) + '</td>' +
        '<td class="mono">' + esc(x.dest) + '</td><td>' + (x.rw ? '<span class="tag warn">rw</span>' :
        '<span class="tag">ro</span>') + '</td></tr>').join('')
        : '<tr><td class="empty">none</td></tr>') + '</tbody></table></div>';
    return;
  }
  if (S.dt === 'net') {
    const n = c.networks || [];
    $('drBody').innerHTML = head + '<div class="sec"><h3>Networks</h3>' +
      n.map(x => '<div class="kv" style="margin-bottom:14px">' +
        kv('network', esc(x.name)) + kv('ip', '<span class="cell-ip">' + esc(x.ip || '–') +
          (x.prefix ? '/' + esc(x.prefix) : '') + '</span>') +
        kv('gateway', '<span class="cell-ip">' + esc(x.gateway || '–') + '</span>') +
        kv('mac', '<span class="mono">' + esc(x.mac || '–') + '</span>') +
        kv('dns aliases', '<div class="chips">' + (x.aliases || []).map(a =>
          '<span class="chip">' + esc(a) + '</span>').join('') + '</div>') +
      '</div>').join('') + '</div>';
    return;
  }
  if (S.dt === 'ports') {
    const p = c.ports || [];
    $('drBody').innerHTML = head + '<div class="sec"><h3>Ports</h3>' +
      '<table class="sub-tbl"><thead><tr><th>Container</th><th>Host</th><th>Host IP</th></tr></thead><tbody>' +
      (p.length ? p.map(x => '<tr><td class="mono">' + esc(x.container) + '/' + esc(x.proto) + '</td>' +
        '<td class="mono">' + (x.host ? esc(x.host) : '<span class="dim">not published</span>') + '</td>' +
        '<td class="mono dim">' + esc(x.host_ip || '–') + '</td></tr>').join('')
        : '<tr><td class="empty">none</td></tr>') + '</tbody></table>' +
      '<div class="dim" style="margin-top:12px;font-size:11.5px">Unpublished ports are reachable ' +
      'inside the docker network only — that is why the orchestrator talks to ' +
      '<span class="mono">engine-current:8101</span> by name.</div></div>';
    return;
  }
  body.innerHTML = head;
}
function kv(k, v){
  return '<div class="k">' + esc(k) + '</div><div class="v">' + v + '</div>';
}
async function loadLogs(){
  if (!S.sel) return;
  const box = $('logBox'); if (!box) return;
  const grep = ($('logGrep') && $('logGrep').value || '').trim();
  const r = await jfetch(API + '/containers/' + encodeURIComponent(S.sel) + '/logs?tail=300' +
                         (grep ? '&grep=' + encodeURIComponent(grep) : ''));
  const j = r.json || {};
  if (j.error) { box.textContent = 'error: ' + j.error; return; }
  const atBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 40;
  box.innerHTML = (j.lines || []).map(l => {
    const cls = /\b(error|exception|traceback|failed|fatal)\b/i.test(l) ? 'lv-err'
              : /\b(warn|warning)\b/i.test(l) ? 'lv-warn' : '';
    return '<span class="' + cls + '">' + esc(l) + '</span>';
  }).join('\n');
  const follow = $('logFollow');
  if (!follow || follow.checked || atBottom) box.scrollTop = box.scrollHeight;
}
$('drBody').addEventListener('input', (e) => { if (e.target.id === 'logGrep') loadLogs(); });

/* ── boot ──────────────────────────────────────────────────────────── */
refresh(true);
loadOwnStatus();
setInterval(loadOwnStatus, 10000);
schedule();
</script>
</body>
</html>
"""


def build_infra_page() -> str:
    """The dashboard HTML (static — the JS derives every URL from location)."""
    return INFRA_HTML
