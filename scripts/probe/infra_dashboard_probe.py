#!/usr/bin/env python3
"""
infra_dashboard_probe.py — offline verification for the shared /infra dashboard.

Two things it proves, neither of which needs the VM:

  1. STRUCTURE — the graph the UI builds from /infra/api/overview is internally
     consistent: every edge endpoint is a node that actually exists. This is a
     real failure mode, not a theoretical one: bare-metal unit nodes are keyed
     ``unit:<id>`` while a holder's display label is its human name, so an edge
     built from the label points at nothing and is silently dropped by the SVG
     renderer (the GPU↔Image-Lab link vanished exactly this way).

  2. RENDERING — writes a self-contained HTML that mocks ``fetch`` with the
     synthetic payload and prepends a ``window.onerror`` collector, so the page
     can be opened in a browser and any JS runtime error shows up as a red
     banner instead of dying silently in the console.

Usage:
    python scripts/probe/infra_dashboard_probe.py            # writes + checks
    python scripts/probe/infra_dashboard_probe.py --open     # also opens it

Exits non-zero when a structural check fails, so it can be wired into CI
(note: ``return 1`` at script scope does NOT set an exit code — see the
PowerShell footguns in the ops notes; this uses ``sys.exit``).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import tempfile
import types
import webbrowser

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# lab_infra imports FastAPI at module scope; stub it when it is not installed
# (a plain dev venv) so the pure graph helpers stay importable and testable.
try:
    import fastapi  # noqa: F401
except ModuleNotFoundError:                       # pragma: no cover
    _fa = types.ModuleType("fastapi")
    _fa.APIRouter = lambda *a, **k: types.SimpleNamespace(
        get=lambda *a, **k: (lambda f: f), post=lambda *a, **k: (lambda f: f))
    _res = types.ModuleType("fastapi.responses")
    _res.HTMLResponse = _res.JSONResponse = lambda *a, **k: None
    sys.modules["fastapi"] = _fa
    sys.modules["fastapi.responses"] = _res

import lab_infra as L                                                    # noqa: E402
import lab_infra_ui as U                                                 # noqa: E402

ENGINES = ["piper", "kokoro", "melo", "chatt", "outetts", "bark", "styletts2",
           "f5tts", "dia", "xtts", "cosyvoice", "fishspeech", "chatterbox",
           "chatterboxturbo", "omnivoice", "openvoice", "zonos", "matcha",
           "manatts", "mmsfas", "csm"]

DEVICE = {"name": "RTX 5060 Ti", "vram_total": 16311, "vram_used": 9130,
          "vram_free": 7181, "proc_allocated_mb": 7800, "proc_reserved_mb": 8100}

# (name, env, stack, ip, gpu?, running?, aliases, health, restarts, published ports)
FLEET = [
    ("tts-lab-orchestrator", "orch", "orchestrator", "172.18.0.2", False, True,
     ("orchestrator",), "healthy", 0, [8009]),
    ("tts-lab-engine-current", {}, "engine-current", "172.18.0.3", True, True,
     ("engine-current",), "healthy", 0, []),
    ("tts-lab-engine-qwen", {}, "engine-qwen", "172.18.0.4", True, True,
     ("engine-qwen",), "healthy", 2, []),
    ("tts-lab-engine-mid", {}, "engine-mid", "172.18.0.5", True, False,
     ("engine-mid",), "", 0, []),
    # stopped, but its LAST health status was 'unhealthy'. Docker keeps that
    # status on exit, which is why the unhealthy counter must ignore it.
    ("tts-lab-engine-legacy", {}, "engine-legacy", "172.18.0.6", False, False,
     ("engine-legacy",), "unhealthy", 5, []),
    ("tts-lab-engine-editx", {}, "engine-editx", "172.18.0.7", True, True,
     ("engine-editx",), "unhealthy", 1, []),
    ("tts-lab-engine-fa", {}, "engine-fa", "172.18.0.8", True, True,
     ("engine-fa",), "healthy", 0, []),
    ("tts-lab-s2pro", {}, "sglang-omni", "172.18.0.9", True, True,
     ("s2pro",), "healthy", 0, [8005]),
    ("tts-lab-vibevoice", {}, "sglang", "172.18.0.10", True, False,
     ("vibevoice",), "", 0, [8003]),
    ("tts-lab-higgs", {}, "sglang", "172.18.0.11", True, False,
     ("higgs",), "", 0, [8004]),
    ("tts-lab-orpheus", {}, "orpheus", "172.18.0.12", True, False,
     ("orpheus",), "", 9, [8002]),
    ("tts-lab-gpu-probe", {}, "engine-editx", "172.18.0.13", True, True,
     (), "unhealthy", 0, []),
]

# Units as the dashboard sees them (systemctl + a localhost /status ping whose
# JSON carries gpu + vram — this is how the bare-metal Image Lab appears).
UNITS = [
    {"id": "imglab", "label": "Image Lab", "unit": "arthur-imglab.service",
     "port": 8002, "lab": "image", "path": "/status", "retired": False,
     "active": "active", "enabled": "enabled",
     "http": {"ok": True, "code": 200, "ms": 12.0, "device": "remote",
              "gpu": {"available": True, "name": "RTX 5060 Ti",
                      "vram_total_mb": 16311, "vram_used_mb": 9130,
                      "vram_free_mb": 7181, "source": "nvidia-smi"},
              "vram": {"available": True, "total_gb": 15.9, "reserved_gb": 4.4}}},
    {"id": "comfy", "label": "ComfyUI sidecar", "unit": "arthur-comfy.service",
     "port": 8188, "lab": "image", "path": "/system_stats", "retired": False,
     "active": "active", "enabled": "enabled",
     "http": {"ok": True, "code": 200, "ms": 4.0}},
    {"id": "ttslab", "label": "TTS bare-metal (retired)", "unit": "arthur-lab.service",
     "port": 8001, "lab": "tts", "path": "/status", "retired": True,
     "active": "inactive", "enabled": "masked"},
    {"id": "arthur", "label": "arthur.service (disabled)", "unit": "arthur.service",
     "port": 8000, "lab": "tts", "path": "/health", "retired": True,
     "active": "failed", "enabled": "disabled"},
]


def _inspect(name, env, stack, ip, gpu, running, aliases, health, restarts, ports):
    return {
        "Id": "sha256:" + name.ljust(16, "0") + "0" * 48,
        "Name": "/" + name,
        "Created": "2026-09-01T10:00:00.000000000Z",
        "Image": "sha256:" + stack.replace("-", "").ljust(12, "0")[:12] + "0" * 52,
        "Config": {"Image": f"tts-lab-{stack}:latest",
                   "Env": [f"{k}={v}" for k, v in (env or {}).items()],
                   "Labels": {"com.docker.compose.project": "tts-lab",
                              "com.docker.compose.service": name.replace("tts-lab-", "")},
                   "Cmd": ["uvicorn", "tts_lab_engine_server:app"],
                   "Healthcheck": {"Test": ["CMD"]}},
        "State": {"Status": "running" if running else "exited", "Running": running,
                  "Paused": False, "Restarting": False,
                  "ExitCode": 0 if running else 137, "OOMKilled": not running,
                  "Pid": 100 if running else 0, "Health": {"Status": health},
                  "StartedAt": "2026-09-14T06:00:00.000000000Z",
                  "FinishedAt": "2026-09-14T09:00:00.000000000Z"},
        "HostConfig": {"RestartPolicy": {"Name": "unless-stopped"},
                       "DeviceRequests": ([{"Driver": "nvidia", "Count": -1,
                                            "Capabilities": [["gpu"]]}] if gpu else None)},
        "Mounts": [{"Type": "bind", "Source": "/opt/models",
                    "Destination": "/opt/models", "RW": True},
                   {"Type": "volume", "Source": "tts_lab_data",
                    "Destination": "/data", "RW": False}],
        "RestartCount": restarts,
        "NetworkSettings": {
            "Ports": {f"{p}/tcp": [{"HostIp": "0.0.0.0", "HostPort": str(p)}]
                      for p in (ports or [])},
            "Networks": {"tts-lab-net": {"IPAddress": ip, "IPPrefixLen": 16,
                                         "Gateway": "172.18.0.1",
                                         "MacAddress": "02:42:ac:12:00:02",
                                         "Aliases": list(aliases)}},
        },
    }


def build_payload() -> dict:
    """A payload shaped exactly like GET /infra/api/overview."""
    orch_env = {f"{e.upper()}_URL": "http://engine-current:8101" for e in ENGINES}
    orch_env.update({
        "QWEN3TTS_URL": "http://engine-qwen:8104",
        "VIBEVOICE_URL": "http://engine-mid:8103",
        "HIGGS_URL": "http://engine-mid:8103",
        "S2PRO_SGLANG_URL": "http://s2pro:8000/v1/audio/speech",
        "EDITX_URL": "http://engine-editx:8105",
        "XTTSFA_URL": "http://engine-fa:8106",
        "INDEXTTS_URL": "http://engine-legacy:8102",
        "PARLER_URL": "http://engine-legacy:8102",
        "VOICE_LIB_URL": "http://engine-current:8101",
        "HF_TOKEN": "hf_example_secret_value",
    })
    containers = []
    for row in FLEET:
        env = orch_env if row[1] == "orch" else row[1]
        c = L._summarise(row[0], _inspect(row[0], env, row[2], row[3], row[4],
                                          row[5], row[6], row[7], row[8], row[9]),
                         None, {})
        if c["running"] and c["gpu"]:
            c["live"] = {"current_engine": "chatterbox", "loaded": ["chatterbox"],
                         "gpu": DEVICE}
        containers.append(c)

    topo = L.build_topology(containers)
    by_name = {c["name"]: c for c in containers}
    pipes = [
        {"target": "tts-lab-engine-current", "port": 8101, "path": "", "ip": "172.18.0.3",
         "kind": "engine", "vars": ["PIPER_URL"], "engines": ENGINES, "ok": True,
         "code": 200, "ms": 1.9,
         "health": {"current_engine": "chatterbox", "loaded": ["chatterbox"], "gpu": DEVICE}},
        {"target": "tts-lab-s2pro", "port": 8000, "path": "/v1/audio/speech",
         "ip": "172.18.0.9", "kind": "engine", "vars": ["S2PRO_SGLANG_URL"],
         "engines": ["s2pro"], "ok": True, "code": 200, "ms": 3.1,
         "health": {"current_engine": None, "loaded": [], "sglang_up": True}},
        {"target": "tts-lab-engine-mid", "port": 8103, "path": "", "ip": "172.18.0.5",
         "kind": "engine", "vars": ["VIBEVOICE_URL"], "engines": ["vibevoice", "higgs"],
         "ok": False, "ms": None, "reason": "target not running"},
    ]
    for p in pipes:
        node = by_name.get(p["target"])
        if node is not None and p.get("health"):
            node["live"] = {"current_engine": p["health"].get("current_engine"),
                            "loaded": p["health"].get("loaded", []),
                            "gpu": p["health"].get("gpu"),
                            "sglang_up": p["health"].get("sglang_up")}
    images = [
        {"id": "a1b2c3d4e5f6", "tags": ["tts-lab-engine-current:latest"], "lab": True,
         "size_mb": 9840.5, "shared_mb": 210.0, "created": 1789000000.0,
         "dangling": False, "used_by": ["tts-lab-engine-current"], "containers": 1},
        {"id": "0f0f0f0f0f0f", "tags": ["<none> (0f0f0f0f0f0f)"], "lab": False,
         "size_mb": 142.0, "shared_mb": 0, "created": 1785000000.0,
         "dangling": True, "used_by": [], "containers": -1},
    ]
    networks = [
        {"id": "n1", "name": "tts-lab-net", "driver": "bridge", "scope": "local",
         "internal": False, "subnets": ["172.18.0.0/16"], "gateway": "172.18.0.1",
         "created": "2026-08-01", "attached": [c["name"] for c in containers],
         "containers": len(containers)},
        {"id": "n2", "name": "bridge", "driver": "bridge", "scope": "local",
         "internal": False, "subnets": ["172.17.0.0/16"], "gateway": "172.17.0.1",
         "created": "2026-06-01", "attached": [], "containers": 0},
    ]
    volumes = [{"name": "tts_lab_data", "driver": "local",
                "mountpoint": "/var/lib/docker/volumes/tts_lab_data/_data",
                "created": "2026-08-01T00:00:00Z", "scope": "local",
                "used_by": ["tts-lab-engine-current:/data"]}]
    return {
        "docker": {"ok": True, "version": "28.3.2", "api": "1.49", "min_api": "1.24",
                   "os": "linux", "arch": "amd64", "socket": "/var/run/docker.sock"},
        "containers": containers,
        "edges": topo["edges"],
        "routes": topo["routes"],
        "pipes": pipes,
        "images": images,
        "networks": networks,
        "volumes": volumes,
        "gpu": L.gpu_summary(containers, UNITS, None),
        "labs": L.LABS,
        "host": {"cpu": 16, "ram": {"total_mb": 64300, "used_mb": 41200, "free_mb": 23100},
                 "load": [1.42, 1.10, 0.88],
                 "systemd": True,
                 "disks": [{"path": "/", "total_gb": 240.0, "used_gb": 180.2,
                            "free_gb": 59.8, "pct": 75.1},
                           {"path": "/opt/models", "total_gb": 900.0, "used_gb": 650.4,
                            "free_gb": 249.6, "pct": 72.3}],
                 "docker": {"name": "arthur-vm", "server": "28.3.2",
                            "os": "Ubuntu 22.04.5 LTS", "kernel": "6.8.0-45-generic",
                            "storage": "overlay2", "root": "/var/lib/docker", "cgroup": "2",
                            "containers": len(containers),
                            "running": sum(1 for c in containers if c["running"]),
                            "paused": 0, "stopped": 0, "images": 14, "ncpu": 16,
                            "mem_total_mb": 64300, "runtimes": ["nvidia", "runc"],
                            "warnings": []},
                 "units": UNITS},
        "counts": L.counts_of(containers, images, volumes),
        "tool": {"readonly": False, "allow_all": False, "pipes": True},
        "ts": 1789000000.0,
    }


def graph_ids(payload: dict) -> set[str]:
    """Node ids the UI can render — mirrors topoModel() in lab_infra_ui.py."""
    ids = {"unit:" + u["id"] for u in payload["host"]["units"]}
    if (payload.get("gpu") or {}).get("available"):
        ids.add("gpu")
    ids |= {c["name"] for c in payload["containers"]}
    return ids


def check(payload: dict, fixture: bool = False) -> list[str]:
    """Structural assertions on a payload. Returns failures (empty = pass).

    ``fixture=True`` adds guards that assert the *synthetic fixture* still
    exercises the cases these checks exist for. Keep them out of the default
    path: the same function is used to validate a payload pulled from a live
    lab, where a guard like "the disposable helper must currently be
    unhealthy" is simply not true at that moment and would read as a failure.
    """
    fails: list[str] = []
    ids = graph_ids(payload)

    for e in payload["edges"]:
        if e["from"] not in ids:
            fails.append(f"edge {e['from']} -> {e['to']}: source is not a node")
        if e["to"] not in ids:
            fails.append(f"edge {e['from']} -> {e['to']}: target is not a node")
        # An edge's engine list and the endpoints' routes lists come from the
        # same env scan, so they must agree. They did not: a substring test
        # classified CosyVoice/OmniVoice/OpenVoice/VibeVoice as services, which
        # dropped them from the node counters while the flows table kept them.
        eng = e.get("engines") or []
        if len(eng) != len(set(eng)):
            fails.append(f"edge {e['from']} -> {e['to']}: duplicate engine entries")
        for endpoint in (e["from"], e["to"]):
            missing = sorted(set(eng) - set(payload["routes"].get(endpoint, [])))
            if missing:
                fails.append(f"edge {e['from']} -> {e['to']}: {missing} counted on the "
                             f"edge but absent from routes[{endpoint}]")
        for svc in e.get("services") or []:
            if svc in payload["routes"].get(e["to"], []):
                fails.append(f"edge {e['from']} -> {e['to']}: {svc!r} is a service but "
                             f"is counted as a routed engine")

    for h in (payload.get("gpu") or {}).get("holders", []):
        if (h.get("node") or h["name"]) not in ids:
            fails.append(f"gpu holder {h['name']!r} has no node id ({h.get('node')!r})")

    for c in payload["containers"]:
        if c["ip"] and not c["networks"]:
            fails.append(f"{c['name']}: ip without a network")
        if c["tier"] not in ("service", "engine", "aux"):
            fails.append(f"{c['name']}: unexpected tier {c['tier']!r}")

    # Header counter: an exited container keeps its last health status, so it
    # must not be reported as failing now. The fixture is asserted to actually
    # contain such a container, otherwise this check would quietly test nothing.
    live_unhealthy = [c["name"] for c in payload["containers"]
                      if c["running"] and c["health"] == "unhealthy"
                      and not c.get("health_ignored")]
    dead_unhealthy = [c["name"] for c in payload["containers"]
                      if not c["running"] and c["health"] == "unhealthy"]
    ignored_helpers = [c["name"] for c in payload["containers"]
                       if c["running"] and c["health"] == "unhealthy"
                       and c.get("health_ignored")]
    if fixture and not dead_unhealthy:
        fails.append("fixture problem: no stopped-but-unhealthy container, so the "
                     "unhealthy counter is not exercised")
    if fixture and not ignored_helpers:
        fails.append("fixture problem: no disposable helper reporting unhealthy, so "
                     "the health_ignored path is not exercised")
    if payload["counts"]["unhealthy"] != len(live_unhealthy):
        fails.append(f"counts.unhealthy={payload['counts']['unhealthy']} but only "
                     f"{len(live_unhealthy)} running container(s) are genuinely "
                     f"unhealthy ({', '.join(dead_unhealthy + ignored_helpers)} must be "
                     f"ignored)")
    if payload["counts"]["volumes"] != len(payload["volumes"]):
        fails.append("counts.volumes disagrees with the volumes list")
    if payload["counts"]["images"] != len(payload["images"]):
        fails.append("counts.images disagrees with the images list")

    # Guard the guard: the substring trap only bites for engines whose env var
    # name contains "VOICE", so the fixture must route at least one of them.
    routed = {en for e in payload["edges"] for en in (e.get("engines") or [])}
    if fixture and not routed & {"cosyvoice", "omnivoice", "openvoice", "vibevoice"}:
        fails.append("fixture problem: no *VOICE*-named engine is routed, so the "
                     "route-classification trap is not exercised")
    return fails


def _function_bodies(js: str):
    """[(name, body)] for top-level ``function name(...) { ... }`` by brace match.

    Scoping matters: two renderers both use a local ``h``, one for a table and
    one for header pills. Tracking variable names globally flags the pills
    assignment as table markup — a false positive, and a check that cries wolf
    gets ignored. Brace counting is safe here because the JS uses string
    concatenation only (no braces inside string literals).
    """
    import re
    out = []
    for m in re.finditer(r"function\s+([A-Za-z0-9_$]+)\s*\([^)]*\)\s*\{", js):
        depth, i = 1, m.end()
        while i < len(js) and depth:
            c = js[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            i += 1
        out.append((m.group(1), js[m.end():i - 1]))
    return out


def check_markup() -> list[str]:
    """DOM-level checks on the static page, no browser required.

    Catches two defects, both real during development:
      * table markup assigned to an element that is NOT a ``<table>`` — the
        HTML parser drops the table tags and the rows collapse into an
        unformatted text blob (this shipped once for the Filesystems card);
      * a ``$('x')`` reference with no matching element id (silent throw).

    Renderers build rows in a local first (``let dh = '<thead…'``) and assign
    later, so the local has to be tracked — matching only the text right after
    ``innerHTML =`` misses that case entirely.
    """
    import re
    html = U.INFRA_HTML
    js = "\n".join(re.findall(r"<script>(.*?)</script>", html, re.S))
    ids = set(re.findall(r'id="([A-Za-z0-9_\-]+)"', html))
    table_ids = set(re.findall(r'<table id="([A-Za-z0-9_\-]+)"', html))
    fails: list[str] = []

    def scan(code: str) -> None:
        """Flag assignments that inject table rows into a non-table element."""
        thead_vars = set(re.findall(r"(?:let|var|const)\s+([A-Za-z0-9_$]+)\s*=\s*'<thead", code))
        for m in re.finditer(r"\$\('([A-Za-z0-9_\-]+)'\)\.innerHTML\s*=", code):
            target = m.group(1)
            rest = code[m.end():]
            stop = re.search(r";\s*\n", rest)        # this statement only
            rhs = rest[:stop.start() if stop else 400]
            carries = "<thead" in rhs or any(
                re.search(r"\b" + re.escape(v) + r"\b", rhs) for v in thead_vars)
            if carries and "<table" not in rhs and target not in table_ids:
                fails.append(f"#{target} receives table rows but is not a <table> "
                             f"(they render as a text blob)")

    for _name, body in _function_bodies(js):
        scan(body)

    for ref in set(re.findall(r"\$\('([A-Za-z0-9_\-]+)'\)", js)):
        if ref not in ids:
            fails.append(f"JS references #{ref} which no element defines")
    return fails


def write_probe_html(payload: dict, out: pathlib.Path) -> pathlib.Path:
    """Dashboard HTML with fetch mocked and a visible JS-error collector.

    The mock routes per endpoint rather than answering everything with the
    overview blob — otherwise the drawer's Env/Mounts/Logs tabs render empty
    and the probe would "pass" while those panels are untested.
    """
    mock = (
        "<script>(function(){var P=%s;"
        "function byPath(u){"
        "  if(String(u).indexOf('/status')>=0) return {device:'remote',gpu:{name:'RTX 5060 Ti',"
        "    vram_total_mb:16311,vram_used_mb:9130,vram_free_mb:7181,source:'nvidia-smi',"
        "    processes:[{pid:4411,mb:8100,process:'python',container:'tts-lab-engine-current'},"
        "    {pid:992,mb:4400,process:'python3.11',container:''}]}};"
        "  if(String(u).indexOf('/logs')>=0) return {lines:["
        "    '2026-09-14T06:00:01Z INFO  uvicorn running on http://0.0.0.0:8009',"
        "    '2026-09-14T06:00:04Z WARNING transformers: ROPE_INIT_FUNCTIONS missing',"
        "    '2026-09-14T06:00:09Z ERROR engine load failed: CUDA out of memory'],"
        "    count:3, name:'x'};"
        "  if(String(u).indexOf('/disk')>=0) return {layers_mb:18420.5,volumes_mb:120.0,"
        "    containers_mb:64.5,build_mb:0,counts:{images:14,volumes:3,containers:12}};"
        "  if(String(u).indexOf('/pipes')>=0) return {pipes:P.pipes};"
        "  var m=String(u).match(/containers\\/([^\\/?]+)/);"
        "  if(m){for(var i=0;i<P.containers.length;i++)"
        "    if(P.containers[i].name===decodeURIComponent(m[1])) return P.containers[i];"
        "    return {error:'no such container'};}"
        "  return P;}"
        "window.fetch=function(u){"
        "return Promise.resolve({ok:true,status:200,"
        "  text:function(){return Promise.resolve(JSON.stringify(byPath(u)));}});};"
        "window.addEventListener('error',function(e){"
        "var d=document.getElementById('__err');"
        "if(!d){d=document.createElement('pre');d.id='__err';"
        "d.style.cssText='position:fixed;bottom:0;left:0;right:0;z-index:9999;"
        "background:#7f1d1d;color:#fff;font:12px monospace;padding:8px;"
        "max-height:30vh;overflow:auto';document.body.appendChild(d);}"
        "d.textContent+='JS ERROR: '+e.message+' @'+e.lineno+'\\n';});})();</script>"
    ) % json.dumps(payload)
    out.write_text(U.INFRA_HTML.replace("</head>", mock + "</head>", 1), encoding="utf-8")
    return out


def main() -> int:
    # stdout may be cp1252 (Windows PowerShell) where printing the findings below
    # raises UnicodeEncodeError -- i.e. the probe would crash exactly while
    # reporting a failure. Force UTF-8 and keep the printed text ASCII anyway
    # (the docstring is printed by --help, so this covers that path too).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--open", action="store_true", help="open the probe in a browser")
    ap.add_argument("--out", default=os.path.join(tempfile.gettempdir(),
                                                  "infra_probe.html"))
    ap.add_argument("--live", metavar="URL", default="",
                    help="also validate a payload fetched from a running lab, e.g. "
                         "http://192.168.0.87:8009 (structural checks only)")
    args = ap.parse_args()

    payload = build_payload()
    fails = check(payload, fixture=True) + check_markup()

    print("fleet:      %d containers, %d running"
          % (len(payload["containers"]), payload["counts"]["running"]))
    print("topology:   %d edges, %d nodes" % (len(payload["edges"]), len(graph_ids(payload))))
    print("pipes:      %d (%d up)" % (len(payload["pipes"]),
                                      sum(1 for p in payload["pipes"] if p["ok"])))
    gpu = payload["gpu"]
    print("gpu:        %s %d/%d MB, holders=%s"
          % (gpu.get("name") or "n/a", gpu.get("used_mb", 0), gpu.get("total_mb", 0),
             ", ".join("%s(%s)" % (h["name"], h["mb"]) for h in gpu.get("holders", []))))

    # Optional: the same structural checks against a running lab. This is what
    # found the route-classification and unhealthy-counter bugs — synthetic data
    # cannot invent the env names and health statuses a live daemon reports.
    if args.live:
        import urllib.request
        url = args.live.rstrip("/") + "/infra/api/overview"
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                live = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"\nlive {url}: FETCH FAILED ({type(e).__name__}: {e})")
            return 1
        lg = live.get("gpu") or {}
        print(f"\nlive {url}")
        print("  counts:  %s" % (live.get("counts"),))
        print("  edges:   %s" % [(e["to"], len(e.get("engines") or []),
                                  e.get("services") or []) for e in live.get("edges", [])])
        print("  gpu:     %s %s/%s MB holders=%s"
              % (lg.get("name") or "n/a", lg.get("used_mb"), lg.get("total_mb"),
                 [h["name"] for h in lg.get("holders", [])]))
        lfails = check(live)
        print("  STRUCTURE:", "OK - consistent with the live daemon" if not lfails else "")
        for f in lfails:
            print("     x %s" % f)
        fails += ["live: " + f for f in lfails]

    print()
    if fails:
        print("STRUCTURE FAILED (%d):" % len(fails))
        for f in fails:
            print("  x " + f)
    else:
        print("structure OK - every edge endpoint, GPU holder, table target and "
              "element reference resolves")

    path = write_probe_html(payload, pathlib.Path(args.out))
    print("\nrender probe: %s" % path)
    print("  open it and look for a red JS-error banner at the bottom")
    if args.open:
        webbrowser.open(path.as_uri())
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
