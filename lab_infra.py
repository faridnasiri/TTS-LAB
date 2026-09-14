"""
lab_infra.py — Shared container/infrastructure dashboard for both labs.

Mounted into BOTH apps so the same page is reachable from either:
    TTS Lab orchestrator  :8009/infra
    Image Lab             :8002/infra

Talks to the Docker Engine API directly over the unix socket (httpx +
HTTPTransport(uds=...)) — no docker CLI, no docker python package. That works
identically in the orchestrator container (compose mounts /var/run/docker.sock)
and in the bare-metal image-lab service (runs as root).

What it exposes
    GET  /infra                              dashboard HTML
    GET  /infra/api/overview                 containers + topology + pipes + host
    GET  /infra/api/containers/{name}        full inspect summary (env/mounts/net)
    GET  /infra/api/containers/{name}/logs   tailed log text (demuxed)
    GET  /infra/api/containers/{name}/stats  one-shot CPU/mem sample
    POST /infra/api/containers/{name}/action/{act}   start|stop|restart|pause|
                                                     unpause|kill
    GET  /infra/api/images                   image list + who runs them
    GET  /infra/api/networks                 networks + attached containers
    GET  /infra/api/volumes                  volumes + consumers
    GET  /infra/api/host                     docker info, GPU, RAM, disks, units
    POST /infra/api/lab/{lab}/action/{act}   bulk action over a lab's containers
    GET  /infra/api/disk                     /system/df (slow, 60 s cached)

Safety
    * Actions are limited to containers named ``tts-lab-*`` unless
      ``INFRA_ALLOW_ALL=1``.
    * ``tts-lab-gpu-probe`` is protected (it is an AutoRemove helper the TTS
      dispatch layer recreates on demand).
    * Stopping/killing the container that *serves* this page and the lab it
      is mounted in is refused unless ``?force=1`` — otherwise the UI would
      kill its own transport mid-request.
    * ``INFRA_READONLY=1`` disables every mutating endpoint.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse

log = logging.getLogger("lab_infra")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

_DOCKER_SOCK  = os.environ.get("INFRA_DOCKER_SOCK", "/var/run/docker.sock")
_DOCKER_HOST  = os.environ.get("DOCKER_HOST", "")
_API_VER      = os.environ.get("INFRA_DOCKER_API", "v1.49")
_CACHE_TTL    = float(os.environ.get("INFRA_CACHE_TTL", "6.0"))   # >= the UI poll (5 s)
_STATS_TTL    = float(os.environ.get("INFRA_STATS_TTL", "10.0"))
_IMAGES_TTL   = float(os.environ.get("INFRA_IMAGES_TTL", "15.0"))
_RAW_TTL      = float(os.environ.get("INFRA_RAW_TTL", "2.0"))
_PROBE_TTL    = float(os.environ.get("INFRA_PROBE_TTL", "8.0"))

READONLY  = os.environ.get("INFRA_READONLY", "").lower() in ("1", "true", "yes")
ALLOW_ALL = os.environ.get("INFRA_ALLOW_ALL", "").lower() in ("1", "true", "yes")
PIPES_ON  = os.environ.get("INFRA_PIPE_PROBE", "1").lower() not in ("0", "false", "no")

TTS_PORT   = int(os.environ.get("INFRA_TTS_PORT", "8009"))
IMAGE_PORT = int(os.environ.get("INFRA_IMAGE_PORT", "8002"))
COMFY_PORT = int(os.environ.get("INFRA_COMFY_PORT", "8188"))

# Containers owned by these labs may be started/stopped from the dashboard.
_MANAGED_PREFIXES = ("tts-lab-",)
# Never touched: AutoRemove helper (recreated by tts_lab_dispatch on demand).
_PROTECTED = {"tts-lab-gpu-probe"}

LABS: dict[str, dict[str, Any]] = {
    "tts": {
        "label": "TTS Lab", "short": "TTS", "color": "#6366f1",
        "port": TTS_PORT, "kind": "docker",
        "note": "Orchestrator container serves the UI; engines are HTTP-remote.",
    },
    "image": {
        "label": "Image Lab", "short": "IMG", "color": "#a78bfa",
        "port": IMAGE_PORT, "kind": "bare-metal",
        "note": "Bare-metal systemd service — runs on the host, not in Docker.",
    },
}

# Bare-metal systemd units shown as nodes in the topology (not containers).
#       id,      label,                  unit,                    port, health path
UNITS: list[dict[str, Any]] = [
    {"id": "imglab", "label": "Image Lab", "unit": "arthur-imglab.service",
     "port": IMAGE_PORT, "path": "/status", "lab": "image"},
    {"id": "comfy", "label": "ComfyUI sidecar", "unit": "arthur-comfy.service",
     "port": COMFY_PORT, "path": "/system_stats", "lab": "image"},
    {"id": "ttslab", "label": "TTS bare-metal (retired)", "unit": "arthur-lab.service",
     "port": 8001, "path": "/status", "lab": "tts", "retired": True},
    {"id": "arthur", "label": "arthur.service (disabled)", "unit": "arthur.service",
     "port": 8000, "path": "/health", "lab": "tts", "retired": True},
]

# Stack fingerprint per engine image (hand-maintained — see CLAUDE.md).
STACK_INFO: dict[str, str] = {
    "orchestrator":    "no ML libs · pure HTTP dispatch · port 8009",
    "engine-current":  "torch 2.12 nightly · transformers 5.12.1 · CUDA 12.8 · 21 engines",
    "engine-mid":      "torch 2.12 nightly · transformers 4.51.3 · CUDA 12.8 · VibeVoice + Higgs",
    "engine-qwen":     "torch 2.12 nightly · transformers 4.51.3 · CUDA 12.8 · Qwen3-TTS",
    "engine-legacy":   "torch 1.13 · transformers 4.46 · CUDA 11.7 · IndexTTS (+ Parler blocked)",
    "engine-editx":    "torch 2.13.0+cu130 stable · vLLM 0.26 nightly · Step Audio EditX AWQ-4bit",
    "engine-fa":       "python 3.12 · torch 2.13.0+cu130 · coqui-tts · ParsVoice XTTS",
    "sglang":          "SGLang + flashinfer (sm_120 JIT)",
    "sglang-omni":     "sgl-omni serve + flashinfer · Fish S2-Pro 5B",
    "orpheus":         "vLLM · CUDA 12.1 · Orpheus 3B (blocked)",
    "base":            "nvidia/cuda:12.8.2-runtime-ubuntu22.04",
}

# tier → column rank in the topology graph
_RANK = {"unit": 0, "service": 1, "engine": 2, "aux": 3}


# ─────────────────────────────────────────────────────────────────────────────
# Docker Engine API client (unix socket, no CLI)
# ─────────────────────────────────────────────────────────────────────────────

class DockerError(RuntimeError):
    """Docker daemon unreachable, or the API returned an error."""


def _transport():
    """(httpx transport, base_url) for the configured daemon."""
    import httpx
    if _DOCKER_HOST.startswith("tcp://"):
        return None, "http://" + _DOCKER_HOST[len("tcp://"):]
    if _DOCKER_HOST.startswith("unix://"):
        return httpx.HTTPTransport(uds=_DOCKER_HOST[len("unix://"):]), "http://localhost"
    return httpx.HTTPTransport(uds=_DOCKER_SOCK), "http://localhost"


def _docker(method: str, path: str, *, params=None, content=None,
            timeout: float = 15.0) -> tuple[int, bytes]:
    """Raw call → (status_code, body_bytes). Content calls need the JSON
    Content-Type header or the API answers 400 'malformed Content-Type'."""
    import httpx
    transport, base = _transport()
    headers = {"Content-Type": "application/json"} if content is not None else None
    with httpx.Client(transport=transport, base_url=base, timeout=timeout) as c:
        r = c.request(method, f"/{_API_VER}{path}", params=params,
                      content=content, headers=headers)
        return r.status_code, r.content


def _call(method: str, path: str, *, expect=(200,), **kw) -> Any:
    """Call and decode JSON, raising DockerError on anything unexpected."""
    try:
        code, body = _docker(method, path, **kw)
    except DockerError:
        raise
    except Exception as e:                       # socket gone, daemon down
        raise DockerError(f"Docker daemon unreachable ({_DOCKER_SOCK}): {e}") from e
    if code not in expect:
        msg = body[:300].decode("utf-8", "replace") if body else ""
        raise DockerError(f"{method} {path} → HTTP {code} {msg}")
    if not body:
        return {}
    try:
        return json.loads(body)
    except Exception:
        return {}


def docker_available() -> dict[str, Any]:
    """Cheap liveness probe → {ok, error, version, api, storage_driver}."""
    try:
        v = _call("GET", "/version", timeout=5.0)
        return {
            "ok": True,
            "version": v.get("Version", ""),
            "api": v.get("ApiVersion", ""),
            "min_api": v.get("MinAPIVersion", ""),
            "os": v.get("Os", ""), "arch": v.get("Arch", ""),
            "socket": _DOCKER_HOST or _DOCKER_SOCK,
        }
    except DockerError as e:
        return {"ok": False, "error": str(e), "socket": _DOCKER_HOST or _DOCKER_SOCK}


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

_lock = threading.Lock()
_cache: dict[str, tuple[float, Any]] = {}


def _cached(key: str, ttl: float, fn):
    now = time.time()
    with _lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
    val = fn()
    with _lock:
        _cache[key] = (time.time(), val)
    return val


def _cache_clear() -> None:
    """Drop every cached value. Used after a container action: the stats cache
    is keyed by container *id* while actions name the container, so clearing
    selected keys would leave a stopped container's CPU/mem on screen."""
    with _lock:
        _cache.clear()


def _parse_iso(ts: str | None) -> float:
    """Docker timestamps → epoch seconds (0.0 when absent/unparseable)."""
    if not ts:
        return 0.0
    s = ts.strip().replace("Z", "+00:00")
    # trim nanoseconds (fromisoformat handles at most 6 digits)
    m = re.match(r"(.*\.\d{6})\d+(.*)", s)
    if m:
        s = m.group(1) + m.group(2)
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return 0.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _stack_of(image: str) -> str:
    """`tts-lab-engine-current:latest` → 'engine-current'."""
    repo = (image or "").split("@", 1)[0].rsplit(":", 1)[0]
    for pre in ("tts-lab-", "tts_lab-"):
        if repo.startswith(pre):
            repo = repo[len(pre):]
    return repo


def _lab_of(name: str, labels: dict[str, str]) -> str:
    """Which lab owns this container (name prefix, then compose project)."""
    if name.startswith("tts-lab-"):
        return "infra" if name == "tts-lab-gpu-probe" else "tts"
    if name.startswith(("img-", "img_", "arthur-img", "comfy", "arthur-comfy")):
        return "image"
    proj = (labels or {}).get("com.docker.compose.project", "")
    if proj.startswith("tts-lab"):
        return "tts"
    if proj.startswith(("img", "arthur-img")):
        return "image"
    return "other"


def _tier_of(name: str) -> str:
    if name == "tts-lab-gpu-probe":
        return "aux"
    if name.endswith("-orchestrator"):
        return "service"
    if name.startswith("tts-lab-engine-"):
        return "engine"
    if name.startswith("tts-lab-"):
        return "engine"
    return "service"


def _cpu_percent(s: dict) -> float:
    """CPU% from a one-shot /stats sample (two samples are returned)."""
    try:
        cpu, pre = s["cpu_stats"], s["precpu_stats"]
        cpu_d = cpu["cpu_usage"]["total_usage"] - pre["cpu_usage"]["total_usage"]
        sys_d = cpu["system_cpu_usage"] - pre["system_cpu_usage"]
        if sys_d <= 0 or cpu_d < 0:
            return 0.0
        ncpu = cpu.get("online_cpus") or len(cpu["cpu_usage"].get("percpu_usage") or [1])
        return round(cpu_d / sys_d * ncpu * 100.0, 1)
    except Exception:
        return 0.0


def _mem_of(s: dict) -> tuple[int, int]:
    """(used_mb, limit_mb) — page cache subtracted when the kernel reports it."""
    try:
        mem = s.get("memory_stats") or {}
        used = int(mem.get("usage", 0))
        st = mem.get("stats") or {}
        for k in ("inactive_file", "total_inactive_file", "cache"):
            if k in st:
                used = max(0, used - int(st[k]))
                break
        return used >> 20, int(mem.get("limit", 0)) >> 20
    except Exception:
        return 0, 0


def _demux(raw: bytes) -> str:
    """Strip Docker's multiplexed log/attach frame headers when present.

    Non-TTY containers send 8-byte frames (stream id + 3 pad + LE... actually
    big-endian uint32 length). TTY streams are plain text — detect and pass
    through, so both shapes decode.
    """
    if len(raw) >= 8 and raw[0] in (0, 1, 2) and raw[1:4] == b"\x00\x00\x00":
        out, i, n = [], 0, len(raw)
        while i + 8 <= n:
            if raw[i] not in (0, 1, 2):
                break
            ln = int.from_bytes(raw[i + 4:i + 8], "big")
            if i + 8 + ln > n:
                break
            out.append(raw[i + 8:i + 8 + ln].decode("utf-8", "replace"))
            i += 8 + ln
        if out:
            return "".join(out)
    return raw.decode("utf-8", "replace")


# ─────────────────────────────────────────────────────────────────────────────
# Container inventory
# ─────────────────────────────────────────────────────────────────────────────

def _summarise(name: str, insp: dict, stats: dict | None,
               image_names: dict[str, str]) -> dict[str, Any]:
    """Flatten one container's inspect payload into the dashboard's shape."""
    cfg = insp.get("Config") or {}
    state = insp.get("State") or {}
    hc = insp.get("HostConfig") or {}
    net = ((insp.get("NetworkSettings") or {}).get("Networks") or {})
    labels = cfg.get("Labels") or {}

    env = {}
    for kv in cfg.get("Env") or []:
        k, _, v = kv.partition("=")
        env[k] = v

    # ── networks / IP addressing ──
    nets = []
    for nname, ninfo in net.items():
        if nname in ("host", "none"):
            nets.append({"name": nname, "ip": "", "aliases": [],
                         "gateway": "", "mac": "", "prefix": ninfo.get("IPPrefixLen", 0)})
            continue
        nets.append({
            "name":    nname,
            "ip":      ninfo.get("IPAddress", ""),
            "prefix":  ninfo.get("IPPrefixLen", 0),
            "gateway": ninfo.get("Gateway", ""),
            "mac":     ninfo.get("MacAddress", ""),
            "aliases": [a for a in (ninfo.get("Aliases") or [])
                        if a and a != insp.get("Id", "")[:12]],
        })

    # ── ports (internal → published) ──
    ports = []
    for cport, binds in ((insp.get("NetworkSettings") or {}).get("Ports") or {}).items():
        cport_num = cport.split("/")[0]
        if binds:
            for b in binds:
                ports.append({"container": cport_num, "host": b.get("HostPort", ""),
                              "host_ip": b.get("HostIp", ""), "proto": cport.split("/")[-1]})
        else:
            ports.append({"container": cport_num, "host": None,
                          "host_ip": "", "proto": cport.split("/")[-1]})

    # ── GPU request ──
    gpu = 0
    for dr in hc.get("DeviceRequests") or []:
        caps = [c for cap in (dr.get("Capabilities") or []) for c in cap]
        if "gpu" in caps:
            gpu = dr.get("Count", -1) if dr.get("Count") not in (0, None) else -1

    # ── mounts ──
    mounts = [{
        "type":   m.get("Type", ""),
        "source": m.get("Source", ""),
        "dest":   m.get("Destination", ""),
        "rw":     bool(m.get("RW", False)),
    } for m in insp.get("Mounts") or []]

    started = _parse_iso(state.get("StartedAt"))
    finished = _parse_iso(state.get("FinishedAt"))
    now = time.time()
    uptime_s = int(now - started) if state.get("Running") and started else 0
    exited_s = int(now - finished) if (not state.get("Running") and finished) else 0

    health = (state.get("Health") or {}).get("Status", "")

    return {
        "id":         insp.get("Id", "")[:12],
        "name":       (insp.get("Name") or name).lstrip("/"),
        "image":      cfg.get("Image", ""),
        "image_id":   (insp.get("Image") or "")[7:19],
        "image_name": image_names.get((insp.get("Image") or "")[:19], cfg.get("Image", "")),
        "stack":      _stack_of(cfg.get("Image", "")),
        "stack_note": STACK_INFO.get(_stack_of(cfg.get("Image", "")), ""),
        "lab":        _lab_of(name, labels),
        "tier":       _tier_of(name),
        "rank":       _RANK[_tier_of(name)],
        "status":     state.get("Status", "unknown"),
        "running":    bool(state.get("Running")),
        "paused":     bool(state.get("Paused")),
        "restarting": bool(state.get("Restarting")),
        "health":     health,
        "exit_code":  state.get("ExitCode", 0),
        "error":      state.get("Error", ""),
        "oom_killed": bool(state.get("OOMKilled")),
        "pid":        state.get("Pid", 0),
        "started_at": state.get("StartedAt", ""),
        "uptime_s":   uptime_s,
        "exited_s":   exited_s,
        "created":    insp.get("Created", ""),
        "restarts":   insp.get("RestartCount", 0),
        "restart_policy": (hc.get("RestartPolicy") or {}).get("Name", ""),
        "cmd":        " ".join(cfg.get("Cmd") or []) if isinstance(cfg.get("Cmd"), list) else (cfg.get("Cmd") or ""),
        "entrypoint": " ".join(cfg.get("Entrypoint") or []) if isinstance(cfg.get("Entrypoint"), list) else "",
        "networks":   nets,
        "ip":         next((n["ip"] for n in nets if n["ip"]), ""),
        "aliases":    sorted({a for n in nets for a in n["aliases"]} | {name}),
        "ports":      sorted(ports, key=lambda p: int(p["container"])),
        "gpu":        gpu,
        "mounts":     mounts,
        "env":        env,
        "labels": {
            "project": labels.get("com.docker.compose.project", ""),
            "service": labels.get("com.docker.compose.service", ""),
            "config":  labels.get("com.docker.compose.config-hash", "")[:12],
        },
        "stats":      stats or {},
        "managed":    ALLOW_ALL or name.startswith(_MANAGED_PREFIXES),
        "protected":  name in _PROTECTED,
        # Disposable helpers have no server inside, so their image HEALTHCHECK
        # can never pass: tts-lab-gpu-probe reports 'unhealthy' forever while
        # working exactly as designed. Its health must not raise an alarm, but
        # the card still shows the raw value (nothing is hidden).
        "health_ignored": name in _PROTECTED,
    }


def _images_json() -> list[dict]:
    """`/images/json`, TTL-cached. Measured at ~2 s on this VM (159 GB of
    images), and the overview needs it twice — once to name each container's
    image, once to list images with their users. Fetching it twice accounted
    for ~4 s of an 8 s overview build.
    """
    return _cached("images", _IMAGES_TTL, lambda: _call("GET", "/images/json"))


def _image_name_map() -> dict[str, str]:
    """Image ID prefix (19 chars, sha256:) → best repo:tag."""
    out: dict[str, str] = {}
    try:
        for img in _images_json():
            tags = img.get("RepoTags") or []
            if not tags:
                continue
            tags = sorted(t for t in tags if t != "<none>:<none>") or tags
            out[img.get("Id", "")[:19]] = tags[0]
    except DockerError:
        pass
    return out


def _containers_raw() -> list[dict[str, Any]]:
    """Full inventory: summaries (parallel inspect + parallel stats).

    Briefly cached: five endpoints rebuild this list (overview, images,
    networks, volumes, pipes) and it is the expensive part — a listing, N
    inspects and a stats sample per running container (measured ~0.9 s warm,
    ~8 s before the image/stats lists were shared). 2 s is safe because every
    mutating action calls ``_cache_clear()``.
    """
    return _cached("containers", _RAW_TTL, _containers_build)


def _containers_build() -> list[dict[str, Any]]:
    """Uncached inventory build — see ``_containers_raw``."""
    listing = _call("GET", "/containers/json", params={"all": 1, "size": 0})
    names = [(c["Names"][0].lstrip("/") if c.get("Names") else c["Id"][:12], c["Id"])
             for c in listing]
    if not names:
        return []
    images = _image_name_map()

    def one(item):
        name, cid = item
        try:
            insp = _call("GET", f"/containers/{cid}/json", timeout=10.0)
        except DockerError as e:
            log.warning("inspect %s failed: %s", name, e)
            return None
        stats = None
        if (insp.get("State") or {}).get("Running"):
            try:
                # Stats cost ~0.1 s per engine container but up to ~2 s for the
                # orchestrator itself (the container answering the request), so
                # cache them: CPU/mem bars at 10 s resolution are plenty, and it
                # keeps every other poll from paying for them.
                stats = _cached(f"stats:{cid}", _STATS_TTL, lambda: _call(
                    "GET", f"/containers/{cid}/stats",
                    params={"stream": "false", "one-shot": "true"}, timeout=12.0))
            except DockerError:
                stats = None
        s = _summarise(name, insp, stats, images)
        if stats:
            used, limit = _mem_of(stats)
            s["stats"] = {"cpu_pct": _cpu_percent(stats), "mem_mb": used,
                          "mem_limit_mb": limit,
                          "mem_pct": round(used / limit * 100, 1) if limit else 0.0}
        return s

    with ThreadPoolExecutor(max_workers=10) as pool:
        rows = [r for r in pool.map(one, names) if r]
    # keep the topology column order stable
    rows.sort(key=lambda r: (r["rank"], r["name"]))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Topology: engine→container routing parsed from the containers' own env
# ─────────────────────────────────────────────────────────────────────────────

_URL_ENV = re.compile(r"^([A-Z0-9_]+)_URL$")
# Routes to a proxied service rather than an engine. Matched by EXACT env name:
# a substring test ("VOICE" in key) silently mislabelled CosyVoice, OmniVoice,
# OpenVoice and VibeVoice as services, so those engines disappeared from the
# per-container counters while still listed in the flows table — a mismatch only
# visible when the two are compared.
_SERVICE_URLS = {"VOICE_LIB_URL"}


def _edge_key(env_key: str) -> str:
    k = env_key[:-4]                        # drop _URL
    k = k.replace("_SGLANG", "")
    return k.lower()


def _resolve_host(host: str, index: dict[str, str]) -> str | None:
    """Container name for a compose hostname/alias (host or host:port)."""
    if host in index:
        return index[host]
    base = host.split(".")[0]
    return index.get(base)


def build_topology(containers: list[dict[str, Any]]) -> dict[str, Any]:
    """Nodes + edges for the graph, derived from live container config."""
    index: dict[str, str] = {}
    for c in containers:
        index[c["name"]] = c["name"]
        for a in c["aliases"]:
            index.setdefault(a, c["name"])
        # compose service name is usually the unprefixed alias (engine-current)
        if c["labels"]["service"]:
            index.setdefault(c["labels"]["service"], c["name"])

    edges: dict[tuple[str, str], dict[str, Any]] = {}
    routes: dict[str, list[str]] = {c["name"]: [] for c in containers}

    for c in containers:
        for k, v in c["env"].items():
            m = _URL_ENV.match(k)
            if not m or not isinstance(v, str) or "://" not in v:
                continue
            parts = urlsplit(v)
            if not parts.hostname:
                continue
            target = _resolve_host(parts.hostname, index)
            if not target or target == c["name"]:
                continue
            engine = _edge_key(k)
            kind = "service" if k in _SERVICE_URLS else "engine"
            key = (c["name"], target)
            e = edges.setdefault(key, {
                "from": c["name"], "to": target,
                "var": k, "host": parts.hostname,
                "port": parts.port or (443 if parts.scheme == "https" else 80),
                "path": parts.path or "",
                "kind": kind, "engines": [], "services": [],
            })
            if kind == "engine":
                e["engines"].append(engine)
                routes[c["name"]].append(engine)
                routes[target].append(engine)
            else:
                e["services"].append(engine)

    for e in edges.values():
        # Several env vars can normalise to one key; dedupe so an edge's engine
        # count matches the target's routes list exactly.
        e["engines"] = sorted(set(e["engines"]))
        e["services"] = sorted(set(e["services"]))
        if not e["engines"]:
            e["kind"] = "service"

    for c in containers:
        c["routes"] = sorted(set(routes.get(c["name"], [])))
    return {"edges": sorted(edges.values(), key=lambda e: (e["from"], e["to"])),
            "routes": {k: sorted(set(v)) for k, v in routes.items()}}


# ─────────────────────────────────────────────────────────────────────────────
# Pipe health probe — the actual HTTP hops the labs depend on
# ─────────────────────────────────────────────────────────────────────────────

def probe_pipes(edges: list[dict], by_name: dict[str, dict]) -> list[dict[str, Any]]:
    """GET <container-ip>:<internal-port>/health for every routing edge.

    Probing the *container IP* (not the compose DNS name) works from the
    orchestrator container AND from the bare-metal image lab, because the
    bridge network is routable from the host. Stopped targets are recorded as
    such without a network round trip.
    """
    import httpx
    if not PIPES_ON:
        return []
    targets: dict[tuple[str, int], list[dict]] = {}
    for e in edges:
        targets.setdefault((e["to"], e["port"]), []).append(e)

    def one(item):
        (tname, port), es = item
        target = by_name.get(tname) or {}
        ip = target.get("ip") or ""
        base = {"target": tname, "port": port, "path": es[0]["path"] or "/health",
                "ip": ip, "kind": es[0]["kind"],
                "vars": sorted({e["var"] for e in es}),
                "engines": sorted({en for e in es for en in e["engines"]}),
                "services": sorted({s for e in es for s in e.get("services", [])})}
        if not target.get("running") or not ip:
            return {**base, "ok": False, "ms": None,
                    "reason": "target not running" if not target.get("running") else "no IP"}
        url = f"http://{ip}:{port}{es[0]['path'] or '/health'}"
        t0 = time.perf_counter()
        try:
            r = httpx.get(url, timeout=2.5)
            ms = round((time.perf_counter() - t0) * 1000, 1)
            body: Any = None
            try:
                body = r.json()
            except Exception:
                body = r.text[:200]
            health = {}
            if isinstance(body, dict):
                health = {
                    "current_engine": body.get("current_engine"),
                    "loaded":         sorted([k for k, v in (body.get("engines") or {}).items()
                                              if isinstance(v, dict) and v.get("loaded")]),
                    "gpu":            body.get("gpu") or None,
                    "sglang_up":      body.get("sglang_up"),
                }
            return {**base, "ok": r.status_code == 200, "code": r.status_code,
                    "ms": ms, "health": health}
        except Exception as e:
            return {**base, "ok": False, "ms": None, "reason": type(e).__name__}

    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(one, sorted(targets.items())))


# ─────────────────────────────────────────────────────────────────────────────
# Host view: docker info, systemd units, disks, GPU (borrowed from /status)
# ─────────────────────────────────────────────────────────────────────────────

def _units() -> list[dict[str, Any]]:
    """systemd unit state — absent inside the orchestrator container.

    Each active unit also gets a localhost HTTP ping (the labs are on the same
    box), so the dashboard shows "service up AND answering" rather than just
    "systemd says active" — a listening-but-broken service is the failure mode
    that matters here.
    """
    have = shutil.which("systemctl") is not None
    units = []
    for u in UNITS:
        row = {k: u[k] for k in ("id", "label", "unit", "port", "lab", "path")}
        row["retired"] = bool(u.get("retired"))
        if not have:
            row.update({"active": None, "enabled": None,
                        "note": "systemctl not available here (container)"})
            units.append(row)
            continue
        for key, verb in (("active", "is-active"), ("enabled", "is-enabled")):
            try:
                p = subprocess.run(["systemctl", verb, u["unit"]],
                                   capture_output=True, text=True, timeout=4)
                row[key] = (p.stdout or p.stderr).strip() or "unknown"
            except Exception as e:
                row[key] = f"error: {e}"
        if row.get("active") == "active":
            row["http"] = _ping_local(u["port"], u.get("path", "/"),
                                      want=("gpu", "vram", "device", "active_engine"))
        units.append(row)
    return units


def _ping_local(port: int, path: str = "/", timeout: float = 2.0,
                want: tuple[str, ...] = ()) -> dict[str, Any]:
    """GET http://127.0.0.1:<port><path> → {ok, ms, code, **want}.

    ``want`` pulls named top-level keys out of a JSON body (keeping the
    response small — the labs' /status payloads are large).
    """
    import httpx
    t0 = time.perf_counter()
    try:
        r = httpx.get(f"http://127.0.0.1:{port}{path}", timeout=timeout)
        out: dict[str, Any] = {"ok": r.status_code < 500, "code": r.status_code,
                               "ms": round((time.perf_counter() - t0) * 1000, 1)}
        if want and r.status_code < 400:
            try:
                body = r.json()
                for k in want:
                    if isinstance(body, dict) and k in body:
                        out[k] = body[k]
            except Exception:
                pass
        return out
    except Exception as e:
        return {"ok": False, "code": None, "ms": None, "reason": type(e).__name__}


def gpu_from_status(data: dict) -> dict[str, Any] | None:
    """Normalise the two labs' GPU payload shapes into one.

    engine /health  : {name, vram_total, vram_used, vram_free, proc_allocated_mb,
                       proc_reserved_mb}      → vram_* is DEVICE-WIDE, proc_* is own
    image /status   : gpu {vram_total_mb, vram_used_mb, ...} + vram {reserved_gb, ...}
    """
    g = data.get("gpu") or {}
    v = data.get("vram") or {}
    if not g and not v:
        return None
    def _pick(*vals):
        for x in vals:
            if x:
                return int(x)
        return 0
    total = _pick(g.get("vram_total_mb"), g.get("vram_total"),
                  round((v.get("total_gb") or 0) * 1024))
    used = _pick(g.get("vram_used_mb"), g.get("vram_used"),
                 round((v.get("reserved_gb") or 0) * 1024))
    free = _pick(g.get("vram_free_mb"), g.get("vram_free", 0) or None,
                 max(0, total - used))
    # this process's own footprint (what it *can* be evicted for)
    share = _pick(g.get("proc_reserved_mb"), g.get("proc_allocated_mb"),
                  round((v.get("reserved_gb") or 0) * 1024))
    return {"name": g.get("name") or v.get("device_name") or "",
            "total_mb": total, "used_mb": used, "free_mb": free,
            "share_mb": share, "source": g.get("source") or "",
            "processes": g.get("processes") or []}


def gpu_summary(containers: list[dict], units: list[dict],
                own: dict | None = None) -> dict[str, Any]:
    """Who is holding the one GPU — containers *and* bare-metal services.

    This is the coupling that crosses labs: the TTS engine containers and the
    bare-metal Image Lab service contend for the same 16 GB card, which is why
    the two labs evict each other. Nothing in the compose routing expresses
    that, so it is derived from each process's own view of the device.
    """
    name, total, used, free = "", 0, 0, 0
    holders: list[dict[str, Any]] = []
    for c in containers:
        live = c.get("live") or {}
        g = gpu_from_status({"gpu": live.get("gpu")})
        if not g:
            continue
        name = name or g["name"]
        total = total or g["total_mb"]
        used = max(used, g["used_mb"])
        free = free or g["free_mb"]
        holders.append({"node": c["name"], "name": c["name"], "kind": "container",
                        "mb": g["share_mb"], "running": c["running"],
                        "note": live.get("current_engine") or ""})
    for u in units:
        g = gpu_from_status(u.get("http") or {})
        if not g:
            continue
        name = name or g["name"]
        total = total or g["total_mb"]
        used = max(used, g["used_mb"])
        free = free or g["free_mb"]
        # `node` is the graph id of the unit node ("unit:<id>") — the display
        # label is not an id, and an edge to a non-existent node is dropped.
        holders.append({"node": "unit:" + u["id"], "name": u["label"],
                        "kind": "bare-metal", "mb": g["share_mb"],
                        "running": u.get("active") == "active", "note": u["unit"]})
    if own:
        g = gpu_from_status(own)
        if g and not any(h["kind"] == "bare-metal" for h in holders):
            name = name or g["name"]
            total = total or g["total_mb"]
            used = max(used, g["used_mb"])
            free = free or g["free_mb"]
    if not (name or total):
        return {"available": False, "holders": [],
                "note": "no GPU telemetry reachable from this process"}
    holders.sort(key=lambda h: -h["mb"])
    return {"available": True, "name": name, "total_mb": total,
            "used_mb": used, "free_mb": free,
            "pct": round(used / total * 100, 1) if total else 0,
            "holders": holders,
            "processes": (own or {}).get("gpu", {}).get("processes") or []}



def _host_view() -> dict[str, Any]:
    """docker info + RAM + disks + units, all best-effort."""
    out: dict[str, Any] = {"units": _units(), "cpu": os.cpu_count(),
                           # False inside the orchestrator container: it cannot see
                           # systemd, so unit state must be read from the image lab.
                           "systemd": shutil.which("systemctl") is not None}
    try:
        info = _call("GET", "/info", timeout=8.0)
        out["docker"] = {
            "name":        info.get("Name", ""),
            "server":      info.get("ServerVersion", ""),
            "os":          info.get("OperatingSystem", ""),
            "kernel":      info.get("KernelVersion", ""),
            "storage":     info.get("Driver", ""),
            "root":        info.get("DockerRootDir", ""),
            "cgroup":      info.get("CgroupVersion", ""),
            "containers":  info.get("Containers", 0),
            "running":     info.get("ContainersRunning", 0),
            "paused":      info.get("ContainersPaused", 0),
            "stopped":     info.get("ContainersStopped", 0),
            "images":      info.get("Images", 0),
            "ncpu":        info.get("NCPU", 0),
            "mem_total_mb": int(info.get("MemTotal", 0)) >> 20,
            "runtimes":    sorted((info.get("Runtimes") or {}).keys()),
            "warnings":    info.get("Warnings") or [],
        }
    except DockerError as e:
        out["docker"] = {"error": str(e)}
    try:
        out["ram"] = _ram()
    except Exception:
        out["ram"] = None
    out["disks"] = _disks()
    try:
        out["load"] = list(os.getloadavg())
    except Exception:
        out["load"] = None
    return out


def _ram():
    """(total, used, free) MB from /proc/meminfo — reports HOST totals even
    inside a container, which is what the lab dashboards already show."""
    try:
        mem = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, v = line.partition(":")
                mem[k] = int(v.strip().split()[0]) // 1024
        total = mem.get("MemTotal", 0)
        avail = mem.get("MemAvailable", mem.get("MemFree", 0))
        return {"total_mb": total, "used_mb": max(0, total - avail), "free_mb": avail}
    except Exception:
        return None


def _disks() -> list[dict[str, Any]]:
    seen, out = set(), []
    for p in ("/", "/opt/models", "/opt/arthur-img-models", "/var/lib/docker", "/tmp"):
        if p in seen or not os.path.isdir(p):
            continue
        seen.add(p)
        try:
            u = shutil.disk_usage(p)
            out.append({"path": p, "total_gb": round(u.total / 1e9, 1),
                        "used_gb": round(u.used / 1e9, 1),
                        "free_gb": round(u.free / 1e9, 1),
                        "pct": round(u.used / u.total * 100, 1)})
        except Exception:
            continue
    return out


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI router
# ─────────────────────────────────────────────────────────────────────────────

router = APIRouter()

# Every handler below is a *sync* ``def`` on purpose. The work is blocking
# socket I/O against the Docker daemon (a stop can take 20 s), and the TTS
# orchestrator runs uvicorn with one worker — an ``async def`` here would stall
# the event loop and therefore synthesis polling. FastAPI runs sync handlers in
# its threadpool, which is exactly the behaviour wanted.


@router.get("/infra", response_class=HTMLResponse, include_in_schema=False)
async def infra_page():
    from lab_infra_ui import build_infra_page
    return HTMLResponse(build_infra_page())


@router.get("/infra/api/overview")
def overview(probe: int = 1, stats: int = 1):
    """Everything the dashboard renders in one round trip (2 s cache)."""
    def build():
        avail = docker_available()
        if not avail["ok"]:
            return {"docker": avail, "containers": [], "edges": [], "routes": {},
                    "pipes": [], "images": [], "networks": [], "volumes": [],
                    "gpu": {"available": False, "holders": [],
                            "note": "docker unavailable — no GPU telemetry"},
                    "labs": LABS, "host": {"units": _units(), "cpu": os.cpu_count(),
                                           "systemd": shutil.which("systemctl") is not None},
                    "counts": {"total": 0, "running": 0, "images": 0, "volumes": 0}}
        containers = _containers_raw()
        if not stats:
            for c in containers:
                c.pop("stats", None)
        topo = build_topology(containers)
        by_name = {c["name"]: c for c in containers}
        pipes = probe_pipes(topo["edges"], by_name) if probe else []
        # fold pipe results back into the nodes (resident engine / VRAM)
        for p in pipes:
            node = by_name.get(p["target"])
            if node is not None and p.get("health"):
                node["live"] = {
                    "current_engine": p["health"].get("current_engine"),
                    "loaded":         p["health"].get("loaded") or [],
                    "gpu":            p["health"].get("gpu"),
                    "sglang_up":      p["health"].get("sglang_up"),
                }
        images = image_usage(containers)
        volumes = _volumes_raw(containers)
        host = _host_view()
        labs = {}
        for key, meta in LABS.items():
            own = [c for c in containers if c["lab"] == key]
            labs[key] = {**meta, "total": len(own),
                         "running": sum(1 for c in own if c["running"]),
                         "url": f"http://{{host}}:{meta['port']}/"}
        return {
            "docker":     avail,
            "containers": containers,
            "edges":      topo["edges"],
            "routes":     topo["routes"],
            "pipes":      pipes,
            "images":     images,
            "networks":   _networks_raw(containers),
            "volumes":    volumes,
            "gpu":        gpu_summary(containers, host.get("units") or []),
            "labs":       labs,
            "host":       host,
            "counts":     counts_of(containers, images, volumes),
            "tool": {"readonly": READONLY, "allow_all": ALLOW_ALL,
                     "pipes": PIPES_ON},
            "ts": time.time(),
        }

    key, ttl = ("overview-probe" if probe else "overview-noprobe"), _CACHE_TTL
    try:
        return JSONResponse(_cached(key, ttl, build))
    except DockerError as e:
        return JSONResponse({"docker": {"ok": False, "error": str(e)},
                             "containers": [], "edges": [], "pipes": [],
                             "images": [], "networks": [], "volumes": [],
                             "labs": LABS, "host": {"units": []},
                             "counts": {"total": 0, "running": 0}}, status_code=200)


def counts_of(containers: list[dict], images: list[dict],
              volumes: list[dict]) -> dict[str, int]:
    """Header counters for the overview payload.

    ``unhealthy`` deliberately counts only RUNNING containers. Docker keeps the
    last health status on an exited container, so counting every
    ``health == "unhealthy"`` reports stopped containers as if they were
    failing right now — observed live on this VM, where 4 long-exited engine
    containers turned the header pill red. Containers marked
    ``health_ignored`` (disposable helpers whose healthcheck cannot succeed) are
    excluded too. Extracted as a function so the offline probe can exercise
    both cases (see scripts/probe/).
    """
    return {
        "total":     len(containers),
        "running":   sum(1 for c in containers if c["running"]),
        "unhealthy": sum(1 for c in containers
                          if c["running"] and c["health"] == "unhealthy"
                          and not c.get("health_ignored")),
        "images":    len(images),
        "volumes":   len(volumes),
    }


# ── containers ──────────────────────────────────────────────────────────────

@router.get("/infra/api/containers/{name}")
def container_detail(name: str):
    try:
        insp = _call("GET", f"/containers/{name}/json")
    except DockerError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    stats = None
    if (insp.get("State") or {}).get("Running"):
        try:
            stats = _call("GET", f"/containers/{name}/stats",
                          params={"stream": "false", "one-shot": "true"}, timeout=12.0)
        except DockerError:
            pass
    row = _summarise(name, insp, stats, _image_name_map())
    if stats:
        used, limit = _mem_of(stats)
        row["stats"] = {"cpu_pct": _cpu_percent(stats), "mem_mb": used,
                        "mem_limit_mb": limit}
    row["healthcheck"] = (insp.get("Config") or {}).get("Healthcheck") or {}
    row["raw_state"] = insp.get("State") or {}
    return JSONResponse(row)


@router.get("/infra/api/containers/{name}/stats")
def container_stats(name: str):
    try:
        s = _call("GET", f"/containers/{name}/stats",
                  params={"stream": "false", "one-shot": "true"}, timeout=12.0)
    except DockerError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    used, limit = _mem_of(s)
    return JSONResponse({"cpu_pct": _cpu_percent(s), "mem_mb": used,
                         "mem_limit_mb": limit, "ts": time.time()})


@router.get("/infra/api/containers/{name}/logs")
def container_logs(name: str, tail: int = 200, since: int = 0, grep: str = ""):
    params: dict[str, Any] = {"stdout": 1, "stderr": 1, "tail": max(1, min(tail, 2000)),
                              "timestamps": 1}
    if since:
        params["since"] = since
    try:
        code, raw = _docker("GET", f"/containers/{name}/logs", params=params, timeout=15.0)
    except Exception as e:
        return JSONResponse({"error": f"docker unreachable: {e}"}, status_code=502)
    if code not in (200,):
        return JSONResponse({"error": _demux(raw)[:400]}, status_code=code)
    text = _demux(raw)
    lines = text.splitlines()
    if grep:
        lines = [ln for ln in lines if grep.lower() in ln.lower()]
    return JSONResponse({"name": name, "lines": lines, "count": len(lines),
                         "tail": tail, "ts": time.time()})


@router.post("/infra/api/containers/{name}/action/{act}")
def container_action(name: str, act: str, force: int = 0, timeout: int = 20):
    """start | stop | restart | pause | unpause | kill"""
    if READONLY:
        return JSONResponse({"ok": False, "error": "read-only mode (INFRA_READONLY=1)"},
                            status_code=403)
    if act not in ("start", "stop", "restart", "pause", "unpause", "kill"):
        return JSONResponse({"ok": False, "error": f"unsupported action {act!r}"},
                            status_code=400)
    if not name.startswith(_MANAGED_PREFIXES) and not ALLOW_ALL:
        return JSONResponse({
            "ok": False,
            "error": f"{name!r} is outside this lab (only tts-lab-* is managed). "
                     f"Set INFRA_ALLOW_ALL=1 to lift the restriction.",
        }, status_code=403)
    if name in _PROTECTED:
        return JSONResponse({
            "ok": False,
            "error": f"{name} is a disposable helper container — the TTS dispatch "
                     f"layer recreates it on demand. Not managing it from here.",
        }, status_code=403)
    # Refuse to cut the branch we are sitting on.
    if act in ("stop", "kill") and force != 1:
        import socket as _s
        me = _s.gethostname() or ""
        try:
            ident = "docker"
            with open("/proc/self/cgroup") as f:
                for m in re.finditer(r"docker[-/]([0-9a-f]{12,64})", f.read()):
                    ident = m.group(1)[:12]
                    break
        except Exception:
            ident = ""
        try:
            insp = _call("GET", f"/containers/{name}/json", timeout=8.0)
            cid = insp.get("Id", "")[:12]
            hosts = insp.get("Config", {}).get("Hostname", "")
        except DockerError as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=404)
        if ident and (cid == ident or hosts == me):
            return JSONResponse({
                "ok": False,
                "error": f"refusing to {act} {name}: this dashboard is served from that "
                         f"container. Use `docker compose {act} ...` on the VM, or "
                         f"pass force=1 to override.",
            }, status_code=409)

    path = f"/containers/{name}/{act}"
    params = {"t": timeout} if act in ("stop", "restart", "kill") else None
    try:
        code, body = _docker("POST", path, params=params, timeout=timeout + 10)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"docker unreachable: {e}"},
                            status_code=502)
    ok = code in (204, 304)
    log.info("infra action %s %s → HTTP %s", act, name, code)
    _cache_clear()                      # stop showing the pre-action state
    return JSONResponse({"ok": ok, "action": act, "container": name, "http": code,
                         "error": "" if ok else _demux(body)[:300]},
                        status_code=200 if ok else 409)


@router.post("/infra/api/lab/{lab}/action/{act}")
def lab_action(lab: str, act: str, force: int = 0):
    """Bulk start/stop/restart over every container belonging to a lab."""
    if READONLY:
        return JSONResponse({"ok": False, "error": "read-only mode"}, status_code=403)
    if lab not in LABS:
        return JSONResponse({"ok": False, "error": f"unknown lab {lab!r}"}, status_code=400)
    if act not in ("start", "stop", "restart"):
        return JSONResponse({"ok": False, "error": f"unsupported bulk action {act!r}"},
                            status_code=400)
    try:
        containers = _containers_raw()
    except DockerError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=502)
    targets = [c["name"] for c in containers
               if c["lab"] == lab and not c["protected"]
               and (c["name"].startswith(_MANAGED_PREFIXES) or ALLOW_ALL)]
    results = {}
    for n in targets:
        path = f"/containers/{n}/{act}"
        params = {"t": 20} if act in ("stop", "restart") else None
        try:
            code, body = _docker("POST", path, params=params, timeout=40)
            results[n] = {"ok": code in (204, 304), "http": code}
        except Exception as e:
            results[n] = {"ok": False, "error": str(e)}
    _cache_clear()
    return JSONResponse({"action": act, "lab": lab, "results": results,
                         "ok": all(r.get("ok") for r in results.values()) if results else False})


# ── images / networks / volumes ─────────────────────────────────────────────

def image_usage(containers: list[dict]) -> list[dict[str, Any]]:
    """Images with the containers currently running them (lab images first)."""
    users: dict[str, list[str]] = {}
    for c in containers:
        users.setdefault(c["image_id"], []).append(c["name"])
    try:
        raw = _images_json()
    except DockerError:
        return []
    out = []
    for img in raw:
        tags = [t for t in (img.get("RepoTags") or []) if t != "<none>:<none>"]
        iid = img.get("Id", "")
        prefix = iid[7:19] if iid.startswith("sha256:") else iid[:12]
        short = iid[len("sha256:"):][:12] if iid.startswith("sha256:") else iid[:12]
        out.append({
            "id":       short,
            "tags":     tags or [f"<none> ({short})"],
            "lab":      bool(tags) and all(t.startswith("tts-lab-") for t in tags),
            "size_mb":  round(img.get("Size", 0) / 1048576, 1),
            "shared_mb": round(img.get("SharedSize", 0) / 1048576, 1),
            "created":  img.get("Created", 0),
            "dangling": not tags,
            "used_by":  users.get(prefix, []),
            "containers": img.get("Containers", -1),
        })
    out.sort(key=lambda i: (not i["lab"], not i["used_by"], -i["size_mb"]))
    return out


def _networks_raw(containers: list[dict]) -> list[dict[str, Any]]:
    try:
        raw = _call("GET", "/networks")
    except DockerError:
        return []
    members: dict[str, list[str]] = {}
    for c in containers:
        for n in c["networks"]:
            members.setdefault(n["name"], []).append(c["name"])
    out = []
    for n in raw:
        cfg = (n.get("IPAM") or {}).get("Config") or []
        subnets = [c.get("Subnet", "") for c in cfg if c.get("Subnet")]
        gw = next((c.get("Gateway", "") for c in cfg if c.get("Gateway")), "")
        out.append({
            "id":      n.get("Id", "")[:12],
            "name":    n.get("Name", ""),
            "driver":  n.get("Driver", ""),
            "scope":   n.get("Scope", ""),
            "internal": bool(n.get("Internal")),
            "subnets": subnets,
            "gateway": gw,
            "created": n.get("Created", ""),
            "attached": sorted(members.get(n.get("Name", ""), [])),
            "containers": len(n.get("Containers") or {}),
        })
    out.sort(key=lambda n: (n["driver"] != "bridge", n["name"]))
    return out


def _volumes_raw(containers: list[dict]) -> list[dict[str, Any]]:
    try:
        raw = _call("GET", "/volumes")
    except DockerError:
        return []
    users: dict[str, list[str]] = {}
    for c in containers:
        for m in c["mounts"]:
            if m["type"] == "volume":
                users.setdefault(m["source"], []).append(f"{c['name']}:{m['dest']}")
    out = []
    for v in raw.get("Volumes") or []:
        out.append({
            "name":       v.get("Name", ""),
            "driver":     v.get("Driver", ""),
            "mountpoint": v.get("Mountpoint", ""),
            "created":    v.get("CreatedAt", ""),
            "scope":      v.get("Scope", ""),
            "used_by":    sorted(users.get(v.get("Name", ""), [])),
        })
    out.sort(key=lambda v: (not v["used_by"], v["name"]))
    return out


@router.get("/infra/api/images")
def images():
    try:
        return JSONResponse({"images": image_usage(_containers_raw())})
    except DockerError as e:
        return JSONResponse({"images": [], "error": str(e)}, status_code=200)


@router.get("/infra/api/networks")
def networks():
    try:
        return JSONResponse({"networks": _networks_raw(_containers_raw())})
    except DockerError as e:
        return JSONResponse({"networks": [], "error": str(e)}, status_code=200)


@router.get("/infra/api/volumes")
def volumes():
    try:
        return JSONResponse({"volumes": _volumes_raw(_containers_raw())})
    except DockerError as e:
        return JSONResponse({"volumes": [], "error": str(e)}, status_code=200)


@router.get("/infra/api/host")
def host():
    return JSONResponse(_host_view())


@router.get("/infra/api/disk")
def disk():
    """`/system/df` — layer/volume/build-cache sizes. Slow, 60 s cached."""
    def build():
        try:
            df = _call("GET", "/system/df", timeout=45.0)
        except DockerError as e:
            return {"error": str(e)}
        return {
            "layers_mb":  round(df.get("LayersSize", 0) / 1048576, 1),
            "images_mb":  sum((i.get("Size", 0)) for i in df.get("Images") or []) / 1048576,
            "volumes_mb": sum((v.get("UsageData") or {}).get("Size", -1)
                              for v in df.get("Volumes") or []) / 1048576,
            "build_mb":   round(sum(b.get("Size", 0) for b in df.get("BuildCache") or [])
                                / 1048576, 1),
            "containers_mb": round(sum((c.get("SizeRw", 0)) for c in df.get("Containers") or [])
                                   / 1048576, 1),
            "counts": {"images": len(df.get("Images") or []),
                       "volumes": len(df.get("Volumes") or []),
                       "containers": len(df.get("Containers") or [])},
            "ts": time.time(),
        }
    return JSONResponse(_cached("disk", 60.0, build))


@router.get("/infra/api/units")
def units():
    return JSONResponse({"units": _units(), "available": shutil.which("systemctl") is not None})


@router.get("/infra/api/pipes")
def pipes():
    """Live health of every engine→container HTTP hop."""
    try:
        containers = _containers_raw()
    except DockerError as e:
        return JSONResponse({"pipes": [], "error": str(e)}, status_code=200)
    topo = build_topology(containers)
    by_name = {c["name"]: c for c in containers}
    return JSONResponse({"pipes": _cached(
        "pipes", _PROBE_TTL, lambda: probe_pipes(topo["edges"], by_name))})
