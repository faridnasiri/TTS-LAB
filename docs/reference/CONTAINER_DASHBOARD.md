# Container / Infrastructure Dashboard (`/infra`)

> **Added:** 2026-09-14 · **Served by:** both labs — `:8009/infra` (TTS orchestrator) and `:8002/infra` (Image Lab)
> **Source:** `lab_infra.py` (Docker client + API), `lab_infra_ui.py` (page), `scripts/probe/infra_dashboard_probe.py` (offline checks)

One page that answers: *what containers exist, what runs inside each, what talks to
what, who holds the GPU, what images/volumes/networks exist, and start/stop them.*
It is **common to both labs** — the same router is mounted in `tts_lab.py` and
`image_lab.py`, and each lab's UI links to it from a new tab.

---

## Why it is one module, not two

The two labs differ only in *where they run*:

| | TTS Lab | Image Lab |
|---|---|---|
| Where | `tts-lab-orchestrator` container, port 8009 | bare-metal `arthur-imglab.service`, port 8002 |
| Docker access | `/var/run/docker.sock` bind-mounted (already in `docker-compose.yml`) | same socket; the unit runs as **root** |
| Engines | 30 in engine containers | 11 engines in-process |

Everything the dashboard needs is in the Docker API, so a single module serves
both. There is no per-lab code path — `docker_available()` is the only branch,
and it degrades to a banner when the socket is missing.

## How it talks to Docker

The Engine API over the unix socket via `httpx.HTTPTransport(uds=...)` — **no
`docker` CLI and no `docker` Python package**, so it works in the ML-free
orchestrator image and on bare metal alike. This mirrors the approach already
used by `tts_lab_dispatch.py`'s VRAM eviction helpers.

```
GET /containers/json?all=1     → id/name list
GET /containers/{id}/json      → parallel inspect (10 workers), flattened
GET /containers/{id}/stats?stream=false&one-shot=true → CPU/mem
GET /containers/{id}/logs      → multiplexed stream, demuxed server-side
POST /containers/{id}/{act}    → start|stop|restart|pause|unpause|kill
GET /images/json · /networks · /volumes · /info · /version · /system/df
```

## What makes it lab-specific: the derived views

Three things are **computed from the containers themselves** rather than
configured, which is why the page cannot drift out of date:

1. **Engine → container routing.** Every `*_URL=http://host:port` env var in a
   container's config becomes a directed edge. The compose service name is
   resolved against each container's name + network aliases, so
   `PIPER_URL=http://engine-current:8101` draws `orchestrator → engine-current`
   and labels it with all 21 engines routed there. Nothing is hardcoded: change
   `docker-compose.yml`, reload the page.
2. **Pipe health.** Each target is probed at `http://<container-ip>:<port>/health`
   (2.5 s timeout, 8-way parallel, 8 s cached). Probing the **container IP**
   rather than the compose DNS name is deliberate — it works from the
   orchestrator container *and* from the bare-metal image lab, because the
   bridge network is routable from the host. The response also yields the
   resident engine, which is how "what is running on each container" is shown.
3. **The GPU contention view.** The two labs do not share an HTTP pipe; they
   share **one 16 GB card** and evict each other. That coupling is expressed
   nowhere in the config, so it is derived: every container's `/health` `gpu`
   block and every bare-metal unit's `/status` (`gpu` + `vram`) are normalised
   into one card-level view, and rendered as dashed `GPU ↔ holder` edges plus a
   "VRAM holders" table. This is the single most useful cross-lab signal.

The module also knows the bare-metal units that are *not* containers
(`arthur-imglab`, `arthur-comfy`, and the two retired TTS units) and probes them
on `127.0.0.1`, so "systemd says active" is distinguished from "and it answers".

## Safety rails

* Actions are limited to containers named `tts-lab-*` unless `INFRA_ALLOW_ALL=1`.
* `tts-lab-gpu-probe` is protected — it is an AutoRemove helper the TTS dispatch
  layer recreates on demand.
* **Stopping or killing the container that serves the page** is refused with
  HTTP 409 (`?force=1` overrides). Otherwise the dashboard would cut its own
  transport mid-request and the UI would look broken rather than stopped.
* `INFRA_READONLY=1` disables every mutating endpoint.
* Env values matching `TOKEN|SECRET|PASS|KEY|CREDENTIAL|API` are blurred until
  clicked, so `HF_TOKEN` is not readable over someone's shoulder.

## Endpoints

| Method | Path | Notes |
|---|---|---|
| `GET` | `/infra` | the dashboard page |
| `GET` | `/infra/api/overview` | everything: containers + edges + pipes + images + networks + volumes + gpu + host. 2 s cache |
| `GET` | `/infra/api/containers/{name}` | full flattening: env, mounts, networks, ports, health, restart policy |
| `GET` | `/infra/api/containers/{name}/logs?tail=300&grep=` | demuxed text |
| `GET` | `/infra/api/containers/{name}/stats` | one-shot CPU/mem |
| `POST` | `/infra/api/containers/{name}/action/{act}` | `start stop restart pause unpause kill` |
| `POST` | `/infra/api/lab/{lab}/action/{act}` | bulk over a lab (`start stop restart`) |
| `GET` | `/infra/api/images` · `/networks` · `/volumes` | with "used by" attribution |
| `GET` | `/infra/api/host` | `docker info`, RAM, load, disks, systemd units (+ 127.0.0.1 probe) |
| `GET` | `/infra/api/disk` | `/system/df` — layer/volume/build-cache sizes, 60 s cache |
| `GET` | `/infra/api/pipes` · `/units` | refreshed on demand |

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `INFRA_READONLY` | off | disable all actions |
| `INFRA_ALLOW_ALL` | off | allow actions on containers outside `tts-lab-*` |
| `INFRA_PIPE_PROBE` | on | set `0` to skip the HTTP health probes |
| `INFRA_CACHE_TTL` / `INFRA_PROBE_TTL` | `2` / `8` s | overview / pipe cache |
| `INFRA_DOCKER_SOCK` | `/var/run/docker.sock` | socket path |
| `DOCKER_HOST` | — | `tcp://host:port` or `unix:///path` override |
| `INFRA_TTS_PORT` / `INFRA_IMAGE_PORT` / `INFRA_COMFY_PORT` | `8009` / `8002` / `8188` | lab links + unit probes |

---

## Verification

`scripts/probe/infra_dashboard_probe.py` runs offline — no VM, no Docker, no
browser required:

```bash
python scripts/probe/infra_dashboard_probe.py            # static checks
python scripts/probe/infra_dashboard_probe.py --open     # + render probe
```

It builds a synthetic 12-container fleet through the **real** `lab_infra`
functions and asserts:

* every graph edge endpoint and every GPU holder resolves to a node that exists
  (bare-metal nodes are keyed `unit:<id>`, so an edge built from a display label
  points at nothing and is silently dropped by the SVG renderer — this happened);
* every `$('id')` reference in the JS has a matching element;
* no renderer injects table rows into a non-`<table>` element — the parser drops
  `<thead>/<tbody>` and the rows collapse into an unformatted text blob (this
  happened to the Filesystems card; it renders as one run-on line, not as an
  error, so only a check catches it).

It then writes `infra_probe.html` — the real dashboard with `fetch` mocked
per-endpoint and a `window.onerror` collector that prints a red banner. Open it
and a green screen means every panel rendered; the checks and the page were both
used to find the three defects above during development.

Exits non-zero on failure (via `sys.exit`, not `return 1` — see the PowerShell
footgun notes: `return 1` at script scope does not set an exit code).

---

## Deploying it

The two labs load the page in **different ways**, and that asymmetry is the
whole procedure — deploying to one and not the other leaves the labs serving
two different dashboards:

| | TTS Lab (`:8009`) | Image Lab (`:8002`) |
|---|---|---|
| Code lives | **baked into the orchestrator image** (`Dockerfile.orchestrator` `COPY lab_infra.py`) | on disk in `/opt/arthur-img/`, read at process start |
| Takes effect by | `docker build` + `docker compose up -d` | `systemctl restart arthur-imglab.service` |

VM paths: checkout `/opt/arthur-tts-lab` (compose project `arthur-tts-lab`,
config `docker-compose.yml`, service `orchestrator`), bare-metal lab
`/opt/arthur-img`. `lab_infra.py` / `lab_infra_ui.py` are **untracked** in the VM
checkout, so the files are copied individually — a `git pull` does not deliver
them.

```bash
# 1. copy to BOTH locations
scp lab_infra.py lab_infra_ui.py arthur@192.168.0.87:/opt/arthur-tts-lab/
scp lab_infra.py lab_infra_ui.py arthur@192.168.0.87:/opt/arthur-img/

# 2. TTS lab — rebuild. Only the two COPY layers change, so it is ~2 s warm
cd /opt/arthur-tts-lab && docker build -f docker/Dockerfile.orchestrator \
    -t tts-lab-orchestrator:latest .

# 3. TTS lab — recreate the container
cd /opt/arthur-tts-lab && docker compose up -d --no-deps orchestrator

# 4. Image Lab — reload the bare-metal service
sudo systemctl restart arthur-imglab.service
```

⚠️ **`docker compose up -d orchestrator` without `--no-deps` also starts
`tts-lab-engine-qwen`** — it is a declared dependency and sits in the default
profile, and it is deliberately left stopped. The deploy silently re-creates it
(and loads its model into VRAM); stop it again afterwards, or use `--no-deps`.

### Confirming the deploy actually took

A `md5sum` comparison against the local file is *not* sufficient: the
orchestrator serves the copy **inside its image**, so after a `docker build` a
container that was not recreated still runs the old code — and the file on the
host looks correct the whole time.

```bash
docker exec tts-lab-orchestrator md5sum /opt/arthur/lab_infra.py   # == local hash?
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8009/infra
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8002/infra
```

Timings are the cheapest behavioural check. A cold `/infra/api/overview` costs
1–3 s (it samples container stats); it is then cached, so 10 ms responses mean
the caching layer is alive. The five endpoints that rebuild the container list
(`overview`, `images`, `networks`, `volumes`, `pipes`) share the 2 s
`INFRA_RAW_TTL` inventory cache, so hitting them back-to-back should cost ~75 ms
each after the first — if they cost ~0.95 s each, the cache is not in the build.

Rollback is a file restore plus the same three actions: nothing in the dashboard
is stateful, so there is nothing to migrate in either direction.
