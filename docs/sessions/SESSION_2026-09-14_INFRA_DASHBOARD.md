# Session 2026-09-14 — Container / Infrastructure dashboard (`/infra`)

> Host: `arthur@192.168.0.87` (Proxmox VM 104) · GPU: RTX 5060 Ti 16 GB (sm_120)
> Status: **deployed and verified live on both labs** · Downtime: ~10 s (orchestrator recreate) + imglab restart
> Reference (API surface, env vars, design rationale): [`docs/reference/CONTAINER_DASHBOARD.md`](../reference/CONTAINER_DASHBOARD.md)

## TL;DR

One page that answers *what containers exist, what runs inside each, what talks to what, who
holds the GPU, and start/stop them* — served identically by **both** labs, because it is one
router mounted twice:

* `http://192.168.0.87:8009/infra` — TTS orchestrator container (port 8009)
* `http://192.168.0.87:8002/infra` — Image Lab, bare-metal `arthur-imglab.service` (port 8002)

It talks to the Docker Engine API **over the unix socket** (`httpx` + `uds=`), so it needs no
`docker` CLI and no `docker` Python package — which matters because the orchestrator image
deliberately contains no ML libraries and the image lab runs outside Docker entirely.

Three of the views are **derived from the containers themselves** rather than configured, so the
page cannot drift out of date: engine→container routing (parsed from `*_URL` env vars and
resolved against container names/network aliases), pipe health (HTTP `/health` probes at
container IP:port), and GPU contention (from each engine's `/health` `gpu` block + the image
lab's `/status`).

## Wiring — six places that must stay in sync

| Place | Change |
|---|---|
| `lab_infra.py` (new) | Docker client + FastAPI router; `docker_available()` is the only branch |
| `lab_infra_ui.py` (new) | the page, inlined HTML/CSS/JS |
| `tts_lab.py` | `from lab_infra import router as infra_router` + `app.include_router(...)` |
| `image_lab.py` | same two lines |
| `docker/Dockerfile.orchestrator` | `COPY lab_infra.py` + `COPY lab_infra_ui.py` |
| `scripts/deploy/deploy_lab.ps1`, `scripts/deploy/deploy_image_lab.ps1` | file lists (+ the `ast` syntax-check list in `deploy_lab.ps1` — 7 → 9 modules) |

UI entry points: TTS sidebar button `showInfra()` / `#pane-infra` / header `#btn-infra`; Image
Lab `data-view="infra"` tab / `#viewInfra`. Both are iframes whose `src` is attached **on first
open** so the dashboard does not poll the daemon at boot.

⚠️ `tts_lab.py` imports `lab_infra` at module scope, so a deploy that copies `tts_lab.py` without
`lab_infra.py` ships a lab that dies on `ImportError`.

## The three defects found during development

All three were **silent** — no exception, no console error, just wrong output. They are the
reason `scripts/probe/infra_dashboard_probe.py` exists.

| Symptom | Cause |
|---|---|
| The Filesystems card rendered as one run-on line of text | the renderer assigned `<thead>/<tbody>` markup into a **`<div>`**; the parser drops them, so the table structure vanished |
| The GPU ↔ Image Lab link was missing from the topology, with no error | bare-metal unit nodes are keyed `unit:<id>` while the GPU holder's display label was its human name → `pos[e.to]` was `undefined` |
| A freshly opened tab stayed empty, reading as "no data" | renderers ran only from the 5 s poll; `setTab()` must call the panel's renderer itself |

## Deployment — the asymmetry is the whole procedure

| | TTS Lab (`:8009`) | Image Lab (`:8002`) |
|---|---|---|
| Code lives | **baked into the orchestrator image** | on disk in `/opt/arthur-img/` |
| Takes effect by | `docker build` + `docker compose up -d` | `systemctl restart arthur-imglab.service` |

`lab_infra*.py` are **untracked** in the VM checkout (`/opt/arthur-tts-lab`), so the files are
copied individually — a `git pull` does not deliver them.

```bash
scp lab_infra.py lab_infra_ui.py arthur@192.168.0.87:/opt/arthur-tts-lab/
scp lab_infra.py lab_infra_ui.py arthur@192.168.0.87:/opt/arthur-img/
cd /opt/arthur-tts-lab && docker build -f docker/Dockerfile.orchestrator \
    -t tts-lab-orchestrator:latest .          # only the two COPY layers change (~2 s warm)
cd /opt/arthur-tts-lab && docker compose up -d --no-deps orchestrator
sudo systemctl restart arthur-imglab.service
```

Two traps, both hit for real:

1. **`docker compose up -d orchestrator` without `--no-deps` also starts `tts-lab-engine-qwen`**
   (declared dependency, default profile, deliberately left stopped). It re-created the container
   and began loading its model into VRAM; stopped again afterwards — GPU back to 271 MiB.
2. **An `md5sum` on the host is not proof of deployment.** The orchestrator serves the copy
   *inside its image*, so a container that was not recreated keeps running the old code while the
   host file looks correct. Compare `docker exec tts-lab-orchestrator md5sum /opt/arthur/lab_infra.py`
   — not the host path — and then check behaviour.

`.gitignore` also had to change: `scripts/` was ignored wholesale even though 40 files under it
were already tracked, which silently blocked the new probe/utility scripts (and the NVIDIA driver
repair scripts from the previous session) from ever being committed.

## Verification performed

Offline (`scripts/probe/infra_dashboard_probe.py`, no VM/Docker/browser needed): a synthetic
12-container fleet run through the **real** `lab_infra` functions, asserting every graph edge
endpoint and GPU holder resolves to a node, every `$('id')` has a matching element, and no
renderer injects table rows into a non-`<table>`; then a render probe writing `infra_probe.html`
with `fetch` mocked per endpoint and a `window.onerror` collector.

Live, after the deploy:

| Check | Result |
|---|---|
| `/infra` on `:8009` and `:8002` | 200, identical 70 557 B page |
| all 11 endpoints (`overview host disk pipes units images networks volumes`, container detail/stats/logs) | 200 with real data |
| browser, `:8009/infra` | full render: nav `Containers 3/9`, `Images 12`, 15 topology nodes (119 SVG elements), pipes table 5 rows with health + latency, disk 3, units 5, GPU holders 1, PID/VRAM 2 — **no red error banner, 0 JS errors** |
| browser, `:8002` → Infrastructure tab | iframe lazily attaches and the nested dashboard renders |
| header facts | docker 29.6.0 / api 1.55 · 3/9 running · 12 images · RAM 6 % (4.0/63 GB) · pipes 1/5 up |

## Performance notes (measured, not assumed)

* Cold `/infra/api/overview` = 1–3 s — it samples container stats. It is then cached, so warm
  responses are ~10 ms.
* The five endpoints that rebuild the container inventory (`overview`, `images`, `networks`,
  `volumes`, `pipes`) were each paying ~0.95 s for a listing + N inspects + a stats sample.
  They now share a 2 s `INFRA_RAW_TTL` cache: `images` 2.78 s cold, then `networks` **0.078 s**,
  `volumes` **0.074 s**, repeat `images` 0.005 s.
* `one-shot=true` stats still need `precpu_stats` for a CPU %; container `Id` has **no** `sha256:`
  prefix (images do), which is why `[:12]` is clean hex for containers.

## State left behind

* The dashboard is live on both labs and in sync with the working tree, except one deliberate
  difference: the VM's `tts_lab.py` carries only the two router edits
  (`scripts/utils/patch_tts_lab_infra.py`), because the working tree also holds an unrelated
  `xttsfa` entry in the `/voices` map that must not be deployed.
* Still uncommitted from other work streams: the Persian/`engine-fa` engine, the NVIDIA driver
  hardening scripts, and the `/infra`-adjacent repo-hygiene edits.
* The rollout is not under Ansible or CI yet — it is the `scp` + rebuild sequence above, written
  down here because it is asymmetric and easy to half-do.
