#!/usr/bin/env python3
"""Decide which container images a push needs to rebuild.

One table is the single source of truth for CI coverage: every image, the
Dockerfile that builds it, its parent image, and the files that invalidate it.

This replaced a per-image list of `dorny/paths-filter` globs in the workflow,
and the replacement is the point: that list silently omitted engine-qwen,
engine-mid, engine-fa, engine-editx, the sglang image and both py311 stacks, so
those images were never built by CI at all — and nothing reported the omission.
Coverage is now a property of the table, not of remembering to edit two places.

Invalidation is derived from what each image actually copies, not guessed: the
source lists below mirror the COPY lines of each Dockerfile (verified against
them), and a child also inherits a rebuild when an *ancestor's Dockerfile*
changes, because that changes the layer it is built on top of. App-code changes
inherit only where the file is really copied — `tts_lab_ui.py` invalidates base
and the orchestrator, but no engine, because no engine copies the UI.

Emits three JSON arrays (tier 1 = base, tier 2 = stacks, tier 3 = engines and
apps) to $GITHUB_OUTPUT for a matrix build, so a tier only starts once the tier
below it has been pushed and can be pulled as its parent.

Usage:
    python3 build_plan.py                 # reads BEFORE_SHA / AFTER_SHA / IMAGE_PREFIX
    python3 build_plan.py --all           # plan every image (manual dispatch)
    python3 build_plan.py --list          # print the table
    python3 build_plan.py --explain NAME  # why would this image rebuild?
"""
from __future__ import annotations

import fnmatch
import json
import os
import subprocess
import sys

# Parents outside the lab (used only as the BASE_IMAGE value; their Dockerfiles
# do not consume it, so it is inert metadata for those rows).
CUDA22 = "nvidia/cuda:12.8.2-runtime-ubuntu22.04"
CUDA24 = "nvidia/cuda:12.8.2-runtime-ubuntu24.04"
CUDA121 = "nvidia/cuda:12.1-runtime-ubuntu22.04"
CUDA_DEVEL = "nvidia/cuda:12.8.2-devel-ubuntu22.04"

# Files copied by every image that carries the application code (base + the
# orchestrator use this set or a subset of it).
APP_SOURCES = [
    "tts_lab.py", "tts_lab_shims.py", "tts_lab_config.py", "tts_lab_utils.py",
    "tts_lab_engines.py", "tts_lab_dispatch.py", "tts_lab_ui.py",
    "tts_lab_history.py", "lab_infra.py", "lab_infra_ui.py",
]
# Copied by the engine images (see their COPY lines).
ENGINE_SOURCES = [
    "tts_lab_shims.py", "tts_lab_config.py", "tts_lab_engines.py",
    "tts_lab_dispatch.py", "tts_lab_engine_server.py",
]

# tier, image, dockerfile, parent, sources, needs-free-disk
TABLE = [
    (1, "tts-lab-base", "docker/Dockerfile.base", CUDA22,
     ["docker/Dockerfile.base", "patches/**"] + APP_SOURCES, False),
    (1, "tts-lab-base-py311", "docker/Dockerfile.base-py311", CUDA22,
     ["docker/Dockerfile.base-py311"], False),

    (2, "tts-lab-stack-current", "docker/Dockerfile.stack.current", "tts-lab-base",
     ["docker/Dockerfile.stack.current"], False),
    (2, "tts-lab-stack-legacy", "docker/Dockerfile.stack.legacy", "tts-lab-base",
     ["docker/Dockerfile.stack.legacy"], False),
    (2, "tts-lab-stack-mid", "docker/Dockerfile.stack.mid", "tts-lab-base",
     ["docker/Dockerfile.stack.mid"], False),
    (2, "tts-lab-stack-py311", "docker/Dockerfile.stack-py311", "tts-lab-base-py311",
     ["docker/Dockerfile.stack-py311"], False),

    # engine-current also copies voice_library.py; the legacy engine copies only
    # the legacy shim and the engine server.
    (3, "tts-lab-engine-current", "docker/Dockerfile.engine-current", "tts-lab-stack-current",
     ["docker/Dockerfile.engine-current", "voice_library.py"] + ENGINE_SOURCES, False),
    (3, "tts-lab-engine-legacy", "docker/Dockerfile.engine-legacy", "tts-lab-stack-legacy",
     ["docker/Dockerfile.engine-legacy", "tts_lab_shims_legacy.py",
      "tts_lab_engine_server.py"], False),
    (3, "tts-lab-engine-mid", "docker/Dockerfile.engine-mid", "tts-lab-stack-mid",
     ["docker/Dockerfile.engine-mid"] + ENGINE_SOURCES, False),
    (3, "tts-lab-engine-qwen", "docker/Dockerfile.engine-qwen", "tts-lab-stack-mid",
     ["docker/Dockerfile.engine-qwen"] + ENGINE_SOURCES, False),
    # The py311 pair copies `tts_lab*.py` as a glob, plus pyproject.toml.
    (3, "tts-lab-engine-py311", "docker/Dockerfile.engine-py311", "tts-lab-stack-py311",
     ["docker/Dockerfile.engine-py311", "pyproject.toml", "tts_lab*.py"], False),
    (3, "tts-lab-engine-omni", "docker/Dockerfile.engine-omni", "tts-lab-stack-py311",
     ["docker/Dockerfile.engine-omni", "pyproject.toml", "tts_lab*.py"], False),
    # editx and fa build straight on a public CUDA image, so no lab ancestor.
    (3, "tts-lab-engine-editx", "docker/Dockerfile.engine-editx", CUDA24,
     ["docker/Dockerfile.engine-editx"] + ENGINE_SOURCES, True),
    (3, "tts-lab-engine-fa", "docker/Dockerfile.engine-fa", CUDA24,
     ["docker/Dockerfile.engine-fa"] + ENGINE_SOURCES, False),

    (3, "tts-lab-orchestrator", "docker/Dockerfile.orchestrator", "tts-lab-base",
     ["docker/Dockerfile.orchestrator", "tts_lab.py", "tts_lab_config.py",
      "tts_lab_ui.py", "tts_lab_dispatch.py", "tts_lab_history.py",
      "lab_infra.py", "lab_infra_ui.py"], False),
    (3, "tts-lab-orpheus", "docker/Dockerfile.orpheus", CUDA121,
     ["docker/Dockerfile.orpheus", "tts_lab_orpheus_server.py",
      "tts_lab_shims.py", "tts_lab_config.py", "tts_lab_utils.py",
      "tts_lab_engines.py"], False),
    # sglang-omni's only copied file is the S2-Pro engine config, not app code.
    (3, "tts-lab-sglang-omni", "docker/Dockerfile.sglang", CUDA_DEVEL,
     ["docker/Dockerfile.sglang", "docker/s2pro_tts.yaml"], True),
]

BY_NAME = {row[1]: row for row in TABLE}


def effective_sources(name: str) -> list[str]:
    """Own sources, plus the Dockerfiles of every lab ancestor."""
    row = BY_NAME[name]
    files = list(row[4])
    parent = row[3]
    while parent in BY_NAME:
        files.append(BY_NAME[parent][2])
        parent = BY_NAME[parent][3]
    return files


def changed_files(before: str, after: str) -> list[str]:
    if not before or set(before) == {"0"}:
        return ["<all>"]            # first push to a branch: assume everything
    out = subprocess.run(["git", "diff", "--name-only", before, after],
                         capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line.strip()]


def plan(changed: list[str], prefix: str) -> dict[str, list[dict]]:
    tiers: dict[str, list[dict]] = {"tier1": [], "tier2": [], "tier3": []}
    everything = changed == ["<all>"]
    for tier, name, dockerfile, parent, _own, freespace in TABLE:
        if everything:
            hits = ["<all>"]
        else:
            hits = [f for f in changed
                    if any(fnmatch.fnmatch(f, src) for src in effective_sources(name))]
        if not hits:
            continue
        parent_ref = (f"{prefix}/{parent}:latest"
                      if parent.startswith("tts-lab-") else parent)
        tiers[f"tier{tier}"].append({
            "image": name,
            "dockerfile": dockerfile,
            "parent": parent_ref,
            "tag": f"{prefix}/{name}:latest",
            "freespace": freespace,
        })
    return tiers


def main() -> int:
    prefix = os.environ.get("IMAGE_PREFIX", "ghcr.io/faridnasiri/tts-lab")

    # Every emitted tag/parent starts with this string. Without the registry host
    # the refs read as Docker Hub paths, and the failure shows up much later and
    # much less clearly ("pull access denied, repository does not exist") in a
    # build job rather than here.
    if "." not in prefix.split("/")[0]:
        print(f"::error::IMAGE_PREFIX must include the registry host, got {prefix!r}")
        return 1

    if "--list" in sys.argv:
        for tier, name, dockerfile, parent, own, free in TABLE:
            anc = [s for s in effective_sources(name) if s not in own]
            print(f"t{tier} {name:24} {dockerfile:34} parent={parent:24} "
                  f"free={free!s:5} own={len(own)} inherited={anc}")
        return 0

    if "--explain" in sys.argv:
        name = sys.argv[sys.argv.index("--explain") + 1]
        print(f"{name}:")
        print(f"  dockerfile : {BY_NAME[name][2]}")
        print(f"  parent     : {BY_NAME[name][3]}")
        print(f"  rebuilds on: {sorted(effective_sources(name))}")
        return 0

    if "--all" in sys.argv or os.environ.get("FORCE_ALL"):
        print("planning every image (manual run)")
        changed = ["<all>"]
    else:
        before = os.environ.get("BEFORE_SHA", "")
        after = os.environ.get("AFTER_SHA", "HEAD")
        changed = changed_files(before, after)
        print(f"changed files ({len(changed)}): {', '.join(changed[:20])}")

    tiers = plan(changed, prefix)
    total = 0
    for key, rows in tiers.items():
        names = [r["image"] for r in rows]
        total += len(names)
        print(f"{key}: {names}")
    print(f"planned {total} of {len(TABLE)} images")

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as fh:
            for key, rows in tiers.items():
                fh.write(f"{key}={json.dumps(rows)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
