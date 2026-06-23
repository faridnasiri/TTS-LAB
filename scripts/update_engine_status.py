#!/usr/bin/env python3
"""
update_engine_status.py — runtime gate updater for engine_compatibility.yaml

Prevents documentation drift: the test harness writes validation results
directly into the compatibility matrix instead of requiring manual editing.

Usage (CLI):
  python scripts/update_engine_status.py vibevoice model_load passed --duration 41 --vram-mb 6420
  python scripts/update_engine_status.py qwen3tts build_import failed --error "ROPE_INIT_FUNCTIONS missing default key"
  python scripts/update_engine_status.py vibevoice --promote   # promote if all gates passed

Usage (Python):
  from scripts.update_engine_status import update_gate, load_matrix, save_matrix
  m = load_matrix()
  update_gate(m, "vibevoice", "model_load", "passed", duration_s=41, vram_mb=6420)
  save_matrix(m)

Auto-populated fields:
  - last_tested: ISO-8601 timestamp (now)
  - validated_on: {torch, transformers, cuda, driver} from runtime
  - duration_seconds: optional, from --duration
  - vram_mb: optional, from --vram-mb
  - error: optional, from --error (stored on failure)
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:
    sys.exit(
        "PyYAML is required: pip install pyyaml\n"
        "Or: pip install ruamel.yaml"
    )

# ── Paths ──────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
MATRIX_PATH = REPO_ROOT / "docs" / "engine_compatibility.yaml"


# ── Runtime fingerprint ─────────────────────────────────────────────

_FP_SCRIPT = r"""
import json
result = {}
try:
    import torch
    result['torch'] = str(torch.__version__)
    if torch.cuda.is_available():
        result['cuda'] = str(torch.version.cuda)
except Exception:
    pass
try:
    import transformers
    result['transformers'] = str(transformers.__version__)
except Exception:
    pass
print(json.dumps(result))
"""


def _collect_from_host() -> Dict[str, Optional[str]]:
    """Collect fingerprint from the current Python process."""
    fp: Dict[str, Optional[str]] = {}
    try:
        import torch
        fp["torch"] = str(torch.__version__)
        if torch.cuda.is_available():
            fp["cuda"] = str(torch.version.cuda)
    except ImportError:
        pass
    try:
        import transformers
        fp["transformers"] = str(transformers.__version__)
    except ImportError:
        pass
    # Driver version is always collected from the host
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, timeout=10,
        ).strip()
        fp["driver"] = out.split("\n")[0].strip() if out else None
    except Exception:
        fp["driver"] = None
    return fp


def _collect_from_container(container: str) -> Dict[str, Optional[str]]:
    """Collect fingerprint from a running Docker container."""
    try:
        out = subprocess.check_output(
            ["docker", "exec", container, "python3", "-c", _FP_SCRIPT],
            text=True, timeout=30,
        ).strip()
        fp = json.loads(out)
    except FileNotFoundError:
        print(f"WARNING: docker not found -- is Docker installed? "
              f"Falling back to host fingerprint.",
              file=sys.stderr)
        return _collect_from_host()
    except subprocess.CalledProcessError as e:
        print(f"WARNING: docker exec failed for container '{container}': {e.stderr.strip() if e.stderr else e}",
              file=sys.stderr)
        print("Is the container running? Falling back to host fingerprint.", file=sys.stderr)
        return _collect_from_host()
    except json.JSONDecodeError as e:
        print(f"WARNING: failed to parse container fingerprint: {e}",
              file=sys.stderr)
        return {}
    # Driver version is host-level — collected separately
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, timeout=10,
        ).strip()
        fp["driver"] = out.split("\n")[0].strip() if out else None
    except Exception:
        fp["driver"] = None
    return fp


def collect_fingerprint(container: Optional[str] = None) -> Dict[str, Optional[str]]:
    """Collect the runtime environment fingerprint.

    If `container` is given, introspects that Docker container.
    Otherwise uses the current Python process.
    Driver version is always collected from the host.
    """
    if container:
        return _collect_from_container(container)
    return _collect_from_host()


def _get_vram_used_mb() -> Optional[int]:
    """Current GPU memory usage in MB (first GPU only)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        ).strip()
        return int(out.split("\n")[0].strip()) if out else None
    except Exception:
        return None


# ── Matrix I/O ─────────────────────────────────────────────────────

def load_matrix(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the engine compatibility matrix."""
    p = path or MATRIX_PATH
    if not p.exists():
        raise FileNotFoundError(f"Matrix not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def recompute_summary(matrix: Dict[str, Any]) -> Dict[str, Any]:
    """Derive summary counts from engine entries — single source of truth.

    Called automatically by save_matrix(). No manual summary maintenance.
    """
    engines = matrix.get("engines", {})
    counts = {"total": 0, "supported": 0, "deprecated": 0, "experimental": 0, "blocked": 0}
    containers: Dict[str, Dict[str, int]] = {}

    for entry in engines.values():
        status = entry.get("status", "experimental")
        container = entry.get("container", "unknown")

        counts["total"] += 1
        counts[status] = counts.get(status, 0) + 1

        if container not in containers:
            containers[container] = {"total": 0, "supported": 0, "deprecated": 0, "experimental": 0, "blocked": 0}
        containers[container]["total"] += 1
        containers[container][status] = containers[container].get(status, 0) + 1

    matrix["summary"] = {
        "total": counts["total"],
        "supported": counts["supported"],
        "deprecated": counts["deprecated"],
        "experimental": counts["experimental"],
        "blocked": counts["blocked"],
        "containers": dict(sorted(containers.items())),
    }
    return matrix["summary"]


def save_matrix(matrix: Dict[str, Any], path: Optional[Path] = None) -> None:
    """Write the matrix back to disk, recomputing the summary first."""
    p = path or MATRIX_PATH
    recompute_summary(matrix)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            matrix, f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=120,
        )


# ── Gate update ────────────────────────────────────────────────────

def update_gate(
    matrix: Dict[str, Any],
    engine: str,
    gate: str,
    status: str,
    *,
    duration_s: Optional[float] = None,
    vram_mb: Optional[int] = None,
    error: Optional[str] = None,
    container: Optional[str] = None,
) -> bool:
    """Update a single validation gate for an engine.

    Returns True if the gate was found and updated.
    """
    engines = matrix.get("engines", {})
    if engine not in engines:
        print(f"ERROR: engine '{engine}' not found in matrix", file=sys.stderr)
        return False

    entry = engines[engine]
    validation = entry.setdefault("validation", {})

    if gate not in validation:
        print(f"ERROR: gate '{gate}' not found on engine '{engine}'", file=sys.stderr)
        print(f"Available gates: {list(validation.keys())}", file=sys.stderr)
        return False

    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    gate_data = validation[gate]

    # Save previous attempt to history before overwriting (keep last 5)
    prev_entry = {
        "timestamp": gate_data.get("last_tested"),
        "status": gate_data.get("status"),
    }
    if gate_data.get("error"):
        prev_entry["error"] = gate_data["error"]
    if gate_data.get("duration_seconds") is not None:
        prev_entry["duration_seconds"] = gate_data["duration_seconds"]
    if gate_data.get("peak_vram_mb") is not None:
        prev_entry["peak_vram_mb"] = gate_data["peak_vram_mb"]

    history = gate_data.setdefault("history", [])
    # Only push if there's a meaningful previous state
    if prev_entry["timestamp"] is not None:
        history.append(prev_entry)
        # Keep only the last 5
        if len(history) > 5:
            gate_data["history"] = history[-5:]

    gate_data["status"] = status
    gate_data["last_tested"] = now

    # Clear stale error when flipping to passed
    if status == "passed":
        gate_data.pop("error", None)
    elif error is not None:
        gate_data["error"] = error

    if duration_s is not None:
        gate_data["duration_seconds"] = round(duration_s, 1)

    if vram_mb is not None:
        gate_data["peak_vram_mb"] = vram_mb

    # Auto-populate validated_on fingerprint from target runtime
    fp = collect_fingerprint(container=container)
    validated_on = entry.setdefault("validated_on", {})
    for key in ("torch", "transformers", "cuda", "driver"):
        if fp.get(key) is not None:
            validated_on[key] = fp[key]

    print(f"[OK] {engine}.{gate} -> {status}  ({now})")
    if duration_s is not None:
        print(f"  duration: {duration_s:.1f}s")
    if vram_mb is not None:
        print(f"  vram:     {vram_mb} MB")
    if error is not None:
        print(f"  error:    {error}")
    print(f"  fingerprint: {fp}")

    return True


# ── Promotion ──────────────────────────────────────────────────────

def check_promotable(matrix: Dict[str, Any], engine: str) -> Optional[Dict[str, Any]]:
    """Check if an engine is eligible for promotion.

    Returns a dict with {eligible, passed, failed, pending, missing} or None if
    the engine has no promotion_requirements defined.
    """
    entry = matrix["engines"].get(engine, {})
    required = entry.get("promotion_requirements", [])

    if not required:
        return None

    validation = entry.get("validation", {})
    passed = []
    failed = []
    pending = []
    missing = []

    for gate in required:
        if gate not in validation:
            missing.append(gate)
            continue
        s = validation[gate].get("status", "pending")
        if s == "passed":
            passed.append(gate)
        elif s == "failed":
            failed.append(gate)
        else:
            pending.append(gate)

    eligible = (
        len(failed) == 0
        and len(pending) == 0
        and len(missing) == 0
        and len(passed) == len(required)
    )

    return {
        "eligible": eligible,
        "passed": passed,
        "failed": failed,
        "pending": pending,
        "missing": missing,
        "total_required": len(required),
    }


def promote_engine(matrix: Dict[str, Any], engine: str) -> bool:
    """Promote an engine from EXPERIMENTAL to SUPPORTED if all gates passed.

    Returns True if promotion occurred.
    """
    entry = matrix["engines"].get(engine, {})
    current = entry.get("status", "")

    if current != "experimental":
        print(f"SKIP: {engine} is '{current}', not 'experimental' — cannot promote", file=sys.stderr)
        return False

    result = check_promotable(matrix, engine)
    if result is None:
        print(f"SKIP: {engine} has no promotion_requirements defined", file=sys.stderr)
        return False

    if not result["eligible"]:
        print(f"BLOCKED: {engine} not eligible for promotion", file=sys.stderr)
        if result["failed"]:
            print(f"  failed gates:  {result['failed']}", file=sys.stderr)
        if result["pending"]:
            print(f"  pending gates: {result['pending']}", file=sys.stderr)
        if result["missing"]:
            print(f"  missing gates: {result['missing']}", file=sys.stderr)
        print(f"  passed: {len(result['passed'])}/{result['total_required']}", file=sys.stderr)
        return False

    entry["status"] = "supported"
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry.setdefault("promoted_on", now)
    entry["promoted_on"] = now

    # Summary counts are recomputed automatically by save_matrix() —
    # no manual arithmetic here. Single source of truth.

    print(f"[OK] PROMOTED {engine}: experimental -> supported  ({now})")
    print(f"  gates: {' '.join(result['passed'])}")
    return True


# ── CLI ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update engine compatibility matrix from test results",
    )
    parser.add_argument("engine", nargs="?", help="Engine name (e.g. vibevoice, qwen3tts)")
    parser.add_argument("gate", nargs="?", help="Gate name (e.g. model_load, inference)")
    parser.add_argument("status", nargs="?", choices=["passed", "failed", "pending"],
                        help="Gate status")
    parser.add_argument("--duration", type=float, default=None,
                        help="Gate duration in seconds")
    parser.add_argument("--vram-mb", type=int, default=None,
                        help="Peak VRAM in MB during this gate")
    parser.add_argument("--error", type=str, default=None,
                        help="Error message (on failure)")
    parser.add_argument("--promote", action="store_true",
                        help="Promote engine if all gates passed")
    parser.add_argument("--check", action="store_true",
                        help="Print promotion eligibility without modifying")
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute summary from engine data, fix drift, then exit")
    parser.add_argument("--vram-now", action="store_true",
                        help="Print current VRAM usage and exit")
    parser.add_argument("--fingerprint", action="store_true",
                        help="Print current environment fingerprint and exit")
    parser.add_argument("--matrix", type=str, default=None,
                        help="Path to matrix file (default: docs/engine_compatibility.yaml)")
    parser.add_argument("--container", "-c", type=str, default=None,
                        help="Collect fingerprint from Docker container (e.g. engine-mid)")

    args = parser.parse_args()

    matrix_path = Path(args.matrix) if args.matrix else MATRIX_PATH

    # ── Read-only queries ──────────────────────────────────────────
    if args.vram_now:
        vram = _get_vram_used_mb()
        if vram is not None:
            print(f"{vram}")
        else:
            print("UNAVAILABLE", file=sys.stderr)
            sys.exit(1)
        return

    if args.recompute:
        m = load_matrix(matrix_path)
        summary = recompute_summary(m)
        save_matrix(m, matrix_path)
        print(f"Summary recomputed from engine data:")
        print(f"  total:        {summary['total']}")
        print(f"  supported:    {summary['supported']}")
        print(f"  experimental: {summary['experimental']}")
        print(f"  blocked:      {summary['blocked']}")
        for cname, counts in summary.get("containers", {}).items():
            print(f"  {cname}: {counts}")
        print(f"Matrix updated: {matrix_path}")
        return

    if args.fingerprint:
        fp = collect_fingerprint(container=args.container)
        for k, v in fp.items():
            print(f"{k}: {v}")
        return

    if args.check:
        m = load_matrix(matrix_path)
        result = check_promotable(m, args.engine)
        if result is None:
            print(f"{args.engine}: no promotion_requirements defined")
            return
        status = "ELIGIBLE" if result["eligible"] else "BLOCKED"
        print(f"{args.engine}: {status}")
        print(f"  passed:  {result['passed']}")
        print(f"  failed:  {result['failed']}")
        print(f"  pending: {result['pending']}")
        print(f"  missing: {result['missing']}")
        print(f"  progress: {len(result['passed'])}/{result['total_required']}")
        sys.exit(0 if result["eligible"] else 1)

    # ── Mutating operations ────────────────────────────────────────
    if args.promote:
        if args.gate or args.status:
            print("ERROR: --promote cannot be combined with gate/status", file=sys.stderr)
            sys.exit(1)
        m = load_matrix(matrix_path)
        ok = promote_engine(m, args.engine)
        if ok:
            save_matrix(m, matrix_path)
            print(f"Matrix updated: {matrix_path}")
        sys.exit(0 if ok else 1)

    if not args.engine:
        parser.error("engine is required for gate updates (or use --fingerprint / --vram-now standalone)")
    if not args.gate or not args.status:
        parser.error("gate and status are required (or use --promote, --check, --fingerprint, --vram-now)")

    # ── Gate update ────────────────────────────────────────────────
    m = load_matrix(matrix_path)
    ok = update_gate(
        m, args.engine, args.gate, args.status,
        duration_s=args.duration,
        vram_mb=args.vram_mb,
        error=args.error,
        container=args.container,
    )
    if not ok:
        sys.exit(1)

    save_matrix(m, matrix_path)
    print(f"Matrix updated: {matrix_path}")

    # Auto-check promotion
    result = check_promotable(m, args.engine)
    if result and result["eligible"]:
        print(f"\n>>> {args.engine} is ELIGIBLE for promotion!")
        print(f"    Run: python scripts/update_engine_status.py {args.engine} --promote")


if __name__ == "__main__":
    main()
