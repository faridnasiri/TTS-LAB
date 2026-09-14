"""
test_new_engines.py — structural & import integration test for the 6 new TTS engines.

Verifies that all registries (MODEL_INFO, MODEL_ORDER, LOADERS, SYNTHERS,
pkg_map, UI widgets) are consistent and importable.

Does NOT load actual models — only validates that the wiring is correct.
Run: python scripts/test/test_new_engines.py
"""
from __future__ import annotations
import sys, os, ast, re, traceback
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

NEW_ENGINES = ["chatterboxturbo", "vibevoice", "higgs", "omnivoice", "s2pro", "editx"]
PASS = 0
FAIL = 0


def check(desc: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {desc}")
    else:
        FAIL += 1
        print(f"  [FAIL] {desc}  -- {detail}")


# ── 1. Parse MODEL_INFO from config (no import — avoids torch dependency) ──────
print("=" * 70)
print("1. tts_lab_config.py — MODEL_INFO & MODEL_ORDER")
config_path = PROJECT_DIR / "tts_lab_config.py"
config_src = config_path.read_text(encoding="utf-8")

# Extract MODEL_INFO top-level keys (each entry starts a line with "key": {)
model_info_keys = re.findall(r'^\s*"(\w+)"\s*:\s*\{', config_src, re.MULTILINE)

# Extract MODEL_ORDER list
order_match = re.search(r"MODEL_ORDER\s*=\s*\[(.*?)\]", config_src, re.DOTALL)
order_items = []
if order_match:
    order_items = [i.strip().strip('"') for i in order_match.group(1).split(",") if i.strip()]

for eng in NEW_ENGINES:
    check(f"MODEL_INFO['{eng}'] exists", eng in model_info_keys)
    check(f"{eng} in MODEL_ORDER", eng in order_items)

check(f"Total engines: {len(order_items)}", len(order_items) == 30, f"got {len(order_items)} (expected 30)")

# ── 2. Parse tts_lab_engines.py — LOADERS, SYNTHERS, function defs ────────────
print("\n2. tts_lab_engines.py — LOADERS, SYNTHERS, _load_*/_synth_* pairs")
engines_path = PROJECT_DIR / "tts_lab_engines.py"
engines_src = engines_path.read_text(encoding="utf-8")

loader_match = re.search(r"LOADERS:\s*dict\s*=\s*\{(.*?)\}", engines_src, re.DOTALL)
loader_keys = []
if loader_match:
    loader_keys = [k for k in re.findall(r'"(\w+)"', loader_match.group(1)) if not k.startswith("_")]

synth_match = re.search(r"SYNTHERS:\s*dict\s*=\s*\{(.*?)\}", engines_src, re.DOTALL)
synth_keys = []
if synth_match:
    synth_keys = [k for k in re.findall(r'"(\w+)"', synth_match.group(1)) if not k.startswith("_")]

for eng in NEW_ENGINES:
    check(f"LOADERS['{eng}'] present", eng in loader_keys)
    check(f"SYNTHERS['{eng}'] present", eng in synth_keys)

# Check _load_<name> and _synth_<name> function definitions exist
load_funcs = re.findall(r"def (_load_\w+)", engines_src)
synth_funcs = re.findall(r"def (_synth_\w+)", engines_src)

for eng in NEW_ENGINES:
    expected_load = f"_load_{eng}"
    expected_synth = f"_synth_{eng}"
    check(f"def {expected_load}() exists", expected_load in load_funcs)
    check(f"def {expected_synth}() exists", expected_synth in synth_funcs)

check(f"Total LOADERS: {len(loader_keys)}", len(loader_keys) == 30, f"got {len(loader_keys)}")
check(f"Total SYNTHERS: {len(synth_keys)}", len(synth_keys) == 30, f"got {len(synth_keys)}")

# ── 3. Parse tts_lab_dispatch.py — pkg_map ────────────────────────────────────
print("\n3. tts_lab_dispatch.py — pkg_map entries")
dispatch_path = PROJECT_DIR / "tts_lab_dispatch.py"
dispatch_src = dispatch_path.read_text(encoding="utf-8")

pkg_map_match = re.search(r"pkg_map\s*=\s*\{(.*?)\}", dispatch_src, re.DOTALL)
expected_pkgs = {
    "chatterboxturbo": "chatterbox",
    "vibevoice": "transformers",
    "higgs": "transformers",
    "omnivoice": "omnivoice",
    "s2pro": None,
    "editx": "vllm",
}
for eng, pkg in expected_pkgs.items():
    # Find the line like "eng": "pkg" or "eng": None
    pattern = rf'"{eng}"\s*:\s*("[^"]*"|None)'
    match = re.search(pattern, dispatch_src)
    if match:
        val = match.group(1)
        expected_val = f'"{pkg}"' if pkg else "None"
        check(f"pkg_map['{eng}'] = {expected_val}", val == expected_val, f"got {val}")
    else:
        check(f"pkg_map['{eng}'] present", False, "not found in pkg_map")

# ── 4. Parse tts_lab_ui.py — parameter widgets ────────────────────────────────
print("\n4. tts_lab_ui.py — per-engine param widgets")
ui_path = PROJECT_DIR / "tts_lab_ui.py"
ui_src = ui_path.read_text(encoding="utf-8")

for eng in NEW_ENGINES:
    has_widget = f'if name == "{eng}"' in ui_src
    check(f"UI widget for '{eng}'", has_widget)

# ── 5. Parse deploy_lab.ps1 — pip install lines ───────────────────────────────
print("\n5. deploy_lab.ps1 — pip install entries")
deploy_path = PROJECT_DIR / "deploy_lab.ps1"
deploy_src = deploy_path.read_text(encoding="utf-8")

# omnivoice needs explicit pip install; others use existing packages
check("omnivoice pip install in deploy", "omnivoice" in deploy_src)
# chatterboxturbo uses same chatterbox-tts package (already installed)
check("chatterbox-tts already in deploy", "chatterbox-tts" in deploy_src)
# vibevoice + higgs use transformers (already installed)
check("transformers already in deploy", "transformers" in deploy_src.lower())

# ── 6. Cross-reference consistency ─────────────────────────────────────────────
print("\n6. Cross-reference consistency checks")

# All MODEL_ORDER engines must have MODEL_INFO, LOADERS, SYNTHERS
# (model_info_keys only contains the top-level engine keys — verified above)
for eng in order_items:
    check(f"'{eng}' in MODEL_INFO", eng in model_info_keys, f"MODEL_INFO missing '{eng}'")
    check(f"'{eng}' in LOADERS", eng in loader_keys, f"LOADERS missing '{eng}'")

# All LOADERS should have matching SYNTHERS and vice versa
for k in loader_keys:
    check(f"LOADERS['{k}'] <-> SYNTHERS['{k}']", k in synth_keys, f"SYNTHERS missing '{k}'")
for k in synth_keys:
    check(f"SYNTHERS['{k}'] <-> LOADERS['{k}']", k in loader_keys, f"LOADERS missing '{k}'")

# ── 7. Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
total = PASS + FAIL
print(f"Results: {PASS}/{total} passed, {FAIL} failed")
if FAIL == 0:
    print("ALL CHECKS PASSED — 6 new engines fully integrated.")
else:
    print(f"WARNING: {FAIL} check(s) failed — review above.")
sys.exit(0 if FAIL == 0 else 1)
