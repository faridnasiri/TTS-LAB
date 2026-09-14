#!/usr/bin/env python3
"""Check TTS Lab UI for JavaScript/HTML issues."""
import re, sys
from collections import Counter

with open("/tmp/page.html") as f:
    html = f.read()

errors = []

# 1. Extract script blocks
scripts = re.findall(r"<script>(.*?)</script>", html, re.DOTALL)
all_js = "\n".join(scripts)

# 2. Check JS brace/paren balance
opens = all_js.count("{")
closes = all_js.count("}")
if opens != closes:
    errors.append(f"JS brace imbalance: {{ {opens} vs }} {closes}")

opens_p = all_js.count("(")
closes_p = all_js.count(")")
if opens_p != closes_p:
    errors.append(f"JS paren imbalance: ( {opens_p} vs ) {closes_p}")

# 3. Check for broken regex (split across lines)
for i, s in enumerate(scripts):
    lines = s.split("\n")
    for j, line in enumerate(lines):
        stripped = line.strip()
        if "replace(/" in stripped:
            # Check if regex closes on same line
            rest = stripped[stripped.index("replace(/")+9:]
            if "/g" not in rest and "/gi" not in rest and "/i" not in rest:
                # Might be continued on next line - check
                if j+1 < len(lines) and ("/g" in lines[j+1] or "/gi" in lines[j+1]):
                    errors.append(f"Script {i} line {j}: regex continues on next line: {stripped[:60]}")

# 4. Check for duplicate IDs
ids = re.findall(r'id="([^"]+)"', html)
dupes = {k: v for k, v in Counter(ids).items() if v > 1}
if dupes:
    for k, v in dupes.items():
        errors.append(f"Duplicate ID: '{k}' appears {v} times")

# 5. Check key functions exist
key_funcs = ["selectEngine", "refreshStatus", "synth", "refreshAvailability"]
for f in key_funcs:
    if f"function {f}" not in all_js and f"async function {f}" not in all_js:
        errors.append(f"Missing function: {f}")

print(f"Script blocks: {len(scripts)}")
print(f"Total JS size: {len(all_js)} chars")
print(f"Issues found: {len(errors)}")
for e in errors:
    print(f"  FAIL: {e}")
if not errors:
    print("  PASS: No issues found")

# Show first 5 duplicate IDs
print()
for k, v in Counter(ids).most_common(10):
    if v > 1:
        print(f"  DUPE ID: {k} ({v}x)")
