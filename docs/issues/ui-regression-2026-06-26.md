# UI Regression — 2026-06-26 Qwen 3.6 LLM Integration

> **Status:** RESOLVED
> **Date:** 2026-06-26
> **Session:** Qwen 3.6 LLM integration (Claude Code)
> **Symptom:** Left sidebar unresponsive (clicking TTS model buttons does nothing), VRAM bars not updating at top of page

---

## Root Causes

Three separate bugs introduced during LLM integration, all in `tts_lab_ui.py`.

### Bug 1 — JavaScript Regex Split Across Lines (CRITICAL)

**Symptom:** ALL JavaScript on page dead. Sidebar not responding, VRAM bars not updating, no debug logs.

**Cause:** The `addChatMessage()` function was inserted via a Python script that used regular (non-raw) strings. The line:

```python
"html = html.replace(/\\n/g, '<br>');"
```

In a regular Python string, `\n` is an actual newline character (0x0A). This was written into the Python source file as:

```python
# Line 1260: html = html.replace(/
# Line 1261: /g, '<br>');
```

The JavaScript regex literal `/.../` was split across two lines. JavaScript regex literals **cannot span lines** — this is a syntax error.

When the browser parsed the `<script>` block, it hit `Invalid regular expression: missing /` and stopped executing ALL JavaScript. Every JS function on the page was dead:
- `selectEngine()` — sidebar buttons
- `refreshStatus()` — VRAM/RAM bars
- `synth()` — synthesis
- `dbg()` — debug logging
- Everything else

**Fix:** Used `sed` to replace the literal newline with `\n` (backslash-n, two characters):
```bash
sed -i '1260s|replace(/$|replace(/\\n/g;|' tts_lab_ui.py
sed -i '1261d' tts_lab_ui.py
```
Result: single line `html = html.replace(/\n/g, '<br>');`

**Lesson:** When inserting JavaScript into a Python raw string (`r"""..."""`), always use raw strings or double-escape backslashes (`\\n`). Any Python string operation on the JS code must preserve `\n` as two characters, not a newline byte.

**Detection:** Node.js `new Function(js)` immediately throws `Invalid regular expression: missing /`. Static brace counting doesn't catch this because braces are balanced — the regex `/` is not a brace.

---

### Bug 2 — Duplicate Button ID

**Symptom:** Evict VRAM button replaced the Refresh button's functionality. Caused `getElementById('btn-refresh')` to return wrong button, cascading into JS errors.

**Cause:** When inserting the "Evict VRAM" button into the top header, the string replacement matched on `class="btn-action" onclick="refreshAvailability()"` but the original HTML had `id="btn-refresh"` BEFORE `class="btn-action"`:

```html
<!-- Original -->
<button id="btn-refresh" class="btn-action" onclick="refreshAvailability()">Refresh</button>

<!-- After broken insertion -->
<button id="btn-refresh" class="btn-action" onclick="evictAllVRAM()">Evict VRAM</button>
<button id="btn-refresh" class="btn-action" onclick="refreshAvailability()">Refresh</button>
```

Both buttons had `id="btn-refresh"`. The `evictAllVRAM()` JS function used `getElementById('btn-evict')` which didn't exist, so it silently did nothing.

**Fix:** Changed first button ID from `id="btn-refresh"` to `id="btn-evict"`:
```html
<button id="btn-evict" class="btn-action" onclick="evictAllVRAM()">Evict VRAM</button>
<button id="btn-refresh" class="btn-action" onclick="refreshAvailability()">Refresh</button>
```

**Detection:** `grep -oP '<button[^>]*id="[^"]*"'` shows duplicate IDs. Static checker should flag any `id=` used more than once.

---

### Bug 3 — Mismatched Script Tags (3rd `<script>` Block)

**Symptom:** Possible HTML parsing issues. Script open/close count was 2/3 (imbalanced).

**Cause:** The `_build_params()` function for the "manatts" engine included an inline `<script>...</script>` block (lines 497-528). This was a separate script block from the main `_JS` script, embedded inside the HTML body's engine pane section.

The page had:
1. `<script>` inline block (manatts params) 
2. `<script>` main JS block (`_JS` variable)
3. `<script src="...bootstrap..."></script>` (CDN, self-closing-like)

That's 3 `<script` opens. Closes:
1. `</script>` (manatts inline)
2. `</script>` (main JS)
3. `</script>` (bootstrap CDN)

= 3 closes. BUT the rendered HTML showed open=2 because one `<script>` tag was inside a Python string that wasn't being emitted correctly, or the `_JS` raw string had its `<script>` tag counted differently.

The 2/3 imbalance meant one too many `</script>` tags — the browser might have closed the main script block prematurely.

**Fix:** Removed the manatts inline script entirely (32 lines deleted). It was not critical functionality (just populated a dropdown with reference WAV files for the ManaTTS engine, which is marked unavailable anyway).

After removal: open=2, close=2 (balanced).

**Detection:** Count `<script` opens vs `</script>` closes in rendered HTML. Should be exactly equal.

---

## Detection Methods

### Static checks to run after any UI change:

```bash
# 1. Fetch rendered page
curl -s http://localhost:8009/ > /tmp/page.html

# 2. Tag balance
echo "script: open=$(grep -c '<script' /tmp/page.html), close=$(grep -c '</script>' /tmp/page.html)"

# 3. JS syntax via Node.js
python3 -c "
import re
html = open('/tmp/page.html').read()
js = re.findall(r'<script>(.*?)</script>', html, re.DOTALL)
open('tmp.js','w').write(js[0] if js else '')
"
node -c tmp.js  # throws if syntax error

# 4. Duplicate IDs
grep -oP 'id="[^"]*"' /tmp/page.html | sort | uniq -c | sort -rn | head -10

# 5. Brace/paren balance
python3 -c "
import re
html = open('/tmp/page.html').read()
js = re.findall(r'<script>(.*?)</script>', html, re.DOTALL)[0]
print(f'Braces: {{ {js.count(\"{\")}  }} {js.count(\"}\")}')
print(f'Parens: ( {js.count(\"(\")}  ) {js.count(\")\")}')
print(f'Brackets: [ {js.count(\"[\")}  ] {js.count(\"]\")}')
"
```

### The checker script

`scripts/utils/check_ui.py` — SCP to VM, fetch page, runs all checks. Output is PASS/FAIL.

---

## Timeline

| Time | Event |
|------|-------|
| ~14:00 | LLM integration code deployed. UI appears to work initially (TTS synthesis OK via curl). |
| ~15:00 | User reports sidebar not responding, VRAM not showing. |
| ~15:30 | Bug 1 found: broken JS regex. Fixed with `sed`. |
| ~16:00 | Bug 2 found: duplicate `btn-refresh` ID. Fixed in source. |
| ~16:30 | Both fixes deployed. User still reports issue. |
| ~17:00 | Deep investigation: Node.js validation, brace counting, pane/button matching. All pass. |
| ~17:30 | Bug 3 found: imbalanced script tags from manatts inline block. Removed entire block. |
| ~18:00 | All fixes deployed. Page verified: 1 JS block, 0 issues, tag balance 2/2. |

**Root cause summary:** The `addChatMessage()` JS function was inserted via a Python script using regular strings. The `\n` in `replace(/\n/g, '<br>')` became a literal newline in the source file, splitting the regex across lines. JavaScript regex literals cannot span lines — this broke ALL JS execution on the page. Two additional minor bugs (duplicate ID, imbalanced script tags) contributed to fragility.

**Key lesson for future sessions:** When modifying `tts_lab_ui.py`, test the rendered page by fetching it and validating the JavaScript with `node -c`. Never assume the Python string manipulations produce valid JS — always verify the rendered output.
