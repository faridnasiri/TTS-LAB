#!/usr/bin/env python3
"""
ElevenLabs Persian Voice Batch Generator — Rich Terminal UI
============================================================
Discovers male voices from the Voice Library, tests pricing,
shows full cost estimate, then generates Persian TTS organized by category.

Usage:
  python scripts\elevenlabs_persian_batch.py               # discover all, confirm before generate
  python scripts\elevenlabs_persian_batch.py --yes         # auto-generate after discovery
  python scripts\elevenlabs_persian_batch.py --max 10      # limit to 10 voices
  python scripts\elevenlabs_persian_batch.py --search persian  # Persian voices only
  python scripts\elevenlabs_persian_batch.py --discover-only  # just count, don't generate
"""

import os
import sys
import time
import json
from pathlib import Path
from collections import defaultdict, OrderedDict

# ── Force UTF-8 on Windows ──────────────────────────────────────────
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Dependencies ────────────────────────────────────────────────────
try:
    import requests
except ImportError:
    print("Missing: requests. Run: pip install requests")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn,
    )
    from rich.rule import Rule
    from rich.align import Align
    from rich import box
    RICH_OK = True
except ImportError:
    RICH_OK = False
    print("Tip: pip install rich  for a prettier UI\n")

console = Console() if RICH_OK else None

# ── Config ──────────────────────────────────────────────────────────
API_KEY_FILE = Path(r"C:\temp\11.txt")
BASE_URL = "https://api.elevenlabs.io/v1"

PERSIAN_TEXT = (
    "قشنگ‌ترین خاطره‌ی زندگی‌ام، "
    "روزی بود که با ژاله و چنگیز رفتیم شمال"
)

GENDER = "male"
MODEL_ID = "eleven_v3"
VOICE_SETTINGS = {
    "stability": 0.0,
    "similarity_boost": 0.0,
    "style": 1.0,
    "use_speaker_boost": False,
    "speed": 1.0,
}

# Premade voices (fallback)
PREMADE_MALE_VOICES = {
    "Adam":   {"voice_id": "pNInz6obpgDQGcFmaJgB", "style": "Warm, deep, authoritative"},
    "Antoni": {"voice_id": "ErXwobaYiN019PkySvjV", "style": "Smooth, calm, well-rounded"},
    "Arnold": {"voice_id": "VR6AewLTigWG4xSOukaG", "style": "Strong, powerful, mature"},
    "Liam":   {"voice_id": "TX3LPaxmHKxFdv7VOQHJ", "style": "Young, bright, friendly"},
}

# Persian voices found via search
PERSIAN_MALE_VOICES = {
    "Shahram - Natural Documentary Narrator": {"voice_id": "rNb3hdSf0n4ROIbYC8Bl"},
    "Amir - Breathy and Calm":                {"voice_id": "PleK417YVMP2SUWm8Btb"},
    "Adrian - Versatile, Smooth and Adaptive": {"voice_id": "BognUUMX6W1qmZKB2TOw"},
    "IMan - Iranian Male":                    {"voice_id": "3AA408tBxTzz5dPx3TsR"},
}

OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "output" / "elevenlabs_persian"
PAGE_SIZE = 100
REQ_DELAY = 0.25

# PAYG pricing
COST_PER_1K_V3 = 0.10
COST_PER_1K_FLASH = 0.05


# ── Helpers ─────────────────────────────────────────────────────────

def banner():
    if RICH_OK:
        console.print()
        console.print(Panel(
            Align.center(
                "[bold cyan][mic]  ElevenLabs Persian Voice Batch Generator[/bold cyan]\n\n"
                "[dim]Discover | Estimate | Generate | Save by category[/dim]"
            ),
            border_style="cyan", padding=(1, 2),
        ))
        console.print()
    else:
        print("\n" + "=" * 64)
        print("  ElevenLabs Persian Voice Batch Generator")
        print("=" * 64 + "\n")


def read_api_key():
    if not API_KEY_FILE.exists():
        banner()
        print(f"API key file not found: {API_KEY_FILE}")
        return input("Paste your ElevenLabs API key: ").strip()
    raw = API_KEY_FILE.read_text(encoding="utf-8").strip()
    if "\t" in raw:
        raw = raw.split("\t", 1)[1]
    return raw


def count_chars(text: str) -> int:
    return len(text)


def safe_filename(name: str, voice_id: str) -> str:
    safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in name)
    safe = safe.strip().replace(" ", "_")[:50]
    return f"{safe}_{voice_id[:8]}.mp3"


def parse_args():
    args = {
        "yes": False,
        "max": None,
        "search": None,
        "discover_only": False,
    }
    for i, a in enumerate(sys.argv):
        if a in ("--yes", "-y"):
            args["yes"] = True
        elif a == "--max" and i + 1 < len(sys.argv):
            try:
                args["max"] = int(sys.argv[i + 1])
            except ValueError:
                pass
        elif a == "--search" and i + 1 < len(sys.argv):
            args["search"] = sys.argv[i + 1]
        elif a == "--discover-only":
            args["discover_only"] = True
    return args


# ── API Capability Detection ────────────────────────────────────────

def detect_api_capabilities(api_key: str):
    caps = {
        "can_tts": False, "can_list_shared": False,
        "can_list_voices": False, "can_read_user": False,
        "tier": "unknown", "characters_used": 0, "characters_limit": 0,
    }
    headers = {"xi-api-key": api_key}

    # TTS
    r = requests.post(
        f"{BASE_URL}/text-to-speech/pNInz6obpgDQGcFmaJgB",
        headers={**headers, "Content-Type": "application/json"},
        json={"text": "test", "model_id": "eleven_flash_v2_5"}, timeout=15,
    )
    caps["can_tts"] = (r.status_code == 200)

    # Shared voices
    r = requests.get(f"{BASE_URL}/shared-voices?page_size=3", headers=headers, timeout=15)
    caps["can_list_shared"] = (r.status_code == 200)

    # Own voices
    r = requests.get(f"{BASE_URL}/voices", headers=headers, timeout=15)
    caps["can_list_voices"] = (r.status_code == 200)

    # User info
    r = requests.get(f"{BASE_URL}/user", headers=headers, timeout=15)
    caps["can_read_user"] = (r.status_code == 200)
    if caps["can_read_user"]:
        try:
            sub = r.json().get("subscription", {})
            caps["tier"] = sub.get("tier", "unknown")
            caps["characters_used"] = sub.get("character_count", 0)
            caps["characters_limit"] = sub.get("character_limit", 0)
        except Exception:
            pass

    return caps


# ── Voice Discovery ─────────────────────────────────────────────────

def discover_library_voices(api_key: str, search_term: str = None, max_voices: int = None):
    """
    Fetch male voices from the Voice Library.
      - search_term: optional text search (e.g., 'persian')
      - max_voices: stop discovery after this many unique voices
    Returns: (voices OrderedDict, category_counts dict)
    """
    headers = {"xi-api-key": api_key}
    all_voices = OrderedDict()
    category_counts = defaultdict(int)
    page = 0
    total_count = "?"
    sample_printed = False

    while True:
        params = {"gender": GENDER, "page_size": PAGE_SIZE, "page": page}
        if search_term:
            params["search"] = search_term

        r = requests.get(
            f"{BASE_URL}/shared-voices",
            headers=headers, params=params, timeout=30,
        )
        if r.status_code != 200:
            if page == 0:
                console.print(f"  [red]Voice Library HTTP {r.status_code}[/red]") if RICH_OK \
                    else print(f"  Voice Library HTTP {r.status_code}")
            break

        data = r.json()
        voices = data.get("voices", [])

        if page == 0:
            total_count = data.get("total_count", "?")
            label = f"search='{search_term}'" if search_term else "all"
            if RICH_OK:
                console.print(
                    f"  [dim]Voice Library ({label}): [cyan]{total_count}[/cyan] male voices[/dim]"
                )
            else:
                print(f"  Voice Library ({label}): {total_count} male voices")

        if not voices:
            break

        for v in voices:
            vid = v.get("voice_id")
            if not vid:
                continue

            labels = v.get("labels", {}) or {}
            voice_category = v.get("category", "voice_library")

            # Build a human-readable category from available data
            voice_use_cases = []
            if isinstance(labels, dict) and labels.get("use_cases"):
                voice_use_cases = labels["use_cases"]
            elif voice_category:
                voice_use_cases = [voice_category]

            if not voice_use_cases:
                voice_use_cases = ["Voice Library"]

            # Print one sample for debugging
            if not sample_printed and RICH_OK:
                console.print(
                    f"  [dim]Sample: [yellow]{v.get('name','?')[:50]}[/yellow] "
                    f"cat=[cyan]{voice_category}[/cyan] "
                    f"labels={list(labels.keys()) if isinstance(labels, dict) else type(labels).__name__}[/dim]"
                )
                sample_printed = True

            if vid not in all_voices:
                all_voices[vid] = {
                    "voice_id": vid,
                    "name": v.get("name", "Unknown"),
                    "category": voice_category,
                    "accent": labels.get("accent", "N/A") if isinstance(labels, dict) else "N/A",
                    "age": labels.get("age", "N/A") if isinstance(labels, dict) else "N/A",
                    "descriptives": labels.get("descriptives", []) if isinstance(labels, dict) else [],
                    "use_cases_found": voice_use_cases,
                }
                for uc in voice_use_cases:
                    category_counts[uc] += 1

            # Stop if we hit the max
            if max_voices and len(all_voices) >= max_voices:
                break

        page += 1
        if len(voices) < PAGE_SIZE:
            break
        if max_voices and len(all_voices) >= max_voices:
            break
        time.sleep(0.2)

        if page % 10 == 0 and RICH_OK:
            console.print(f"  [dim]...page {page}, {len(all_voices)} voices so far[/dim]")

    return all_voices, dict(category_counts)


def discover_premade_voices(api_key: str, candidates: dict):
    """Test which candidate voices work for TTS."""
    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
    working = OrderedDict()
    for name, info in candidates.items():
        r = requests.post(
            f"{BASE_URL}/text-to-speech/{info['voice_id']}",
            headers=headers,
            json={"text": "test", "model_id": "eleven_flash_v2_5"},
            timeout=15,
        )
        if r.status_code == 200:
            working[info["voice_id"]] = {
                "voice_id": info["voice_id"],
                "name": name,
                "style": info.get("style", ""),
                "category": "premade",
                "accent": info.get("accent", "N/A"),
                "age": info.get("age", "N/A"),
                "descriptives": [],
                "use_cases_found": ["premade"],
            }
        time.sleep(0.15)
    return working


# ── UI Components ───────────────────────────────────────────────────

def show_voice_browser(voices: OrderedDict, limit: int = 50):
    """Display voice table. If > limit, show first N + count summary."""
    count = len(voices)
    display_voices = OrderedDict(list(voices.items())[:limit])

    if RICH_OK:
        title = f"[v]  Male Voices ({count} total)"
        if count > limit:
            title += f" — showing first {limit}"
        table = Table(title=title, box=box.ROUNDED, border_style="cyan",
                      header_style="bold white", show_lines=False)
        table.add_column("#", style="dim", width=4, justify="right")
        table.add_column("Voice Name", style="bold yellow", min_width=25)
        table.add_column("Category", style="cyan", width=16)
        table.add_column("Accent", style="magenta", width=12)

        for i, (vid, info) in enumerate(display_voices.items(), 1):
            table.add_row(
                str(i), info["name"][:55],
                info.get("category", info.get("use_cases_found", [""])[0])[:14],
                info.get("accent", "N/A"),
            )
        console.print(table)
        console.print()
    else:
        print(f"\n  Male Voices ({count} total)")
        print(f"  {'Name':<45s} {'Category':<15s} {'Accent'}")
        print(f"  {'-'*45} {'-'*15} {'-'*12}")
        for i, (vid, info) in enumerate(display_voices.items(), 1):
            print(f"  {info['name'][:43]:<45s} {info.get('category','')[:13]:<15s} {info.get('accent','N/A')}")


def show_distribution(voices: OrderedDict):
    uc_counts = defaultdict(int)
    for info in voices.values():
        for uc in info.get("use_cases_found", []):
            uc_counts[uc] += 1

    if not uc_counts:
        return

    if RICH_OK:
        table = Table(title="[=]  Category Distribution", box=box.SIMPLE,
                      border_style="blue", header_style="bold")
        table.add_column("Category", style="bold cyan")
        table.add_column("Voices", style="yellow", justify="right")
        table.add_column("Bar", style="green", min_width=30)
        max_count = max(uc_counts.values()) if uc_counts else 1
        for uc in sorted(uc_counts, key=uc_counts.get, reverse=True):
            count = uc_counts[uc]
            bar_len = max(1, int(count / max_count * 30))
            table.add_row(uc, str(count), "#" * bar_len)
        console.print(table)
        console.print()
    else:
        for uc in sorted(uc_counts, key=uc_counts.get, reverse=True):
            count = uc_counts[uc]
            print(f"  {uc:<20s} {count:>5d}  {'#' * min(count, 40)}")


# ── Pricing Test ────────────────────────────────────────────────────

def pricing_test(api_key: str, sample_voice_id: str, sample_voice_name: str):
    """Generate ONE audio to verify settings and cost."""
    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
    payload = {"text": PERSIAN_TEXT, "model_id": MODEL_ID, "voice_settings": VOICE_SETTINGS}

    test_dir = OUTPUT_ROOT / "_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_file = test_dir / safe_filename(sample_voice_name, sample_voice_id)

    if RICH_OK:
        console.print(Rule("[bold yellow][TEST]  PRICING TEST", style="yellow"))
        console.print(f"  Voice: [bold]{sample_voice_name}[/bold]  |  "
                      f"Model: [cyan]{MODEL_ID}[/cyan]  |  "
                      f"stability=[yellow]{VOICE_SETTINGS['stability']}[/yellow]  "
                      f"style=[yellow]{VOICE_SETTINGS['style']}[/yellow]")
    else:
        print(f"\n--- PRICING TEST: {sample_voice_name} | {MODEL_ID} ---")

    try:
        resp = requests.post(
            f"{BASE_URL}/text-to-speech/{sample_voice_id}",
            headers=headers, json=payload, timeout=120,
        )
    except requests.RequestException as e:
        msg = f"Connection error: {e}"
        console.print(f"  [red]FAIL: {msg}[/red]") if RICH_OK else print(f"  FAIL: {msg}")
        return False, count_chars(PERSIAN_TEXT), 0.0

    if resp.status_code != 200:
        try:
            err = resp.json()
            msg = err.get("detail", {}).get("message", resp.text[:200])
        except Exception:
            msg = resp.text[:200]
        console.print(f"  [red]FAIL HTTP {resp.status_code}: {msg}[/red]") if RICH_OK \
            else print(f"  FAIL HTTP {resp.status_code}: {msg}")
        return False, count_chars(PERSIAN_TEXT), 0.0

    test_file.write_bytes(resp.content)
    chars_billed = count_chars(PERSIAN_TEXT)
    test_cost = chars_billed / 1000 * COST_PER_1K_V3

    if RICH_OK:
        console.print(Panel(
            f"  Voice:          [yellow]{sample_voice_name}[/yellow]\n"
            f"  Chars billed:   [cyan]{chars_billed}[/cyan]\n"
            f"  Status:         [green]OK 200[/green]\n"
            f"  Audio file:     [dim]{test_file.name} ({len(resp.content)} bytes)[/dim]\n"
            f"  Est. cost:      [yellow]~${test_cost:.4f}[/yellow]",
            title="[bold]Test Result[/bold]", border_style="green", padding=(1, 2),
        ))
        console.print()
    else:
        print(f"  OK: {chars_billed} chars | {len(resp.content)} bytes | ~${test_cost:.4f}\n")

    return True, chars_billed, test_cost


# ── Final Estimate ──────────────────────────────────────────────────

def show_final_estimate(voices: OrderedDict, chars_per_call: int,
                        cost_per_call: float, caps: dict, search_term: str = None):
    total_voices = len(voices)
    total_chars = total_voices * chars_per_call
    total_cost_v3 = total_chars / 1000 * COST_PER_1K_V3
    total_cost_flash = total_chars / 1000 * COST_PER_1K_FLASH

    used = caps.get("characters_used", 0)
    limit = caps.get("characters_limit", 0)
    remaining = max(0, limit - used) if limit else 0
    quota_ok = total_chars <= remaining if limit else True

    if RICH_OK:
        console.print(Rule("[bold yellow][$]  FINAL ESTIMATE", style="yellow"))
        console.print()

        quota_color = "green" if quota_ok else "red"
        quota_line = ""
        if limit:
            quota_line = (
                f"  Plan quota:              [dim]{remaining:,} remaining of {limit:,}[/dim]\n"
                f"  Quota sufficient:        [{quota_color}]{'YES' if quota_ok else f'NO — short by {total_chars - remaining:,} chars'}[/{quota_color}]\n"
            )

        cost_panel = Panel(
            f"  Total voices:            [bold cyan]{total_voices}[/bold cyan]\n"
            f"  Characters per voice:    [bold]{chars_per_call}[/bold]\n"
            f"  Total characters:        [bold yellow]{total_chars:,}[/bold yellow]\n"
            f"  Model:                   [cyan]{MODEL_ID}[/cyan] (max creativity)\n"
            f"{quota_line}"
            f"  ---\n"
            f"  Est. cost (v3):          [yellow]~${total_cost_v3:.2f}[/yellow]\n"
            f"  Est. cost (Flash):       [dim]~${total_cost_flash:.2f}[/dim]\n"
            f"  Output folder:           [dim]{OUTPUT_ROOT}[/dim]",
            title="Cost Breakdown", border_style="green", padding=(1, 2),
        )
        console.print(cost_panel)
        console.print()
    else:
        print(f"\n  FINAL ESTIMATE")
        print(f"  Voices: {total_voices} | Chars/voice: {chars_per_call} | Total chars: {total_chars:,}")
        print(f"  Cost (v3): ~${total_cost_v3:.2f} | Cost (Flash): ~${total_cost_flash:.2f}")
        if limit:
            print(f"  Quota: {remaining:,}/{limit:,} remaining — {'OK' if quota_ok else 'NOT ENOUGH'}")
        print()

    if not quota_ok:
        overage_chars = total_chars - remaining
        overage_cost = overage_chars / 1000 * COST_PER_1K_V3
        if RICH_OK:
            console.print(
                f"  [yellow][!] Overage needed: +{overage_chars:,} chars = ~${overage_cost:.2f} (PAYG)[/yellow]\n"
            )
        else:
            print(f"  [!] Overage needed: +{overage_chars:,} chars = ~${overage_cost:.2f}\n")

    return total_cost_v3


# ── Generation ──────────────────────────────────────────────────────

def generate_all(api_key: str, voices: OrderedDict, chars_per_call: int):
    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
    payload_base = {
        "text": PERSIAN_TEXT, "model_id": MODEL_ID, "voice_settings": VOICE_SETTINGS,
    }

    # Build work items — one per (voice, category) pair
    work_items = []
    for vid, info in voices.items():
        ucs = info.get("use_cases_found", ["uncategorized"])
        for uc in ucs:
            work_items.append((vid, info["name"], uc))

    total_items = len(work_items)

    # Create folders
    for _, _, uc in work_items:
        (OUTPUT_ROOT / uc).mkdir(parents=True, exist_ok=True)
    all_folder = OUTPUT_ROOT / "_all"
    all_folder.mkdir(parents=True, exist_ok=True)

    success = 0
    fail = 0
    cat_success = defaultdict(int)
    cat_fail = defaultdict(int)
    manifest = []

    if RICH_OK:
        console.print(Rule("[bold yellow][GEN]  GENERATION PHASE", style="yellow"))
        console.print(
            f"  Generating [bold]{total_items}[/bold] files "
            f"([cyan]{len(voices)}[/cyan] voices x categories)"
        )
        console.print(
            f"  Model: [cyan]{MODEL_ID}[/cyan]  |  "
            f"stability=[yellow]{VOICE_SETTINGS['stability']}[/yellow]  "
            f"style=[yellow]{VOICE_SETTINGS['style']}[/yellow]"
        )
        console.print()

        progress = Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=35), TaskProgressColumn(),
            TextColumn("[dim]{task.fields[detail]}[/dim]"),
            console=console,
        )
        task = progress.add_task("[cyan]Generating...", total=total_items, detail=f"0/{total_items}")

        with progress:
            for idx, (vid, name, uc) in enumerate(work_items):
                try:
                    resp = requests.post(
                        f"{BASE_URL}/text-to-speech/{vid}",
                        headers=headers, json=dict(payload_base), timeout=120,
                    )
                except requests.RequestException as e:
                    progress.log(f"[red]CONN[/red] {name[:35]} -> {uc}")
                    fail += 1; cat_fail[uc] += 1
                    progress.update(task, advance=1, detail=f"{idx+1}/{total_items}")
                    continue

                if resp.status_code == 200:
                    fname = safe_filename(name, vid)
                    (OUTPUT_ROOT / uc / fname).write_bytes(resp.content)
                    ap = all_folder / fname
                    if not ap.exists():
                        ap.write_bytes(resp.content)
                    success += 1; cat_success[uc] += 1
                    manifest.append({
                        "voice_id": vid, "name": name, "category": uc,
                        "file": str(OUTPUT_ROOT / uc / fname),
                        "chars": chars_per_call, "model": MODEL_ID,
                    })
                else:
                    try:
                        err = resp.json()
                        msg = err.get("detail", {}).get("message", "")[:80]
                    except Exception:
                        msg = ""
                    progress.log(f"[red]HTTP{resp.status_code}[/red] {name[:35]} -> {uc} {msg}")
                    fail += 1; cat_fail[uc] += 1

                progress.update(task, advance=1, detail=f"{idx+1}/{total_items}")
                if idx < total_items - 1:
                    time.sleep(REQ_DELAY)
        console.print()
    else:
        print(f"\n--- GENERATION: {total_items} files ---")
        for idx, (vid, name, uc) in enumerate(work_items):
            try:
                resp = requests.post(
                    f"{BASE_URL}/text-to-speech/{vid}",
                    headers=headers, json=dict(payload_base), timeout=120,
                )
            except requests.RequestException:
                print(f"  [{idx+1}/{total_items}] CONN ERR {name[:35]} -> {uc}")
                fail += 1; cat_fail[uc] += 1
                continue
            if resp.status_code == 200:
                fname = safe_filename(name, vid)
                (OUTPUT_ROOT / uc / fname).write_bytes(resp.content)
                ap = all_folder / fname
                if not ap.exists():
                    ap.write_bytes(resp.content)
                success += 1; cat_success[uc] += 1
                manifest.append({
                    "voice_id": vid, "name": name, "category": uc,
                    "file": str(OUTPUT_ROOT / uc / fname),
                    "chars": chars_per_call, "model": MODEL_ID,
                })
                print(f"  [{idx+1}/{total_items}] OK  {name[:35]} -> {uc}")
            else:
                print(f"  [{idx+1}/{total_items}] ERR {resp.status_code} {name[:35]} -> {uc}")
                fail += 1; cat_fail[uc] += 1
            if idx < total_items - 1:
                time.sleep(REQ_DELAY)
        print()

    return success, fail, cat_success, cat_fail, manifest


# ── Summary ─────────────────────────────────────────────────────────

def show_summary(voices, success, fail, cat_success, cat_fail,
                 manifest, total_est_cost):
    if RICH_OK:
        console.print(Rule("[bold yellow][OK]  DONE", style="yellow"))
        console.print()

        console.print(Panel(
            f"  Voices discovered:  [bold cyan]{len(voices)}[/bold cyan]\n"
            f"  Files generated:    [bold green]{success}[/bold green]\n"
            f"  Failed:             [bold red]{fail}[/bold red]\n"
            f"  Est. total cost:    [bold yellow]~${total_est_cost:.2f}[/bold yellow]\n"
            f"  Model:              [cyan]{MODEL_ID}[/cyan] (max creativity)\n"
            f"  Output:             [dim]{OUTPUT_ROOT}[/dim]",
            title="Run Summary", border_style="green", padding=(1, 2),
        ))
        console.print()

        # Output folders table
        cat_table = Table(title="[+]  Output Folders", box=box.SIMPLE, border_style="blue")
        cat_table.add_column("Category", style="bold cyan")
        cat_table.add_column("OK", justify="right", style="green")
        cat_table.add_column("FAIL", justify="right", style="red")
        cat_table.add_column("Path", style="dim")
        for uc in sorted(set(list(cat_success) + list(cat_fail))):
            s = cat_success.get(uc, 0); f = cat_fail.get(uc, 0)
            cat_table.add_row(uc, str(s), str(f) if f else "-", str(OUTPUT_ROOT / uc))
        console.print(cat_table)
        console.print()

        # Save manifest
        manifest_path = OUTPUT_ROOT / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"[dim]Manifest: {manifest_path}[/dim]")
        console.print()
    else:
        print(f"\n  DONE — Voices: {len(voices)} | OK: {success} | FAIL: {fail}")
        print(f"  Cost: ~${total_est_cost:.2f}  |  Output: {OUTPUT_ROOT}\n")
        for uc in sorted(set(list(cat_success) + list(cat_fail))):
            s = cat_success.get(uc, 0); f = cat_fail.get(uc, 0)
            print(f"  {uc:<20s}  OK {s}  FAIL {f}")


# ── Main ────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    banner()

    # 1. Read key
    api_key = read_api_key()
    if not api_key:
        print("ERROR: No API key. Aborting.")
        sys.exit(1)
    if RICH_OK:
        console.print(f"[dim]Key: {api_key[:12]}...{api_key[-8:]}[/dim]\n")
    else:
        print(f"Key: {api_key[:12]}...{api_key[-8:]}\n")

    # 2. Detect capabilities
    if RICH_OK:
        console.print(Rule("[bold yellow][i]  CHECKING API KEY CAPABILITIES", style="yellow"))
    else:
        print("Checking API key capabilities...\n")
    caps = detect_api_capabilities(api_key)

    if RICH_OK:
        lines = [
            f"  TTS generation:     {'[green]YES[/green]' if caps['can_tts'] else '[red]NO[/red]'}",
            f"  Voice Library:      {'[green]YES[/green]' if caps['can_list_shared'] else '[red]NO[/red]'}",
            f"  Account info:       {'[green]YES[/green]' if caps['can_read_user'] else '[red]NO[/red]'}",
        ]
        console.print(Panel("\n".join(lines), title="API Key Capabilities",
                            border_style="cyan", padding=(1, 2)))
        console.print()
    else:
        print(f"  TTS: {'YES' if caps['can_tts'] else 'NO'}")
        print(f"  Voice Library: {'YES' if caps['can_list_shared'] else 'NO'}")
        print(f"  Account: {'YES' if caps['can_read_user'] else 'NO'}\n")

    if not caps["can_tts"]:
        print("ERROR: This key cannot generate TTS. Check your API key.")
        sys.exit(1)

    # 3. Discover voices
    voices = OrderedDict()
    search_term = args.get("search")
    max_voices = args.get("max")

    if caps["can_list_shared"]:
        if RICH_OK:
            label = f"search='{search_term}'" if search_term else "ALL male"
            console.print(Rule(f"[bold yellow]>>>  VOICE DISCOVERY ({label})", style="yellow"))
        else:
            print(f"\n--- Voice Discovery ({search_term or 'all male'}) ---")

        voices, cat_counts = discover_library_voices(api_key, search_term, max_voices)

        # If search gave 0 results, warn
        if not voices and search_term:
            print(f"\n  WARNING: Search '{search_term}' returned 0 voices. Trying without search...\n")
            voices, cat_counts = discover_library_voices(api_key, None, max_voices)

        # If still empty or API error, fallback
        if not voices:
            print("\n  Voice Library returned 0 results. Falling back to premade.\n")
            voices = discover_premade_voices(api_key, PREMADE_MALE_VOICES)
    else:
        if RICH_OK:
            console.print(Rule("[bold yellow]>>>  VOICE DISCOVERY (premade only)", style="yellow"))
        print("  Voice Library not accessible on this plan.")
        print("  Testing premade male voices...\n")
        voices = discover_premade_voices(api_key, PREMADE_MALE_VOICES)

    if not voices:
        print("ERROR: No working male voices found.")
        sys.exit(1)

    # 4. Show what we found
    if RICH_OK:
        search_info = f" (search: '{search_term}')" if search_term else ""
        console.print(f"[green]{len(voices)} male voices found{search_info}[/green]\n")
    else:
        print(f"\n{len(voices)} male voices found\n")

    show_distribution(voices)
    show_voice_browser(voices)

    # 5. If > 50 voices, warn and ask to cap
    if len(voices) > 50 and not max_voices and not args["yes"]:
        if RICH_OK:
            console.print(
                f"  [yellow][!] {len(voices)} voices is a lot. "
                f"Use [bold]--max N[/bold] to limit, or type 'y' to proceed with all.[/yellow]\n"
            )

    # 6. Pricing test
    first_vid, first_info = next(iter(voices.items()))
    test_ok, chars_billed, test_cost = pricing_test(api_key, first_vid, first_info["name"])
    if not test_ok:
        print("WARNING: Pricing test failed. Using defaults (66 chars).")
        chars_billed = count_chars(PERSIAN_TEXT)
        test_cost = chars_billed / 1000 * COST_PER_1K_V3

    cost_per_voice = test_cost
    chars_per_voice = chars_billed

    # 7. Final estimate
    total_est = show_final_estimate(voices, chars_per_voice, cost_per_voice, caps, search_term)

    # 8. If discover-only, stop here
    if args.get("discover_only"):
        print("--discover-only flag set. Stopping after estimate. Test audio saved in _test/")
        sys.exit(0)

    # 9. Confirm
    is_tty = sys.stdin.isatty()
    auto_yes = args.get("yes")
    if auto_yes:
        print("--yes: auto-proceeding with generation.\n")
    elif not is_tty:
        print("Non-interactive terminal. Use --yes to auto-proceed, or --discover-only to just count.")
        print("Test audio saved in output/elevenlabs_persian/_test/")
        sys.exit(0)
    else:
        answer = input("Proceed with full generation? (y/n): ").strip().lower()
        if answer != "y":
            print("Aborted. Test audio saved in output/elevenlabs_persian/_test/")
            sys.exit(0)

    # 10. Generate
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    success, fail, cat_success, cat_fail, manifest = generate_all(
        api_key, voices, chars_per_voice,
    )

    # 11. Summary
    show_summary(voices, success, fail, cat_success, cat_fail, manifest, total_est)


if __name__ == "__main__":
    main()
