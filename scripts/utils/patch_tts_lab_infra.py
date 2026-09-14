#!/usr/bin/env python3
"""One-shot patcher: mount the shared /infra router into the deployed tts_lab.py.

Surgical on purpose. The local working tree also carries an unrelated
uncommitted change to tts_lab.py (an "xttsfa" entry in the /voices map) which
must NOT be deployed, so the whole file is not copied over -- only these two
edits are applied.

Idempotent: re-running reports ALREADY for each edit and changes nothing.
Exits non-zero (via sys.exit) if an anchor is missing, so a silent no-op cannot
be mistaken for success.
"""
from __future__ import annotations

import pathlib
import sys

TARGET = pathlib.Path("/opt/arthur-tts-lab/tts_lab.py")

# (marker that proves it is applied, anchor in the unpatched file, replacement)
EDITS = [
    (
        "from lab_infra import router as infra_router",
        "from tts_lab_history import (\n"
        "    save_generation, list_history, get_history_path,\n"
        "    delete_history_entry, history_stats,\n"
        ")",
        "from tts_lab_history import (\n"
        "    save_generation, list_history, get_history_path,\n"
        "    delete_history_entry, history_stats,\n"
        ")\n"
        "# Shared container/infrastructure dashboard (GET /infra) -- same router is\n"
        "# mounted in image_lab.py so both labs serve one identical page.\n"
        "from lab_infra import router as infra_router",
    ),
    (
        "app.include_router(infra_router)",
        'app = FastAPI(title="Arthur TTS Lab")',
        'app = FastAPI(title="Arthur TTS Lab")\napp.include_router(infra_router)',
    ),
]


def main() -> int:
    if not TARGET.exists():
        print(f"FAILED: {TARGET} does not exist")
        return 1
    text = original = TARGET.read_text(encoding="utf-8")
    failed = []
    for i, (marker, old, new) in enumerate(EDITS, 1):
        if marker in text:
            print(f"  edit {i}: ALREADY present")
        elif old in text:
            text = text.replace(old, new, 1)
            print(f"  edit {i}: APPLIED")
        else:
            print(f"  edit {i}: MISSING anchor")
            failed.append(i)
    if failed:
        print(f"FAILED: edits {failed} could not be located — file NOT written")
        return 1
    if text != original:
        backup = TARGET.with_name(TARGET.name + ".bak-infra")
        backup.write_text(original, encoding="utf-8")
        TARGET.write_text(text, encoding="utf-8")
        print(f"  written: {TARGET}  (backup: {backup.name})")
    else:
        print("  no change needed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
