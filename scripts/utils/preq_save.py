#!/opt/arthur-img-env/bin/python
"""
preq_save.py — SUPERSEDED 2026-09-07 (kept as reference recipe, no-ops).

sd35 + wan were the only engines that loaded pre-saved "shared" pipeline
components from /opt/arthur-img-models/quantized/<model>/shared (the dirs
preq_save.py built with rsync + bnb-quantized transformers). Both engines
were removed — see docs/sessions/SESSION_2026-09-07_IMGLAB_T2I_SWAP.md — and
no remaining engine reads the quantized/ tree, so this script has nothing to
do. The full rsync/copy-shared/quantize implementation is preserved in git
history if a future engine ever needs it again.

Run (as arthur, service stopped):
  sudo systemctl stop arthur-imglab
  /opt/arthur-img-env/bin/python /opt/arthur-img/preq_save.py
"""

import sys
import time


def main() -> None:
    t0 = time.time()
    print("[preq_save.py] SUPERSEDED 2026-09-07 — sd35/wan removed; nothing to save.")
    print(f"[preq_save.py] exited in {time.time() - t0:.1f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
