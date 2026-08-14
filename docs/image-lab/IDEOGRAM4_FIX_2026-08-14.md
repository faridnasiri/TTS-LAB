# Ideogram 4 — Blank Images from Short Prompts (Caption Starvation) + Seed Randomization (Aug 14, 2026)

## Incident

User reported "generated images are blank — safety filter warning". Ideogram 4
generations from short plain-text prompts came back uniform gray at 1024×1024,
while the service logged a "caption verifier" warning that looked like a safety
filter.

## Root Cause — Caption Starvation

Ideogram 4 is a **caption-driven flow-matching** model: the prompt reaches the
DiT *only* through attention to text tokens (LLM features are zeroed at image
slots). A short plain-text prompt at high resolution produces so few text
tokens relative to the image grid that the positive/negative CFG branches
collapse to near-equal velocities and the latent decays to uniform gray.

**Image-token math:** `patch = patch_size × ae_scale_factor = 2 × 8 = 16`;
`num_image_tokens = (height÷16) × (width÷16)` → 1024² = **4096** tokens,
768² = 2304, 1536×864 = 5184.

**Measured boundary** (tokcount2.py on VM): blank at text-ratio 0.24–0.41%,
real at 0.73–2.85% → trigger threshold chosen: **ratio < 1%**.

**"Safety filter warning" identity:** there is no safety filter in the
pipeline. The warning is the caption verifier's JSON warning
(`_verify_prompts`, `pipeline_ideogram4.py` ~line 552), e.g.
`elements[0]: key order is ('type', 'desc'), expected ('type', 'text', 'desc')`.
It is non-blocking (`raise_on_issues=False`).

## Key Discovery — the pipe's text encoder is headless

`pipe.text_encoder` is a **base `Qwen3VLModel`** (no `lm_head`, no
`generate()`). Verified by introspection + safetensors key scan: 1853 keys,
zero `head`/lm_head keys. `config.architectures=['Qwen3VLModel']`,
`vocab_size=151936`, `tie_word_embeddings=False`. So on-device Qwen3-VL
caption generation is **impossible**.

Model alternatives evaluated:
- **Cached `Qwen/Qwen3-8B`** (`/opt/arthur-img-models/huggingface/models--Qwen--Qwen3-8B/`) — full
  `Qwen3ForCausalLM`, but the **base** checkpoint (repo name canonical;
  no `generation_config.json`) — cannot follow the v1.txt instruction recipe.
- **`Qwen3-8B-Instruct` / `-bnb-4bit` / unsloth mirror** — all **gated** on HF;
  the lab token (`farid-nasiri`) has *no* access (404 with valid token —
  terms never accepted; the token is otherwise valid, `whoami` works, and it
  *does* access `ideogram-ai/ideogram-4-nf4`).

**Decision (user preference: no local LLM for ideogram):** use Ideogram's own
hosted magic-prompt API instead. Local-model code was fully removed.

## Fix 1 — Auto-expansion via Ideogram hosted API (default ON)

`ideogram4_lab_engine.py`:

- `_count_text_tokens(pipe, prompt)` — counts prompt tokens exactly like the
  pipeline's `_tokenize` (chat template, `add_special_tokens=False`).
- `_needs_caption_expansion(pipe, prompt, w, h)` — `True` when the prompt is
  non-empty, not already a JSON caption, and `ratio < 0.01`. Any heuristic
  failure returns `False` — expansion must never block generation.
- `generate_ideogram4` — when the heuristic fires, calls the existing
  `_expand_via_ideogram(prompt, aspect_ratio)` → `POST
  https://api.ideogram.ai/v1/ideogram-v4/magic-prompt` with
  `{"text_prompt": ..., "aspect_ratio": "WxH"}` (no space in WxH), header
  `Api-Key: $IDEOGRAM_API_KEY`, reads `data.json_prompt` (a dict), minifies.
  **Free** endpoint; key already in `/opt/arthur-img/.env` (verified live).

**API-safety (explicit user constraint):** JSON captions and long prompts
pass through **byte-identical** — `_needs_caption_expansion` returns False for
them, so existing API callers are untouched. Response shape unchanged (still
`{"results": [...]}`). If expansion fails (network/API/key), it logs a warning
and uses the prompt as-is — **a request can never fail because of expansion**.

## Fix 3 — Seed randomization (byte-identical images)

The pipeline sets `generator = torch.Generator(device=...)`; `seed=None` →
default seed **67280421310721** → repeated requests produced byte-identical
images. Now:

- `generate_ideogram4`: `gen_seed = seed if seed >= 0 else random.randrange(2**31 - 1)`
  (random drawn **server-side per request**).
- Returns `(images, caption, seed_used)`; `image_lab_engines._generate_ideogram4`
  records the **actual** seed in `final_params` → visible in API response +
  gallery for reproducibility.

## Verification (VM, 2026-08-14)

Engine-level (`test_fix1.py` / `test_fix2.py` on VM venv):

| Test | Result |
|---|---|
| Heuristic: short prompt @1024 and @768 → expand; JSON → no expand | ✅ |
| Short prompt 1024² → real image + cloud-expanded JSON caption (1166 chars) | ✅ |
| Production June-9 JSON caption 1024² → byte-identical + real image | ✅ |
| seed=-1 → randomized (≠ default), seed=42 → recorded as-is | ✅ |

Live API (service restarted, `POST /generate/ideogram4` + `quant=nf4`):

| Test | Result |
|---|---|
| Short prompt → real image, caption expanded (1103 chars), seed=1644938916 | ✅ |
| Real JSON caption → byte-identical + real image | ✅ |
| Empty prompt → HTTP 422 (unchanged error behavior) | ✅ |

Note: the API **requires `quant=nf4`/`fp8`** — empty `quant` → 400
"Unknown quantization" (pre-existing validation, UI always sends it).

## Deployment State

- `ideogram4_lab_engine.py` md5 `26b0e41e…` (repo ↔ VM identical)
- `image_lab_engines.py` md5 `6c98ffcb…` (repo ↔ VM identical)
- `arthur-imglab.service` restarted 2026-08-14 ~19:43, all engines available
- Commit: `ea24d9c`

## Known Tradeoffs / Gotchas

- **Cloud dependency by choice:** if the Ideogram API is down/rate-limited or
  the key dies, short prompts go blank again (no local fallback). The
  endpoint is currently free.
- **Test-harness env gotcha:** SSH sessions inherit empty `HF_HOME`/`HF_TOKEN`;
  `os.environ.setdefault()` does **not** override them, so engine tests run
  without the right cache/token and hit HF with a stale legacy token (401
  "gated"). Always run VM tests via
  `set -a; . /opt/arthur-img/.env; set +a; export HF_HOME=/opt/arthur-img-models/huggingface`
  (see memory `systemd-hf-token-gotcha`).
- **Not approved (do not implement):** fix #2 (cap resolution for short
  prompts) and fix #4 (idle-eviction `last_used` bug).
