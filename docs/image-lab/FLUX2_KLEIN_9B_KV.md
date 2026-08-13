# FLUX.2 Klein 9B-KV — Image Lab Engine

> Added 2026-08-13. The mid-size Klein variant from Black Forest Labs, integrated
> as a **GGUF-quantised** engine because the official `black-forest-labs/FLUX.2-klein-9b-kv`
> repo is gated (401 for our HF token).

## What it is

FLUX.2 Klein 9B-KV is the 9B-parameter sibling of the Klein 4B already in the lab.
The "KV" variant caches reference-token attention K/V during image editing, so
image-to-image runs pay the reference-image cost once instead of every step.

| | Klein 4B (existing) | Klein 9B-KV (this engine) |
|---|---|---|
| Transformer | 5 double + 20 single blocks, hidden 3072 | **8 double + 24 single blocks, hidden 4096** |
| Text encoder | Qwen3-4B (hidden 2560) | **Qwen3-8B (hidden 4096)** |
| Joint attention dim | 7680 (= 3×2560) | **12288 (= 3×4096)** |
| MLP | ratio 3.0 (SwiGLU doubled) | ratio 3.0 (SwiGLU doubled, bigger dims) |
| Distilled steps | 4 | 4 |
| Source | HF repo (bf16) | GGUF Q4_K_M (5.7 GB) |

## Why GGUF and not the official repo

The official repo requires HF access approval (`gated: auto`) that our token does
not have. Verified 2026-08-13:

- `black-forest-labs/FLUX.2-klein-9b-kv` → 401 on every file, including plain configs
- `black-forest-labs/FLUX.2-klein-9b-kv-fp8` → public but **single-file only**
  (transformer safetensors, no model_index.json, no text encoder / VAE / tokenizer)
- `black-forest-labs/FLUX.2-klein-9b-fp8` → 401

So the transformer comes from the public **QuantStack/FLUX.2-Klein-9B-KV-GGUF**
repo (`Flux-2-Klein-9B-KV-Q4_K_M.gguf`, 5.7 GB), and the shared components
(text encoder, VAE, scheduler, tokenizer) are reused from the accessible
`black-forest-labs/FLUX.2-klein-4B` repo / Qwen3-8B.

## Config derivation (the tricky part)

The QuantStack GGUF carries **no `mmdit_*` config metadata** (only
`general.architecture = 'flux'`), and diffusers' auto-config inference
(`DIFFUSERS_DEFAULT_PIPELINE_PATHS`) would pick the wrong FLUX.2-dev-32B
config from key names. The loader therefore writes a local config
(`GGUF_ROOT/flux2klein9b/transformer_cfg/config.json`) derived from the
**actual GGUF tensor shapes**:

| GGUF tensor (logical shape) | Derived config value |
|---|---|
| `double_blocks` 0..7 | `num_layers = 8` |
| `single_blocks` 0..23 | `num_single_layers = 24` |
| `img_attn.qkv` [12288, 4096] | `num_attention_heads = 32`, `attention_head_dim = 128` (hidden 4096) |
| `img_mlp.0` [24576, 4096] | `mlp_ratio = 3.0` — **not 6.0!** diffusers `Flux2SwiGLU` doubles the FFN dim internally (4096 × 3 × 2 = 24576) |
| `txt_in` [12288, 4096] | `joint_attention_dim = 12288` (= 3 × Qwen3-8B hidden 4096) |
| `time_in` [256, 4096] | `timestep_guidance_channels = 256` |
| no `guidance_in` tensors | `guidance_embeds = false` (distilled) |
| — | `patch_size = 1`, `in_channels = 128`, `rope_theta = 2000`, `axes_dims_rope = [32,32,32,32]` (from Klein 4B) |

### GGUF shape convention note

QuantStack GGUFs store quantised tensors with **byte dims** in the last position
(e.g. `img_mlp.0` stored as `[24576, 2304]` where 2304 = 4096 × 144/256 bytes/row
for Q4_K). diffusers handles this transparently: `GGUFParameter.quant_shape`
recovers the logical shape, `check_quantized_param_shape` verifies it against the
model, and the fused-QKV chunking in `convert_flux2_transformer_checkpoint_to_diffusers`
works on the flat byte stream (dim-0 chunking of a row-major quantised blob).

## Text encoder: Qwen3-8B

The 9B-KV pipeline stacks text-encoder hidden states from layers **(9, 18, 27)**:
`prompt_embeds = concat(hs_9, hs_18, hs_27)` → dim = 3 × hidden. To hit the
transformer's `joint_attention_dim = 12288` the encoder must have hidden 4096 —
that's **Qwen3-8B** (`Qwen/Qwen3-8B`, public, 36 layers, hidden 4096, heads 32).
The gated repo's text encoder is 4 shards ≈ 9.4B params, consistent with Qwen3-8B
+ embedding overhead.

- **GPU_ONLY (default):** NF4 4-bit via bitsandbytes (0.49.2, installed),
  **always on GPU** (`device_map="cuda"`, `bnb_4bit_quant_type="nf4"`) → ~2.5 GB
  VRAM. No HF token needed (public repo). If free VRAM < 14800 MiB at load
  time, the loader first POSTs `/evict` to the TTS engine containers (ports
  8101-8104) to reclaim the shared GPU, then re-checks; if still short it
  raises a clear `RuntimeError`. There is **no CPU fallback** — GPU-only policy.
- **Non-GPU mode (`IMGLAB_GPU_ONLY=0`):** bf16 on CPU + leaf-level group
  offloading (62 GB RAM is fine) — historical path kept for GPU-less machines.

## VRAM budget (16 GB RTX 5060 Ti, measured)

| Component | Size on GPU |
|---|---|
| Transformer Q4_K_M GGUF | ~5.7 GB (dequant fused at forward) |
| Qwen3-8B text encoder (NF4) | ~2.5 GB |
| VAE (AutoencoderKLFlux2) | ~3 GB (slicing + tiling enabled) |
| Latents / activations @1024² | < 1.5 GB |
| **Total, everything on CUDA** | **~10 GiB — measured 9.98 GiB CUDA (2026-08-13)** |

Fits on the 15.5 GiB card with ~4-5 GiB headroom for activations. The TTS
engine containers normally keep ~3.4 GiB resident, so the loader evicts them
when free VRAM drops below the 14800 MiB threshold (max achievable free with
TTS CUDA contexts resident is ~15354 MiB).

## Loading path (what actually runs)

```python
transformer = Flux2Transformer2DModel.from_single_file(
    gguf_path, config=_flux2klein9b_config_dir(),
    quantization_config=_gguf_quant_config(), torch_dtype=torch.bfloat16)
text_encoder = AutoModel.from_pretrained("Qwen/Qwen3-8B",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
    device_map="cuda")                                  # NF4 — ALWAYS on GPU (GPU_ONLY)
pipe = Flux2KleinKVPipeline(transformer=transformer.to("cuda"),
                            text_encoder=text_encoder,
                            tokenizer=Qwen2TokenizerFast("Qwen/Qwen3-8B"),
                            vae=AutoencoderKLFlux2(4B repo), scheduler=…, is_distilled=True)
pipe = pipe.to("cuda")
# token=None for all from_pretrained calls — public repos, service has no HF_TOKEN
# If free VRAM < 14800 MiB before the load: POST /evict to TTS containers (8101-8104),
# re-check, raise RuntimeError if still short. No CPU fallback.
```

Verified end-to-end on the VM (diffusers 0.38.0, gguf 0.19.0):
- `from_single_file` loads the 5.7 GB GGUF in **15 s**
- all **233 converted tensors match** the model state_dict (0 shape mismatches)
- CUDA forward pass (8×8 latent, 64 text tokens) in **1.5 s** — clean output
- **Live generation 2026-08-13:** `POST /generate/flux2klein9b` 1024×1024/4-step
  → HTTP 200 in 39-41 s, **9.98 GiB CUDA total, encoder NF4 on GPU**
  (eviction not needed — free VRAM exceeded the threshold)

## UI / API

- Engine key: `flux2klein9b`, label "FLUX.2 Klein 9B-KV" (tab "Klein 9B-KV")
- `POST /generate/flux2klein9b` — params: prompt, reference_image (optional),
  width/height (64-multiples), num_inference_steps (default 4 — distilled),
  seed. `guidance_scale` is shown in the UI for consistency but **not sent**
  (step-distilled models run without CFG — the KV pipeline has no such arg).
- Reference image → KV-cache path: step 0 extracts ref-token K/V, steps 1+
  reuse them.

## Troubleshooting

- **`LocalTokenNotFoundError: Token is required (token=True)`** (fixed
  2026-08-13): hf_hub 1.16.1 treats `token=True` as "a token is REQUIRED" and
  raises *before any cache check* — even for fully-cached public files. The
  systemd service has no `HF_TOKEN` (root's shell gets it from
  `/etc/environment`, which systemd does not load). Fix: the loader passes
  `token=None` — all components come from public repos. Same fix applied to
  the klein-4B engine, which had the identical latent bug.
- **`CUDA out of memory` at generation despite the budget looking fine**
  (fixed 2026-08-13): the GPU is shared with the TTS engine containers
  (engine-current holds ~3.35 GiB resident). The loader now checks free VRAM
  at load time and, when free < 14800 MiB, POSTs `/evict` to the TTS engine
  containers (8101-8104) before proceeding — the eviction helper logs each
  `/evict` response with the reclaimed `vram_free_mb`. If free VRAM is still
  insufficient after eviction, the loader raises a clear `RuntimeError`
  (GPU-only policy — it never falls back to CPU). klein-4B got the same
  treatment (NF4 encoder, threshold 12500 MiB).
- **Shape mismatch on load** (`check_quantized_param_shape`): the local config
  was regenerated from a different GGUF — delete
  `GGUF_ROOT/flux2klein9b/transformer_cfg/` and retry.
- **OOM at high res**: raise steps/width only on the 4B engine; 9B-KV at 2048²
  needs VAE tiling (enabled) and may still OOM — keep to 1024²/1536².
- **Slow first load**: first run downloads Qwen3-8B (~18 GB bf16, quantised to
  NF4 in RAM) — allow 10+ min.
- **`torchao` import warning** at startup is benign (no torchao in the env).
