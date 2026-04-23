"""
tts_lab_ui.py — CSS, JS, per-engine param widgets, and HTML page builder.
"""
from __future__ import annotations

from tts_lab_shims  import DEVICE_NAME, VRAM_TOTAL_MB, DEVICE
from tts_lab_config import (
    MODEL_ORDER, MODEL_INFO, ARTHUR_PRESETS, BARK_ARTHUR_PRESETS,
    ALL_KOKORO_VOICES, ALL_XTTS_SPEAKERS, XTTS_LANGUAGES, BARK_PRESETS,
    CHATTTS_SPEEDS, OUTETTS_MODELS, OUTETTS_SPEAKERS, PARLER_MODELS,
    ORPHEUS_VOICES, ZONOS_VARIANTS, CSM_SPEAKERS,
    OUTETTS_DEFAULT_GGUF,
)
from tts_lab_dispatch import _available, _import_cache
from tts_lab_utils    import _piper_voices


# ── HTML helpers ──────────────────────────────────────────────────────────────
def _stars(n: int) -> str:
    return "&#9733;" * n + "&#9734;" * (5 - n)

def _sel(param: str, opts, cur=None) -> str:
    o = "".join(
        f'<option value="{v}"{" selected" if v == cur else ""}>{l}</option>'
        for v, l in opts)
    return (f'<select class="form-select form-select-sm bg-dark text-light border-secondary" '
            f'data-param="{param}">{o}</select>')

def _rng(param, lo, hi, step, val, note="") -> str:
    return (f'<input type="range" class="form-range" data-param="{param}" '
            f'min="{lo}" max="{hi}" step="{step}" value="{val}" oninput="rangeUpdate(this)">'
            + (f'<small class="text-muted">{note}</small>' if note else ""))

def _grp(label, ctrl) -> str:
    return f'<div class="param-group"><label>{label}</label>{ctrl}</div>'

def _row(*cols) -> str:
    return f'<div class="param-row">{"".join(cols)}</div>'

def _upload_widget(prompt_file_id, prompt_status_id, prompt_hidden_id,
                   label="Reference audio WAV (5-30s)") -> str:
    return (
        f'<div class="param-group" style="max-width:100%;width:100%"><label>{label}</label>'
        f'<div class="d-flex gap-2 align-items-center">'
        f'<input type="file" id="{prompt_file_id}" '
        f'class="form-control form-control-sm bg-dark text-light border-secondary" '
        f'accept="audio/wav,audio/*" style="max-width:320px">'
        f'<button class="btn btn-sm btn-outline-info" '
        f"onclick=\"uploadPrompt('{prompt_file_id}','{prompt_status_id}','{prompt_hidden_id}')\">Upload</button>"
        f'<span id="{prompt_status_id}" class="text-muted small"></span>'
        f'<input type="hidden" data-param="audio_prompt_id" id="{prompt_hidden_id}">'
        f'</div></div>'
    )


# ── Per-engine parameter widgets ──────────────────────────────────────────────
def _build_params(name: str) -> str:

    if name == "piper":
        voices = _piper_voices() or ["en_US-ryan-high"]
        vopts = "\n".join(
            f'<option value="{v}">{"[GB]" if "GB" in v else "[US]"} {v}'
            f'{"  default" if "ryan-high" in v else ""}</option>'
            for v in voices)
        sel = (f'<select class="form-select form-select-sm bg-dark text-light border-secondary" '
               f'data-param="voice">{vopts}</select>')
        return (
            _row(_grp("Voice (6 downloaded)", sel),
                 _grp('Speed <span class="range-val">1.0</span>', _rng("speed", "0.5", "2.0", "0.05", "1.0")))
            + _row(
                _grp('Length scale <span class="range-val">1.0</span>', _rng("length_scale", "0.5", "2.0", "0.05", "1.0", "higher=slower")),
                _grp('Noise scale <span class="range-val">0.667</span>', _rng("noise_scale", "0.1", "1.5", "0.05", "0.667", "voice variation")),
                _grp('Noise-W <span class="range-val">0.8</span>', _rng("noise_w", "0.1", "1.5", "0.05", "0.8", "duration variation")),
            )
        )

    if name == "kokoro":
        grps = [
            ("British Male (Arthur pick)", [v for v in ALL_KOKORO_VOICES if v.startswith("bm_")]),
            ("British Female",             [v for v in ALL_KOKORO_VOICES if v.startswith("bf_")]),
            ("American Male",              [v for v in ALL_KOKORO_VOICES if v.startswith("am_")]),
            ("American Female",            [v for v in ALL_KOKORO_VOICES if v.startswith("af_")]),
            ("Spanish",                    [v for v in ALL_KOKORO_VOICES if v.startswith(("ef_", "em_"))]),
            ("French",                     [v for v in ALL_KOKORO_VOICES if v.startswith("ff_")]),
            ("Hindi",                      [v for v in ALL_KOKORO_VOICES if v.startswith(("hf_", "hm_"))]),
            ("Italian",                    [v for v in ALL_KOKORO_VOICES if v.startswith(("if_", "im_"))]),
            ("Japanese",                   [v for v in ALL_KOKORO_VOICES if v.startswith(("jf_", "jm_"))]),
            ("Portuguese",                 [v for v in ALL_KOKORO_VOICES if v.startswith(("pf_", "pm_"))]),
            ("Chinese",                    [v for v in ALL_KOKORO_VOICES if v.startswith(("zf_", "zm_"))]),
        ]
        opts = "".join(
            f'<optgroup label="{gl}">{"".join(f"""<option value="{v}"{" selected" if v == "bm_lewis" else ""}>{v}{"  (Arthur pick)" if v == "bm_lewis" else ""}</option>""" for v in vl)}</optgroup>'
            for gl, vl in grps if vl)
        sel = (f'<select class="form-select form-select-sm bg-dark text-light border-secondary" '
               f'data-param="voice">{opts}</select>')
        return _row(_grp("Voice (54, grouped by language)", sel),
                    _grp('Speed <span class="range-val">0.85</span>', _rng("speed", "0.5", "1.5", "0.05", "0.85")))

    if name == "melo":
        sp = [("EN-Default", "EN-Default"), ("EN-US", "EN-US American"),
              ("EN-BR", "EN-BR British"), ("EN-AU", "EN-AU Australian"), ("EN_INDIA", "EN_INDIA Indian")]
        return _row(_grp("Speaker (5 accents)", _sel("speaker", sp, "EN-US")),
                    _grp('Speed <span class="range-val">0.85</span>', _rng("speed", "0.5", "1.5", "0.05", "0.85")))

    if name == "chattts":
        return (
            _row(
                _grp("Prompt speed token", _sel("prompt", CHATTTS_SPEEDS, "[speed_5]")),
                _grp('Temperature <span class="range-val">0.3</span>', _rng("temperature", "0.1", "1.5", "0.05", "0.3")),
                _grp('Top-P <span class="range-val">0.7</span>', _rng("top_p", "0.1", "1.0", "0.01", "0.7")),
                _grp('Top-K <span class="range-val">20</span>', _rng("top_k", "1", "100", "1", "20")),
            )
            + _row(
                _grp('Repetition penalty <span class="range-val">1.05</span>', _rng("repetition_penalty", "1.0", "2.0", "0.01", "1.05")),
                _grp('Max new tokens <span class="range-val">512</span>', _rng("max_new_token", "128", "2048", "64", "512")),
                _grp('Seed <span class="range-val">0</span>', _rng("seed", "0", "9999", "1", "0", "0=random")),
            )
            + _row(_grp("Skip refine text", _sel("skip_refine_text", [("true", "true"), ("false", "false")], "true")))
            + f'<div class="param-row">{_upload_widget("ct-file", "ct-status", "ct-prompt-id", "Reference WAV (optional — derive ChatTTS speaker embedding)")}</div>'
        )

    if name == "outetts":
        vc = ('<textarea class="form-control form-control-sm bg-dark text-light border-secondary" '
              'data-param="voice_characteristics" rows="3" '
              'placeholder="Optional character description, e.g. elderly man, warm, raspy, hesitant"></textarea>')
        transcript = ('<input type="text" class="form-control form-control-sm bg-dark text-light border-secondary" '
                      'data-param="transcript" placeholder="Optional transcript of uploaded reference WAV">')
        return (
            _row(
                _grp("Model", _sel("model_path", OUTETTS_MODELS, OUTETTS_DEFAULT_GGUF)),
                _grp("Default speaker", _sel("speaker", OUTETTS_SPEAKERS, "en-female-1-neutral")),
            )
            + _row(
                _grp('Temperature <span class="range-val">0.4</span>', _rng("temperature", "0.1", "1.5", "0.05", "0.4")),
                _grp('Repetition penalty <span class="range-val">1.1</span>', _rng("repetition_penalty", "1.0", "2.0", "0.01", "1.1")),
                _grp('Top-K <span class="range-val">40</span>', _rng("top_k", "1", "100", "1", "40")),
                _grp('Top-P <span class="range-val">0.9</span>', _rng("top_p", "0.1", "1.0", "0.01", "0.9")),
                _grp('Min-P <span class="range-val">0.05</span>', _rng("min_p", "0.0", "0.5", "0.01", "0.05")),
            )
            + _row(_grp('Max tokens <span class="range-val">0 (auto)</span>',
                        _rng("max_length", "0", "4096", "128", "0", "0=auto from text length (~30 tok/word)")))
            + f'<div class="param-row">{_upload_widget("ot-file", "ot-status", "ot-prompt-id", "Reference WAV (optional — create OuteTTS speaker)")}</div>'
            + '<div class="param-row">' + _grp("Reference transcript", transcript) + '</div>'
            + f'<div class="param-row" style="flex-direction:column"><div class="param-group" style="max-width:100%;width:100%"><label>Voice characteristics</label>{vc}</div></div>'
        )

    if name == "bark":
        preset_opts = list(BARK_PRESETS)
        token_hint = (
            '<div class="alert alert-info py-2 small mt-2 mb-0">'
            '<strong>Emotion tokens you can embed in text:</strong><br>'
            '<code>[laughs]</code> &nbsp; <code>[sighs]</code> &nbsp; <code>[clears throat]</code> &nbsp; '
            '<code>[hesitantly]</code> &nbsp; <code>[gasps]</code> &nbsp; <code>[long pause]</code> &nbsp; '
            '<code>[nervously]</code> &nbsp; <code>[quietly]</code> &nbsp; <code>[MAN]</code> &nbsp; <code>[WOMAN]</code><br>'
            '<em>Example: "Hello? [sighs] Oh my goodness, just a moment dear. [clears throat]"</em>'
            '</div>'
        )
        bark_presets_html = "".join(
            f'<button class="btn btn-sm btn-outline-secondary mb-1" onclick="setPreset(this.dataset.txt)" '
            f'data-txt="{t.replace(chr(34), chr(39))}" style="font-size:.72rem">{l}</button>'
            for t, l in BARK_ARTHUR_PRESETS)
        return (
            _row(_grp("Voice preset", _sel("voice_preset", preset_opts, "v2/en_speaker_6")))
            + f'<div class="mt-2">{bark_presets_html}</div>'
            + token_hint
        )

    if name == "styletts2":
        return (
            _row(
                _grp('Alpha (style weight) <span class="range-val">0.3</span>', _rng("alpha", "0.0", "1.0", "0.05", "0.3", "0=copy ref style exactly")),
                _grp('Beta (prosody weight) <span class="range-val">0.7</span>', _rng("beta", "0.0", "1.0", "0.05", "0.7", "0=copy ref prosody")),
                _grp('Diffusion steps <span class="range-val">5</span>', _rng("diffusion_steps", "3", "15", "1", "5", "more=better+slower")),
                _grp('Embedding scale <span class="range-val">1.0</span>', _rng("embedding_scale", "0.5", "3.0", "0.1", "1.0")),
            )
            + f'<div class="param-row">{_upload_widget("sty-file", "sty-status", "sty-prompt-id", "Style reference WAV (optional — sets voice timbre)")}</div>'
            + '<p class="text-muted small mt-1">Without reference: uses built-in neutral voice. With reference: clones timbre/style.</p>'
        )

    if name == "f5tts":
        ref_box = (
            '<div class="param-group" style="max-width:100%;width:100%">'
            '<label>Reference text <small class="text-muted">(what the speaker says in the reference WAV)</small></label>'
            '<input type="text" class="form-control form-control-sm bg-dark text-light border-secondary" '
            'data-param="ref_text" placeholder="Exact words spoken in the reference WAV clip...">'
            '</div>'
        )
        return (
            f'<div class="param-row">{_upload_widget("f5-file", "f5-status", "f5-prompt-id", "Reference WAV (REQUIRED — 5-15s of the target voice)")}</div>'
            + f'<div class="param-row">{ref_box}</div>'
            + _row(
                _grp('Speed <span class="range-val">1.0</span>', _rng("speed", "0.5", "2.0", "0.1", "1.0")),
                _grp('NFE steps <span class="range-val">32</span>', _rng("nfe_step", "8", "64", "4", "32", "more=better quality+slower")),
            )
            + '<p class="text-muted small mt-1">F5-TTS REQUIRES a reference WAV — upload a 5-15s clip of any voice you want to clone.</p>'
        )

    if name == "dia":
        tags_hint = (
            '<div class="alert alert-info py-2 small mt-2 mb-0">'
            '<strong>Dia text format:</strong><br>'
            '<code>[S1]</code> and <code>[S2]</code> = speaker turns &nbsp;|&nbsp; '
            '<code>[laughs]</code> <code>[sighs]</code> <code>[coughs]</code> <code>[groans]</code> '
            '<code>[gasps]</code> <code>[sobs]</code> <code>[clears throat]</code><br>'
            '<em>Auto-prefixes [S1] if no speaker tag found. Upload a voice WAV to clone a speaker.</em>'
            '</div>'
        )
        return (
            tags_hint
            + _row(
                _grp('CFG scale <span class="range-val">3.0</span>', _rng("cfg_scale", "1.0", "5.0", "0.1", "3.0", "guidance strength")),
                _grp('Temperature <span class="range-val">1.2</span>', _rng("temperature", "0.5", "2.0", "0.1", "1.2")),
                _grp('Top-P <span class="range-val">0.95</span>', _rng("top_p", "0.5", "1.0", "0.01", "0.95")),
            )
            + _row(_grp('Max tokens <span class="range-val">auto</span>', _rng("max_tokens", "128", "1024", "64", "0", "0=auto from text length")))
            + f'<div class="param-row">{_upload_widget("dia-file", "dia-status", "dia-prompt-id", "Voice reference WAV (optional — speaker cloning)")}</div>'
        )

    if name == "xtts":
        sp_opts   = [(s, s + ("  [Arthur pick]" if s == "Torcull Diarmuid" else "")) for s in ALL_XTTS_SPEAKERS]
        lang_opts = [(k, f"{k}  {v}") for k, v in XTTS_LANGUAGES.items()]
        return (
            '<div class="alert alert-warning py-2 small mb-2">3.2 GB RAM — evicts other heavy models.</div>'
            + _row(
                _grp("Speaker (58 total)", _sel("speaker", sp_opts, "Torcull Diarmuid")),
                _grp("Language (17 total)", _sel("language", lang_opts, "en")),
            )
            + _row(_grp('Temperature <span class="range-val">0.3</span>', _rng("temperature", "0.01", "1.0", "0.01", "0.3", "lower=more stable")))
        )

    if name == "cosyvoice":
        sp = [("English Female", "English Female"), ("English Male", "English Male")]
        return (
            '<div class="alert alert-secondary py-2 small mb-2">SFT mode: English Female/Male pre-trained speakers.</div>'
            + _row(_grp("Speaker", _sel("speaker", sp)))
        )

    if name == "parler":
        presets = [
            "An elderly man with a slow, warm, slightly confused voice speaks gently and unhurriedly.",
            "A tired old man with a Southern American accent speaks very slowly, stumbling over words.",
            "An elderly gentleman with a British accent speaks politely and hesitantly.",
            "A friendly 78-year-old man speaks in a clear American accent, with long natural pauses.",
        ]
        preset_btns = "".join(
            f'<button class="btn btn-sm btn-outline-secondary mb-1" '
            f"onclick=\"document.querySelector('#tab-parler [data-param=description]').value=this.dataset.txt\" "
            f'data-txt="{p}" style="font-size:.72rem">{p[:42]}...</button>'
            for p in presets)
        ta = (f'<textarea class="form-control form-control-sm bg-dark text-light border-secondary"'
              f' data-param="description" rows="3">{presets[0]}</textarea>')
        return (
            _row(_grp("Model", _sel("model_id", PARLER_MODELS, "parler-tts/parler-tts-mini-v1")))
            + f'<div class="param-row" style="flex-direction:column">'
              f'<div class="param-group" style="max-width:100%;width:100%"><label>Voice description</label>{ta}</div>'
              f'<div class="d-flex gap-1 flex-wrap mt-1">{preset_btns}</div></div>'
            + _row(
                _grp('Temperature <span class="range-val">1.0</span>', _rng("temperature", "0.1", "2.0", "0.1", "1.0")),
                _grp('Max tokens <span class="range-val">1000</span>', _rng("max_new_tokens", "200", "2000", "50", "1000")),
            )
        )

    if name == "chatterbox":
        return (
            _row(
                _grp('Exaggeration <span class="range-val">0.65</span>', _rng("exaggeration", "0.0", "1.0", "0.05", "0.65", "0=flat, 1=expressive")),
                _grp('CFG weight <span class="range-val">0.5</span>', _rng("cfg_weight", "0.1", "1.0", "0.05", "0.5", "lower=natural")),
                _grp('Seed <span class="range-val">0</span>', _rng("seed", "0", "9999", "1", "0", "0=random")),
            )
            + f'<div class="param-row">{_upload_widget("cb-file", "cb-status", "cb-prompt-id", "Voice cloning reference WAV (optional)")}</div>'
        )

    if name == "fishspeech":
        return (
            f'<div class="param-row">{_upload_widget("fs2-file", "fs2-status", "fs2-prompt-id", "Reference WAV (optional — enables voice cloning)")}</div>'
            + _row(_grp('Speed <span class="range-val">1.0</span>', _rng("speed", "0.5", "2.0", "0.1", "1.0")))
            + '<p class="text-muted small mt-1">Without reference: default voice. Upload a 5-30s WAV to clone any voice.</p>'
        )

    if name == "csm":
        hint = (
            '<div class="alert alert-info py-2 small mt-2 mb-0">'
            'Gated model — run <code>huggingface-cli login</code> before first load. '
            'Speaker 0 is male, 1-2 are alternatives.</div>'
        )
        return (
            _row(
                _grp("Speaker", _sel("speaker_id", CSM_SPEAKERS, "0")),
                _grp('Max audio (ms) <span class="range-val">30000</span>', _rng("max_audio_length_ms", "5000", "60000", "1000", "30000")),
            )
            + hint
        )

    if name == "qwen3tts":
        q3_speakers = [("aiden","Aiden"),("dylan","Dylan"),("eric","Eric"),
                       ("ono_anna","Ono Anna"),("ryan","Ryan"),("serena","Serena"),
                       ("sohee","Sohee"),("uncle_fu","Uncle Fu"),("vivian","Vivian")]
        q3_langs    = [("english","English"),("chinese","Chinese"),("japanese","Japanese"),
                       ("korean","Korean"),("french","French"),("german","German"),
                       ("spanish","Spanish"),("portuguese","Portuguese")]
        return (
            '<div class="alert alert-info py-2 small mb-3">'
            '🎙 <strong>Qwen3-TTS 1.7B CustomVoice</strong> — 9 built-in speakers · '
            'style via <em>Instruct</em> · voice clone via reference WAV</div>'
            + _row(_grp("Speaker", _sel("voice", q3_speakers, "aiden")),
                   _grp("Language", _sel("language", q3_langs, "english")))
            + _row(_grp('Style instruction <span style="font-size:.7rem;color:#aaa">(optional)</span>',
                        '<input type="text" class="form-control form-control-sm" data-param="instruct" '
                        'placeholder="e.g. speak like a confused elderly man, slowly and gently">'))
            + '<div class="mt-3 mb-1" style="font-size:.72rem;font-weight:700;color:#7eb8f7;text-transform:uppercase;letter-spacing:.08em">Main talker sampling</div>'
            + _row(
                _grp('Temperature <span class="range-val">0.9</span>', _rng("temperature", "0.1", "2.0", "0.05", "0.9", "lower=stable, higher=expressive")),
                _grp('Top-p <span class="range-val">1.0</span>', _rng("top_p", "0.1", "1.0", "0.05", "1.0", "nucleus cutoff")),
                _grp('Top-k <span class="range-val">50</span>', _rng("top_k", "1", "200", "1", "50", "vocab cutoff")),
                _grp('Repetition penalty <span class="range-val">1.05</span>', _rng("repetition_penalty", "1.0", "1.5", "0.01", "1.05")),
            )
            + '<div class="mt-3 mb-1" style="font-size:.72rem;font-weight:700;color:#7eb8f7;text-transform:uppercase;letter-spacing:.08em">Sub-talker sampling <span style="font-weight:400;color:#888">(tokenizer-v2 / 1.7B)</span></div>'
            + _row(
                _grp('Sub temperature <span class="range-val">0.9</span>', _rng("subtalker_temperature", "0.1", "2.0", "0.05", "0.9")),
                _grp('Sub top-p <span class="range-val">1.0</span>', _rng("subtalker_top_p", "0.1", "1.0", "0.05", "1.0")),
                _grp('Sub top-k <span class="range-val">50</span>', _rng("subtalker_top_k", "1", "200", "1", "50")),
            )
            + '<div class="mt-3 mb-1" style="font-size:.72rem;font-weight:700;color:#7eb8f7;text-transform:uppercase;letter-spacing:.08em">Generation</div>'
            + _row(_grp('Max tokens <span class="range-val">2048</span>', _rng("max_new_tokens", "256", "4096", "64", "2048", "codec tokens ≈ audio length cap")))
            + '<div class="mt-3 mb-1" style="font-size:.72rem;font-weight:700;color:#7eb8f7;text-transform:uppercase;letter-spacing:.08em">Voice clone <span style="font-weight:400;color:#888">(overrides speaker)</span></div>'
            + f'<div class="param-row">{_upload_widget("q3-file", "q3-status", "q3-prompt-id", "Reference WAV — 5–30s of target voice")}</div>'
            + _row(_grp('Ref transcript <span style="font-size:.7rem;color:#aaa">(required for voice clone)</span>',
                        '<input type="text" class="form-control form-control-sm" data-param="ref_text" '
                        'placeholder="Exact words spoken in the reference audio…">'))
        )

    if name == "orpheus":
        emotion_hint = (
            '<div class="alert alert-info py-2 small mt-2 mb-0">'
            '<strong>Emotion tags (embed in text):</strong><br>'
            '<code>&lt;laugh&gt;</code> &nbsp; <code>&lt;chuckle&gt;</code> &nbsp; <code>&lt;sigh&gt;</code> &nbsp; '
            '<code>&lt;cough&gt;</code> &nbsp; <code>&lt;sniffle&gt;</code> &nbsp; <code>&lt;groan&gt;</code> &nbsp; '
            '<code>&lt;yawn&gt;</code> &nbsp; <code>&lt;gasp&gt;</code><br>'
            '<em>Example: "Oh my goodness &lt;sigh&gt; just a moment dear &lt;cough&gt; I need to find my glasses."</em>'
            '</div>'
        )
        gpu_warn = (
            '<div class="alert alert-danger py-2 small mt-2 mb-0">'
            '⚠ <strong>Orpheus 3B requires a CUDA GPU</strong> — vllm will not run on CPU.</div>'
        )
        return _row(_grp("Voice", _sel("voice", ORPHEUS_VOICES, "tara"))) + emotion_hint + gpu_warn

    if name == "neutts":
        return (
            '<div class="alert alert-warning py-2 small">'
            '⚠ <strong>NeuTTS Air is not yet configured.</strong><br>'
            'Edit <code>_load_neutts()</code> and <code>_synth_neutts()</code> in '
            '<code>tts_lab_engines.py</code>.</div>'
        )

    if name == "indextts":
        return (
            f'<div class="param-row">{_upload_widget("idx-file", "idx-status", "idx-prompt-id", "Reference WAV (REQUIRED — 5-30s of target voice)")}</div>'
            + '<p class="text-muted small mt-1">IndexTTS-2 requires a reference WAV for every synthesis call.</p>'
        )

    if name == "zonos":
        lang_opts = [("en-us","English US"),("en-gb","English GB"),("de","German"),("fr","French"),
                     ("ja","Japanese"),("ko","Korean"),("zh","Chinese"),("es","Spanish")]
        emotion_info = (
            '<div class="alert alert-info py-2 small mt-2 mb-0">'
            '<strong>Emotion vector</strong> — sliders control the 8-dim emotion blend. '
            'Higher <em>neutral</em> + low rest = calm elderly speech.</div>'
        )
        return (
            _row(
                _grp("Variant", _sel("variant", ZONOS_VARIANTS, "transformer")),
                _grp("Language", _sel("language", lang_opts, "en-us")),
                _grp('Speaking rate <span class="range-val">13.0</span>', _rng("speaking_rate", "5.0", "25.0", "0.5", "13.0", "words/sec")),
                _grp('Max tokens <span class="range-val">1024</span>', _rng("max_new_tokens", "256", "2048", "64", "1024")),
            )
            + emotion_info
            + _row(
                _grp('Happiness <span class="range-val">0.3</span>', _rng("happiness", "0.0", "1.0", "0.05", "0.3")),
                _grp('Sadness <span class="range-val">0.05</span>', _rng("sadness", "0.0", "1.0", "0.05", "0.05")),
                _grp('Surprise <span class="range-val">0.1</span>', _rng("surprise", "0.0", "1.0", "0.05", "0.1")),
                _grp('Neutral <span class="range-val">0.2</span>', _rng("neutral", "0.0", "1.0", "0.05", "0.2")),
                _grp('Other <span class="range-val">0.2</span>', _rng("other", "0.0", "1.0", "0.05", "0.2")),
            )
            + f'<div class="param-row">{_upload_widget("zn-file", "zn-status", "zn-prompt-id", "Reference WAV (optional — speaker voice cloning)")}</div>'
        )

    if name == "openvoice":
        sp = [("EN-US", "EN-US American"), ("EN-BR", "EN-BR British"), ("EN-AU", "EN-AU Australian")]
        return (
            '<div class="alert alert-secondary py-2 small mb-2">MeloTTS synthesises; tone-color conversion adapts timbre. '
            'Without reference WAV: uses the selected base speaker directly.</div>'
            + _row(
                _grp("Base speaker", _sel("speaker", sp, "EN-US")),
                _grp('Speed <span class="range-val">0.85</span>', _rng("speed", "0.5", "1.5", "0.05", "0.85")),
                _grp('Tau (blend) <span class="range-val">0.3</span>', _rng("tau", "0.0", "1.0", "0.05", "0.3", "0=original, 1=full clone")),
            )
            + f'<div class="param-row">{_upload_widget("ov-file", "ov-status", "ov-prompt-id", "Reference WAV (optional — voice to clone)")}</div>'
        )

    return ""


# ── Page builder ──────────────────────────────────────────────────────────────
_CSS = """
<style>
:root{--bg:#13131f;--panel:#1a1a2e;--card:#1e2235;--card2:#141428;--border:#2d3050;
      --accent:#7eb8f7;--accent2:#4caf50;--text:#e0e0e0;--muted:#8899aa;}
*{box-sizing:border-box;}
body{background:var(--bg);color:var(--text);font-family:system-ui,sans-serif;margin:0;min-height:100vh;}
.top-header{background:var(--panel);border-bottom:1px solid var(--border);padding:10px 20px;
            display:flex;align-items:center;gap:16px;flex-wrap:wrap;position:sticky;top:0;z-index:100;}
.top-header h1{margin:0;font-size:1.15rem;font-weight:700;color:var(--accent);white-space:nowrap;}
.bars-wrap{display:flex;gap:12px;flex-wrap:wrap;flex:1;min-width:0;}
.bar-item{display:flex;flex-direction:column;gap:2px;min-width:180px;}
.bar-track{background:#2a3050;border-radius:6px;height:12px;width:100%;}
.bar-fill{height:100%;border-radius:6px;transition:width .6s;}
.bar-fill.ram{background:linear-gradient(90deg,#4caf50,#7eb8f7);}
.bar-fill.vram{background:linear-gradient(90deg,#4caf50,#39c0c0);}
.bar-label{font-size:.7rem;color:var(--muted);}
.gpu-badge{padding:3px 10px;border-radius:20px;font-size:.72rem;font-weight:700;white-space:nowrap;}
.gpu-badge.ok{background:#1e3a1e;color:#4caf50;border:1px solid #4caf50;}
.gpu-badge.cpu{background:#3a1e1e;color:#f44336;border:1px solid #f44336;}
.main-wrap{display:flex;height:calc(100vh - 57px);}
.sidebar{width:240px;min-width:200px;background:var(--panel);border-right:1px solid var(--border);
         overflow-y:auto;display:flex;flex-direction:column;flex-shrink:0;}
.sidebar-section{padding:8px 10px 4px;font-size:.68rem;font-weight:700;color:var(--muted);
                 text-transform:uppercase;letter-spacing:.08em;border-bottom:1px solid var(--border);}
.engine-btn{display:flex;align-items:center;gap:8px;padding:9px 12px;cursor:pointer;
            border:none;background:transparent;color:var(--text);width:100%;text-align:left;
            border-bottom:1px solid #1e2235;transition:background .12s;}
.engine-btn:hover{background:#22253a;}
.engine-btn.active{background:#2d3561;border-left:3px solid var(--accent);}
.engine-btn .eng-name{font-size:.82rem;font-weight:600;flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.engine-btn .eng-rtf{font-size:.65rem;color:var(--muted);white-space:nowrap;}
.engine-btn .eng-dot{font-size:.7rem;flex-shrink:0;}
.engine-btn .eng-stars{font-size:.65rem;color:#f4b942;flex-shrink:0;}
.sidebar-search{padding:8px 10px;border-bottom:1px solid var(--border);}
.sidebar-search input{width:100%;background:#141428;color:var(--text);border:1px solid var(--border);
                      border-radius:6px;padding:5px 9px;font-size:.8rem;}
.sidebar-search input::placeholder{color:var(--muted);}
.sidebar-search input:focus{outline:none;border-color:var(--accent);}
.content{flex:1;min-width:0;overflow-y:auto;display:flex;flex-direction:column;}
.text-panel{background:var(--card);border-bottom:1px solid var(--border);padding:14px 18px;}
.text-panel label{font-size:.78rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;display:block;}
.preset-bar{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:8px;}
.preset-btn{background:#242840;color:#aab8d0;border:1px solid var(--border);border-radius:14px;
            font-size:.72rem;padding:3px 11px;cursor:pointer;transition:all .15s;}
.preset-btn:hover{background:#2d3561;color:var(--accent);}
.text-box{width:100%;height:88px;background:#0f0f1c;color:var(--text);border:1px solid var(--border);
          border-radius:8px;padding:10px 12px;font-size:.9rem;resize:vertical;transition:border .15s;}
.text-box:focus{outline:none;border-color:var(--accent);}
.engine-panel{flex:1;padding:18px;}
.engine-header{display:flex;align-items:baseline;gap:10px;margin-bottom:4px;flex-wrap:wrap;}
.engine-title{font-size:1.3rem;font-weight:800;color:var(--accent);}
.rtf-badge{background:#2a3561;color:var(--accent);border:1px solid #4a5580;border-radius:4px;
           padding:2px 8px;font-size:.72rem;font-weight:700;}
.avail-badge{font-size:.72rem;padding:2px 9px;border-radius:12px;font-weight:700;}
.avail-badge.ok{background:#1e3a1e;color:#4caf50;border:1px solid #2a5a2a;}
.avail-badge.missing{background:#3a1e1e;color:#f44336;border:1px solid #5a2a2a;}
.engine-meta{font-size:.75rem;color:var(--muted);margin-bottom:14px;}
.engine-meta span{margin-right:14px;}
.params-area{display:flex;flex-direction:column;gap:10px;margin-bottom:16px;}
.param-row{display:flex;flex-wrap:wrap;gap:12px;align-items:flex-end;}
.param-group{display:flex;flex-direction:column;gap:4px;min-width:150px;}
.param-group label{font-size:.75rem;color:var(--muted);margin:0;}
.range-val{font-weight:700;color:var(--accent);}
.form-range::-webkit-slider-thumb{background:var(--accent);}
.form-control,.form-select{background:#0f0f1c !important;color:var(--text) !important;
  border-color:var(--border) !important;font-size:.82rem;}
.form-control:focus,.form-select:focus{outline:none;border-color:var(--accent) !important;box-shadow:none !important;}
.synth-bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:4px;}
.btn-synth{background:#3d5af1;border:none;color:#fff;font-weight:700;padding:9px 22px;
           border-radius:8px;font-size:.9rem;cursor:pointer;transition:background .15s;}
.btn-synth:hover{background:#2d4ae1;}
.btn-synth:disabled{background:#2a3050;cursor:not-allowed;}
.btn-action{background:#1e2235;border:1px solid var(--border);color:var(--muted);
            font-size:.78rem;padding:7px 13px;border-radius:7px;cursor:pointer;transition:all .15s;}
.btn-action:hover{color:var(--text);border-color:var(--accent);}
.spinner{display:none;width:1.1rem;height:1.1rem;border:2px solid var(--accent);
         border-top-color:transparent;border-radius:50%;animation:spin .7s linear infinite;}
@keyframes spin{to{transform:rotate(360deg)}}
.result-card{background:var(--card2);border:1px solid var(--border);border-radius:10px;padding:14px;display:none;margin-top:10px;}
.metric-row{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px;}
.metric-pill{background:#2a3561;border:1px solid #4a5580;border-radius:16px;padding:3px 12px;font-size:.78rem;color:var(--accent);}
audio{width:100%;border-radius:6px;}
.error-panel{display:none;margin-top:10px;border:1px solid #7f2a2a;border-radius:8px;overflow:hidden;}
.error-panel-header{background:#3a1515;padding:7px 12px;display:flex;justify-content:space-between;align-items:center;gap:8px;cursor:pointer;user-select:none;}
.error-title{color:#ff6b6b;font-weight:700;font-size:.85rem;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.error-actions{display:flex;gap:5px;flex-shrink:0;}
.error-panel-body{background:#1a0a0a;padding:10px 12px;max-height:280px;overflow-y:auto;}
.error-panel-body pre{color:#ff9999;font-size:.75rem;margin:0;white-space:pre-wrap;word-break:break-all;}
.err-toggle,.err-copy-btn{font-size:.72rem;color:#aaa;border:1px solid #555;background:transparent;border-radius:3px;padding:1px 6px;cursor:pointer;}
.err-toggle:hover,.err-copy-btn:hover{color:#fff;}
.alert-info{background:#1a2a3a;border-color:#4a6a8a;color:#8ab8e8;}
.alert-warning{background:#2a2010;border-color:#6a5010;color:#d0a060;}
.alert-secondary{background:#1e2235;border-color:#3a4060;color:#aabbcc;}
.alert-danger{background:#2a1010;border-color:#6a2020;color:#e08080;}
code{background:#2a3050;padding:1px 5px;border-radius:4px;font-size:.82em;}
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:#2a3050;border-radius:3px;}
::-webkit-scrollbar-thumb:hover{background:var(--accent);}
@media(max-width:680px){
  .sidebar{width:100%;height:auto;flex-direction:row;overflow-x:auto;overflow-y:hidden;
           border-right:none;border-bottom:1px solid var(--border);}
  .main-wrap{flex-direction:column;height:auto;}
  .engine-btn{border-bottom:none;border-right:1px solid #1e2235;white-space:nowrap;width:auto;}
  .sidebar-section,.sidebar-search{display:none;}
}
</style>"""

_JS = r"""
<script>
const API = '';

document.addEventListener('input', e => {
  if (e.target.type === 'range') {
    const span = e.target.closest('.param-group')?.querySelector('.range-val');
    if (span) span.textContent = parseFloat(e.target.value).toFixed(
      e.target.step < 0.1 ? 3 : e.target.step < 1 ? 2 : 0);
  }
});

function getParams(modelId) {
  const pane = document.getElementById('pane-' + modelId);
  const params = {};
  if (!pane) return params;
  pane.querySelectorAll('[data-param]').forEach(el => {
    if (el.id && el.id.endsWith('-prompt-id') && !el.value) return;
    params[el.dataset.param] = el.value;
  });
  return params;
}

function setPreset(text) { document.getElementById('text-input').value = text; }

let activeEngine = null;
function selectEngine(name) {
  document.querySelectorAll('.engine-btn').forEach(b => b.classList.toggle('active', b.dataset.engine === name));
  document.querySelectorAll('.engine-pane').forEach(p => p.style.display = p.id === 'pane-' + name ? 'block' : 'none');
  activeEngine = name;
}

document.addEventListener('DOMContentLoaded', () => {
  const search = document.getElementById('sidebar-search');
  if (search) search.addEventListener('input', () => {
    const q = search.value.toLowerCase();
    document.querySelectorAll('.engine-btn').forEach(b => {
      b.style.display = b.dataset.label.toLowerCase().includes(q) ? '' : 'none';
    });
  });
});

async function synth(model) {
  const text = document.getElementById('text-input').value.trim();
  if (!text) { alert('Enter some text first'); return; }
  const btn  = document.getElementById('btn-' + model);
  const spin = document.getElementById('spin-' + model);
  const card = document.getElementById('result-' + model);
  if (btn)  btn.disabled = true;
  if (spin) spin.style.display = 'inline-block';
  clearError(model);
  try {
    const res = await fetch(`${API}/synthesize/${model}`, {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({text, params: getParams(model)})
    });
    const data = await res.json();
    if (data.error) { showError(model, data.error + (data.trace ? '\n\n' + data.trace : '')); return; }
    const blob = new Blob([Uint8Array.from(atob(data.audio_b64), c => c.charCodeAt(0))], {type:'audio/wav'});
    const url  = URL.createObjectURL(blob);
    if (card) {
      card.style.display = 'block';
      clearError(model);
      card.querySelector('.audio-player').src = url;
      card.querySelector('.audio-player').load();
      card.querySelector('.m-synth').textContent = data.synth_time_ms + ' ms';
      card.querySelector('.m-dur').textContent   = data.audio_dur_ms  + ' ms';
      card.querySelector('.m-rtf').textContent   = data.rtf + '×';
      card.querySelector('.m-load').textContent  = data.load_time_s   + ' s';
      card.querySelector('.m-sr').textContent    = data.sample_rate   + ' Hz';
      const rtf = parseFloat(data.rtf);
      card.querySelector('.m-rtf').style.color = rtf <= 1 ? '#4caf50' : rtf <= 5 ? '#ff9800' : '#f44336';
    }
  } catch(e) { showError(model, e.toString()); }
  finally {
    if (btn)  btn.disabled = false;
    if (spin) spin.style.display = 'none';
  }
}

function showError(model, msg) {
  const card = document.getElementById('result-' + model);
  if (card) card.style.display = 'block';
  const lines  = msg.split('\n');
  const panel  = document.getElementById('errpanel-'  + model);
  const title  = document.getElementById('errtitle-'  + model);
  const body   = document.getElementById('errmsg-'    + model);
  const toggle = document.getElementById('errtoggle-' + model);
  const bodyEl = document.getElementById('errbody-'   + model);
  if (!panel) { alert(msg); return; }
  panel.style.display = 'block';
  title.textContent   = '❌ ' + (lines[0].trim() || msg.substring(0, 120));
  body.textContent    = msg;
  bodyEl.style.display = lines.length > 3 ? 'block' : 'none';
  toggle.textContent   = lines.length > 3 ? '▲ hide' : '▼ show';
  const audio = card?.querySelector('.audio-player');
  if (audio) audio.src = '';
}
function clearError(model) {
  const panel = document.getElementById('errpanel-' + model);
  if (panel) panel.style.display = 'none';
}
function toggleErrBody(model) {
  const body   = document.getElementById('errbody-'   + model);
  const toggle = document.getElementById('errtoggle-' + model);
  const hidden = body.style.display === 'none';
  body.style.display = hidden ? 'block' : 'none';
  toggle.textContent = hidden ? '▲ hide' : '▼ show';
}
function copyErr(model, ev) {
  ev.stopPropagation();
  const msg = document.getElementById('errmsg-' + model);
  if (msg) navigator.clipboard.writeText(msg.textContent).then(() => {
    const btn = ev.target; btn.textContent = '✅';
    setTimeout(() => btn.textContent = '📋 Copy', 1500);
  });
}

async function preload(model) {
  const spin = document.getElementById('spin-' + model);
  const btn  = document.getElementById('btn-'  + model);
  if (spin) spin.style.display = 'inline-block';
  if (btn)  btn.disabled = true;
  try {
    const res = await fetch(`${API}/models/${model}/load`, {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({params: getParams(model)})
    });
    const d = await res.json();
    if (d.error) { showError(model, d.error); }
    else {
      const card = document.getElementById('result-' + model);
      if (card) {
        card.style.display = 'block';
        clearError(model);
        card.querySelector('.m-load').textContent = d.load_time_s + ' s';
        const panel = document.getElementById('errpanel-' + model);
        if (panel) {
          panel.style.display = 'block';
          document.getElementById('errtitle-' + model).textContent =
            d.status === 'already_loaded' ? '✅ Already loaded' : `✅ Loaded in ${d.load_time_s}s`;
          panel.style.borderColor = '#2a7f2a';
          panel.querySelector('.error-panel-header').style.background = '#153015';
        }
      }
    }
  } catch(e) { showError(model, e.toString()); }
  finally {
    if (spin) spin.style.display = 'none';
    if (btn)  btn.disabled = false;
    refreshStatus();
  }
}
async function unload(model) {
  await fetch(`${API}/models/${model}`, {method:'DELETE'});
  refreshStatus();
}

async function uploadPrompt(fileId, statusId, hiddenId) {
  const input  = document.getElementById(fileId);
  const status = document.getElementById(statusId);
  if (!input.files[0]) { status.textContent = 'No file selected'; return; }
  status.textContent = 'Uploading...';
  const fd = new FormData(); fd.append('file', input.files[0]);
  try {
    const r = await fetch(`${API}/upload`, {method:'POST', body:fd});
    const d = await r.json();
    document.getElementById(hiddenId).value = d.id;
    status.textContent = `✅ ${input.files[0].name} (${(d.size/1024).toFixed(0)} KB)`;
  } catch(e) { status.textContent = '❌ ' + e; }
}

async function refreshStatus() {
  const d = await (await fetch(`${API}/status`)).json();
  const {total, used} = d.system;
  const pct = (used/total*100).toFixed(1);
  document.getElementById('ram-bar').style.width = pct + '%';
  document.getElementById('ram-text').textContent = `RAM ${used}/${total} MB  (${pct}%)`;
  if (d.gpu && d.gpu.vram_total) {
    const gUsed = d.gpu.vram_used || 0, gTot = d.gpu.vram_total;
    const gPct  = (gUsed/gTot*100).toFixed(1);
    document.getElementById('vram-bar').style.width = gPct + '%';
    document.getElementById('vram-text').textContent = `VRAM ${gUsed}/${gTot} MB  (${gPct}%)`;
  }
  Object.entries(d.models).forEach(([n, m]) => {
    const dot = document.getElementById('dot-' + n);
    if (!dot) return;
    dot.textContent = m.status==='loaded' ? '🟢' : m.status==='loading' ? '🟡' : m.status==='error' ? '🔴' : '⚫';
  });
}

async function refreshAvailability() {
  const btn = document.getElementById('btn-refresh');
  if (btn) btn.disabled = true;
  try { await fetch(`${API}/refresh`, {method:'POST'}); await refreshStatus(); }
  finally { if (btn) btn.disabled = false; }
}

setInterval(refreshStatus, 6000);
window.addEventListener('load', () => { refreshStatus(); });
</script>"""


def _result_card(n: str) -> str:
    return (
        f'<div class="result-card" id="result-{n}">'
        f'<div class="metric-row">'
        f'<span class="metric-pill">⏱ <b class="m-synth">—</b></span>'
        f'<span class="metric-pill">🔊 <b class="m-dur">—</b></span>'
        f'<span class="metric-pill">RTF <b class="m-rtf">—</b></span>'
        f'<span class="metric-pill">⬇ <b class="m-load">—</b></span>'
        f'<span class="metric-pill">🎚 <b class="m-sr">—</b></span>'
        f'</div>'
        f'<audio class="audio-player" controls preload="none"></audio>'
        f'<div class="error-panel" id="errpanel-{n}">'
        f'  <div class="error-panel-header" onclick="toggleErrBody(\'{n}\')">'
        f'    <span class="error-title" id="errtitle-{n}"></span>'
        f'    <div class="error-actions">'
        f'      <button class="err-copy-btn" onclick="copyErr(\'{n}\', event)">📋 Copy</button>'
        f'      <button class="err-toggle" id="errtoggle-{n}">▼ show</button>'
        f'    </div>'
        f'  </div>'
        f'  <div class="error-panel-body" id="errbody-{n}" style="display:none">'
        f'    <pre class="error-msg" id="errmsg-{n}"></pre>'
        f'  </div>'
        f'</div>'
        f'</div>'
    )


def build_page() -> str:
    """Render the complete single-page web UI."""
    sidebar_items: list[str] = []
    pane_items:    list[str] = []
    first = True

    for n in MODEL_ORDER:
        info = MODEL_INFO[n]
        ok, reason = _available(n)
        stars    = _stars(info["arthur_fit"])
        dot_cls  = "🟢" if ok else "⚫"
        avail_badge = (
            '<span class="avail-badge ok">✓ available</span>' if ok else
            '<span class="avail-badge missing">✗ missing</span>'
        )
        sidebar_items.append(
            f'<button class="engine-btn{"  active" if first else ""}" '
            f'data-engine="{n}" data-label="{info["label"]}" '
            f"onclick=\"selectEngine('{n}')\">"
            f'<span class="eng-dot" id="dot-{n}">{dot_cls}</span>'
            f'<span class="eng-name">{info["label"]}</span>'
            f'<span class="eng-rtf">{info["rtf_est"]}</span>'
            f'<span class="eng-stars">{stars}</span>'
            f'</button>'
        )
        pane_items.append(
            f'<div class="engine-pane" id="pane-{n}" style="display:{"block" if first else "none"}">'
            f'<div class="engine-header">'
            f'  <span class="engine-title">{info["label"]}</span>'
            f'  <span class="rtf-badge">{info["rtf_est"]}</span>'
            f'  {avail_badge}'
            f'</div>'
            f'<div class="engine-meta">'
            f'  <span>💾 {info["size"]}</span>'
            f'  <span>🧠 ~{info["ram_est_mb"]} MB</span>'
            f'  <span>🎭 Arthur fit: {stars}</span>'
            f'</div>'
            f'<div class="params-area">{_build_params(n)}</div>'
            + (f'<p class="text-warning small mt-1">⚠ {reason}</p>' if not ok else "")
            + f'<div class="synth-bar">'
            + f"  <button id=\"btn-{n}\" class=\"btn-synth\" onclick=\"synth('{n}')\">▶ Synthesise</button>"
            + f"  <button class=\"btn-action\" onclick=\"preload('{n}')\" title=\"Load model into VRAM\">⬇ Preload</button>"
            + f"  <button class=\"btn-action\" onclick=\"unload('{n}')\">⏏ Unload</button>"
            + f'  <span id="spin-{n}" class="spinner"></span>'
            + f'</div>'
            + _result_card(n)
            + f'</div>'
        )
        first = False

    gpu_badge = (
        f'<span class="gpu-badge ok">🟢 {DEVICE_NAME} · {VRAM_TOTAL_MB} MB VRAM</span>'
        if DEVICE == "cuda" else
        '<span class="gpu-badge cpu">🔴 CPU only</span>'
    )
    presets_html = " ".join(
        f'<button class="preset-btn" onclick="setPreset(this.dataset.txt)" data-txt="{t}">{l}</button>'
        for l, t in ARTHUR_PRESETS
    )

    return f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>🎙 Arthur TTS Lab</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
{_CSS}</head><body>

<div class="top-header">
  <h1>🎙 Arthur TTS Lab <span style="font-size:.75rem;font-weight:400;color:var(--muted);">{len(MODEL_ORDER)} engines</span></h1>
  <div class="bars-wrap">
    <div class="bar-item">
      <div class="bar-track"><div class="bar-fill ram" id="ram-bar" style="width:0%"></div></div>
      <span class="bar-label" id="ram-text">Loading RAM…</span>
    </div>
    <div class="bar-item">
      <div class="bar-track"><div class="bar-fill vram" id="vram-bar" style="width:0%"></div></div>
      <span class="bar-label" id="vram-text">Loading VRAM…</span>
    </div>
  </div>
  {gpu_badge}
  <button id="btn-refresh" class="btn-action" onclick="refreshAvailability()" style="white-space:nowrap">🔄 Refresh</button>
</div>

<div class="main-wrap">
  <div class="sidebar">
    <div class="sidebar-section">TTS Engines</div>
    <div class="sidebar-search"><input id="sidebar-search" placeholder="🔍  Filter engines…"></div>
    {"".join(sidebar_items)}
  </div>
  <div class="content">
    <div class="text-panel">
      <label>Arthur's text <span style="font-weight:400;text-transform:none;letter-spacing:0;color:var(--muted)">(shared across all engines)</span></label>
      <div class="preset-bar">{presets_html}</div>
      <textarea id="text-input" class="text-box">{ARTHUR_PRESETS[0][1]}</textarea>
    </div>
    <div class="engine-panel">
      {"".join(pane_items)}
    </div>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
{_JS}</body></html>"""
