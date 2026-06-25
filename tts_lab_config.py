"""
tts_lab_config.py — static catalogues, model registry, shared runtime state.

Imports: tts_lab_shims (for _N_CORES, DEVICE, DEVICE_NAME, VRAM_TOTAL_MB)
Exports: all catalogues, MODEL_INFO, MODEL_ORDER, HEAVY, _state, paths
"""
from __future__ import annotations
import threading
from pathlib import Path
# In orchestrator mode, torch isn't available. tts_lab_shims won't import.
# Provide safe defaults — the orchestrator doesn't need GPU info anyway.
try:
    from tts_lab_shims import _N_CORES, DEVICE, DEVICE_NAME, VRAM_TOTAL_MB
except (ImportError, ModuleNotFoundError):
    import os as _os_config
    _N_CORES = _os_config.cpu_count() or 6
    DEVICE = "remote"
    DEVICE_NAME = "orchestrator"
    VRAM_TOTAL_MB = 0

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR    = Path(__file__).parent / "models"
COSYVOICE_DIR = Path("/opt/CosyVoice")
UPLOAD_DIR    = Path("/tmp/tts_uploads")
INDEXTTS_DIR  = Path("/opt/models/indextts")
OPENVOICE_MODELS_DIR = Path("/opt/models/openvoice_v2")
MODELS_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

# ── Kokoro voice catalogue (54 voices) ────────────────────────────────────────
ALL_KOKORO_VOICES = [
    "bm_daniel","bm_fable","bm_george","bm_lewis",
    "bf_alice","bf_emma","bf_isabella","bf_lily",
    "am_adam","am_echo","am_eric","am_fenrir","am_liam","am_michael","am_onyx","am_puck","am_santa",
    "af_alloy","af_aoede","af_bella","af_heart","af_jessica","af_kore",
    "af_nicole","af_nova","af_river","af_sarah","af_sky",
    "ef_dora","em_alex","em_santa","ff_siwis",
    "hf_alpha","hf_beta","hm_omega","hm_psi",
    "if_sara","im_nicola",
    "jf_alpha","jf_gongitsune","jf_nezumi","jf_tebukuro","jm_kumo",
    "pf_dora","pm_alex","pm_santa",
    "zf_xiaobei","zf_xiaoni","zf_xiaoxiao","zf_xiaoyi",
    "zm_yunjian","zm_yunxi","zm_yunxia","zm_yunyang",
]
KOKORO_LANG_MAP = {
    "am":"en-us","af":"en-us","bm":"en-gb","bf":"en-gb",
    "ef":"es","em":"es","ff":"fr","hf":"hi","hm":"hi",
    "if":"it","im":"it","jf":"ja","jm":"ja","pf":"pt-br","pm":"pt-br","zf":"zh","zm":"zh",
}

# ── XTTS catalogues ───────────────────────────────────────────────────────────
ALL_XTTS_SPEAKERS = [
    "Aaron Dreschner","Abrahan Mack","Adde Michal","Alexandra Hisakawa","Alison Dietlinde",
    "Alma Maria","Ana Florence","Andrew Chipper","Annmarie Nele","Asya Anara",
    "Badr Odhiambo","Baldur Sanjin","Barbora MacLean","Brenda Stern","Camilla Holmstrom",
    "Chandra MacFarland","Claribel Dervla","Craig Gutsy","Daisy Studious","Damien Black",
    "Damjan Chapman","Dionisio Schuyler","Eugenio Mataracı","Ferran Simen","Filip Traverse",
    "Gilberto Mathias","Gitta Nikolina","Gracie Wise","Henriette Usha","Ige Behringer",
    "Ilkin Urbano","Kazuhiko Atallah","Kumar Dahl","Lidiya Szekeres","Lilya Stainthorpe",
    "Ludvig Milivoj","Luis Moray","Maja Ruoho","Marcos Rudaski","Narelle Moon",
    "Nova Hogarth","Rosemary Okafor","Royston Min","Sofia Hellen","Suad Qasim",
    "Szofi Granger","Tammie Ema","Tammy Grit","Tanja Adelina","Torcull Diarmuid",
    "Uta Obando","Viktor Eka","Viktor Menelaos","Vjollca Johnnie","Wulf Carlevaro",
    "Xavier Hayasaka","Zacharie Aimilios","Zofija Kendrick",
]
XTTS_LANGUAGES = {
    "en":"English","fr":"French","de":"German","es":"Spanish","it":"Italian",
    "pt":"Portuguese","pl":"Polish","cs":"Czech","ar":"Arabic","hi":"Hindi",
    "hu":"Hungarian","ja":"Japanese","ko":"Korean","nl":"Dutch","ru":"Russian",
    "tr":"Turkish","zh-cn":"Chinese",
}

# ── Bark voice presets ────────────────────────────────────────────────────────
BARK_PRESETS = [
    ("v2/en_speaker_6", "en_speaker_6 — male, measured (best Arthur)"),
    ("v2/en_speaker_9", "en_speaker_9 — male, older"),
    ("v2/en_speaker_0", "en_speaker_0 — male, deep"),
    ("v2/en_speaker_1", "en_speaker_1 — male, warm"),
    ("v2/en_speaker_7", "en_speaker_7 — male, elderly"),
    ("v2/en_speaker_3", "en_speaker_3 — male, gravelly"),
    ("v2/en_speaker_4", "en_speaker_4 — male, neutral"),
    ("v2/en_speaker_2", "en_speaker_2 — female"),
    ("v2/en_speaker_5", "en_speaker_5 — female"),
    ("v2/en_speaker_8", "en_speaker_8 — female, soft"),
]

CHATTTS_SPEEDS   = [(f"[speed_{i}]", f"speed_{i}") for i in range(1, 10)]
OUTETTS_MODELS   = [
    ("/opt/models/outetts-gguf/OuteTTS-1.0-0.6B-Q4_K_M.gguf", "OuteTTS 1.0 0.6B Q4 (default, ~400 MB)"),
    ("/opt/models/outetts-gguf/OuteTTS-1.0-0.6B-Q8_0.gguf",   "OuteTTS 1.0 0.6B Q8 (higher quality, ~650 MB)"),
    ("/opt/models/outetts-gguf/OuteTTS-0.3-500M-Q4_K_M.gguf", "OuteTTS 0.3 500M Q4 (smaller, ~300 MB)"),
]
OUTETTS_SPEAKERS = [("en-female-1-neutral", "en-female-1-neutral")]
PARLER_MODELS    = [
    ("parler-tts/parler-tts-mini-v1",       "Mini v1"),
    ("parler-tts/parler-tts-mini-expresso",  "Mini Expresso"),
]
ORPHEUS_VOICES   = [("tara","tara"),("leah","leah"),("jess","jess"),("leo","leo"),
                    ("dan","dan"),("mia","mia"),("zac","zac"),("zoe","zoe")]
ZONOS_VARIANTS   = [("transformer","Transformer (quality, ~1.2 GB)"),
                    ("hybrid","Hybrid (faster, ~1.5 GB)")]
CSM_SPEAKERS     = [(str(i), f"Speaker {i}") for i in range(3)]
MATCHA_VOICES    = [
    ("khadijah", "Khadijah — Female, 22050 Hz, FA+EN"),
    ("musa",     "Musa — Male, 22050 Hz, FA+EN"),
]
MATCHA_MODEL_REPOS = {
    "khadijah": "csukuangfj/matcha-tts-fa_en-khadijah",
    "musa":     "csukuangfj/matcha-tts-fa_en-musa",
}
MATCHA_VOCODER_REPO = "csukuangfj/sherpa-onnx-hifigan"
MATCHA_VOCODER_FILE = "hifigan_v2.onnx"
MANATTS_REPO_DIR   = Path("/opt/models/Persian-MultiSpeaker-Tacotron2")
MANATTS_MODEL_REPO = "MahtaFetrat/Persian-Tacotron2-on-ManaTTS"
MANATTS_MAX_CHARS  = 200

# ── OmniVoice language catalogue (600+ languages — key ones for the UI) ────────
OMNIVOICE_LANGUAGES = [
    ("fa",  "Persian / Farsi (فارسی)"),
    ("en",  "English"),
    ("zh",  "Chinese (中文)"),
    ("ar",  "Arabic (العربية)"),
    ("fr",  "French (Français)"),
    ("de",  "German (Deutsch)"),
    ("es",  "Spanish (Español)"),
    ("it",  "Italian (Italiano)"),
    ("pt",  "Portuguese (Português)"),
    ("ru",  "Russian (Русский)"),
    ("ja",  "Japanese (日本語)"),
    ("ko",  "Korean (한국어)"),
    ("hi",  "Hindi (हिन्दी)"),
    ("tr",  "Turkish (Türkçe)"),
    ("ur",  "Urdu (اردو)"),
    ("he",  "Hebrew (עברית)"),
    ("nl",  "Dutch (Nederlands)"),
    ("pl",  "Polish (Polski)"),
    ("sv",  "Swedish (Svenska)"),
    ("th",  "Thai (ไทย)"),
    ("vi",  "Vietnamese (Tiếng Việt)"),
    ("id",  "Indonesian (Bahasa)"),
    ("ms",  "Malay (Bahasa Melayu)"),
    ("bn",  "Bengali (বাংলা)"),
    ("pa",  "Punjabi (ਪੰਜਾਬੀ)"),
    ("ta",  "Tamil (தமிழ்)"),
    ("te",  "Telugu (తెలుగు)"),
    ("mr",  "Marathi (मराठी)"),
    ("gu",  "Gujarati (ગુજરાતી)"),
    ("kn",  "Kannada (ಕನ್ನಡ)"),
    ("ml",  "Malayalam (മലയാളം)"),
    ("si",  "Sinhala (සිංහල)"),
    ("my",  "Burmese (မြန်မာ)"),
    ("km",  "Khmer (ខ្មែរ)"),
    ("lo",  "Lao (ລາວ)"),
    ("fil","Filipino / Tagalog"),
    ("uk",  "Ukrainian (Українська)"),
    ("ro",  "Romanian (Română)"),
    ("hu",  "Hungarian (Magyar)"),
    ("cs",  "Czech (Čeština)"),
    ("sk",  "Slovak (Slovenčina)"),
    ("el",  "Greek (Ελληνικά)"),
    ("da",  "Danish (Dansk)"),
    ("fi",  "Finnish (Suomi)"),
    ("no",  "Norwegian (Norsk)"),
    ("bg",  "Bulgarian (Български)"),
    ("sr",  "Serbian (Српски)"),
    ("hr",  "Croatian (Hrvatski)"),
    ("sw",  "Swahili (Kiswahili)"),
    ("am",  "Amharic (አማርኛ)"),
    ("ha",  "Hausa"),
    ("yo",  "Yoruba (Èdè Yorùbá)"),
    ("zu",  "Zulu (isiZulu)"),
    ("az",  "Azerbaijani (Azərbaycan)"),
    ("kk",  "Kazakh (Қазақ)"),
    ("uz",  "Uzbek (Oʻzbek)"),
    ("tk",  "Turkmen (Türkmen)"),
    ("ky",  "Kyrgyz (Кыргыз)"),
    ("tg",  "Tajik (Тоҷикӣ)"),
    ("ps",  "Pashto (پښتو)"),
    ("ku",  "Kurdish (Kurdî)"),
]

# ── OuteTTS GGUF defaults ─────────────────────────────────────────────────────
OUTETTS_DEFAULT_GGUF      = "/opt/models/outetts-gguf/OuteTTS-1.0-0.6B-Q4_K_M.gguf"
OUTETTS_DEFAULT_TOKENIZER = "OuteAI/OuteTTS-1.0-0.6B"

# ── Qwen3-TTS model ID ────────────────────────────────────────────────────────
# Base variant supports generate_voice_clone() — 3s voice cloning + x-vector mode.
# CustomVoice variant (built-in speakers + instruct) does NOT support voice cloning.
QWEN3TTS_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

# ── Arthur text presets ───────────────────────────────────────────────────────
ARTHUR_PRESETS = [
    ("Greeting",   "Hello? Oh my goodness, who is this? I almost didn't hear the phone. Let me turn the TV down a moment."),
    ("Confused",   "Now where did I put that... oh, I'm sorry dear, I've gone and confused myself again. What was that number you gave me?"),
    ("Intel hunt", "Can you give me that badge number one more time, nice and slow? And the website, could you spell that out letter by letter for me?"),
    ("Stage 4",    "I've been very patient but I can't find my reading glasses and Mr. Whiskers knocked the paper off the table and I keep losing the case number."),
    ("Short",      "Oh, just a moment dear."),
]
BARK_ARTHUR_PRESETS = [
    ("[sighs] Hello? Oh my goodness, who is this? [clears throat] Let me turn the TV down.",
     "Bark: Greeting with sighs"),
    ("[hesitantly] Now where did I put that... [long pause] oh I'm sorry dear, I've gone and confused myself. [laughs nervously]",
     "Bark: Confused with tokens"),
    ("Can you give me that badge number one more time [pause] nice and slow? I keep writing it down and losing it. [sighs]",
     "Bark: Intel hunt"),
]

# ── Model registry ────────────────────────────────────────────────────────────
MODEL_INFO = {
    # ⚡ = real-time capable (RTF < 1.0), 🐌 = very slow (RTF > 10)
    "piper":      {"label":"Piper TTS",     "size":"61-116 MB",          "rtf_est":"RTF 0.43× ⚡",          "ram_est_mb":200,  "heavy":False,"notes":"6 voices. ONNX CPU-only. Real-time on any hardware.","arthur_fit":2},
    "kokoro":     {"label":"Kokoro-82M",    "size":"89 MB",              "rtf_est":"RTF 3.20×",            "ram_est_mb":500,  "heavy":False,"notes":"54 voices, 9 languages. bm_lewis is best Arthur voice.","arthur_fit":5},
    "melo":       {"label":"MeloTTS",       "size":"200 MB",             "rtf_est":"RTF 0.46× ⚡",          "ram_est_mb":1200, "heavy":False,"notes":"5 English accents. EN-BR sounds slightly older.","arthur_fit":3},
    "chattts":    {"label":"ChatTTS",       "size":"1.2-2.3 GB",         "rtf_est":"RTF 2.14×",            "ram_est_mb":1800, "heavy":True, "notes":"Conversational TTS. Ref voice: library bug, falls back to random.","arthur_fit":4},
    "outetts":    {"label":"OuteTTS 1.0",   "size":"384 MB (Q4_K_M)",    "rtf_est":"RTF 15-26× 🐌",        "ram_est_mb":800,  "heavy":True, "notes":"GGUF LLM-based. ~19 tok/s. Auto-capped max_length for speed.","arthur_fit":4},
    "bark":       {"label":"Bark",          "size":"2.5 GB (full)",      "rtf_est":"RTF 5.92×",            "ram_est_mb":3000, "heavy":True, "notes":"Emotion tokens: [laughs] [sighs]. ~12 GB VRAM.","arthur_fit":5},
    "styletts2":  {"label":"StyleTTS 2",    "size":"0.7 GB",             "rtf_est":"RTF 0.22× ⚡",          "ram_est_mb":1500, "heavy":True, "notes":"Fast high-quality TTS. Style transfer from ref WAV.","arthur_fit":4},
    "f5tts":      {"label":"F5-TTS",        "size":"1.2 GB",             "rtf_est":"RTF 5.45×",            "ram_est_mb":2000, "heavy":True, "notes":"Best zero-shot voice cloning. Needs ref WAV + hf-hub>=1.0.","arthur_fit":4},
    "dia":        {"label":"Dia-1.6B",      "size":"3 GB",               "rtf_est":"RTF 7.20×",            "ram_est_mb":3000, "heavy":True, "notes":"Dialogue-native. [S1]/[S2] speakers + emotion tags.","arthur_fit":5},
    "xtts":       {"label":"XTTS-v2",       "size":"1.8 GB",             "rtf_est":"⚠ broken",              "ram_est_mb":3200, "heavy":True, "notes":"⚠ SKIPPED: torchcodec vs torch nightly.","arthur_fit":5},
    "cosyvoice":  {"label":"CosyVoice2",    "size":"2 GB",               "rtf_est":"not built",             "ram_est_mb":2500, "heavy":True, "notes":"⚠ git clone needed + openai-whisper build failure.","arthur_fit":3},
    "parler":     {"label":"Parler-TTS",    "size":"2.5-3.3 GB",         "rtf_est":"skipped",               "ram_est_mb":1500, "heavy":True, "notes":"⚠ SKIPPED: needs legacy stack (torch 1.x + tf 4.x).","arthur_fit":4},
    "chatterbox": {"label":"Chatterbox",    "size":"3.0 GB",             "rtf_est":"RTF 2.42×",            "ram_est_mb":1800, "heavy":True, "notes":"Persian T3 (30-layer, 2454 tokens). Voice cloning. Auto-chunks long text.","arthur_fit":5},
    "fishspeech": {"label":"Fish Speech",   "size":"~1.1 GB",            "rtf_est":"RTF 3.48×",            "ram_est_mb":1500, "heavy":True, "notes":"Zero-shot voice cloning. Persian via LM tokenizer.","arthur_fit":4},
    "csm":        {"label":"Sesame CSM 1B", "size":"~2 GB",              "rtf_est":"blocked (Meta)",        "ram_est_mb":2000, "heavy":True, "notes":"⚠ All deps ready. Blocked: meta-llama/Llama-3.2-1B gated.","arthur_fit":4},
    "qwen3tts":   {"label":"Qwen3-TTS 1.7B", "size":"~3 GB",           "rtf_est":"RTF ~3-6×",         "ram_est_mb":6000, "heavy":True, "notes":"Voice cloning 1.7B Base. 3s ref audio. x-vector-only mode. 10 languages.","arthur_fit":5},
    "orpheus":    {"label":"Orpheus 3B",    "size":"~3 GB",              "rtf_est":"needs container",       "ram_est_mb":3000, "heavy":True, "notes":"⚠ Installed but vllm vs torch nightly. Needs Dockerfile.orpheus.","arthur_fit":5},
    "neutts":     {"label":"NeuTTS Air",    "size":"TBD",                "rtf_est":"not configured",        "ram_est_mb":1000, "heavy":True, "notes":"⚠ Not yet configured — edit _load_neutts() in engines.","arthur_fit":3},
    "indextts":   {"label":"IndexTTS-2",    "size":"~1.5 GB",            "rtf_est":"skipped",               "ram_est_mb":2000, "heavy":True, "notes":"⚠ SKIPPED: needs legacy stack.","arthur_fit":4},
    "zonos":      {"label":"Zonos v0.1",    "size":"~1.2 GB",            "rtf_est":"RTF 4.29×",            "ram_est_mb":2500, "heavy":True, "notes":"Emotion vector + speaking-rate. 44 kHz. Voice cloning.","arthur_fit":4},
    "openvoice":  {"label":"OpenVoice v2",  "size":"~600 MB",            "rtf_est":"not built",             "ram_est_mb":1500, "heavy":True, "notes":"⚠ Build failure — av package Cython.","arthur_fit":3},
    "matcha":     {"label":"Matcha-TTS (FA/EN)","size":"~74 MB per voice",  "rtf_est":"RTF 0.24× ⚡",          "ram_est_mb":400,  "heavy":False,"notes":"Fast flow-matching ONNX. Real-time! Khadijah (F) + Musa (M).","arthur_fit":3},
    "manatts":    {"label":"ManaTTS (FA)",  "size":"~371 MB + encoder",    "rtf_est":"not available",         "ram_est_mb":2500, "heavy":True, "notes":"⚠ parallel-wavegan not available on PyPI.","arthur_fit":3},
    "mmsfas":     {"label":"MMS Persian (Meta)","size":"~150 MB",           "rtf_est":"RTF ~0.5×",            "ram_est_mb":200,  "heavy":False,"notes":"Meta MMS-TTS Persian VITS. Reference baseline. CC-BY-NC.","arthur_fit":5},
    "chatterboxturbo": {"label":"Chatterbox-Turbo","size":"~700 MB",       "rtf_est":"RTF 1.11×",            "ram_est_mb":1500, "heavy":True, "notes":"350M distilled one-step TTS. Voice cloning. Near real-time!","arthur_fit":4},
    "vibevoice":  {"label":"VibeVoice-1.5B","size":"~6 GB (BF16)",         "rtf_est":"needs SGLang",          "ram_est_mb":6500, "heavy":True, "notes":"⚠ SGLang image tf too old. Needs upstream update.","arthur_fit":3},
    "higgs":      {"label":"Higgs Audio v3","size":"~8 GB (BF16)",         "rtf_est":"needs SGLang",          "ram_est_mb":8500, "heavy":True, "notes":"⚠ SGLang image tf too old. Needs upstream update.","arthur_fit":3},
    "omnivoice":  {"label":"OmniVoice",     "size":"~1.2 GB (BF16)",       "rtf_est":"RTF 0.67× ⚡",          "ram_est_mb":2000, "heavy":True, "notes":"0.6B diffusion LM. 600+ languages. Real-time!","arthur_fit":4},
    "s2pro":      {"label":"Fish S2-Pro",   "size":"~10 GB (BF16, 5B)",   "rtf_est":"needs SGLang",          "ram_est_mb":10000,"heavy":True,"notes":"⚠ SGLang image tf too old. Needs upstream update.","arthur_fit":3},
    "qwen36":     {"label":"Qwen3.6-35B-A3B","size":"~13 GB (TQ3_4S GGUF)","rtf_est":"LLM — ~107 tok/s","ram_est_mb":13000,"heavy":True,"notes":"Alibaba Qwen 3.6 MoE. 35B total, 3B active params. Reasoning + coding via llama.cpp. Evicts ALL TTS engines before loading.","arthur_fit":3,"engine_type":"llm"},
}

MODEL_ORDER = [
    "piper","kokoro","melo","matcha",
    "chattts","outetts","bark","styletts2","f5tts","dia","xtts",
    "cosyvoice","parler","chatterbox","chatterboxturbo","fishspeech","csm","qwen3tts","orpheus",
    "neutts","indextts","manatts","mmsfas","zonos","openvoice",
    "vibevoice","higgs","omnivoice","s2pro",
    "qwen36",
]

HEAVY = {n for n, i in MODEL_INFO.items() if i["heavy"]}

# ── Per-engine shared runtime state ──────────────────────────────────────────
_state: dict = {
    n: {
        "instance":    None,
        "status":      "unloaded",
        "lock":        threading.Lock(),
        "error":       "",
        "load_time_s": 0.0,
        "loaded_voice": None,
        "loaded_model": None,
    }
    for n in MODEL_ORDER
}

# ── Server-side ring-buffer log (last 400 entries) ────────────────────────────
import collections, datetime as _dt
_server_log: collections.deque = collections.deque(maxlen=400)
_server_log_seq = 0

def slog(cat: str, engine: str, msg: str) -> None:
    """Append a categorised entry to the server-side log ring-buffer."""
    global _server_log_seq
    _server_log_seq += 1
    _server_log.append({
        "seq": _server_log_seq,
        "ts":  _dt.datetime.now().strftime("%H:%M:%S.") + f"{_dt.datetime.now().microsecond//1000:03d}",
        "cat": cat,
        "engine": engine,
        "msg": msg,
    })


SYNTH_TIMEOUT: dict[str, int] = {
    "orpheus":  240,
    "dia":      180,
    "bark":     180,
    "qwen3tts": 180,
    "outetts":  120,
    "f5tts":    120,
    "manatts":  120,
    "fishspeech":360,
    "chattts":   90,
}
DEFAULT_SYNTH_TIMEOUT = 300
