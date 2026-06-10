# Ideogram 4 HTTP API — Developer Reference v2

> For content generation tools calling the Arthur Image Lab's Ideogram 4 engine via HTTP/HTTPS.
> **No authentication required** (internal service, same network).
> **Base URL:** `http://192.168.0.87:8002`

---

## 1. Quick Start

### Direct JSON caption (fastest — no API calls)

```bash
curl -X POST http://192.168.0.87:8002/generate/ideogram4 \
  -F 'prompt={"high_level_description":"a clean explainer slide about solar energy","style_description":{"art_style":"editorial explainer, paper texture, clean typography, black text with orange accents","medium":"educational slide","color_palette":["#FFFFFF","#1A1A1A","#FF6B35"]},"compositional_deconstruction":{"background":"white paper texture with subtle grid","elements":[{"type":"text","text":"HOW SOLAR PANELS WORK","desc":"large bold headline at top"},{"type":"text","text":"1. Sunlight hits panel    2. Electrons flow   3. Inverter converts to AC","desc":"three step process in clean list with orange arrows"}]}}' \
  -F 'width=1280' \
  -F 'height=720' \
  -F 'preset=V4_QUALITY_48' \
  -F 'guidance_scale=10.0' \
  -F 'quant=nf4' \
  -o output.png
```

### Magic Prompt (plain text → auto-expand to JSON)

```bash
curl -X POST http://192.168.0.87:8002/generate/ideogram4 \
  -F 'prompt=a vintage newspaper front page with headline BREAKING NEWS, two columns of article text, old paper texture, black ink on aged yellowish paper, dated June 9 2026' \
  -F 'use_magic_prompt=true' \
  -F 'magic_prompt_aspect_ratio=3:4' \
  -F 'width=768' \
  -F 'height=1024' \
  -F 'preset=V4_QUALITY_48' \
  -F 'guidance_scale=10.0' \
  -F 'quant=nf4' \
  -o newspaper.png
```

---

## 2. Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| `POST` | `/generate/ideogram4` | Generate one image |
| `GET` | `/status` | Check engine availability, VRAM, model state |
| `GET` | `/files/images/{filename}` | Download generated image |

### `/generate/ideogram4` Response

```json
{
  "results": [
    {
      "id": "a48b6908-...",
      "engine": "ideogram4",
      "filename": "ideogram4_a48b6908-....png",
      "url": "/files/images/ideogram4_a48b6908-....png",
      "base64": "iVBORw0KGgo...",
      "type": "image"
    }
  ]
}
```

Download: `http://192.168.0.87:8002/files/images/ideogram4_[id].png`

---

## 3. Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | **required** | JSON caption or plain text (see §4) |
| `width` | int | 1024 | Output width (16–2048, multiples of 16) |
| `height` | int | 1024 | Output height (16–2048, multiples of 16) |
| `preset` | string | `V4_DEFAULT_20` | Sampler preset (see §6) |
| `num_inference_steps` | int | 0 | Override steps (0 = use preset default) |
| `guidance_scale` | float | 7.0 | CFG strength (1.0–30.0). Use ≥7.0 for text rendering |
| `seed` | int | -1 | Random seed (-1 = random) |
| `quant` | string | `nf4` | Quantization: `nf4` or `fp8` |
| `mu` | float | 0.0 | Schedule mean (advanced, 0 = use preset) |
| `std` | float | 1.75 | Schedule std (advanced, 1.75 = use preset) |
| `use_magic_prompt` | bool | false | Expand prompt from plain text → JSON via LLM API (see §5) |
| `magic_prompt_aspect_ratio` | string | `1:1` | Target ratio for expansion: `16:9`, `3:4`, `1:1`, `2:3`, etc. |

---

## 4. Two Prompt Modes

### Mode A: Direct JSON (`use_magic_prompt=false`, default)

You construct the JSON caption yourself. **Zero API overhead** — fastest path. Use when you have a pre-built caption or generate JSON programmatically.

### Mode B: Magic Prompt (`use_magic_prompt=true`)

You write **plain text**. The engine expands it into a structured JSON caption via an LLM API. The `prompt` field is the only input — no separate "magic input" field.

**Expansion priority chain** (first configured key wins):

| Pri | Provider | Env Var | System Prompt | Latency | Cost | Quality |
|:---:|----------|---------|---------------|:-------:|:----:|:-------:|
| 1 | **Ideogram hosted** | `IDEOGRAM_API_KEY` | None (server-side) | ~2–5s | **Free** | 🏆 Best |
| 2 | **DeepSeek native** | `DEEPSEEK_API_KEY` | v1.txt (28 KB, ~6.9K tokens) | ~3–8s | ~$0.14/M tok | Very good |
| 3 | **OpenRouter → DeepSeek** | `OPENROUTER_API_KEY` | v1.txt (28 KB, ~6.9K tokens) | ~3–8s | ~$0.14/M tok | Very good |

If no keys are configured, the plain text is passed as-is to Ideogram 4 — which will produce poor results since it expects JSON.

### v1.txt — Ideogram's Official Expansion Recipe

When DeepSeek or OpenRouter is used, the system prompt is Ideogram's own hand-crafted **v1.txt** (296 lines, ~6,900 tokens) shipped with the `ideogram4` package. It encodes 19 specific formatting rules including:

- Single-line minified JSON, no markdown fences
- Three top-level keys only: `aspect_ratio`, `high_level_description`, `compositional_deconstruction`
- 50-word hard cap on HLD
- **SINGLE SUBJECT = SINGLE ELEMENT** (prevents fragmentation artifacts — a bee is one element, not 8)
- **Ground/floor ALWAYS in background** (prevents "legs buried in ground" rendering bug)
- **Shell-affixed objects → DUAL MENTION** (name in background + emit as first element — prevents floating chalkboards)
- Non-ASCII preservation (CJK, Cyrillic, Arabic), single-quote conventions for embedded references
- No shadows or camera language in element descriptions
- 30–60 word element descs with 60-word hard cap
- Pop-culture named references (never genericize brands/characters)
- Transparent background: exact verbatim string required

Full source: `/opt/arthur-img/ideogram4/src/ideogram4/magic_prompt_system_prompts/v1.txt`

---

## 5. JSON Caption Schema

Two schema variants are accepted:

### Variant A: With `style_description`

```json
{
  "high_level_description": "concise scene summary (50 words max)",
  "style_description": {
    "art_style": "editorial explainer graphic, paper texture, clean white...",
    "color_palette": ["#FFFFFF", "#1A1A1A", "#FF6B35"],
    "medium": "educational slide"
  },
  "compositional_deconstruction": {
    "background": "detailed background description",
    "elements": [
      {"type": "text", "text": "RENDERED TEXT HERE", "desc": "visual description of how it looks"},
      {"type": "obj",  "desc": "description of a non-text visual element"}
    ]
  }
}
```

### Variant B: Native v1.txt format

```json
{
  "aspect_ratio": "16:9",
  "high_level_description": "concise summary",
  "compositional_deconstruction": {
    "background": "detailed background",
    "elements": [
      {"type": "text", "bbox": [100, 50, 150, 400], "text": "LINE ONE\nLINE TWO", "desc": "visual description"},
      {"type": "obj",  "bbox": [200, 200, 500, 600], "desc": "object description"}
    ]
  }
}
```

### Critical Rules

| Rule | Detail |
|------|--------|
| `type: "text"` elements | **Must** have both `"text"` (literal characters to render) AND `"desc"` (visual description) |
| `type: "obj"` elements | Only need `"type"` and `"desc"` |
| Multi-line text | Use `\n`: `"LINE ONE\nLINE TWO"` |
| `high_level_description` | 50-word hard cap. Reads like a search query, not prose |
| Color palette | 3–5 hex codes defining the scene's color identity |
| `bbox` | Optional `[y1, x1, y2, x2]` in 0–1000 normalised coordinates |
| Element descriptions | 30–60 words recommended, 60-word hard cap |
| No shadows in elements | Lighting/shadow belongs in `background`, not individual elements |

---

## 6. Presets

| Preset | Steps | ~Time | Best For |
|--------|:-----:|:-----:|----------|
| `V4_TURBO_12` | 12 | ~15s | Fast preview, iteration, low quality |
| `V4_DEFAULT_20` | 20 | ~25s | Good balance of speed and quality |
| `V4_QUALITY_48` | 48 | ~60s | Maximum quality, text rendering, production use |

### Recommended Configurations by Use Case

| Use Case | W × H | Preset | Guidance | Notes |
|----------|:-------:|--------|:--------:|-------|
| Explainer slide (16:9) | 1280 × 720 | V4_QUALITY_48 | 10.0 | Text-heavy content needs high guidance |
| Newspaper (portrait) | 768 × 1024 | V4_QUALITY_48 | 10.0 | Dense text layout, old paper aesthetic |
| Photograph (1:1) | 1024 × 1024 | V4_DEFAULT_20 | 7.0 | Natural scenes, lower guidance for realism |
| Quick preview | 512 × 512 | V4_TURBO_12 | 7.0 | Fast iteration during development |
| High-res photo | 1024 × 1280 | V4_QUALITY_48 | 7.0 | Maximum resolution tested on 16 GB GPU |

---

## 7. Converting GPT Prompts to Ideogram 4 JSON

If you have a GPT-style prompt describing a style system, convert it in two steps:

### Step 1: Style → `high_level_description` + `style_description`

**GPT prompt excerpt:**
> *"clean editorial explainer visuals, paper-style layouts, text, simple supporting graphics, clean white or very light paper-like background, black text as main text color, orange highlight color, clean and minimal layout..."*

**Becomes:**
```json
{
  "high_level_description": "a clean 16:9 editorial explainer slide with paper texture background, black text, orange accents, minimal design",
  "style_description": {
    "art_style": "editorial explainer graphic, paper texture background, clean white layout, black typography with orange accent highlights, premium educational style, minimal uncluttered design, 16:9 horizontal format, soft paper shadows, subtle grid lines",
    "color_palette": ["#FFFFFF", "#1A1A1A", "#FF6B35", "#F5F0E8"],
    "medium": "educational slide"
  }
}
```

### Step 2: Content → `compositional_deconstruction.elements`

Each key point, timestamp, or content block becomes one element. Text elements need both `"text"` (what appears) and `"desc"` (how it looks).

### Full Example: Photosynthesis Explainer Slide

```json
{
  "high_level_description": "a clean 16:9 explainer slide about photosynthesis in 3 steps, white paper background, black text, orange accents",
  "style_description": {
    "art_style": "editorial explainer graphic, paper texture, clean white background, black typography with orange accents, minimal design, educational slide, soft paper shadows, clean icons and arrows",
    "color_palette": ["#FFFFFF", "#1A1A1A", "#FF6B35", "#F5F0E8"],
    "medium": "educational slide"
  },
  "compositional_deconstruction": {
    "background": "clean white paper texture with very subtle grid lines, soft paper edge shadow on all four sides",
    "elements": [
      {
        "type": "text",
        "text": "HOW PHOTOSYNTHESIS WORKS",
        "desc": "large bold headline at top center in black, 48pt sans-serif font, with a thin orange underline accent"
      },
      {
        "type": "text",
        "text": "Step 1: Sunlight is absorbed by chlorophyll in the leaves",
        "desc": "left column, medium black text with a small orange circle icon containing the number 1"
      },
      {
        "type": "obj",
        "desc": "a simple orange arrow pointing right, connecting step 1 to step 2"
      },
      {
        "type": "text",
        "text": "Step 2: Water molecules split, releasing oxygen",
        "desc": "center column, medium black text with a small orange circle icon containing the number 2"
      },
      {
        "type": "obj",
        "desc": "a simple orange arrow pointing right, connecting step 2 to step 3"
      },
      {
        "type": "text",
        "text": "Step 3: Carbon dioxide is converted into glucose",
        "desc": "right column, medium black text with a small orange circle icon containing the number 3"
      },
      {
        "type": "text",
        "text": "CO2 + H2O + Light \u2192 C6H12O6 + O2",
        "desc": "small chemical equation in a light gray rounded box at the bottom center, monospace font, with an orange arrow"
      }
    ]
  }
}
```

### Full Example: Vintage Newspaper Front Page

```json
{
  "high_level_description": "a vintage newspaper front page with bold BREAKING NEWS headline, two columns of article text, a scientific diagram, old paper texture, black ink on aged yellowish paper",
  "style_description": {
    "art_style": "vintage newspaper front page, aged yellowish paper texture, black ink, serif typography, 19th century newspaper aesthetic, column layout, slightly yellowed and weathered paper, authentic newsprint feel",
    "color_palette": ["#F5E6C8", "#1A1A1A", "#8B7355", "#D4C5A0"],
    "medium": "newspaper print"
  },
  "compositional_deconstruction": {
    "background": "aged yellowish paper with subtle fiber texture, slightly yellowed with age, soft vignette at edges from old paper wear",
    "elements": [
      {
        "type": "text",
        "text": "THE DAILY CHRONICLE",
        "desc": "small newspaper nameplate at very top in elegant serif font, centered, black ink, decorative thin line below"
      },
      {
        "type": "text",
        "text": "June 9, 2026  |  Vol. CXLVII  |  Price: Two Cents",
        "desc": "date line below nameplate in small serif font, thin horizontal rules above and below"
      },
      {
        "type": "text",
        "text": "BREAKING NEWS",
        "desc": "massive bold headline spanning full width in heavy serif font, the main focal point of the page, black ink, slightly letterpressed texture"
      },
      {
        "type": "text",
        "text": "Scientists Announce Revolutionary Discovery That Could Reshape Modern Understanding of Quantum Mechanics",
        "desc": "subheadline below main headline in slightly smaller bold serif, single line spanning full width, black ink"
      },
      {
        "type": "text",
        "text": "WASHINGTON \u2014 In a stunning announcement that has sent shockwaves through the scientific community, researchers at the National Quantum Institute revealed findings that fundamentally challenge long-held assumptions about particle behavior at the subatomic level. The discovery, published in yesterday\u2019s issue of Nature, suggests that quantum entanglement may operate across temporal boundaries previously thought impossible.\n\nThe research team, led by Dr. Elena Vasquez, spent seven years conducting experiments at the Large Hadron Collider before reaching their conclusions. Their work could pave the way for quantum computing applications that were once considered purely theoretical.",
        "desc": "left column of body text in small serif font, justified alignment, black ink, approximately 150 words of dense newsprint copy"
      },
      {
        "type": "obj",
        "desc": "a scientific diagram illustration showing a simplified atom model with electron orbits, rendered as a woodcut-style line engraving, centered between the two text columns, approximately 200px square"
      },
      {
        "type": "text",
        "text": "The implications are profound and far-reaching. If quantum states can indeed maintain coherence across temporal displacement, it would fundamentally alter our understanding of causality at the quantum level. Several prominent physicists have already described the findings as the most significant development in the field since Bell\u2019s Theorem.",
        "desc": "right column of body text continuing the article, small serif font, justified alignment, black ink, approximately 80 words"
      },
      {
        "type": "text",
        "text": "Continued on Page A12",
        "desc": "small continuation notice at bottom right in italic serif font, black ink"
      }
    ]
  }
}
```

---

## 8. Prompt Template Functions

### TypeScript

```typescript
interface ExplainerFrame {
  headline: string;
  bodyPoints: string[];
  accentColor: string;
}

function buildExplainerCaption(frame: ExplainerFrame): string {
  const elements = frame.bodyPoints.map((point, i) => ({
    type: "text" as const,
    text: point,
    desc: `key point ${i + 1} in clean black sans-serif font with small orange circle indicator`
  }));

  return JSON.stringify({
    high_level_description: `a clean 16:9 editorial explainer slide about: ${frame.headline}`,
    style_description: {
      art_style: "editorial explainer graphic, paper texture, clean white layout, black typography, orange accent highlights, premium educational style, minimal uncluttered design, soft paper shadows, 16:9 horizontal format",
      color_palette: ["#FFFFFF", "#1A1A1A", frame.accentColor, "#F5F0E8"],
      medium: "educational slide"
    },
    compositional_deconstruction: {
      background: "clean white paper texture with subtle grid, soft paper edge shadow",
      elements: [
        {
          type: "text",
          text: frame.headline.toUpperCase(),
          desc: "large bold black headline at top with thin orange underline, 48pt sans-serif"
        },
        ...elements
      ]
    }
  });
}
```

### Python

```python
import json

def build_explainer_caption(headline: str, body_points: list[str], accent_color: str = "#FF6B35") -> str:
    elements = [
        {
            "type": "text",
            "text": point,
            "desc": f"key point {i+1} in clean black sans-serif font with small orange circle indicator"
        }
        for i, point in enumerate(body_points)
    ]

    return json.dumps({
        "high_level_description": f"a clean 16:9 editorial explainer slide about: {headline}",
        "style_description": {
            "art_style": "editorial explainer graphic, paper texture, clean white layout, black typography, orange accent highlights, premium educational style, minimal uncluttered design, soft paper shadows, 16:9 horizontal format",
            "color_palette": ["#FFFFFF", "#1A1A1A", accent_color, "#F5F0E8"],
            "medium": "educational slide"
        },
        "compositional_deconstruction": {
            "background": "clean white paper texture with subtle grid, soft paper edge shadow",
            "elements": [
                {
                    "type": "text",
                    "text": headline.upper(),
                    "desc": "large bold black headline at top with thin orange underline, 48pt sans-serif"
                },
                *elements
            ]
        }
    }, ensure_ascii=False)
```

---

## 9. Batch Generation Patterns

### Python — Direct JSON Mode

```python
import urllib.request
import urllib.parse
import json

BASE = "http://192.168.0.87:8002"

frames = [
    {"headline": "Solar Energy Basics", "points": ["..."], "ts": "00:42"},
    {"headline": "How Panels Work",      "points": ["..."], "ts": "01:15"},
]

for frame in frames:
    prompt = build_explainer_caption(frame["headline"], frame["points"])
    data = urllib.parse.urlencode({
        "prompt": prompt,
        "width": "1280", "height": "720",
        "preset": "V4_QUALITY_48",
        "guidance_scale": "10.0",
        "seed": "-1", "quant": "nf4"
    }, doseq=True).encode()

    req = urllib.request.Request(f"{BASE}/generate/ideogram4", data=data, method="POST")
    with urllib.request.urlopen(req, timeout=600) as resp:
        result = json.loads(resp.read())

    url = result["results"][0]["url"]
    with urllib.request.urlopen(f"{BASE}{url}") as img_resp:
        img_data = img_resp.read()

    with open(f"frame_{frame['ts']}.png", "wb") as f:
        f.write(img_data)
    print(f"Generated: frame_{frame['ts']}.png")
```

### Python — Magic Prompt Mode (Plain Text)

```python
data = urllib.parse.urlencode({
    "prompt": "a clean 16:9 explainer slide about solar energy basics, white paper background, black text, orange accents, educational style, minimal layout",
    "use_magic_prompt": "true",
    "magic_prompt_aspect_ratio": "16:9",
    "width": "1280", "height": "720",
    "preset": "V4_QUALITY_48",
    "guidance_scale": "10.0",
    "seed": "-1", "quant": "nf4"
}, doseq=True).encode()

req = urllib.request.Request(f"{BASE}/generate/ideogram4", data=data, method="POST")
with urllib.request.urlopen(req, timeout=600) as resp:
    result = json.loads(resp.read())
```

---

## 10. VRAM & Performance

| Metric | Value |
|--------|-------|
| GPU | RTX 5060 Ti 16 GB (Blackwell) |
| Idle VRAM (model loaded, no generation) | ~10.6 GB |
| Peak VRAM (during generation) | ~11.9 GB |
| TURBO_12 (12 steps) | ~12–15 seconds |
| DEFAULT_20 (20 steps) | ~22–28 seconds |
| QUALITY_48 (48 steps) | ~55–65 seconds |
| Magic prompt overhead (API call) | +3–8 seconds |
| Maximum tested resolution | 1024 × 1280 |
| Concurrent requests | 1 (single GPU, sequential only) |
| Model load time (first request after restart) | ~6–8 minutes |
| Model stays loaded? | Yes — persists between requests |

**Important:** The first request after a service restart triggers model loading (~6–8 min). Poll `GET /status` and check `loaded: true` before sending batch jobs.

---

## 11. Common Issues & Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `503 Service Unavailable` | Model loading or another generation in progress | Wait 10s, retry |
| `CUDA out of memory` | Resolution exceeds available VRAM | Reduce to 1280×720, use `nf4` quant |
| `caption verifier flagged` | JSON format doesn't match expected schema | Ensure `"text"` key present in text elements, `"photo"` or `"art_style"` in style |
| Garbled or missing text | `guidance_scale` too low | Use ≥7.0 (10.0 recommended for text-heavy content) |
| Magic prompt returns poor results | No API key configured | Set `DEEPSEEK_API_KEY` or `IDEOGRAM_API_KEY` |
| `Unknown quantization: .` | Empty `quant` parameter | Always explicitly pass `quant=nf4` |
| Image has floating objects | Missing dual-mention for shell-affixed items | Name wall-mounted objects in BOTH background and elements |
| Subject fragmented | Too many elements for one subject | Single subject = single element (bee = 1, not 8) |

---

## 12. Style Checklist (Explainer Slides)

Before submitting a generation, verify:

- [ ] Correct aspect ratio configured (`width` + `height` match `magic_prompt_aspect_ratio`)
- [ ] Clean white or very light paper-like background
- [ ] Black text as primary content color
- [ ] Orange (`#FF6B35`) used for accents only (arrows, underlines, circle indicators)
- [ ] Minimal, uncluttered layout — not busy
- [ ] Editorial explainer aesthetic — not cinematic 3D rendering
- [ ] Paper texture feel, soft shadows on edges
- [ ] Headline is bold, large, immediately readable
- [ ] Body text is short, clear, scannable
- [ ] Simple supporting graphics when helpful (arrows, icons, rounded boxes)
- [ ] NO logos, branding, watermarks
- [ ] NO dark backgrounds, flashy visuals, particle effects
- [ ] Each frame communicates ONE clear idea only
- [ ] Layout style is consistent across frames in a series
- [ ] 30–60 words per element description
- [ ] `guidance_scale` ≥ 7.0 for frames containing text

---

## 13. Full Shell Examples

### Explainer Slide (direct JSON)

```bash
curl -s -X POST http://192.168.0.87:8002/generate/ideogram4 \
  -F 'prompt={"high_level_description":"a clean 16:9 explainer slide","style_description":{"art_style":"editorial explainer, paper texture, black text, orange accents","color_palette":["#FFFFFF","#1A1A1A","#FF6B35"],"medium":"educational slide"},"compositional_deconstruction":{"background":"white paper with subtle grid","elements":[{"type":"text","text":"HOW IT WORKS","desc":"bold headline"},{"type":"text","text":"Key concept explained clearly","desc":"body text, clean sans-serif"}]}}' \
  -F 'width=1280' -F 'height=720' \
  -F 'preset=V4_QUALITY_48' -F 'guidance_scale=10.0' \
  -F 'quant=nf4' \
  -o output.png
```

### Newspaper (magic prompt — plain text auto-expanded)

```bash
curl -s -X POST http://192.168.0.87:8002/generate/ideogram4 \
  -F 'prompt=a vintage newspaper front page with headline BREAKING NEWS, two columns of dense article text, old paper texture, black ink on aged yellowish paper, dated June 9 2026' \
  -F 'use_magic_prompt=true' \
  -F 'magic_prompt_aspect_ratio=3:4' \
  -F 'width=768' -F 'height=1024' \
  -F 'preset=V4_QUALITY_48' -F 'guidance_scale=10.0' \
  -F 'seed=42' -F 'quant=nf4' \
  -o newspaper.png
```

### Batch Explainer (3 frames, direct JSON)

```bash
#!/bin/bash
BASE="http://192.168.0.87:8002"
HEADLINES=("SOLAR ENERGY BASICS" "HOW PANELS WORK" "FUTURE OF SOLAR")
TIMESTAMPS=("00:42" "01:15" "02:30")

for i in "${!HEADLINES[@]}"; do
  HL="${HEADLINES[$i]}"
  TS="${TIMESTAMPS[$i]}"

  PROMPT=$(cat <<EOF
{"high_level_description":"clean 16:9 explainer slide: $HL","style_description":{"art_style":"editorial explainer, paper texture, black text, orange accents","color_palette":["#FFFFFF","#1A1A1A","#FF6B35"],"medium":"educational slide"},"compositional_deconstruction":{"background":"white paper with subtle grid","elements":[{"type":"text","text":"$HL","desc":"bold headline, 48pt, orange underline"},{"type":"text","text":"Key concept for $TS","desc":"body text, clean sans-serif"}]}}
EOF
)

  curl -s -X POST "$BASE/generate/ideogram4" \
    -F "prompt=$PROMPT" \
    -F 'width=1280' -F 'height=720' \
    -F 'preset=V4_QUALITY_48' -F 'guidance_scale=10.0' \
    -F 'seed=-1' -F 'quant=nf4' \
    -o "frame_${TS}.json"

  URL=$(python3 -c "import json; print(json.load(open('frame_${TS}.json'))['results'][0]['url'])")
  curl -s "$BASE$URL" -o "frame_${TS}.png"
  echo "Generated: frame_${TS}.png"
done
```
