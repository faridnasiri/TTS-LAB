# Ideogram 4 HTTP API — Developer Reference

> For content generation tools calling the Arthur Image Lab's Ideogram 4 engine via HTTP/HTTPS.
> **No authentication required** (internal service, same network).
> **Base URL:** `http://192.168.0.87:8002`

---

## 1. Quick Start

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

---

## 2. Endpoint

| Method | URL | Description |
|--------|-----|-------------|
| `POST` | `/generate/ideogram4` | Generate one image |
| `GET` | `/status` | Check engine availability, VRAM, model state |

### `/status` Response
```json
{
  "engines": [
    { "key": "ideogram4", "label": "Ideogram 4", "available": true, "loaded": true }
  ],
  "active_engine": "ideogram4",
  "vram": { "total_gb": 16.0, "reserved_gb": 11.9 }
}
```

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

To download: `http://192.168.0.87:8002/files/images/ideogram4_[id].png`

---

## 3. Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | **required** | JSON caption (see Section 4) |
| `width` | int | 1024 | Output width (16–2048, multiples of 16) |
| `height` | int | 1024 | Output height (16–2048, multiples of 16) |
| `preset` | string | `V4_DEFAULT_20` | Sampler preset (see Section 5) |
| `num_inference_steps` | int | 0 | Override steps (0 = use preset default) |
| `guidance_scale` | float | 7.0 | CFG strength (1.0–30.0) |
| `seed` | int | -1 | Random seed (-1 = random) |
| `quant` | string | `nf4` | Quantization: `nf4` or `fp8` |
| `mu` | float | 0.0 | Schedule mean (advanced) |
| `std` | float | 1.75 | Schedule std (advanced) |
| `use_magic_prompt` | bool | false | Auto-expand plain text to JSON via LLM |
| `magic_prompt_input` | string | "" | Plain text to expand (if use_magic_prompt) |
| `magic_prompt_aspect_ratio` | string | `1:1` | Target ratio for magic prompt |

### Recommended 16:9 settings
```
width=1280  height=720   preset=V4_QUALITY_48  guidance_scale=10.0
```

---

## 4. The JSON Caption — Your Prompt Format

Ideogram 4 requires a **structured JSON caption** — not plain text. This is the most critical part.

### Schema

```json
{
  "high_level_description": "...",
  "style_description": {
    "photo": "..."          // OR "art_style": "..."
  },
  "compositional_deconstruction": {
    "background": "...",
    "elements": [
      {
        "type": "text",     // "text" or "obj"
        "text": "...",      // THE ACTUAL TEXT TO RENDER
        "desc": "..."       // Description of how it looks
      }
    ]
  }
}
```

### Key Rules

| Rule | Detail |
|------|--------|
| `style_description` | Must contain either `"photo"` (photorealistic) or `"art_style"` (illustration) as the main key |
| `type: "text"` elements | Must have BOTH `"text"` (the literal characters to render) AND `"desc"` (visual description) |
| `type: "obj"` elements | Only need `"type"` and `"desc"` (for non-text objects) |
| `high_level_description` | A concise summary like a search query |
| Text rendering | Ideogram 4 natively renders the characters you put in `"text"` — make them exactly what you want to appear |

---

## 5. Presets

| Preset | Steps | Use Case |
|--------|:-----:|----------|
| `V4_TURBO_12` | 12 | Fast preview (~15s), lower quality |
| `V4_DEFAULT_20` | 20 | Good balance (~25s) |
| `V4_QUALITY_48` | 48 | Best quality, text rendering (~60s) |

---

## 6. Converting Your GPT Prompt to Ideogram 4 JSON

Your GPT prompt describes a **style system** — now we convert it into Ideogram 4's structured format.

### Step 1: Style becomes `high_level_description` + `style_description`

From your GPT prompt:
```
"clean editorial explainer visuals, paper-style layouts, text, simple supporting graphics, 
clean white or very light paper-like background, black text as main text color, 
orange highlight color, clean and minimal layout..."
```

Becomes:
```json
{
  "high_level_description": "a clean editorial explainer slide in 16:9 format with paper texture background, black text, orange accents, minimal design",
  "style_description": {
    "art_style": "editorial explainer graphic, paper texture background, clean white layout, black typography with orange highlight accents, premium educational style, minimal uncluttered design, 16:9 horizontal format, soft paper shadows, subtle grid lines",
    "color_palette": ["#FFFFFF", "#1A1A1A", "#FF6B35", "#F5F0E8", "#E8E0D5"],
    "medium": "educational slide"
  }
}
```

### Step 2: Content becomes `compositional_deconstruction`

Each "timestamp" becomes one `element` in the elements array. Use `type: "text"` for text you want rendered, `type: "obj"` for graphics.

### Full Conversion Example

Given a timestamp idea: *"The process of photosynthesis in 3 steps"*

```json
{
  "high_level_description": "a clean 16:9 explainer slide about photosynthesis in 3 steps, white paper background, black text, orange accents",
  "style_description": {
    "art_style": "editorial explainer graphic, paper texture, clean white background, black typography with orange accents, minimal design, educational slide, soft paper shadows, clean icons and arrows, premium educational style",
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
        "desc": "left column, medium black text with a small orange circle icon with the number 1 inside"
      },
      {
        "type": "obj",
        "desc": "a simple orange arrow pointing right, connecting step 1 to step 2"
      },
      {
        "type": "text",
        "text": "Step 2: Water molecules split, releasing oxygen",
        "desc": "center column, medium black text with a small orange circle icon with the number 2 inside"
      },
      {
        "type": "obj",
        "desc": "a simple orange arrow pointing right, connecting step 2 to step 3"
      },
      {
        "type": "text",
        "text": "Step 3: Carbon dioxide is converted into glucose",
        "desc": "right column, medium black text with a small orange circle icon with the number 3 inside"
      },
      {
        "type": "text",
        "text": "CO2 + H2O + Light → C6H12O6 + O2",
        "desc": "small chemical equation in a light gray rounded box at the bottom center, monospace font, with an orange equals sign"
      }
    ]
  }
}
```

### Prompt Template Function (JavaScript/TypeScript)

```typescript
interface ExplainerFrame {
  headline: string;       // Main title shown on the slide
  bodyPoints: string[];   // Key points rendered as text
  accentColor: string;    // "#FF6B35"
  timestamp: string;      // For filename
}

function buildIdeogram4Prompt(frame: ExplainerFrame): string {
  const elements = frame.bodyPoints.map((point, i) => ({
    type: "text" as const,
    text: point,
    desc: `key point ${i + 1} in clean black sans-serif font with small orange circle indicator`
  }));

  const caption = {
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
          desc: `large bold black headline at top with thin orange underline, 48pt sans-serif`
        },
        ...elements
      ]
    }
  };

  return JSON.stringify(caption);
}
```

---

## 7. Batch Generation Pattern

Generate multiple slides by calling the endpoint sequentially. The model stays loaded between calls (~12s per image after first load).

```python
import urllib.request, urllib.parse, json, time

BASE = "http://192.168.0.87:8002"
FRAMES = [
    {"headline": "Solar Energy Basics", "points": ["..."], "ts": "00:42"},
    {"headline": "How Panels Work",      "points": ["..."], "ts": "01:15"},
    # ...
]

for frame in FRAMES:
    prompt = build_ideogram4_prompt(frame)
    payload = {
        "prompt": prompt,
        "width": "1280", "height": "720",
        "preset": "V4_QUALITY_48",
        "guidance_scale": "10.0",
        "seed": "-1",
        "quant": "nf4"
    }
    data = urllib.parse.urlencode(payload, doseq=True).encode()
    req = urllib.request.Request(f"{BASE}/generate/ideogram4", data=data, method="POST")
    resp = urllib.request.urlopen(req, timeout=600)
    result = json.loads(resp.read())
    
    # Download and save
    url = result["results"][0]["url"]
    img_data = urllib.request.urlopen(f"{BASE}{url}").read()
    with open(f"frame_{frame['ts']}.png", "wb") as f:
        f.write(img_data)
    
    print(f"Frame {frame['ts']} done")
    time.sleep(2)  # brief cooldown
```

---

## 8. VRAM / Performance

| Metric | Value |
|--------|-------|
| GPU | RTX 5060 Ti 16 GB |
| Model VRAM (NF4) | ~11.9 GB loaded |
| Per-image time (QUALITY_48) | ~60s |
| Per-image time (TURBO_12) | ~12s |
| Max resolution tested | 1024×1280 |
| Concurrent requests | 1 only (single GPU) |
| Model load time (first request) | ~6-8 minutes |

**Important:** The first request triggers model loading (~6-8 min). Subsequent requests reuse the loaded model. Poll `/status` to check `loaded: true`.

---

## 9. Common Issues

| Error | Cause | Fix |
|-------|-------|-----|
| `503 Service Unavailable` | Model loading or another generation in progress | Wait 10s, retry |
| `CUDA out of memory` | Resolution too high for available VRAM | Lower to 1280×720, use nf4 quant |
| `caption verifier flagged` | JSON format doesn't match schema | Use `"text"` key in text elements, `"photo"` or `"art_style"` in style |
| Garbled/missing text | `guidance_scale` too low | Use ≥7.0, recommend 10.0 |
| `raise_on_caption_issues` | Warnings converted to errors | Engine uses `raise_on_caption_issues=False` by default |

---

## 10. Style Checklist (from your GPT prompt)

For each frame, verify:

- [ ] 16:9 horizontal (`width=1280, height=720`)
- [ ] Clean white/light background
- [ ] Black text as primary, orange (`#FF6B35`) for accents only
- [ ] Minimal uncluttered layout
- [ ] Editorial explainer style, not cinematic 3D
- [ ] Paper texture feel, soft shadows
- [ ] Headline is bold and readable
- [ ] Body text is short and clear
- [ ] Simple supporting graphics (arrows, icons, boxes) when helpful
- [ ] No logos or branding
- [ ] No dark backgrounds, flashy visuals, or cluttered layouts
- [ ] Each frame focuses on ONE clear idea only
- [ ] Layout style consistent across frames, content changes per timestamp

---

## 11. Full Working Example

```bash
#!/bin/bash
# Generate a 3-frame explainer series

BASE="http://192.168.0.87:8002"
TIMESTAMPS=("00:42" "01:15" "02:30")
HEADLINES=("SOLAR ENERGY BASICS" "HOW PANELS WORK" "THE FUTURE OF SOLAR")

for i in "${!TIMESTAMPS[@]}"; do
  TS="${TIMESTAMPS[$i]}"
  HL="${HEADLINES[$i]}"
  
  PROMPT=$(cat <<EOF
{
  "high_level_description": "a clean 16:9 explainer slide about: $HL",
  "style_description": {
    "art_style": "editorial explainer graphic, paper texture, clean white, black text, orange accents, minimal, educational, 16:9",
    "color_palette": ["#FFFFFF", "#1A1A1A", "#FF6B35", "#F5F0E8"],
    "medium": "educational slide"
  },
  "compositional_deconstruction": {
    "background": "white paper with subtle grid, soft shadow",
    "elements": [
      {"type":"text","text":"$HL","desc":"bold headline top center with orange underline"},
      {"type":"text","text":"Key concept for timestamp $TS explained clearly with supporting graphics","desc":"body text in clean black sans-serif, short and readable"}
    ]
  }
}
EOF
)

  curl -s -X POST "$BASE/generate/ideogram4" \
    -F "prompt=$PROMPT" \
    -F 'width=1280' -F 'height=720' \
    -F 'preset=V4_QUALITY_48' \
    -F 'guidance_scale=10.0' \
    -F 'seed=-1' -F 'quant=nf4' \
    -o "frame_${TS}.json"
  
  URL=$(python3 -c "import json; print(json.load(open('frame_${TS}.json'))['results'][0]['url'])")
  curl -s "$BASE$URL" -o "frame_${TS}.png"
  echo "Generated: frame_${TS}.png"
done
```
