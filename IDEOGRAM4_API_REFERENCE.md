# Ideogram 4 HTTP API — Developer Reference

> For content generation tools calling the Arthur Image Lab'"'"'s Ideogram 4 engine via HTTP/HTTPS.
> **No authentication required** (internal service, same network).
> **Base URL:** `http://192.168.0.87:8002`

---

## 1. Quick Start

### Direct JSON caption (fastest — no API calls)

```bash
curl -X POST http://192.168.0.87:8002/generate/ideogram4 \
  -F '"'"'prompt={"high_level_description":"a clean explainer slide about solar energy","style_description":{"art_style":"editorial explainer, paper texture, clean typography, black text with orange accents","medium":"educational slide","color_palette":["#FFFFFF","#1A1A1A","#FF6B35"]},"compositional_deconstruction":{"background":"white paper texture with subtle grid","elements":[{"type":"text","text":"HOW SOLAR PANELS WORK","desc":"large bold headline at top"},{"type":"text","text":"1. Sunlight hits panel    2. Electrons flow   3. Inverter converts to AC","desc":"three step process in clean list with orange arrows"}]}}'"'"' \
  -F '"'"'width=1280'"'"' \
  -F '"'"'height=720'"'"' \
  -F '"'"'preset=V4_QUALITY_48'"'"' \
  -F '"'"'guidance_scale=10.0'"'"' \
  -F '"'"'quant=nf4'"'"' \
  -o output.png
```

### Magic Prompt (plain text → auto-expand to JSON)

```bash
curl -X POST http://192.168.0.87:8002/generate/ideogram4 \
  -F '"'"'prompt=a vintage newspaper front page with headline BREAKING NEWS, two columns of article text, old paper texture, black ink on aged yellowish paper, dated June 9 2026'"'"' \
  -F '"'"'use_magic_prompt=true'"'"' \
  -F '"'"'magic_prompt_aspect_ratio=3:4'"'"' \
  -F '"'"'width=768'"'"' \
  -F '"'"'height=1024'"'"' \
  -F '"'"'preset=V4_QUALITY_48'"'"' \
  -F '"'"'guidance_scale=10.0'"'"' \
  -F '"'"'quant=nf4'"'"' \
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

---

## 3. Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | **required** | JSON caption or plain text (see §4) |
| `width` | int | 1024 | Output width (16–2048, multiples of 16) |
| `height` | int | 1024 | Output height (16–2048, multiples of 16) |
| `preset` | string | `V4_DEFAULT_20` | Sampler preset (see §6) |
| `num_inference_steps` | int | 0 | Override steps (0 = use preset default) |
| `guidance_scale` | float | 7.0 | CFG strength (1.0–30.0). ≥7.0 for text |
| `seed` | int | -1 | Random seed (-1 = random) |
| `quant` | string | `nf4` | Quantization: `nf4` or `fp8` |
| `mu` | float | 0.0 | Schedule mean (advanced) |
| `std` | float | 1.75 | Schedule std (advanced) |
| `use_magic_prompt` | bool | false | Expand prompt via LLM (see §5) |
| `magic_prompt_aspect_ratio` | string | `1:1` | Target ratio (`16:9`, `3:4`, `1:1`) |

---

## 4. Two Prompt Modes

### Mode A: Direct JSON (`use_magic_prompt=false`, default)

You write the JSON caption yourself. Fastest — zero API overhead.

### Mode B: Magic Prompt (`use_magic_prompt=true`)

You write plain text. The engine auto-expands it via an LLM API.

**Expansion priority chain:**

| Pri | Provider | Env Var | Prompt | Cost | Quality |
|:---:|----------|---------|--------|:----:|:-------:|
| 1 | Ideogram hosted | `IDEOGRAM_API_KEY` | Server-side | **Free** | Best |
| 2 | DeepSeek native | `DEEPSEEK_API_KEY` | v1.txt (28KB) | ~$0.14/M | Very good |
| 3 | OpenRouter→DeepSeek | `OPENROUTER_API_KEY` | v1.txt (28KB) | ~$0.14/M | Very good |

First configured key wins. None configured → plain text passed as-is (poor results).

### v1.txt — Ideogram'"'"'s Official Recipe

~6,900 tokens, 19 formatting rules: single-line minified JSON, 50-word HLD cap, SINGLE SUBJECT = SINGLE ELEMENT, ground ALWAYS in background, dual-mention for shell-affixed objects, non-ASCII preservation, no shadows in element descs, etc.

---

## 5. JSON Caption Schema

Two variants supported:

### Variant A: With `style_description`

```json
{
  "high_level_description": "concise scene summary",
  "style_description": {
    "art_style": "illustration style",
    "color_palette": ["#HEX1", "#HEX2", "#HEX3"],
    "medium": "digital art | photograph | educational slide"
  },
  "compositional_deconstruction": {
    "background": "detailed background",
    "elements": [
      {"type": "text", "text": "RENDERED TEXT", "desc": "visual desc"},
      {"type": "obj",  "desc": "non-text object desc"}
    ]
  }
}
```

### Variant B: Native v1.txt format (magic prompt output)

```json
{
  "aspect_ratio": "16:9",
  "high_level_description": "concise summary",
  "compositional_deconstruction": {
    "background": "detailed background",
    "elements": [
      {"type":"text","bbox":[y1,x1,y2,x2],"text":"LINE ONE\nLINE TWO","desc":"desc"},
      {"type":"obj","bbox":[y1,x1,y2,x2],"desc":"desc"}
    ]
  }
}
```

### Key Rules

| Rule | Detail |
|------|--------|
| `type: "text"` | Must have BOTH `"text"` (literal chars) AND `"desc"` (visual) |
| `type: "obj"` | Only need `"type"` and `"desc"` |
| Multi-line | Use `\n`: `"LINE ONE\nLINE TWO"` |
| HLD cap | 50 words, reads like a search query |
| Color palette | 3–5 hex codes |
| `bbox` | Optional `[y1,x1,y2,x2]` positioning |

---

## 6. Presets

| Preset | Steps | ~Time | Use Case |
|--------|:-----:|:-----:|----------|
| `V4_TURBO_12` | 12 | ~15s | Fast preview |
| `V4_DEFAULT_20` | 20 | ~25s | Good balance |
| `V4_QUALITY_48` | 48 | ~60s | Best quality, text |

### Recommended Configs

| Use Case | W×H | Preset | Guidance |
|----------|:-----:|--------|:--------:|
| Explainer (16:9) | 1280×720 | QUALITY_48 | 10.0 |
| Newspaper (portrait) | 768×1024 | QUALITY_48 | 10.0 |
| Photo (1:1) | 1024×1024 | DEFAULT_20 | 7.0 |
| Quick preview | 512×512 | TURBO_12 | 7.0 |

---

## 7. Converting GPT Prompts to JSON

### Style → `style_description`

From GPT: *"clean editorial explainer, paper texture, black text, orange accents..."*

```json
{
  "high_level_description": "a clean 16:9 explainer slide, white paper, black text, orange accents",
  "style_description": {
    "art_style": "editorial explainer graphic, paper texture, clean white, black typography, orange accents, minimal design, 16:9, soft shadows",
    "color_palette": ["#FFFFFF", "#1A1A1A", "#FF6B35", "#F5F0E8"],
    "medium": "educational slide"
  }
}
```

### Content → `elements`

Each key point = one element. Text elements: `"text"` (what to render) + `"desc"` (how it looks).

### Example: Photosynthesis Slide

```json
{
  "high_level_description": "clean 16:9 explainer slide about photosynthesis, white paper, black text, orange accents",
  "style_description": {
    "art_style": "editorial explainer, paper texture, black typography, orange accents, minimal, educational",
    "color_palette": ["#FFFFFF", "#1A1A1A", "#FF6B35", "#F5F0E8"],
    "medium": "educational slide"
  },
  "compositional_deconstruction": {
    "background": "clean white paper texture with subtle grid lines, soft edge shadow",
    "elements": [
      {"type":"text","text":"HOW PHOTOSYNTHESIS WORKS","desc":"large bold headline at top, black, 48pt sans-serif, thin orange underline"},
      {"type":"text","text":"Step 1: Sunlight absorbed by chlorophyll","desc":"left column, medium black text, small orange circle with number 1"},
      {"type":"obj","desc":"simple orange arrow pointing right"},
      {"type":"text","text":"Step 2: Water molecules split, releasing oxygen","desc":"center column, medium black text, small orange circle with number 2"},
      {"type":"obj","desc":"simple orange arrow pointing right"},
      {"type":"text","text":"Step 3: CO2 converted into glucose","desc":"right column, medium black text, small orange circle with number 3"},
      {"type":"text","text":"CO2 + H2O + Light → C6H12O6 + O2","desc":"chemical equation in gray rounded box at bottom, monospace, orange equals sign"}
    ]
  }
}
```

### Example: Newspaper Front Page

```json
{
  "high_level_description": "vintage newspaper front page, BREAKING NEWS headline, two columns, old paper, black ink on aged yellowish paper",
  "style_description": {
    "art_style": "vintage newspaper, aged yellowish paper texture, black ink, serif typography, 19th century aesthetic, column layout, weathered newsprint",
    "color_palette": ["#F5E6C8", "#1A1A1A", "#8B7355", "#D4C5A0"],
    "medium": "newspaper print"
  },
  "compositional_deconstruction": {
    "background": "aged yellowish paper with subtle fiber texture, soft vignette at edges from old paper wear",
    "elements": [
      {"type":"text","text":"THE DAILY CHRONICLE","desc":"small nameplate at top, elegant serif, centered, black ink, decorative thin line below"},
      {"type":"text","text":"June 9, 2026  |  Vol. CXLVII  |  Price: Two Cents","desc":"date line, small serif, thin rules above and below"},
      {"type":"text","text":"BREAKING NEWS","desc":"massive bold headline spanning full width, heavy serif, main focal point, letterpressed texture"},
      {"type":"text","text":"Scientists Announce Revolutionary Discovery","desc":"subheadline, slightly smaller bold serif, single line"},
      {"type":"text","text":"WASHINGTON — In a stunning announcement, researchers revealed findings that challenge long-held assumptions about particle behavior. The discovery suggests quantum entanglement may operate across temporal boundaries. The team spent seven years on experiments.","desc":"left column body text, small serif, justified, ~100 words of dense newsprint"},
      {"type":"obj","desc":"scientific diagram: simplified atom model with electron orbits, woodcut-style line engraving, centered between columns, ~200px square"},
      {"type":"text","text":"The implications are profound. If quantum states maintain coherence across temporal displacement, it alters our understanding of causality. Several physicists called it the most significant development since Bell'"'"'s Theorem.","desc":"right column body text, small serif, justified, ~50 words"},
      {"type":"text","text":"Continued on Page A12","desc":"small continuation notice, bottom right, italic serif"}
    ]
  }
}
```

---

## 8. Prompt Template (TypeScript)

```typescript
interface Frame {
  headline: string;
  bodyPoints: string[];
  accentColor: string;
}

function buildCaption(f: Frame): string {
  const elements = f.bodyPoints.map((p, i) => ({
    type: "text" as const,
    text: p,
    desc: `key point ${i+1}, clean black sans-serif, small orange circle indicator`
  }));

  return JSON.stringify({
    high_level_description: `clean 16:9 explainer slide: ${f.headline}`,
    style_description: {
      art_style: "editorial explainer, paper texture, clean white, black typography, orange accents, minimal, 16:9, soft shadows",
      color_palette: ["#FFFFFF", "#1A1A1A", f.accentColor, "#F5F0E8"],
      medium: "educational slide"
    },
    compositional_deconstruction: {
      background: "white paper texture with subtle grid, soft edge shadow",
      elements: [
        { type: "text", text: f.headline.toUpperCase(), desc: "bold black headline, 48pt sans-serif, thin orange underline" },
        ...elements
      ]
    }
  });
}
```

---

## 9. Batch Generation (Python)

```python
import urllib.request, urllib.parse, json

BASE = "http://192.168.0.87:8002"

# Direct JSON mode
for frame in frames:
    prompt = build_caption(frame)
    data = urllib.parse.urlencode({
        "prompt": prompt, "width": "1280", "height": "720",
        "preset": "V4_QUALITY_48", "guidance_scale": "10.0",
        "seed": "-1", "quant": "nf4"
    }, doseq=True).encode()
    req = urllib.request.Request(f"{BASE}/generate/ideogram4", data=data, method="POST")
    resp = urllib.request.urlopen(req, timeout=600)
    r = json.loads(resp.read())
    url = r["results"][0]["url"]
    img = urllib.request.urlopen(f"{BASE}{url}").read()
    with open(f"frame_{frame['ts']}.png", "wb") as f:
        f.write(img)

# Magic prompt mode (plain text)
data = urllib.parse.urlencode({
    "prompt": "a clean explainer slide about solar energy, white paper, black text, orange accents",
    "use_magic_prompt": "true", "magic_prompt_aspect_ratio": "16:9",
    "width": "1280", "height": "720",
    "preset": "V4_QUALITY_48", "guidance_scale": "10.0",
    "seed": "-1", "quant": "nf4"
}, doseq=True).encode()
```

---

## 10. VRAM / Performance

| Metric | Value |
|--------|-------|
| GPU | RTX 5060 Ti 16 GB |
| Idle VRAM | ~10.6 GB |
| Peak VRAM | ~11.9 GB |
| TURBO_12 | ~12–15s |
| DEFAULT_20 | ~22–28s |
| QUALITY_48 | ~55–65s |
| Magic prompt overhead | +3–8s |
| Max resolution | 1024×1280 |
| First-request load | ~6–8 min |
| Concurrency | 1 (single GPU) |

---

## 11. Common Issues

| Error | Cause | Fix |
|-------|-------|-----|
| `503` | Model loading / gen in progress | Wait 10s |
| `CUDA OOM` | Resolution too high | 1280×720, nf4 |
| Bad JSON | Wrong format | Include `"text"` key |
| Garbled text | Low guidance | ≥7.0, recommend 10.0 |
| Magic prompt fails | No API key | Set `DEEPSEEK_API_KEY` or `IDEOGRAM_API_KEY` |
| `Unknown quantization` | Empty quant | Always pass `quant=nf4` |

---

## 12. Style Checklist

- [ ] Correct aspect ratio (16:9, 3:4, etc.)
- [ ] Clean white/light background
- [ ] Black text primary, accent color sparingly
- [ ] Minimal uncluttered layout
- [ ] Paper texture feel, soft shadows
- [ ] Headline bold and readable
- [ ] Body text short and clear
- [ ] Simple graphics (arrows, icons, boxes)
- [ ] No logos, branding, dark backgrounds
- [ ] One clear idea per frame
- [ ] 30–60 words per element description
