#!/usr/bin/env python3
"""
Persian TTS Benchmark Suite — Standardized evaluation sentences for Persian engines.

Usage:
    # Generate benchmark audio from all Persian engines:
    python scripts/benchmark/persian_bench.py --output-dir /tmp/persian_bench

    # Test a single engine:
    python scripts/benchmark/persian_bench.py --engine chatterbox --output-dir /tmp/persian_bench

    # Compare two engines side-by-side:
    python scripts/benchmark/persian_bench.py --engines chatterbox,mmsfas --output-dir /tmp/persian_bench

    # List all benchmark sentences:
    python scripts/benchmark/persian_bench.py --list

Integration:
    Run this BEFORE exposing new Persian engines to users. Any engine scoring
    < 2 on pronunciation or naturalness should be marked experimental/hidden.

    Phase 0 of: docs/reference/PERSIAN_TTS_INTEGRATION_PLAN.md
"""

from __future__ import annotations
import argparse, json, sys, time, wave, io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════
# Benchmark Sentences
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkSentence:
    id: str
    text: str
    category: str  # basic, numbers, punct, chars, mixed, informal, long

PERSIAN_BENCHMARK: list[BenchmarkSentence] = [
    # ── Basic pronunciation ──
    BenchmarkSentence("basic_hello",       "سلام، حال شما چطور است؟",                    "basic"),
    BenchmarkSentence("basic_weather",     "امروز هوا در سیاتل بسیار خوب است.",            "basic"),
    BenchmarkSentence("basic_quality",     "این یک تست کیفیت برای سامانه تبدیل متن به گفتار است.", "basic"),

    # ── Numbers — Persian digits (۰۱۲۳۴۵۶۷۸۹) are critical ──
    BenchmarkSentence("numbers_phone",     "شماره تماس من ۱۲۳۴۵۶۷۸۹۰ است.",               "numbers"),
    BenchmarkSentence("numbers_price",     "قیمت بیت کوین امروز چقدر است؟",                "numbers"),
    BenchmarkSentence("numbers_date",      "تاریخ امروز بیست و سوم ژوئن سال دو هزار و بیست و شش است.", "numbers"),

    # ── Punctuation & structure ──
    BenchmarkSentence("punct_question",    "آیا واقعاً فکر میکنی که این راهحل درستی است؟",    "punct"),
    BenchmarkSentence("punct_exclamation", "چه روز قشنگی! واقعاً که هوا عالی است.",           "punct"),
    BenchmarkSentence("punct_quotes",      "او گفت: «من فردا به تهران میروم.»",              "punct"),

    # ── Persian-specific characters ──
    # These test critical phoneme distinctions: آ (alef+maddah), ؤ (waw+hamza), ی/ي
    BenchmarkSentence("chars_alef",        "آب آمد و آن آسیاب را باد برد.",                 "chars"),
    BenchmarkSentence("chars_hamza",       "مؤمنان مؤثر در جامعه مؤاخذه نمیشوند.",             "chars"),
    BenchmarkSentence("chars_yeh",         "خانهای بزرگ در کنار رودخانه.",                   "chars"),

    # ── Mixed Persian/Arabic vocabulary ──
    BenchmarkSentence("mixed_arabic",      "قرآن کریم کتاب مقدس مسلمانان است.",               "mixed"),
    BenchmarkSentence("mixed_formal",      "اداره کل امور مالیاتی اعلام کرد که مهلت تسلیم اظهارنامه تمدید شد.", "mixed"),

    # ── Informal / colloquial ──
    BenchmarkSentence("informal_chat",     "سلام رفیق، چطوری؟ دلم برات تنگ شده بود.",         "informal"),
    BenchmarkSentence("informal_slang",    "بابا این چه وضعشه؟ آخه کی این کارو کرده؟",         "informal"),

    # ── Long-form (tests chunking / max-length handling) ──
    BenchmarkSentence("long_paragraph",
        "امروز میخواهم درباره یکی از مهمترین پیشرفتهای علمی قرن اخیر صحبت کنم. "
        "هوش مصنوعی در سالهای اخیر تحول عظیمی در صنعت و فناوری ایجاد کرده است. "
        "از خودروهای خودران گرفته تا دستیارهای پزشکی، هوش مصنوعی همه جا حضور دارد. "
        "اما چالشهای اخلاقی زیادی هم پیش روی ما قرار دارد که باید به آنها فکر کنیم.",
        "long"),
]

# Arthur-specific sentences (scam-baiting decoy use case)
ARTHUR_PERSIAN: list[BenchmarkSentence] = [
    BenchmarkSentence("arthur_greeting",   "الو؟ ببخشید، کی هستید؟ من تقریباً صدای تلفن را نشنیدم.",   "arthur"),
    BenchmarkSentence("arthur_confused",   "حالا این کاغذ را کجا گذاشتم... اوه، ببخشید عزیزم، باز گیج شدم.", "arthur"),
    BenchmarkSentence("arthur_numbers",    "میشه اون شماره رو یک بار دیگه برام بگید؟ آروم آروم لطفاً.",      "arthur"),
    BenchmarkSentence("arthur_stall",      "وایسید ببینم... عینکم کجاست؟ آهان پیداش کردم. حالا بفرمایید.",   "arthur"),
    BenchmarkSentence("arthur_long",
        "من واقعاً متأسفم که نمیتونم الان کمکتون کنم. پسرم که معمولاً این کارا رو برام انجام میده "
        "الان خونه نیست. شاید بتونید یه کم دیرتر زنگ بزنین؟ البته من نمیدونم کی برمیگرده. "
        "راستی، شما از کجا تماس میگیرین؟",
        "arthur"),
]


# ═══════════════════════════════════════════════════════════════════════
# Synthesis Client
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SynthResult:
    engine: str
    sentence_id: str
    text: str
    wav_bytes: Optional[bytes] = None
    sample_rate: int = 0
    duration_s: float = 0.0
    synth_time_s: float = 0.0
    rtf: float = 0.0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.wav_bytes is not None


class BenchmarkRunner:
    """Synthesize all benchmark sentences against one or more engine servers."""

    def __init__(self, base_url: str = "http://192.168.0.87:8001", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def synth(self, engine: str, text: str, params: dict | None = None) -> tuple[bytes, float]:
        """Synthesize text and return (wav_bytes, elapsed_seconds)."""
        import requests
        url = f"{self.base_url}/synthesize/{engine}"
        payload = {"text": text, "params": params or {}}
        t0 = time.time()
        resp = requests.post(url, json=payload, timeout=self.timeout)
        elapsed = time.time() - t0
        resp.raise_for_status()
        return resp.content, elapsed

    def wav_duration(self, wav_bytes: bytes) -> float:
        """Return duration in seconds from WAV bytes."""
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            return wf.getnframes() / wf.getframerate()

    def run_engine(self, engine: str, sentences: list[BenchmarkSentence],
                   params: dict | None = None) -> list[SynthResult]:
        """Run all benchmark sentences against one engine."""
        results = []
        for s in sentences:
            try:
                wav, elapsed = self.synth(engine, s.text, params)
                dur = self.wav_duration(wav)
                rtf = elapsed / dur if dur > 0 else float("inf")
                results.append(SynthResult(
                    engine=engine, sentence_id=s.id, text=s.text,
                    wav_bytes=wav, sample_rate=0,  # parsed from WAV if needed
                    duration_s=dur, synth_time_s=elapsed, rtf=rtf,
                ))
            except Exception as e:
                results.append(SynthResult(
                    engine=engine, sentence_id=s.id, text=s.text,
                    error=str(e),
                ))
        return results

    def save(self, results: list[SynthResult], output_dir: Path):
        """Save WAV files and a JSON summary to output_dir."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Group by engine
        by_engine: dict[str, list[SynthResult]] = {}
        for r in results:
            by_engine.setdefault(r.engine, []).append(r)

        summary = {}
        for engine, engine_results in by_engine.items():
            engine_dir = output_dir / engine
            engine_dir.mkdir(exist_ok=True)

            engine_summary = {}
            ok_count = 0
            total_rtf = 0.0

            for r in engine_results:
                if r.ok:
                    path = engine_dir / f"{r.sentence_id}.wav"
                    path.write_bytes(r.wav_bytes)  # type: ignore[arg-type]
                    engine_summary[r.sentence_id] = {
                        "duration_s": round(r.duration_s, 2),
                        "synth_time_s": round(r.synth_time_s, 2),
                        "rtf": round(r.rtf, 2),
                    }
                    ok_count += 1
                    total_rtf += r.rtf
                else:
                    engine_summary[r.sentence_id] = {"error": r.error}

            avg_rtf = total_rtf / ok_count if ok_count > 0 else 0
            summary[engine] = {
                "ok": ok_count,
                "total": len(engine_results),
                "avg_rtf": round(avg_rtf, 2),
                "results": engine_summary,
            }

        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"\nSaved {len(results)} WAVs to {output_dir}/")
        print(f"Summary: {summary_path}")
        for engine, s in summary.items():
            status = "✅" if s["ok"] == s["total"] else f"⚠️ {s['ok']}/{s['total']}"
            print(f"  {engine:20s} {status}  avg RTF: {s['avg_rtf']:.2f}×")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Persian TTS Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --output-dir /tmp/persian_bench
  %(prog)s --engine chatterbox --output-dir /tmp/chat_test
  %(prog)s --engines chatterbox,mmsfas,kamtera_f --output-dir /tmp/compare
  %(prog)s --list
        """,
    )
    parser.add_argument("--output-dir", default="/tmp/persian_bench",
                        help="Directory to save WAV files and summary.json")
    parser.add_argument("--engine", default=None,
                        help="Single engine to test")
    parser.add_argument("--engines", default=None,
                        help="Comma-separated list of engines to test")
    parser.add_argument("--url", default="http://192.168.0.87:8001",
                        help="Base URL of TTS Lab server")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Synthesis timeout in seconds")
    parser.add_argument("--arthur", action="store_true",
                        help="Also run Arthur-specific sentences")
    parser.add_argument("--list", action="store_true",
                        help="List all benchmark sentences and exit")
    args = parser.parse_args()

    if args.list:
        print("=== Standard Persian Benchmark ===")
        for s in PERSIAN_BENCHMARK:
            print(f"  [{s.category}] {s.id}: {s.text[:80]}{'...' if len(s.text) > 80 else ''}")
        print("\n=== Arthur-Specific ===")
        for s in ARTHUR_PERSIAN:
            print(f"  [{s.category}] {s.id}: {s.text[:80]}{'...' if len(s.text) > 80 else ''}")
        return

    # Determine engine list
    if args.engine:
        engines = [args.engine]
    elif args.engines:
        engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    else:
        # All known Persian-capable engines
        engines = [
            # Existing in lab
            "chatterbox", "matcha", "manatts", "piper", "fishspeech", "f5tts",
            # Phase 1 (after deployment)
            # "mmsfas",
            # Phase 2 (after deployment)
            # "kamtera_f", "kamtera_m", "gptinf_fa", "zabanzad_f", "zabanzad_m",
            # Phase 4 (after verification)
            # "speecht5_fa",
        ]

    sentences = list(PERSIAN_BENCHMARK)
    if args.arthur:
        sentences.extend(ARTHUR_PERSIAN)

    runner = BenchmarkRunner(base_url=args.url, timeout=args.timeout)
    all_results = []

    for engine in engines:
        print(f"\n{'='*60}")
        print(f"Testing: {engine} ({len(sentences)} sentences)")
        print(f"{'='*60}")
        results = runner.run_engine(engine, sentences)
        ok = sum(1 for r in results if r.ok)
        for r in results:
            if r.ok:
                print(f"  ✅ {r.sentence_id:25s}  {r.duration_s:5.1f}s  RTF {r.rtf:.2f}×")
            else:
                print(f"  ❌ {r.sentence_id:25s}  {r.error}")
        print(f"  ── {ok}/{len(results)} OK ──")
        all_results.extend(results)

    runner.save(all_results, Path(args.output_dir))


if __name__ == "__main__":
    main()
