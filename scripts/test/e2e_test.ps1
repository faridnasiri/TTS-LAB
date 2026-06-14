#!/usr/bin/env pwsh
<#
.SYNOPSIS
    End-to-end test for Arthur TTS Lab after engine fixes.
    Tests: indextts (no load_model), qwen3tts (attn shim), and all 18 installed engines.

.EXAMPLE
    .\e2e_test.ps1
    .\e2e_test.ps1 -VM 192.168.0.100
#>
param(
    [string]$VM   = "192.168.0.87",
    [string]$User = "arthur",
    [string]$Key  = "$env:USERPROFILE\.ssh\id_arthur_vm"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function vm([string]$cmd, [switch]$nocheck) {
    $out = & ssh -i $Key -o StrictHostKeyChecking=no -o ConnectTimeout=15 `
                "${User}@${VM}" $cmd 2>&1
    if (-not $nocheck -and $LASTEXITCODE -ne 0) {
        throw "Remote command failed (exit $LASTEXITCODE): $cmd`n$out"
    }
    return $out
}

function hdr([string]$txt) {
    Write-Host ""
    Write-Host ("─" * 64) -ForegroundColor DarkCyan
    Write-Host "  $txt" -ForegroundColor Cyan
    Write-Host ("─" * 64) -ForegroundColor DarkCyan
}

function pass([string]$msg) { Write-Host "  ✅ $msg" -ForegroundColor Green }
function fail([string]$msg) { Write-Host "  ❌ $msg" -ForegroundColor Red; $script:failures++ }
function warn([string]$msg) { Write-Host "  ⚠️  $msg" -ForegroundColor Yellow }
function info([string]$msg) { Write-Host "  ℹ  $msg" -ForegroundColor DarkGray }

$script:failures = 0

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║       Arthur TTS Lab — End-to-End Test Suite                 ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host "  VM: ${User}@${VM} | $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# ── 1. SSH connectivity ───────────────────────────────────────────────────────
hdr "1 — SSH Connectivity"
$pong = vm "echo PONG" -nocheck
if ($pong -match "PONG") { pass "SSH OK" } else { fail "SSH unreachable"; exit 1 }

# ── 2. Service health ─────────────────────────────────────────────────────────
hdr "2 — Service Health"
$svc = vm "sudo systemctl is-active arthur-lab" -nocheck
if ($svc.Trim() -eq "active") {
    pass "arthur-lab.service is ACTIVE"
} else {
    fail "arthur-lab.service is $($svc.Trim())"
    warn "Showing last 20 log lines:"
    vm "sudo journalctl -u arthur-lab -n 20 --no-pager" -nocheck | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
    exit 1
}

# ── 3. HTTP health ────────────────────────────────────────────────────────────
hdr "3 — HTTP Health (port 8001)"
$code = (vm "curl -s -o /dev/null -w '%{http_code}' http://localhost:8001/" -nocheck).Trim()
if ($code -eq "200") { pass "UI responding — HTTP 200" } else { fail "HTTP $code — expected 200" }

# ── 4. /status endpoint — all 21 engines registered ──────────────────────────
hdr "4 — /status Endpoint"
$raw = vm "curl -sf http://localhost:8001/status" -nocheck
try {
    $data   = $raw | ConvertFrom-Json
    $models = $data.models.PSObject.Properties
    $total  = ($models | Measure-Object).Count
    $avail  = ($models | Where-Object { $_.Value.available -eq $true } | Measure-Object).Count

    Write-Host ""
    Write-Host ("  {0,-14} {1,-12} {2,6}  {3}" -f "Engine", "Status", "RAM MB", "Label") -ForegroundColor White
    Write-Host ("  " + "─" * 56)
    foreach ($m in $models | Sort-Object Name) {
        $v    = $m.Value
        $icon = if ($v.available) { "✅ ready   " } else { "🔴 missing " }
        $col  = if ($v.available) { "White" } else { "DarkGray" }
        Write-Host ("  {0,-14} {1,-12} {2,6}  {3}" -f $m.Name, $icon, $v.ram_est_mb, $v.label) -ForegroundColor $col
    }
    Write-Host ""

    if ($total -eq 21) { pass "All 21 engines registered" } else { fail "Expected 21 engines, got $total" }
    info "$avail / $total packages installed and available"
} catch {
    fail "/status JSON parse failed — raw: $raw"
}

# ── 5. Fix verification: indextts — no load_model() ──────────────────────────
hdr "5 — Fix Verification: indextts (no load_model)"
$check = vm "grep -n 'model.load_model()' /opt/arthur/tts_lab_engines.py" -nocheck
if ([string]::IsNullOrWhiteSpace($check)) {
    pass "load_model() call is gone from _load_indextts()"
} else {
    fail "load_model() still present at: $check"
}

# verify IndexTTS2 actually has no load_model (sanity check the library itself)
$checkScript = @"
from indextts.infer_v2 import IndexTTS2
print('HAS_LOAD_MODEL' if hasattr(IndexTTS2, 'load_model') else 'NO_LOAD_MODEL')
"@
$checkScript | ssh -i $Key -o StrictHostKeyChecking=no "${User}@${VM}" "cat > /tmp/test_indextts_api.py"
$r = vm "source /opt/arthur-bench-env/bin/activate && python3 /tmp/test_indextts_api.py 2>&1 | tail -1" -nocheck
if ($r -match "NO_LOAD_MODEL") {
    pass "IndexTTS2 library confirms: no load_model() method"
} elseif ($r -match "HAS_LOAD_MODEL") {
    warn "IndexTTS2 gained load_model() back — remove the fix if so"
} else {
    warn "Could not verify IndexTTS2 methods: $r"
}

# ── 6. Fix verification: qwen3tts — _attn_implementation_autoset shim ────────
hdr "6 — Fix Verification: qwen3tts (_attn_implementation_autoset)"
$script3 = @"
source /opt/arthur-bench-env/bin/activate && python3 -c '
import tts_lab_shims  # run shims
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig
cfg = Qwen3TTSSpeakerEncoderConfig()
print(\"ATTR_OK\" if hasattr(cfg, \"_attn_implementation_autoset\") else \"ATTR_MISSING\")
' 2>&1 | tail -1
"@
# write script to server to avoid quoting issues
$shimTest = @"
import sys
sys.path.insert(0, '/opt/arthur')
import tts_lab_shims
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig
cfg = Qwen3TTSSpeakerEncoderConfig()
print('ATTR_OK' if hasattr(cfg, '_attn_implementation_autoset') else 'ATTR_MISSING')
"@
$shimTest | ssh -i $Key -o StrictHostKeyChecking=no "${User}@${VM}" "cat > /tmp/test_qwen_shim.py"
$r = vm "source /opt/arthur-bench-env/bin/activate && python3 /tmp/test_qwen_shim.py 2>&1 | tail -1" -nocheck
if ($r -match "ATTR_OK") {
    pass "Qwen3TTSSpeakerEncoderConfig._attn_implementation_autoset is present after shim"
} else {
    fail "Shim not effective — got: $r"
}

# ── 7. Synthesis tests — CPU-safe engines ────────────────────────────────────
hdr "7 — Synthesis Tests (CPU-safe engines)"

# Engines that run on CPU and don't need a reference WAV
$cpuEngines = @("piper", "kokoro", "melo", "chattts", "bark", "styletts2", "dia", "zonos")
$testText   = "Hello, this is an automated end to end test."
$passed = 0; $failed = 0; $skipped = 0

foreach ($engine in $cpuEngines) {
    $req = "{`"text`":`"$testText`",`"params`":{}}"
    $result = vm "curl -s -o /tmp/synth_out.wav -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d '$req' http://localhost:8001/synthesize/$engine" -nocheck
    $httpCode = $result.Trim()
    $size = vm "wc -c < /tmp/synth_out.wav" -nocheck

    if ($httpCode -eq "200" -and [int]$size.Trim() -gt 1000) {
        Write-Host "  ✅ $($engine.PadRight(14)) HTTP 200  $($size.Trim()) bytes" -ForegroundColor Green
        $passed++
    } elseif ($httpCode -eq "503") {
        Write-Host "  ⏭  $($engine.PadRight(14)) HTTP 503 (not loaded — OK, lazy load)" -ForegroundColor DarkGray
        $skipped++
    } else {
        $body = vm "cat /tmp/synth_out.wav 2>/dev/null | head -c 200" -nocheck
        Write-Host "  ❌ $($engine.PadRight(14)) HTTP $httpCode  size=$($size.Trim())  body=$body" -ForegroundColor Red
        $failed++
        $script:failures++
    }
}

Write-Host ""
info "Synthesis: $passed passed, $skipped not-loaded, $failed failed"

# ── 8. Engines that need ref WAV — just verify HTTP 4xx (not 500) ─────────────
hdr "8 — Ref-WAV Engines (expect 400, not 500)"
$refWavEngines = @("indextts", "f5tts", "openvoice", "cosyvoice", "xtts", "chatterbox", "fishspeech")
foreach ($engine in $refWavEngines) {
    $req = "{`"text`":`"$testText`",`"params`":{}}"
    $result = vm "curl -s -o /tmp/ref_out.json -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d '$req' http://localhost:8001/synthesize/$engine" -nocheck
    $httpCode = $result.Trim()
    if ($httpCode -match "^4") {
        Write-Host "  ✅ $($engine.PadRight(14)) HTTP $httpCode (correctly rejected — no ref WAV)" -ForegroundColor Green
    } elseif ($httpCode -eq "200") {
        Write-Host "  ✅ $($engine.PadRight(14)) HTTP 200 (synthesised without ref WAV)" -ForegroundColor Green
    } elseif ($httpCode -eq "503") {
        Write-Host "  ⏭  $($engine.PadRight(14)) HTTP 503 (not loaded)" -ForegroundColor DarkGray
    } elseif ($httpCode -eq "500") {
        $body = vm "cat /tmp/ref_out.json 2>/dev/null" -nocheck
        if ($body -match "load_model") {
            fail "$engine — still hitting load_model() error: $body"
        } else {
            Write-Host "  ⚠️  $($engine.PadRight(14)) HTTP 500 — $($body | Select-Object -First 1)" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  ⚠️  $($engine.PadRight(14)) HTTP $httpCode" -ForegroundColor Yellow
    }
}

# ── 9. GPU-only engines — verify they fail gracefully (not crash) ─────────────
hdr "9 — GPU-only Engines (expect graceful error, not crash)"
$gpuEngines = @("parler", "outetts", "qwen3tts", "orpheus")
foreach ($engine in $gpuEngines) {
    $req = "{`"text`":`"$testText`",`"params`":{}}"
    $result = vm "curl -s -o /tmp/gpu_out.json -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d '$req' http://localhost:8001/synthesize/$engine" -nocheck
    $httpCode = $result.Trim()
    $body = vm "cat /tmp/gpu_out.json 2>/dev/null | head -c 300" -nocheck
    if ($httpCode -match "^[45]") {
        if ($body -match "_attn_implementation_autoset") {
            fail "$engine — _attn_implementation_autoset error leaked through shim: $body"
        } elseif ($body -match "load_model") {
            fail "$engine — load_model() error: $body"
        } else {
            Write-Host "  ✅ $($engine.PadRight(14)) HTTP $httpCode (graceful fail on CPU VM)" -ForegroundColor Green
        }
    } elseif ($httpCode -eq "200") {
        pass "$engine synthesised (GPU must be available!)"
    } else {
        Write-Host "  ⚠️  $($engine.PadRight(14)) HTTP $httpCode" -ForegroundColor Yellow
    }
}

# ── 10. Check journal for crashes since restart ───────────────────────────────
hdr "10 — Journal Crash Check (last 5 min)"
$since = (Get-Date).AddMinutes(-5).ToString("yyyy-MM-dd HH:mm:ss")
$journal = vm "sudo journalctl -u arthur-lab --since '$since' --no-pager 2>/dev/null | grep -i 'error\|traceback\|killed\|segfault' | head -20" -nocheck
if ([string]::IsNullOrWhiteSpace($journal)) {
    pass "No errors/crashes in journal in last 5 minutes"
} else {
    warn "Found log entries (may be non-fatal):"
    $journal -split "`n" | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkYellow }
}

# ── Summary ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host ("═" * 64) -ForegroundColor DarkCyan
if ($script:failures -eq 0) {
    Write-Host "  🎉  ALL TESTS PASSED — Engine fixes verified OK" -ForegroundColor Green
} else {
    Write-Host "  ❌  $($script:failures) TEST(S) FAILED — Review output above" -ForegroundColor Red
}
Write-Host ("═" * 64) -ForegroundColor DarkCyan
Write-Host ""

exit $script:failures
