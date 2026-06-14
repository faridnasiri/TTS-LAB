# Session Report — 2026-04-06
## Android 14 ICS Binding Bypass — Incoming Call UI Architecture

---

## Overview

Two full conversation threads spent diagnosing why `IncomingCallActivity` never
appeared when a call rang on the Pixel 5 (Android 14). The root cause turned out
to be a permanent Android 14 `isBoundAndConnectedToServices()` regression
combined with three independent secondary bugs that each had to be peeled away
one at a time.

**Commits in this session (oldest → newest):**

| Hash | Message |
|---|---|
| `284b69f` | fix: add IN_CALL_SERVICE_RINGING metadata to BaiterInCallService *(prior session, shown for context)* |
| `0bff488` | fix: bypass Android 14 Emergency ICS binding; screen service posts notification, activity uses TelecomManager fallback |
| `70c300d` | fix: use StartActivity() as primary incoming call UI trigger |

---

## Starting State

- `IN_CALL_SERVICE_RINGING = true` metadata was added in the prior session as a
  hypothesis fix for Android 14 silently skipping ICS binding.
- Default dialer confirmed (`adb shell telecom get-default-dialer` →
  `com.companyname.spamblocker`).
- `BaiterWatchdogService` confirmed running as foreground service.
- `BaiterInCallService` was **never** being bound for incoming calls; no
  `bind_ics` event ever appeared in `dumpsys telecom`.

---

## Root Cause A — Android 14 Emergency ICS Permanently Connected

**Symptom:** `adb shell dumpsys telecom` captured live during TC@10 (audibly
ringing, `ringerMode=2`, `START_RINGER` logged) showed:

```
Emergency: BindingConnection [connected, bound, Google Dialer InCallServiceImpl]
Default-Dialer: BindingConnection [not connected, not bound, BaiterInCallService]
```

`bind_ics` was **completely absent** from the Timings section even while the
phone was actively ringing.

**Root cause:**
Android 14 keeps `com.google.android.dialer/InCallServiceImpl` permanently
connected as the "Emergency" InCallService. `InCallController.isBoundAndConnectedToServices()`
checks whether *any* ICS connection is active — because Emergency is always
connected, it always returns `true`. `bindToServices()` is therefore **never
called** for BaiterInCallService on incoming calls.

**Why TC@1 worked (previously):**
TC@1 was an incoming call that appeared to work. Investigation revealed the user
had made an outgoing call (TC@2) immediately before. Outgoing calls go through
`placeOutgoingCall()` which **always** calls `bindToServices()` regardless of
`isBoundAndConnectedToServices()`. BaiterInCallService was already bound from
TC@2 when TC@1 arrived.

**Fix (commit `0bff488`):** Bypass ICS binding entirely for the incoming call UI.
`BaiterScreeningService` (which is always called before any ICS decision) became
the new trigger for `IncomingCallActivity`.

---

## Root Cause B — IncomingCallActivity Immediately Finished on null CurrentCall

**Symptom (pre-fix code):**
```csharp
var call = BaiterInCallService.CurrentCall;
if (call == null) { Finish(); return; }   // line 43
```

**Root cause:** Since ICS was never bound, `CurrentCall` was always `null` when
the activity launched. Even if a notification had somehow launched the activity,
it would have self-destructed in `OnCreate`.

**Fix (commit `0bff488`):** Replaced the null guard with multi-source number
extraction and a graceful fallback chain:
```csharp
_number = call?.GetDetails()?.GetHandle()?.SchemeSpecificPart
          ?? Intent?.GetStringExtra("number")
          ?? BaiterScreeningService.PendingCallNumber
          ?? string.Empty;

if (string.IsNullOrEmpty(_number)) { Finish(); return; }
```

---

## Three-Part Architectural Fix (commit `0bff488`)

### Part 1 — BaiterScreeningService as notification source

`CallScreeningService.OnScreenCall()` is guaranteed to fire for every incoming
call before Telecom makes any ICS binding decision.

Added:
- `public static string? PendingCallNumber { get; set; }` — static field readable
  by `IncomingCallActivity` in-process.
- `PostScreeningNotification(string number)` — posts a `CategoryCall`,
  `USE_FULL_SCREEN_INTENT`, ongoing notification pointing to
  `IncomingCallActivity` with the number passed as an intent extra.
- After `RespondToCall()` for non-contact calls: sets `PendingCallNumber` and
  calls `PostScreeningNotification`.

### Part 2 — IncomingCallActivity null-safe TelecomManager fallbacks

- `ShowRingingPhase(Call? call, ...)` — call is now nullable throughout.
- `AnswerCall(Call? call)`:
  - If `call != null` → `call.Answer(0)` + `BaiterInCallService.BeginActiveCallStatic()`
  - If `call == null` → `TelecomManager.AcceptRingingCall()` (deprecated API,
    valid for default dialer without `MODIFY_PHONE_STATE`)
- `RejectCall(Call? call)`:
  - If `call != null` → `call.Reject(false, null)`
  - If `call == null` → `TelecomManager.EndCall()`

### Part 3 — BaiterInCallService STATE_ACTIVE unified init

- `OnCallAdded()` STATE_ACTIVE else-branch now calls `BeginActiveCallStatic()`
  (overlay + transcription + bridge) instead of duplicating the same logic inline.
- `OnCallRemoved()` clears `BaiterScreeningService.PendingCallNumber = null` to
  prevent stale state from re-opening the activity on the next call.

---

## Bug 2 — RespondToCall IllegalStateException Swallowed the Notification

**Symptom (first test after `0bff488`):**
```
W Baiter/Screen: RespondToCall IllegalState: Invalid response state for allowed call. — falling back to allow call
```
No further Baiter logs. `IncomingCallActivity` never launched.

**Root cause:**
`RespondToCall` threw `Java.Lang.IllegalStateException` with the message
`"Invalid response state for allowed call"`. This is thrown by Android's
`CallScreeningServiceAdapter.allowCall()` when `getCallById(callId)` returns
`null` — meaning the call object had become stale in the `CallsManager` between
`OnScreenCall` being invoked and our `RespondToCall` being called (~123 ms later,
occupied by the `IsKnownContact` content-resolver query).

The catch block correctly retried `RespondToCall` with a clean builder, but
`PostScreeningNotification()` was placed **after** `RespondToCall` and was
therefore never reached.

**Fix (included in `70c300d`):**
Moved `PendingCallNumber = number; PostScreeningNotification(number)` to
**before** `RespondToCall`, inside the try block. Even if `RespondToCall` throws,
the notification has already been posted.

---

## Bug 3 — USE_FULL_SCREEN_INTENT Rejected, Release Log.Debug Invisible

**Symptom (second test — TC@13, TC@15):**
No `Baiter/*` logs at all despite the Telecom dump confirming:
```
SCREENING_COMPLETED, [Allow, logged, notified, mCallScreeningAppName = Spamblocker]
```

**Two sub-bugs:**

### 3a — Log.Debug suppressed in release builds

`Log.Debug` (priority D) is suppressed by Android's log system for
non-debuggable (release) apps. All diagnostic logs were invisible, making it
impossible to confirm whether `PostScreeningNotification` ran.

**Fix:** Changed key diagnostic calls to `Log.Info` / `Log.Warn` which are
always visible regardless of app debuggability.

### 3b — USE_FULL_SCREEN_INTENT rejected

```
adb shell appops get com.companyname.spamblocker USE_FULL_SCREEN_INTENT
→ USE_FULL_SCREEN_INTENT: default; rejectTime=+3h32m4s203ms ago
```

The `rejectTime` confirmed the full-screen intent had been previously rejected.
Even though the notification was being posted, Android was silently downgrading
it to a normal shade notification — the activity was never launched.

**Fix:**
1. Granted the permission explicitly:
   ```
   adb shell appops set com.companyname.spamblocker android:use_full_screen_intent allow
   ```
2. Added `StartActivity()` as the **primary** trigger so full-screen intent
   permission is not required at all. The default dialer app receives a
   Background Activity Launch (BAL) exemption from Android during incoming calls.

**Final architecture in `LaunchIncomingCallUi()` (commit `70c300d`):**
```csharp
private void LaunchIncomingCallUi(string number)
{
    // Primary: direct activity start — BAL-exempt as default dialer
    try
    {
        var activityIntent = new Intent(this, typeof(IncomingCallActivity));
        activityIntent.AddFlags(ActivityFlags.NewTask | ActivityFlags.ClearTop
                                                      | ActivityFlags.SingleTop);
        activityIntent.PutExtra("number", number);
        StartActivity(activityIntent);
        Android.Util.Log.Warn("Baiter/Screen", $"StartActivity(IncomingCallActivity) OK for {number}");
    }
    catch (Exception ex)
    {
        Android.Util.Log.Warn("Baiter/Screen", $"StartActivity failed ({ex.Message}) — falling back to notification");
    }

    // Secondary: full-screen notification — persistent in shade + USE_FULL_SCREEN_INTENT backup
    PostScreeningNotification(number);
}
```

---

## Call Flow (Post-Fix)

```
Incoming call
    │
    ▼
BaiterScreeningService.OnScreenCall()
    │  score, IsKnownContact check
    │  if blocked → RespondToCall(reject) → done
    │
    ├─ if !inContacts:
    │      PendingCallNumber = number
    │      LaunchIncomingCallUi(number)
    │          ├─ StartActivity(IncomingCallActivity) ← primary (BAL exempt)
    │          └─ PostScreeningNotification()         ← backup (shade + full-screen)
    │
    └─ RespondToCall(allow / silence)
           ↑ may throw IllegalStateException → catch retries, UI already shown

IncomingCallActivity.OnCreate()
    │  call = BaiterInCallService.CurrentCall     ← may be null on Android 14
    │  _number = call?.number
    │           ?? Intent.GetStringExtra("number")  ← from notification/startActivity
    │           ?? BaiterScreeningService.PendingCallNumber
    │
    └─ ShowRingingPhase(call?, score, autoFlag)
           btnAnswer → AnswerCall(call?)
               ├─ call != null → call.Answer(0) + BeginActiveCallStatic()
               └─ call == null → TelecomManager.AcceptRingingCall()
           btnDecline → RejectCall(call?)
               ├─ call != null → call.Reject(false, null)
               └─ call == null → TelecomManager.EndCall()

BaiterInCallService.OnCallAdded() [fires when ICS eventually binds, post-answer]
    │  STATE_ACTIVE → BeginActiveCallStatic()
    │      CancelNotification(42)
    │      BaiterOverlayManager.Show()
    │      StartTranscription()
    │      if PendingAutoBait → InitiateHomeBridge()
    │
OnCallRemoved()
    └─ BaiterScreeningService.PendingCallNumber = null  ← prevent stale re-open
```

---

## Key Lessons Learned

| Lesson | Detail |
|---|---|
| Android 14 Emergency ICS | Google Dialer is permanently bound as "Emergency" ICS. `isBoundAndConnectedToServices()` always returns `true`. `bindToServices()` is never called for third-party default dialers on incoming calls. |
| Outgoing calls bypass the bug | `placeOutgoingCall()` always calls `bindToServices()` unconditionally. Once the outgoing call binds the ICS, subsequent incoming calls in the same session also see it bound. |
| `RespondToCall` can throw | `IllegalStateException: "Invalid response state for allowed call"` means the call ID is no longer in `CallsManager`. Put any work that must happen (UI launch) **before** `RespondToCall`, not after. |
| Release builds suppress `Log.Debug` | D-level logs are filtered out for non-debuggable APKs. Use `Log.Info` or `Log.Warn` for anything you need to see from release builds. |
| `USE_FULL_SCREEN_INTENT` is not auto-granted | Even for the default dialer, the `appops` mode can show `default` while still having a historical `rejectTime`. Grant explicitly via `appops set ... allow` and/or use `StartActivity()` which bypasses the permission entirely via the BAL exemption. |
| BAL exemption for default dialer | The default phone app receives a temporary Background Activity Launch exemption when an incoming call arrives. `StartActivity()` works from any service in the package without `SYSTEM_ALERT_WINDOW`. |
| `Exported = false` + same-UID `StartActivity()` | A non-exported activity can still be started by any component in the same package (same UID), including services running in a separate process of the same package. |

---

## Debugging Commands Used

```powershell
# Full telecom state dump — captures mInCallController binding status
adb shell dumpsys telecom > $env:TEMP\telecom.txt

# Confirm default dialer
adb shell telecom get-default-dialer

# Check USE_FULL_SCREEN_INTENT grant status
adb shell appops get com.companyname.spamblocker USE_FULL_SCREEN_INTENT

# Grant USE_FULL_SCREEN_INTENT explicitly
adb shell appops set com.companyname.spamblocker android:use_full_screen_intent allow

# Filter logcat to Baiter tags (works in release: Info/Warn, not Debug)
adb logcat -s "Baiter/Screen","Baiter/UI","Baiter/Call","Baiter/ICS"

# Find all Baiter-related lines in full logcat dump
adb logcat -d "*:D" 2>&1 | Select-String "Baiter|spamblocker|IncomingCall"

# Find lines from a specific PID
adb logcat -d "*:D" 2>&1 | Select-String "23557"

# Build Release APK (VS has Debug output locked during debug session)
dotnet build Spamblocker/Spamblocker.csproj -f net10.0-android -c Release

# Install to connected device
adb install -r Spamblocker\bin\Release\net10.0-android\com.companyname.spamblocker-Signed.apk
```

---

## Files Changed

| File | Changes |
|---|---|
| `Spamblocker/Services/BaiterScreeningService.cs` | Added `PendingCallNumber`, `LaunchIncomingCallUi()`, `PostScreeningNotification()`; moved UI launch before `RespondToCall`; upgraded logs to Info/Warn |
| `Spamblocker/IncomingCallActivity.cs` | Multi-source number extraction; `ShowRingingPhase(Call?)` nullable; `AnswerCall(Call?)` / `RejectCall(Call?)` helpers with TelecomManager fallbacks |
| `Spamblocker/Services/BaiterInCallService.cs` | STATE_ACTIVE branch uses `BeginActiveCallStatic()`; `OnCallRemoved` clears `PendingCallNumber` |

---

## Pending / Next Steps

- [ ] **Test the full fix** — make an incoming call from an unknown number and
  confirm `IncomingCallActivity` appears, shows the caller number, and Answer/Bait/Block all work.
- [ ] **Confirm `BaiterInCallService.OnCallAdded` STATE_ACTIVE fires after answer** — after tapping Answer (via `TelecomManager.AcceptRingingCall()`), the ICS should bind and `BeginActiveCallStatic()` should start transcription and the bridge. Watch logcat for `Baiter/ICS` or `Baiter/Call` after answer.
- [ ] **Verify notification dismissal** — after the call is answered and `BeginActiveCallStatic()` runs, the notification (id=42) should be cancelled by `CancelNotification(42)` in the ICS. Confirm it disappears from the shade.
- [ ] **Test Bait path** — tap Bait 🎣 and confirm `PendingAutoBait=true` flows into `BeginActiveCallStatic()` which initiates the Twilio home bridge.
- [ ] **Test Block path** — tap Block and confirm `BlocklistManager.Block()` is called and `TelecomManager.EndCall()` terminates the call.
