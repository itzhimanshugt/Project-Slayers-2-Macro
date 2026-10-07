# Slayers Macro

External screen-reading automation for Project Slayers 2 (Roblox).

It reads your screen and sends real key presses. It does **not** touch the
Roblox client process — no injection, no memory reading, no DLL.

---

## Quick start

```powershell
cd slayers-macro
python -m pip install -r requirements.txt

# draw boxes over the bars this macro needs (once per window size)
python tools/calibrate.py fishing

# if detection misreads the bar, let it tune itself
python tools/autotune.py fishing_bar

# run it
python run.py fishing
```

**F6** start/stop · **F12** panic · **F7** show calibrated regions

---

## What it does

| | |
|---|---|
| **Macros** | `combat`, `fishing`, `boss`, `afk`, `diagnose` |
| **Safety** | Focus gate, window-stability gate, occlusion check, panic key |
| **Recovery** | Escalating ladder: wait → nudge → back off → turn → re-walk → respaw |
| **Adaptation** | Learns your real attack interval; can retune detection thresholds |
| **Prediction** | Kalman filter decides at the actuation horizon, not the current frame |

---

## Measured on this machine

Legion Pro 7i · Core Ultra 9 275HX · RTX 5070 Ti · 2560×1600 @ 240 Hz

| Thing | Measured |
|---|---|
| `SendInput` batched keypress | **135 µs** min, 308 µs p50 |
| Region capture (small strip) | **4.18 ms** p50 |
| Full 2560×1600 capture | 59 ms — never used |
| Sustained capture+press | 239 cycles/sec |
| Filter vs raw, position | ~3× better |
| Filter vs naive extrapolation, 9 ms ahead | **~10× better** |

Two things that shaped the design:

**Capture is vsync-locked.** 4.18 ms is 1/240 s — your display refresh. A
4,800-pixel region costs the same as a 65,536-pixel one, so the cost is the
frame wait, not the pixels. Every macro grabs a calibrated region rather than
the full screen.

**`pydirectinput`'s default `PAUSE=0.1` adds 100 ms to every call.** Measured
201 ms per press against 302 µs for raw `SendInput`. We don't use it in the
hot path. If you use it anywhere, set `PAUSE = 0` first.

### The latency budget

```
capture (vsync-locked)     4.17ms   60.8%
detect + filter            0.30ms    4.4%
SendInput                  0.30ms    4.4%
game frame wait (avg)      2.08ms   30.4%
                            -------
TOTAL to effect            6.85ms
```

You cannot beat this by optimising. ~60% of it is waiting for the display and
~30% is the game waiting for its own frame boundary. The only thing under your
control is the ~8% in the middle, which is why that is the only part worth
optimising.

---

## Safety

This module exists because a macro once attached to a Roblox window that
existed but was not the window being looked at, and began pressing keys
against whatever was actually on screen. Two things were wrong with the
original code, and both are now checked every frame:

- **A window existing is not a window being visible.** Screen capture reads
  whatever is composited at a screen rectangle. If something else covers
  Roblox, you get that instead. `guard.py` compares against a reference frame
  and refuses to act when the rect shows a static image.
- **A moving window makes every coordinate wrong.** Acting mid-resize is a
  misfire. The guard requires the rect to have been stable for 400 ms.

Every macro routes input through `ctx.may_act()`. There is one gate, so a
macro cannot forget a check by not knowing about it.

`F12` releases every held key. So does a crash, and so does process exit.

### Recovery ladder

A macro that knows one way to do a thing breaks the first time the world
disagrees. This escalates only after each rung has actually been tried and
failed:

```
hold → recheck → nudge → backoff → turn → rewalk → respawn → [serverhop]
 free      free     input    input   input   input    input      manual
```

`serverhop` is the most disruptive rung and requires explicitly opting in.
Most "stuck" detections are a menu, a cutscene, or you touching the keyboard,
so the ladder confirms for several frames before escalating at all.

---

## Adaptation

Not self-evolution — it does not modify its own code, train a network, or
choose strategy. Three bounded, verifiable mechanisms:

- **Timing** — learns your real attack interval from observed presses, using
  a median so one lag spike can't poison it. Clamped so a bad session can't
  permanently halve it.
- **Thresholds** — nudges detection bounds based on observed success rate.
  One-directional on purpose: an adapter that chases a target rate
  oscillates. Off by default, because silently retuning a good hand calibration
  is the worst failure mode available.
- **Obstruction vs standing still** — a still screen with nothing held is not
  stuck. This discriminator is why the ladder no longer walks itself into a menu.

---

## Honest limits

**Latency floor.** Roblox samples input twice per frame. At 240 FPS the best
case is ~6.9 ms from screen to effect. Still ~25× faster than a human's 250 ms,
but not zero.

**240 FPS is already the ceiling.** Fishstrap's framerate unlock past 240
doesn't work — those fastflags are off Roblox's allowlist, and a Roblox
staffer stated above 240 Hz is intentionally unsupported for physics reasons.

**Detection thresholds are unvalidated.** No primary source describes Slayers
2's UI. SEO sites that claim to — one published a fabricated "official PC
bindings" table that does not exist on the real game page. So nothing is
hardcoded from a guide: you draw the boxes, and the tuner fits to what is
really on screen. **Run `diagnose` and check the log before trusting any
macro.**

**Navigation is recovery, not pathfinding.** Pixels cannot see rocks. The
ladder notices you're not moving and escalates; it does not know where to go.

**Keys are Roblox defaults, verified not guessed.** Space = jump, WASD =
movement, Escape = menu, Shift = sprint. M1 is **unbound** in Roblox, so the
attack defaults to the left mouse button, not Space. Ability keys 1–4 are
`configurable` and unverified.

---

## Not doing code injection

Injection makes the Roblox client a *modified client*, which Roblox says it
auto-actions with account termination ([devforum](https://devforum.roblox.com/t/an-update-on-automated-action-against-modified-clients/3640609)).
External input macros are a different thing and are not what this does.

The real risk is the game developer, not Roblox. Roblox's own
[server-side detection guidance](https://create.roblox.com/docs/en-us/scripting/security/server-side-detection.md)
names **action cadence** — identical intervals — as the signal, which is why
`jitter_ms` exists in the config.

---

## Layout

```
run.py                  entry point
config/config.example.toml   annotated config - copy to config.toml
config/calibration.json      your drawn regions (gitignored)
src/slayersmacro/
  input.py    SendInput, scancodes, batched taps
  capture.py  region-only screen capture
  window.py   window tracking, DPI awareness
  detect.py   bar/marker/health detection
  kalman.py   1D constant-velocity filter + lookahead
  latency.py  where the milliseconds go
  guard.py    safety interlocks
  recovery.py escalation ladder
  actions.py  concrete recovery actions
  adaptation.py  timing learning, threshold drift
  engine.py   macro loop, panic, held-key safety
  macros.py   combat, fishing, boss, afk, diagnose
  hotkeys.py  global hooks on their own thread
tools/
  calibrate.py  draw the regions
  autotune.py   learn thresholds from live frames
  bench_input.py / bench_capture.py
tests/          169 tests, no game required
```

---

## Tests

```powershell
python -m pytest tests/ -q
```

Input tests use **F24** as the test key: real events, nothing bound to it.
