# Roblox reconnect supervisor

Watches the Roblox client and recovers when a connection error appears:
spots the error dialog, clicks **Rejoin / Try Again**, and waits. It also
works from the main menu, where it joins the last experience again.

It is a separate tool from the Slayers macro in this repository. It knows
nothing about fishing, bosses or keys - it only reacts to the Roblox client
showing an error.

## What it does

1. Finds the Roblox client window (by process name, not window class).
2. Captures it with Windows Graphics Capture, so it keeps working when the
   window is covered or the display is asleep.
3. Classifies each frame as in-game, blank, or an error dialog.
4. Only after the same error screen persists does it click anything.
5. Clicks the primary button. Retries once. Then stops and waits, unless you
   explicitly allow it to press Escape and leave.

## Install

```sh
cd roblox-reconnect
python -m pip install -e .
```

## Use

Safe first step - report what it sees, click nothing:

```sh
python -m reconnect --check
```

Watch, but click nothing. **Do this first.** It tells you whether the
detector is right on your machine before it can do anything:

```sh
python -m reconnect --dry-run
```

Let it recover:

```sh
python -m reconnect
```

Also allow Escape/Leave when rejoining has failed. This can end your
session, so it is off by default:

```sh
python -m reconnect --allow-destructive
```

## First run: please read this

**A real Roblox connection-error dialog has never been seen on the machine
this was built on.** The error side of the detector is calibrated against a
dialog drawn to match Roblox's own UI conventions. The false-positive side
*has* been tested against real captured game frames.

So the asymmetry is: clicking too eagerly is well tested and now blocked;
recognising a real error dialog is not yet confirmed. Run `--dry-run`
through a real disconnect and check the log before trusting it unattended.

## Why it will not click your game

Early versions clicked the game's own **server browser**, which is a dark
screen with a "Waiting for response..." modal and a column of green JOIN
buttons - and would have thrown you out of your session. Two more false
positives came from ordinary night gameplay.

Both are now blocked, by requiring all four of:

| Test | Dialog | Server browser | Night gameplay |
|---|---|---|---|
| colour buckets | 7-20 | 12-34 | 49-98 |
| button count | 2 | 6-8 | 0-1 |
| vertical span | 6% | 39-46% | 2% |
| horizontal centring | 0.50 | 0.77 | 0.63 |

The bucket test is the important one: a Roblox dialog is flat-shaded UI, a
game frame is a textured 3D render. That is a physical difference, not a
tuned constant. Confirmed over 30 live frames - zero errors.

Those real frames are committed in `tests/fixtures/real/`, so this cannot
silently regress.

## Other safety rules

- The error screen must persist unchanged for several polls before acting.
- At most 2 clicks per incident. Then it stops and waits for you.
- Escape and Leave require `--allow-destructive`.
- It never clicks a window it cannot bring to the foreground.
- If it cannot find a button, it waits rather than escalating.

## Tests

```sh
python -m pytest tests/ -q
```

71 tests. Most use synthetic frames; the ones in `test_real_frames.py` use
real captures from a live client.

## Requirements

- Windows
- Python 3.11+
- Roblox running (client or launcher)