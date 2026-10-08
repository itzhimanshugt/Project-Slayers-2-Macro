"""Slayers macro tool - Roblox reconnect and error-recovery supervisor.

Watches the Roblox client and recovers from connection errors: detects the
error dialog, clicks Rejoin / Try Again, and gives up without acting rather
than clicking blindly when it cannot read the screen.

External only: reads the window's own surface via Windows Graphics Capture
and sends synthetic mouse input. Never touches the Roblox process.

Run:
    python -m reconnect                    # watch and recover
    python -m reconnect --check            # report what it sees, click nothing
    python -m reconnect --allow-destructive # also Escape/Leave on repeated failure
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from . import log
from .supervisor import RecoveryConfig, Supervisor
from .ui import classify
from .wgc import open_capture
from .window import find_roblox

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"

BANNER = r"""
  Roblox reconnect supervisor
  detects connection errors, clicks Rejoin/Try Again, recovers
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="reconnect", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="report the current screen and exit; click nothing")
    ap.add_argument("--allow-destructive", action="store_true",
                    help="permit Escape/Leave/relaunch, not just Rejoin")
    ap.add_argument("--dry-run", action="store_true",
                    help="report every action instead of performing it. Use "
                         "this the first time: it is the only way to check "
                         "behaviour against a real error dialog without "
                         "clicking it.")
    ap.add_argument("--no-wgc", action="store_true",
                    help="use screen capture instead of Windows Graphics "
                         "Capture (cannot see occluded or display-off)")
    ap.add_argument("--confirm-frames", type=int, default=8)
    ap.add_argument("--cooldown", type=float, default=3.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log.setup("DEBUG" if args.verbose else "INFO", "reconnect")
    lg = log.get("reconnect")
    print(BANNER)

    win = find_roblox()
    if win is None:
        lg.error("no Roblox client or launcher window found. Start Roblox "
                 "first (the launcher counts, so this also works from the "
                 "main menu).")
        return 1
    lg.info(f"watching: {win.describe()}")

    try:
        cap = open_capture(prefer_wgc=not args.no_wgc)
    except RuntimeError as exc:
        lg.error(str(exc))
        return 2
    lg.info(f"capture backend: {cap.backend}")
    if cap.backend == "mss":
        lg.warning(
            "using screen capture. That returns whatever is composited at "
            "Roblox's coordinates, so another window on top will be read as "
            "the game, and capture stops when the display sleeps. Install "
            "windows-capture for the reliable path.")

    if args.check:
        frame = cap.grab()
        det = classify(frame)
        print(f"\nscreen   : {det.screen.value}")
        print(f"confidence: {det.confidence:.2f}")
        print(f"reason   : {det.reason}")
        print(f"mean BGR : ({det.mean_bgr[0]:.0f}, {det.mean_bgr[1]:.0f}, "
              f"{det.mean_bgr[2]:.0f})")
        print(f"buttons  : {len(det.buttons)}")
        for b in det.buttons:
            print(f"   {b.kind:<10} {b.w}x{b.h} at ({b.x},{b.y}) "
                  f"score={b.score}")
        if frame is not None:
            out = LOG_DIR / "shots" / f"check_{int(time.time())}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            import cv2
            cv2.imwrite(str(out), frame[:, :, :3])
            print(f"\nsaved: {out}")
        cap.close()
        return 0

    cfg = RecoveryConfig(
        confirm_frames=args.confirm_frames,
        action_cooldown_s=args.cooldown,
    )
    sup = Supervisor(cap, cfg, allow_destructive=args.allow_destructive,
                     logger=lg, dry_run=args.dry_run)
    if args.dry_run:
        lg.warning("DRY RUN: nothing will be clicked or pressed")
    lg.info(f"running. Ctrl+C to stop."
            + (" (destructive actions disabled)" if not args.allow_destructive
               else ""))

    actions = 0
    try:
        while not sup.should_stop:
            win = find_roblox()
            if win is None:
                time.sleep(1.0)
                continue
            frame = cap.grab()
            if frame is None:
                # WGC can drop frames when the display is off or the client
                # is busy. Not an error by itself.
                time.sleep(0.2)
                continue
            outcome, det = sup.tick(frame, win)
            if outcome == "acted":
                actions += 1
            if not cap.healthy(max_stale_ms=8000.0):
                lg.warning("capture has gone stale; is the display asleep?")
                cap.close()
                try:
                    cap = open_capture(prefer_wgc=not args.no_wgc)
                    sup.cap = cap
                    lg.info("capture re-opened")
                except RuntimeError as exc:
                    lg.error(f"could not re-open capture: {exc}")
                    return 3
            time.sleep(cfg.poll_interval_ms / 1000.0)
    except KeyboardInterrupt:
        lg.info("Ctrl+C, stopping")
    finally:
        sup.stop()
        cap.close()
        lg.info(f"stopped after {actions} recovery action(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
