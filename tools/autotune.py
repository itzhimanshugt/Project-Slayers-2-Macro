"""Offline threshold tuner.

Records a burst of frames of a calibrated region and learns the bar
geometry plus working HSV thresholds, then writes them to config/tuned.json.

Run it when detection is failing, or after the game updates. It refuses to
write anything if the frames do not actually contain a bar - a plausible
looking config derived from the wrong pixels is worse than no config.

Usage:
    python tools/autotune.py fishing_bar
    python tools/autotune.py fishing_bar --frames 24 --interval 90
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from slayersmacro import config, log  # noqa: E402
from slayersmacro.autotune import (  # noqa: E402
    learn_region, otsu_split, record_frames,
)
from slayersmacro.detect import Calibration  # noqa: E402
from slayersmacro.engine import MacroContext  # noqa: E402
from slayersmacro.window import find_window  # noqa: E402

from slayersmacro.capture import Capture  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("region")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--interval", type=float, default=80.0)
    ap.add_argument("--config", default=str(
        Path(__file__).resolve().parent.parent / "config" / "config.toml"))
    args = ap.parse_args()

    log.setup()
    cfg = config.load(args.config)
    win = find_window(cfg["general"].get("window_title", "Roblox"))
    if win is None:
        print("No Roblox window found. Open the game first.")
        return 1

    calib = Calibration.load()
    if args.region not in calib.regions:
        have = ", ".join(calib.regions) or "(none)"
        print(f"No region named '{args.region}'. Calibrated: {have}")
        print(f"Add it with: python tools/calibrate.py fishing")
        return 1

    ctx = MacroContext(cfg, Capture(), win, calib)
    print(f"Recording {args.frames} frames of '{args.region}' "
          f"every {args.interval:.0f}ms...")
    print("Get the bar on screen now. Sampling starting in 2s.\n")

    import time
    time.sleep(2.0)
    frames = record_frames(ctx.region_frame, args.region,
                           count=args.frames, interval_ms=args.interval)
    if not frames:
        print("No frames captured.")
        return 1
    print(f"captured {len(frames)} frames "
          f"({frames[0].shape[1]}x{frames[0].shape[0]})")

    tuned = learn_region(frames, args.region)
    if tuned is None:
        print("\nFAILED: no bar-shaped object in those frames.")
        print("Likely causes:")
        print("  - the box is in the wrong place (re-run calibrate)")
        print("  - the bar is not currently on screen")
        print("  - the region is too large or too small")
        return 2

    path = tuned.save()
    print(f"\nwrote {path}\n")
    print(f"  bar brightness : {tuned.bar_v_min} .. {tuned.bar_v_max}")
    print(f"  marker bright  : > {tuned.marker_v_min}")
    print(f"  saturation min : {tuned.bar_s_min}")
    print(f"  geometry       : {tuned.geometry}")
    print(f"  confidence     : {tuned.confidence:.2f}  "
          f"(below 0.5 means the frames disagreed - try more)")
    print("\nverify with: python run.py diagnose")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())