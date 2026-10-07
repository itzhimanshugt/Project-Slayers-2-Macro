"""Crop and upscale a region of the live game so small text is readable.

Usage:
    python tools/zoom.py 0,1140,400,120          # player health, bottom-left
    python tools/zoom.py 620,230,720,180          # boss bar + drop table
    python tools/zoom.py 780,1130,500,130 --scale 3
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

_state: dict = {"frame": None, "control": None}
_ready = threading.Event()


def on_frame_arrived(frame, control=None) -> None:  # noqa: N802
    _state["frame"] = frame
    _state["control"] = control
    _ready.set()


def on_closed(_frame=None) -> None:  # noqa: N802
    pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("region", help="x,y,w,h in frame coords")
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--title", default="Roblox")
    args = ap.parse_args()

    from windows_capture import WindowsCapture
    from slayersmacro.window import find_window

    win = find_window(args.title)
    if win is None:
        print(f"no window matching {args.title!r}")
        return 1
    c = WindowsCapture(cursor_capture=False, draw_border=False,
                       minimum_update_interval=0, window_hwnd=win.hwnd)
    c.event(on_frame_arrived)
    c.event(on_closed)
    c.start_free_threaded()
    if not _ready.wait(20):
        print("no frame")
        return 2

    bgr = np.array(_state["frame"].frame_buffer, copy=True)
    x, y, w, h = (int(v) for v in args.region.split(","))
    crop = bgr[y:y + h, x:x + w, :3]
    if crop.size == 0:
        print(f"region {args.region} is outside the {bgr.shape[1]}x{bgr.shape[0]} frame")
        return 3
    if args.scale != 1.0:
        crop = cv2.resize(crop, None, fx=args.scale, fy=args.scale,
                          interpolation=cv2.INTER_CUBIC)
    out = Path(args.out or f"logs/shots/zoom_{x}_{y}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), crop)
    print(f"frame {bgr.shape[1]}x{bgr.shape[0]}  region {args.region}  "
          f"-> {crop.shape}  saved {out}")
    if _state.get("control"):
        _state["control"].stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())