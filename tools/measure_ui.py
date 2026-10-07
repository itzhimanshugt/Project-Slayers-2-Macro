"""Measure the real Slayers 2 UI from a single WGC frame.

Everything in one grab, because the game state changes between captures
and measuring across several frames produced contradictory readings.

Layout priors are fractions of the frame, observed from real 2560x1600
captures of the game - not from any guide. The online sources for this
game include one that published a fabricated controls table, so nothing
here is taken on trust.

Usage:
    python tools/measure_ui.py
    python tools/measure_ui.py --out logs/shots/measured.png
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from slayersmacro import log  # noqa: E402
from slayersmacro.window import find_window, is_foreground  # noqa: E402

log.setup()
_st: dict = {"f": None, "c": None, "n": 0}
_ev = threading.Event()


def on_frame_arrived(frame, control=None) -> None:  # noqa: N802
    _st["f"] = frame
    _st["c"] = control
    _st["n"] += 1
    _ev.set()


def on_closed(_frame=None) -> None:  # noqa: N802
    pass


def red_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    a = cv2.inRange(hsv, (0, 120, 80), (12, 255, 255))
    b = cv2.inRange(hsv, (168, 120, 80), (180, 255, 255))
    return cv2.bitwise_or(a, b)


def bars_in(bgr, x0, y0, x1, y1, min_w=40, min_h=5, max_h=None):
    """Find solid red bars inside a region, returning filled spans.

    Returns (x, y, w, h) of the *filled* portion, so a half-full health bar
    reports half the width. That is the whole point of measuring a bar.
    """
    h_img, w_img = bgr.shape[:2]
    x0, x1 = max(0, int(x0)), min(w_img, int(x1))
    y0, y1 = max(0, int(y0)), min(h_img, int(y1))
    roi = bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return []
    mask = red_mask(roi)
    max_h = max_h or max(min_h * 2, int((y1 - y0) * 0.5))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 2))
    solid = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    n, _, stats, _ = cv2.connectedComponentsWithStats(solid)
    out = []
    for i in range(1, n):
        bx, by, bw, bh, area = stats[i]
        if bw < min_w or bh < min_h or bh > max_h:
            continue
        if area / max(1, bw * bh) < 0.7:
            continue
        out.append({"x": x0 + int(bx), "y": y0 + int(by),
                    "w": int(bw), "h": int(bh),
                    "fill": round(area / max(1, bw * bh), 3)})
    return sorted(out, key=lambda d: -d["w"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/shots/measured.png")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    from windows_capture import WindowsCapture
    win = find_window("Roblox")
    if win is None:
        print("no Roblox window")
        return 1
    print(f"window rect: {win.width}x{win.height} at ({win.left},{win.top})  "
          f"foreground={is_foreground(win.hwnd)}")

    c = WindowsCapture(cursor_capture=False, draw_border=False,
                       minimum_update_interval=0, window_hwnd=win.hwnd)
    c.event(on_frame_arrived)
    c.event(on_closed)
    c.start_free_threaded()
    if not _ev.wait(20):
        print("no frame")
        return 2

    bgr = np.array(_st["f"].frame_buffer, copy=True)
    h, w = bgr.shape[:2]
    print(f"WGC frame:   {w}x{h}")
    if h != win.height or w != win.width:
        print(f"  !! frame is {w}x{h} but GetClientRect says "
              f"{win.width}x{win.height} (DPI scale active)")
        print(f"     window-relative coordinates must be scaled by "
              f"{h / max(1, win.height):.4f} vertically")

    # Search windows as fractions, chosen from observation. Generous, then
    # we keep the widest match inside each.
    zones = {
        "player_health": (0.00, 0.700, 0.28, 0.10),
        "boss_health":   (0.20, 0.130, 0.45, 0.075),
        "target_health": (0.25, 0.270, 0.50, 0.075),
        "stamina":       (0.33, 0.590, 0.35, 0.055),
    }
    found = {}
    for name, (fx0, fy0, fx1, fy1) in zones.items():
        bars = bars_in(bgr, w * fx0, h * fy0, w * fx1, h * fy1)
        found[name] = bars
        best = bars[0] if bars else None
        print(f"\n{name}: {len(bars)} candidate bar(s)")
        for b in bars[:3]:
            print(f"    {b['w']:>4}x{b['h']:<3} at ({b['x']},{b['y']}) "
                  f"fill={b['fill']}  width={b['w'] / w:.4f} of frame")
        if best:
            print(f"    -> normalised: x={best['x'] / w:.4f} "
                  f"y={best['y'] / h:.4f} w={best['w'] / w:.4f} "
                  f"h={best['h'] / h:.4f}")

    # Skill slots: small icons in a row near the bottom centre.
    roi = bgr[int(h * 0.655):int(h * 0.72), int(w * 0.32):int(w * 0.50)]
    print(f"\nskill-key row: sampled {roi.shape[1]}x{roi.shape[0]} "
          f"at ({int(w * 0.32)},{int(h * 0.655)})")
    # Count bright icon squares on the dark plate.
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bright = (gray > 90).sum(axis=0)
    runs = []
    on = bright > 3
    start = None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > 8:
                runs.append((start, i))
            start = None
    print(f"  icon-ish column runs: {len(runs)}  -> "
          f"{[f'{a}-{b}' for a, b in runs[:10]]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    marked = bgr[:, :, :3].copy()
    colours = {"player_health": (0, 255, 0), "boss_health": (0, 0, 255),
               "target_health": (255, 160, 0), "stamina": (255, 0, 255)}
    for name, bars in found.items():
        for b in bars[:1]:
            col = colours.get(name, (255, 255, 255))
            cv2.rectangle(marked, (b["x"], b["y"]),
                          (b["x"] + b["w"], b["y"] + b["h"]), col, 3)
            cv2.putText(marked, name, (b["x"], b["y"] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    cv2.imwrite(str(out), marked)
    print(f"\nmarked: {out}")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        payload = {"frame": [w, h], "window": [win.width, win.height],
                   "bars": {k: v[:1] for k, v in found.items()}}
        Path(args.json).write_text(json.dumps(payload, indent=2),
                                   encoding="utf-8")
        print(f"json: {args.json}")

    if _st.get("c"):
        _st["c"].stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())