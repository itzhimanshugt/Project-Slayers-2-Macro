"""Locate Project Slayers 2 UI elements automatically.

Replaces manual drag-calibration for the elements that are reliably
findable by colour and shape. Roblox health bars are saturated red
rectangles in known screen quarters, which is a far more reliable signal
than a hand-drawn box that goes stale on the next UI change.

Everything here is derived from a real 2560x1600 WGC frame of the game, not
from any guide - the sources for this game that circulate online include
one that fabricated its own controls table.

Usage:
    python tools/locate_ui.py
    python tools/locate_ui.py --out logs/shots/ui_marked.png
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from slayersmacro import log  # noqa: E402
from slayersmacro.window import find_window  # noqa: E402

log.setup()
_state: dict = {"frame": None, "n": 0, "control": None}
_ready = threading.Event()


def on_frame_arrived(frame, control=None) -> None:  # noqa: N802
    _state["frame"] = frame
    _state["control"] = control
    _state["n"] += 1
    _ready.set()


def on_closed(_frame=None) -> None:  # noqa: N802
    pass


def grab(title: str = "Roblox", timeout: float = 20.0):
    from windows_capture import WindowsCapture
    win = find_window(title)
    if win is None:
        raise SystemExit(f"no window matching {title!r}")
    c = WindowsCapture(cursor_capture=False, draw_border=False,
                       minimum_update_interval=0, window_hwnd=win.hwnd)
    c.event(on_frame_arrived)
    c.event(on_closed)
    c.start_free_threaded()
    if not _ready.wait(timeout):
        if _state.get("control"):
            _state["control"].stop()
        raise SystemExit("no frame within timeout")
    return win, _state


def frame_to_bgr(frame_obj) -> np.ndarray:
    return np.array(frame_obj.frame_buffer, copy=True)


# --- element finders ----------------------------------------------------

# Saturated reds: HP bars and damage numbers. The two hue bands cover the
# difference between OpenCV's H in [0,180) and the wrap at 180.
def red_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo = cv2.inRange(hsv, (0, 130, 90), (10, 255, 255))
    hi = cv2.inRange(hsv, (170, 130, 90), (180, 255, 255))
    return cv2.bitwise_or(lo, hi)


def find_health_bars(bgr: np.ndarray) -> list[dict]:
    """Find wide, short, saturated-red rectangles. Health bars.

    Shape filters, not position filters, so this survives a window move:
    a health bar is wide relative to its height and sits on a dark plate.
    """
    h, w = bgr.shape[:2]
    mask = red_mask(bgr)
    # Bars are solid blocks; erode away text and small red UI accents.
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 3))
    solid = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    n, _, stats, _ = cv2.connectedComponentsWithStats(solid)

    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bw < w * 0.05:            # too short to be a bar
            continue
        if bh < 6 or bh > h * 0.08:  # too thin or too tall
            continue
        fill = area / max(1, bw * bh)
        if fill < 0.75:              # bars are solid; text is not
            continue
        # A health bar is drawn on a dark plate, so the area around its
        # top edge is dark. Skip anything sitting on a bright background.
        pad = 6
        y0, y1 = max(0, y - pad), min(h, y + bh + pad)
        around = bgr[y0:y1, max(0, x - 4):min(w, x + bw + 4)]
        if around.size and around.mean() > 90:
            continue
        out.append({"x": int(x), "y": int(y), "w": int(bw), "h": int(bh),
                    "fill": round(float(fill), 3),
                    "frac_full": round(float(bw) / w, 4)})
    return sorted(out, key=lambda d: d["y"])


def classify_bars(bars: list[dict], h: int) -> dict:
    """Assign roles by position: the player's bar is bottom-left, a boss's
    is in the upper half and usually wider on screen."""
    player = boss = None
    for b in bars:
        centre_y = b["y"] + b["h"] / 2
        if centre_y > h * 0.70 and b["x"] < b["w"] * 0.0 + 400:
            if player is None or b["x"] < player["x"]:
                player = b
        elif centre_y < h * 0.45:
            if boss is None or b["w"] > boss["w"]:
                boss = b
    return {"player_health": player, "boss_health": boss}


def find_skill_bar(bgr: np.ndarray) -> dict | None:
    """The ability bar: a row of square icons near the bottom centre.

    Located by finding the densest run of near-uniform dark squares in the
    lower third, which is the hotbar plate.
    """
    h, w = bgr.shape[:2]
    roi = bgr[int(h * 0.62):int(h * 0.85), int(w * 0.25):int(w * 0.75)]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    closed = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 40 or bh < 40:
            continue
        if not (0.7 < bw / max(1, bh) < 1.4):
            continue
        boxes.append((x, y, bw, bh))
    if len(boxes) < 3:
        return None
    boxes.sort(key=lambda b: b[0])
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    ox, oy = int(w * 0.25), int(h * 0.62)
    return {"x": ox + x0, "y": oy + y0, "w": x1 - x0, "h": y1 - y0,
            "icons": len(boxes)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default="Roblox")
    ap.add_argument("--out", default="logs/shots/ui_marked.png")
    args = ap.parse_args()

    win, st = grab(args.title)
    bgr = frame_to_bgr(st["frame"])
    h, w = bgr.shape[:2]
    print(f"frame {w}x{h}  (window rect {win.width}x{win.height})")

    bars = find_health_bars(bgr)
    print(f"\nred bar-shaped regions: {len(bars)}")
    for b in bars:
        print(f"  {b['w']:>5}x{b['h']:<4} at ({b['x']},{b['y']}) "
              f"fill={b['fill']}")

    roles = classify_bars(bars, h)
    print("\nassigned roles:")
    for name, b in roles.items():
        print(f"  {name:<14} {b if b else 'NOT FOUND'}")

    hot = find_skill_bar(bgr)
    print(f"\nskill bar: {hot}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    marked = bgr[:, :, :3].copy()
    for b in bars:
        cv2.rectangle(marked, (b["x"], b["y"]),
                      (b["x"] + b["w"], b["y"] + b["h"]), (0, 0, 255), 2)
    for name, b in roles.items():
        if not b:
            continue
        cv2.rectangle(marked, (b["x"] - 3, b["y"] - 3),
                      (b["x"] + b["w"] + 3, b["y"] + b["h"] + 3), (0, 255, 0), 3)
        cv2.putText(marked, name, (b["x"], max(14, b["y"] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if hot:
        cv2.rectangle(marked, (hot["x"], hot["y"]),
                      (hot["x"] + hot["w"], hot["y"] + hot["h"]), (255, 200, 0), 2)
        cv2.putText(marked, f'skills x{hot["icons"]}',
                    (hot["x"], hot["y"] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 200, 0), 2)
    cv2.imwrite(str(out), marked)
    print(f"\nmarked image: {out}")

    if st.get("control"):
        st["control"].stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())