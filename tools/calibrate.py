"""Calibration helper: pick the screen regions the macros need.

Run once per resolution/window-size. It shows a live preview of the Roblox
window and asks you to drag boxes over the bars you care about. Regions are
stored as fractions of the window, so they survive a resize.

Usage:  python tools/calibrate.py fishing
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from slayersmacro import log  # noqa: E402
from slayersmacro.capture import Capture  # noqa: E402
from slayersmacro.detect import Calibration, Region  # noqa: E402
from slayersmacro.window import find_window  # noqa: E402

log.setup()

# name -> what to tell the user to draw the box around
PROMPTS = {
    "fishing": [
        ("fishing_bar", "drag a box around the WHOLE fishing minigame track"),
        ("fishing_ball", "drag a tight box around the moving white marker"),
        ("reel_prompt", "drag a box around the 'press to reel' prompt or progress bar"),
    ],
    "boss": [
        ("boss_health", "drag a box around the BOSS health bar"),
        ("player_health", "drag a box around YOUR health bar"),
        ("target_marker", "drag a box around the crosshair or target indicator"),
    ],
    "combat": [
        ("player_health", "drag a box around YOUR health bar"),
        ("target_health", "drag a box around the enemy's health bar, if shown"),
    ],
    "quest": [
        ("quest_accept", "drag a box around the ACCEPT button"),
        ("quest_turnin", "drag a box around the TURN IN button"),
        ("progress_bar", "drag a box around the quest progress bar"),
    ],
}

_state = {"img": None, "win": None, "drag": None, "made": []}
WIN_NAME = "calibrate"


def mouse_cb(event, x, y, flags, _param):
    if event == cv2.EVENT_LBUTTONDOWN:
        _state["drag"] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and _state["drag"]:
        x0, y0 = _state["drag"]
        _state["drag"] = None
        if abs(x - x0) > 6 and abs(y - y0) > 6:
            rx, ry = min(x0, x), min(y0, y)
            rw, rh = abs(x - x0), abs(y - y0)
            _state["made"].append((rx, ry, rw, rh))


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "fishing"
    if which not in PROMPTS:
        print(f"unknown target '{which}'. choose from: {', '.join(PROMPTS)}")
        return 2

    win = find_window("Roblox")
    if win is None:
        print("No Roblox window found. Open the game first.")
        return 1

    cap = Capture()
    calib = Calibration.load()
    if not calib.anchor_w or not calib.anchor_h:
        calib.anchor_w, calib.anchor_h = win.width, win.height

    print(f"attached to '{win.title}'  {win.width}x{win.height}")
    print("For each box: drag it on the preview, then press ENTER in this")
    print("terminal to accept it. Press Q in the preview to save and quit.\n")

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN_NAME, mouse_cb)

    for name, prompt in PROMPTS[which]:
        print(f"--- {name}: {prompt}")
        scale = 0.55
        while True:
            frame = cap.grab_window(win)
            if frame is None:
                print("lost the window")
                return 1
            disp = cv2.resize(frame, None, fx=scale, fy=scale)
            for made in _state["made"]:
                mx, my, mw, mh = [int(v * scale) for v in made]
                cv2.rectangle(disp, (mx, my), (mx + mw, my + mh), (0, 255, 0), 2)
            cv2.putText(disp, f"{name} - drag a box",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)
            cv2.imshow(WIN_NAME, disp)
            k = cv2.waitKey(30) & 0xFF

            if k in (ord("q"), 27):
                cv2.destroyAllWindows()
                cap.close()
                print("cancelled, nothing saved")
                return 1
            if k in (10, 13):  # enter
                if not _state["made"]:
                    print("  no box drawn - try again")
                    continue
                rx, ry, rw, rh = _state["made"][-1]
                calib.regions[name] = Region(name, rx, ry, rw, rh)
                calib.anchor_w, calib.anchor_h = win.width, win.height
                calib.save()
                print(f"  saved {name}: {rw}x{rh} at ({rx},{ry})")
                _state["made"] = []
                break

    cv2.destroyAllWindows()
    cap.close()
    print(f"\ndone. {len(PROMPTS[which])} regions saved to config/calibration.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())