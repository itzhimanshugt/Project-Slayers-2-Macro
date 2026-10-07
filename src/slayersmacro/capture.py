"""Screen capture, region only.

Capturing 2560x1600 every loop would cost far more than the loop itself,
so everything here grabs just the rectangle we need. The mss instance is
kept open because creating one per frame dominates the cost.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import mss
import numpy as np


class Capture:
    """Thread-safe region capture returning OpenCV BGR arrays."""

    def __init__(self) -> None:
        self._sct = mss.mss()
        self._lock = threading.Lock()
        self._monitor = self._sct.monitors[1]
        self.last_capture_ms = 0.0

    @property
    def virtual_screen(self) -> dict:
        with self._lock:
            return self._sct.monitors[0]

    def grab(self, left: int, top: int, width: int, height: int) -> np.ndarray | None:
        """Grab a region. Returns a fresh BGR array, or None if off-screen.

        The region is clipped to the virtual desktop, because a window on a
        second monitor can easily ask for coordinates that do not exist on
        this one.
        """
        region = {
            "left": int(left),
            "top": int(top),
            "width": int(width),
            "height": int(height),
        }
        with self._lock:
            mon = self._sct.monitors[0]
            l = max(region["left"], mon["left"])
            t = max(region["top"], mon["top"])
            r = min(region["left"] + region["width"], mon["left"] + mon["width"])
            b = min(region["top"] + region["height"], mon["top"] + mon["height"])
            if r <= l or b <= t:
                return None
            shot = self._sct.grab({"left": l, "top": t, "width": r - l, "height": b - t})

        # mss returns BGRA. Drop alpha, and copy out of mss' own buffer.
        frame = np.asarray(shot)[:, :, :3].copy()
        self.last_capture_ms = 0.0
        return frame

    def grab_window(self, win) -> np.ndarray | None:
        """Grab the whole client area of a WindowInfo."""
        return self.grab(win.left, win.top, win.width, win.height)

    def save(self, frame: np.ndarray, name: str) -> Path:
        out_dir = Path(__file__).resolve().parent.parent.parent / "logs" / "shots"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%H%M%S")
        path = out_dir / f"{stamp}_{name}.png"
        cv2.imwrite(str(path), frame)
        return path

    def close(self) -> None:
        try:
            self._sct.close()
        except Exception:
            pass


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute pixel difference. Used for change detection."""
    if a is None or b is None or a.shape != b.shape:
        return 0.0
    return float(cv2.absdiff(a, b).mean())