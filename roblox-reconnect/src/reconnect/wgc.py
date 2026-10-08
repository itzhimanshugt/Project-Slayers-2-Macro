"""Capture layer for the reconnect tool.

Two backends, because they fail in different situations and the whole
point of this tool is to keep working when something has gone wrong.

  WgcCapture - Windows Graphics Capture. Asks the window to render its own
      surface, so it sees through occlusion AND keeps working when the
      display is off. That last property is the reason it exists here: a
      closed-lid laptop still runs the game, and the point of this tool is
      to recover from a disconnect nobody is present to see.

  MssCapture - ordinary screen-region capture. Faster for small regions,
      but returns whatever is composited at those coordinates and stops
      producing useful frames when the display sleeps.

WGC is the default and the fallback is only used if WGC cannot start.

Everything here is read-only. This module never sends input.
"""

from __future__ import annotations

import ctypes
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .window import WindowInfo, find_roblox, is_foreground


@dataclass
class CaptureStats:
    frames: int = 0
    last_frame_at: float = 0.0
    stale_ms: float = 0.0
    backend: str = ""
    note: str = ""


class CaptureBase:
    """Interface both backends satisfy."""

    backend = "none"

    def __init__(self, win: WindowInfo) -> None:
        self.win = win
        self.stats = CaptureStats(backend=self.backend)
        self._closed = False

    def grab(self) -> np.ndarray | None:
        raise NotImplementedError

    def close(self) -> None:
        self._closed = True

    def stale_ms(self) -> float:
        if not self.stats.last_frame_at:
            return float("inf")
        return (time.perf_counter() - self.stats.last_frame_at) * 1000.0

    def healthy(self, max_stale_ms: float = 3000.0) -> bool:
        return self.stats.frames > 0 and self.stale_ms() < max_stale_ms


# --- Windows Graphics Capture -----------------------------------------

class _WgcCallbacks:
    """Handler names are load-bearing.

    windows-capture dispatches on the handler's __name__ and raises at
    start() if either is missing, so these cannot be renamed or made
    nested.
    """

    def __init__(self) -> None:
        self.frame = None
        self.control = None
        self.closed = False
        self.ready = threading.Event()

    def on_frame_arrived(self, frame, control=None) -> None:  # noqa: N802
        self.frame = frame
        self.control = control
        self.ready.set()

    def on_closed(self, _frame=None) -> None:  # noqa: N802
        self.closed = True


class WgcCapture(CaptureBase):
    """Capture through Windows Graphics Capture.

    Contract established empirically against windows-capture 1.x, and each
    point cost a round to find:
      - start() blocks on the calling thread; start_free_threaded() does not
      - the callback receives (Frame, InternalCaptureControl)
      - Frame.convert_to_bgr() returns ANOTHER Frame, not pixels. The
        pixels are Frame.frame_buffer, already an ndarray.
      - frame_buffer is reused between frames, so it must be copied.
    """

    backend = "wgc"

    def __init__(self, win: WindowInfo, min_interval_ms: int = 0) -> None:
        super().__init__(win)
        try:
            from windows_capture import WindowsCapture
        except ImportError as exc:
            raise RuntimeError(
                "windows-capture is required for the WGC backend: "
                "python -m pip install windows-capture") from exc

        self._cb = _WgcCallbacks()
        self._control = WindowsCapture(
            cursor_capture=False,
            draw_border=False,
            minimum_update_interval=min_interval_ms,
            window_hwnd=win.hwnd,
        )
        # The library matches on __name__, so pass the bound methods.
        self._control.event(self._cb.on_frame_arrived)
        self._control.event(self._cb.on_closed)
        self._control.start_free_threaded()

        if not self._cb.ready.wait(15.0):
            self.close()
            raise RuntimeError("WGC produced no frame within 15s")

    def grab(self) -> np.ndarray | None:
        if self._cb.closed:
            self.stats.note = "WGC session closed by the system"
            return None
        frame = self._cb.frame
        if frame is None:
            return None
        buf = getattr(frame, "frame_buffer", None)
        if buf is None:
            return None
        arr = np.array(buf, copy=True)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            arr = arr[:, :, :3]
        self.stats.frames += 1
        self.stats.last_frame_at = time.perf_counter()
        return arr

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        ctrl = self._cb.control
        if ctrl is None:
            # WindowsCapture itself exposes no stop(); only the control
            # object handed to the callback does.
            ctrl = getattr(self, "_control", None)
        if ctrl is not None and hasattr(ctrl, "stop"):
            try:
                ctrl.stop()
            except Exception:
                pass


# --- mss fallback -------------------------------------------------------

class MssCapture(CaptureBase):
    """Ordinary screen-region capture. The fallback, not the default."""

    backend = "mss"

    def __init__(self, win: WindowInfo) -> None:
        super().__init__(win)
        import mss
        self._sct = mss.mss()

    def grab(self) -> np.ndarray | None:
        with self._sct:
            shot = self._sct.grab({
                "left": self.win.left, "top": self.win.top,
                "width": self.win.width, "height": self.win.height,
            })
        arr = np.asarray(shot)[:, :, :3].copy()
        self.stats.frames += 1
        self.stats.last_frame_at = time.perf_counter()
        return arr

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._sct.close()
        except Exception:
            pass


def open_capture(prefer_wgc: bool = True,
                 title: str = "Roblox") -> CaptureBase:
    """Best available capture for the Roblox window.

    WGC first, always. It is the only backend that works when something
    else is on top, and the only one that keeps working with the display
    off - which is precisely the situation this tool exists for.
    """
    win = find_roblox()
    if win is None:
        raise RuntimeError(f"no Roblox window found (looking for {title!r})")
    errors = []
    if prefer_wgc:
        try:
            return WgcCapture(win)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"wgc: {exc}")
    try:
        return MssCapture(win)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"mss: {exc}")
    raise RuntimeError("no capture backend available: " + "; ".join(errors))
