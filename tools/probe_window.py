"""Capture a specific window's own surface, ignoring whatever covers it.

The ordinary capture path (mss, GDI, DXGI) reads whatever is COMPOSITED at a
screen rectangle. If Chrome sits on top of Roblox, a normal screenshot of
"the Roblox rect" returns Chrome. That is not a bug - it is what every
screen-capture API does - and it is the reason guard.py exists.

Windows Graphics Capture is different: the window renders its own surface
into a texture, so an occluded or background window still produces correct
frames. That is the only way to see the game while working in a browser,
which is exactly the normal situation for this tool.

Two handler functions are required by the library and identified BY NAME, so
they must be module-level with these exact names.

Usage:
    python tools/probe_window.py
    python tools/probe_window.py --region 0,0,400,20
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from slayersmacro import log  # noqa: E402
from slayersmacro.window import find_window, is_foreground  # noqa: E402

log.setup()
lg = log.get("probe")

# windows_capture dispatches on the handler's __name__, so these two names
# are load-bearing. Renaming them silently disables the capture.
_state: dict = {"frame": None, "frames": 0, "control": None}
_frame_ready = threading.Event()


def _to_bgr(obj) -> np.ndarray:
    """WGC Frame -> an OpenCV BGR array.

    Established empirically against windows-capture 1.x. Two traps here, both
    of which cost a round each:

      - the callback receives (Frame, InternalCaptureControl), and the
        Frame is NOT an array
      - Frame.convert_to_bgr() returns ANOTHER Frame, not pixels. It is a
        description of the buffer, and numpy ops on it fail with
        "unsupported operand type(s) for /: 'Frame' and 'int'".

    The pixels are Frame.frame_buffer, which is already a real ndarray.
    """
    if isinstance(obj, np.ndarray):
        return obj
    buf = getattr(obj, "frame_buffer", None)
    if buf is None:
        raise TypeError(
            f"{type(obj).__name__} has no frame_buffer; "
            f"available: {[n for n in dir(obj) if not n.startswith('_')]}")
    arr = np.asarray(buf)
    # The buffer is reused between frames, so copy before the next callback
    # overwrites it. Without this the caller holds a view of moving memory.
    return arr.copy()


def on_frame_arrived(frame, control=None) -> None:  # noqa: N802
    # Signature is (Frame, InternalCaptureControl). Keep the control: it is
    # the only way to end the session, since WindowsCapture itself exposes
    # no stop().
    _state["frame"] = frame
    _state["control"] = control
    _state["frames"] += 1
    _frame_ready.set()


def on_closed(_frame=None) -> None:  # noqa: N802
    lg.warning("WGC session closed by the system")


def top_window_at(x: int, y: int) -> tuple[int, str, int]:
    """What is actually on screen at this point. Proves occlusion."""
    u = ctypes.WinDLL("user32")
    hwnd = u.WindowFromPoint(ctypes.wintypes.POINT(int(x), int(y)))
    n = u.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    u.GetWindowTextW(hwnd, buf, n + 1)
    pid = ctypes.wintypes.DWORD()
    u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return hwnd, buf.value, pid.value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default="Roblox")
    ap.add_argument("--out", default="")
    ap.add_argument("--region", default="", help="x,y,w,h window-local")
    ap.add_argument("--count", type=int, default=30,
                    help="frames to collect, for a latency figure")
    args = ap.parse_args()

    win = find_window(args.title)
    if win is None:
        print(f"No window matching {args.title!r}.")
        return 1

    top_hwnd, top_title, _ = top_window_at(*win.center)
    print(f"target : {win.title!r} {win.width}x{win.height} at "
          f"({win.left},{win.top})  foreground={is_foreground(win.hwnd)}")
    print(f"on top : {top_title!r}  occluded={top_hwnd != win.hwnd}")
    if top_hwnd != win.hwnd:
        print("  -> a normal screen grab of that rect would return the "
              "covering window.\n     WGC is required to see the real one.\n")

    try:
        from windows_capture import WindowsCapture
    except ImportError:
        print("python -m pip install windows-capture")
        return 2

    control = WindowsCapture(
        cursor_capture=False,
        draw_border=False,
        minimum_update_interval=0,
        window_hwnd=win.hwnd,
    )
    control.event(on_frame_arrived)
    control.event(on_closed)
    # start() runs the capture loop ON THE CALLING THREAD and never returns.
    # start_free_threaded() is the one that returns, which is what makes
    # this usable from a script at all.
    control.start_free_threaded()

    if not _frame_ready.wait(25):
        print("no frame within 25s. The window may be minimized, or the")
        print("compositor may not be producing frames for an occluded window.")
        return 3

    # Collect a few frames to measure the real per-frame cost.
    t0 = time.perf_counter()
    deadline = t0 + 2.0
    while _state["frames"] < args.count and time.perf_counter() < deadline:
        time.sleep(0.001)
    elapsed = time.perf_counter() - t0
    fps = _state["frames"] / elapsed if elapsed else 0

    frame = _state["frame"]
    img = _to_bgr(frame)
    print(f"frames: {_state['frames']} in {elapsed:.2f}s -> {fps:.1f} fps")
    print(f"image : {img.shape} {img.dtype} min={img.min()} max={img.max()} "
          f"mean={float(img.mean()):.1f}")

    h, w = img.shape[:2]
    if (h, w) != (win.height, win.width):
        print(f"note: frame is {w}x{h}, window rect is "
              f"{win.width}x{win.height} - DPI scaling is active")

    out_img = img
    if args.region:
        x, y, rw, rh = (int(v) for v in args.region.split(","))
        out_img = img[max(0, y):max(0, y + rh), max(0, x):max(0, x + rw)]
        print(f"cropped {args.region} -> {out_img.shape}")

    out = args.out or str(Path("logs/shots") / f"wgc_{int(time.time())}.png")
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import cv2
    bgr = out_img[:, :, :3] if out_img.ndim == 3 else \
        np.stack([out_img] * 3, -1)
    cv2.imwrite(str(out_path), bgr)
    print(f"saved {out_path}")

    # WindowsCapture has no .stop(); the control object carried in the
    # callback does. Holding it is what lets us shut the session down.
    ctrl = _state.get("control")
    if ctrl is not None:
        ctrl.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())