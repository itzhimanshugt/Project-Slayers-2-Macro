"""Roblox window tracking.

Everything screen-based depends on knowing where the Roblox window is, in
real screen pixels. That changes when the user moves, resizes, or goes
fullscreen, so it is re-read every frame rather than cached.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass

user32 = ctypes.WinDLL("user32", use_last_error=True)

_dpi_ready = False


def enable_dpi_awareness() -> None:
    """Opt into per-monitor DPI awareness.

    Must happen before any coordinate call. Without it Windows hands back
    scaled coordinates on a 125%/150% display and every screen region is
    offset from where it should be. The 2560x1600 Legion panel is very
    likely scaled, so this is not optional.
    """
    global _dpi_ready
    if _dpi_ready:
        return
    try:  # Windows 10 1703+
        ctypes.windll.user32.SetProcessDpiAwarenessContext(
            ctypes.c_void_p(-4)  # PER_MONITOR_AWARE_V2
        )
    except (AttributeError, OSError):
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            user32.SetProcessDPIAware()
    _dpi_ready = True


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str
    left: int
    top: int
    width: int
    height: int

    @property
    def rect(self) -> tuple[int, int, int, int]:
        """Screen rect as (left, top, right, bottom)."""
        return self.left, self.top, self.left + self.width, self.top + self.height

    @property
    def center(self) -> tuple[int, int]:
        return self.left + self.width // 2, self.top + self.height // 2

    def contains(self, x: int, y: int) -> bool:
        return self.left <= x < self.left + self.width and \
               self.top <= y < self.top + self.height

    def to_local(self, x: int, y: int) -> tuple[int, int]:
        """Screen point -> point relative to the window's top-left."""
        return x - self.left, y - self.top


def _title_of(hwnd: int) -> str:
    length = user32.GetWindowTextLengthW(hwnd)
    if length == 0:
        return ""
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


def find_window(title_substr: str = "Roblox") -> WindowInfo | None:
    """Locate a top-level window whose title contains title_substr.

    Matches on title because Fishstrap titles the game window like any
    other Roblox client; the class name is not stable enough to rely on.
    """
    enable_dpi_awareness()
    needle = title_substr.lower()
    found: list[WindowInfo] = []

    WNDENUMPROC = ctypes.WINFUNCTYPE(
        wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
    )

    def callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        title = _title_of(hwnd)
        if needle in title.lower():
            rect = wintypes.RECT()
            # Client rect excludes the title bar and borders, which is what
            # we want: it is the part the game actually renders into.
            if user32.GetClientRect(hwnd, ctypes.byref(rect)):
                pt = wintypes.POINT(0, 0)
                # ClientToScreen gives us the client area's origin on screen.
                user32.ClientToScreen(hwnd, ctypes.byref(pt))
                found.append(WindowInfo(
                    hwnd=hwnd,
                    title=title,
                    left=pt.x,
                    top=pt.y,
                    width=rect.right - rect.left,
                    height=rect.bottom - rect.top,
                ))
        return True

    user32.EnumWindows(WNDENUMPROC(callback), 0)
    # Biggest match wins: Roblox can leave small helper windows around.
    return max(found, key=lambda w: w.width * w.height, default=None)


def is_foreground(hwnd: int) -> bool:
    return user32.GetForegroundWindow() == hwnd


def bring_to_front(hwnd: int) -> bool:
    return bool(user32.SetForegroundWindow(hwnd))


def virtual_screen_size() -> tuple[int, int]:
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)  # 0=width 1=height