"""Mouse input for the reconnect tool, kept separate from the macro tool.

Same reasoning as the macro tool: SendInput, batched down+up in one call.
Different emphasis though - this clicks UI rather than pressing keys in a
rhythm, so what matters most is that the click lands and that we can aim
at a specific absolute point.
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _INPUTunion(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTunion)]


user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
user32.SendInput.restype = wintypes.UINT

_batch = (INPUT * 2)()
_batch_ptr = ctypes.cast(_batch, ctypes.POINTER(INPUT))


def virtual_screen() -> tuple[int, int, int, int]:
    """(left, top, width, height) of the whole desktop, all monitors."""
    SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN = 76, 77
    SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN = 78, 79
    return (user32.GetSystemMetrics(SM_XVIRTUALSCREEN),
            user32.GetSystemMetrics(SM_YVIRTUALSCREEN),
            user32.GetSystemMetrics(SM_CXVIRTUALSCREEN),
            user32.GetSystemMetrics(SM_CYVIRTUALSCREEN))


def move_abs(x: int, y: int) -> None:
    """Move to absolute desktop coordinates.

    SendInput wants 0..65535 normalised over the VIRTUAL screen, not the
    primary one. Getting that wrong on a multi-monitor setup means every
    click is offset, which on a multi-monitor laptop is exactly the case
    that matters.
    """
    vx, vy, vw, vh = virtual_screen()
    ev = INPUT(type=INPUT_MOUSE)
    ev.u.mi.dx = int((x - vx) * 65535 / max(1, vw - 1))
    ev.u.mi.dy = int((y - vy) * 65535 / max(1, vh - 1))
    ev.u.mi.dwFlags = (MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE
                       | MOUSEEVENTF_VIRTUALDESK)
    _batch[0] = ev
    user32.SendInput(1, _batch_ptr, ctypes.sizeof(INPUT))


def left_click(x: int, y: int, settle_ms: float = 40.0) -> None:
    """Move then click, in one batched pair where possible.

    Roblox registers a click from the button events alone, but moving
    first means the UI's hover state has settled by the time the click
    lands, which matters for a dialog that only enables a button on hover.
    """
    move_abs(x, y)
    time.sleep(max(0.0, settle_ms) / 1000.0)
    down = INPUT(type=INPUT_MOUSE)
    down.u.mi.dwFlags = MOUSEEVENTF_LEFTDOWN
    up = INPUT(type=INPUT_MOUSE)
    up.u.mi.dwFlags = MOUSEEVENTF_LEFTUP
    _batch[0] = down
    _batch[1] = up
    sent = user32.SendInput(2, _batch_ptr, ctypes.sizeof(INPUT))
    if sent != 2:
        raise OSError(ctypes.get_last_error(), "SendInput click rejected")


def cursor_pos() -> tuple[int, int]:
    pt = wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


def bring_to_front(hwnd: int) -> bool:
    """Focus a window before clicking it.

    Necessary because SendInput delivers to the FOREGROUND window's input
    queue. Clicking at coordinates over a background window would instead
    click whatever is on top - the same class of bug the macro tool's
    guard exists to prevent, and it is worth doing properly here too.
    """
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, 9)          # SW_RESTORE
    ok = bool(user32.SetForegroundWindow(hwnd))
    if not ok:
        # SetForegroundWindow is refused when the caller is not foreground.
        # Flash instead so the user can bring it up, and say so.
        user32.FlashWindow(hwnd, True)
    return ok
