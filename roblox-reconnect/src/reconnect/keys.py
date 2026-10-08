"""Keyboard input for the reconnect tool: Escape, and the launcher hotkey."""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002

VK_ESCAPE = 0x1B


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _INPUTunion(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTunion)]


user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
user32.SendInput.restype = wintypes.UINT

_batch = (INPUT * 2)()
_batch_ptr = ctypes.cast(_batch, ctypes.POINTER(INPUT))


def tap(vk: int, hold_ms: float = 25.0) -> None:
    """Press and release a virtual key in one batched call."""
    down = INPUT(type=INPUT_KEYBOARD)
    down.u.ki.wVk = vk
    up = INPUT(type=INPUT_KEYBOARD)
    up.u.ki.wVk = vk
    up.u.ki.dwFlags = KEYEVENTF_KEYUP
    _batch[0] = down
    _batch[1] = up
    sent = user32.SendInput(2, _batch_ptr, ctypes.sizeof(INPUT))
    if sent != 2:
        raise OSError(ctypes.get_last_error(), "SendInput key rejected")
    if hold_ms > 0:
        time.sleep(hold_ms / 1000.0)


def press_escape(hold_ms: float = 25.0) -> None:
    tap(VK_ESCAPE, hold_ms)
