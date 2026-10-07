"""Synthetic keyboard/mouse input via direct SendInput.

SendInput puts events into the same queue a physical keyboard uses, so the
game cannot tell the difference. That is why this is used instead of
PostMessage, which most modern engines ignore entirely.

Also supports scancodes (KEYEVENTF_SCANCODE). Some engines read raw input
and only look at scancodes, so if virtual keys are ignored, scancodes are
the fallback that works.
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

import pydirectinput

user32 = ctypes.WinDLL("user32", use_last_error=True)

# --- Windows constants -------------------------------------------------
INPUT_KEYBOARD = 1
INPUT_MOUSE = 0
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040

# --- Key tables -------------------------------------------------------
# Scan codes, set 1. Needed for engines that read raw input.
SCANCODES = {
    "escape": 0x01, "1": 0x02, "2": 0x03, "3": 0x04, "4": 0x05, "5": 0x06,
    "6": 0x07, "7": 0x08, "8": 0x09, "9": 0x0A, "0": 0x0B, "-": 0x0C,
    "=": 0x0D, "backspace": 0x0E, "tab": 0x0F,
    "q": 0x10, "w": 0x11, "e": 0x12, "r": 0x13, "t": 0x14, "y": 0x15,
    "u": 0x16, "i": 0x17, "o": 0x18, "p": 0x19, "[": 0x1A, "]": 0x1B,
    "enter": 0x1C, "lctrl": 0x1D, "space": 0x39,
    "a": 0x1E, "s": 0x1F, "d": 0x20, "f": 0x21, "g": 0x22, "h": 0x23,
    "j": 0x24, "k": 0x25, "l": 0x26, ";": 0x27, "'": 0x28, "`": 0x29,
    "lshift": 0x2A, "\\": 0x2B,
    "z": 0x2C, "x": 0x2D, "c": 0x2E, "v": 0x2F, "b": 0x30, "n": 0x31,
    "m": 0x32, ",": 0x33, ".": 0x34, "/": 0x35, "rshift": 0x36,
    "up": 0x48, "left": 0x4B, "right": 0x4D, "down": 0x50,
}

VK = {
    "escape": 0x1B, "tab": 0x09, "enter": 0x0D, "lctrl": 0x11, "space": 0x20,
    "lshift": 0x10, "backspace": 0x08, "capslock": 0x14,
    "1": 0x31, "2": 0x32, "3": 0x33, "4": 0x34, "5": 0x35,
    "6": 0x36, "7": 0x37, "8": 0x38, "9": 0x39, "0": 0x30,
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
}
for _i in range(1, 13):
    VK[f"f{_i}"] = 0x6F + _i
# F13-F24 are rarely used by games and nothing in a typical UI binds them,
# which makes them the safe choice for benchmarking without side effects.
VK["f24"] = 0x87
for _c in "abcdefghijklmnopqrstuvwxyz":
    VK[_c] = ord(_c.upper())

# Scan codes for keys only present in the table above.
for _k, _v in list(SCANCODES.items()):
    VK.setdefault(_k, _v)

EXTENDED = {"up", "down", "left", "right", "lctrl", "rctrl", "lalt", "ralt",
            "insert", "delete", "home", "end", "pageup", "pagedown"}


# --- Structures -------------------------------------------------------
ULONG_PTR = wintypes.WPARAM


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class _INPUTunion(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTunion)]


user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
user32.SendInput.restype = wintypes.UINT

BATCH_SIZE = 8
_batch = (INPUT * BATCH_SIZE)()
# Cast once. ctypes cannot pass byref() of an array where a POINTER(INPUT)
# is expected, and doing the cast per call would add overhead we are trying
# to avoid.
_batch_ptr = ctypes.cast(_batch, ctypes.POINTER(INPUT))


def _norm(key: str) -> str:
    return key.strip().lower()


def _key_event(key: str, up: bool, use_scancode: bool) -> INPUT:
    key = _norm(key)
    if key not in VK and key not in SCANCODES:
        # Silently sending vk=0 would be a no-op nobody notices until the
        # macro just does not work, so fail loudly instead.
        raise KeyError(f"unknown key: {key!r}")
    ev = INPUT(type=INPUT_KEYBOARD)
    flags = KEYEVENTF_KEYUP if up else 0
    if use_scancode and key in SCANCODES:
        ev.u.ki.wScan = SCANCODES[key]
        flags |= KEYEVENTF_SCANCODE
    else:
        ev.u.ki.wVk = VK.get(key, 0)
    if key in EXTENDED:
        flags |= KEYEVENTF_EXTENDEDKEY
    ev.u.ki.dwFlags = flags
    return ev


def press_release(key: str, hold_ms: float = 18.0, use_scancode: bool = False) -> None:
    """Press and release a key in one batched SendInput call.

    Both events go out together, so the key is genuinely down for hold_ms
    without costing two API round trips.
    """
    down = _key_event(key, up=False, use_scancode=use_scancode)
    up = _key_event(key, up=True, use_scancode=use_scancode)
    _batch[0] = down
    _batch[1] = up
    sent = user32.SendInput(2, _batch_ptr, ctypes.sizeof(INPUT))
    if sent != 2:
        raise ctypes.WinError(ctypes.get_last_error())
    if hold_ms > 0:
        # Busy-wait for very short holds. Sleep has ~1-15ms granularity
        # on Windows, which would make short taps land unpredictably.
        if hold_ms <= 2.0:
            end = time.perf_counter_ns() + int(hold_ms * 1_000_000)
            while time.perf_counter_ns() < end:
                pass
        else:
            time.sleep(hold_ms / 1000.0)


def down(key: str, use_scancode: bool = False) -> None:
    _batch[0] = _key_event(key, up=False, use_scancode=use_scancode)
    user32.SendInput(1, _batch_ptr, ctypes.sizeof(INPUT))


def up(key: str, use_scancode: bool = False) -> None:
    _batch[0] = _key_event(key, up=True, use_scancode=use_scancode)
    user32.SendInput(1, _batch_ptr, ctypes.sizeof(INPUT))


def tap(key: str, use_scancode: bool = False) -> None:
    """Press and release with no hold. Fastest possible single key press."""
    press_release(key, hold_ms=0.0, use_scancode=use_scancode)


# --- Mouse ------------------------------------------------------------
def move_relative(dx: int, dy: int) -> None:
    ev = INPUT(type=INPUT_MOUSE)
    ev.u.mi.dx = dx
    ev.u.mi.dy = dy
    ev.u.mi.dwFlags = MOUSEEVENTF_MOVE
    _batch[0] = ev
    user32.SendInput(1, _batch_ptr, ctypes.sizeof(INPUT))


def move_absolute(x: int, y: int, screen_w: int, screen_h: int) -> None:
    """Move to absolute screen coords. SendInput wants 0..65535 normalised."""
    ev = INPUT(type=INPUT_MOUSE)
    ev.u.mi.dx = int(x * 65535 / max(1, screen_w - 1))
    ev.u.mi.dy = int(y * 65535 / max(1, screen_h - 1))
    ev.u.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE
    _batch[0] = ev
    user32.SendInput(1, _batch_ptr, ctypes.sizeof(INPUT))


def click(button: str = "left", hold_ms: float = 0.0) -> None:
    """Press and release a mouse button in one batched call."""
    down_map = {"left": MOUSEEVENTF_LEFTDOWN, "right": MOUSEEVENTF_RIGHTDOWN,
                "middle": MOUSEEVENTF_MIDDLEDOWN}
    up_map = {"left": MOUSEEVENTF_LEFTUP, "right": MOUSEEVENTF_RIGHTUP,
              "middle": MOUSEEVENTF_MIDDLEUP}
    b = button.lower()
    _batch[0] = INPUT(type=INPUT_MOUSE)
    _batch[0].u.mi.dwFlags = down_map[b]
    _batch[1] = INPUT(type=INPUT_MOUSE)
    _batch[1].u.mi.dwFlags = up_map[b]
    user32.SendInput(2, _batch_ptr, ctypes.sizeof(INPUT))
    if hold_ms > 0:
        time.sleep(hold_ms / 1000.0)


BUTTON_DOWN = {"left": MOUSEEVENTF_LEFTDOWN, "right": MOUSEEVENTF_RIGHTDOWN,
               "middle": MOUSEEVENTF_MIDDLEDOWN}
BUTTON_UP = {"left": MOUSEEVENTF_LEFTUP, "right": MOUSEEVENTF_RIGHTUP,
             "middle": MOUSEEVENTF_MIDDLEUP}


def button_down(button: str = "left") -> None:
    """Press a mouse button and leave it down until button_up().

    Needed for hold-to-move style inputs, where a batched down+up would be
    too short for the game to register at all.
    """
    _batch[0] = INPUT(type=INPUT_MOUSE)
    _batch[0].u.mi.dwFlags = BUTTON_DOWN[button.lower()]
    user32.SendInput(1, _batch_ptr, ctypes.sizeof(INPUT))


def button_up(button: str = "left") -> None:
    _batch[0] = INPUT(type=INPUT_MOUSE)
    _batch[0].u.mi.dwFlags = BUTTON_UP[button.lower()]
    user32.SendInput(1, _batch_ptr, ctypes.sizeof(INPUT))


# --- pydirectinput compatibility shim ---------------------------------
def pd_tap(key: str) -> None:
    """Same as tap() but through pydirectinput. Used by the benchmark to
    compare the two on real hardware."""
    pydirectinput.press(key)
    pydirectinput.release(key)


def cursor_pos() -> tuple[int, int]:
    pt = wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y