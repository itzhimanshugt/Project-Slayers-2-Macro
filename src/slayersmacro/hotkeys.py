"""Global hotkeys with a dedicated hook thread.

Runs its own message loop on its own thread. That matters for the panic
key: it has to fire even when the main loop is busy or wedged, otherwise
"emergency stop" is not really an emergency stop.
"""

from __future__ import annotations

import ctypes
import threading
from ctypes import wintypes
from typing import Callable

user32 = ctypes.WinDLL("user32", use_last_error=True)

WH_KEYBOARD_LL = 13
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_QUIT = 0x0012

VK_CODES = {
    "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73, "f5": 0x74, "f6": 0x75,
    "f7": 0x76, "f8": 0x77, "f9": 0x78, "f10": 0x79, "f11": 0x7A, "f12": 0x7B,
}


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode", wintypes.DWORD),
        ("scanCode", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
    ]


HOOKPROC = ctypes.WINFUNCTYPE(
    ctypes.c_longlong, ctypes.c_int, wintypes.WPARAM, ctypes.POINTER(KBDLLHOOKSTRUCT)
)


class HotkeyManager:
    """Fires callbacks on key-down. Callbacks run on the hook thread, so
    they should be fast and must not block."""

    def __init__(self) -> None:
        self._callbacks: dict[int, Callable[[], None]] = {}
        self._thread: threading.Thread | None = None
        self._thread_id: int | None = None
        self._hook_proc = None  # must stay referenced or Windows frees it
        self._hook_handle = None
        self._running = threading.Event()
        self._ready = threading.Event()
        self._error: BaseException | None = None

    def register(self, key: str, callback: Callable[[], None]) -> None:
        vk = VK_CODES.get(key.strip().lower())
        if vk is None:
            raise ValueError(f"unsupported hotkey: {key}")
        self._callbacks[vk] = callback

    def start(self, timeout: float = 2.0) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="hotkeys", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise RuntimeError(f"hotkey hook failed to start: {self._error}")
        if self._error is not None:
            raise RuntimeError(f"hotkey hook failed to start: {self._error}")

    def stop(self) -> None:
        if self._thread_id is not None:
            user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None

    def _run(self) -> None:
        try:
            self._thread_id = ctypes.windll.kernel32.GetCurrentThreadId()
            self._install()
            self._running.set()
            self._ready.set()
            msg = wintypes.MSG()
            while self._running.is_set():
                # -1 = wait on all messages. Must pump, or the LL hook
                # never fires.
                got = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
                if got in (0, -1):
                    break
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
        except BaseException as exc:  # noqa: BLE001
            self._error = exc
            self._ready.set()
        finally:
            self._cleanup()

    def _install(self) -> None:
        def hook(code, wparam, lparam):
            if code >= 0 and wparam == WM_KEYDOWN:
                kbd = ctypes.cast(lparam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
                cb = self._callbacks.get(int(kbd.vkCode))
                if cb is not None:
                    try:
                        cb()
                    except Exception:
                        # A broken callback must not kill the hook thread;
                        # that would take the panic key down with it.
                        pass
            return user32.CallNextHookEx(None, code, wparam, lparam)

        self._hook_proc = HOOKPROC(hook)
        hmod = ctypes.windll.kernel32.GetModuleHandleW(None)
        self._hook_handle = user32.SetWindowsHookExW(
            WH_KEYBOARD_LL, self._hook_proc, hmod, 0
        )
        if not self._hook_handle:
            raise ctypes.WinError(ctypes.get_last_error())

    def _cleanup(self) -> None:
        if self._hook_handle:
            try:
                user32.UnhookWindowsHookEx(self._hook_handle)
            except Exception:
                pass
            self._hook_handle = None