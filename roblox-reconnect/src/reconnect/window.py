"""Window discovery for Roblox, including the launcher and main menu.

Roblox presents several different windows depending on state, and this tool
has to survive all of them:

  - the game client, in-game
  - the game client showing a connection-error dialog
  - the Roblox launcher / main menu, when the client exited
  - the client "white screen" or black screen during load

HOW THE WINDOW IS FOUND

Measured on this machine against a live RobloxPlayerBeta process, the game
window's class name is "WINDOWSCLIENT" and its title is "Roblox" - NOT
"RobloxPlayerBeta", which is the process name and the intuitive thing to
guess. An earlier version of this file matched on a hardcoded list of
guessed class names and returned no window at all against a running game.

So discovery is by PROCESS NAME, not by class name: enumerate the pids of
processes called RobloxPlayerBeta, then take the top-level window owned by
one of them. That does not depend on a name Roblox could change, and it
cannot accidentally match an unrelated window.

Class and title are kept only as a fallback, for the case where the process
enumeration is unavailable.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

#: Process names that host the in-game client. Matched case-insensitively.
GAME_PROCESSES = ("RobloxPlayerBeta", "RobloxPlayer")
#: Process names for the standalone launcher.
LAUNCHER_PROCESSES = ("RobloxLauncher",)

# Fallback class names. Only used if process enumeration fails. The first
# two are guesses that did NOT match anything measured on this machine; the
# third is the real one. Listed in that order so the measured value is what
# normally wins even if the others are tried first.
GAME_CLASSES = ("RobloxPlayerBeta", "RobloxWindow", "WINDOWSCLIENT")
LAUNCHER_CLASSES = ("RobloxLauncher", "RobloxStudioLauncher")

_dpi_ready = False


def enable_dpi_awareness() -> None:
    """Opt into per-monitor DPI awareness before any coordinate call.

    Without this, Windows returns scaled coordinates on a scaled display and
    every click lands somewhere else. Must happen first.
    """
    global _dpi_ready
    if _dpi_ready:
        return
    try:
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # PER_MONITOR_V2
    except (AttributeError, OSError):
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            user32.SetProcessDPIAware()
    _dpi_ready = True


# --- process enumeration ------------------------------------------------

TH32CS_SNAPPROCESS = 0x00000002
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
MAX_PATH = 260


class _PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", ctypes.c_wchar * MAX_PATH),
    ]


# Without these, ctypes truncates the snapshot HANDLE to 32 bits on x64 and
# the snapshot API fails silently. See _pids_named.
kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
kernel32.Process32FirstW.restype = wintypes.BOOL
kernel32.Process32FirstW.argtypes = [wintypes.HANDLE,
                                     ctypes.POINTER(_PROCESSENTRY32W)]
kernel32.Process32NextW.restype = wintypes.BOOL
kernel32.Process32NextW.argtypes = [wintypes.HANDLE,
                                    ctypes.POINTER(_PROCESSENTRY32W)]
kernel32.CloseHandle.restype = wintypes.BOOL
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]


def _pids_named(wanted: tuple[str, ...]) -> set[int]:
    """pids of running processes whose image name matches any of `wanted`.

    Uses the Toolhelp snapshot so this needs no third-party dependency.
    Returns an empty set if the snapshot cannot be taken, which is the
    signal for the caller to fall back to class-name matching.

    szExeFile arrives WITH its extension - measured on this machine it
    reads "RobloxPlayerBeta.exe", not "RobloxPlayerBeta" - so the extension
    is stripped before comparing. Comparing the raw name against
    "RobloxPlayerBeta" silently matched nothing while a real client was
    running.

    The argtypes/restype block above is load-bearing, not decoration.
    Without it ctypes assumes a 32-bit int return, which truncates a 64-bit
    HANDLE, so Process32FirstW is handed a garbage pointer and this
    silently returns an empty set.
    """
    lower = {w.lower() for w in wanted}
    snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if not snap or snap == INVALID_HANDLE_VALUE:
        return set()
    out: set[int] = set()
    try:
        entry = _PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(entry)
        if not kernel32.Process32FirstW(ctypes.c_void_p(snap),
                                         ctypes.byref(entry)):
            return out
        while True:
            name = entry.szExeFile.lower()
            if name.endswith(".exe"):
                name = name[:-4]
            if name in lower:
                out.add(int(entry.th32ProcessID))
            if not kernel32.Process32NextW(ctypes.c_void_p(snap),
                                           ctypes.byref(entry)):
                break
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(snap))
    return out


# --- window info --------------------------------------------------------

@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str
    cls: str
    left: int
    top: int
    width: int
    height: int
    pid: int = 0

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.left + self.width, self.top + self.height

    @property
    def center(self) -> tuple[int, int]:
        return self.left + self.width // 2, self.top + self.height // 2

    @property
    def is_minimised(self) -> bool:
        return bool(user32.IsIconic(self.hwnd))

    def describe(self) -> str:
        state = "minimised" if self.is_minimised else "visible"
        return (f"{self.cls!r} title={self.title!r} {self.width}x{self.height} "
                f"at ({self.left},{self.top}) pid={self.pid} {state}")


def _class_of(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buf, 256)
    return buf.value


def _title_of(hwnd: int) -> str:
    n = user32.GetWindowTextLengthW(hwnd)
    if n <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def _client_rect(hwnd: int) -> tuple[int, int, int, int] | None:
    """(left, top, width, height) of the client area in screen coords."""
    r = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(r)):
        return None
    pt = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(pt)):
        return None
    return pt.x, pt.y, r.right - r.left, r.bottom - r.top


_ENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


def _top_level_windows(min_size: int = 100) -> list[WindowInfo]:
    """Every top-level window with a client area worth capturing."""
    found: list[WindowInfo] = []

    def cb(hwnd, _lparam):
        cr = _client_rect(hwnd)
        if cr and cr[2] >= min_size and cr[3] >= min_size:
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            found.append(WindowInfo(
                hwnd=int(hwnd), title=_title_of(hwnd), cls=_class_of(hwnd),
                left=cr[0], top=cr[1], width=cr[2], height=cr[3],
                pid=int(pid.value)))
        return True

    user32.EnumWindows(_ENUMPROC(cb), 0)
    return found


def find_roblox(include_launcher: bool = True) -> WindowInfo | None:
    """The Roblox client window, or the launcher if the client is gone.

    Minimised and background windows are still candidates. That is
    deliberate: a minimised or occluded client is exactly the state this
    tool has to recover from, and excluding it would leave the tool blind
    in the one situation it exists for.
    """
    enable_dpi_awareness()
    wins = _top_level_windows()

    game_pids = _pids_named(GAME_PROCESSES)
    launcher_pids = _pids_named(LAUNCHER_PROCESSES) if include_launcher else set()

    # Prefer the largest window owned by a game process. A client can own
    # several top-level windows (tooltips, hidden helpers); the game
    # surface is the big one.
    by_pid = [w for w in wins if w.pid in game_pids]
    if by_pid:
        return max(by_pid, key=lambda w: w.width * w.height)

    if launcher_pids:
        by_lpid = [w for w in wins if w.pid in launcher_pids]
        if by_lpid:
            return max(by_lpid, key=lambda w: w.width * w.height)

    # Fallback: process enumeration unavailable or found nothing. Match on
    # class name, game classes before launcher classes.
    wanted = list(GAME_CLASSES) + (list(LAUNCHER_CLASSES) if include_launcher else [])
    by_cls = [w for w in wins if w.cls in wanted]
    if by_cls:
        def rank(w: WindowInfo) -> tuple:
            return (0 if w.cls in GAME_CLASSES else 1, -w.width * w.height)
        return sorted(by_cls, key=rank)[0]

    return None


def find_by_title_substring(needle: str) -> WindowInfo | None:
    """Largest top-level window whose title contains `needle`."""
    enable_dpi_awareness()
    needle_l = needle.lower()
    hits = [w for w in _top_level_windows() if needle_l in w.title.lower()]
    return max(hits, key=lambda w: w.width * w.height, default=None)


def is_foreground(hwnd: int) -> bool:
    return user32.GetForegroundWindow() == hwnd


def process_alive(pid: int) -> bool:
    """Whether a pid is still running.

    Checked with OpenProcess rather than a window lookup, because after a
    client crash the window is gone but the process may still be winding
    down - and clicking at that hwnd would hit whatever took its place.
    """
    if pid <= 0:
        return False
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return False
    try:
        code = wintypes.DWORD()
        if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
            return False
        return code.value == STILL_ACTIVE
    finally:
        k32.CloseHandle(h)