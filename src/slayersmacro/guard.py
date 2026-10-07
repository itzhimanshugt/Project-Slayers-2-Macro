"""Safety interlocks. Nothing sends input without passing through here.

Two real bugs motivated this file, both found by accident:

1. A Roblox window was found and the macro attached to it and started
   pressing keys - while the actual game was not visible and the player was
   looking at something else entirely.

2. The window finder matched a window titled "Roblox", but screen capture
   returned a completely different window's pixels, because capture reads
   whatever is COMPOSITED AT THAT SCREEN RECT. This is not a bug in the
   capture code; it is the documented behaviour of every screen-capture API
   used here. A window existing is not the same as a window being visible.

So "is Roblox running?" is the wrong question. The right ones are:
  - is Roblox the foreground window?
  - are the pixels at Roblox's rect actually Roblox's?
  - did that answer change since the last frame?

The second is the important one and the one nobody checks.
"""

from __future__ import annotations

import ctypes
import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .window import WindowInfo, is_foreground

user32 = ctypes.WinDLL("user32", use_last_error=True)


class GuardState(Enum):
    OK = "ok"                 # foreground and pixels look stable
    NOT_FOCUSED = "not_focused"
    OCCLUDED = "occluded"     # another window is covering Roblox's rect
    NO_WINDOW = "no_window"
    UNSTABLE = "unstable"     # window rect jumping around (resize/move)


@dataclass
class GuardConfig:
    # Require Roblox to be the focused window. Non-negotiable in practice.
    require_focus: bool = True

    # Compare against a reference frame to catch "something else is covering
    # the Roblox rect". Worth keeping on: it is the check that would have
    # caught the macro attaching to a window it could not actually see.
    detect_occlusion: bool = True

    # How many consecutive frames must pass all checks before we act.
    settle_frames: int = 3

    # A fingerprint that changes every frame means the window is moving or
    # being resized; acting on a coordinate frame mid-resize is a misfire.
    stable_for_ms: int = 400

    # Abort a macro after this many consecutive unsafe observations rather
    # than waiting indefinitely for the user to alt-tab back.
    abort_after: float = 0.0     # 0 = never abort, just pause

    # Log a screenshot the first time we see each new state.
    log_on_change: bool = True


@dataclass
class GuardReport:
    state: GuardState
    detail: str = ""
    should_act: bool = False
    frames: int = 0


class SafetyGuard:
    """Checks, in order of severity, that acting right now is sane."""

    def __init__(self, cfg: GuardConfig | None = None,
                 log=None, logger=None) -> None:
        self.cfg = cfg or GuardConfig()
        # Accept both names: "log" shadows nothing here but reads ambiguously
        # next to a module-level `log`, and callers reasonably reach for
        # either.
        self.log = log if log is not None else logger
        self._ok_streak = 0
        self._unsafe_since: float | None = None
        self._last_rect: tuple | None = None
        self._rect_since = time.perf_counter()
        self._last_state: GuardState | None = None
        self._last_frame_hash: str | None = None

    # -- individual checks -------------------------------------------------

    @staticmethod
    def foreground_ok(win: WindowInfo) -> tuple[bool, str]:
        if is_foreground(win.hwnd):
            return True, ""
        return False, "Roblox is not the foreground window"

    def stable_ok(self, win: WindowInfo) -> tuple[bool, str]:
        """Window rect must have been still for stable_for_ms."""
        rect = (win.left, win.top, win.width, win.height)
        now = time.perf_counter()
        if rect != self._last_rect:
            self._last_rect = rect
            self._rect_since = now
            return False, (f"window rect just changed to "
                           f"{win.width}x{win.height} at ({win.left},{win.top})")
        held_ms = (now - self._rect_since) * 1000.0
        if held_ms < self.cfg.stable_for_ms:
            return False, f"window moved {held_ms:.0f}ms ago, waiting"
        return True, ""

    @staticmethod
    def fingerprint(frame: np.ndarray) -> str:
        """Cheap content hash. Used to tell 'same static screen' from
        'live game' and to notice the pixels belong to something else."""
        if frame is None or frame.size == 0:
            return "empty"
        small = frame[::16, ::16, :]
        return hashlib.blake2b(small.tobytes(), digest_size=8).hexdigest()

    # -- the combined check -----------------------------------------------

    def check(self, win: WindowInfo | None, frame: np.ndarray | None,
              reference_frame: np.ndarray | None = None) -> GuardReport:
        """One frame of verification. Cheap enough for the hot loop.

        reference_frame: a previously-accepted frame of the same region.
        Used only to report *that* the content changed unexpectedly, not to
        block on it - the game is legitimately animating.
        """
        if win is None:
            return self._transition(GuardState.NO_WINDOW,
                                    "no Roblox window", False)

        ok, detail = self.stable_ok(win)
        if not ok:
            return self._transition(GuardState.UNSTABLE, detail, False)

        if self.cfg.require_focus:
            ok, detail = self.foreground_ok(win)
            if not ok:
                return self._transition(GuardState.NOT_FOCUSED, detail, False)

        if frame is None:
            return self._transition(GuardState.OCCLUDED,
                                    "could not capture the window rect", False)

        h = self.fingerprint(frame)
        if self._last_frame_hash is not None and h != self._last_frame_hash:
            # Content changed. Expected while playing, so this is not a block
            # by itself - but a completely static hash across many frames
            # while we are trying to act means we are probably reading a
            # different window's pixels.
            pass
        self._last_frame_hash = h

        if self.cfg.detect_occlusion and reference_frame is not None:
            try:
                import cv2
                same = cv2.absdiff(frame, reference_frame).mean() < 1.0
            except Exception:
                same = False
            if same:
                return self._transition(
                    GuardState.OCCLUDED,
                    "window rect is showing a static image - something else "
                    "is probably on top", False)

        self._ok_streak += 1
        if self._ok_streak >= self.cfg.settle_frames:
            self._unsafe_since = None
            return self._transition(GuardState.OK, "", True)

        return self._transition(
            GuardState.OK,
            f"settling, {self._ok_streak}/{self.cfg.settle_frames}", False)

    def _transition(self, state: GuardState, detail: str,
                    should_act: bool) -> GuardReport:
        if state is not self._last_state:
            if self.cfg.log_on_change and self.log:
                if state is GuardState.OK:
                    self.log.info("guard: clear")
                else:
                    self.log.warning(f"guard: {state.value} - {detail}")
            if state is not GuardState.OK:
                self._ok_streak = 0
                self._unsafe_since = time.perf_counter()
            self._last_state = state

        if state is not GuardState.OK and self.cfg.abort_after > 0:
            since = self._unsafe_since or time.perf_counter()
            if time.perf_counter() - since > self.cfg.abort_after:
                raise UnsafeStateError(
                    f"aborting: {state.value} for longer than "
                    f"{self.cfg.abort_after:.0f}s ({detail})")

        return GuardReport(state=state, detail=detail,
                           should_act=should_act, frames=self._ok_streak)


class UnsafeStateError(RuntimeError):
    """Raised when the environment has been unsafe for too long."""