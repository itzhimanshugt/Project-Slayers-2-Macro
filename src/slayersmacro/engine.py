"""Macro loop: the engine that runs a macro and keeps it honest.

Responsibilities:
  - hold the window and refuse to run when Roblox is not focused
  - handle start/stop/panic from another thread
  - release every held key on exit, so a crash cannot leave W stuck down
  - rate-limit the loop and log what happened
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import input as sin
from . import log
from .capture import Capture
from .detect import Calibration, MovementWatch
from .guard import GuardConfig, SafetyGuard, UnsafeStateError
from .window import (  # noqa: F401
    WindowInfo, find_window, is_foreground, virtual_screen_size,
)


@dataclass
class LoopStats:
    cycles: int = 0
    actions: int = 0
    frames: int = 0
    started: float = field(default_factory=time.perf_counter)
    last_action_ms: float = 0.0
    stuck_events: int = 0

    @property
    def uptime_s(self) -> float:
        return time.perf_counter() - self.started

    def summary(self) -> str:
        up = self.uptime_s
        return (
            f"{self.cycles} cycles / {self.actions} actions / "
            f"{self.frames} frames in {up:.1f}s "
            f"({self.frames / up:.0f} fps) "
            f"stuck={self.stuck_events}"
        )


class MacroContext:
    """Everything a macro needs, handed to its step() each iteration."""

    def __init__(self, cfg: dict, cap: Capture, win: WindowInfo,
                 calib: Calibration) -> None:
        self.cfg = cfg
        self.cap = cap
        self.win = win
        self.calib = calib
        self.log = log.get("macro")
        self.stats = LoopStats()
        self.held: set[str] = set()
        # Populated by the engine. The guard is the only thing allowed to
        # answer "may I act right now".
        self.guard: SafetyGuard | None = None
        self.reference_frame = None
        # Set by the engine while it still holds the latch. A macro may
        # refuse to act (return False early) but must never bypass it.
        self._engine = None

    def may_act(self) -> bool:
        """The safety gate. False means: send nothing this iteration."""
        if self._engine is None:
            return True
        return self._engine.gate(self)

    def action(self, name: str) -> None:
        self.stats.actions += 1
        self.stats.last_action_ms = time.perf_counter() * 1000

    def region_frame(self, name: str) -> np.ndarray | None:
        """Grab a calibrated region, scaled to the current window size."""
        cal = self.calib.scaled(self.win)
        region = cal.regions.get(name)
        if region is None:
            return None
        l, t, w, h = region.to_screen(self.win)
        return self.cap.grab(l, t, w, h)

    def full_frame(self) -> np.ndarray | None:
        return self.cap.grab_window(self.win)

    def jitter(self) -> float:
        """Random ms offset, 0 when disabled."""
        j = float(self.cfg["general"].get("jitter_ms", 0) or 0)
        if j <= 0:
            return 0.0
        import random
        return random.uniform(-j, j)

    def hold(self, key: str, ms: float) -> None:
        sin.down(key)
        self.held.add(key)
        time.sleep(max(0.0, ms + self.jitter()) / 1000.0)
        sin.up(key)
        self.held.discard(key)
        self.action(f"hold {key} {ms:.0f}ms")

    def press(self, key: str) -> None:
        ms = float(self.cfg["input"].get("tap_ms", 18))
        sin.press_release(key, hold_ms=ms)
        self.action(f"press {key}")

    @property
    def screen_w(self) -> int:
        return virtual_screen_size()[0]

    @property
    def screen_h(self) -> int:
        return virtual_screen_size()[1]

    def attack(self) -> None:
        """Primary attack.

        Defaults to the left mouse button because Roblox ships M1 unbound -
        Space is jump. Honour an explicit key in config if the user has
        rebound it, and translate the conventional aliases.
        """
        key = str(self.cfg["combat"].get("attack_key", "mouse1")).lower()
        if key in ("mouse1", "lmb", "left", "m1"):
            sin.click("left")
        elif key in ("mouse2", "rmb", "right"):
            sin.click("right")
        else:
            sin.press_release(key,
                              hold_ms=float(self.cfg["combat"].get(
                                  "m1_hold_ms", 18)))
        self.action(f"attack via {key}")

    def release_attack(self) -> None:
        key = str(self.cfg["combat"].get("attack_key", "mouse1")).lower()
        if key in ("mouse1", "lmb", "left", "m1"):
            sin.button_up("left")
        elif key in ("mouse2", "rmb", "right"):
            sin.button_up("right")
        else:
            sin.up(key)


class Macro:
    """Base class. Subclasses implement run() and return False to stop."""

    name = "macro"

    def __init__(self, ctx: MacroContext) -> None:
        self.ctx = ctx

    def setup(self) -> None: ...
    def teardown(self) -> None: ...
    def run(self) -> bool:
        raise NotImplementedError


class Engine:
    """Runs one macro on a background thread, with panic from any thread."""

    def __init__(self, macro_factory: Callable[[MacroContext], Macro],
                 cfg: dict, max_seconds: float = 0.0) -> None:
        self.macro_factory = macro_factory
        self.cfg = cfg
        self.max_seconds = max_seconds
        self.cap = Capture()
        self.calib = Calibration.load()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.log = log.get("engine")

    def request_stop(self) -> None:
        self._stop.set()

    def release_all(self) -> None:
        """Let go of anything we are holding. Safe to call twice.

        Releases an explicit list rather than trusting internal bookkeeping,
        because the failure this exists to prevent is a key stuck DOWN. If
        the bookkeeping is wrong - which is exactly when it matters - this
        still works.
        """
        # Everything a macro could plausibly be holding. Releasing an
        # already-up key is a harmless no-op in SendInput.
        for key in ("w", "a", "s", "d", "space", "lshift", "lctrl", "up",
                    "down", "left", "right"):
            try:
                sin.up(key)
            except (OSError, KeyError):
                pass
        for btn in ("left", "right", "middle"):
            try:
                sin.button_up(btn)
            except (OSError, KeyError):
                pass
        self._held_now = set()

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="macro", daemon=True)
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    def _build_guard(self) -> SafetyGuard:
        s = self.cfg.get("safety", {})
        return SafetyGuard(GuardConfig(
            require_focus=bool(self.cfg["general"].get("require_focus", True)),
            settle_frames=int(s.get("settle_frames", 3)),
            stable_for_ms=int(s.get("stable_for_ms", 400)),
            detect_occlusion=bool(s.get("detect_occlusion", True)),
            abort_after=float(s.get("abort_after_s", 0.0)),
        ), logger=self.log)

    def _run(self) -> None:
        win = find_window(self.cfg["general"].get("window_title", "Roblox"))
        if win is None:
            self.log.error(
                "No Roblox window found. Open the game, then run the macro."
            )
            self._stop.set()
            return
        self.log.info(f"attached to '{win.title}' {win.width}x{win.height} at "
                      f"({win.left},{win.top})")

        ctx = MacroContext(self.cfg, self.cap, win, self.calib)
        ctx._engine = self
        ctx.guard = self._build_guard()
        # A reference frame so the guard can tell "the game is animating"
        # from "something else is covering the window".
        ctx.reference_frame = self.cap.grab_window(win)

        macro = self.macro_factory(ctx)
        macro.setup()
        try:
            macro.run()
        except Exception as exc:  # noqa: BLE001
            self.log.exception(f"macro '{macro.name}' crashed: {exc}")
            ctx.action(f"crash:{exc}")
        except UnsafeStateError as exc:
            self.log.error(f"macro '{macro.name}' aborted: {exc}")
        finally:
            macro.teardown()
            self.release_all()
            self.log.info(f"{macro.name}: {ctx.stats.summary()}")
            try:
                self.cap.close()
            except Exception:
                pass
            self._stop.set()

    def gate(self, ctx: MacroContext) -> bool:
        """Run one guard check. Macros call this every iteration.

        This is the single choke point for "may I send input right now".
        It is a method on the engine rather than something each macro
        reimplements, because every macro reimplementing it is a chance to
        forget the occlusion check.
        """
        if self._stop.is_set():
            return False
        win = find_window(self.cfg["general"].get("window_title", "Roblox"))
        if win is None:
            ctx.guard.check(None, None)
            return False
        ctx.win = win
        frame = self.cap.grab_window(win)
        report = ctx.guard.check(win, frame, ctx.reference_frame)
        if report.should_act and frame is not None:
            # Refresh the reference periodically so a genuinely static game
            # screen (a menu) does not get flagged as an occluder forever.
            ctx.reference_frame = frame
        return report.should_act

    # --- pacing helpers used by macros ---------------------------------

    def should_stop(self) -> bool:
        return self._stop.is_set()

    def sleep_ms(self, ms: float) -> bool:
        """Interruptible sleep. Returns True if a stop was requested."""
        if self._stop.wait(max(0.0, ms) / 1000.0):
            return True
        return False

    def guard_focus(self, win: WindowInfo) -> bool:
        """Pause while Roblox is not the focused window.

        Without this the macro happily types into whatever you clicked.
        """
        if not self.cfg["general"].get("require_focus", True):
            return False
        if is_foreground(win.hwnd):
            return False
        self.log.warning("Roblox lost focus - pausing. Click the game window.")
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            if self._stop.is_set():
                return True
            if is_foreground(win.hwnd):
                self.log.info("focus regained")
                return False
            self.sleep_ms(100)
        return True

    def loop_rate_limit(self, ctx: MacroContext) -> None:
        """Respect loop_hz if the user set one. 0 means flat out."""
        hz = float(self.cfg["input"].get("loop_hz", 0) or 0)
        if hz > 0:
            self.sleep_ms(1000.0 / hz)


def anti_idle_nudge() -> None:
    """Tiny mouse move to dodge the 20-minute idle kick.

    Roblox treats a completely stationary session as idle. A 1px jiggle is
    enough and costs nothing.
    """
    try:
        sin.move_relative(1, 0)
        time.sleep(0.02)
        sin.move_relative(-1, 0)
    except OSError:
        pass


__all__ = [
    "Engine", "Macro", "MacroContext", "LoopStats",
    "MovementWatch", "anti_idle_nudge",
]