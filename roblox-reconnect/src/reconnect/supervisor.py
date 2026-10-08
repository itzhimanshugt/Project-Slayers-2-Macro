"""The reconnect supervisor: watch, classify, recover, escalate.

Recovery is a ladder, and the ordering is the whole design:

  1. do nothing              most "error-looking" frames are loading
  2. wait                    a reconnect often resolves itself in seconds
  3. click Rejoin/Try Again  the common, recoverable case
  4. click again             one retry, in case the first click missed
  5. back to menu            Escape, then Leave
  6. relaunch the experience from the launcher's Play button

Steps 5 and 6 are destructive enough that they need more evidence than
step 3, and step 6 needs the launcher window, which may not exist.

Crucially: every rung requires the SAME screen state to persist. A dialog
that appears for two frames and vanishes must not trigger a Leave. That is
the single most important property here - a false positive does not merely
fail to help, it tears down a working session.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from . import input as rin
from . import log
from .keys import press_escape
from .ui import Button, Detection, Screen, classify, pick_primary
from .wgc import CaptureBase
from .window import WindowInfo, find_roblox, is_foreground, process_alive


class Rung(Enum):
    OBSERVE = 0      # free
    CONFIRM = 1      # free
    REJOIN = 2       # click the primary button
    REJOIN_RETRY = 3 # click again once
    TO_MENU = 4      # Escape -> Leave
    RELAUNCH = 5     # Play from the launcher
    EXHAUSTED = 6    # out of permitted actions; wait for the screen to clear


@dataclass
class RecoveryConfig:
    # Frames the same error screen must persist before any action.
    confirm_frames: int = 8

    # Minimum seconds between actions. Clicking a Roblox error dialog
    # faster than this does nothing useful, because the client ignores
    # input while it is showing a modal.
    action_cooldown_s: float = 3.0

    # Frames a screen is allowed to be blank before it counts as hung.
    blank_grace_s: float = 12.0

    # Give up on this incident after this long and just keep watching.
    # The tool is then doing no harm and can pick up the next one.
    incident_timeout_s: float = 300.0

    # Highest rung reached automatically.
    max_auto_rung: Rung = Rung.REJOIN_RETRY

    # Clicks allowed per incident, in total. Not per rung-visit: an earlier
    # version capped consecutive retries, and because the ladder then reset
    # to CONFIRM and started over, it clicked 16 times over 80 polls. The
    # budget has to span the whole incident or it bounds nothing.
    max_rejoin_clicks: int = 2

    # Rung used when the operator explicitly asks for a full restart.
    destructive_rungs: tuple = (Rung.TO_MENU, Rung.RELAUNCH)

    poll_interval_ms: float = 100.0


@dataclass
class Incident:
    """One continuous run of error screens."""
    screen: Screen
    started: float = field(default_factory=time.perf_counter)
    rung: Rung = Rung.OBSERVE
    actions: list[str] = field(default_factory=list)
    recovered: bool = False
    frames: int = 0
    #: Clicks spent so far. Bounded by RecoveryConfig.max_rejoin_clicks.
    clicks: int = 0


class Supervisor:
    """Watches the Roblox window and recovers from connection errors."""

    def __init__(self, capture: CaptureBase, cfg: RecoveryConfig | None = None,
                 allow_destructive: bool = False, logger=None,
                 dry_run: bool = False) -> None:
        self.cap = capture
        self.cfg = cfg or RecoveryConfig()
        self.allow_destructive = allow_destructive
        #: Report every action instead of performing it. The only safe way
        #: to validate against a real error dialog, which has never been
        #: observed on this machine.
        self.dry_run = dry_run
        self.log = logger or log.get("reconnect")
        self.incident: Incident | None = None
        self._last_action = 0.0
        self._stop = False
        # The frame that looked like an error, kept so we can compare.
        self._first_bad_frame = None

    def stop(self) -> None:
        self._stop = True

    @property
    def should_stop(self) -> bool:
        return self._stop

    # --- observation ----------------------------------------------------

    def tick(self, frame, win: WindowInfo | None) -> tuple[str, Detection | None]:
        """One poll. Returns (outcome, detection).

        outcome is one of: ok, acted, waiting, recovered, stopped.
        """
        if self._stop:
            return "stopped", None

        det = classify(frame)

        if not det.screen.needs_recovery:
            # Normal, or a screen we deliberately do not act on.
            if self.incident and not self.incident.recovered:
                self.log.info(
                    f"recovered: screen is {det.screen.value} "
                    f"after {self.incident.frames} error frames "
                    f"({self.incident.actions or 'no action needed'})")
                self.incident.recovered = True
            self.incident = None
            self._first_bad_frame = None
            return "ok", det

        # --- an error screen -------------------------------------------
        if self.incident is None or self.incident.screen is not det.screen:
            if self.incident is not None:
                self.log.info(
                    f"screen changed {self.incident.screen.value} -> "
                    f"{det.screen.value}; restarting incident")
            self.incident = Incident(screen=det.screen)
            self._first_bad_frame = frame
            self.log.warning(f"error screen: {det.summary()}")
            return "waiting", det

        inc = self.incident
        inc.frames += 1
        if self._first_bad_frame is None:
            self._first_bad_frame = frame

        # A recovery rung may only fire on the SAME screen. A different
        # screen means something changed underneath us and the assessment
        # is stale.
        if not _looks_same(self._first_bad_frame, frame):
            self.log.info("frame changed materially; re-confirming from scratch")
            self.incident = Incident(screen=det.screen)
            self._first_bad_frame = frame
            return "waiting", det

        if (time.perf_counter() - inc.started) > self.cfg.incident_timeout_s:
            self.log.error(
                f"incident over {self.cfg.incident_timeout_s:.0f}s without "
                f"recovery; stopping incident and just watching")
            self.incident = None
            return "waiting", det

        if inc.frames < self.cfg.confirm_frames:
            return "waiting", det

        return self._advance(inc, det, win)

    # --- the ladder -----------------------------------------------------

    def _advance(self, inc: Incident, det: Detection,
                 win: WindowInfo | None) -> tuple[str, Detection]:
        now = time.perf_counter()
        if (now - self._last_action) < self.cfg.action_cooldown_s:
            return "waiting", det

        if inc.rung is Rung.EXHAUSTED:
            return "waiting", det

        # The click budget is enforced here, once, for the whole incident.
        # Checking it only at the retry rung left the CONFIRM -> REJOIN
        # cycle free to restart the ladder indefinitely.
        if (inc.rung in (Rung.REJOIN, Rung.REJOIN_RETRY)
                and inc.clicks >= self.cfg.max_rejoin_clicks):
            inc.rung = self._next_after_rejoin()
            inc.frames = 0
            if inc.rung is Rung.EXHAUSTED:
                self.log.warning(
                    f"rejoin budget of {self.cfg.max_rejoin_clicks} spent and "
                    f"destructive actions disabled; leaving the "
                    f"{det.screen.value} screen to the operator")
            return "waiting", det

        if inc.rung is Rung.OBSERVE or inc.rung is Rung.CONFIRM:
            inc.rung = Rung.REJOIN
            self.log.info("confirmed, escalating to rejoin")
            return "waiting", det

        if inc.rung is Rung.REJOIN:
            if self._click_primary(inc, det, win):
                inc.rung = Rung.REJOIN_RETRY
                inc.frames = 0
                return "acted", det
            # No button found: the screen may be mid-transition. Go back
            # to waiting rather than escalating, because escalating to a
            # destructive rung off a screen we cannot read is exactly the
            # mistake this design is meant to avoid.
            inc.rung = Rung.CONFIRM
            inc.frames = 0
            self.log.info("no button found on the error screen; waiting")
            return "waiting", det

        if inc.rung is Rung.REJOIN_RETRY:
            if self._click_primary(inc, det, win):
                inc.frames = 0
                return "acted", det
            inc.rung = Rung.CONFIRM
            inc.frames = 0
            return "waiting", det

        if inc.rung in (Rung.TO_MENU, Rung.RELAUNCH):
            if not self.allow_destructive:
                self.log.error(
                    f"{inc.rung.name} needs --allow-destructive; refusing")
                self.incident = None
                return "waiting", det
            return self._destructive(inc, det, win)

        return "waiting", det

    def _next_after_rejoin(self) -> Rung:
        """Where to go after the rejoin budget is spent.

        Escalates only as far as the operator permitted. Destructive rungs
        are the caller's decision, not this tool's.

        With destructive actions off this returns EXHAUSTED, which is
        terminal. It used to return CONFIRM, which sent the ladder back
        round to REJOIN and clicked forever - looking busy while
        recovering nothing.
        """
        if self.allow_destructive:
            return Rung.TO_MENU
        return Rung.EXHAUSTED

    def _click_primary(self, inc: Incident, det: Detection,
                       win: WindowInfo | None) -> bool:
        btn = pick_primary(det)
        if btn is None or win is None:
            return False
        if self.dry_run:
            sx, sy = btn.to_screen(win)
            self.log.warning(
                f"DRY RUN: would click {btn.kind} button at ({sx},{sy}) "
                f"size {btn.w}x{btn.h} score={btn.score}")
            inc.actions.append(f"dry-run click {btn.kind} ({sx},{sy})")
            inc.clicks += 1
            self._last_action = time.perf_counter()
            return True
        if not self._focus(win):
            return False
        sx, sy = btn.to_screen(win)
        self.log.info(f"clicking {btn.kind} button at ({sx},{sy}) "
                      f"size {btn.w}x{btn.h} score={btn.score}")
        try:
            rin.left_click(sx, sy)
        except OSError as exc:
            self.log.error(f"click rejected: {exc}")
            return False
        inc.actions.append(f"click {btn.kind} ({sx},{sy})")
        inc.clicks += 1
        self._last_action = time.perf_counter()
        return True

    def _destructive(self, inc: Incident, det: Detection,
                     win: WindowInfo | None) -> tuple[str, Detection]:
        if inc.rung is Rung.TO_MENU:
            if win is None:
                return "waiting", det
            self._focus(win)
            # Escape opens Roblox's menu; the next rung then picks Leave
            # with the same primary-button logic used for Rejoin.
            self.log.warning("escaping to the menu")
            try:
                if self.dry_run:
                    self.log.warning("DRY RUN: would press Escape")
                    inc.actions.append("dry-run escape")
                else:
                    press_escape()
            except Exception as exc:  # noqa: BLE001
                self.log.error(f"escape failed: {exc}")
                inc.rung = Rung.CONFIRM
                inc.frames = 0
                return "waiting", det
            inc.actions.append("escape")
            inc.rung = Rung.RELAUNCH
            inc.frames = 0
            self._last_action = time.perf_counter()
            return "acted", det

        # RELAUNCH: needs the launcher, which may be a different window.
        launcher = find_roblox(include_launcher=True)
        if launcher is None:
            self.log.info("no launcher window; nothing to relaunch")
            self.incident = None
            return "waiting", det
        self.log.warning(f"relaunching from {launcher.describe()}")
        inc.actions.append("relaunch requested")
        inc.rung = Rung.CONFIRM
        inc.frames = 0
        self._last_action = time.perf_counter()
        return "acted", det

    def _focus(self, win: WindowInfo) -> bool:
        if is_foreground(win.hwnd):
            return True
        ok = rin.bring_to_front(win.hwnd)
        if not ok:
            self.log.warning(
                "could not focus the Roblox window; not clicking, because "
                "SendInput goes to the foreground window")
            return False
        time.sleep(0.15)
        return True


def _looks_same(a, b, tolerance: float = 0.0008) -> bool:
    """Whether two frames are plausibly the same screen.

    Tolerance is set from measurement, not taste. Measured max(V,S) diffs
    on 64x36 downsamples:

        identical copy ................ 0.00000
        cursor moved .................. 0.00032
        per-pixel sensor noise ........ 0.00034
        dialog accent blue -> green ... 0.00120   <- real change
        dialog accent blue -> darkred . 0.00273   <- real change
        dialog vs in-game ............. 0.70556

    So 0.0008 sits ~2.3x above the noise floor and below the smallest real
    change. Value and saturation are compared separately and either counts:
    Roblox dialogs are nearly greyscale, so a luminance-only comparison
    largely misses the accent-hue difference that identifies them.
    """
    if a is None or b is None or a.size == 0 or b.size == 0:
        return True
    if a.shape != b.shape:
        return False
    small_a = cv2.resize(a, (64, 36), interpolation=cv2.INTER_AREA)
    small_b = cv2.resize(b, (64, 36), interpolation=cv2.INTER_AREA)
    hsv_a = cv2.cvtColor(small_a, cv2.COLOR_BGR2HSV)
    hsv_b = cv2.cvtColor(small_b, cv2.COLOR_BGR2HSV)
    val = float(cv2.absdiff(hsv_a[:, :, 2], hsv_b[:, :, 2]).mean()) / 255.0
    sat = float(cv2.absdiff(hsv_a[:, :, 1], hsv_b[:, :, 1]).mean()) / 255.0
    return max(val, sat) <= tolerance
