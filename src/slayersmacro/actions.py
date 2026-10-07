"""Named actions, and the recovery ladder wired to them.

The ladder in recovery.py is deliberately game-agnostic: it knows what
escalation looks like but not what "back off" means. This module supplies
the actual actions for Project Slayers 2 and exposes them as a menu.

Every action is a small, individually testable unit that returns success or
failure, because the ladder can only escalate correctly if each rung reports
honestly whether it worked.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from . import input as sin
from .recovery import ProblemReport, RecoveryConfig, RecoveryLadder, Rung


@dataclass
class ActionResult:
    ok: bool
    detail: str = ""


class GameActions:
    """Concrete recovery actions.

    Key bindings are Roblox defaults, verified rather than guessed:
    Space = jump, WASD = movement, Escape = menu, Shift = sprint.

    Positions are relative to the window, not absolute screen pixels, so
    they survive a resolution change. Centre-relative coordinates are
    expressed as fractions: 0.5 is the middle of the window.
    """

    def __init__(self, ctx, win=None) -> None:
        self.ctx = ctx
        self.win = win
        self.log = ctx.log
        # Fractional window coordinates for UI buttons. These are guesses
        # that need calibration - run tools/calibrate.py and verify. Kept
        # here as named constants so there is one place to correct them.
        self.esc_button = (0.5, 0.78)      # Escape menu's Leave button
        self.rejoin_button = (0.5, 0.72)   # after a disconnect dialog

    # -- individual actions ------------------------------------------------

    def nudge(self) -> ActionResult:
        """A tiny movement. Often enough to clear a soft lock."""
        self.ctx.hold("space", 90)
        return ActionResult(True, "jumped")

    def backoff(self) -> ActionResult:
        """Retreat from whatever is blocking us."""
        self.ctx.hold("s", 420)
        self.ctx.hold("space", 90)
        self.ctx.hold("s", 320)
        return ActionResult(True, "stepped back")

    def turn(self) -> ActionResult:
        """Change facing so the next attempt is not the same blocked one."""
        # A = turn left, D = turn right in Roblox's default scheme.
        self.ctx.hold("a", 280)
        self.ctx.hold("d", 560)
        return ActionResult(True, "turned")

    def rewalk(self) -> ActionResult:
        """Back away, turn around, and come back at the target."""
        self.ctx.hold("s", 700)
        self.ctx.hold("a", 400)
        self.ctx.hold("w", 600)
        return ActionResult(True, "re-walked to target")

    def respawn(self) -> ActionResult:
        """Give up on this target and start the encounter again.

        Relies on the game respawning us when killed, or on walking back to
        the spawn. Deliberately conservative: it does not touch the menu.
        """
        self.ctx.hold("s", 900)
        time.sleep(0.3)
        self.ctx.hold("w", 700)
        return ActionResult(True, "walked back toward spawn")

    def serverhop(self) -> ActionResult:
        """Leave and rejoin, landing in a different server.

        THE most disruptive action here and the reason max_auto_rung
        defaults below it. Uses the Escape menu rather than any deep link,
        because a deep link is a client-side navigation and not something
        an external tool should be constructing.

        The button positions are fractional and almost certainly need
        calibration against the live menu. If they miss, this returns
        failure and the ladder stops rather than clicking blindly - which
        is the safe outcome, since blind clicking in a menu is how a macro
        ends up deleting a character.
        """
        if self.win is None:
            return ActionResult(False, "no window to click in")
        self.ctx.hold("escape", 90)
        time.sleep(0.6)

        left = (int(self.win.left + self.win.width * self.esc_button[0]),
                int(self.win.top + self.win.height * self.esc_button[1]))
        frame = self.ctx.cap.grab(self.win.left, self.win.top,
                                  self.win.width, self.win.height)
        if frame is None:
            return ActionResult(False, "could not read the menu")

        # Only click if something that looks like a button is there. A blind
        # click in a menu is not recoverable.
        import cv2
        import numpy as np
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Roblox menus use a red-ish leave button.
        red = cv2.inRange(hsv, (0, 120, 120), (12, 255, 255))
        red = cv2.bitwise_or(red, cv2.inRange(hsv, (168, 120, 120),
                                             (180, 255, 255)))
        if (red > 0).mean() < 0.0015:
            return ActionResult(False, "no red button found - not clicking blindly")
        if (red > 0).mean() > 0.15:
            return ActionResult(False, "too much red - menu state unclear")

        sin.move_absolute(left[0], left[1],
                          self.ctx.screen_w, self.ctx.screen_h)
        time.sleep(0.08)
        sin.click("left")
        self.log.warning("clicked Leave - check the game")
        return ActionResult(True, "clicked leave, verification needed")

    # -- wiring ------------------------------------------------------------

    def as_ladder(self, cfg: RecoveryConfig | None = None) -> RecoveryLadder:
        cfg = cfg or RecoveryConfig()
        actions = {
            Rung.NUDGE: self.nudge,
            Rung.BACKOFF: self.backoff,
            Rung.TURN: self.turn,
            Rung.REWALK: self.rewalk,
            Rung.RESPAWN: self.respawn,
            Rung.SERVERHOP: self.serverhop,
        }
        wrapped = {
            r: (lambda fn=fn: fn().ok)
            for r, fn in actions.items()
        }
        return RecoveryLadder(cfg, wrapped, logger=self.log)

    def from_config(self, cfg: dict) -> RecoveryLadder:
        c = cfg.get("recovery", {})
        rc = RecoveryConfig(
            confirm_frames=int(c.get("confirm_frames", 4)),
            step_pause_ms=int(c.get("step_pause_ms", 900)),
            max_auto_rung=Rung.parse(c.get("max_auto_rung", "respawn")),
            max_cycles=int(c.get("max_cycles", 6)),
        )
        return self.as_ladder(rc)


def obstruction_report(moving: bool, control_down: bool,
                       focused: bool) -> ProblemReport:
    """Translate the obstruction detector's verdict into a ladder report."""
    from .adaptation import ObstructionDetector
    d = ObstructionDetector(still_ms=0)
    stuck = d.observe(moving, control_down, focused)
    return ProblemReport.stuck("view static while holding a movement key") \
        if stuck else ProblemReport.fine()