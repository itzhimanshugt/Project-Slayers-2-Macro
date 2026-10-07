"""The recovery ladder.

A macro that only knows one way to do a thing breaks the first time the
world disagrees. This escalates through progressively heavier responses,
stopping as soon as the problem goes away.

The escalation order matters. Each rung costs more than the last:

  1. do nothing      - most "stuck" detections are a menu, a cutscene, or the
                       player touching the keyboard. Acting here is wrong.
  2. re-check        - wait and confirm it is real. Cheap, and it catches
                       the common transient case.
  3. nudge           - a small in-game action that often clears a soft lock.
  4. back off        - physically retreat from whatever is blocking us.
  5. turn            - change facing, which changes what we walk into.
  6. walk the loop   - re-traverse a known-good path.
  7. respaw          - give up on this target and start over.
  8. server hop      - last resort; changes the instance entirely.

The one hard rule: every rung requires CONFIRMATION that the previous one
failed. Without that, a transient detection failure walks the whole ladder
in a second and you end up server-hopping while your character is fine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable

from . import log


class Rung(IntEnum):
    HOLD = 0        # do nothing
    RECHECK = 1     # wait and confirm
    NUDGE = 2       # small in-game action
    BACKOFF = 3     # retreat
    TURN = 4        # change facing
    REWALK = 5      # re-traverse known path
    RESPAWN = 6     # restart the target
    SERVERHOP = 7   # change instance

    @classmethod
    def _missing_(cls, value):
        # Config files are written by hand in lowercase. Accepting "respawn"
        # for Rung.RESPAWN is the difference between a working setting and a
        # KeyError the first time someone edits their own config.
        if isinstance(value, str):
            v = value.strip().lower()
            for member in cls:
                if member.name.lower() == v:
                    return member
            # Also accept the no-separator spellings people reach for.
            squashed = v.replace("_", "").replace("-", "")
            for member in cls:
                if member.name.lower() == squashed:
                    return member
        raise KeyError(f"unknown recovery rung: {value!r}. "
                       f"Valid: {', '.join(m.name.lower() for m in cls)}")

    @classmethod
    def parse(cls, value) -> "Rung":
        if isinstance(value, cls):
            return value
        return cls(value)


@dataclass
class RecoveryConfig:
    # Confirm a problem this many frames in a row before escalating.
    confirm_frames: int = 4

    # Milliseconds between escalation steps. Long enough for the game to
    # visibly react, short enough that a real problem resolves quickly.
    step_pause_ms: int = 900

    # Never escalate past this rung automatically. SERVERHOP by default:
    # it is the most disruptive thing here and should need a human.
    max_auto_rung: Rung = Rung.RESPAWN

    # Give up on the whole macro after this many failed cycles.
    max_cycles: int = 6

    # Extra frames needed to confirm at higher rungs. A respaw takes longer
    # to become visible than a nudge does.
    rung_settle_frames: dict = field(default_factory=lambda: {
        Rung.RESPAWN: 12,
        Rung.REWALK: 8,
        Rung.SERVERHOP: 20,
    })


@dataclass
class ProblemReport:
    """Is the thing we are trying to do actually broken?"""

    ok: bool                       # the macro's goal is achievable
    progressing: bool              # and it is changing state, not frozen
    detail: str = ""

    @classmethod
    def fine(cls, detail: str = "") -> "ProblemReport":
        return cls(ok=True, progressing=True, detail=detail)

    @classmethod
    def stuck(cls, detail: str = "") -> "ProblemReport":
        return cls(ok=True, progressing=False, detail=detail)

    @classmethod
    def dead(cls, detail: str = "") -> "ProblemReport":
        """Target gone, session over. Needs a restart, not a nudge."""
        return cls(ok=False, progressing=False, detail=detail)


class RecoveryLadder:
    """Escalates only when the rung below it has been tried and failed."""

    def __init__(self, cfg: RecoveryConfig | None = None,
                 actions: dict | None = None,
                 logger=None) -> None:
        self.cfg = cfg or RecoveryConfig()
        self.log = logger or log.get("recovery")
        # Rung -> callable performing that rung. Supplied by the macro,
        # because only the macro knows what "back off" means for its game.
        self.actions: dict[Rung, Callable[[], bool]] = actions or {}
        self._suspect_frames = 0
        self._rung = Rung.HOLD
        # Negative infinity, not 0: needs_action() compares against a
        # caller-supplied timestamp that may start at 0 in tests, and
        # "never fired" must always satisfy the pause.
        self._since_action = float("-inf")
        self._cycle = 0
        self._rung_failures = 0

    @property
    def rung(self) -> Rung:
        return self._rung

    @property
    def cycle(self) -> int:
        return self._cycle

    def required_frames(self) -> int:
        return int(self.cfg.rung_settle_frames.get(
            self._rung, self.cfg.confirm_frames))

    def observe(self, report: ProblemReport) -> None:
        """Feed one frame's assessment. Drives escalation only."""
        if report.ok and report.progressing:
            self._on_success()
            return
        self._suspect_frames += 1

    def _on_success(self) -> None:
        if self._rung is not Rung.HOLD:
            self.log.info(f"recovered at rung {self._rung.name}")
        self._rung = Rung.HOLD
        self._suspect_frames = 0
        self._rung_failures = 0
        self._cycle = 0

    def needs_action(self, now: float | None = None) -> bool:
        """True when the current rung should fire now."""
        if self._suspect_frames < self.required_frames():
            return False
        now = now if now is not None else time.perf_counter()
        return (now - self._since_action) >= self.cfg.step_pause_ms / 1000.0

    @property
    def exhausted(self) -> bool:
        """True once max_cycles give-ups have been reached.

        The caller must check this. Without a consumer, giving up just
        resets the ladder to HOLD and it starts climbing again on the next
        frame - an infinite retry loop that looks like work but is not.
        """
        return self._cycle >= self.cfg.max_cycles

    def fire(self, now: float | None = None) -> Rung | None:
        """Attempt the current rung. Returns the rung attempted, or None."""
        if self.exhausted:
            return None
        if not self.needs_action(now):
            return None
        now = now if now is not None else time.perf_counter()
        self._since_action = now

        if self._rung in (Rung.HOLD, Rung.RECHECK):
            # These two rungs cost nothing: HOLD is "not confirmed yet" and
            # RECHECK is "confirmed, now do something". RECHECK must never
            # resolve as success, or the ladder resets here and never
            # reaches the rungs that actually send input.
            previous = self._rung
            self._rung = self._next_rung() if previous is Rung.HOLD \
                else Rung(self._rung + 1)
            self._suspect_frames = 0
            self.log.info(f"recovery: {previous.name} elapsed -> "
                          f"{self._rung.name}")
            return self._rung

        fn = self.actions.get(self._rung)
        if fn is None:
            self.log.warning(f"recovery: no action for {self._rung.name}, "
                             f"stopping escalation here")
            return self._rung

        try:
            worked = fn()
        except Exception as exc:  # noqa: BLE001
            # A rung that throws is a failed rung, not a crash. Treat it
            # exactly like one that returned False.
            self.log.exception(f"recovery: {self._rung.name} raised {exc}")
            worked = False

        if worked:
            self.log.info(f"recovery: {self._rung.name} worked")
            self._rung = Rung.HOLD
            self._suspect_frames = 0
            self._rung_failures = 0
            self._cycle = 0
        else:
            self._rung_failures += 1
            self._suspect_frames = 0
            self.log.warning(f"recovery: {self._rung.name} did not work "
                             f"(failure {self._rung_failures})")
            if self._rung_failures > self.cfg.max_cycles:
                self.log.error("recovery: giving up on this target")
                self._cycle += 1
                self._rung_failures = 0
                self._rung = Rung.HOLD
                self._suspect_frames = 0
                return Rung.HOLD
            self._rung = self._next_rung()
            if self._rung > self.cfg.max_auto_rung:
                self.log.error(
                    f"recovery: reached {self._rung.name}, which is above "
                    f"max_auto_rung {self.cfg.max_auto_rung.name}. Needs a human.")
                self._rung = Rung.RESPAWN
        return self._rung

    def _next_rung(self) -> Rung:
        nxt = self._rung + 1
        return Rung(min(int(nxt), int(self.cfg.max_auto_rung)))

    def reset(self) -> None:
        self._rung = Rung.HOLD
        self._suspect_frames = 0
        self._rung_failures = 0
        self._cycle = 0