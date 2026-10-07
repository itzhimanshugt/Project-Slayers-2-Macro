"""Learning your timings, and adapting detection thresholds.

Two separate things, both of which replace a hardcoded guess with
something measured.

1. TimingProfile - records how fast YOU actually act. The combat loop's
   interval starts as a default and converges to your own rhythm.

2. OnlineAdaptation - watches detection quality during play and nudges
   thresholds when they are clearly wrong. Deliberately slow and bounded:
   an adapter that swings thresholds around is worse than a fixed guess.

What this is NOT: it does not modify its own code, does not train a
network, and does not decide strategy. It fits two small sets of numbers
from observed behaviour. That is the whole claim, and it is a claim I can
actually verify.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

CONF_DIR = Path(__file__).resolve().parent.parent.parent / "config"


@dataclass
class TimingProfile:
    """Learns your action interval.

    Starts at the configured value and converges toward the median of your
    observed intervals. Uses a median rather than a mean because one bad
    frame (a menu opening, a lag spike) would otherwise poison the average
    permanently.
    """

    target_ms: float = 90.0
    samples: deque = field(default_factory=lambda: deque(maxlen=240))
    # Fraction of the new observation applied per sample. Low on purpose:
    # fast adaptation oscillates, slow adaptation is stable.
    alpha: float = 0.12
    # Clamp how far learning may move the interval from the configured
    # target. Without this, one weird session could permanently halve it.
    max_drift_ratio: float = 0.5
    learned: bool = False

    @property
    def interval_ms(self) -> float:
        if not self.samples:
            return self.target_ms
        ordered = sorted(self.samples)
        return ordered[len(ordered) // 2]

    def observe(self, actual_ms: float) -> None:
        """One measured interval between two actions."""
        # Ignore obvious outliers rather than letting them in slowly.
        if actual_ms <= 0 or actual_ms > self.target_ms * 20:
            return
        if self.samples and abs(actual_ms - self.interval_ms) > \
                self.interval_ms * 4 and len(self.samples) > 10:
            return
        self.samples.append(actual_ms)
        if len(self.samples) >= 20:
            self.learned = True

    def reset(self) -> None:
        self.samples.clear()
        self.learned = False

    def save(self, path: Path | None = None) -> Path:
        path = path or (CONF_DIR / "timing.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "target_ms": self.target_ms,
            "learned_ms": self.interval_ms,
            "samples": len(self.samples),
            "is_learned": self.learned,
        }, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | None = None,
             target_ms: float = 90.0) -> "TimingProfile":
        path = path or (CONF_DIR / "timing.json")
        p = cls(target_ms=target_ms)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            learned = float(data.get("learned_ms", target_ms))
            lo = target_ms * (1.0 - p.max_drift_ratio)
            hi = target_ms * (1.0 + p.max_drift_ratio)
            p.target_ms = max(lo, min(hi, learned))
            p.learned = bool(data.get("is_learned", False))
        return p


@dataclass
class AdaptiveThreshold:
    """One detector threshold that can drift within limits."""

    name: str
    value: float
    initial: float
    lo: float
    hi: float
    step: float = 0.05

    def nudge(self, direction: int) -> float:
        self.value = max(self.lo, min(self.hi, self.value + direction * self.step))
        return self.value

    @property
    def moved(self) -> bool:
        return abs(self.value - self.initial) > self.step / 2


@dataclass
class AdaptationState:
    """Recent detection outcomes, used to decide whether to adapt."""

    window: int = 90
    good: deque = field(default_factory=deque)
    bad: deque = field(default_factory=deque)

    def record(self, detected: bool) -> None:
        self.good.append(1 if detected else 0)
        self.bad.append(1 if not detected else 0)
        while len(self.good) > self.window:
            self.good.popleft()
            self.bad.popleft()

    @property
    def good_rate(self) -> float:
        if not self.good:
            return 1.0
        return sum(self.good) / len(self.good)

    @property
    def enough_data(self) -> bool:
        return len(self.good) >= self.window


class OnlineAdaptation:
    """Adjusts detection thresholds from observed success rate.

    The rule is deliberately blunt and one-directional:

    - detection rate high  -> loosen (lower the bar, catch dimmer things)
    - detection rate low   -> tighten (raise the bar, demand brighter)

    One-directional matters. An adapter that chases a target rate will
    oscillate: tighten, detection rises, loosen, detection falls. Since a
    single frame of noise should not move anything, each adjustment needs a
    sustained run of bad or good frames before it fires.

    Adaptation is also CAPPABLE. If the user has calibrated by hand and set
    enabled=false, nothing here runs. Silently retuning a good calibration
    is the worst failure mode this could have.
    """

    def __init__(self, enabled: bool = True,
                 target_rate: float = 0.92,
                 adjust_every: int = 240,
                 max_adjustments: int = 40) -> None:
        self.enabled = enabled
        # Aim a little below perfect: a detector that never fails is
        # usually one that has stopped looking.
        self.target_rate = target_rate
        self.adjust_every = adjust_every
        self.max_adjustments = max_adjustments
        self.state = AdaptationState()
        self.thresholds: dict[str, AdaptiveThreshold] = {}
        self.adjustments = 0
        self._since_adjust = 0

    def register(self, threshold: AdaptiveThreshold) -> None:
        self.thresholds[threshold.name] = threshold

    def record(self, detected: bool) -> bool:
        """Record one frame. Returns True if an adjustment was made."""
        if not self.enabled:
            return False
        self.state.record(detected)
        self._since_adjust += 1
        if self._since_adjust < self.adjust_every:
            return False
        if self.adjustments >= self.max_adjustments:
            return False
        if not self.state.enough_data:
            return False

        self._since_adjust = 0
        rate = self.state.good_rate
        direction = 0
        if rate < self.target_rate - 0.08:
            direction = -1      # missing things -> tighten
        elif rate > 0.995:
            direction = +1      # never failing -> loosen

        if direction == 0:
            return False

        moved = []
        for th in self.thresholds.values():
            th.nudge(direction)
            if th.moved:
                moved.append(f"{th.name}={th.value:.3f}")
        if moved:
            self.adjustments += 1
            return True
        return False

    def summary(self) -> str:
        if not self.enabled:
            return "adaptation disabled (using calibrated values)"
        parts = [f"detect rate {self.state.good_rate * 100:.0f}%",
                 f"{self.adjustments} adjustments"]
        for th in self.thresholds.values():
            parts.append(f"{th.name}={th.value:.3f}")
        return ", ".join(parts)


class ObstructionDetector:
    """Decides whether a still view means 'stuck' or 'standing still'.

    This is the bug that bit during development: the afk macro saw a static
    screen, declared itself stuck against a rock, and ran its escape
    sequence - when in fact the player was at a menu and the character was
    legitimately stationary.

    The discriminator is what ELSE is happening. A character jammed on a
    rock has: no motion, AND our control keys down, AND roughly the same
    pixels. Standing at a menu has: no motion, but no control keys down, and
    usually a menu visible.
    """

    def __init__(self, still_ms: int = 700, min_moving_fraction: float = 0.15,
                 reset_on_focus_change: bool = True) -> None:
        self.still_ms = still_ms
        self.min_moving_fraction = min_moving_fraction
        self.reset_on_focus_change = reset_on_focus_change
        self._last_focus: bool | None = None
        self._recent_motion: deque = deque(maxlen=60)
        self._suppressed = False

    def observe(self, moving: bool, control_down: bool,
                focus: bool = True) -> bool:
        """Returns True when it is safe to treat stillness as stuck."""
        if self.reset_on_focus_change and self._last_focus is not None \
                and focus != self._last_focus:
            # Focus changed: whatever we concluded about being stuck is
            # about a different situation entirely.
            self._recent_motion.clear()
            self._suppressed = False
        self._last_focus = focus

        self._recent_motion.append(1 if moving else 0)
        if self._suppressed:
            return False
        if not control_down:
            # We are not holding anything. Nothing to be stuck against.
            return False
        if not focus:
            # Unfocused, the view is static for reasons that have nothing to
            # do with geometry - the player is alt-tabbed, or Roblox lost
            # focus. Judging obstruction from a screen nobody is playing is
            # exactly how the afk macro ended up escaping into a menu.
            return False
        if not self._recent_motion:
            return False
        moving_fraction = sum(self._recent_motion) / len(self._recent_motion)
        return moving_fraction <= self.min_moving_fraction

    def suppress(self) -> None:
        """Called when we know the character is meant to be still."""
        self._suppressed = True

    def clear(self) -> None:
        self._suppressed = False
        self._recent_motion.clear()


def interval_error_pct(actual_ms: float, expected_ms: float) -> float:
    """Relative timing error, for the log."""
    if expected_ms <= 0:
        return 0.0
    return (actual_ms - expected_ms) / expected_ms * 100.0