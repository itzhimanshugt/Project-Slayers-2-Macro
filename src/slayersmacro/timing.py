"""Humanised timing.

Why not random jitter in [-15, +15]: because it is wrong in two ways.

First, uniform noise is symmetric and obviously synthetic. Second, and
worse, it is uncorrelated, which produces a signal with high variance at
exactly the timescale an anti-cheat would measure. Roblox's own server-side
detection docs name "action cadence" as the signal: identical intervals are
the giveaway. What a human produces is not identical intervals, and the
deviation is *correlated over time* - your hands are not independent noise
sources.

So: log-normal (right-skewed, matches human reaction-time data) plus an
AR(1) filter for correlation. Two parameters, no ML needed.
"""

from __future__ import annotations

import random


class JitterModel:
    """Generates human-plausible intervals around a target.

    Intervals are log-normal: mostly near the target with a longer right
    tail, which is how human timing actually behaves. Successive samples are
    correlated via an AR(1) process so the sequence does not look like white
    noise.
    """

    def __init__(self, magnitude: float = 20.0, sigma: float = 0.22,
                 smoothing: float = 0.8, seed: int | None = None,
                 min_ms: float = 1.0) -> None:
        """
        magnitude: mean deviation in ms. 0 disables jitter entirely.

        sigma: log-normal spread. 0.22 gives roughly +/-25% at one sigma,
            which sits in the range of real human keypress timing. Higher
            looks less robotic but also degrades a timing minigame, so keep
            it modest where accuracy matters.

        smoothing: AR(1) coefficient, 0 = independent samples, 0.9 = slow
            drift. Around 0.8 reproduces the "lumpy" feel of human timing
            without drifting far from the target.
        """
        self.magnitude = magnitude
        self.sigma = sigma
        self.smoothing = smoothing
        self.min_ms = min_ms
        self._rng = random.Random(seed)
        self._state = 0.0

    def sample(self, target_ms: float) -> float:
        if self.magnitude <= 0 or target_ms <= 0:
            return max(0.0, target_ms)
        # Log-normal deviation, centred on 1.0 so the mean is ~target.
        raw = self._rng.lognormvariate(0.0, self.sigma)
        centred = raw - 1.0
        self._state = (self.smoothing * self._state
                       + (1.0 - self.smoothing) * centred)
        value = target_ms + self.magnitude * self._state
        return max(self.min_ms, value)

    def sleep_ms(self, target_ms: float) -> None:
        """Jitter and sleep. Returns the value slept, for logging."""
        import time
        v = self.sample(target_ms)
        time.sleep(v / 1000.0)
        return v