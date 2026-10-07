"""Tests for the safety guard.

The scenarios here are the ones that actually bit us: a Roblox window
existing while the game is not visible, and a window that is moving.
"""

import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from slayersmacro.guard import (  # noqa: E402
    GuardConfig, GuardState, SafetyGuard, UnsafeStateError,
)


class FakeWin:
    def __init__(self, hwnd=1, left=0, top=0, width=800, height=600):
        self.hwnd = hwnd
        self.left, self.top = left, top
        self.width, self.height = width, height
        self.title = "Roblox"


def noisy(h=60, w=200, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


class TestGuardBlocksUnsafeStates:
    def test_missing_window_blocks(self):
        g = SafetyGuard()
        r = g.check(None, noisy())
        assert r.state is GuardState.NO_WINDOW
        assert r.should_act is False

    def test_unfocused_window_blocks(self):
        g = SafetyGuard(GuardConfig(stable_for_ms=0, settle_frames=1))
        g._last_rect = (0, 0, 800, 600)      # pretend it settled
        g._rect_since = time.perf_counter() - 10
        # hwnd 1 is never the foreground window in a headless test, so this
        # exercises the focus requirement rather than the settle timer.
        r = g.check(FakeWin(), noisy())
        assert r.state is GuardState.NOT_FOCUSED
        assert r.should_act is False

    @staticmethod
    def _settled(g):
        """Fast-forward the window-stability clock so tests exercise the
        check they are actually about rather than the stability timer."""
        g._last_rect = (0, 0, 800, 600)
        g._rect_since = time.perf_counter() - 10
        g.cfg.require_focus = False
        return g

    def test_missing_capture_blocks(self):
        g = self._settled(SafetyGuard(GuardConfig(stable_for_ms=0)))
        r = g.check(FakeWin(), None)
        assert r.state is GuardState.OCCLUDED
        assert r.should_act is False

    def test_static_image_identical_to_reference_blocks(self):
        """The bug that motivated this module.

        A Roblox window existed, but something else was covering it, so the
        capture returned that other window's pixels. A macro acting on those
        coordinates would be clicking on a browser.
        """
        g = self._settled(SafetyGuard(GuardConfig(stable_for_ms=0)))
        ref = noisy(seed=4)
        r = g.check(FakeWin(), ref.copy(), reference_frame=ref)
        assert r.state is GuardState.OCCLUDED
        assert r.should_act is False

    def test_moving_window_blocks(self):
        g = SafetyGuard(GuardConfig(stable_for_ms=500))
        r1 = g.check(FakeWin(left=0), noisy())
        assert r1.should_act is False
        r2 = g.check(FakeWin(left=100), noisy())
        assert r2.state is GuardState.UNSTABLE
        assert r2.should_act is False

    def test_settle_frames_required(self):
        """Even a perfect environment needs N clean frames before acting."""
        g = self._settled(SafetyGuard(GuardConfig(settle_frames=3,
                                                 stable_for_ms=0)))
        seen = [g.check(FakeWin(), noisy(seed=i)).should_act for i in range(5)]
        # settle_frames=3 means the 3rd consecutive clean frame is the first
        # one allowed to act. Frames 1 and 2 must be blocked.
        assert seen[0] is False and seen[1] is False
        assert seen[2:] == [True, True, True]

    def test_state_change_resets_streak(self):
        g = self._settled(SafetyGuard(GuardConfig(settle_frames=2,
                                                 stable_for_ms=0)))
        g.check(FakeWin(), noisy(seed=1))
        g.check(None, None)                       # go unsafe
        assert g._ok_streak == 0
        self._settled(g)
        assert g.check(FakeWin(), noisy(seed=2)).should_act is False


class TestGuardAbort:
    def test_aborts_after_timeout(self):
        g = SafetyGuard(GuardConfig(abort_after=0.05, stable_for_ms=0))
        with pytest.raises(UnsafeStateError):
            for _ in range(50):
                time.sleep(0.005)
                g.check(None, None)

    def test_no_abort_when_disabled(self):
        g = SafetyGuard(GuardConfig(abort_after=0.0))
        for _ in range(10):
            g.check(None, None)
        assert g._last_state is GuardState.NO_WINDOW


class TestFingerprint:
    def test_same_pixels_same_hash(self):
        g = SafetyGuard()
        a = noisy(seed=3)
        assert g.fingerprint(a) == g.fingerprint(a.copy())

    def test_different_pixels_differ(self):
        g = SafetyGuard()
        assert g.fingerprint(noisy(seed=3)) != g.fingerprint(noisy(seed=9))

    def test_none_and_empty(self):
        g = SafetyGuard()
        assert g.fingerprint(None) == "empty"
        assert g.fingerprint(np.zeros((0, 0, 3), np.uint8)) == "empty"

    def test_cheap_on_full_frame(self):
        """Must not be the bottleneck: it runs every frame."""
        g = SafetyGuard()
        big = np.random.randint(0, 255, (1600, 2560, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        for _ in range(50):
            g.fingerprint(big)
        per = (time.perf_counter() - t0) / 50
        assert per < 0.01, f"fingerprint took {per * 1000:.2f}ms"