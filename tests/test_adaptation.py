"""Tests for adaptation: timing learning, online threshold drift, and the
obstruction-vs-standing-still discriminator.

The obstruction test is the important one. Getting it wrong is what caused
a macro to run its escape sequence against a menu screen.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from slayersmacro.adaptation import (  # noqa: E402
    AdaptiveThreshold, AdaptationState, ObstructionDetector,
    OnlineAdaptation, TimingProfile,
)


class TestTimingProfile:
    def test_starts_at_target(self):
        p = TimingProfile(target_ms=90.0)
        assert p.interval_ms == 90.0
        assert p.learned is False

    def test_converges_to_observed_median(self):
        p = TimingProfile(target_ms=90.0)
        for _ in range(40):
            p.observe(75.0)
        assert p.interval_ms == pytest.approx(75.0, abs=1.0)
        assert p.learned is True

    def test_median_ignores_outliers(self):
        """One lag spike must not poison the average."""
        p = TimingProfile(target_ms=90.0)
        for _ in range(30):
            p.observe(80.0)
        p.observe(4000.0)
        assert p.interval_ms == pytest.approx(80.0, abs=2.0)

    def test_rejects_absurd_intervals(self):
        p = TimingProfile(target_ms=90.0)
        before = p.interval_ms
        p.observe(-5.0)
        p.observe(1e9)
        assert p.interval_ms == before

    def test_slow_adaptation_is_stable(self):
        """A jump in input should move the profile, not teleport it."""
        p = TimingProfile(target_ms=90.0)
        for _ in range(30):
            p.observe(90.0)
        p.observe(300.0)          # one wild sample
        assert p.interval_ms < 150.0, "must not lurch on a single sample"

    def test_save_load_roundtrip(self, tmp_path):
        p = TimingProfile(target_ms=90.0)
        for _ in range(30):
            p.observe(70.0)
        f = p.save(tmp_path / "t.json")
        back = TimingProfile.load(f, target_ms=90.0)
        assert back.interval_ms == pytest.approx(70.0, abs=1.0)
        assert back.learned is True

    def test_load_clamps_drift(self, tmp_path):
        """A saved profile must not be able to move the interval wildly."""
        (tmp_path / "t.json").write_text(
            '{"target_ms": 90, "learned_ms": 9, "is_learned": true}')
        back = TimingProfile.load(tmp_path / "t.json", target_ms=90.0)
        assert back.target_ms >= 90.0 * 0.5


class TestAdaptiveThreshold:
    def test_clamps_to_bounds(self):
        t = AdaptiveThreshold("lo", 0.5, 0.5, 0.1, 0.9, step=0.2)
        for _ in range(10):
            t.nudge(-1)
        assert t.value == pytest.approx(0.1)
        for _ in range(20):
            t.nudge(1)
        assert t.value == pytest.approx(0.9)

    def test_reports_movement(self):
        t = AdaptiveThreshold("lo", 0.5, 0.5, 0.1, 0.9, step=0.05)
        assert t.moved is False
        t.nudge(1)
        assert t.moved is True


class TestAdaptationState:
    def test_counts(self):
        s = AdaptationState(window=10)
        for i in range(6):
            s.record(True)
        for i in range(4):
            s.record(False)
        assert s.good_rate == pytest.approx(0.6)
        assert s.enough_data is True

    def test_empty_is_trusted(self):
        assert AdaptationState().good_rate == 1.0
        assert AdaptationState().enough_data is False

    def test_window_slides(self):
        s = AdaptationState(window=5)
        for _ in range(5):
            s.record(False)
        assert s.good_rate == 0.0
        for _ in range(5):
            s.record(True)
        assert s.good_rate == 1.0


class TestOnlineAdaptation:
    def _adapt(self, good):
        a = OnlineAdaptation(enabled=True, target_rate=0.92,
                             adjust_every=20, max_adjustments=40)
        a.register(AdaptiveThreshold("lo", 0.5, 0.5, 0.1, 0.9, step=0.05))
        for _ in range(good):
            a.record(True)
        for _ in range(good):
            a.record(False)
        return a

    def test_disabled_never_adapts(self):
        a = OnlineAdaptation(enabled=False)
        a.register(AdaptiveThreshold("lo", 0.5, 0.5, 0.1, 0.9))
        for _ in range(200):
            a.record(False)
        assert a.adjustments == 0
        assert "disabled" in a.summary()

    def test_tightens_when_missing(self):
        a = self._adapt(60)
        assert a.thresholds["lo"].value < 0.5, "low detection must raise lo"

    def test_no_adjustment_when_rate_is_healthy(self):
        a = OnlineAdaptation(enabled=True, target_rate=0.92, adjust_every=20)
        a.register(AdaptiveThreshold("lo", 0.5, 0.5, 0.1, 0.9, step=0.05))
        for i in range(60):
            a.record(i % 10 < 9)        # 90% detection
        assert a.adjustments == 0

    def test_does_not_oscillate(self):
        """The point of one-directional adaptation.

        Feed alternating good/bad blocks and confirm the threshold settles
        instead of swinging across its whole range.
        """
        a = OnlineAdaptation(enabled=True, target_rate=0.92,
                             adjust_every=20, max_adjustments=200)
        a.register(AdaptiveThreshold("lo", 0.5, 0.5, 0.0, 1.0, step=0.05))
        seen = []
        for block in range(12):
            good = block % 2 == 0
            for _ in range(20):
                a.record(good)
            seen.append(a.thresholds["lo"].value)
        reversals = sum(1 for i in range(1, len(seen))
                        if (seen[i] - seen[i - 1]) * (seen[i - 1] - seen[i - 2]) < 0)
        assert reversals <= 1, f"oscillating: {seen}"

    def test_adjustment_budget_is_finite(self):
        a = OnlineAdaptation(enabled=True, adjust_every=10, max_adjustments=5)
        a.register(AdaptiveThreshold("lo", 0.5, 0.5, 0.0, 1.0, step=0.01))
        for _ in range(2000):
            a.record(False)
        assert a.adjustments <= 5


class TestObstructionDetector:
    def test_still_with_no_input_is_not_stuck(self):
        """THE bug: a menu screen with nothing held is not an obstruction."""
        d = ObstructionDetector(still_ms=0)
        for _ in range(30):
            assert d.observe(moving=False, control_down=False) is False

    def test_still_while_holding_moving_key_is_stuck(self):
        d = ObstructionDetector(still_ms=0)
        got = [d.observe(moving=False, control_down=True) for _ in range(30)]
        assert got[-1] is True

    def test_motion_suppresses_stuck(self):
        d = ObstructionDetector(still_ms=0)
        for _ in range(30):
            assert d.observe(moving=True, control_down=True) is False

    def test_unfocused_is_never_stuck(self):
        d = ObstructionDetector(still_ms=0)
        for _ in range(30):
            assert d.observe(moving=False, control_down=True,
                             focus=False) is False

    def test_focus_change_clears_history(self):
        """Losing focus mid-stuck must not leave a stale verdict behind."""
        d = ObstructionDetector(still_ms=0)
        for _ in range(20):
            d.observe(moving=False, control_down=True, focus=True)
        assert d.observe(moving=False, control_down=True, focus=False) is False
        for _ in range(30):
            d.observe(moving=False, control_down=True, focus=False)
        # Still suppressed while unfocused.
        assert d.observe(moving=False, control_down=True, focus=False) is False

    def test_explicit_suppress(self):
        """Fishing holds a key while standing still - that is not stuck."""
        d = ObstructionDetector(still_ms=0)
        d.suppress()
        for _ in range(30):
            assert d.observe(moving=False, control_down=True) is False
        d.clear()
        got = [d.observe(moving=False, control_down=True) for _ in range(30)]
        assert got[-1] is True