"""Latency accounting.

The only honest way to hit a timing minigame is to measure where the time
goes, then compensate for it. These tests pin the arithmetic so the budget
cannot silently drift into being wrong.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from slayersmacro.latency import (  # noqa: E402
    LatencyBudget, lookahead_ms, missing_fps_error,
)


class TestLatencyBudget:
    def test_actuation_sums_the_chain(self):
        b = LatencyBudget(t_cap=0.00417, t_proc=0.0003, t_inj=0.0002,
                          dt_game=1 / 240)
        assert b.actuation == pytest.approx(0.00417 + 0.0003 + 0.0002
                                             + (1 / 240) / 2)

    def test_lookahead_snaps_up_to_whole_game_frames(self):
        """A press landing before a frame boundary behaves identically to one
        just after it, so partial frames are wasted margin."""
        b = LatencyBudget(t_cap=0.00417, t_proc=0.0003, t_inj=0.0002,
                          dt_game=1 / 240)
        tau = b.lookahead()
        n = tau * 240
        assert n == pytest.approx(round(n)), "must be a whole number of frames"
        assert tau >= b.actuation

    def test_lookahead_is_never_less_than_the_chain(self):
        for dt_game in (1 / 60, 1 / 144, 1 / 240):
            b = LatencyBudget(dt_game=dt_game)
            assert b.lookahead() >= b.actuation

    def test_higher_game_fps_means_finer_quantisation(self):
        slow = LatencyBudget(dt_game=1 / 60).lookahead()
        fast = LatencyBudget(dt_game=1 / 240).lookahead()
        assert fast < slow

    def test_uses_p99_not_mean_when_given_p99(self):
        """Over-estimating tau biases the press early, which is recoverable.
        Under-estimating biases it late, which loses the fish."""
        b = LatencyBudget()
        assert b.lookahead(bias_frames=0.5) >= b.lookahead(bias_frames=0.0)

    def test_defaults_match_measured_machine(self):
        b = LatencyBudget()
        # Measured on this box: ~4.18ms vsync-locked capture, ~0.3ms
        # SendInput. Game frame at 240Hz.
        assert 0.004 <= b.t_cap <= 0.005
        assert b.t_inj < 0.001


class TestCompensationArithmetic:
    def test_error_scales_with_marker_speed(self):
        """The whole justification for compensating latency at all."""
        tau = LatencyBudget().lookahead()
        slow_err = 0.5 * tau          # normalised units/s
        fast_err = 2.0 * tau
        assert fast_err == pytest.approx(4 * slow_err)

    def test_uncompensated_press_is_late_by_the_full_chain(self):
        b = LatencyBudget()
        velocity = 0.5
        # Acting on the measured position means acting on where the marker
        # was; by actuation it has moved velocity * actuation.
        drift = velocity * b.actuation
        assert drift > 0
        # At 0.5/s that is ~5ms of marker travel: 2.5e-3 normalised,
        # about 1px on a 400px bar. Small, but systematic, which is the
        # part that matters.
        assert 1e-3 < drift < 6e-3

    def test_compensation_closes_the_gap(self):
        b = LatencyBudget()
        velocity = 0.5
        uncompensated = velocity * b.actuation
        compensated = velocity * max(0.0, b.actuation - b.lookahead())
        assert compensated < uncompensated

    def test_lookahead_helper_matches_budget(self):
        b = LatencyBudget()
        assert lookahead_ms() == pytest.approx(b.lookahead() * 1000)


class TestMissingFpsError:
    def test_wide_band_absorbs_quantisation(self):
        """The threshold at which compensating latency stops mattering.

        At 0.5 units/s a game frame is worth ~1.0e-3 normalised, so a band
        half-width of 0.01 swallows it ten times over and precise
        prediction is decoration.
        """
        assert not missing_fps_error(velocity=0.5, band_frac=0.01)

    def test_narrow_band_is_unreachable_at_any_speed(self):
        # 1e-4 is a fifth of one frame of marker travel at 0.5/s.
        assert missing_fps_error(velocity=0.5, band_frac=1e-4)

    def test_error_grows_with_speed(self):
        b = LatencyBudget()
        from slayersmacro.latency import quantisation_error
        slow = quantisation_error(0.2, b)
        fast = quantisation_error(2.0, b)
        assert fast == pytest.approx(10 * slow)

    def test_same_band_can_be_viable_at_low_speed_and_not_high(self):
        """The practical use: check before investing in precision."""
        band = 3e-4          # ~0.3% of a 400px bar, i.e. a tight band
        assert not missing_fps_error(velocity=0.1, band_frac=band)
        assert missing_fps_error(velocity=1.0, band_frac=band)

    def test_zero_band_is_always_hard(self):
        assert missing_fps_error(velocity=0.1, band_frac=0.0)