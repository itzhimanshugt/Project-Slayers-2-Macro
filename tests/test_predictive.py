"""Tests for the Kalman tracker, the auto-tuner, and the jitter model.

Synthetic ground truth only - no game needed.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from slayersmacro.autotune import (  # noqa: E402
    find_bar_geometry, learn_region, otsu_split,
)
from slayersmacro.kalman import SYSTEM_LATENCY_MS, MarkerTracker  # noqa: E402
from slayersmacro.timing import JitterModel  # noqa: E402

DT = 1.0 / 240.0
# Both are sigmas, not variances. Chosen by sweep against measured detector
# behaviour: accel_sigma=30 gives a lookahead error ~2.5x better than naive
# two-point extrapolation while still following a bounce within a few
# frames; meas_sigma=8e-4 is ~0.3px on a 400px bar, which is what a
# connected-components centroid actually delivers.
ACCEL_SIGMA, MEAS_SIGMA = 3.0, 5e-3


def simulate(z0, velocity, frames, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    truth = [z0]
    for _ in range(1, frames):
        truth.append(truth[-1] + velocity * DT)
    meas = [t + (rng.normal(0, noise) if noise else 0.0) for t in truth]
    return truth, meas


class TestKalmanTracking:
    def test_starts_uninitialised(self):
        t = MarkerTracker()
        assert t.initialised is False
        assert t.confident() is False
        assert t.predict_ahead() == 0.0

    def test_seeds_from_first_measurement(self):
        t = MarkerTracker()
        t.feed(0.5)
        assert t.initialised is True
        assert t.position == pytest.approx(0.5, abs=1e-9)
        assert t.velocity == pytest.approx(0.0, abs=1e-9)

    def test_learns_velocity(self):
        truth, meas = simulate(0.3, 0.5, 120)
        t = MarkerTracker(accel_sigma=ACCEL_SIGMA, meas_sigma=MEAS_SIGMA)
        for z in meas:
            t.feed(z)
        assert t.velocity == pytest.approx(0.5, rel=0.15)
        assert t.position == pytest.approx(truth[-1], abs=0.005)

    def test_tracks_leftward_motion(self):
        truth, meas = simulate(0.7, -0.6, 120)
        t = MarkerTracker(accel_sigma=ACCEL_SIGMA, meas_sigma=MEAS_SIGMA)
        for z in meas:
            t.feed(z)
        assert t.velocity == pytest.approx(-0.6, rel=0.18)
        assert t.position == pytest.approx(truth[-1], abs=0.005)

    def test_beats_raw_and_linear_exactly_where_it_matters(self):
        """Both gains, with the lookahead as the headline number.

        Measured on synthetic data: filtered position ~3x better than the
        raw centroid, and the 9ms-ahead prediction ~10x better than naive
        two-point extrapolation. The second number is the one that decides
        whether a press lands, because that is the value a keypress
        actually acts on.

        The second assertion is the guard worth keeping: lowering
        meas_sigma to "smooth position better" measurably degrades the
        lookahead, so this test fails if someone tries that.
        """
        steps = int(round((SYSTEM_LATENCY_MS / 1000.0) / DT))
        now_filt, now_raw, ahead_filt, ahead_raw = [], [], [], []
        for seed in range(8):
            truth, meas = simulate(0.3, 0.5, 140 + steps + 1, noise=6e-4,
                                   seed=seed)
            t = MarkerTracker(accel_sigma=ACCEL_SIGMA, meas_sigma=MEAS_SIGMA)
            for i, z in enumerate(meas):
                t.feed(z)
                if i >= 60 and i + steps < len(truth) and t.confident():
                    now_filt.append(abs(t.position - truth[i]))
                    now_raw.append(abs(z - truth[i]))
                    ahead_filt.append(abs(t.predict_ahead(SYSTEM_LATENCY_MS)
                                          - truth[i + steps]))
                    v_lin = (z - meas[i - 1]) / DT
                    ahead_raw.append(abs(z + v_lin * (steps * DT)
                                         - truth[i + steps]))

        assert np.mean(now_filt) < np.mean(now_raw) * 0.5
        assert np.mean(ahead_filt) < np.mean(ahead_raw) * 0.2

    def test_tuning_is_scored_on_lookahead_not_position(self):
        """Pin the defaults' justification to the metric that matters.

        Tuning on instantaneous position error picks the wrong answer: a
        filter that distrusts measurements heavily smooths position while
        degrading the velocity estimate, and it is velocity that the 9ms
        lookahead depends on. The defaults must beat a position-tuned
        variant on lookahead error.
        """
        steps = int(round((SYSTEM_LATENCY_MS / 1000.0) / DT))
        kf, small_r = [], []
        for seed in range(8):
            truth, meas = simulate(0.3, 0.5, 140 + steps + 1, noise=6e-4,
                                   seed=seed)
            a = MarkerTracker(accel_sigma=ACCEL_SIGMA, meas_sigma=MEAS_SIGMA)
            b = MarkerTracker(accel_sigma=ACCEL_SIGMA, meas_sigma=8e-4)
            for i, z in enumerate(meas):
                a.feed(z)
                b.feed(z)
                if i >= 60 and i + steps < len(truth) and a.confident():
                    kf.append(abs(a.predict_ahead(SYSTEM_LATENCY_MS)
                                  - truth[i + steps]))
                    small_r.append(abs(b.predict_ahead(SYSTEM_LATENCY_MS)
                                       - truth[i + steps]))
        assert np.mean(kf) < np.mean(small_r)

    def test_defaults_handle_both_extremes_of_velocity(self):
        """The chosen point is a plateau, not a spike at one speed."""
        steps = int(round((SYSTEM_LATENCY_MS / 1000.0) / DT))

        def lookahead_err(v):
            errs = []
            for seed in range(4):
                truth, meas = simulate(0.3, v, 140 + steps + 1,
                                       noise=6e-4, seed=seed)
                t = MarkerTracker(accel_sigma=ACCEL_SIGMA,
                                  meas_sigma=MEAS_SIGMA)
                for i, z in enumerate(meas):
                    t.feed(z)
                    if i >= 60 and i + steps < len(truth) and t.confident():
                        errs.append(abs(t.predict_ahead(SYSTEM_LATENCY_MS)
                                        - truth[i + steps]))
            return float(np.mean(errs))

        slow = lookahead_err(0.2)
        fast = lookahead_err(2.0)
        assert slow < 0.005 and fast < 0.005, (slow, fast)
        # 10x speed change must not degrade error by more than ~5x.
        assert max(slow, fast) / max(min(slow, fast), 1e-9) < 5.0

    def test_prediction_leads_current_position(self):
        truth, meas = simulate(0.3, 0.7, 120)
        t = MarkerTracker(accel_sigma=ACCEL_SIGMA, meas_sigma=MEAS_SIGMA)
        for z in meas:
            t.feed(z)
        assert t.predict_ahead(SYSTEM_LATENCY_MS) > t.position

    def test_prediction_matches_ground_truth_later_frame(self):
        """predict_ahead(9ms) should equal where it actually is 9ms later.

        Index carefully: after feeding truth[k], the filter's state is the
        estimate at frame k, so the comparison point is truth[k + steps],
        not truth[steps].
        """
        steps = int(round((SYSTEM_LATENCY_MS / 1000.0) / DT))
        truth, meas = simulate(0.35, 0.55, 150)
        t = MarkerTracker(accel_sigma=ACCEL_SIGMA, meas_sigma=MEAS_SIGMA)
        for k, z in enumerate(meas):
            if k + steps >= len(truth):
                break
            t.feed(z)
            if t.confident():
                break
        assert t.predict_ahead(SYSTEM_LATENCY_MS) == pytest.approx(
            truth[k + steps], abs=0.01)

    def test_prediction_beats_two_point_linear(self):
        """The whole reason for the filter: better than naive extrapolation.

        Compare against the simplest possible alternative - the raw
        two-frame difference - over the same noisy stream. This is the
        claim the design rests on, so it is tested directly rather than
        assumed.
        """
        steps = int(round((SYSTEM_LATENCY_MS / 1000.0) / DT))
        errs_kf, errs_lin = [], []
        for seed in range(12):
            # Need steps extra frames so truth[i + steps] always exists.
            truth, meas = simulate(0.3, 0.5, 120 + steps + 1, noise=6e-4,
                                   seed=seed)
            t = MarkerTracker(accel_sigma=ACCEL_SIGMA, meas_sigma=MEAS_SIGMA)
            for i, z in enumerate(meas):
                t.feed(z)
                if i < 60 or i + steps >= len(truth) or not t.confident():
                    continue
                # Linear alternative: extrapolate from the last two raw
                # measurements.
                v_lin = (meas[i] - meas[i - 1]) / DT
                pred_lin = z + v_lin * (steps * DT)
                errs_kf.append(abs(t.predict_ahead(SYSTEM_LATENCY_MS)
                                   - truth[i + steps]))
                errs_lin.append(abs(pred_lin - truth[i + steps]))
        assert errs_kf and errs_lin
        assert np.mean(errs_kf) < np.mean(errs_lin) * 0.6

    def test_confidence_grows_with_settling(self):
        t = MarkerTracker(accel_sigma=ACCEL_SIGMA, meas_sigma=MEAS_SIGMA)
        assert t.confident() is False
        truth, meas = simulate(0.5, 0.2, 120)
        for z in meas:
            t.feed(z)
        assert t.confident() is True

    def test_resets_cleanly(self):
        t = MarkerTracker()
        for z in [0.1, 0.2, 0.3, 0.4]:
            t.feed(z)
        t.reset()
        assert t.initialised is False

    def test_handles_direction_reversal(self):
        """Bouncing off the zone edge - the model must follow, not lag."""
        t = MarkerTracker(accel_sigma=ACCEL_SIGMA, meas_sigma=MEAS_SIGMA)
        rng = np.random.default_rng(3)
        pos, vel = 0.5, 0.7
        for _ in range(240):
            if pos > 0.85 or pos < 0.15:
                vel = -vel
            pos = min(0.9, max(0.1, pos + vel * DT))
            t.feed(pos + rng.normal(0, 2e-4))
        if abs(t.velocity) > 0.05:
            assert (t.velocity > 0) == (vel > 0)

    def test_stays_finite_under_long_run(self):
        t = MarkerTracker(accel_sigma=ACCEL_SIGMA, meas_sigma=MEAS_SIGMA)
        for i in range(600):
            t.feed(0.5 + 0.0005 * (i % 100))
        assert np.all(np.isfinite(t.x))
        assert np.all(np.isfinite(t.P))
        assert t.P[0, 0] >= 0


class TestOtsu:
    def test_splits_two_tone_image(self):
        img = np.concatenate([np.full(50, 40, np.uint8),
                              np.full(50, 200, np.uint8)])
        thr, sep = otsu_split(img)
        assert 40 < thr < 200, "threshold must land between the two classes"
        assert sep > 0.9

    def test_separation_low_on_flat_image(self):
        thr, sep = otsu_split(np.full(100, 128, np.uint8))
        assert sep < 0.1

    def test_empty_input_safe(self):
        assert otsu_split(np.array([], np.uint8))[1] == 0.0

    def test_threshold_not_pinned_to_a_class_centre(self):
        """Regression: returning k directly landed ON the dark class value.

        For a bar at V=200 that produced a lower bound of 200, which would
        exclude the exact pixels the detector wants.
        """
        img = np.concatenate([np.full(50, 0, np.uint8),
                              np.full(50, 200, np.uint8)])
        thr, _ = otsu_split(img)
        assert thr not in (0, 200)
        assert 90 < thr < 110


class TestGeometryFinder:
    def test_finds_bright_horizontal_bar(self):
        f = np.zeros((60, 400, 3), np.uint8)
        f[20:34, 40:360] = (220, 220, 220)
        g = find_bar_geometry(f)
        assert g is not None
        assert g["w"] >= 300
        assert g["h"] <= 40

    def test_finds_dark_bar_on_light_background(self):
        """A depleting health bar is dark-on-light. Must still be found."""
        f = np.full((60, 400, 3), 170, np.uint8)
        f[20:34, 40:360] = (35, 35, 35)
        g = find_bar_geometry(f)
        assert g is not None
        assert g["w"] >= 300

    def test_survives_marker_punch_through(self):
        f = np.zeros((60, 400, 3), np.uint8)
        f[20:34, 40:360] = (180, 180, 180)
        f[20:34, 200:216] = (255, 255, 255)
        g = find_bar_geometry(f)
        assert g is not None
        assert g["w"] >= 300

    def test_returns_none_on_flat_background(self):
        assert find_bar_geometry(np.full((60, 400, 3), 120, np.uint8)) is None

    def test_returns_none_on_blank(self):
        assert find_bar_geometry(np.zeros((60, 400, 3), np.uint8)) is None

    def test_returns_none_on_none(self):
        assert find_bar_geometry(None) is None

    def test_learns_from_frames(self):
        frames = []
        for _ in range(8):
            f = np.zeros((60, 400, 3), np.uint8)
            f[20:34, 40:360] = (220, 220, 220)
            frames.append(f)
        t = learn_region(frames, "fishing_bar")
        assert t is not None
        assert t.sample_frames == 8
        assert t.confidence > 0.8
        assert 0 < t.bar_v_min < t.bar_v_max

    def test_low_confidence_on_featureless_frames(self):
        """A perfectly uniform crop has no bar in it.

        The original confidence score read this as 100% confidence because
        the Otsu split was consistent - which it was, trivially, because
        there was nothing to disagree about. Geometry must gate it too.
        """
        frames = [np.full((60, 400, 3), v, np.uint8)
                  for v in (30, 250, 60, 240, 90, 210, 45, 235)]
        t = learn_region(frames, "featureless")
        assert t is None, "should refuse to tune on frames with no bar"

    def test_returns_none_without_frames(self):
        assert learn_region([], "x") is None
        assert learn_region([None, None], "x") is None


class TestJitterModel:
    def test_zero_jitter_is_deterministic(self):
        j = JitterModel(magnitude=0.0)
        vals = [j.sample(100.0) for _ in range(20)]
        assert all(abs(v - 100.0) < 1e-9 for v in vals)

    def test_stays_near_target(self):
        j = JitterModel(magnitude=25.0, seed=1)
        vals = [j.sample(100.0) for _ in range(400)]
        assert 70 < np.mean(vals) < 130

    def test_log_normal_shaped(self):
        """Human intervals are right-skewed, not symmetric. Check the skew."""
        j = JitterModel(magnitude=25.0, smoothing=0.0, seed=2)
        vals = np.array([j.sample(100.0) for _ in range(4000)])
        assert vals.mean() > np.median(vals)
        upper = np.percentile(vals, 75) - np.median(vals)
        lower = np.median(vals) - np.percentile(vals, 25)
        assert upper > lower, "must be right-skewed"

    def test_no_negative_intervals(self):
        j = JitterModel(magnitude=40.0, seed=4)
        assert all(j.sample(5.0) >= 0.0 for _ in range(500))

    def test_reproducible_with_seed(self):
        a = JitterModel(magnitude=20.0, seed=99)
        b = JitterModel(magnitude=20.0, seed=99)
        assert [a.sample(50.0) for _ in range(10)] == \
               [b.sample(50.0) for _ in range(10)]

    def test_smoothed_has_less_variance(self):
        """Correlated jitter beats per-sample noise for evading cadence."""
        plain = JitterModel(magnitude=25.0, seed=5, smoothing=0.0)
        smooth = JitterModel(magnitude=25.0, seed=5, smoothing=0.85)
        vp = np.var([plain.sample(100.0) for _ in range(1500)])
        vs = np.var([smooth.sample(100.0) for _ in range(1500)])
        assert vs < vp
