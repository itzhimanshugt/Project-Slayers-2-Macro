"""Tests for detection maths. No game needed - synthetic frames only."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from slayersmacro.detect import (  # noqa: E402
    BarDetector, Calibration, HealthReader, MovementWatch, Region,
)


def make_bar(marker_col: int, track=(40, 200), width: int = 300,
             height: int = 60) -> np.ndarray:
    f = np.zeros((height, width, 3), np.uint8)
    f[:, track[0]:track[1]] = (180, 180, 180)
    f[:, marker_col:marker_col + 14] = (255, 255, 255)
    return f


class TestBarDetector:
    def test_track_centre_is_stable_regardless_of_marker(self):
        """The bug this guards: a bright marker split the track in two."""
        det = BarDetector(lock_hits=1)
        results = [det.find_track(make_bar(c)) for c in (60, 120, 180)]
        assert all(r is not None for r in results)
        assert max(results) - min(results) < 0.01
        assert abs(results[0] - 0.400) < 0.01

    @pytest.mark.parametrize("col,expect", [(60, 0.225), (120, 0.425), (180, 0.625)])
    def test_marker_position(self, col, expect):
        det = BarDetector(lock_hits=1)
        got = det.find_marker(make_bar(col))
        assert got is not None
        assert abs(got - expect) < 0.01

    def test_centred_true_only_when_overlapping(self):
        det = BarDetector(lock_hits=1)
        det.find_track(make_bar(120))
        det.find_marker(make_bar(120))
        assert det.centred(0.06) is True
        det.find_track(make_bar(60))
        det.find_marker(make_bar(60))
        assert det.centred(0.06) is False

    def test_centred_is_none_until_locked(self):
        det = BarDetector(lock_hits=3)
        det.find_marker(make_bar(120))
        assert det.centred(0.06) is None, "must not guess before 3 clean frames"

    def test_lock_hits_debounces(self):
        det = BarDetector(lock_hits=3)
        det.find_marker(make_bar(120))
        det.find_marker(make_bar(120))
        assert det.last_marker == 0.0
        det.find_marker(make_bar(120))
        assert det.last_marker > 0.4

    def test_blank_frame_returns_none(self):
        det = BarDetector(lock_hits=1)
        blank = np.zeros((60, 300, 3), np.uint8)
        assert det.find_track(blank) is None
        assert det.find_marker(blank) is None

    def test_none_input_is_safe(self):
        det = BarDetector(lock_hits=1)
        assert det.find_track(None) is None
        assert det.find_marker(None) is None

    def test_reset_clears_lock(self):
        det = BarDetector(lock_hits=2)
        det.find_marker(make_bar(120))
        det.find_marker(make_bar(120))
        assert det.last_marker > 0
        det.reset()
        assert det._marker_hits == 0


class TestHealthReader:
    def test_reads_fill_fraction(self):
        r = HealthReader()
        f = np.zeros((10, 200, 3), np.uint8)
        f[:, :120] = (40, 40, 220)
        assert abs(r.read(f) - 0.60) < 0.02

    def test_empty_bar_returns_none(self):
        r = HealthReader()
        f = np.zeros((10, 200, 3), np.uint8)
        assert r.read(f) is None

    def test_full_bar_returns_none(self):
        r = HealthReader()
        f = np.zeros((10, 200, 3), np.uint8)
        f[:, :] = (40, 40, 220)
        assert r.read(f) is None

    def test_none_input(self):
        assert HealthReader().read(None) is None


class TestMovementWatch:
    def test_motion_resets_timer(self):
        w = MovementWatch()
        a = np.zeros((60, 300, 3), np.uint8)
        b = a.copy()
        b[:, :50] = 255
        w.update(a, 5)
        w.update(b, 5)
        assert w.still_ms == 0

    def test_frozen_scene_accumulates(self):
        w = MovementWatch()
        a = np.zeros((60, 300, 3), np.uint8)
        w.update(a, 5)
        w.update(a, 5)
        assert w.still_ms == 10

    def test_none_frame_ignored(self):
        w = MovementWatch()
        assert w.update(None, 5) is False


class TestCalibrationScaling:
    def test_regions_scale_with_window(self):
        cal = Calibration(regions={"bar": Region("bar", 100, 50, 200, 20)},
                           anchor_w=1000, anchor_h=1000)
        out = cal.scaled(type("W", (), {"width": 2000, "height": 2000})())
        assert out.regions["bar"].x == 200
        assert out.regions["bar"].w == 400

    def test_no_anchor_returns_self(self):
        cal = Calibration(regions={"bar": Region("bar", 1, 2, 3, 4)})
        assert cal.scaled(type("W", (), {"width": 9, "height": 9})()) is cal

    def test_roundtrip_saves_and_loads(self, tmp_path):
        p = tmp_path / "cal.json"
        cal = Calibration(regions={"bar": Region("bar", 5, 6, 7, 8)},
                          anchor_w=800, anchor_h=600)
        cal.save(p)
        back = Calibration.load(p)
        assert back.regions["bar"].x == 5
        assert back.anchor_h == 600