"""HealthReader against the two real bar shapes in Slayers 2.

Measured from live 2560x1600 captures; see config/measured_layout.md.

The segmented case is the one that matters. A boss at full health is four
separate red chunks with dark gaps, so the original "widest red component"
logic reported a full-health boss as nearly dead. These tests exist because
that bug would have made the boss macro permanently retreat.
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from slayersmacro.detect import HealthReader, _count_runs  # noqa: E402

W, H = 400, 12          # a bar crop


# Roblox health bars are RED in BGR, so the blue channel carries the
# signal. This was the source of a confusing test failure: an earlier
# version of this file tested the blue channel as if it were red, and the
# mask matched nothing.
RED_BGR = (30, 30, 220)


def solid_bar(frac: float, w: int = W) -> np.ndarray:
    """Player-style: one solid red block filling `frac` of the track."""
    f = np.zeros((H, w, 3), np.uint8)
    f[:, :int(w * frac)] = RED_BGR
    return f


def segmented_bar(frac: float, segments: int = 4, w: int = W) -> np.ndarray:
    """Boss-style: `segments` red chunks with dark gaps, total span `frac`."""
    f = np.zeros((H, w, 3), np.uint8)
    filled_px = int(w * frac)
    gap = 4
    seg = max(1, (filled_px - gap * (segments - 1)) // segments)
    x = 0
    for i in range(segments):
        if x >= filled_px:
            break
        end = min(filled_px, x + seg)
        f[:, x:end] = RED_BGR
        x = end + gap
    return f


def is_red(frame: np.ndarray) -> np.ndarray:
    """Boolean mask of red pixels, matching the detector's own logic."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return (hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 70)


class TestSolidBars:
    @pytest.mark.parametrize("frac", [0.3, 0.5, 0.75, 0.95])
    def test_reads_fill_fraction(self, frac):
        r = HealthReader(segmented=False, _threshold=0.0)
        got = r.read(solid_bar(frac))
        assert got is not None
        assert abs(got - frac) < 0.05, f"{frac} read as {got}"

    def test_low_health_reads_as_dead_not_none(self):
        """Below `threshold` a solid bar reads as 0.0, not None.

        The distinction matters: 0.0 means "alive but nearly dead" and None
        means "no data". A boss macro that gets None forever will never
        act; one that gets 0.0 will correctly retreat.
        """
        r = HealthReader(segmented=False)
        got = r.read(solid_bar(0.25))
        assert got == 0.0

    def test_essentially_empty_returns_none(self):
        """A couple of stray pixels is a mis-drawn crop, not a health
        value, and must not be reported as a dead player."""
        r = HealthReader(segmented=False)
        f = np.zeros((H, W, 3), np.uint8)
        f[:, :3] = RED_BGR
        assert r.read(f) is None

    def test_empty_bar_returns_none(self):
        assert HealthReader(segmented=False).read(solid_bar(0.0)) is None

    def test_full_bar_returns_none(self):
        """A completely full bar is indistinguishable from a no-bar crop,
        so returning None is honest. Downstream treats None as 'unknown'."""
        r = HealthReader(segmented=False)
        assert r.read(solid_bar(1.0)) is None

    def test_none_and_empty_inputs(self):
        r = HealthReader(segmented=False)
        assert r.read(None) is None
        assert r.read(np.zeros((0, 0, 3), np.uint8)) is None


class TestSegmentedBars:
    """The bug: a four-chunk bar is not one component."""

    @pytest.mark.parametrize("frac", [1.0, 0.75, 0.5, 0.25])
    def test_reads_span_not_largest_chunk(self, frac):
        r = HealthReader(segmented=True)
        got = r.read(segmented_bar(frac))
        assert got is not None, f"{frac} read as None"
        assert abs(got - frac) < 0.08, f"{frac} read as {got:.3f}"

    def test_full_health_is_not_near_death(self):
        """The specific failure this replaces.

        Widest-component logic on a 4-segment bar returns ~0.25 for a full
        health boss, which the boss macro reads as critical.
        """
        r = HealthReader(segmented=True)
        got = r.read(segmented_bar(1.0))
        assert got is not None and got > 0.85

    def test_half_health_reads_about_half(self):
        r = HealthReader(segmented=True)
        got = r.read(segmented_bar(0.5))
        assert 0.4 < got < 0.6

    def test_more_chunks_than_widest_component(self):
        """Six chunks: the widest is a sixth, span is the whole thing."""
        f = segmented_bar(0.9, segments=6)
        # confirm the fixture really is fragmented
        cols = is_red(f).any(axis=0)
        assert _count_runs(cols) >= 5, "fixture is not actually segmented"
        r = HealthReader(segmented=True)
        assert r.read(f) > 0.8


class TestAutoDetection:
    def test_detects_segmented(self):
        r = HealthReader(segmented=None)
        got = r.read(segmented_bar(0.8))
        assert got is not None and got > 0.7

    def test_detects_solid(self):
        r = HealthReader(segmented=None)
        got = r.read(solid_bar(0.6))
        assert got is not None and abs(got - 0.6) < 0.06

    def test_anti_aliased_edges_do_not_fake_gaps(self):
        """A solid bar with soft edges must not be read as segmented, or
        it gets measured by span and over-reports."""
        f = solid_bar(0.8)
        f[:, 318:322] = (90, 90, 150)      # a dimmer band, still "filled"
        r = HealthReader(segmented=None)
        got = r.read(f)
        assert got is not None and 0.7 < got < 0.95


class TestRunCounter:
    def test_counts_runs(self):
        assert _count_runs(np.array([0, 1, 1, 0, 1, 0], bool)) == 2
        assert _count_runs(np.array([0, 0, 0], bool)) == 0
        assert _count_runs(np.array([1, 1, 1], bool)) == 1
        assert _count_runs(np.array([], bool)) == 0
