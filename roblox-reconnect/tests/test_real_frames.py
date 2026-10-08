"""Regression tests against REAL captured Roblox frames.

These exist because synthetic fixtures were not enough. Every threshold in
the classifier was first derived from dialogs I drew myself, and against a
live client the tool flagged the game's own server browser as a connection
error - 96-99% dark, 5-21% saturated, 6-8 button-shaped regions, and it
would have clicked JOIN and ejected the player from their session.

The frames in fixtures/real are captured from RobloxPlayerBeta on this
machine via Windows Graphics Capture. They are the regression evidence:
if a change to the thresholds reintroduces this false positive, these fail.

What is NOT here, and cannot be: a real Roblox connection-error dialog.
None has ever been observed on this machine. The dialog side of the
classifier is still calibrated against a synthetic frame built to
Roblox's CoreUI conventions. That is the honest limit of this coverage and
is why --check and --dry-run exist.
"""

import glob
import os

import cv2
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures", "real")

import sys  # noqa: E402
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from reconnect.ui import (  # noqa: E402
    Screen, classify, find_buttons, frame_stats, looks_like_dialog,
)

REAL_FRAMES = sorted(glob.glob(os.path.join(FIXTURES, "*.png")))


def _load(name: str) -> np.ndarray:
    path = os.path.join(FIXTURES, name)
    frame = cv2.imread(path)
    assert frame is not None, f"could not read {path}"
    return frame


def test_fixtures_are_present():
    """Guard against the fixtures being deleted, which would make this
    file silently test nothing."""
    assert REAL_FRAMES, f"no real frames in {FIXTURES}"


class TestRealFramesAreNeverErrors:
    """The single most important property: a live game frame must never
    be classified as a connection error."""

    @pytest.mark.parametrize("path", REAL_FRAMES, ids=os.path.basename)
    def test_not_an_error(self, path):
        frame = cv2.imread(path)
        det = classify(frame)
        assert det.screen is not Screen.CONNECTION_ERROR, (
            f"{os.path.basename(path)} misread as a connection error: "
            f"{det.summary()}")

    @pytest.mark.parametrize("path", REAL_FRAMES, ids=os.path.basename)
    def test_does_not_need_recovery(self, path):
        """Only blanks and errors may trigger recovery. A dark, busy,
        button-covered game screen is normal play."""
        frame = cv2.imread(path)
        det = classify(frame)
        assert not det.screen.needs_recovery, (
            f"{os.path.basename(path)} would trigger recovery: {det.summary()}")


class TestServerBrowserSpecifically:
    """The screen that caused the false positive, named explicitly so the
    regression is traceable rather than anonymous."""

    @pytest.mark.parametrize("name", ["server_browser_modal.png",
                                      "server_browser_list.png"])
    def test_server_browser_is_not_a_dialog(self, name):
        frame = _load(name)
        det = classify(frame)
        assert det.screen is Screen.IN_GAME, det.summary()

    @pytest.mark.parametrize("name", ["server_browser_modal.png",
                                      "server_browser_list.png"])
    def test_server_browser_fails_the_layout_gate(self, name):
        """Pin the actual reason, so if this ever changes the failure is
        legible instead of mysterious."""
        frame = _load(name)
        buttons = find_buttons(frame)
        assert buttons, "fixture should still contain button-shaped regions"
        ok, why = looks_like_dialog(buttons, frame.shape)
        assert not ok, f"server browser passed the modal layout gate: {why}"

    def test_server_browser_is_dark_and_saturated(self):
        """Confirms the fixture really is the hard case. If this stops
        being true, the test has stopped testing what it claims to."""
        frame = _load("server_browser_modal.png")
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dark = float((hsv[:, :, 2] < 90).mean())
        sat = float((hsv[:, :, 1] > 130).mean())
        assert dark > 0.55, f"only {dark:.0%} dark - not the hard case"
        assert sat > 0.002, f"only {sat:.2%} saturated - not the hard case"


class TestRealGameVariety:
    def test_dark_game_frame(self):
        """A dark in-game frame, the case a naive darkness threshold
        catches."""
        det = classify(_load("in_game_dark.png"))
        assert det.screen is Screen.IN_GAME, det.summary()

    def test_bright_game_frame(self):
        det = classify(_load("in_game_bright.png"))
        assert det.screen is Screen.IN_GAME, det.summary()

    def test_game_frames_are_colourful(self):
        """Game frames are richly coloured; that is the cheap signal that
        separates them from a flat-shaded dialog."""
        for name in ("in_game_dark.png", "in_game_bright.png"):
            _, buckets = frame_stats(_load(name))
            assert buckets > 45, f"{name} only occupied {buckets} buckets"


class TestLayoutGateUnit:
    """The layout gate on its own, including the shapes it must reject."""

    class _B:
        def __init__(self, x, y, w, h):
            self.x, self.y, self.w, self.h = x, y, w, h

    def _shape(self, w=2560, h=1600):
        return (h, w, 3)

    def test_two_centre_buttons_pass(self):
        bs = [self._B(800, 960, 174, 48), self._B(1586, 960, 174, 48)]
        ok, why = looks_like_dialog(bs, self._shape())
        assert ok, why

    def test_many_buttons_fail(self):
        bs = [self._B(1900, 400 + i * 110, 144, 64) for i in range(8)]
        ok, why = looks_like_dialog(bs, self._shape())
        assert not ok
        assert "modal has at most" in why

    def test_column_of_buttons_fails_on_centring(self):
        """Only two buttons, compact, but hard against the right edge - the
        shape a side panel or a column makes. Sized so the vertical-span
        gate passes and the centring gate is what rejects it."""
        bs = [self._B(1900, 700, 144, 48), self._B(1900, 760, 144, 48)]
        ok, why = looks_like_dialog(bs, self._shape())
        assert not ok
        assert "centred" in why

    def test_wide_vertical_spread_fails(self):
        bs = [self._B(1100, 200, 200, 48), self._B(1100, 1300, 200, 48)]
        ok, why = looks_like_dialog(bs, self._shape())
        assert not ok
        assert "span" in why

    def test_empty_fails(self):
        ok, why = looks_like_dialog([], self._shape())
        assert not ok
        assert why == "no buttons"

    def test_degenerate_frame_fails(self):
        ok, why = looks_like_dialog([self._B(0, 0, 100, 30)], (0, 0, 3))
        assert not ok

    def test_complex_frame_fails_even_with_perfect_layout(self):
        """The case that needed the complexity clause.

        Night gameplay: a single flat HUD rectangle, horizontally centred
        and vertically compact, so it passes all three layout tests. It is
        still a 3D render at 92/125 buckets, and must be rejected. This was
        6 false positives in 30 live frames before the clause was added.
        """
        bs = [self._B(1200, 1400, 160, 44)]
        ok_unchecked, _ = looks_like_dialog(bs, self._shape())
        assert ok_unchecked, "layout alone would accept this; fixture is wrong"
        ok, why = looks_like_dialog(bs, self._shape(), buckets=92)
        assert not ok
        assert "too complex" in why

    def test_dialog_complexity_is_accepted(self):
        """A flat-shaded dialog's bucket count must not block it."""
        bs = [self._B(800, 960, 174, 48), self._B(1586, 960, 174, 48)]
        ok, why = looks_like_dialog(bs, self._shape(), buckets=12)
        assert ok, why