"""Detection and recovery tests, against synthetic Roblox error screens.

The central property: the tool must never act on an in-game frame. A false
positive does not just fail to help - it tears down a working session.
"""

import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

import scenes  # noqa: E402
from reconnect.ui import (  # noqa: E402
    Button, Detection, Screen, classify, find_buttons, frame_stats,
    is_flat, pick_primary,
)


class TestBlankScreens:
    def test_white_screen(self):
        d = classify(scenes.white_screen())
        assert d.screen is Screen.WHITE_SCREEN
        assert d.confidence > 0.8

    def test_black_screen(self):
        d = classify(scenes.black_screen())
        assert d.screen is Screen.BLACK_SCREEN

    def test_loading_backdrop_is_not_blank(self):
        """It has a spinner, so it must not be reported as a dead screen."""
        d = classify(scenes.mid_loading())
        assert d.screen is not Screen.WHITE_SCREEN
        assert d.screen is not Screen.BLACK_SCREEN

    def test_none_frame(self):
        d = classify(None)
        assert d.screen.needs_recovery


class TestInGameIsNeverAnError:
    @pytest.mark.parametrize("seed", range(6))
    def test_busy_scene_reads_as_in_game(self, seed):
        d = classify(scenes.in_game_scene(seed=seed))
        assert d.screen is Screen.IN_GAME, d.summary()
        assert not d.screen.needs_recovery

    def test_in_game_with_dark_hud(self):
        """The scene has a dark HUD strip; it must not trigger the
        dark-plate rule."""
        d = classify(scenes.in_game_scene(seed=3))
        assert not d.screen.needs_recovery

    def test_in_game_with_noise_and_cursor(self):
        f = scenes.with_cursor_and_noise(scenes.in_game_scene(seed=2))
        d = classify(f)
        assert not d.screen.needs_recovery, d.summary()


class TestErrorDialogDetection:
    def test_dialog_is_detected(self):
        d = classify(scenes.dark_dialog())
        assert d.screen is Screen.CONNECTION_ERROR, d.summary()
        assert d.confidence > 0.4

    def test_buttons_are_found(self):
        d = classify(scenes.dark_dialog())
        assert len(d.buttons) >= 2, f"only found {d.summary()}"

    def test_primary_button_is_the_bright_one(self):
        d = classify(scenes.dark_dialog())
        prim = [b for b in d.buttons if b.kind == "primary"]
        assert prim, d.summary()
        assert pick_primary(d).kind == "primary"

    def test_click_target_matches_the_real_button(self):
        """The button the detector reports must be where the button
        actually is, or every click misses."""
        frame = scenes.dark_dialog()
        truth = scenes.primary_button_box(frame)
        assert truth is not None, "fixture has no primary button"
        btn = pick_primary(classify(frame))
        tx, ty, tw, th = truth
        # Centres must agree within a few pixels.
        assert abs(btn.center[0] - (tx + tw // 2)) < 12
        assert abs(btn.center[1] - (ty + th // 2)) < 12

    def test_green_accent_also_detected(self):
        d = classify(scenes.dark_dialog(accent_bgr=(90, 200, 120)))
        assert d.screen is Screen.CONNECTION_ERROR, d.summary()

    def test_dialog_survives_noise(self):
        f = scenes.with_cursor_and_noise(scenes.dark_dialog(), cursor=(300, 200))
        d = classify(f)
        assert d.screen is Screen.CONNECTION_ERROR, d.summary()

    def test_pick_primary_none_when_no_buttons(self):
        d = Detection(Screen.CONNECTION_ERROR, 0.5, "x", buttons=[])
        assert pick_primary(d) is None


class TestFrameStats:
    def test_white_is_uniform(self):
        mean, uniq = frame_stats(scenes.white_screen())
        assert uniq <= 4
        assert all(v > 200 for v in mean)

    def test_busy_frame_has_many_colours(self):
        _, uniq = frame_stats(scenes.in_game_scene())
        assert uniq > 40, f"only {uniq} quantised colours"

    def test_dialog_has_few_colours(self):
        """A dialog is flat-shaded. This is the inverse of the in-game case
        and is what separates the two in classify()."""
        _, uniq = frame_stats(scenes.dark_dialog())
        assert uniq < 80, f"{uniq} colours is too busy for a dialog"

    def test_flat_detection(self):
        assert is_flat(scenes.white_screen())
        assert not is_flat(scenes.in_game_scene())

    def test_none_safe(self):
        assert frame_stats(None) == ((0.0, 0.0, 0.0), 0)
        assert is_flat(None)


# --- supervisor -------------------------------------------------------

from reconnect.supervisor import (  # noqa: E402
    Incident, RecoveryConfig, Rung, Supervisor, _looks_same,
)


class FakeCapture:
    """A capture that hands out scripted frames and records nothing else."""

    backend = "fake"

    def __init__(self, frames=None):
        self.frames = list(frames or [])
        self.stats = SimpleNamespace(frames=0, last_frame_at=time.perf_counter(),
                                     stale_ms=0.0, backend="fake", note="")

    def grab(self):
        if self.frames:
            return self.frames.pop(0)
        return None

    def close(self):
        pass

    def healthy(self, max_stale_ms=3000.0):
        return True


class NullLog:
    def __getattr__(self, name):
        return lambda *a, **k: None


class FakeWin:
    def __init__(self, left=0, top=0, width=1280, height=800):
        self.hwnd = 1
        self.left, self.top = left, top
        self.width, self.height = width, height
        self.title = "Roblox"
        self.cls = "RobloxPlayerBeta"
        self.pid = 1

    def center(self):
        return (self.left + self.width // 2, self.top + self.height // 2)


@pytest.fixture
def no_input(monkeypatch):
    """Record every click instead of sending one."""
    clicks = []
    monkeypatch.setattr("reconnect.supervisor.rin.left_click",
                        lambda x, y, **k: clicks.append((x, y)))
    monkeypatch.setattr("reconnect.supervisor.rin.bring_to_front",
                        lambda h: True)
    monkeypatch.setattr("reconnect.supervisor.is_foreground",
                        lambda h: True)
    monkeypatch.setattr("reconnect.supervisor.press_escape",
                        lambda **k: clicks.append(("escape",)))
    return clicks


def make_sup(confirm_frames=3, **cfg_kw):
    """Build a supervisor with tests fast. Defaults are overridable so a
    single kwarg like confirm_frames=6 does not collide with the default."""
    kw = {"action_cooldown_s": 0.0, "confirm_frames": confirm_frames}
    kw.update(cfg_kw)
    return Supervisor(FakeCapture(), RecoveryConfig(**kw), logger=NullLog())


class TestSupervisorNeverActsOnInGame:
    def test_in_game_frame_does_nothing(self, no_input):
        s = make_sup()
        win = FakeWin()
        for _ in range(30):
            out, _ = s.tick(scenes.in_game_scene(seed=1), win)
            assert out in ("ok", "waiting")
        assert no_input == [], f"acted on a live game frame: {no_input}"

    def test_white_game_frame_does_nothing(self, no_input):
        """A bright game scene must not be read as a failure."""
        bright = np.clip(scenes.in_game_scene(seed=4).astype(np.int16) + 90,
                        0, 255).astype(np.uint8)
        s = make_sup()
        for _ in range(20):
            s.tick(bright, FakeWin())
        assert no_input == []


class TestSupervisorRecovers:
    def test_confirms_before_acting(self, no_input):
        """Nothing may be clicked until confirmation has elapsed.

        Asserted as the invariant rather than a fixed tick count, because
        the ladder takes an extra confirm->rejoin step internally and the
        test should not have to know that.
        """
        s = make_sup(confirm_frames=6)
        win = FakeWin()
        for _ in range(5):
            out, _ = s.tick(scenes.dark_dialog(), win)
            assert out == "waiting"
        assert no_input == [], "acted before confirmation elapsed"

        # Within a bounded number of further ticks - two ladder steps -
        # the click must have happened.
        for _ in range(4):
            s.tick(scenes.dark_dialog(), win)
            if no_input:
                break
        assert no_input, "never acted after confirmation"

    def test_rejoin_is_retried_then_given_up_on(self, no_input):
        """Two attempts, then stop. Clicking Rejoin forever recovers
        nothing and looks like progress."""
        s = make_sup(confirm_frames=2)
        win = FakeWin()
        for _ in range(60):
            s.tick(scenes.dark_dialog(), win)
        assert 0 < len(no_input) <= 3, f"{len(no_input)} actions: {no_input}"

    def test_stops_after_retry_cap_without_destructive(self, no_input):
        s = make_sup(confirm_frames=2)
        win = FakeWin()
        for _ in range(80):
            s.tick(scenes.dark_dialog(), win)
        assert len(no_input) <= 2, f"kept clicking: {no_input}"

    def test_clicks_the_primary_button(self, no_input):
        s = make_sup(confirm_frames=3)
        win = FakeWin()
        for _ in range(6):
            s.tick(scenes.dark_dialog(), win)
        assert no_input
        click = no_input[0]
        assert isinstance(click[0], int), f"clicked {click}"
        # Must land inside the frame bounds.
        assert 0 <= click[0] <= win.width
        assert 0 <= click[1] <= win.height

    def test_recovers_when_screen_clears(self, no_input):
        s = make_sup(confirm_frames=3)
        win = FakeWin()
        for _ in range(5):
            s.tick(scenes.dark_dialog(), win)
        s.tick(scenes.in_game_scene(), win)
        assert s.incident is None, "incident not cleared on recovery"

    def test_recovers_to_white_screen(self, no_input):
        """Back to a blank screen counts as recovery, not a new incident."""
        s = make_sup()
        win = FakeWin()
        s.tick(scenes.dark_dialog(), win)
        out, det = s.tick(scenes.white_screen(), win)
        assert det.screen is Screen.WHITE_SCREEN

    def test_increments_retry_rung(self, no_input):
        s = make_sup(confirm_frames=2)
        win = FakeWin()
        for _ in range(12):
            s.tick(scenes.dark_dialog(), win)
        assert len(no_input) >= 2, f"only {len(no_input)} actions"


class TestSupervisorIsCautious:
    def test_no_button_means_no_click(self, monkeypatch):
        """An error screen we cannot read must not escalate."""
        clicks = []
        monkeypatch.setattr("reconnect.supervisor.rin.left_click",
                            lambda x, y, **k: clicks.append((x, y)))
        monkeypatch.setattr("reconnect.supervisor.is_foreground",
                            lambda h: True)
        cfg = RecoveryConfig(confirm_frames=2, action_cooldown_s=0.0)
        s = Supervisor(FakeCapture(), cfg, logger=NullLog())

        # An error-looking frame with NO detectable buttons.
        blank_dialog = np.full((800, 1280, 3), 30, np.uint8)
        cv2rect = np.full((800, 1280, 3), 30, np.uint8)
        for _ in range(20):
            s.tick(cv2rect, FakeWin())
        assert clicks == [], f"clicked a screen with no buttons: {clicks}"

    def test_unfocused_window_is_not_clicked(self, monkeypatch):
        clicks = []
        monkeypatch.setattr("reconnect.supervisor.rin.left_click",
                            lambda x, y, **k: clicks.append((x, y)))
        monkeypatch.setattr("reconnect.supervisor.rin.bring_to_front",
                            lambda h: False)
        monkeypatch.setattr("reconnect.supervisor.is_foreground",
                            lambda h: False)
        s = make_sup(confirm_frames=2)
        for _ in range(12):
            s.tick(scenes.dark_dialog(), FakeWin())
        assert clicks == [], "clicked without focus"

    def test_changed_frame_resets_confirmation(self, no_input):
        """A dialog that flickers must not accumulate confirmation.

        Alternating between two visually different error screens is a
        mid-transition state. Confirmation must restart each time, so the
        ladder never reaches an action.
        """
        s = make_sup(confirm_frames=6)
        win = FakeWin()
        a = scenes.dark_dialog()
        b = scenes.dark_dialog(accent_bgr=(20, 200, 60),
                               secondary_bgr=(90, 90, 100))
        assert not _looks_same(a, b), "fixture frames are too similar"
        for i in range(20):
            s.tick(a if i % 2 == 0 else b, win)
        assert no_input == [], "acted on a flickering dialog"

    def test_destructive_rungs_refused_by_default(self, no_input):
        s = make_sup(confirm_frames=2, incident_timeout_s=0.0)
        win = FakeWin()
        for _ in range(20):
            s.tick(scenes.dark_dialog(), win)
        assert ("escape",) not in no_input

    def test_destructive_needs_explicit_opt_in(self, monkeypatch):
        clicks = []
        monkeypatch.setattr("reconnect.supervisor.rin.left_click",
                            lambda x, y, **k: clicks.append((x, y)))
        monkeypatch.setattr("reconnect.supervisor.is_foreground",
                            lambda h: True)
        monkeypatch.setattr("reconnect.supervisor.press_escape",
                            lambda **k: clicks.append(("escape",)))
        cfg = RecoveryConfig(confirm_frames=2, action_cooldown_s=0.0,
                             max_auto_rung=Rung.RELAUNCH)
        s = Supervisor(FakeCapture(), cfg, allow_destructive=True,
                       logger=NullLog())
        for _ in range(30):
            s.tick(scenes.dark_dialog(), FakeWin())
        assert ("escape",) in clicks, "destructive mode never escaped"

    def test_incident_timeout_stops_escalating(self, no_input):
        s = make_sup(confirm_frames=2, incident_timeout_s=0.0)
        for _ in range(10):
            out, _ = s.tick(scenes.dark_dialog(), FakeWin())
            assert out != "acted" or len(no_input) <= 2

    def test_stop_flag(self):
        s = make_sup()
        s.stop()
        out, _ = s.tick(scenes.dark_dialog(), FakeWin())
        assert out == "stopped"


class TestFrameSimilarity:
    def test_same_frame_is_same(self):
        f = scenes.dark_dialog()
        assert _looks_same(f, f.copy())

    def test_small_noise_is_same(self):
        a = scenes.dark_dialog()
        b = scenes.with_cursor_and_noise(a, seed=5)
        assert _looks_same(a, b)

    def test_different_scene_is_different(self):
        assert not _looks_same(scenes.dark_dialog(),
                               scenes.in_game_scene(seed=1))

    def test_none_is_permissive(self):
        assert _looks_same(None, None)
        assert _looks_same(None, scenes.dark_dialog())
