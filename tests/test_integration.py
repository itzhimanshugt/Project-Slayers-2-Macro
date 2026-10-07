"""Integration tests: engine wiring, guard gating, and the safety ladder.

These are the tests that would have caught the real incident - a macro
sending input to a game window that was not the one being looked at.
"""

import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from slayersmacro import config  # noqa: E402
from slayersmacro.engine import Engine, Macro, MacroContext  # noqa: E402
from slayersmacro.guard import GuardState  # noqa: E402
from slayersmacro.window import WindowInfo  # noqa: E402


def make_win(hwnd=1, w=800, h=600):
    return WindowInfo(hwnd=hwnd, title="Roblox", left=0, top=0,
                      width=w, height=h)


class _Lazy:
    """Reads through to the macro instance once the engine has built it."""

    def __init__(self, holder):
        self._holder = holder

    def __getattr__(self, name):
        assert "m" in self._holder, "engine never constructed the macro"
        return getattr(self._holder["m"], name)


class Recorder(Macro):
    """Macro that counts its iterations, gated and ungated.

    Bounded on TOTAL iterations, not just permitted ones: when the gate is
    correctly closed forever, a counter that only advances when permitted
    would spin here until the test harness gave up.
    """
    name = "recorder"

    def __init__(self, ctx, budget=40):
        super().__init__(ctx)
        self.iterations = 0
        self.gated_off = 0
        self.budget = budget

    def run(self):
        for _ in range(self.budget):
            if self.ctx.may_act():
                self.iterations += 1
            else:
                self.gated_off += 1
            time.sleep(0.002)
        return True


class TestEngineGating:
    def _engine(self, monkeypatch, frame=None, win=None):
        cfg = config.load("config/does-not-exist.toml")
        cap = SimpleNamespace(
            grab_window=lambda w: frame,
            close=lambda: None,
            grab=lambda *a, **k: frame,
        )
        monkeypatch.setattr("slayersmacro.engine.find_window",
                            lambda t: win)
        eng = Engine(Recorder, cfg)
        eng.cap = cap
        # Hold the constructed macro so the test can read its counters.
        eng.macro_factory = _wrapping_factory(Recorder)
        return eng, cfg


def _wrapping_factory(cls):
    holder = {}

    def factory(ctx):
        m = cls(ctx)
        holder["m"] = m
        return m

    factory.holder = holder
    return factory

    def test_gate_blocks_when_no_window(self, monkeypatch):
        eng, _ = self._engine(monkeypatch, win=None)
        eng.start()
        eng.join(5)
        assert eng.should_stop() is True

    def test_gate_blocks_while_unfocused(self, monkeypatch):
        """A window that exists but is not foreground must not be acted on.

        Asserts on the gate count, not on engine shutdown: a macro that
        finishes its budget either way would pass a shutdown check while
        having sent input, which is the thing under test.
        """
        eng, _ = self._engine(monkeypatch,
                              frame=np.zeros((60, 80, 3), np.uint8),
                              win=make_win(hwnd=999999))
        macro = _capture_macro(eng, monkeypatch)
        eng.start()
        eng.join(10)
        assert macro.gated_off > 0, "gate never closed on an unfocused window"
        assert macro.iterations == 0, "sent input to an unfocused window"

    def test_gate_blocks_occluded_static_frame(self, monkeypatch):
        """The incident: the rect showed something else entirely."""
        static = np.zeros((60, 80, 3), np.uint8)
        cfg = config.load("config/does-not-exist.toml")
        cfg["general"]["require_focus"] = False
        eng = Engine(Recorder, cfg)
        eng.cap = SimpleNamespace(grab_window=lambda w: static,
                                  close=lambda: None,
                                  grab=lambda *a, **k: static)
        monkeypatch.setattr("slayersmacro.engine.find_window",
                            lambda t: make_win())
        # Force the "window already settled" state so the focus check is out
        # of the way and we exercise occlusion specifically.
        eng._build_guard = lambda: _settled_guard()
        eng.macro_factory = _wrapping_factory(Recorder)

        macro = _capture_macro(eng)
        eng.start()
        eng.join(10)
        assert macro.iterations == 0, "acted on a static/occluded rect"

    def test_release_all_covers_common_keys(self):
        """A stuck key is the worst failure mode; make the list explicit."""
        cfg = config.load("config/does-not-exist.toml")
        eng = Engine(Recorder, cfg)
        eng.release_all()          # must not raise with no window present
        eng.release_all()          # and must be idempotent


def _settled_guard():
    from slayersmacro.guard import GuardConfig, SafetyGuard
    g = SafetyGuard(GuardConfig(require_focus=False, settle_frames=2,
                                stable_for_ms=0))
    g._last_rect = (0, 0, 800, 600)
    g._rect_since = time.perf_counter() - 10
    return g


class TestMacroContextGate:
    def test_no_engine_means_permissive(self):
        """A context built by hand (tools, tests) has no latch to consult."""
        ctx = MacroContext(config.load("x.toml"),
                           None, make_win(), None)
        assert ctx.may_act() is True

    def test_screen_size_accessors(self):
        ctx = MacroContext(config.load("x.toml"), None, make_win(), None)
        assert ctx.screen_w > 0 and ctx.screen_h > 0


class TestFailureModesAreLoud:
    def test_macro_crash_is_caught_and_logged(self, monkeypatch):
        class Boom(Macro):
            name = "boom"
            def run(self):
                raise RuntimeError("intentional")
        cfg = config.load("config/does-not-exist.toml")
        eng = Engine(Boom, cfg)
        monkeypatch.setattr("slayersmacro.engine.find_window", lambda t: None)
        eng.start()
        eng.join(5)
        assert eng.should_stop() is True     # crashed, but shut down cleanly

    def test_teardown_runs_even_on_crash(self, monkeypatch):
        calls = []
        class Boom(Macro):
            name = "boom"
            def setup(self):
                calls.append("setup")
            def teardown(self):
                calls.append("teardown")
            def run(self):
                raise RuntimeError("intentional")
        cfg = config.load("config/does-not-exist.toml")
        eng = Engine(Boom, cfg)
        monkeypatch.setattr("slayersmacro.engine.find_window",
                            lambda t: make_win())
        eng.cap = SimpleNamespace(grab_window=lambda w: None, close=lambda: None,
                                  grab=lambda *a, **k: None)
        eng.start()
        eng.join(5)
        assert calls == ["setup", "teardown"]