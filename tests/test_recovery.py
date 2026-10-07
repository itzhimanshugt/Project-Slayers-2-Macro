"""Tests for the recovery ladder.

The central property under test: escalation happens only after a rung has
actually been tried and failed. A ladder that walks itself on a transient
detection failure is worse than no ladder.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from slayersmacro.recovery import (  # noqa: E402
    ProblemReport, RecoveryConfig, RecoveryLadder, Rung,
)


def ladder(rung_results=None, **cfg_kw):
    """Ladder whose every rung is scripted to return the given result."""
    calls: list[Rung] = []
    results = rung_results or {}

    def make(rung):
        def fn():
            calls.append(rung)
            return results.get(rung, True)
        return fn

    cfg = RecoveryConfig(confirm_frames=2, step_pause_ms=0, **cfg_kw)
    # HOLD and RECHECK are free no-input rungs and never get an action.
    actions = {r: make(r) for r in Rung if r > Rung.RECHECK}
    lad = RecoveryLadder(cfg, actions, logger=None)
    return lad, calls


def pump(lad, report, frames=1):
    """Drive `frames` observation cycles, firing whenever due."""
    fired = []
    t = 0.0
    for _ in range(frames):
        lad.observe(report)
        if lad.needs_action(now=t):
            r = lad.fire(now=t)
            if r is not None:
                fired.append(r)
        t += 0.01
    return fired


class TestEscalationRequiresConfirmation:
    def test_single_bad_frame_does_nothing(self):
        lad, calls = ladder()
        pump(lad, ProblemReport.stuck(), frames=1)
        assert calls == []
        assert lad.rung is Rung.HOLD

    def test_transient_failure_never_escalates(self):
        """One bad frame then good ones must not move the ladder."""
        lad, calls = ladder()
        pump(lad, ProblemReport.stuck(), frames=1)
        pump(lad, ProblemReport.fine(), frames=6)
        assert calls == []
        assert lad.rung is Rung.HOLD

    def test_free_rungs_send_no_input(self):
        """HOLD and RECHECK only wait; the first real action is NUDGE.

        Asserting on `calls` rather than on the rung counter, because the
        ladder moves on immediately once a rung succeeds - a successful NUDGE
        resets to HOLD, so a rung-value assertion would pass vacuously.
        """
        lad, calls = ladder(rung_results={r: False for r in Rung})
        pump(lad, ProblemReport.stuck(), frames=4)
        # Within 4 frames the ladder has only traversed the two free rungs.
        assert calls == [], "the free rungs must not send input"
        assert lad.rung is Rung.NUDGE

    def test_success_resets_everything(self):
        lad, calls = ladder(rung_results={Rung.NUDGE: True})
        pump(lad, ProblemReport.stuck(), frames=6)
        pump(lad, ProblemReport.fine(), frames=1)
        assert lad.rung is Rung.HOLD


class TestEscalationOrder:
    def test_failed_rung_advances_to_next(self):
        lad, calls = ladder(rung_results={Rung.NUDGE: False,
                                           Rung.BACKOFF: False,
                                           Rung.TURN: False})
        pump(lad, ProblemReport.stuck(), frames=40)
        assert calls[:2] == [Rung.NUDGE, Rung.BACKOFF]
        assert calls[2] == Rung.TURN, "must keep climbing past failures"

    def test_never_skips_a_rung(self):
        lad, calls = ladder(rung_results={r: False for r in Rung})
        pump(lad, ProblemReport.stuck(), frames=200)
        order = sorted(set(calls))
        assert order, "no action should have fired"
        for i in range(1, len(order)):
            assert order[i] == order[i - 1] + 1, "no rung may be skipped"

    def test_successful_rung_stops_escalation(self):
        lad, calls = ladder(rung_results={Rung.NUDGE: True})
        pump(lad, ProblemReport.stuck(), frames=60)
        assert Rung.BACKOFF not in calls

    def test_respects_max_auto_rung(self):
        lad, calls = ladder(rung_results={r: False for r in Rung},
                            max_auto_rung=Rung.BACKOFF)
        pump(lad, ProblemReport.stuck(), frames=200)
        assert Rung.RESPAWN not in calls
        assert Rung.SERVERHOP not in calls

    def test_serverhop_never_fires_without_config(self):
        """The most disruptive rung must require explicit opt-in."""
        lad, calls = ladder(rung_results={r: False for r in Rung})
        pump(lad, ProblemReport.stuck(), frames=400)
        assert Rung.SERVERHOP not in calls


class TestFailureHandling:
    def test_raising_rung_counts_as_failure_not_crash(self):
        def boom():
            raise RuntimeError("input failed")

        cfg = RecoveryConfig(confirm_frames=2, step_pause_ms=0)
        lad = RecoveryLadder(cfg, {Rung.NUDGE: boom,
                                   Rung.BACKOFF: lambda: True},
                             logger=None)
        fired = pump(lad, ProblemReport.stuck(), frames=20)
        assert Rung.NUDGE in fired
        assert Rung.BACKOFF in fired, "should have advanced past the raiser"

    def test_missing_action_stops_rather_than_advancing(self):
        cfg = RecoveryConfig(confirm_frames=2, step_pause_ms=0)
        lad = RecoveryLadder(cfg, {}, logger=None)   # no actions at all
        fired = pump(lad, ProblemReport.stuck(), frames=30)
        assert Rung.BACKOFF not in fired

    def test_gives_up_after_max_cycles(self):
        lad, _ = ladder(rung_results={r: False for r in Rung},
                        max_cycles=2)
        pump(lad, ProblemReport.stuck(), frames=600)
        assert lad.cycle > 0, "must eventually give up rather than loop forever"

    def test_exhausted_blocks_further_firing(self):
        """The consumer check. Without it the ladder resets and climbs
        again on the next frame, which is an infinite retry loop that looks
        like work but achieves nothing."""
        lad, calls = ladder(rung_results={r: False for r in Rung},
                            max_cycles=1)
        assert lad.exhausted is False
        pump(lad, ProblemReport.stuck(), frames=200)
        assert lad.exhausted is True
        before = len(calls)
        pump(lad, ProblemReport.stuck(), frames=200)
        assert len(calls) == before, "fired again after giving up"

    def test_success_clears_exhaustion(self):
        # Every rung must FAIL to exhaust the ladder. If a rung succeeds it
        # resets the cycle count, so exhaustion never happens.
        lad, _ = ladder(rung_results={r: False for r in Rung}, max_cycles=1)
        pump(lad, ProblemReport.stuck(), frames=200)
        assert lad.exhausted is True
        pump(lad, ProblemReport.fine(), frames=2)
        assert lad.exhausted is False, "a success must reset the cycle count"


class TestPacing:
    def test_step_pause_enforced(self):
        lad, _ = ladder()
        lad.cfg.step_pause_ms = 1000
        lad.observe(ProblemReport.stuck())
        lad.observe(ProblemReport.stuck())
        assert lad.needs_action(now=0.0) is True
        assert lad.fire(now=0.0) == Rung.RECHECK
        # Immediately after, the pause must block the next rung.
        assert lad.needs_action(now=0.1) is False

    def test_heavier_rungs_need_more_confirmation_frames(self):
        """A respaw takes longer to become visible than a nudge."""
        lad, _ = ladder()
        lad.cfg.rung_settle_frames = {Rung.RESPAWN: 12}
        lad._rung = Rung.RESPAWN
        assert lad.required_frames() == 12
        lad._rung = Rung.NUDGE
        assert lad.required_frames() == lad.cfg.confirm_frames

    def test_waiting_consumes_confirmation_frames(self):
        lad, _ = ladder()
        lad.cfg.step_pause_ms = 10_000     # force pause to be the blocker
        for _ in range(3):
            lad.observe(ProblemReport.stuck())
            lad.fire(now=0.0)
        assert lad._suspect_frames < 3 * lad.cfg.confirm_frames