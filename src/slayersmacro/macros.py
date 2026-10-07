"""The macros themselves.

Each is a plain class with run() returning False to stop. They share one
loop pattern: look, decide, act, wait. Keeping them boring means each one
can be read and changed on its own.
"""

from __future__ import annotations

import time

from . import input as sin
from .actions import GameActions, obstruction_report
from .recovery import Rung
from .adaptation import ObstructionDetector, OnlineAdaptation, TimingProfile
from .detect import BarDetector, HealthReader, MovementWatch
from .engine import Macro, MacroContext, anti_idle_nudge
from .kalman import MarkerTracker
from .latency import LatencyBudget


class CombatMacro(Macro):
    """M1 combo loop, optionally weaving in skill keys.

    combo_length matters: sources agree the 5th M1 in a string shoves your
    character out of range, so the loop stops at 4 by default and waits for
    the animation to settle.
    """

    name = "combat"

    def __init__(self, ctx: MacroContext) -> None:
        super().__init__(ctx)
        c = ctx.cfg["combat"]
        self.interval = float(c["m1_interval_ms"]) / 1000.0
        self.hold_ms = float(c["m1_hold_ms"])
        self.combo_len = int(c["combo_length"])
        self.skill_interval = float(c["skill_interval_ms"]) / 1000.0
        self.skill_keys = list(c["skill_keys"])
        self.use_skills = bool(c.get("use_skills", True))
        self.reader = HealthReader()
        self.timing = (TimingProfile.load(
            target_ms=float(c["m1_interval_ms"]))
            if bool(ctx.cfg["adaptation"].get("learn_timing", True))
            else None)
        self.adaptation = OnlineAdaptation(
            enabled=bool(ctx.cfg["adaptation"].get("adapt_thresholds", False)))
        self._i = 0
        self._next_skill = 0.0
        self._last_attack = time.perf_counter()

    def setup(self) -> None:
        t = self.timing
        self.ctx.log.info(
            f"combat: {self.combo_len}-hit combo, attack via "
            f"{self.ctx.cfg['combat'].get('attack_key', 'mouse1')}, "
            f"skills {'on' if self.use_skills else 'off'}, interval "
            + (f"learning from {t.target_ms:.0f}ms" if t
               else f"{self.interval * 1000:.0f}ms fixed")
        )

    def teardown(self) -> None:
        if self.timing and self.timing.learned:
            try:
                p = self.timing.save()
                self.ctx.log.info(
                    f"learned interval {self.timing.interval_ms:.0f}ms -> {p}")
            except OSError as exc:
                self.ctx.log.warning(f"could not save timing: {exc}")

    def run(self) -> bool:
        while True:
            # The safety gate. Without it this macro will happily press keys
            # into whatever window happens to be at the Roblox rect.
            if not self.ctx.may_act():
                time.sleep(0.05)
                continue

            hp = self.reader.read(self.ctx.region_frame("player_health"))
            if hp is not None and hp < 0.25:
                # Backing off at low health is a correctness decision, not a
                # detection failure: without the region we cannot read hp, so
                # this simply never engages unless calibrated.
                self.ctx.log.warning(f"health {hp:.0%} - backing off")
                time.sleep(2.0)
                continue

            now = time.perf_counter()
            if self.timing is not None:
                self.timing.observe((now - self._last_attack) * 1000.0)
            self._last_attack = now

            self.ctx.attack()
            self._i = (self._i + 1) % max(1, self.combo_len)
            self.adaptation.record(True)

            if self.use_skills and self.skill_keys and now >= self._next_skill:
                key = self.skill_keys[int(now * 3) % len(self.skill_keys)]
                self.ctx.press(key)
                self._next_skill = now + self.skill_interval

            base = (self.timing.interval_ms / 1000.0
                    if self.timing is not None else self.interval)
            # Let the animation finish rather than fighting it. The extra
            # pause after the last hit of a combo is what stops the 5th hit
            # pushing the character out of range.
            time.sleep(base * 1.8 if self._i == 0 else base)

            if int(now) % 600 < 1:
                anti_idle_nudge()

            self.ctx.stats.cycles += 1
            self.ctx.stats.frames += 1


class FishingMacro(Macro):
    """Cast, then hold the marker inside the zone until progress completes.

    Hold raises, release lowers (per the community descriptions of this
    minigame).

    The decision uses the Kalman prediction at the actuation horizon, not
    the position we just measured. That is the whole point: the measured
    position is ~7ms stale by the time the game acts, and on a fast marker
    that is the difference between landing the fish and not. See
    latency.py for the budget.

    Fails safe rather than spamming: if detection fails N frames running,
    the macro stops rather than mashing the key against a bar it cannot see.
    """

    name = "fishing"

    def __init__(self, ctx: MacroContext) -> None:
        super().__init__(ctx)
        f = ctx.cfg["fishing"]
        self.cast_ms = float(f["cast_interval_ms"]) / 1000.0
        self.band = float(f["hit_band_pct"]) / 100.0
        self.det = BarDetector(lock_hits=3)
        self.collect = bool(f["collect_fish"])
        self.use_prediction = bool(f.get("predictive", True))
        self.max_misses = int(f.get("max_consecutive_misses", 12))
        self.tracker = MarkerTracker()
        self.budget = LatencyBudget()
        self._misses = 0
        self._since_cast = 0.0

    def run(self) -> bool:
        self.ctx.log.info(
            f"fishing: band {self.band * 100:.1f}% of bar, "
            f"predictive={self.use_prediction}, "
            f"horizon {self.budget.lookahead() * 1000:.1f}ms"
        )
        last = time.perf_counter()
        while True:
            now = time.perf_counter()
            dt = now - last
            last = now

            # Fishing holds space, so losing the gate must release it
            # immediately or the character jumps into a wall.
            if not self.ctx.may_act():
                sin.up("space")
                time.sleep(0.05)
                continue

            bar = self.ctx.region_frame("fishing_bar")
            if bar is None:
                self._misses += 1
                if self._misses >= self.max_misses:
                    self.ctx.log.error(
                        "no fishing_bar region - stopping. "
                        "Run: python tools/calibrate.py fishing")
                    return False
                sin.up("space")
                time.sleep(0.05)
                continue

            track = self.det.find_track(bar)
            marker = self.det.find_marker(bar)

            if track is None or marker is None:
                self._misses += 1
                self.tracker.reset()
                if self._misses >= self.max_misses:
                    self.ctx.log.error(
                        f"bar not detected for {self._misses} frames - "
                        f"stopping instead of mashing keys. "
                        f"Try: python tools/autotune.py fishing_bar")
                    return False
                sin.up("space")
                time.sleep(0.004)
                continue

            self._misses = 0

            # Decide on the predicted position at actuation time.
            if self.use_prediction:
                self.tracker.feed(marker)
                target = self.tracker.predict_ahead(self.budget.lookahead())
                ready = self.tracker.confident()
            else:
                target = marker
                ready = True

            if ready:
                # Hysteresis: hold while the marker is behind the zone
                # centre, release once ahead. Predicting means we can act a
                # frame early instead of reacting at the boundary.
                holding = target <= track
                if holding:
                    sin.down("space")
                else:
                    sin.up("space")
                self.ctx.stats.frames += 1

            self._since_cast += dt
            if self._since_cast > self.cast_ms:
                self._since_cast = 0.0
                self.det.reset()
                self.tracker.reset()
                sin.up("space")
                self.ctx.press("space")
                self.ctx.stats.cycles += 1

            time.sleep(0.004)


class BossMacro(Macro):
    """Fight a boss: engage, retreat when low, heal, resume, and give up
    gracefully if it is not making progress.

    The original version of this macro read "boss health bar not visible" as
    "boss died", and reacted by retreating. But the bar is also invisible
    when detection fails, when the boss is off-screen, or during a phase
    transition - so it retreated roughly as often as it won. Absence of
    evidence is not evidence of absence: a death now requires the bar to be
    gone AND our own health to have stopped moving, for several frames in a
    row.
    """

    name = "boss"

    def __init__(self, ctx: MacroContext) -> None:
        super().__init__(ctx)
        b = ctx.cfg["boss"]
        self.interval = float(b["attack_interval_ms"]) / 1000.0
        self.retreat_at = float(b["retreat_hp_pct"]) / 100.0
        self.heal_ms = float(b["heal_ms"]) / 1000.0
        self.max_retreats = int(b["max_retreats"])
        self.boss_hp = HealthReader()
        self.my_hp = HealthReader()
        self.timing = (TimingProfile.load(
            target_ms=float(b["attack_interval_ms"]))
            if bool(ctx.cfg["adaptation"].get("learn_timing", True))
            else None)
        self._retreats = 0
        self._state = "engage"
        self._bars_missing = 0
        self._last_boss_hp = None
        self._no_progress = 0.0
        self._last_attack = time.perf_counter()
        # Latched once the macro decides it is finished. Without it a
        # caller that keeps stepping drives the retreat counter past its own
        # limit, because the "stop" decision has nowhere to be remembered.
        self._stopped = False

    def setup(self) -> None:
        self.ctx.log.info(
            f"boss: {self.interval * 1000:.0f}ms attacks, retreat below "
            f"{self.retreat_at:.0%} hp, {self.max_retreats} retreats allowed")

    def teardown(self) -> None:
        if self.timing and self.timing.learned:
            try:
                self.timing.save()
            except OSError:
                pass

    def step(self, my, boss) -> str:
        """Decide and act for one frame. Returns the action taken.

        Split out from run() deliberately: the loop is infinite, so testing
        the decisions through it means fighting the loop. Everything
        interesting here is one step's worth of state change.

        my, boss are health fractions, or None when unreadable. None means
        "no data", never a default value - a fabricated 0.0 here would read
        as a dead player.
        """
        if self._stopped:
            return "stop"
        if boss is None:
            self._bars_missing += 1
        else:
            if self._bars_missing:
                self.ctx.log.info("boss bar found again")
            self._bars_missing = 0
            # A bar that never moves means we are swinging at air, however
            # healthy it looks.
            if self._last_boss_hp is not None and \
                    abs(boss - self._last_boss_hp) < 0.01:
                self._no_progress += self.interval
            else:
                self._no_progress = 0.0
            self._last_boss_hp = boss

        # --- retreat and heal -------------------------------------------
        if self._state == "retreat":
            if my is not None and my > 0.7:
                self.ctx.log.info("healed, engaging again")
                self._retreats = 0
                self._state = "engage"
                return "engage"
            # Count time spent retreating, not just entries into it. Without
            # this the loop cannot terminate: the low-health branch that
            # increments the counter is unreachable while already retreating.
            self._retreats += 1
            if self._retreats >= self.max_retreats:
                self.ctx.log.error(
                    f"no recovery after {self._retreats} retreats - stopping")
                self._stopped = True
                return "stop"
            return "heal"

        # --- low health ---------------------------------------------------
        if my is not None and my < self.retreat_at:
            self.ctx.log.warning(
                f"hp {my:.0%} - retreating "
                f"({self._retreats + 1}/{self.max_retreats})")
            self.ctx.hold("s", 500)
            self._state = "retreat"
            return "retreat"

        # --- fight over? --------------------------------------------------
        # Requires the bar gone for a sustained stretch, not one frame. A
        # single unreadable frame used to be read as a dead boss, which made
        # this macro retreat about as often as it fought.
        if self._bars_missing > 45:
            self.ctx.log.info(
                f"boss bar absent for {self._bars_missing} frames - "
                f"treating the fight as over")
            self.ctx.stats.cycles += 1
            self._retreats += 1
            self._state = "retreat"
            return "fight_over"

        # --- not landing hits ---------------------------------------------
        if self._no_progress > 20.0:
            self.ctx.log.warning(
                f"boss health unchanged for {self._no_progress:.0f}s - "
                f"probably not in range")
            self.ctx.hold("w", 600)      # walk toward it
            self._no_progress = 0.0
            return "reposition"

        now = time.perf_counter()
        if self.timing is not None:
            self.timing.observe((now - self._last_attack) * 1000.0)
        self._last_attack = now
        self.ctx.attack()
        self.ctx.stats.cycles += 1
        return "attack"

    def _read(self):
        my = self.my_hp.read(self.ctx.region_frame("player_health"))
        boss = self.boss_hp.read(self.ctx.region_frame("boss_health"))
        return my, boss

    def run(self) -> bool:
        while True:
            if not self.ctx.may_act():
                time.sleep(0.05)
                continue

            action = self.step(*self._read())
            if action == "stop":
                return False
            if action in ("heal",):
                time.sleep(self.heal_ms)
                continue
            if action in ("retreat", "fight_over", "engage"):
                continue
            time.sleep(self.timing.interval_ms / 1000.0
                       if self.timing is not None else self.interval)


class AfkFarmMacro(Macro):
    """Walk somewhere, attack forever, back out when stuck.

    Stuck detection is the point of this one. A blind walk bumps into rocks,
    so when the view stops changing while we think we are moving, we try a
    few escape moves and then give up rather than jamming into a wall.
    """

    name = "afk"

    def __init__(self, ctx: MacroContext) -> None:
        super().__init__(ctx)
        s = ctx.cfg["stuck"]
        self.enabled = bool(s["enabled"])
        self.detect_ms = int(s["detect_ms"])
        self.max_retries = int(s["max_retries"])
        self.watch = MovementWatch()
        self.retries = 0
        self._last = time.perf_counter()

        # The three things that make this macro adaptive. None of them
        # rewrites code or trains a model; all of them fit numbers to what
        # is actually happening.
        self.timing = TimingProfile.load(
            target_ms=float(ctx.cfg["combat"]["m1_interval_ms"]))
        self.adaptation = OnlineAdaptation(
            enabled=bool(ctx.cfg["adaptation"].get("adapt_thresholds", False)))
        self.obstruction = ObstructionDetector(still_ms=self.detect_ms)
        self.actions = GameActions(ctx, win=ctx.win)
        self.ladder = self.actions.from_config(ctx.cfg)
        self._last_action = time.perf_counter()
        self._control_down = False

    def _escape(self) -> None:
        """Bump sequence tried in order: back up, jump, sidestep, turn.

        Space is jump in Roblox by default, so it is safe to use here.
        """
        for key, ms in (("s", 300), ("space", 120), ("a", 250), ("space", 150),
                        ("d", 350), ("w", 400)):
            self.ctx.hold(key, ms)
        self.ctx.log.info("escape move done, resuming")

    def run(self) -> bool:
        self.ctx.log.info(
            f"afk: ladder max auto {self.ladder.cfg.max_auto_rung.name}, "
            f"timing {'learning' if self.timing.learned else f'fixed {self.timing.target_ms:.0f}ms'}"
        )
        self.ctx.hold("w", 500)  # start moving
        self._control_down = True
        while True:
            now = time.perf_counter()
            dt_ms = (now - self._last) * 1000
            self._last = now

            # The single safety gate. Nothing below runs unless the engine
            # confirms Roblox is focused, stationary, and actually what we
            # are capturing.
            if not self.ctx.may_act():
                self._control_down = False
                sin.up("w")
                time.sleep(0.05)
                continue

            moving = False
            frame = self.ctx.full_frame()
            if self.enabled and frame is not None:
                before = self.watch.still_ms
                self.watch.update(frame, dt_ms)
                moving = self.watch.still_ms <= before   # motion resets it

            self.ladder.observe(obstruction_report(moving,
                                                   self._control_down, True))
            if self.ladder.needs_action(now):
                rung = self.ladder.fire(now)
                if rung and rung > Rung.RECHECK:
                    self.ctx.stats.stuck_events += 1
                    self._control_down = False
                    sin.up("w")
                    time.sleep(self.ladder.cfg.step_pause_ms / 1000.0)
                    self.watch.reset()
                    if self.ladder.rung is Rung.HOLD:
                        self._control_down = True
                        self.ctx.hold("w", 500)
                continue

            gap_ms = (now - self._last_action) * 1000.0
            self.timing.observe(gap_ms)
            self._last_action = now
            self.ctx.attack()
            self.adaptation.record(True)

            self.ctx.stats.cycles += 1
            time.sleep(self.timing.interval_ms / 1000.0)


class StanceCheckMacro(Macro):
    """Diagnostic: reports what each calibrated region actually contains.

    Run this before trusting any detection. It prints the mean colour and
    brightness per region so a mis-drawn box is obvious immediately.
    """

    name = "diagnose"

    def run(self) -> bool:
        if not self.ctx.calib.regions:
            self.ctx.log.error("nothing calibrated. run: "
                               "python tools/calibrate.py fishing")
            return False
        names = list(self.ctx.calib.regions)
        for i in range(20):
            for name in names:
                f = self.ctx.region_frame(name)
                if f is None:
                    continue
                b, g, r = f[:, :, 0].mean(), f[:, :, 1].mean(), f[:, :, 2].mean()
                self.ctx.log.info(
                    f"{name:<16} mean BGR=({b:5.1f},{g:5.1f},{r:5.1f}) "
                    f"max={int(f.max())}")
            time.sleep(0.5)
        return False


ALL_MACROS = {
    "combat": CombatMacro,
    "fishing": FishingMacro,
    "boss": BossMacro,
    "afk": AfkFarmMacro,
    "diagnose": StanceCheckMacro,
}