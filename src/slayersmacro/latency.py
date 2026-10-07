"""Where the milliseconds go, and what to do about them.

The measured chain on this machine, from the moment the pixels were on
screen to the moment the game acts on the keypress:

    t_content  ->  T_cap  ->  t_grab  ->  T_proc  ->  t_decide
               ->  T_inj  ->  [wait for game frame]  ->  t_act

    T_cap  ~4.2 ms   capture is vsync-locked at 240Hz, so this is one frame
    T_proc ~0.3 ms   threshold + centroid + filter update
    T_inj  ~0.3 ms   SendInput syscall
    wait   ~0..4.2ms Roblox samples input at the top of its own frame

So roughly 9ms from screen to effect, and about half of that is the part we
cannot control: the game's own frame boundary.

Compensating for it is simple arithmetic. You do not press when the marker
is where you measured it; you press when your prediction says the marker
will BE THERE when the game reads the key. The filter in kalman.py supplies
that prediction, this module supplies the horizon.
"""

from __future__ import annotations

import math

# Measured on this box (see tools/bench_capture.py, tools/bench_input.py).
DEFAULT_T_CAP = 0.00417    # 1/240s - capture returns the last presented frame
DEFAULT_T_PROC = 0.0003    # detection + Kalman update
DEFAULT_T_INJ = 0.0003     # SendInput batched call
DEFAULT_DT_GAME = 1.0 / 240.0


class LatencyBudget:
    """All values in seconds.

    Fill these from MEASURED p99 values, not means. The distinction is not
    pedantic: a press that arrives 0.3ms before a frame boundary and one
    that arrives 3.9ms before it behave identically, but a press that
    arrives just AFTER the boundary waits a whole extra frame. Under-running
    the budget therefore fails intermittently and in a way that looks like
    the game's fault.
    """

    def __init__(self, t_cap: float = DEFAULT_T_CAP,
                 t_proc: float = DEFAULT_T_PROC,
                 t_inj: float = DEFAULT_T_INJ,
                 dt_game: float = DEFAULT_DT_GAME) -> None:
        self.t_cap = t_cap
        self.t_proc = t_proc
        self.t_inj = t_inj
        self.dt_game = dt_game

    @property
    def actuation(self) -> float:
        """Time from t_content to t_act if we pressed immediately."""
        return self.t_cap + self.t_proc + self.t_inj + self.dt_game / 2.0

    def lookahead(self, bias_frames: float = 0.5) -> float:
        """Horizon to hand to predict_ahead(), snapped to whole game frames.

        Snapped UP: the game only reads input at frame boundaries, so a
        fractional extra frame buys nothing and can only make us late.

        Biased early (bias_frames > 0): for a centred band, being early by a
        frame still lands inside it, whereas being late means the marker has
        already passed. Asymmetric costs deserve an asymmetric bias.
        """
        raw = self.actuation + bias_frames * self.dt_game
        return math.ceil(raw / self.dt_game) * self.dt_game

    def __repr__(self) -> str:
        return (f"LatencyBudget(t_cap={self.t_cap * 1000:.2f}ms, "
                f"t_proc={self.t_proc * 1000:.2f}ms, "
                f"t_inj={self.t_inj * 1000:.2f}ms, "
                f"dt_game={self.dt_game * 1000:.2f}ms, "
                f"actuation={self.actuation * 1000:.2f}ms, "
                f"lookahead={self.lookahead() * 1000:.2f}ms)")


def lookahead_ms(budget: LatencyBudget | None = None) -> float:
    return (budget or LatencyBudget()).lookahead() * 1000.0


def quantisation_error(velocity: float,
                       budget: LatencyBudget | None = None) -> float:
    """Worst-case marker error from the game reading input at a frame boundary.

    In normalised units. This is the floor on accuracy that no amount of
    filtering removes: the game acts on a frame boundary, and the press can
    land anywhere within one frame of when we wanted it to.
    """
    b = budget or LatencyBudget()
    return abs(velocity) * b.dt_game / 2.0


def missing_fps_error(velocity: float, band_frac: float,
                      budget: LatencyBudget | None = None) -> bool:
    """Is the hit band too narrow to absorb one frame of quantisation?

    Once True, compensating latency precisely buys nothing: the residual
    error already exceeds the band, and the fix is a wider band or a
    faster marker, not a better predictor.

    Args:
        velocity: marker speed in normalised units per second.
        band_frac: hit band half-width as a fraction of the bar.
    """
    return quantisation_error(velocity, budget) > band_frac


def explain(budget: LatencyBudget | None = None) -> str:
    """Human-readable breakdown, for the log at startup."""
    b = budget or LatencyBudget()
    rows = [
        ("capture (vsync-locked)", b.t_cap),
        ("detect + filter", b.t_proc),
        ("SendInput", b.t_inj),
        ("game frame wait (avg)", b.dt_game / 2.0),
    ]
    total = b.actuation
    lines = ["latency budget:"]
    for name, s in rows:
        lines.append(f"  {name:<26} {s * 1000:6.2f}ms  "
                     f"{s / total * 100:4.1f}%")
    lines.append(f"  {'TOTAL to effect':<26} {total * 1000:6.2f}ms")
    lines.append(f"  predict {b.lookahead() * 1000:.2f}ms ahead "
                 f"({b.lookahead() / b.dt_game:.0f} game frames)")
    return "\n".join(lines)