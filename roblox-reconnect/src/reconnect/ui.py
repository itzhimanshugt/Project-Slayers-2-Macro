"""Detect Roblox's connection-error screens and find their buttons.

Roblox shows several distinct failure states, and they need different
responses:

  CONNECTION_ERROR  the "couldn't connect" dialog. Retry, then leave, then
                    rejoin. The main case this tool exists for.
  DISCONNECTED      the client lost the server. It offers Rejoin.
  CRASHED           a fatal error dialog. Retry restarts the client.
  LOADING_HANG      the window is up but nothing has rendered. Usually a
                    sign the client wedged rather than a recoverable error.
  WHITE_SCREEN      blank white. Same as LOADING_HANG.
  IN_GAME           the normal game. Nothing to do.

Detection is deliberately conservative. A false positive here means clicking
"Leave" on a working game, which loses progress. So every state requires
positive evidence - a known colour signature, not merely "this frame is
unusual" - and the detector reports WHY it believes what it believes.

The colour signatures below were chosen from Roblox's own UI conventions
(dark dialog plate, saturated buttons) rather than from screenshots of a
specific game, so they survive a Roblox client update that changes a game's
art but not the error dialog.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np


class Screen(Enum):
    IN_GAME = "in_game"
    LOADING_HANG = "loading_hang"
    WHITE_SCREEN = "white_screen"
    BLACK_SCREEN = "black_screen"
    CONNECTION_ERROR = "connection_error"
    DISCONNECTED = "disconnected"
    CRASHED = "crashed"
    MENU = "menu"

    @property
    def needs_recovery(self) -> bool:
        return self in (Screen.CONNECTION_ERROR, Screen.DISCONNECTED,
                        Screen.CRASHED, Screen.LOADING_HANG,
                        Screen.WHITE_SCREEN, Screen.BLACK_SCREEN)


@dataclass
class Detection:
    screen: Screen
    confidence: float
    reason: str
    buttons: list["Button"] = field(default_factory=list)
    mean_bgr: tuple[float, float, float] = (0.0, 0.0, 0.0)
    unique_colours: int = 0

    def summary(self) -> str:
        return (f"{self.screen.value} conf={self.confidence:.2f} "
                f"buttons={len(self.buttons)} ({self.reason})")


@dataclass
class Button:
    """A clickable rectangle in FRAME coordinates."""
    x: int
    y: int
    w: int
    h: int
    kind: str          # primary | secondary | unknown
    score: float

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2

    def to_screen(self, win) -> tuple[int, int]:
        cx, cy = self.center
        return win.left + cx, win.top + cy


# --- colour helpers -----------------------------------------------------

def saturation(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1]


def value(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 2]


#: Side of the colour cube each channel is quantised into. 5^3 = 125
#: buckets, which is small enough to histogram and large enough to tell a
#: flat-shaded dialog from a textured game scene.
BUCKETS_PER_CHANNEL = 5
#: Total possible buckets, i.e. the maximum value frame_stats can return.
BUCKET_TOTAL = BUCKETS_PER_CHANNEL ** 3


def frame_stats(frame: np.ndarray) -> tuple[tuple[float, float, float], int]:
    """Mean colour and a colour-bucket occupancy score.

    Deliberately NOT an exact unique-colour count. The original
    implementation was, and it cost 521ms per frame on a 2560x1600 image -
    96% of classify()'s total - because np.unique sorts 2.4M rows. That
    has no business being in a 100ms poll loop.

    Instead: subsample 1/16, quantise each channel into 5 levels, and count
    how many of the 125 possible buckets are occupied. The result is
    bounded by 125 regardless of resolution, and costs a single histogram.

    Measured separation (1280x800 fixtures): flat dialogs ~6, busy game
    frames ~110, a blank screen 1. Blank-versus-not is really carried by
    is_flat(); this score separates "flat dialog" from "busy game".
    """
    if frame is None or frame.size == 0:
        return (0.0, 0.0, 0.0), 0
    small = frame[::4, ::4]                      # 1/16 of the pixels
    if small.size == 0:
        small = frame
    mean = tuple(float(v) for v in small.reshape(-1, 3).mean(axis=0))

    # np.unique would sort 2.4M rows here. There are only 125 distinct
    # bucket indices, so a histogram counts occupancy directly and cheaply.
    step = 256 // BUCKETS_PER_CHANNEL
    q = (small // step).astype(np.int32).reshape(-1, 3)
    index = q[:, 0] * (BUCKETS_PER_CHANNEL ** 2) \
        + q[:, 1] * BUCKETS_PER_CHANNEL + q[:, 2]
    occupied = int(np.count_nonzero(
        np.bincount(index, minlength=BUCKET_TOTAL)))
    return mean, occupied


def is_flat(frame: np.ndarray, min_std: float = 6.0) -> bool:
    """A frame with almost no variation is a blank/loading screen."""
    if frame is None or frame.size == 0:
        return True
    return float(frame.reshape(-1, 3).std(axis=0).max()) < min_std


# --- button finding -----------------------------------------------------

#: Longest edge, in pixels, that connected components runs on. Anything
#: bigger is downscaled first.
ANALYSIS_MAX_EDGE = 640


def _analysis_scale(frame: np.ndarray) -> float:
    """Factor to shrink a frame to for component analysis."""
    h, w = frame.shape[:2]
    longest = max(h, w)
    return min(1.0, ANALYSIS_MAX_EDGE / max(1, longest))


def find_buttons(frame: np.ndarray, min_w: int = 90,
                 min_h: int = 28) -> list[Button]:
    """Find button-shaped regions: wide, short, flat-filled rectangles.

    Connected components rather than Canny contours. An earlier version used
    edges, which produced no contours at all on flat synthetic frames and
    would equally struggle on Roblox's flat-shaded dialog, which has almost
    no gradient to find an edge in.

    What a Roblox dialog button reliably has:
      - a wide, short aspect ratio
      - a flat interior (one colour, not a gradient or texture)
      - noticeably different from the dark plate behind it

    That combination does not appear in a textured game scene, which is
    what keeps this from firing mid-fight.
    """
    if frame is None or frame.size == 0:
        return []
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]

    plate_level = float(np.median(v))
    # Everything that is not the dark plate. Measured against the plate's
    # own brightness rather than a fixed number, so it works on both Roblox's
    # near-black dialog and a lighter game HUD.
    mask = (v.astype(np.int16) > plate_level + 18).astype(np.uint8) * 255
    # Work at reduced resolution. Connected components on a full 2560x1600
    # frame costs ~600ms, which is unaffordable in a 100ms poll loop; at
    # this scale it is ~2ms and a button's box is still accurate to a few
    # pixels, which is far finer than the click needs.
    scale = _analysis_scale(frame)
    if scale < 1.0:
        mask_small = cv2.resize(mask, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_AREA)
    else:
        mask_small = mask
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask_small)
    inv = 1.0 / scale

    out: list[Button] = []
    # Size thresholds are applied AFTER scaling up, so they stay in
    # full-resolution pixels and mean the same thing at any window size.
    min_w_s = min_w * scale
    min_h_s = min_h * scale
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bw < min_w_s or bh < min_h_s:
            continue
        aspect = bw / max(1, bh)
        if aspect < 2.0 or aspect > 9.0:
            continue          # buttons are wide and short
        # A dialog button is a small element on a large plate. A candidate
        # filling most of the frame is scenery, not a control.
        if area > (w * h * scale * scale) * 0.20:
            continue
        fill = area / max(1, bw * bh)
        if fill < 0.72:
            continue          # buttons are solid, text is not

        # Back to full resolution for the colour tests: a button face is
        # flat at 1:1, and a downsampled crop would blur the label into it
        # and destroy the modal-colour measurement.
        fx0, fy0 = int(round(x * inv)), int(round(y * inv))
        fx1 = min(w, int(round((x + bw) * inv)))
        fy1 = min(h, int(round((y + bh) * inv)))
        fw, fh = fx1 - fx0, fy1 - fy0
        if fw <= 8 or fh <= 8:
            continue

        # Interior flatness: the label is a minority of a button's pixels,
        # so measure the MODAL colour's share, not the standard deviation.
        inner = frame[fy0 + 2:fy1 - 2, fx0 + 2:fx1 - 2]
        if inner.size == 0:
            continue
        iq = (inner.reshape(-1, 3) // 20).astype(np.int32)
        idx = iq[:, 0] * 4096 + iq[:, 1] * 64 + iq[:, 2]
        modal_frac = float(np.bincount(idx).max() / inner.shape[0])
        if modal_frac < 0.55:
            continue

        region_s = float(s[fy0:fy1, fx0:fx1].mean())
        region_v = float(v[fy0:fy1, fx0:fx1].mean())

        score = 0.35
        score += 0.30 if modal_frac > 0.8 else 0.0
        score += 0.20 if region_s > 60 else 0.0
        score += 0.15 if region_v > 80 else 0.0
        out.append(Button(x=fx0, y=fy0, w=fw, h=fh,
                          kind="primary" if region_s > 60 else "secondary",
                          score=round(min(score, 1.0), 3)))
    return sorted(out, key=lambda b: -b.score)[:8]


# --- state detection ----------------------------------------------------

def classify(frame: np.ndarray) -> Detection:
    """Classify one frame. Conservative by design.

    Order matters: the blank-screen checks come first because they are
    unambiguous, and a dark blank screen must not be mistaken for a dark
    error dialog.
    """
    mean, buckets = frame_stats(frame)
    flat = is_flat(frame)

    if frame is None or frame.size == 0:
        return Detection(Screen.BLACK_SCREEN, 0.0, "no pixels captured")

    # Blank screens are decided by is_flat() alone, with the mean colour
    # naming which blank it is. The earlier version also required the
    # colour-bucket score to be <= 4, which is a threshold on a signal
    # that is now bounded differently - a coupling that broke the moment
    # frame_stats changed, and misread a white screen as in-game. Blank
    # versus not-blank is a variance question; is_flat() answers it.
    if flat:
        b, g, r = mean
        if b > 200 and g > 200 and r > 200:
            return Detection(Screen.WHITE_SCREEN, 0.95,
                             f"uniform white (mean {b:.0f},{g:.0f},{r:.0f})",
                             mean_bgr=mean, unique_colours=buckets)
        if b < 25 and g < 25 and r < 25:
            return Detection(Screen.BLACK_SCREEN, 0.9,
                             f"uniform black (mean {b:.0f},{g:.0f},{r:.0f})",
                             mean_bgr=mean, unique_colours=buckets)
        return Detection(Screen.LOADING_HANG, 0.7,
                         f"uniform colour, mean {b:.0f},{g:.0f},{r:.0f}",
                         mean_bgr=mean, unique_colours=buckets)

    s = saturation(frame)
    v = value(frame)

    # Roblox's error dialog: a large dark plate with bright, saturated
    # button(s) on it. Require both the dark plate AND saturated buttons,
    # so a dark in-game scene does not qualify.
    dark_plate = float((v < 90).mean())
    has_saturated = float((s > 130).mean())

    # Button finding is the expensive step (~35ms of a 40ms classify), and
    # the CONNECTION_ERROR branch is the only consumer that can change the
    # verdict. A frame that is not dark cannot reach that branch, so do not
    # pay for the analysis. In-game frames are the common case, so this is
    # where the poll loop actually spends its time.
    buttons: list[Button] = []
    if dark_plate > 0.55:
        buttons = find_buttons(frame)

    if dark_plate > 0.55 and has_saturated > 0.002 and buttons:
        modal, layout_note = looks_like_dialog(buttons, frame.shape, buckets)
        if modal:
            conf = min(0.95, 0.4 + dark_plate * 0.4 + min(0.3, has_saturated * 6))
            return Detection(Screen.CONNECTION_ERROR, conf,
                             f"dark plate {dark_plate:.0%} + {layout_note}",
                             buttons=buttons, mean_bgr=mean,
                             unique_colours=buckets)
        # Dark, saturated and full of buttons, but the buttons are laid out
        # like a list or a column, not a modal. This is what the server
        # browser looks like. Say so explicitly, because otherwise this
        # reads as a silent miss.
        return Detection(Screen.IN_GAME, 0.5,
                         f"dark and saturated but not a modal: {layout_note} "
                         f"(dark {dark_plate:.0%})",
                         buttons=buttons, mean_bgr=mean,
                         unique_colours=buckets)

    # In-game: lots of colour, no dominant dark plate. Threshold from
    # measurement - busy game frames occupy 82-92 of the 125 buckets, a
    # flat-shaded dialog 7-20. 45 sits in the gap with margin on both sides.
    if buckets > 45 and dark_plate < 0.55:
        return Detection(Screen.IN_GAME, 0.8,
                         f"rich frame, buckets={buckets}/{BUCKET_TOTAL}, "
                         f"dark {dark_plate:.0%}",
                         buttons=buttons, mean_bgr=mean,
                         unique_colours=buckets)

    return Detection(Screen.IN_GAME, 0.3,
                     f"unclassified, buckets={buckets}/{BUCKET_TOTAL}, "
                     f"dark {dark_plate:.0%}",
                     buttons=buttons, mean_bgr=mean, unique_colours=buckets)


# --- layout gate --------------------------------------------------------
#
# Measured against the LIVE Roblox client, not only synthetic frames. Two
# separate false positives were found and fixed here, both in normal play:
#
#  1. The server browser: a dark screen with a "Waiting for response..."
#     modal and a column of green JOIN buttons. It satisfied every colour
#     test for an error dialog - 96-99% dark, 5-21% saturated, 6-8
#     button-shaped regions. Left alone the tool would have clicked JOIN
#     and thrown the player out of their own session.
#
#  2. Night gameplay: the sky is near-black, so the frame reads as 86-97%
#     dark with 30-60% saturated pixels, and a HUD element (the ability
#     bar) is a single flat rectangle. Colour and button-layout alone both
#     pass on that, so the first fix still produced 6 false positives in 30
#     live frames.
#
# What finally separates them is COLOUR COMPLEXITY. Roblox's error dialog is
# flat-shaded CoreUI: a handful of solid rectangles, so it occupies very few
# colour buckets. Gameplay is a textured 3D render - gradients, sprites,
# textures - and occupies many more. Measured on this machine:
#
#   screen                 buckets occupied (of 125)
#   error dialog (synth)       7 - 20
#   server browser (live)     12 - 34
#   night gameplay (live)     49 - 98
#   day gameplay (live)       80 - 97
#
# The gap between the UI screens (max 34) and gameplay (min 49) is the
# cleanest separation available, and it is exactly the physical reason a
# dialog is not a game frame. So a dialog must be LOW COMPLEXITY as well as
# few, compact and centred.
#
# The gate is deliberately conjunctive: low complexity AND few AND compact
# AND centred. Every clause rejects a screen observed to cause a real false
# positive, so removing any one of them reopens a known failure.

#: Most buttons a Roblox modal may contain. Measured dialog: 2.
MAX_DIALOG_BUTTONS = 4
#: Largest fraction of frame height the button row may span. Dialog 0.06,
#: server browser 0.46.
MAX_BUTTON_SPAN_Y = 0.20
#: How far the button block's centre may sit from the frame's centre
#: horizontally. Dialog centre 0.50, server browser 0.77.
MAX_BUTTON_OFFCENTRE_X = 0.18

#: Highest colour-bucket occupancy a dialog may have. Measured UI screens
#: peaked at 34; the darkest gameplay frame observed was 49. 42 sits in
#: that gap. Below it, the frame is flat-shaded UI rather than a 3D render -
#: which is the physical reason a dialog differs from a game frame.
MAX_DIALOG_BUCKETS = 42


def looks_like_dialog(buttons: list[Button], frame_shape: tuple[int, ...],
                      buckets: int | None = None) -> tuple[bool, str]:
    """Whether a frame is a modal dialog rather than a game screen.

    Returns (ok, reason). The reason is logged either way so a refusal can
    be explained rather than looking like the tool did nothing.

    `buckets` is frame_stats' colour-complexity score. When supplied it is
    checked FIRST, because it is the cheapest of the four tests and the
    strongest: it rejects every game frame measured, including night
    gameplay that passes all three layout tests.
    """
    if buckets is not None and buckets > MAX_DIALOG_BUCKETS:
        return False, (f"too complex for a dialog: {buckets}/{BUCKET_TOTAL} "
                       f"colour buckets, a flat-shaded dialog is under "
                       f"{MAX_DIALOG_BUCKETS}")

    if not buttons:
        return False, "no buttons"
    h, w = frame_shape[:2]
    if h <= 0 or w <= 0:
        return False, "degenerate frame"

    if len(buttons) > MAX_DIALOG_BUTTONS:
        return False, (f"{len(buttons)} buttons, a modal has at most "
                       f"{MAX_DIALOG_BUTTONS}")

    x0 = min(b.x for b in buttons)
    x1 = max(b.x + b.w for b in buttons)
    y0 = min(b.y for b in buttons)
    y1 = max(b.y + b.h for b in buttons)
    span_y = (y1 - y0) / h
    if span_y > MAX_BUTTON_SPAN_Y:
        return False, (f"buttons span {span_y:.0%} of frame height, a modal's "
                       f"row is under {MAX_BUTTON_SPAN_Y:.0%}")

    centre_x = ((x0 + x1) / 2) / w
    if abs(centre_x - 0.5) > MAX_BUTTON_OFFCENTRE_X:
        return False, (f"button block centred at {centre_x:.2f}, a modal is "
                       f"centred near 0.50")

    return True, (f"{len(buttons)} button(s), span {span_y:.0%} of height, "
                  f"centred at {centre_x:.2f}"
                  + (f", {buckets}/{BUCKET_TOTAL} buckets" if buckets is not None
                     else ""))


def pick_primary(det: Detection) -> Button | None:
    if not det.buttons:
        return None
    prim = [b for b in det.buttons if b.kind == "primary"]
    pool = prim or det.buttons
    return max(pool, key=lambda b: (b.score, b.w))
