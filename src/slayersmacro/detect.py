"""Detection: finding bars, bars-in-bars, and health on screen.

Design note - everything here is self-calibrating on purpose. Research
found no primary source describing Slayers 2's UI (the one site claiming
to publish official keybindings had fabricated them), so there are no
trusted hardcoded pixel values. Regions come from the user picking them,
and thresholds adapt to what is actually on screen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

CONF_DIR = Path(__file__).resolve().parent.parent.parent / "config"


@dataclass
class Region:
    """A rect in window-local coordinates, with a name."""

    name: str
    x: int
    y: int
    w: int
    h: int

    def to_screen(self, win) -> tuple[int, int, int, int]:
        return win.left + self.x, win.top + self.y, self.w, self.h

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y:self.y + self.h, self.x:self.x + self.w]


@dataclass
class Calibration:
    """Persisted user-picked regions, in window-local coords."""

    regions: dict[str, Region] = field(default_factory=dict)
    # Anchor used to translate between window sizes. Detection ratios are
    # measured relative to this, so moving between windowed and fullscreen
    # does not break the macros.
    anchor_w: int = 0
    anchor_h: int = 0

    def save(self, path: Path | None = None) -> Path:
        path = path or (CONF_DIR / "calibration.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "anchor_w": self.anchor_w,
            "anchor_h": self.anchor_h,
            "regions": {k: vars(v) for k, v in self.regions.items()},
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "Calibration":
        path = path or (CONF_DIR / "calibration.json")
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        # vars(v) already includes "name", so do not pass it twice.
        regions = {k: Region(**v) for k, v in data.get("regions", {}).items()}
        return cls(regions=regions,
                   anchor_w=data.get("anchor_w", 0),
                   anchor_h=data.get("anchor_h", 0))

    def scaled(self, win) -> "Calibration":
        """Return a copy whose regions are scaled to the window's current size.

        This is what makes the tool survive a resize or a switch to
        fullscreen without recalibrating.
        """
        if not self.anchor_w or not self.anchor_h:
            return self
        sx = win.width / self.anchor_w
        sy = win.height / self.anchor_h
        out = {}
        for k, r in self.regions.items():
            out[k] = Region(
                name=r.name,
                x=int(r.x * sx), y=int(r.y * sy),
                w=max(1, int(r.w * sx)), h=max(1, int(r.h * sy)),
            )
        return Calibration(regions=out, anchor_w=self.anchor_w,
                           anchor_h=self.anchor_h)


class BarDetector:
    """Finds a moving marker inside a track - the fishing minigame shape.

    Two passes:
      1. find_track(): locate the light bar (the zone the marker goes in)
      2. find_marker(): locate the bright ball

    Both use HSV thresholds with a LOCK_HITS debounce, so a single noisy
    frame cannot trigger a keypress. That debounce matters more than it
    sounds: without it, one frame of compression noise fires the reel.
    """

    def __init__(self, track_lo: int = 60, track_hi: int = 200,
                 marker_lo: int = 200, lock_hits: int = 3) -> None:
        self.track_lo, self.track_hi = track_lo, track_hi
        self.marker_lo = marker_lo
        self.lock_hits = lock_hits
        self._track_hits = 0
        self._marker_hits = 0
        self.last_track = 0.0
        self.last_marker = 0.0

    def reset(self) -> None:
        self._track_hits = self._marker_hits = 0

    @staticmethod
    def _brightness_mask(hsv: np.ndarray, lo: int, hi: int) -> np.ndarray:
        v = hsv[:, :, 2]
        return cv2.inRange(v, lo, hi)

    def find_track(self, frame: np.ndarray) -> float | None:
        """Centre of the target zone as a 0..1 fraction across the frame."""
        if frame is None or frame.size == 0:
            return None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = self._brightness_mask(hsv, self.track_lo, self.track_hi)
        # The bright marker sits ABOVE track_hi's brightness, so inRange
        # punches a hole straight through the track and splits it in two.
        # Flood-filling across any dark pixel would swallow the hole and
        # also swallow the background, so instead: take the span from the
        # leftmost to the rightmost track component instead of trusting a
        # single connected blob.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8))
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if n < 2:
            self._track_hits = 0
            return None
        comps = [i for i in range(1, n)
                 if stats[i, cv2.CC_STAT_WIDTH] >= 4
                 and stats[i, cv2.CC_STAT_HEIGHT] >= max(3, frame.shape[0] // 6)]
        if not comps:
            self._track_hits = 0
            return None
        left = min(stats[i, cv2.CC_STAT_LEFT] for i in comps)
        right = max(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]
                    for i in comps)
        span = right - left
        if span < 8:
            self._track_hits = 0
            return None
        centre = left + span / 2.0
        self._track_hits += 1
        pos = float(centre / max(1, frame.shape[1]))
        if self._track_hits >= self.lock_hits:
            self.last_track = pos
        return pos

    def find_marker(self, frame: np.ndarray) -> float | None:
        """Centre of the bright marker as a 0..1 fraction across the frame."""
        if frame is None or frame.size == 0:
            return None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = self._brightness_mask(hsv, self.marker_lo, 255)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if n < 2:
            self._marker_hits = 0
            return None
        # Squarish and small: that is the ball, not the track.
        best, best_score = None, 0.0
        for i in range(1, n):
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            a = stats[i, cv2.CC_STAT_AREA]
            if w < 3 or h < 3 or w > frame.shape[1] * 0.4:
                continue
            aspect = min(w, h) / max(w, h)
            score = aspect * a
            if score > best_score:
                best_score, best = score, i
        if best is None:
            self._marker_hits = 0
            return None
        cx = centroids[best][0]
        self._marker_hits += 1
        if self._marker_hits >= self.lock_hits:
            self.last_marker = float(cx / max(1, frame.shape[1]))
        return float(cx / max(1, frame.shape[1]))

    def centred(self, band_frac: float = 0.06) -> bool | None:
        """True when marker sits inside the track's centre band.

        Returns None until both sides have locked, so the caller can tell
        "not in the band" apart from "no idea yet".
        """
        if self._track_hits < self.lock_hits or self._marker_hits < self.lock_hits:
            return None
        return abs(self.last_marker - self.last_track) <= band_frac


class HealthReader:
    """Reads a health bar as a 0..1 fraction.

    Assumes the bar fills from the left and depletes toward it. Works on
    the usual red/green health bar. Returns None rather than guessing when
    the frame does not look like a bar.
    """

    def __init__(self, pixels_per_pct: float = 0.0) -> None:
        self.pixels_per_pct = pixels_per_pct
        self._calibrate: np.ndarray | None = None

    def calibrate(self, frame: np.ndarray) -> None:
        """Learn the bar's colours from a known-full frame."""
        if frame is not None and frame.size:
            self._calibrate = frame.reshape(-1, 3)

    def read(self, frame: np.ndarray, threshold: float = 0.35) -> float | None:
        """Fraction of the bar that is still filled."""
        if frame is None or frame.size == 0:
            return None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        # Saturated and bright reads as "filled". Dark and desaturated as
        # "empty". This survives the bar changing colour between states.
        filled = (s > 90) & (v > 70)
        frac = float(filled.mean())
        if frac <= 0.02 or frac >= 0.98:
            return None
        # Normalise against a learned full bar when we have one.
        if self.pixels_per_pct > 0:
            frac = min(1.0, frac / self.pixels_per_pct)
        return frac if frac >= threshold else 0.0


class MovementWatch:
    """Detects that the character is not moving, by watching scene motion.

    We cannot read the character's real position from pixels, so we watch
    whether the frame is changing at all while we think we are walking.
    A rock in the way means the view stops moving even though W is held.
    """

    def __init__(self, change_threshold: float = 1.2) -> None:
        self.change_threshold = change_threshold
        self._prev: np.ndarray | None = None
        self.still_ms = 0
        self.total_ms = 0

    def reset(self) -> None:
        self._prev = None
        self.still_ms = 0
        self.total_ms = 0

    def update(self, frame: np.ndarray, dt_ms: float) -> bool:
        """Returns True when we believe we are stuck."""
        if frame is None:
            return False
        small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
        moving = False
        if self._prev is not None and self._prev.shape == small.shape:
            diff = cv2.absdiff(small, self._prev).mean()
            moving = diff > self.change_threshold
        self._prev = small
        self.total_ms += dt_ms
        self.still_ms = 0 if moving else self.still_ms + dt_ms
        return False