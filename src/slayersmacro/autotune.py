"""Self-calibrating detection thresholds.

The problem this solves: hand-tuned HSV ranges break when the game's
lighting, the weather, the map, or the player's own colour changes. And we
have no verified ground truth for Slayers 2's UI to hand-tune from in the
first place.

So instead: record a few frames, find the bar geometry and the marker's
colour automatically, and write them into the config. Run it once, look at
the numbers, and adjust if the auto-detect grabbed the wrong thing.

Otsu is the right tool for the geometry split. It finds the threshold that
maximises between-class variance, so it does not need to be told what the
bar looks like - only that it is one bright thing on a dark background.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

CONF_DIR = Path(__file__).resolve().parent.parent.parent / "config"


@dataclass
class TunedThresholds:
    """Learned detection parameters for one region."""
    region: str
    # Otsu split point on the V channel, 0-255.
    bar_v_min: int
    bar_v_max: int
    marker_v_min: int
    # Otsu on saturation separates a coloured bar from grey UI.
    bar_s_min: int
    geometry: dict          # width, height, fill ratio of the detected bar
    sample_frames: int
    confidence: float       # 0-1, how consistent the frames agreed

    def save(self, path: Path | None = None) -> Path:
        path = path or (CONF_DIR / "tuned.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        data[self.region] = asdict(self)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load_all(cls, path: Path | None = None) -> dict:
        path = path or (CONF_DIR / "tuned.json")
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))


def otsu_split(channel: np.ndarray) -> tuple[int, float]:
    """Return (threshold, separation) splitting a channel into two classes.

    `separation` is the normalised between-class variance: 1.0 means a
    perfect two-tone image, near 0 means the region has no obvious bar in
    it. Using it as a confidence check stops the tuner confidently writing
    nonsense when the box was drawn in the wrong place.
    """
    if channel.size == 0:
        return 0, 0.0
    chan = channel if channel.ndim == 2 else channel.reshape(-1, 1)
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256]).flatten()
    total = hist.sum()
    if total == 0:
        return 0, 0.0
    prob = hist / total

    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]

    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = np.where(denom > 0, (mu_t * omega - mu) ** 2 / denom, 0.0)

    k = int(np.argmax(sigma_b))
    # Return the MIDPOINT between the two class centres, not k itself.
    # cv2's own Otsu returns the bin that opens class 2, which for a clean
    # two-tone image lands exactly on the dark class's value - so a bar at
    # V=200 would report a threshold of 200 and, used as a lower bound,
    # exclude the very pixels we want. The midpoint is the safer convention.
    lo_centre = float(np.sum(prob[:k + 1] * np.arange(k + 1))
                      / max(1e-9, omega[k]))
    hi_mass = 1.0 - omega[k]
    hi_centre = float(np.sum(prob[k + 1:] * np.arange(k + 1, 256))
                      / hi_mass) if hi_mass > 1e-9 else k + 1.0
    midpoint = int(round((lo_centre + hi_centre) / 2.0))

    # Absolute between-class variance, normalised by the total variance.
    # Using sigma_b.max() here would always return 1.0 and make the score
    # meaningless - it is meant to answer "how cleanly did this split?",
    # not "how many candidate thresholds were there?".
    total_var = float(np.sum(prob * (np.arange(256) - mu_t) ** 2))
    if total_var <= 1e-9:
        return midpoint, 0.0
    best = float(sigma_b[k]) / total_var
    return midpoint, best


def find_bar_geometry(frame: np.ndarray,
                      min_width_ratio: float = 0.25) -> dict | None:
    """Find the widest horizontal bar and measure it.

    Returns None when nothing bar-shaped is present, which is the honest
    answer when the user drew the box in the wrong place.
    """
    if frame is None or frame.size == 0:
        return None
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    thr, sep = otsu_split(hsv[:, :, 2])
    if sep < 0.35:
        return None          # no two-tone structure: refuse to guess
    v = hsv[:, :, 2]
    # The bar is whichever Otsu class is MINORITY. A bar occupies part of
    # the crop, never all of it, so flipping to the smaller class handles
    # both a bright bar and a dark (depleting) one without a polarity
    # heuristic that breaks on an all-bright frame.
    above = v >= thr
    # Keep whichever Otsu class is the minority: a bar covers part of the
    # crop, never all of it, so this picks the bar for both a bright bar
    # and a dark depleting one without a polarity guess.
    minority = above if above.mean() <= 0.5 else ~above
    mask = (minority.astype(np.uint8)) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 15), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None
    best, best_score = None, 0.0
    for i in range(1, n):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if w < frame.shape[1] * min_width_ratio:
            continue
        if h > frame.shape[0] * 0.6:   # too tall to be a bar
            continue
        # Prefer wide AND full rectangles. The marker sitting on the bar is
        # taller than the bar, so height alone is a poor discriminator.
        score = w * min(h, frame.shape[0] * 0.5)
        if score > best_score:
            best_score, best = score, i
    if best is None:
        return None
    x, y, w, h, area = stats[best, cv2.CC_STAT_LEFT], stats[best, cv2.CC_STAT_TOP], \
        stats[best, cv2.CC_STAT_WIDTH], stats[best, cv2.CC_STAT_HEIGHT], \
        stats[best, cv2.CC_STAT_AREA]
    return {
        "x": int(x), "y": int(y), "w": int(w), "h": int(h),
        "fill_ratio": round(float(area) / max(1, w * h), 3),
        "otsu_threshold": int(thr),
        "separation": round(sep, 3),
    }


def learn_region(frames: list[np.ndarray], region: str) -> TunedThresholds | None:
    """Learn thresholds from several frames of the same region.

    Multiple frames matter: one frame might catch the marker mid-bounce or
    catch an animation frame with different lighting. Agreement across
    frames is what the confidence score measures.
    """
    frames = [f for f in frames if f is not None and f.size]
    if not frames:
        return None

    geos = [find_bar_geometry(f) for f in frames]
    geos = [g for g in geos if g]
    if not geos:
        # No frame contained a bar-shaped object. Returning thresholds here
        # would write plausible-looking numbers derived from nothing - the
        # exact failure that made the original confidence score useless.
        return None

    v_thresholds = [g["otsu_threshold"] for g in geos]
    separations = [g["separation"] for g in geos]
    sats = []
    marker_v = []
    for f in frames:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        sats.append(float(hsv[:, :, 1].mean()))
        marker_v.append(int(np.percentile(hsv[:, :, 2], 97)))

    v_med = int(np.median(v_thresholds))
    spread = float(np.std(v_thresholds))
    marker_min = int(np.median(marker_v))
    s_med = int(np.median(sats))

    # Two independent quality signals:
    #   spread      - how much the Otsu split moved between frames. A stable
    #                 structure gives a consistent split; an animation or a
    #                 mis-drawn box gives a jumpy one.
    #   separation  - how cleanly each frame split into two tones at all.
    # Consistency alone is not enough: eight identical blank frames are
    # perfectly consistent and completely useless.
    consistency = 1.0 - min(1.0, spread / 60.0)
    clarity = float(np.median(separations))
    confidence = float(np.clip(consistency * clarity, 0.0, 1.0))

    median_geo: dict = {}
    for k in geos[0]:
        if k in ("otsu_threshold", "separation"):
            continue
        median_geo[k] = int(np.median([g[k] for g in geos])) if k != "fill_ratio" \
            else round(float(np.median([g[k] for g in geos])), 3)

    return TunedThresholds(
        region=region,
        bar_v_min=max(1, v_med - 25),
        bar_v_max=min(255, v_med + 90),
        marker_v_min=max(1, marker_min - 18),
        bar_s_min=max(0, s_med - 30),
        geometry=median_geo,
        sample_frames=len(frames),
        confidence=round(confidence, 3),
    )


def record_frames(grab, region_name: str, count: int = 12,
                  interval_ms: float = 80) -> list[np.ndarray]:
    """Grab a burst of frames of one calibrated region for the tuner.

    `grab` is a callable taking a region name and returning a frame.
    """
    import time
    out = []
    for _ in range(count):
        f = grab(region_name)
        if f is not None:
            out.append(f)
        time.sleep(interval_ms / 1000.0)
    return out