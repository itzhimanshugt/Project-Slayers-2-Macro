"""A recoverable Roblox error dialog, rendered synthetically.

Two jobs: prove the detector recognises the shape, and give the supervisor
something realistic to recover from without waiting for a real disconnect.

The colours follow Roblox's own dialog conventions - a very dark plate with
a saturated primary button - rather than any specific game's artwork, so
the classifier should not be tuned to these pixels.
"""

from __future__ import annotations

import cv2
import numpy as np


def dark_dialog(w: int = 1280, h: int = 800,
                accent_bgr: tuple[int, int, int] = (40, 60, 220),
                secondary_bgr: tuple[int, int, int] = (70, 70, 78)
                ) -> np.ndarray:
    """A Roblox-style modal error: dark plate, one bright primary button,
    one dim secondary."""
    frame = np.full((h, w, 3), 26, np.uint8)

    # Dialog plate, centred.
    pw, ph = int(w * 0.52), int(h * 0.34)
    px, py = (w - pw) // 2, (h - ph) // 2
    cv2.rectangle(frame, (px, py), (px + pw, py + ph), (38, 38, 42), -1)
    cv2.rectangle(frame, (px, py), (px + pw, py + ph), (70, 70, 78), 2)

    # Title and body text: grey blocks, which is enough to make the frame
    # non-flat so it is not classified as a loading hang.
    for i, (rel_y, rel_w) in enumerate([(0.16, 0.44), (0.40, 0.62),
                                        (0.52, 0.55), (0.62, 0.38)]):
        bw = int(pw * rel_w)
        bh = 13
        bx = px + int((pw - bw) / 2)
        by = py + int(ph * rel_y)
        shade = 150 if i == 0 else 105
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh),
                      (shade, shade, shade), -1)

    # Buttons along the bottom of the plate.
    bh = 46
    by = py + ph - int(ph * 0.20)
    gap = 26
    bw = int(pw * 0.26)

    sec_x = px + int(pw * 0.12)
    cv2.rectangle(frame, (sec_x, by), (sec_x + bw, by + bh), secondary_bgr, -1)
    cv2.rectangle(frame, (sec_x, by), (sec_x + bw, by + bh), (110, 110, 120), 1)
    # label
    lw = int(bw * 0.45)
    cv2.rectangle(frame,
                  (sec_x + (bw - lw) // 2, by + bh // 2 - 4),
                  (sec_x + (bw + lw) // 2, by + bh // 2 + 4),
                  (170, 170, 175), -1)

    pri_x = px + pw - int(pw * 0.12) - bw
    cv2.rectangle(frame, (pri_x, by), (pri_x + bw, by + bh), accent_bgr, -1)
    cv2.rectangle(frame, (pri_x, by), (pri_x + bw, by + bh),
                  (min(255, accent_bgr[0] + 40),
                   min(255, accent_bgr[1] + 40),
                   min(255, accent_bgr[2] + 40)), 1)
    lw = int(bw * 0.50)
    cv2.rectangle(frame,
                  (pri_x + (bw - lw) // 2, by + bh // 2 - 5),
                  (pri_x + (bw + lw) // 2, by + bh // 2 + 5),
                  (245, 245, 245), -1)
    return frame


def primary_button_box(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    """Where the primary button actually is, for asserting the click target."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] > 130) & (hsv[:, :, 2] > 90)).astype(np.uint8) * 255
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    best = None
    best_area = 0
    for i in range(1, n):
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if bw < 80 or bh < 25 or bw < bh * 2:
            continue
        a = stats[i, cv2.CC_STAT_AREA]
        if a > best_area:
            best_area = a
            best = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], bw, bh)
    return best


def in_game_scene(w: int = 1280, h: int = 800, seed: int = 0) -> np.ndarray:
    """A busy, colourful frame that must NOT be classified as an error.

    Deliberately rich in colour: a real game frame has sky gradients,
    textured terrain and sprites, so it quantises to hundreds of distinct
    colours. An earlier version of this fixture used only flat rectangles
    and produced 24 quantised colours, which is fewer than a Roblox dialog -
    making the in-game case fail for the wrong reason, and quietly making
    the test easy to pass by loosening thresholds.
    """
    rng = np.random.default_rng(seed)
    # Sky: a smooth gradient across both axes, so every row differs.
    gy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    gx = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
    frame = np.concatenate([
        (150 + 60 * gx + 90 * gy) * np.ones_like(gx),
        (120 + 40 * gx + 110 * gy) * np.ones_like(gx),
        (70 + 120 * gx + 60 * gy) * np.ones_like(gx),
    ], axis=2)
    frame = np.clip(frame, 0, 255).astype(np.uint8)

    # Terrain with per-pixel variation, so the ground is not a flat block.
    ground_y = int(h * 0.62)
    gh = h - ground_y
    tex = rng.integers(-22, 22, (gh, w, 3))
    frame[ground_y:] = np.clip(
        frame[ground_y:].astype(np.int16) + tex, 0, 255).astype(np.uint8)

    # Structures with their own texture.
    for _ in range(12):
        x = int(rng.integers(0, max(1, w - 140)))
        bw = int(rng.integers(60, 200))
        bh = int(rng.integers(50, 280))
        y = int(rng.integers(0, max(1, h - bh)))
        base = rng.integers(40, 235, 3)
        patch = np.clip(base[None, None, :] +
                        rng.integers(-28, 28, (bh, bw, 3)), 0, 255)
        frame[y:y + bh, x:x + bw] = patch.astype(np.uint8)

    # A dark HUD strip, but small enough not to dominate the frame.
    cv2.rectangle(frame, (0, h - 60), (240, h), (30, 30, 34), -1)
    return frame


def white_screen(w: int = 1280, h: int = 800) -> np.ndarray:
    return np.full((h, w, 3), 246, np.uint8)


def black_screen(w: int = 1280, h: int = 800) -> np.ndarray:
    return np.full((h, w, 3), 3, np.uint8)


def mid_loading(w: int = 1280, h: int = 800) -> np.ndarray:
    """Roblox's dark blue-grey loading backdrop with a spinner."""
    frame = np.full((h, w, 3), (28, 26, 40), np.uint8)
    cx, cy, r = w // 2, h // 2, 46
    cv2.circle(frame, (cx, cy), r, (90, 80, 200), 4, cv2.LINE_AA)
    cv2.ellipse(frame, (cx, cy), (r, r), 0, 0, 120, (210, 200, 250), 5,
                cv2.LINE_AA)
    return frame


def with_cursor_and_noise(frame: np.ndarray, seed: int = 1,
                          cursor: tuple[int, int] = (640, 500)) -> np.ndarray:
    """Add the things a real frame always has: a cursor and compression
    noise. Tests must not pass only on pristine synthetic frames."""
    out = frame.copy()
    cx, cy = cursor
    cv2.arrowedLine(out, (cx - 12, cy), (cx, cy), (250, 250, 250), 2,
                    cv2.LINE_AA, tipLength=0.4)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 2.2, out.shape).astype(np.int16)
    out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return out
