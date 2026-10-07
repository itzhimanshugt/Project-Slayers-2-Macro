"""Config load/save backed by TOML.

Kept deliberately flat and commented so it can be edited by hand.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import tomli_w

DEFAULT_CONFIG: dict[str, Any] = {
    "general": {
        # Roblox window title substring to attach to.
        "window_title": "Roblox",
        # Refuse to run unless the Roblox window is focused. Stops the tool
        # from typing into Discord or your browser.
        "require_focus": True,
        # Randomise delays by +/- this many ms so timing is not mechanical.
        "jitter_ms": 0,
    },
    "hotkeys": {
        "start_stop": "f6",
        "panic": "f12",
        "recalibrate": "f7",
        "toggle_logging": "f8",
    },
    "input": {
        # Hold time for a single key press. Must exceed one game frame or
        # Roblox may not register it at all.
        "tap_ms": 18,
        # Cap the loop rate. 0 means run flat out.
        "loop_hz": 0,
    },
    "safety": {
        "settle_frames": 3,
        "stable_for_ms": 400,
        "detect_occlusion": True,
    },
    "adaptation": {
        "learn_timing": True,
        "adapt_thresholds": False,
    },
    "recovery": {
        "confirm_frames": 4,
        "step_pause_ms": 900,
        "max_auto_rung": "respawn",
        "max_cycles": 6,
    },
    "combat": {
        # Roblox's default attack is unbound, so it is NOT Space - Space is
        # jump. This is the one binding that matters and it is verified from
        # Roblox's own default controls, not from a guide.
        "attack_key": "mouse1",
        "m1_hold_ms": 18,
        "m1_interval_ms": 90,
        # Sources consistently report a 5th hit in an M1 string pushing the
        # character out of range, so the loop stops at 4 and waits.
        "combo_length": 4,
        "skill_interval_ms": 4000,
        "skill_keys": ["1", "2", "3"],
        "use_skills": True,
    },
    "fishing": {
        "cast_interval_ms": 2600,
        # Half-width of the target band, as a percentage of the bar. Wider
        # is safer and less precise. Below about 0.5% the game running at
        # less than 240fps starts to cause misses on its own.
        "hit_band_pct": 12.0,
        # Decide using the Kalman prediction at the actuation horizon rather
        # than the position just measured. Off = react to the current frame,
        # which is always ~7ms stale.
        "predictive": True,
        # Stop after this many consecutive frames with no bar detected,
        # instead of pressing keys against a bar the macro cannot see.
        "max_consecutive_misses": 12,
        "buy_bait": False,
        "collect_fish": True,
    },
    "boss": {
        "attack_interval_ms": 110,
        "retreat_hp_pct": 30,
        "heal_ms": 2200,
        "max_retreats": 6,
    },
    "stuck": {
        "enabled": True,
        # If position has not changed for this long while we think we are
        # moving, treat it as stuck.
        "detect_ms": 700,
        "max_retries": 4,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load(path: str | Path) -> dict[str, Any]:
    """Load config, filling in any missing keys from the defaults."""
    path = Path(path)
    if not path.exists():
        return dict(DEFAULT_CONFIG)
    with path.open("rb") as fh:
        user = tomllib.load(fh)
    return _deep_merge(DEFAULT_CONFIG, user)


def save(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        tomli_w.dump(data, fh)


def write_default(path: str | Path) -> None:
    """Create a fully commented config the first time we run."""
    path = Path(path)
    if not path.exists():
        save(path, DEFAULT_CONFIG)