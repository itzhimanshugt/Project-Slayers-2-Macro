"""Entry point:  python run.py fishing

Hotkeys: F6 start/stop, F12 panic, F7 recalibrate reminder.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from slayersmacro import config, log  # noqa: E402
from slayersmacro.engine import Engine  # noqa: E402
from slayersmacro.hotkeys import HotkeyManager  # noqa: E402
from slayersmacro.macros import ALL_MACROS  # noqa: E402
from slayersmacro.window import find_window, is_foreground  # noqa: E402

CFG_PATH = Path(__file__).resolve().parent / "config" / "config.toml"

BANNER = r"""
  Slayers Macro - external screen-reading automation
  F6 start/stop   F7 which regions   F12 PANIC (releases everything)
"""


def usage() -> int:
    print(BANNER)
    print("usage: python run.py <macro>\n")
    print("macros:")
    for name in ALL_MACROS:
        print(f"  {name}")
    print("\nfirst time? calibrate the boxes this macro needs:")
    print("  python tools/calibrate.py fishing   # or boss / combat / quest")
    print("\nbenchmarks:")
    print("  python tools/bench_input.py")
    print("  python tools/bench_capture.py")
    return 2


def main() -> int:
    if len(sys.argv) < 2:
        return usage()
    name = sys.argv[1].lower()
    if name in ("-h", "--help", "help"):
        return usage()
    if name not in ALL_MACROS:
        print(f"unknown macro '{name}'")
        return usage()

    cfg = config.load(CFG_PATH)
    if not CFG_PATH.exists():
        config.write_default(CFG_PATH)
        log.setup().info(f"wrote a default config to {CFG_PATH}")

    lg = log.setup()
    print(BANNER)

    win = find_window(cfg["general"].get("window_title", "Roblox"))
    if win is None:
        lg.error("No Roblox window found. Open Project Slayers 2 first.")
        return 1
    lg.info(f"Roblox window: '{win.title}' {win.width}x{win.height} "
            f"at ({win.left},{win.top})")
    if not is_foreground(win.hwnd):
        lg.warning("Roblox is not the focused window. Click it, then press F6.")

    engine = Engine(ALL_MACROS[name], cfg)
    state = {"running": False}

    def toggle() -> None:
        if state["running"]:
            engine.request_stop()
            lg.info("stopping...")
        else:
            if engine.should_stop():
                engine = Engine(ALL_MACROS[name], cfg)
            state["running"] = True
            lg.info(f"starting {name}")
            engine.start()

    def panic() -> None:
        engine.request_stop()
        engine.release_all()
        state["running"] = False
        lg.warning("PANIC - all input released")

    def show_regions() -> None:
        from slayersmacro.detect import Calibration
        cal = Calibration.load()
        if not cal.regions:
            lg.info("nothing calibrated yet - "
                    "run: python tools/calibrate.py fishing")
        for rname, r in cal.regions.items():
            lg.info(f"  {rname:<16} {r.w}x{r.h} at ({r.x},{r.y})")

    hk = HotkeyManager()
    hk.register("f6", toggle)
    hk.register("f12", panic)
    hk.register("f7", show_regions)
    try:
        hk.start()
    except Exception as exc:  # noqa: BLE001
        lg.error(f"could not install hotkeys: {exc}")
        return 1
    lg.info("hotkeys ready - F6 to start, F12 to panic, Ctrl+C to quit")

    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        lg.info("Ctrl+C - releasing input")
    finally:
        panic()
        hk.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())