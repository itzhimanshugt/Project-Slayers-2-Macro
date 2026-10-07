"""Measure real screen capture cost at several region sizes.

The whole design rests on this: if a small region capture is cheap, the
macro loop can run fast enough to be useful.

Run:  python tools/bench_capture.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from slayersmacro.capture import Capture  # noqa: E402

# Regions that matter: a fishing bar, a health bar, a minimap corner,
# and the full client area as the worst case.
SIZES = [
    ("fishing bar strip", 12, 400),
    ("boss health bar", 24, 600),
    ("minimap corner", 256, 256),
    ("half screen", 800, 1280),
    ("full 2560x1600", 1600, 2560),
]

N = 300


def main() -> None:
    cap = Capture()
    vs = cap.virtual_screen
    print(f"virtual desktop: {vs['width']}x{vs['height']} at ({vs['left']},{vs['top']})")
    print(f"\nregion capture cost, {N} iterations each\n")

    print(f"  {'region':<20}{'px':>12}{'p50 us':>11}{'p99 us':>11}{'max us':>11}")
    for name, h, w in SIZES:
        cap.grab(0, 0, w, h)  # warm
        samples = []
        for _ in range(N):
            t0 = time.perf_counter_ns()
            cap.grab(0, 0, w, h)
            t1 = time.perf_counter_ns()
            samples.append((t1 - t0) / 1000.0)
        samples.sort()
        px = w * h
        print(
            f"  {name:<20}{px:>12,}{statistics.median(samples):>11.1f}"
            f"{samples[int(N * 0.99)]:>11.1f}{samples[-1]:>11.1f}"
        )

    # Practical ceiling: strip region plus a press, looped.
    print("\nsustained loop, fishing-bar strip + one key press:")
    loop_n = 3000
    from slayersmacro import input as sin
    t0 = time.perf_counter_ns()
    for _ in range(loop_n):
        cap.grab(0, 0, 400, 12)
        sin.tap("f24")
    total_ms = (time.perf_counter_ns() - t0) / 1e6
    print(f"  total {total_ms:8.1f}ms  ->  {total_ms / loop_n * 1000:7.2f}us per cycle")
    print(f"  ceiling {loop_n / (total_ms / 1000):,.0f} cycles/sec")

    cap.close()


def bench_dxcam(N: int = 300) -> None:
    """DXGI Desktop Duplication, for comparison against mss.

    mss shows a flat ~4.1ms floor regardless of region size, which suggests
    a fixed per-call cost rather than per-pixel cost. If dxcam is cheaper
    the whole loop budget changes.
    """
    try:
        import dxcam
    except ImportError:
        print("\ndxcam not installed, skipping")
        return

    print("\ndxcam (DXGI Desktop Duplication):")
    cam = dxcam.create(output_color="BGR")
    if cam is None:
        print("  no DXGI output available")
        return
    try:
        for name, left, top, w, h in [
            ("fishing bar strip", 0, 0, 400, 12),
            ("minimap corner", 0, 0, 256, 256),
            ("full 2560x1600", 0, 0, 2560, 1600),
        ]:
            cam.grab(region=(left, top, left + w, top + h))
            samples = []
            for _ in range(N):
                t0 = time.perf_counter_ns()
                cam.grab(region=(left, top, left + w, top + h))
                t1 = time.perf_counter_ns()
                samples.append((t1 - t0) / 1000.0)
            samples.sort()
            print(
                f"  {name:<24}{w * h:>12,}{statistics.median(samples):>11.1f}"
                f"{samples[int(N * 0.99)]:>11.1f}{samples[-1]:>11.1f}"
            )
    finally:
        cam.release()


if __name__ == "__main__":
    main()
    bench_dxcam()