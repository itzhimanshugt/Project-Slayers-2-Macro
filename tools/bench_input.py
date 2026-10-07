"""Measure real synthetic input latency on this machine.

Uses F24 as the test key. It produces genuine scan-code events but almost
nothing binds it, so the benchmark does not type into whatever window is
focused.

Run:  python tools/bench_input.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pydirectinput  # noqa: E402

# Critical: pydirectinput's default PAUSE is 0.1s, which adds 100ms to
# EVERY call. Measured on this machine: 201ms per keyDown/keyUp pair with
# the default, vs 143us with PAUSE=0. That is a 1000x difference, and it
# is the single biggest footgun in the pydirectinput API.
pydirectinput.PAUSE = 0

from slayersmacro import input as sin  # noqa: E402

N = 1000
TEST_KEY = "f24"  # harmless: real events, nothing listens to it


def bench(fn, n=N, label="") -> dict:
    fn()  # warm up
    samples = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1000.0)  # microseconds
    samples.sort()
    return {
        "label": label,
        "min": samples[0],
        "p50": statistics.median(samples),
        "p99": samples[int(len(samples) * 0.99)],
        "max": samples[-1],
        "mean": statistics.fmean(samples),
    }


def show(r: dict) -> None:
    print(
        f"  {r['label']:<34} min {r['min']:7.2f}us  "
        f"p50 {r['p50']:7.2f}us  p99 {r['p99']:7.2f}us  max {r['max']:8.2f}us"
    )


def main() -> None:
    print(f"SendInput benchmark, {N} iterations each")
    print("Test key is F24 - no visible effect on any app.\n")

    results = []

    # 1. Batched down+up in one SendInput call. This is the fast path the
    #    tool actually uses.
    results.append(bench(lambda: sin.press_release(TEST_KEY, hold_ms=0.0),
                         label="ctypes SendInput (batched)"))

    # 2. Same, but asking for scan codes.
    results.append(bench(lambda: sin.press_release(TEST_KEY, hold_ms=0.0,
                                                   use_scancode=True),
                         label="ctypes SendInput (scancode)"))

    # 3. Down then up as two separate calls - the non-batched path.
    def two_calls():
        sin.down(TEST_KEY)
        sin.up(TEST_KEY)

    results.append(bench(two_calls, label="ctypes SendInput (two calls)"))

    # 4. pydirectinput, for comparison.
    def pd():
        pydirectinput.keyDown(TEST_KEY)
        pydirectinput.keyUp(TEST_KEY)

    results.append(bench(pd, label="pydirectinput press+release"))

    # 5. Cost of a whole sustained loop: one key press per iteration.
    print("\nPer-call cost:")
    for r in results:
        show(r)

    print("\nSustained loop (one press per iteration, 1000 iters):")
    loop_n = 1000
    t0 = time.perf_counter_ns()
    for _ in range(loop_n):
        sin.tap(TEST_KEY)
    total_ms = (time.perf_counter_ns() - t0) / 1e6
    print(f"  total {total_ms:8.2f}ms  ->  {total_ms / loop_n * 1000:7.2f}us per press")
    print(f"  ceiling {loop_n / (total_ms / 1000):,.0f} presses/sec")

    best = min(results, key=lambda r: r["p50"])
    print(f"\nfastest usable path: {best['label']} at {best['p50']:.2f}us p50")


if __name__ == "__main__":
    main()