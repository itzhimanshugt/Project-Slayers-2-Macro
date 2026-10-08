"""What happens over time on a live client. Not a unit test - an observation.

The false positives that mattered were all found by watching a real game for
a while, not by reading code. This module supports that: it runs the shipped
supervisor against the live window with clicking disabled and reports what
it would have done.

Run it directly:

    python -m tests.test_response --seconds 60

It is not collected by pytest (the name and the __main__ guard keep it that
way) because it needs a running Roblox and takes minutes.
"""

import sys
import time

import cv2

from reconnect.supervisor import RecoveryConfig, Supervisor
from reconnect.ui import Screen, classify
from reconnect.wgc import open_capture
from reconnect.window import find_roblox


def watch(seconds: float = 60.0, interval: float = 0.2,
          shot_dir: str | None = None) -> int:
    """Run the supervisor against the live client and report.

    Always dry run: this never clicks. Exit code is non-zero if any frame
    was classified as an error, so it can be wired into a check.
    """
    win = find_roblox()
    if win is None:
        print("no Roblox window found; start the game first")
        return 2
    cap = open_capture(prefer_wgc=True)
    print(f"watching {win.describe()} for {seconds:.0f}s via {cap.backend} "
          f"(dry run, nothing will be clicked)")

    sup = Supervisor(cap, RecoveryConfig(confirm_frames=4,
                                         action_cooldown_s=1.0),
                     logger=None, dry_run=True)

    counts: dict[str, int] = {}
    false_positives = 0
    polls = 0
    deadline = time.perf_counter() + seconds
    try:
        while time.perf_counter() < deadline:
            frame = cap.grab()
            if frame is None:
                time.sleep(interval)
                continue
            polls += 1
            det = classify(frame)
            counts[det.screen.value] = counts.get(det.screen.value, 0) + 1
            if det.screen is Screen.CONNECTION_ERROR:
                false_positives += 1
                print(f"  FALSE POSITIVE at poll {polls}: {det.summary()}")
                if shot_dir:
                    path = f"{shot_dir}/fp_{polls:04d}.png"
                    cv2.imwrite(path, frame[:, :, :3])
                    print(f"    saved {path}")
            _, _ = sup.tick(frame, win)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        cap.close()

    print(f"\npolls={polls}")
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<18} {n:>5}  ({100 * n / max(1, polls):.1f}%)")
    print(f"\nerror classifications: {false_positives}")
    print("Any non-zero here means the tool would have acted during normal "
          "play. Fix the thresholds before trusting it.")
    return 1 if false_positives else 0


if __name__ == "__main__":
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    raise SystemExit(watch(secs))