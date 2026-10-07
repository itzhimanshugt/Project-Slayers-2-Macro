"""Thread-safe logging: rotating file + console.

Every action the macro takes goes through here so there is always a record
of what it did and when. That log is the main debugging tool.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
import threading
from pathlib import Path

_lock = threading.Lock()
_configured = False

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"


def setup(level: str = "INFO", log_name: str = "macro") -> logging.Logger:
    """Configure and return the root logger. Safe to call more than once."""
    global _configured
    with _lock:
        root = logging.getLogger("slayers")
        if _configured:
            return root

        root.setLevel(logging.DEBUG)
        # Screenshots and traces are the noisy part; keep them at DEBUG.
        root.propagate = False

        fmt = logging.Formatter(
            "%(asctime)s.%(msecs)03d | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S",
        )

        console = logging.StreamHandler(sys.stdout)
        console.setLevel(getattr(logging, level.upper(), logging.INFO))
        console.setFormatter(fmt)
        root.addHandler(console)

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_DIR / f"{log_name}.log",
            maxBytes=4 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

        _configured = True
        return root


def get(name: str = "macro") -> logging.Logger:
    if not _configured:
        setup()
    return logging.getLogger(f"slayers.{name}")


# Timing helper: "took 4.2ms" reads better in a log than raw numbers.
def elapsed_ms(start_ns: int, end_ns: int) -> float:
    return (end_ns - start_ns) / 1_000_000