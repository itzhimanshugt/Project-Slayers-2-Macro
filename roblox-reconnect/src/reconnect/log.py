"""Logging, mirroring the macro tool's setup so both write the same way."""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"

_lock = __import__("threading").Lock()
_configured = False


def setup(level: str = "INFO", log_name: str = "reconnect"):
    global _configured
    with _lock:
        root = logging.getLogger("reconnect")
        if _configured:
            return root
        root.setLevel(logging.DEBUG)
        root.propagate = False

        fmt = logging.Formatter(
            "%(asctime)s.%(msecs)03d | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S")

        console = logging.StreamHandler(sys.stdout)
        console.setLevel(getattr(logging, level.upper(), logging.INFO))
        console.setFormatter(fmt)
        root.addHandler(console)

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            LOG_DIR / f"{log_name}.log", maxBytes=4 * 1024 * 1024,
            backupCount=5, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root.addHandler(fh)

        _configured = True
        return root


def get(name: str = "reconnect"):
    if not _configured:
        setup()
    return logging.getLogger(f"reconnect.{name}")
