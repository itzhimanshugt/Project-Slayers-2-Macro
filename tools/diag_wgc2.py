"""Find the real pixel-extraction call on a windows-capture Frame."""
import threading

import numpy as np
from windows_capture import WindowsCapture

ev = threading.Event()
box = {}


def on_frame_arrived(frame, control=None):
    box["f"] = frame
    box["c"] = control
    ev.set()


def on_closed(frame=None):
    pass


c = WindowsCapture(cursor_capture=False, draw_border=False,
                   minimum_update_interval=0, window_name="Roblox")
c.event(on_frame_arrived)
c.event(on_closed)
c.start_free_threaded()
ev.wait(15)
f = box["f"]
print("Frame type:", type(f).__name__)
print("width/height:", f.width, f.height)
print("public:", [n for n in dir(f) if not n.startswith("_")])
print()

bgr = f.convert_to_bgr()
print("convert_to_bgr() ->", type(bgr).__name__)
print("  public:", [n for n in dir(bgr) if not n.startswith("_")][:25])
print()

for name in ("save", "save_as_array", "to_numpy", "as_numpy", "read"):
    m = getattr(bgr, name, None)
    if callable(m):
        try:
            v = m()
            print(f"bgr.{name}() -> {type(v).__name__} "
                  f"{getattr(v, 'shape', '')}")
        except Exception as exc:
            print(f"bgr.{name}() raised {type(exc).__name__}: {exc}")

buf = getattr(f, "frame_buffer", None)
print("\nframe_buffer ->", type(buf).__name__ if buf is not None else None)
if buf is not None:
    print("  public:", [n for n in dir(buf) if not n.startswith("_")][:25])
    for name in ("save", "to_numpy", "read", "__array__"):
        m = getattr(buf, name, None)
        if callable(m):
            try:
                v = m() if name != "__array__" else m(f)
                print(f"  buf.{name} -> {type(v).__name__} "
                      f"{getattr(v, 'shape', '')}")
            except Exception as exc:
                print(f"  buf.{name} raised {type(exc).__name__}: {exc}")

if box.get("c") is not None:
    box["c"].stop()
