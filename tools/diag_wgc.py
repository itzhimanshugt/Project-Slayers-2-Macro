"""Work out the windows-capture callback contract empirically.

The library dispatches on handler __name__ and the callback signature has
changed between releases. Rather than guess, print what actually arrives.
"""
import threading

from windows_capture import WindowsCapture

ev = threading.Event()
calls = []


def on_frame_arrived(*a):
    if not calls:
        for i, x in enumerate(a):
            meths = [n for n in dir(x) if not n.startswith("_")]
            print(f"arg{i}: {type(x).__name__}")
            print(f"        methods: {meths}")
    calls.append(a)
    ev.set()


def on_closed(*a):
    print("closed handler got", [type(x).__name__ for x in a])


c = WindowsCapture(cursor_capture=False, draw_border=False,
                   minimum_update_interval=0, window_name="Roblox")
c.event(on_frame_arrived)
c.event(on_closed)
c.start_free_threaded()
ev.wait(15)
print("frame callbacks:", len(calls))
if calls:
    for i, x in enumerate(calls[0]):
        for attr in ("save_as_array", "frame", "to_array"):
            if hasattr(x, attr):
                print(f"  arg{i} has {attr}")
                try:
                    if attr != "frame":
                        v = getattr(x, attr)
                        v = v() if callable(v) else v
                        print(f"    -> {getattr(v, 'shape', type(v))}")
                except Exception as exc:
                    print(f"    -> raised {type(exc).__name__}: {exc}")
c.stop()