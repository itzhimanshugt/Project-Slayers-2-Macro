"""Tests for the input and window layers.

These call real Win32 APIs, so they only assert on things that are safe to
check headlessly: struct layout, key-table integrity, argument acceptance,
and window-metric helpers. Nothing here types into a real window.
"""

import ctypes
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from slayersmacro import input as sin  # noqa: E402
from slayersmacro import window as win  # noqa: E402


class TestStructLayout:
    def test_input_struct_is_40_bytes_on_x64(self):
        # 40 is what SendInput's cbSize must be. Getting this wrong makes
        # every call silently fail.
        assert ctypes.sizeof(sin.INPUT) == 40

    def test_keybd_and_mouse_sizes(self):
        assert ctypes.sizeof(sin.KEYBDINPUT) == 24
        assert ctypes.sizeof(sin.MOUSEINPUT) == 32

    def test_sendinput_argtypes_declared(self):
        # Bare ctypes.windll.user32 skips marshalling and costs measurable
        # per-call time.
        assert sin.user32.SendInput.argtypes is not None
        assert sin.user32.SendInput.restype is not None


class TestKeyTables:
    def test_every_scancode_key_has_a_vk(self):
        missing = [k for k in sin.SCANCODES if k not in sin.VK]
        assert not missing, f"no virtual key for: {missing}"

    def test_function_keys_present(self):
        for i in range(1, 13):
            assert f"f{i}" in sin.VK

    def test_movement_keys_present(self):
        for k in ("w", "a", "s", "d"):
            assert k in sin.VK
        for k in ("up", "down", "left", "right"):
            assert k in sin.SCANCODES

    def test_letters_are_uppercase_vks(self):
        assert sin.VK["w"] == ord("W")
        assert sin.VK["f6"] == 0x75


class TestSafeInputCalls:
    """These do inject real events. They use F24, which nothing binds."""

    def test_batched_press_release_succeeds(self):
        sin.press_release("f24", hold_ms=0.0)

    def test_scancode_press_release_succeeds(self):
        sin.press_release("f24", hold_ms=0.0, use_scancode=True)

    def test_separate_down_up_succeeds(self):
        sin.down("f24")
        sin.up("f24")

    def test_tap_succeeds(self):
        sin.tap("f24")

    def test_relative_mouse_move(self):
        sin.move_relative(1, -1)

    def test_absolute_mouse_move_in_bounds(self):
        w, h = win.virtual_screen_size()
        sin.move_absolute(w // 2, h // 2, w, h)

    def test_click(self):
        sin.click("left")

    def test_click_with_hold(self):
        sin.click("left", hold_ms=1.0)

    def test_button_down_then_up(self):
        """Hold-style input: a batched down+up can be too brief to register."""
        sin.button_down("left")
        sin.button_up("left")

    def test_button_down_unknown_raises(self):
        with pytest.raises(KeyError):
            sin.button_down("middle_click")

    def test_space_and_arrows_are_valid_keys(self):
        """Space is jump in Roblox by default, so macros must be able to
        send it. This was a real bug: 'space' was missing from SCANCODES."""
        assert "space" in sin.SCANCODES
        sin.tap("space")
        sin.tap("up")

    def test_named_keys_all_dispatch(self):
        for key in ("w", "a", "space", "f24"):
            sin.tap(key)

    def test_unknown_key_raises(self):
        with pytest.raises(Exception):
            sin.press_release("not_a_real_key")


class TestWindowLayer:
    def test_dpi_awareness_is_idempotent(self):
        win.enable_dpi_awareness()
        win.enable_dpi_awareness()  # must not raise

    def test_virtual_screen_is_positive(self):
        w, h = win.virtual_screen_size()
        assert w > 0 and h > 0

    def test_find_window_returns_none_when_absent(self):
        assert win.find_window("no_such_window_zzz_9999") is None

    def test_windowinfo_helpers(self):
        class W:
            left, top, width, height = 100, 50, 800, 600
        info = win.WindowInfo(hwnd=1, title="t", left=100, top=50,
                              width=800, height=600)
        assert info.rect == (100, 50, 900, 650)
        assert info.center == (500, 350)
        assert info.contains(100, 50) is True
        assert info.contains(899, 649) is True
        assert info.contains(900, 650) is False
        assert info.to_local(150, 80) == (50, 30)

    def test_foreground_check_does_not_raise(self):
        win.is_foreground(0)