"""Tests for config loading: defaults, merging, and the safety invariants.

The safety tests matter more than the parsing tests. A config file is the
one thing a user edits by hand, so a typo in it should fail loudly rather
than quietly disabling a guard.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from slayersmacro import config  # noqa: E402
from slayersmacro.recovery import Rung  # noqa: E402


EXAMPLE = os.path.join(os.path.dirname(__file__), "..", "config",
                       "config.example.toml")


class TestDefaults:
    def test_missing_file_returns_defaults(self, tmp_path):
        cfg = config.load(tmp_path / "nope.toml")
        assert cfg["general"]["require_focus"] is True
        assert cfg["input"]["tap_ms"] == 18

    def test_all_sections_present(self):
        cfg = config.load("config/does-not-exist.toml")
        for section in ("general", "hotkeys", "input", "safety", "combat",
                        "adaptation", "fishing", "boss", "recovery", "stuck"):
            assert section in cfg, f"missing section: {section}"

    def test_write_default_creates_file(self, tmp_path):
        p = tmp_path / "c.toml"
        config.write_default(p)
        assert p.exists()
        assert config.load(p)["input"]["tap_ms"] == 18


class TestMerging:
    def test_user_values_win(self, tmp_path):
        p = tmp_path / "c.toml"
        p.write_text('[input]\ntap_ms = 42\n')
        assert config.load(p)["input"]["tap_ms"] == 42

    def test_missing_keys_fall_back(self, tmp_path):
        p = tmp_path / "c.toml"
        p.write_text('[input]\ntap_ms = 42\n')
        cfg = config.load(p)
        # loop_hz was not in the file, so it must come from the defaults
        # rather than being absent.
        assert "loop_hz" in cfg["input"]

    def test_partial_section_keeps_siblings(self, tmp_path):
        p = tmp_path / "c.toml"
        p.write_text('[combat]\ncombo_length = 2\n')
        cfg = config.load(p)
        assert cfg["combat"]["combo_length"] == 2
        assert cfg["combat"]["attack_key"] == "mouse1"

    def test_new_section_is_not_dropped(self, tmp_path):
        """A user's older config must not lose whole new sections."""
        p = tmp_path / "c.toml"
        p.write_text('[general]\nwindow_title = "X"\n')
        assert "safety" in config.load(p)


class TestExampleConfigIsValid:
    def test_example_loads(self):
        cfg = config.load(EXAMPLE)
        assert cfg["general"]["window_title"] == "Roblox"

    def test_example_keeps_safety_on(self):
        """The shipped example must not ship with guards disabled."""
        cfg = config.load(EXAMPLE)
        assert cfg["general"]["require_focus"] is True
        assert cfg["safety"]["settle_frames"] >= 1
        assert cfg["safety"]["detect_occlusion"] is True

    def test_example_attack_is_mouse_not_space(self):
        """M1 is unbound in Roblox; Space is jump."""
        cfg = config.load(EXAMPLE)
        assert cfg["combat"]["attack_key"].lower() in (
            "mouse1", "lmb", "left", "m1")
        assert cfg["combat"]["attack_key"].lower() != "space"

    def test_example_tap_exceeds_one_frame(self):
        cfg = config.load(EXAMPLE)
        assert cfg["input"]["tap_ms"] > 4.2


class TestRungParsing:
    def test_every_rung_name_parses(self):
        for r in Rung:
            assert Rung[r.name].name == r.name

    @pytest.mark.parametrize("spelling", ["respawn", "RESPAWN", "Respawn",
                                          " server_hop ", "serverhop"])
    def test_case_and_separator_insensitive(self, spelling):
        """Config files are hand-edited in lowercase."""
        assert isinstance(Rung.parse(spelling), Rung)

    def test_config_value_is_a_valid_rung(self):
        cfg = config.load(EXAMPLE)
        name = cfg["recovery"]["max_auto_rung"]
        assert Rung.parse(name) is Rung.RESPAWN

    def test_unknown_rung_name_is_rejected(self):
        """A typo must not silently disable escalation."""
        with pytest.raises(KeyError):
            Rung.parse("hop_server")

    def test_error_message_lists_valid_options(self):
        with pytest.raises(KeyError) as exc:
            Rung.parse("nonsense")
        assert "serverhop" in str(exc.value).lower()


class TestSaveRoundtrip:
    def test_roundtrip(self, tmp_path):
        cfg = config.load(EXAMPLE)
        p = tmp_path / "out.toml"
        config.save(p, cfg)
        back = config.load(p)
        assert back["combat"]["combo_length"] == cfg["combat"]["combo_length"]
        assert back["safety"] == cfg["safety"]