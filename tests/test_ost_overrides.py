"""Tests for the user-final-say OST override mechanism.

Some drives have local names the algorithm can't preserve (small places that
Nominatim classifies weirdly or doesn't index at all). The override file
lets the user pin exact text for specific episodes without fighting the
derivation.

File format: <output_dir>/ost_overrides.json
    {
      "episode_0": "Frogmore to Port Royal via Overland Trail",
      "episode_3": "Port Royal to Gardens Corner via Trask Parkway",
      "drive": "Lowcountry to Blue Ridge"
    }

Applied AFTER derivation and AFTER validation, but BEFORE duplicate
differentiation — an override is the user's final say.
"""

import json
from pathlib import Path


class TestLoadOverrides:
    def test_missing_file_returns_empty(self, tmp_path):
        from gomoveshift.pipeline.ost_overrides import load_overrides

        assert load_overrides(tmp_path) == {}

    def test_empty_file_returns_empty(self, tmp_path):
        from gomoveshift.pipeline.ost_overrides import load_overrides

        (tmp_path / "ost_overrides.json").write_text("{}")
        assert load_overrides(tmp_path) == {}

    def test_valid_overrides_loaded(self, tmp_path):
        from gomoveshift.pipeline.ost_overrides import load_overrides

        (tmp_path / "ost_overrides.json").write_text(
            json.dumps(
                {
                    "episode_0": "Custom start OST",
                    "episode_3": "Custom ep 3 OST",
                    "drive": "Custom drive OST",
                }
            )
        )
        result = load_overrides(tmp_path)
        assert result == {
            "episode_0": "Custom start OST",
            "episode_3": "Custom ep 3 OST",
            "drive": "Custom drive OST",
        }

    def test_malformed_json_returns_empty_with_warning(self, tmp_path, caplog):
        from gomoveshift.pipeline.ost_overrides import load_overrides

        (tmp_path / "ost_overrides.json").write_text("{not valid json")
        result = load_overrides(tmp_path)
        assert result == {}
        # Should log a warning so the operator knows their file was ignored
        assert any("override" in r.message.lower() for r in caplog.records)

    def test_non_dict_top_level_returns_empty(self, tmp_path):
        from gomoveshift.pipeline.ost_overrides import load_overrides

        (tmp_path / "ost_overrides.json").write_text(json.dumps(["not", "a", "dict"]))
        assert load_overrides(tmp_path) == {}

    def test_str_path_accepted(self, tmp_path):
        """load_overrides accepts str in addition to Path."""
        from gomoveshift.pipeline.ost_overrides import load_overrides

        (tmp_path / "ost_overrides.json").write_text(json.dumps({"drive": "X"}))
        assert load_overrides(str(tmp_path)) == {"drive": "X"}


class TestApplyOverrides:
    def test_empty_overrides_noop(self):
        from gomoveshift.pipeline.ost_overrides import apply_overrides

        episodes = ["A to B", "C to D"]
        drive = "A to D"
        new_episodes, new_drive = apply_overrides(episodes, drive, {})
        assert new_episodes == episodes
        assert new_drive == drive

    def test_single_episode_override(self):
        from gomoveshift.pipeline.ost_overrides import apply_overrides

        episodes = ["Auto A", "Auto B", "Auto C"]
        drive = "Auto drive"
        new_episodes, new_drive = apply_overrides(episodes, drive, {"episode_1": "Hand-written B"})
        assert new_episodes == ["Auto A", "Hand-written B", "Auto C"]
        assert new_drive == "Auto drive"  # untouched

    def test_drive_override(self):
        from gomoveshift.pipeline.ost_overrides import apply_overrides

        episodes = ["Auto A", "Auto B"]
        drive = "Auto drive"
        new_episodes, new_drive = apply_overrides(episodes, drive, {"drive": "Hand drive"})
        assert new_episodes == ["Auto A", "Auto B"]  # untouched
        assert new_drive == "Hand drive"

    def test_both_overrides(self):
        from gomoveshift.pipeline.ost_overrides import apply_overrides

        episodes = ["X", "Y", "Z"]
        drive = "auto"
        new_episodes, new_drive = apply_overrides(
            episodes,
            drive,
            {"episode_0": "A", "episode_2": "C", "drive": "hand"},
        )
        assert new_episodes == ["A", "Y", "C"]
        assert new_drive == "hand"

    def test_episode_index_out_of_range_ignored(self, caplog):
        from gomoveshift.pipeline.ost_overrides import apply_overrides

        episodes = ["A", "B"]
        new_episodes, _ = apply_overrides(episodes, "d", {"episode_5": "never"})
        assert new_episodes == ["A", "B"]
        # Should log a warning about the out-of-range override
        assert any(
            "episode_5" in r.message or "out of range" in r.message.lower() for r in caplog.records
        )

    def test_unknown_key_ignored(self, caplog):
        from gomoveshift.pipeline.ost_overrides import apply_overrides

        new_episodes, new_drive = apply_overrides(
            ["A"], "d", {"unknown_key": "foo", "episode_0": "real"}
        )
        assert new_episodes == ["real"]
        assert new_drive == "d"
        # Unknown key logged
        assert any("unknown_key" in r.message for r in caplog.records)

    def test_non_string_value_ignored(self, caplog):
        from gomoveshift.pipeline.ost_overrides import apply_overrides

        new_episodes, _ = apply_overrides(
            ["A", "B"],
            "d",
            {"episode_0": 123, "episode_1": "valid"},
        )
        assert new_episodes == ["A", "valid"]  # numeric ignored, string applied

    def test_invalid_value_revalidated_and_ignored(self, caplog):
        """A typo'd/placeholder override must
        not burn verbatim — validate_ost rejects it and the derived value
        keeps flowing."""
        from gomoveshift.pipeline.ost_overrides import apply_overrides

        new_episodes, new_drive = apply_overrides(
            ["A", "B"],
            "Derived Drive",
            {"episode_0": "TBD", "episode_1": "None", "drive": "12345"},
        )
        assert new_episodes == ["A", "B"]
        assert new_drive == "Derived Drive"
        assert sum("fails validation" in r.message for r in caplog.records) == 3
