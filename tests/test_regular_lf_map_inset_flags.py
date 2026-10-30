"""Flag, config and resume plumbing for the opt-in ``--map-inset`` options.

The pipeline behaviour is covered in ``test_regular_lf_map_inset.py``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


# -- config, flag registry, CLI --------------------------------------------------------


def test_config_defaults_and_validation():
    from gomoveshift.config import DriveConfig

    d = DriveConfig()
    assert d.map_inset == "off" and d.map_inset_bundle is None
    assert d.map_inset_trim_start_m == 0.0 and d.map_inset_trim_end_m == 0.0
    assert d.map_inset_privacy_zones == []
    with pytest.raises(ValueError, match="map_inset"):
        DriveConfig(map_inset="satellite")
    with pytest.raises(ValueError, match="trim"):
        DriveConfig(map_inset_trim_start_m=-1.0)
    with pytest.raises(ValueError, match="privacy"):
        DriveConfig(map_inset_privacy_zones=["95,0,10"])
    assert DriveConfig(map_inset_privacy_zones=["35.2,-83.6,200"]).map_inset_privacy_zones == [
        "35.2,-83.6,200"
    ]


def _flag(parser, dest):
    sub = next(
        a for a in parser._actions if hasattr(a, "choices") and a.choices and "drive" in a.choices
    )
    return next(a for a in sub.choices["drive"]._actions if a.dest == dest)


def test_cli_flags_parse_with_sentinel_defaults():
    with patch("importlib.metadata.version", return_value="0"):
        from gomoveshift.cli.parser import _build_parser

        parser = _build_parser()
    mode = _flag(parser, "map_inset")
    assert mode.choices == ["off", "route", "roads"] and mode.default is None
    for dest in ("map_inset_bundle", "map_inset_trim_start_m", "map_inset_trim_end_m"):
        assert _flag(parser, dest).default is None
    zone = _flag(parser, "map_inset_privacy_zones")
    assert zone.default is None and zone.option_strings == ["--map-inset-privacy-zone"]
    assert zone.type("35.2,-83.6,200") == "35.2,-83.6,200"
    import argparse

    for bad in ("1,2", "a,b,c", "95,0,10", "1,2,0"):
        with pytest.raises(argparse.ArgumentTypeError):
            zone.type(bad)


MAP_FIELDS = {
    "map_inset": ("route", "off"),
    "map_inset_bundle": ("/data/a", "/data/b"),
    "map_inset_trim_start_m": (100.0, 200.0),
    "map_inset_trim_end_m": (100.0, 200.0),
    "map_inset_privacy_zones": (["1,2,3"], ["4,5,6"]),
}


def test_every_option_is_a_registered_sentinel_with_a_config_mirror():
    from gomoveshift.pipeline.flag_registry import FLAG_REGISTRY, Semantic

    rules = {r.state_field: r for r in FLAG_REGISTRY}
    for field in MAP_FIELDS:
        assert rules[field].semantic is Semantic.SENTINEL_CLI_WINS
        assert rules[field].config_mirror == field


def test_resume_without_the_flags_keeps_the_saved_options():
    from gomoveshift.config import Config
    from gomoveshift.pipeline.drive_state_model import PipelineState
    from gomoveshift.pipeline.flag_registry import apply_cli_to_state, merge_on_resume

    first = Config()
    state = PipelineState()
    apply_cli_to_state(state, {k: v[0] for k, v in MAP_FIELDS.items()}, first)
    saved = state.to_dict()
    assert first.drive.map_inset == "route"
    cfg = Config()
    resumed = PipelineState()
    apply_cli_to_state(resumed, {k: None for k in MAP_FIELDS}, cfg)
    merge_on_resume(resumed, saved, cfg)
    for field, (value, _other) in MAP_FIELDS.items():
        assert getattr(resumed, field) == value
        assert getattr(cfg.drive, field) == value


def test_a_differing_flag_beats_the_saved_choice():
    from gomoveshift.config import Config
    from gomoveshift.pipeline.drive_state_model import PipelineState
    from gomoveshift.pipeline.flag_registry import apply_cli_to_state, merge_on_resume

    saved_state = PipelineState()
    apply_cli_to_state(saved_state, {k: v[0] for k, v in MAP_FIELDS.items()}, Config())
    cfg = Config()
    state = PipelineState()
    apply_cli_to_state(state, {"map_inset": "off"}, cfg)
    merge_on_resume(state, saved_state.to_dict(), cfg)
    assert state.map_inset == "off" and cfg.drive.map_inset == "off"
