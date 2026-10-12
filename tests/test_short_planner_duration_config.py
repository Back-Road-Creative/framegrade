"""Configurable short duration (min/max/target).

``short_planner.plan_shorts`` already accepted ``min/max/target_duration_sec``
as kwargs (with module-level ``DEFAULT_*`` fallbacks), but nothing let an
operator or a config file override them: no config field, no CLI flag, no
threading from ``config.drive`` into the ``plan_shorts`` call in
``day_pipeline.py``. These tests cover the new config fields, their
validation, and the new CLI flags.
"""

from __future__ import annotations

import pytest

from gomoveshift.config import DriveConfig
from gomoveshift.video_types import SHORTS_MAX_DURATION


class TestDriveConfigShortsDurationDefaults:
    def test_defaults_match_short_planner_constants(self):
        from gomoveshift.pipeline.short_planner import (
            DEFAULT_MAX_DURATION_SEC,
            DEFAULT_MIN_DURATION_SEC,
            DEFAULT_TARGET_DURATION_SEC,
        )

        cfg = DriveConfig()
        assert cfg.shorts_min_duration_sec == DEFAULT_MIN_DURATION_SEC
        assert cfg.shorts_max_duration_sec == DEFAULT_MAX_DURATION_SEC
        assert cfg.shorts_target_duration_sec == DEFAULT_TARGET_DURATION_SEC

    def test_custom_values_accepted(self):
        cfg = DriveConfig(
            shorts_min_duration_sec=20.0,
            shorts_max_duration_sec=50.0,
            shorts_target_duration_sec=35.0,
        )
        assert cfg.shorts_min_duration_sec == 20.0
        assert cfg.shorts_max_duration_sec == 50.0
        assert cfg.shorts_target_duration_sec == 35.0


class TestDriveConfigShortsDurationValidation:
    def test_max_over_hard_ceiling_raises(self):
        with pytest.raises(ValueError, match="60"):
            DriveConfig(shorts_max_duration_sec=SHORTS_MAX_DURATION + 1)

    def test_target_below_min_raises(self):
        with pytest.raises(ValueError, match="shorts_min_duration_sec"):
            DriveConfig(
                shorts_min_duration_sec=30.0,
                shorts_target_duration_sec=20.0,
                shorts_max_duration_sec=45.0,
            )

    def test_target_above_max_raises(self):
        with pytest.raises(ValueError, match="shorts_max_duration_sec"):
            DriveConfig(
                shorts_min_duration_sec=30.0,
                shorts_target_duration_sec=50.0,
                shorts_max_duration_sec=45.0,
            )


class TestDriveParserShortsDurationFlags:
    def _parse(self, *argv, tmp_path):
        from gomoveshift.cli.parser import _build_parser

        return _build_parser().parse_args(["drive", str(tmp_path), *argv])

    def test_flags_default_none(self, tmp_path):
        args = self._parse(tmp_path=tmp_path)
        assert args.shorts_min_duration is None
        assert args.shorts_max_duration is None
        assert args.shorts_target_duration is None

    def test_flags_parse_floats(self, tmp_path):
        args = self._parse(
            "--shorts-min-duration",
            "20",
            "--shorts-max-duration",
            "50",
            "--shorts-target-duration",
            "35",
            tmp_path=tmp_path,
        )
        assert args.shorts_min_duration == 20.0
        assert args.shorts_max_duration == 50.0
        assert args.shorts_target_duration == 35.0
