"""``--superlong-day``: per-run override for the superlong part publish
weekday, exercised end to end (parser -> orchestrator wiring -> resume) so a
live flag can't be mistaken for dead scaffolding.

Mirrors the existing ``--superlong`` / ``superlong_enabled`` sentinel wiring
(flag_registry.py, drive_orchestrator._build_initial_state / _try_resume).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


def _make_orch():
    from gomoveshift.config import Config
    from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator

    return DrivePipelineOrchestrator(config=Config())


def _fresh_state(orch, tmp_path: Path, **overrides):
    """Build a fresh state through the REAL apply layer (_build_initial_state)."""
    kwargs = dict(
        source_dir=Path("/tmp/src"),
        project_dir=tmp_path,
        stages_base=tmp_path,
        region="ireland",
        publish_day="wed",
        location="",
        fresh=False,
        skip_upload=None,
        skip_drive_upload=None,
        skip_youtube_upload=None,
        youtube_upload=None,
        skip_csv=False,
        start_date=None,
        lf_publish_date=None,
        force_no_rebase=None,
        shorts_per_day=None,
        shorts_target=None,
        shorts_gap=None,
        keep_intermediates=None,
        keep_masters=None,
        allow_short_count=None,
        force_sidecar_reset=None,
        allow_seo_degraded=None,
        per_lf_overrides=None,
        ost_audit_confirmed=None,
        gate_cd_confirmed=None,
        no_youtube_upload=None,
    )
    kwargs.update(overrides)
    state, _explicit, _skip, _yt = orch._build_initial_state(**kwargs)
    return state


def _resume(orch, state, tmp_path: Path, saved: dict):
    state_file = tmp_path / "pipeline_state.json"
    state_file.write_text(json.dumps({"phases_completed": ["phase_b_preview"], **saved}))
    orch._try_resume(state, state_file, tmp_path, [], fresh=False)
    return state


def _parse_drive_args(argv):
    with patch("importlib.metadata.version", return_value="0"):
        from gomoveshift.cli.parser import _build_parser

        parser = _build_parser()
    return parser.parse_args(argv)


class TestSuperlongDayFlag:
    def test_flag_parses_to_superlong_publish_day_dest(self):
        args = _parse_drive_args(
            ["drive", "/tmp/src", "--output-dir", "/tmp/out", "--superlong-day", "sun"]
        )
        assert args.superlong_publish_day == "sun"

    def test_flag_defaults_to_none(self):
        args = _parse_drive_args(["drive", "/tmp/src", "--output-dir", "/tmp/out"])
        assert args.superlong_publish_day is None

    def test_flag_rejects_invalid_day(self):
        with pytest.raises(SystemExit):
            _parse_drive_args(
                ["drive", "/tmp/src", "--output-dir", "/tmp/out", "--superlong-day", "someday"]
            )

    def test_flows_through_build_initial_state_and_config_mirror(self, tmp_path):
        orch = _make_orch()
        state = _fresh_state(orch, tmp_path, superlong_publish_day="sun")
        assert state.get("superlong_publish_day") == "sun"
        assert orch.config.drive.superlong_publish_day == "sun"

    def test_resume_without_flag_preserves_saved_value(self, tmp_path):
        orch = _make_orch()
        state = _fresh_state(orch, tmp_path)  # flag NOT passed this invocation
        _resume(orch, state, tmp_path, {"superlong_publish_day": "wed"})
        assert state.get("superlong_publish_day") == "wed", (
            "saved superlong_publish_day did not survive a CLI-absent resume — "
            "a concrete default re-stomped it"
        )

    def test_explicit_cli_value_beats_saved_on_resume(self, tmp_path):
        orch = _make_orch()
        state = _fresh_state(orch, tmp_path, superlong_publish_day="sun")
        _resume(orch, state, tmp_path, {"superlong_publish_day": "wed"})
        assert state.get("superlong_publish_day") == "sun"
