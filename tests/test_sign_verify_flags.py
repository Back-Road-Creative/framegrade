"""CLI + config wiring for road-sign GPS verification (--sign-verify).

Pins the three layers the feature's operator intent travels through:

  1. the drive subparser (argparse-sentinel ``None`` defaults, and the
     paired ``--no-sign-verify`` off switch on the SAME dest),
  2. the declarative ``FLAG_REGISTRY`` rows (SENTINEL_CLI_WINS + the
     ``config.drive`` mirror the pipeline actually reads),
  3. ``DriveConfig``'s defaults — the feature is OFF and non-strict until
     conflict precision has been measured on real drives.

The resume direction is the load-bearing one: a run that omits the flag
must preserve a saved ``True`` (a ``False`` argparse default would
re-stomp it), while an explicit ``--no-sign-verify`` must still win.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline.drive_state_model import PipelineState
from gomoveshift.pipeline.flag_registry import (
    FLAG_REGISTRY,
    Semantic,
    apply_cli_to_state,
    merge_on_resume,
)

_FIELDS = ("sign_verify", "sign_verify_strict")


def _parse(tmp_path: Path, *extra: str):
    from gomoveshift.cli import _build_parser

    src = tmp_path / "source"
    src.mkdir(exist_ok=True)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    parser = _build_parser()
    return parser.parse_args(
        ["drive", str(src), "--region", "usa", "--output-dir", str(out), *extra]
    )


def _rule(field: str):
    return next(r for r in FLAG_REGISTRY if r.state_field == field)


class TestParser:
    @pytest.mark.parametrize("field", _FIELDS)
    def test_absent_parses_to_none_sentinel(self, field, tmp_path):
        """``None`` (not ``False``) is load-bearing: it is what lets a resume
        that omits the flag preserve the operator's saved intent."""
        args = _parse(tmp_path)
        assert getattr(args, field) is None

    def test_sign_verify_flag_parses_true(self, tmp_path):
        assert _parse(tmp_path, "--sign-verify").sign_verify is True

    def test_sign_verify_strict_flag_parses_true(self, tmp_path):
        assert _parse(tmp_path, "--sign-verify-strict").sign_verify_strict is True

    def test_no_sign_verify_parses_false(self, tmp_path):
        """Explicit off must be distinguishable from absent — same dest."""
        assert _parse(tmp_path, "--no-sign-verify").sign_verify is False

    def test_last_flag_wins_on_the_shared_dest(self, tmp_path):
        assert _parse(tmp_path, "--sign-verify", "--no-sign-verify").sign_verify is False


class TestRegistryRows:
    @pytest.mark.parametrize("field", _FIELDS)
    def test_row_exists_with_sentinel_semantic(self, field):
        rule = _rule(field)
        assert rule.cli_dest == field
        assert rule.semantic is Semantic.SENTINEL_CLI_WINS
        assert rule.coerce is bool

    @pytest.mark.parametrize("field", _FIELDS)
    def test_row_mirrors_onto_config_drive(self, field):
        """The pipeline reads ``config.drive``, not state — without the
        mirror the flag would land on state and change nothing."""
        assert _rule(field).config_mirror == field


class TestApplyCliToState:
    @pytest.mark.parametrize("field", _FIELDS)
    def test_writes_state_and_config_mirror(self, field):
        state = PipelineState()
        config = Config()
        apply_cli_to_state(state, {field: True}, config)
        assert getattr(state, field) is True
        assert getattr(config.drive, field) is True

    @pytest.mark.parametrize("field", _FIELDS)
    def test_absent_key_leaves_state_untouched(self, field):
        state = PipelineState()
        config = Config()
        apply_cli_to_state(state, {}, config)
        assert getattr(state, field) is None
        assert getattr(config.drive, field) is False


class TestResumeSentinel:
    @pytest.mark.parametrize("field", _FIELDS)
    def test_saved_true_survives_a_flagless_resume(self, field, tmp_path):
        state = PipelineState()
        config = Config()
        apply_cli_to_state(state, {field: None}, config)  # flag absent this run
        merge_on_resume(state, json.loads(json.dumps({field: True})), config)
        assert getattr(state, field) is True, (
            f"{field}: saved True did not survive a CLI-absent resume — "
            f"a concrete (non-None) default re-stomped it"
        )
        assert getattr(config.drive, field) is True

    @pytest.mark.parametrize("field", _FIELDS)
    def test_explicit_off_clears_a_saved_true(self, field, tmp_path):
        state = PipelineState()
        config = Config()
        apply_cli_to_state(state, {field: False}, config)  # --no-sign-verify
        merge_on_resume(state, json.loads(json.dumps({field: True})), config)
        assert getattr(state, field) is False, (
            f"{field}: explicit CLI False could not clear a saved True — one-directional wiring"
        )


class TestDriveConfigDefaults:
    def test_feature_is_opt_in(self):
        drive = Config().drive
        assert drive.sign_verify is False
        assert drive.sign_verify_strict is False

    def test_cost_governors(self):
        drive = Config().drive
        assert drive.sign_verify_max_samples_per_day == 120
        assert drive.sign_verify_min_speed_mps == 5.0
        assert drive.sign_verify_min_corroborating_hits == 2

    def test_no_parallel_confidence_knob(self):
        """Detection/OCR confidence already lives on SignDetectionConfig /
        OCRConfig — a second copy on DriveConfig would drift."""
        drive_fields = set(type(Config().drive).model_fields)
        assert not {f for f in drive_fields if f.startswith("sign_verify") and "conf" in f}
