"""Registry-generated CLI↔state authority tests.

Parametrized straight from ``flag_registry.FLAG_REGISTRY`` and exercised through
BOTH layers where wiring bugs live — ``_build_initial_state`` (apply) and
``_try_resume`` (merge) — not ``_try_resume`` alone. Three directions per
sentinel bool flag:

  (a) CLI-set beats saved          — explicit flag wins over the checkpoint.
  (b) saved survives CLI-absent    — no flag → checkpoint value preserved.
  (c) CLI-opposite clears saved    — explicit False clears a saved True.

The former KNOWN-VIOLATION rows are closed: ``force_no_rebase`` is a sentinel
(all 3 directions below), ``no_youtube_upload``/``skip_csv`` are ALWAYS_CLI
(every invocation re-decides; covered by TestAlwaysCliUploadFlags).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from gomoveshift.pipeline.flag_registry import FLAG_REGISTRY, Apply, Semantic


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


# Registry rows applied via apply_cli_to_state AND merged via merge_on_resume —
# the population the 3-direction contract covers. per_lf_overrides is excluded
# by merge_capture=False (resume handled by preserve_per_lf_overrides_on_resume).
_MERGED_RULES = [
    r
    for r in FLAG_REGISTRY
    if r.apply is not Apply.CONSTRUCTOR
    and r.merge_capture
    and r.semantic is Semantic.SENTINEL_CLI_WINS
]
_BOOL_RULES = [r for r in _MERGED_RULES if r.coerce is bool]
_VALUE_SAMPLES = {  # non-bool sentinel rows: (cli_value, saved_value)
    "start_date": ("2026-01-05", "2026-02-06"),
    "lf_publish_date": ("2026-01-05", "2026-02-06"),
    "shorts_per_day": (3, 4),
    "superlong_publish_day": ("sun", "wed"),
    "lf_mode": ("full_drive", "best_moments"),
    "gps_captions": ("events", "off"),
    "map_inset": ("route", "off"),
    "map_inset_bundle": ("/data/a", "/data/b"),
    "map_inset_trim_start_m": (100.0, 200.0),
    "map_inset_trim_end_m": (100.0, 200.0),
    "map_inset_privacy_zones": (["1,2,3"], ["4,5,6"]),
}
_VALUE_RULES = [r for r in _MERGED_RULES if r.state_field in _VALUE_SAMPLES]
assert len(_BOOL_RULES) + len(_VALUE_RULES) == len(_MERGED_RULES)

_ids = lambda rules: [r.state_field for r in rules]  # noqa: E731


class TestSentinelThreeDirections:
    @pytest.mark.parametrize("rule", _BOOL_RULES, ids=_ids(_BOOL_RULES))
    def test_a_cli_set_beats_saved(self, rule, tmp_path):
        orch = _make_orch()
        state = _fresh_state(orch, tmp_path, **{rule.state_field: True})
        _resume(orch, state, tmp_path, {rule.state_field: False})
        assert state.get(rule.state_field) is True, (
            f"{rule.state_field}: explicit CLI True was stomped by the saved False on resume"
        )

    @pytest.mark.parametrize("rule", _BOOL_RULES, ids=_ids(_BOOL_RULES))
    def test_b_saved_survives_cli_absent(self, rule, tmp_path):
        orch = _make_orch()
        state = _fresh_state(orch, tmp_path)  # flag NOT passed this invocation
        _resume(orch, state, tmp_path, {rule.state_field: True})
        assert state.get(rule.state_field) is True, (
            f"{rule.state_field}: saved True did not survive a CLI-absent resume — "
            f"a concrete (non-None) default re-stomped it"
        )

    @pytest.mark.parametrize("rule", _BOOL_RULES, ids=_ids(_BOOL_RULES))
    def test_c_cli_opposite_clears_saved(self, rule, tmp_path):
        """The historically missing third direction: explicit False clears a
        saved True (the historically one-directional bug class)."""
        orch = _make_orch()
        state = _fresh_state(orch, tmp_path, **{rule.state_field: False})
        _resume(orch, state, tmp_path, {rule.state_field: True})
        assert state.get(rule.state_field) is not True, (
            f"{rule.state_field}: explicit CLI False could not clear a saved True "
            f"— one-directional wiring"
        )

    @pytest.mark.parametrize("rule", _VALUE_RULES, ids=_ids(_VALUE_RULES))
    def test_a_cli_value_beats_saved(self, rule, tmp_path):
        cli_val, saved_val = _VALUE_SAMPLES[rule.state_field]
        orch = _make_orch()
        state = _fresh_state(orch, tmp_path, **{rule.state_field: cli_val})
        _resume(orch, state, tmp_path, {rule.state_field: saved_val})
        assert state.get(rule.state_field) == cli_val

    @pytest.mark.parametrize("rule", _VALUE_RULES, ids=_ids(_VALUE_RULES))
    def test_b_saved_value_survives_cli_absent(self, rule, tmp_path):
        _cli_val, saved_val = _VALUE_SAMPLES[rule.state_field]
        orch = _make_orch()
        state = _fresh_state(orch, tmp_path)
        _resume(orch, state, tmp_path, {rule.state_field: saved_val})
        assert state.get(rule.state_field) == saved_val


class TestAlwaysCliUploadFlags:
    """skip_upload family + youtube_upload + no_youtube_upload and
    skip_csv: every invocation re-decides — even the coerced default beats a
    saved value (intended: an audit run's saved skip must not bleed into the
    next publish invocation, and a saved --no-youtube-upload is cleared by
    simply omitting the flag)."""

    @pytest.mark.parametrize(
        "field", ["skip_upload", "youtube_upload", "no_youtube_upload", "skip_csv"]
    )
    def test_current_invocation_default_beats_saved_true(self, field, tmp_path):
        orch = _make_orch()
        state = _fresh_state(orch, tmp_path)  # flag absent → coerced False
        _resume(orch, state, tmp_path, {field: True})
        assert state.get(field) is False

    @pytest.mark.parametrize("field", ["no_youtube_upload", "skip_csv"])
    def test_explicit_true_survives_resume(self, field, tmp_path):
        """The fix direction: the flag passed THIS invocation is not
        silently discarded by the checkpoint merge."""
        orch = _make_orch()
        state = _fresh_state(orch, tmp_path, **{field: True})
        _resume(orch, state, tmp_path, {field: False})
        assert state.get(field) is True


class TestLocationEmptyStringSentinel:
    """Parser default '' means "not supplied" — a location-less resume
    must preserve the saved location instead of stomping it with ''."""

    def test_saved_location_survives_location_less_resume(self, tmp_path):
        orch = _make_orch()
        state = _fresh_state(orch, tmp_path)  # location='' (parser default)
        _resume(orch, state, tmp_path, {"location": "County Kerry, Ireland"})
        assert state.get("location") == "County Kerry, Ireland"

    def test_cli_location_beats_saved(self, tmp_path):
        orch = _make_orch()
        state = _fresh_state(orch, tmp_path, location="Dingle, Kerry, Ireland")
        _resume(orch, state, tmp_path, {"location": "County Cork, Ireland"})
        assert state.get("location") == "Dingle, Kerry, Ireland"


class TestRegistryIntegrity:
    def test_no_known_violations_remain(self):
        """The one-directional rows are closed; the marker must stay empty
        (a new violation row needs a fix, not an honest encoding)."""
        assert not {r.state_field for r in FLAG_REGISTRY if r.known_violation}

    def test_registry_completeness_against_pipeline_state(self):
        """Every persisted bool/Optional[bool] PipelineState field that has a
        CLI dest on the drive subparser must appear in the registry — catches
        the next skip_csv (a new flag wired to state but never merged)."""
        import argparse

        from gomoveshift.pipeline.drive_state_model import PipelineState

        with patch("importlib.metadata.version", return_value="0"):
            from gomoveshift.cli.parser import _build_parser

            parser = _build_parser()
        sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
        drive_dests = {a.dest for a in sub.choices["drive"]._actions}
        registered = {r.state_field for r in FLAG_REGISTRY if r.state_field}
        missing = [
            f.name
            for f in dataclasses.fields(PipelineState)
            if str(f.type) in ("bool", "Optional[bool]")
            and f.name in drive_dests
            and f.name not in registered
        ]
        assert not missing, (
            f"PipelineState bool flag(s) with a drive CLI dest missing from "
            f"FLAG_REGISTRY: {missing} — add a row in "
            f"gomoveshift/pipeline/flag_registry.py declaring the resume semantic."
        )
