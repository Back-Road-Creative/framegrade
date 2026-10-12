"""Regression tests for ``config.drive.<field>`` threading bugs.

Guards against the silent-drop class of bug (see
``test_day_pipeline_brand_logo_path.py`` for the canonical example): an
orchestrator/phase module reads a field from ``config.drive``, passes it
as a kwarg to a downstream callee, and the downstream silently takes a
fallback path when the kwarg arrives wrong. Zero warnings, zero test
failures, shipped output missing the feature.

Each test here builds a minimal config with a sentinel value, stubs
every heavyweight downstream (ffmpeg, disk, network), invokes the
public entry point, and asserts the downstream callee received the
sentinel verbatim.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from _state_helpers import make_state


def _make_config(**drive_overrides):
    """Return a minimal config exposing config.drive.<field>.

    The day_pipeline touches ``brand_logo_path``,
    ``shorts_publish_times``, and ``shorts_min_gap_sec`` by name — keep
    stubs for all of them so an override for one field doesn't crash
    unrelated threading.
    """
    drive_defaults = {
        "brand_logo_path": "/tmp/fake_logo.png",
        "shorts_publish_times": ["15:00", "18:00"],
        "shorts_min_gap_sec": 60.0,
        "skip_audio": False,
    }
    drive_defaults.update(drive_overrides)
    drive = SimpleNamespace(**drive_defaults)
    return SimpleNamespace(drive=drive)


def _minimal_day_inputs(tmp_path):
    """Shared setup for run_day_pipeline tests: one LF plan, one episode."""
    lf_plan = SimpleNamespace(
        index=1,
        segments=[SimpleNamespace(source_file_index=0, start=0.0, end=60.0)],
        montage_peaks=[SimpleNamespace(source_file_index=0, start=10.0, end=12.0)],
    )
    day_batch = SimpleNamespace(
        date="2026-02-20",
        episodes=[
            SimpleNamespace(
                index=1,
                concat_path=tmp_path / "ep.mp4",
                total_duration=300.0,
            )
        ],
    )
    (tmp_path / "ep.mp4").write_bytes(b"\x00")
    (tmp_path / "master.mp4").write_bytes(b"\x00")
    (tmp_path / "seg.mp4").write_bytes(b"\x00")
    (tmp_path / "wm.mp4").write_bytes(b"\x00")
    return day_batch, lf_plan


class TestShortsPlanThreading:
    """run_day_pipeline must thread the shorts target (Waypost's decided
    count, via ``volume_source``) and ``shorts_min_gap_sec`` from config.drive into plan_shorts as
    target_count / min_gap_sec kwargs.

    Silent-drop risk: if day_pipeline reads the wrong attribute,
    plan_shorts falls back to its built-in default of 9 — and you'd ship
    the wrong number of shorts with no warning.
    """

    def _invoke_day_pipeline(self, tmp_path, cfg, shorts_per_day=None, decided_per_lf=7):
        """Call run_day_pipeline with everything stubbed; return the
        captured plan_shorts kwargs. ``shorts_per_day`` lands on state to
        exercise the ``--shorts-per-day N`` requirement cap."""
        from gomoveshift.pipeline import day_pipeline, volume_source

        day_batch, lf_plan = _minimal_day_inputs(tmp_path)

        captured: dict = {}

        def fake_plan_shorts(scores, **kwargs):
            captured.update(kwargs)
            return []

        def fake_decision(db=None):
            return volume_source.ShortsDecision(decided_per_lf, "policy", "TEST")

        with (
            # The per-LF count is Waypost's decision, never config -- pin it.
            patch.object(day_pipeline.volume_source, "shorts_per_lf", side_effect=fake_decision),
            # This class asserts config/derived-target threading, not brand-policy
            # capping (covered by TestShortsTargetBrandPolicyCap) -- pin a brand
            # with no policy block so the default gomoveshift per_location_cap=5
            # doesn't clip these explicit/derived target assertions.
            patch.dict(os.environ, {"GMS_CONSOLE_BRAND": "examplebrand"}),
            patch.object(day_pipeline, "encode_day_master", return_value=tmp_path / "master.mp4"),
            patch.object(
                day_pipeline, "extract_and_concat_segments", return_value=tmp_path / "seg.mp4"
            ),
            patch.object(day_pipeline, "prepend_montage", return_value=tmp_path / "wm.mp4"),
            patch.object(day_pipeline, "apply_ost", side_effect=lambda **kw: kw["output_path"]),
            patch.object(day_pipeline, "plan_shorts", side_effect=fake_plan_shorts),
            patch.object(day_pipeline, "compute_day_file_boundaries", return_value=[(0.0, 300.0)]),
        ):
            day_pipeline.run_day_pipeline(
                day_batch=day_batch,
                day_scores=[[0.0, 0.5]],  # truthy, non-None
                output_dir=tmp_path / "out",
                location="Longwood",
                config=cfg,
                lf_plans=[lf_plan],
                # plan_shorts is stubbed — escape the SHORTS-CONTRACT gate;
                # these tests assert kwarg threading, not planner yield.
                state=make_state(
                    ost_audit_confirmed=True,
                    allow_short_count=True,
                    shorts_per_day=shorts_per_day,
                ),
            )
        return captured

    def test_decision_threaded_to_plan_shorts_over_config(self, tmp_path):
        """The per-LF count is Waypost's decision -- plan_shorts receives
        the decided value × num_lfs (here 1 LF), whatever config carries."""
        cfg = _make_config(shorts_min_gap_sec=60.0)
        kw = self._invoke_day_pipeline(tmp_path, cfg, decided_per_lf=5)
        assert kw.get("target_count") == 5, (
            f"plan_shorts did not receive the decided 5; "
            f"got target_count={kw.get('target_count')!r}"
        )

    def test_derived_target_not_threaded_to_plan_shorts(self, tmp_path):
        """The publish-plan-derived count (3 windows × 7-day interval = 21)
        is only the slot-capacity check; plan_shorts still gets the decision."""
        cfg = _make_config(shorts_publish_times=["08:00", "12:00", "16:00"])
        kw = self._invoke_day_pipeline(tmp_path, cfg, decided_per_lf=5)
        assert kw.get("target_count") == 5, (
            f"plan_shorts must receive the decided 5, not the derived 21; "
            f"got target_count={kw.get('target_count')!r}"
        )

    def test_shorts_per_day_caps_derived_target(self, tmp_path):
        """``--shorts-per-day N`` truncates the scheduler to N windows, so
        the planner target must derive from the SAME N — else the planner
        demands full-window shorts the scheduler has no slots for and every
        surplus ships slotless. Default 2 windows derives 14, but
        ``--shorts-per-day 1`` caps to 1 window × 7-day interval = 7."""
        cfg = _make_config()  # 2 windows → uncapped 14
        kw = self._invoke_day_pipeline(tmp_path, cfg, shorts_per_day=1)
        assert kw.get("target_count") == 7, (
            f"--shorts-per-day 1 must cap the derived target to 1×7=7 so planner "
            f"demand == scheduler supply; got target_count={kw.get('target_count')!r}"
        )

    def test_shorts_min_gap_sec_threaded_to_plan_shorts(self, tmp_path):
        """config.drive.shorts_min_gap_sec must arrive as plan_shorts(min_gap_sec=X)."""
        cfg = _make_config(shorts_min_gap_sec=45.0)
        kw = self._invoke_day_pipeline(tmp_path, cfg)
        assert kw.get("min_gap_sec") == 45.0, (
            f"plan_shorts did not receive shorts_min_gap_sec=45.0; "
            f"got min_gap_sec={kw.get('min_gap_sec')!r}"
        )

    def test_both_shorts_fields_threaded_together(self, tmp_path):
        """Both fields threaded in a single call — catches a partial-read bug."""
        cfg = _make_config(shorts_min_gap_sec=75.0)
        kw = self._invoke_day_pipeline(tmp_path, cfg, decided_per_lf=4)
        assert kw.get("target_count") == 4
        assert kw.get("min_gap_sec") == 75.0

    def test_shorts_duration_fields_threaded_to_plan_shorts(self, tmp_path):
        """config.drive.shorts_{min,max,target}_duration_sec must arrive as
        plan_shorts({min,max,target}_duration_sec=X)."""
        cfg = _make_config(
            shorts_min_duration_sec=20.0,
            shorts_max_duration_sec=50.0,
            shorts_target_duration_sec=35.0,
        )
        kw = self._invoke_day_pipeline(tmp_path, cfg)
        assert kw.get("min_duration_sec") == 20.0
        assert kw.get("max_duration_sec") == 50.0
        assert kw.get("target_duration_sec") == 35.0

    def test_shorts_duration_fields_default_when_absent(self, tmp_path):
        """A config stub without the new duration fields must not crash —
        plan_shorts falls back to the short_planner module defaults."""
        from gomoveshift.pipeline.short_planner import (
            DEFAULT_MAX_DURATION_SEC,
            DEFAULT_MIN_DURATION_SEC,
            DEFAULT_TARGET_DURATION_SEC,
        )

        cfg = _make_config()
        kw = self._invoke_day_pipeline(tmp_path, cfg)
        assert kw.get("min_duration_sec") == DEFAULT_MIN_DURATION_SEC
        assert kw.get("max_duration_sec") == DEFAULT_MAX_DURATION_SEC
        assert kw.get("target_duration_sec") == DEFAULT_TARGET_DURATION_SEC


class TestSkipAudioThreading:
    """config.drive.skip_audio is threaded in three independent places:

    * PipelineShortsExtractor → render_shorts_batched
    * PipelineShortsExtractor → render_short_ffmpeg (fallback)
    * PhaseD.run() → PipelineValidator.validate

    A broken read at any of those three sites would ship output with the
    wrong audio policy (extra stream, or validator miscounting streams).
    """

    def _build_extractor_with_mocks(self, skip_audio_value):
        """Construct PipelineShortsExtractor with config.drive.skip_audio
        set to sentinel, everything else MagicMock."""
        from gomoveshift.pipeline.shorts_extractor import PipelineShortsExtractor

        extractor = PipelineShortsExtractor()
        extractor.config = MagicMock()
        extractor.config.drive.skip_audio = skip_audio_value
        return extractor

    @pytest.mark.parametrize("skip_audio_value", [True, False])
    def test_skip_audio_threaded_to_batched_renderer(self, skip_audio_value, tmp_path):
        """PipelineShortsExtractor must pass config.drive.skip_audio into
        render_shorts_batched verbatim — catches both True→False (silent
        audio on no-audio runs) and False→True (silent no-audio on normal
        runs)."""
        from gomoveshift.pipeline import shorts_extractor as se_mod

        # One mock short to render
        mock_short = MagicMock()
        mock_short.index = 0
        mock_short.rank = 1
        mock_short.start_time = 0.0
        mock_short.end_time = 35.0
        mock_short.duration = 35.0
        mock_short.avg_motion_energy = 0.05
        mock_short.peak_motion_energy = 0.08

        mock_extraction = MagicMock()
        mock_extraction.shorts = [mock_short]
        mock_extraction.num_shorts_found = 1
        mock_extraction.source_duration = 600.0
        mock_extraction.dead_spots_avoided = 0

        video_path = tmp_path / "source.mp4"
        video_path.write_bytes(b"\x00" * 1024)
        output_dir = tmp_path / "shorts"

        with (
            patch.object(se_mod, "render_shorts_batched", return_value={0: True}) as mock_batch,
            patch.object(se_mod, "_detect_nvenc_cached", return_value=False),
            patch.object(se_mod, "get_video_dimensions", return_value=(3840, 2160)),
            patch("gomoveshift.pipeline.cache.is_analysis_cached", return_value=False),
            patch("gomoveshift.pipeline.cache.is_output_fresh", return_value=False),
            patch(
                "gomoveshift.analysis.shorts_extractor.ShortsExtractor.extract_shorts",
                return_value=mock_extraction,
            ),
        ):
            extractor = self._build_extractor_with_mocks(skip_audio_value)
            extractor.extract_and_render(
                video_path=str(video_path),
                output_dir=str(output_dir),
                location="Test",
                num_shorts=1,
            )

        assert mock_batch.call_count == 1
        call_kwargs = mock_batch.call_args.kwargs
        assert call_kwargs.get("skip_audio") is skip_audio_value, (
            f"render_shorts_batched did not receive skip_audio={skip_audio_value}; "
            f"got {call_kwargs.get('skip_audio')!r}"
        )

    @pytest.mark.parametrize("skip_audio_value", [True, False])
    def test_phase_d_source_reads_skip_audio_from_config_drive(self, skip_audio_value):
        """Source-level check: the validation-gate block must
        read ``self.config.drive.skip_audio`` (not ``config.skip_audio``
        or similar) and pass it to ``validate(skip_audio=...)``.

        We can't easily monkey-patch ``PipelineValidator`` (it's resolved
        dynamically via ``resolve()`` inside PhaseD.run), so we do the
        next best thing: verify the source still reads from the nested
        DriveConfig field AND forwards it to validate().

        This catches the same class of bug as the brand_logo_path drop:
        a rename to ``config.skip_audio`` or ``config.output.skip_audio``
        would silently drop to None/False.
        """
        from pathlib import Path as _Path

        # The validation gate lives in phase_d_gates.py.
        src = (
            _Path(__file__).resolve().parent.parent
            / "gomoveshift"
            / "pipeline"
            / "drive_phases"
            / "phase_d_gates.py"
        ).read_text()

        # The validation gate is a
        # module-level helper that receives ``config`` as a parameter, so
        # the read is now ``config.drive.skip_audio`` (not
        # ``self.config.drive.skip_audio``). Either form is the silent-drop
        # vector this guards against — accept both so future
        # decomposition (back into a method, or down into another helper)
        # doesn't break the contract.
        assert "self.config.drive.skip_audio" in src or "config.drive.skip_audio" in src, (
            "phase_d_gates.py no longer reads skip_audio from config.drive — "
            "this is the silent-drop vector this test guards against. If "
            "this read moved, update the test to point at the new read."
        )
        # And the value is threaded into validate(skip_audio=...)
        assert "skip_audio=skip_audio" in src, (
            "phase_d_gates.py must forward the local skip_audio to "
            "PipelineValidator.validate(skip_audio=...) verbatim"
        )
        # Bind skip_audio_value so the parametrize loop runs twice;
        # the source check is value-independent but running both
        # parametrizations documents intent (test True and False).
        assert skip_audio_value in (True, False)


class TestAdaptiveThresholdDispatch:
    """phase_bc_day_first._build_drive_plan_from_state gates on
    config.drive.adaptive_threshold_enabled to pick between
    LongFormAssembler.build_drive_plan_per_day and .build_drive_plan.

    Silent-drop risk: if the gate used ``getattr(config.drive,
    "adaptive_threshold", False)`` (wrong name) or ``config.adaptive_*``
    (wrong parent), the dispatch would silently pick the non-adaptive
    branch regardless of config. Existing tests verify the LF assembler's
    internal adaptive behavior but NOT the dispatch site itself.
    """

    def _make_minimal_state(self, tmp_path):
        """Build the minimum state phase_bc_day_first._build_drive_plan_from_state needs."""
        ep = tmp_path / "ep.mp4"
        ep.write_bytes(b"\x00")
        import numpy as np

        return make_state(
            drive_scores=np.zeros((10, 9), dtype=np.float64),
            file_boundaries=[(0.0, 300.0)],
            episode_files=[str(ep)],
            episode_dates=["2026-02-20"],
            region="usa",
            gps_per_episode=None,
        )

    @pytest.mark.parametrize("enabled", [True, False])
    def test_dispatch_respects_adaptive_threshold_enabled(self, enabled, tmp_path):
        """When adaptive_threshold_enabled=True AND episode_dates align, dispatch goes
        to build_drive_plan_per_day; when False, goes to build_drive_plan."""
        from gomoveshift.pipeline.drive_phases import phase_bc_day_first as bc_mod

        state = self._make_minimal_state(tmp_path)

        cfg = SimpleNamespace(drive=SimpleNamespace(adaptive_threshold_enabled=enabled))

        per_day = MagicMock(return_value=SimpleNamespace(regular_lfs=[]))
        plain = MagicMock(return_value=SimpleNamespace(regular_lfs=[]))
        fake_assembler = MagicMock()
        fake_assembler.build_drive_plan_per_day = per_day
        fake_assembler.build_drive_plan = plain

        with patch.object(bc_mod, "LongFormAssembler", return_value=fake_assembler):
            bc_mod._build_drive_plan_from_state(cfg, state)

        if enabled:
            assert per_day.call_count == 1, (
                "adaptive_threshold_enabled=True but build_drive_plan_per_day not called"
            )
            assert plain.call_count == 0
        else:
            assert plain.call_count == 1, (
                "adaptive_threshold_enabled=False but build_drive_plan not called"
            )
            assert per_day.call_count == 0

    def test_longform_assembler_receives_drive_config(self, tmp_path):
        """The LongFormAssembler must be constructed with config.drive —
        if someone passed `config` or `config.drive_cfg`, the assembler's
        internal `self.config.min_segment_duration` etc. reads would
        either crash loudly (good) or silently hit wrong attrs (bad)."""
        from gomoveshift.pipeline.drive_phases import phase_bc_day_first as bc_mod

        state = self._make_minimal_state(tmp_path)
        sentinel_drive = SimpleNamespace(adaptive_threshold_enabled=False)
        cfg = SimpleNamespace(drive=sentinel_drive)

        captured: dict = {}

        class FakeAssembler:
            def __init__(self, drive_cfg):
                captured["drive_cfg"] = drive_cfg

            def build_drive_plan(self, **kwargs):
                return SimpleNamespace(regular_lfs=[])

            def build_drive_plan_per_day(self, **kwargs):
                return SimpleNamespace(regular_lfs=[])

        with patch.object(bc_mod, "LongFormAssembler", FakeAssembler):
            bc_mod._build_drive_plan_from_state(cfg, state)

        assert captured.get("drive_cfg") is sentinel_drive, (
            f"LongFormAssembler received {captured.get('drive_cfg')!r} — "
            f"expected the config.drive object verbatim"
        )


class TestCliShortsOverridesReachConfig:
    """CLI ``--shorts-target N`` and ``--shorts-gap N`` must actually flow into
    ``config.drive.shorts_target_count`` / ``shorts_min_gap_sec`` before
    day_pipeline runs. Past bug: the orchestrator stored these in
    ``state["shorts_target"]`` / ``state["shorts_gap"]`` but nothing
    downstream read those state keys, so the CLI values were printed in
    the header and silently ignored in the actual plan (same silent-drop
    class as the brand_logo_path fix)."""

    def test_run_applies_shorts_target_to_config_drive(self, tmp_path):
        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator

        # Source dir must exist — the orchestrator short-circuits if not, and
        # the CLI-override mutation lives inside the state-building block that
        # runs only after source validation.
        source = tmp_path / "source"
        source.mkdir()

        cfg = Config()
        orch = DrivePipelineOrchestrator(config=cfg)
        # Preflight will fail on an empty source, but that's fine — the config
        # mutation runs before preflight. We only assert on config state, not
        # on the final run result.
        orch.run(
            source_dir=source,
            output_base=tmp_path / "out",
            region="ireland",
            shorts_target=14,
            shorts_gap=45.0,
        )
        # Config must carry the CLI values forward to day_pipeline's reads.
        assert orch.config.drive.shorts_target_count == 14, (
            f"--shorts-target 14 did not reach config.drive.shorts_target_count: "
            f"{orch.config.drive.shorts_target_count}"
        )
        assert orch.config.drive.shorts_min_gap_sec == 45.0, (
            f"--shorts-gap 45 did not reach config.drive.shorts_min_gap_sec: "
            f"{orch.config.drive.shorts_min_gap_sec}"
        )

    def test_shorts_overrides_write_to_pipeline_state(self, tmp_path, monkeypatch):
        """CLI overrides MUST write to state so they survive
        resume-without-flag.

        The planner reads through state as well as config (read-priority:
        state wins over config), so the state writes are required. The positive control
        on ``config.drive.*`` stays (backward compat for in-process
        planners / test harnesses that read config directly).

        The end-to-end resume path is pinned in test_silent_degradation_upload.py.
        """
        from gomoveshift.config import Config
        from gomoveshift.pipeline import drive_orchestrator as drive_orch_mod
        from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator

        source = tmp_path / "source"
        source.mkdir()

        # Capture the state dict at the moment the orchestrator first writes
        # the checkpoint — that's the authoritative snapshot the planner and
        # resume path see. Monkeypatch ``_save_state`` to grab it and
        # short-circuit before preflight runs.
        captured_state: dict = {}

        class _StopError(Exception):
            pass

        def fake_save_state(self, path, state):
            captured_state["state"] = state
            raise _StopError()

        monkeypatch.setattr(
            drive_orch_mod.DrivePipelineOrchestrator,
            "_save_state",
            fake_save_state,
        )

        cfg = Config()
        orch = DrivePipelineOrchestrator(config=cfg)
        try:
            orch.run(
                source_dir=source,
                output_base=tmp_path / "out",
                region="ireland",
                shorts_target=14,
                shorts_gap=45.0,
            )
        except _StopError:
            pass

        state = captured_state.get("state")
        assert state is not None, (
            "Test harness never reached _save_state — the CLI-override block may have moved."
        )

        # ── Core assertion: both state writes land ──────────
        assert state.get("shorts_target") == 14, (
            "shorts_target must persist on state so it survives resume-without-flag."
        )
        assert state.get("shorts_gap") == 45.0, (
            "shorts_gap must persist on state so it survives resume-without-flag."
        )

        # ── Positive control: config.drive.* still carries the override ─
        # (backward-compat: in-process planners that read config directly
        # must still see it; planner's state-first read wins on resume.)
        assert orch.config.drive.shorts_target_count == 14, (
            "CLI override lost: --shorts-target must still mutate "
            "config.drive.shorts_target_count for in-process readers."
        )
        assert orch.config.drive.shorts_min_gap_sec == 45.0, (
            "CLI override lost: --shorts-gap must still mutate "
            "config.drive.shorts_min_gap_sec for in-process readers."
        )


class TestRetiredVolumeKeys:
    """The hand-set volume keys are gone; a leftover key in a user YAML must
    warn (never raise) because pydantic drops unknown keys silently."""

    _KEYS = ("shorts_per_lf_count", "lf_length_min_n", "lf_length_bucket_edges_sec")

    def _load(self, tmp_path, drive_block):
        import yaml

        from gomoveshift.config import Config

        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(drive_block))
        return Config.load(path)

    def test_user_config_with_shorts_per_lf_count_logs_retired_warning(self, tmp_path, caplog):
        import logging

        drive = {"shorts_per_lf_count": 7, "lf_length_min_n": 5, "lf_length_bucket_edges_sec": [1]}
        with caplog.at_level(logging.WARNING, logger="gomoveshift.config"):
            cfg = self._load(tmp_path, {"drive": drive})
        for key in self._KEYS:
            hits = [
                r
                for r in caplog.records
                if r.levelno == logging.WARNING
                and f"RETIRED_CONFIG_KEY key={key} (retired: Waypost decides)" in r.getMessage()
            ]
            assert len(hits) == 1, (key, [r.getMessage() for r in caplog.records])
            assert not hasattr(cfg.drive, key)

    def test_config_without_retired_keys_logs_no_retired_warning(self, tmp_path, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="gomoveshift.config"):
            self._load(
                tmp_path,
                {"drive": {"shorts_min_gap_sec": 50.0}, "other": {"shorts_per_lf_count": 3}},
            )
        assert not [r for r in caplog.records if "RETIRED_CONFIG_KEY" in r.getMessage()]

    def test_same_named_key_outside_drive_section_does_not_warn(self, tmp_path, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="gomoveshift.config"):
            self._load(tmp_path, {"seo": {"shorts_per_lf_count": 3}})
        assert not [r for r in caplog.records if "RETIRED_CONFIG_KEY" in r.getMessage()]
