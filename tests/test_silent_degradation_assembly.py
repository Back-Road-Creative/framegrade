"""Silent-degradation regression tests for the assembly/validation path.

Each test forces a log-only fallback path in one of the assembly modules and
asserts that a WARNING was emitted naming both the preferred path and the
fallback, so an operator watching the console can see the degradation.

These guard against the "silent fallback" class of bug surfaced in the
silent-degradation review.

Findings covered (log-only, no behavior change):
  * phase_d: empty drive_plan -> no-op deliverables
  * montage_builder.build_montage: no peaks -> return None
  * montage_builder.prepend_montage: missing master/peaks -> return None
  * day_pipeline.extract_and_concat_segments: missing master -> return None
  * day_pipeline.extract_and_concat_segments: empty segments -> return None
  * day_pipeline.extract_and_concat_segments: stream-copy failure -> None
  * longform_metadata.generate: SEO Manager unavailable -> _generate_basic
  * pipeline_validator._check_file_integrity: missing file -> passed=False
  * shorts_validator.validate_shorts: no short files -> silent passed=True skip
  * cleanup.plan_cleanup: pipeline incomplete -> skipped list
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from _state_helpers import make_state


MONTAGE_LOGGER = "gomoveshift.pipeline.montage_builder"
# The extractor moved onto the ``keycut`` package; its GMS-shaped wrapper
# (and therefore its degradation warnings) now lives in day_pipeline.
SEGEXT_LOGGER = "gomoveshift.pipeline.day_pipeline"
LFMETA_LOGGER = "gomoveshift.pipeline.longform_metadata"
VALIDATOR_LOGGER = "gomoveshift.pipeline.pipeline_validator"
SHORTS_VAL_LOGGER = "gomoveshift.pipeline.validation.shorts_validator"
CLEANUP_LOGGER = "gomoveshift.pipeline.cleanup"
PHASE_D_LOGGER = "gomoveshift.pipeline.drive_phases.phase_d"


# ---------------------------------------------------------------------------
# phase_d — empty drive_plan defensive init
# ---------------------------------------------------------------------------


class TestPhaseDEmptyPlanWarns:
    """Empty drive_plan must emit a summary WARNING listing what's no-op'd."""

    def test_empty_drive_plan_logs_no_op_summary(self, caplog):
        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_phases.phase_d import PhaseD

        phase = PhaseD(Config())

        state = make_state(
            region="ireland",
            publish_day="wed",
            location="Test",
            output_base="/tmp/out",
            source_dir="/tmp/src",
            skip_upload=True,
            skip_csv=True,
            drive_plan=None,
        )

        with caplog.at_level("WARNING", logger=PHASE_D_LOGGER):
            phase.run(state, drive_plan=None)

        # Fallback ran — empty state keys installed.
        assert state["metadata_per_lf"] == {}
        assert state["drive_schedule"] is None
        assert state["social_posts_per_lf"] == {}
        assert state["upload_manifests"] == {}

        messages = [r.getMessage() for r in caplog.records if r.name == PHASE_D_LOGGER]
        # The new summary warning names every no-op'd deliverable.
        assert any(
            "drive_plan empty" in m
            and "no-op deliverables" in m
            and "0 metadata" in m
            and "0 schedule" in m
            and "0 posts" in m
            and "0 uploads" in m
            for m in messages
        ), f"expected phase_d no-op summary warning, got: {messages!r}"


# ---------------------------------------------------------------------------
# montage_builder
# ---------------------------------------------------------------------------


class TestBuildMontageNoPeaksWarns:
    """build_montage with no peaks must return None and WARN."""

    def test_no_peaks_returns_none_and_warns(self, tmp_path, caplog):
        from gomoveshift.pipeline import montage_builder

        master = tmp_path / "master.mp4"
        master.touch()
        output = tmp_path / "montage.mp4"
        work = tmp_path / "work"

        with caplog.at_level("WARNING", logger=MONTAGE_LOGGER):
            result = montage_builder.build_montage(
                master_path=master,
                peaks=[],
                file_boundaries=[(0.0, 60.0)],
                output_path=output,
                work_dir=work,
            )

        assert result is None

        messages = [r.getMessage() for r in caplog.records if r.name == MONTAGE_LOGGER]
        assert any("build_montage" in m and "no peaks" in m and "no-op" in m for m in messages), (
            f"expected build_montage no-peaks warning, got: {messages!r}"
        )


class TestPrependMontageMissingMasterWarns:
    """prepend_montage with missing master must WARN with full context."""

    def test_missing_master_returns_none_and_warns(self, tmp_path, caplog):
        from gomoveshift.pipeline import montage_builder

        raw_lf = tmp_path / "raw_lf.mp4"
        raw_lf.touch()
        output = tmp_path / "out.mp4"
        work = tmp_path / "work"

        with caplog.at_level("WARNING", logger=MONTAGE_LOGGER):
            result = montage_builder.prepend_montage(
                raw_lf_path=raw_lf,
                peaks=[MagicMock()],
                output_path=output,
                work_dir=work,
                master_path=None,
                file_boundaries=[(0.0, 60.0)],
            )

        assert result is None

        messages = [r.getMessage() for r in caplog.records if r.name == MONTAGE_LOGGER]
        assert any(
            "prepend_montage" in m and "master_path_present=False" in m and "no-op" in m
            for m in messages
        ), f"expected prepend_montage missing-master warning, got: {messages!r}"


# ---------------------------------------------------------------------------
# segment extraction (day_pipeline's keycut-backed wrapper)
# ---------------------------------------------------------------------------


class TestSegmentExtractorMissingMasterWarns:
    """extract_and_concat_segments with missing master must WARN."""

    def test_missing_master_returns_none_and_warns(self, tmp_path, caplog):
        from gomoveshift.pipeline import day_pipeline

        master = tmp_path / "does_not_exist.mp4"
        assert not master.exists()
        output = tmp_path / "out.mp4"
        work = tmp_path / "work"

        with caplog.at_level("WARNING", logger=SEGEXT_LOGGER):
            result = day_pipeline.extract_and_concat_segments(
                master_path=master,
                segments=[MagicMock()],
                file_boundaries=[(0.0, 60.0)],
                output_path=output,
                work_dir=work,
            )

        assert result is None

        messages = [r.getMessage() for r in caplog.records if r.name == SEGEXT_LOGGER]
        assert any(
            "segment extract" in m and "master file missing" in m and "no-op" in m for m in messages
        ), f"expected segment-extract missing-master warning, got: {messages!r}"


class TestSegmentExtractorEmptySegmentsWarns:
    """extract_and_concat_segments with no segments must WARN."""

    def test_empty_segments_returns_none_and_warns(self, tmp_path, caplog):
        from gomoveshift.pipeline import day_pipeline

        master = tmp_path / "master.mp4"
        master.touch()
        output = tmp_path / "out.mp4"
        work = tmp_path / "work"

        with caplog.at_level("WARNING", logger=SEGEXT_LOGGER):
            result = day_pipeline.extract_and_concat_segments(
                master_path=master,
                segments=[],
                file_boundaries=[(0.0, 60.0)],
                output_path=output,
                work_dir=work,
            )

        assert result is None

        messages = [r.getMessage() for r in caplog.records if r.name == SEGEXT_LOGGER]
        assert any("no segments to extract" in m and "no-op" in m for m in messages), (
            f"expected segment-extract empty-segments warning, got: {messages!r}"
        )


class TestSegmentExtractorStreamCopyFailureWarns:
    """extract_and_concat_segments: stream-copy failure must emit a summary WARN."""

    def test_stream_copy_failure_returns_none_and_warns(self, tmp_path, monkeypatch, caplog):
        import keycut.extract
        from gomoveshift.pipeline import day_pipeline

        master = tmp_path / "master.mp4"
        master.touch()
        output = tmp_path / "out.mp4"
        work = tmp_path / "work"

        segment = SimpleNamespace(source_file_index=0, start_time=0.0, end_time=5.0)
        # Provide keyframes so the extractor takes the stream-copy path (the
        # path this test is named for). Without them it re-encodes on
        # probe-failure and never reaches the mocked stream_copy_range.
        monkeypatch.setattr(
            keycut.extract, "probe_keyframes", lambda *_a, **_kw: [0.0, 2.002, 4.004]
        )
        monkeypatch.setattr(keycut.extract, "stream_copy_range", lambda *_a, **_kw: False)

        with caplog.at_level("WARNING", logger=SEGEXT_LOGGER):
            result = day_pipeline.extract_and_concat_segments(
                master_path=master,
                segments=[segment],
                file_boundaries=[(0.0, 60.0)],
                output_path=output,
                work_dir=work,
            )

        assert result is None

        messages = [r.getMessage() for r in caplog.records if r.name == SEGEXT_LOGGER]
        assert any("stream-copy/concat failed" in m and "no-op" in m for m in messages), (
            f"expected segment-extract stream-copy-failure warning, got: {messages!r}"
        )


# ---------------------------------------------------------------------------
# longform_metadata — fallback to _generate_basic
# ---------------------------------------------------------------------------


class TestLongformMetadataSeoDegradedWarns:
    """generate() with no SEO manager must WARN per-video about degradation."""

    def test_seo_unavailable_falls_back_and_warns(self, caplog, monkeypatch):
        from gomoveshift.pipeline import longform_metadata

        # Block SEO init so self._manager stays None and _seo_degraded=True.
        def broken_init(self):
            self._manager = None
            self._optimizer = None
            self._seo_degraded = True

        monkeypatch.setattr(
            longform_metadata.LongFormMetadataGenerator,
            "_init_seo_manager",
            broken_init,
        )
        monkeypatch.setattr(
            longform_metadata.LongFormMetadataGenerator,
            "_init_ledger",
            lambda self: None,
        )
        # ContentValidator is independent — skip its enforcement to isolate the warning.
        monkeypatch.setattr(
            longform_metadata.ContentValidator,
            "enforce",
            lambda self, **kw: None,
        )

        gen = longform_metadata.LongFormMetadataGenerator()

        # Stub _generate_basic so we don't need full deps to execute.
        monkeypatch.setattr(
            gen,
            "_generate_basic",
            lambda *a, **kw: {
                "title": "POV Drive Through County Tipperary",
                "description": "desc",
                "tags": [],
            },
        )

        with caplog.at_level("WARNING", logger=LFMETA_LOGGER):
            result = gen.generate(
                location="County Tipperary, Ireland",
                publish_day="Wednesday",
                video_duration=600.0,
                project_name="test",
            )

        assert result["title"]
        messages = [r.getMessage() for r in caplog.records if r.name == LFMETA_LOGGER]
        assert any(
            "SEO Manager unavailable" in m and "_generate_basic" in m and "degraded" in m
            for m in messages
        ), f"expected longform_metadata fallback warning, got: {messages!r}"


# ---------------------------------------------------------------------------
# pipeline_validator — file-integrity gate failures
# ---------------------------------------------------------------------------


class TestPipelineValidatorFileMissingWarns:
    """_check_file_integrity on a missing file must WARN (not silent passed=False)."""

    def test_missing_file_logs_warning(self, tmp_path, caplog):
        from gomoveshift.pipeline.pipeline_validator import PipelineValidator

        v = PipelineValidator()
        missing = tmp_path / "does-not-exist.mp4"
        assert not missing.exists()

        with caplog.at_level("WARNING", logger=VALIDATOR_LOGGER):
            result = v._check_file_integrity(missing)

        assert result.passed is False
        messages = [r.getMessage() for r in caplog.records if r.name == VALIDATOR_LOGGER]
        assert any(
            "file_integrity gate FAILED" in m and "file does not exist" in m for m in messages
        ), f"expected pipeline_validator missing-file warning, got: {messages!r}"


# ---------------------------------------------------------------------------
# shorts_validator — silent-skip on empty dir
# ---------------------------------------------------------------------------


class TestShortsValidatorNoFilesWarns:
    """validate_shorts on an empty dir must WARN about the silent skip."""

    def test_no_shorts_logs_warning_but_keeps_passed_true(self, tmp_path, caplog):
        from gomoveshift.pipeline.validation.ffprobe_client import FFprobeClient
        from gomoveshift.pipeline.validation.shorts_validator import ShortsValidator

        v = ShortsValidator(ffprobe=MagicMock(spec=FFprobeClient))
        empty_dir = tmp_path / "shorts"
        empty_dir.mkdir()

        with caplog.at_level("WARNING", logger=SHORTS_VAL_LOGGER):
            result = v.validate_shorts(empty_dir)

        # Behavior preserved: still passed=True (log-only, not behavioral).
        assert result.passed is True

        messages = [r.getMessage() for r in caplog.records if r.name == SHORTS_VAL_LOGGER]
        assert any("no short_*.mp4 files found" in m and "no-op" in m for m in messages), (
            f"expected shorts_validator silent-skip warning, got: {messages!r}"
        )


# ---------------------------------------------------------------------------
# cleanup — pipeline incomplete silent skip
# ---------------------------------------------------------------------------


class TestCleanupIncompletePipelineWarns:
    """plan_cleanup skipping an incomplete pipeline must WARN."""

    def test_incomplete_pipeline_logs_warning(self, tmp_path, caplog, monkeypatch):
        from gomoveshift.pipeline import cleanup as cleanup_mod

        # Build a planner with age_days>min so it reaches the pipeline check.
        planner = cleanup_mod.PipelineCleanup(
            output_base=tmp_path / "output",
            staging_dir=tmp_path / "staging",
            d_drive_base=tmp_path / "d_drive",
        )

        project_dir = tmp_path / "proj_a"
        project_dir.mkdir()

        # age_days forcibly old enough; pipeline_complete False.
        monkeypatch.setattr(planner, "_project_age_days", lambda p: 30.0)
        monkeypatch.setattr(planner, "_load_pipeline_state", lambda p: {})
        monkeypatch.setattr(planner, "_is_pipeline_complete", lambda s: False)

        with caplog.at_level("WARNING", logger=CLEANUP_LOGGER):
            plan = planner.plan_cleanup([project_dir], min_age_days=7)

        # Behavior preserved: still appended to skipped list.
        assert any(s["path"] == str(project_dir) for s in plan.skipped)

        messages = [r.getMessage() for r in caplog.records if r.name == CLEANUP_LOGGER]
        assert any(
            "cleanup" in m and "pipeline not complete" in m and "no-op" in m for m in messages
        ), f"expected cleanup incomplete-pipeline warning, got: {messages!r}"
