"""Tests for deterministic pipeline paths and auto-cleanup."""

import os
from pathlib import Path

import pytest

from gomoveshift.pipeline.cleanup import (
    CleanupPlan,
    PipelineCleanup,
    drop_day_master,
    purge_day_intermediates,
    verify_shorts_integrity,
)
from gomoveshift.pipeline.orchestrator import _PipelinePaths


class TestPipelinePaths:
    """Pipeline paths must resolve deterministically from output_base."""

    def test_resolve_creates_consistent_paths(self):
        """All paths should derive from output_base + project_name.

        When output is on drvfs (/mnt/), stages_dir redirects to ext4 for
        I/O performance. State file stays on drvfs for persistence.
        """
        paths = _PipelinePaths.resolve(Path("/mnt/d/output"), "testproject_3840p")

        assert paths.output_base == Path("/mnt/d/output")
        assert paths.project_dir == Path("/mnt/d/output/testproject_3840p")
        # stages_dir redirects to ext4 when output is on drvfs
        assert not str(paths.stages_dir).startswith("/mnt/")
        assert "testproject_3840p" in str(paths.stages_dir)
        # state_file stays on drvfs for persistence across /tmp wipes
        assert str(paths.state_file).startswith("/mnt/")

    def test_staging_dir_from_env(self, monkeypatch):
        """Staging dir should respect GMS_STAGING_DIR env var."""
        monkeypatch.setenv("GMS_STAGING_DIR", "/custom/staging")
        paths = _PipelinePaths.resolve(Path("/data/output"), "test")
        assert paths.staging_dir == Path("/custom/staging")

    def test_staging_dir_default(self, monkeypatch, tmp_path):
        """Staging dir defaults to ``<data home>/input``."""
        monkeypatch.delenv("GMS_STAGING_DIR", raising=False)
        monkeypatch.setenv("GMS_DATA_HOME", str(tmp_path))
        paths = _PipelinePaths.resolve(Path("/data/output"), "test")
        assert paths.staging_dir == tmp_path / "input"
        monkeypatch.setenv("GMS_STAGING_DIR", str(tmp_path / "elsewhere"))
        paths = _PipelinePaths.resolve(Path("/data/output"), "test")
        assert paths.staging_dir == tmp_path / "elsewhere"


class TestCleanupDefaults:
    """Cleanup module defaults must match pipeline defaults."""

    def test_output_base_matches_pipeline(self, monkeypatch, tmp_path):
        """Cleanup output_base default is ``<data home>/output`` (GMS_OUTPUT_DIR overrides)."""
        monkeypatch.delenv("GMS_OUTPUT_DIR", raising=False)
        monkeypatch.setenv("GMS_DATA_HOME", str(tmp_path))
        cleanup = PipelineCleanup()
        assert cleanup.output_base == tmp_path / "output"
        monkeypatch.setenv("GMS_OUTPUT_DIR", str(tmp_path / "elsewhere"))
        assert PipelineCleanup().output_base == tmp_path / "elsewhere"

    def test_staging_dir_matches_pipeline(self, monkeypatch, tmp_path):
        """Cleanup staging_dir default is ``<data home>/input`` (GMS_STAGING_DIR overrides)."""
        monkeypatch.delenv("GMS_STAGING_DIR", raising=False)
        monkeypatch.setenv("GMS_DATA_HOME", str(tmp_path))
        cleanup = PipelineCleanup()
        assert cleanup.staging_dir == tmp_path / "input"
        monkeypatch.setenv("GMS_STAGING_DIR", str(tmp_path / "elsewhere"))
        assert PipelineCleanup().staging_dir == tmp_path / "elsewhere"

    def test_env_override_respected(self, monkeypatch):
        """Both output_base and staging_dir should respect env vars."""
        monkeypatch.setenv("GMS_OUTPUT_DIR", "/custom/output")
        monkeypatch.setenv("GMS_STAGING_DIR", "/custom/staging")
        cleanup = PipelineCleanup()
        assert cleanup.output_base == Path("/custom/output")
        assert cleanup.staging_dir == Path("/custom/staging")


class TestCleanupPlanStagingFiles:
    """Cleanup must include FINAL copies in staging, not just source copies."""

    def test_plan_includes_final_staging_copy(self, tmp_path):
        """FINAL.mp4 copies in staging dir should be marked for cleanup."""
        # Set up project dir with completed pipeline
        project_dir = tmp_path / "output" / "testproject"
        stages_dir = project_dir / "_stages"
        stages_dir.mkdir(parents=True)

        # Create final video
        final = project_dir / "testproject_FINAL.mp4"
        final.write_bytes(b"x" * 1000)

        # Create pipeline state showing completion
        import json

        state = {
            "stages_completed": ["csv_generation"],
            "source_video": "/data/input/TestProject.mp4",
        }
        (stages_dir / "pipeline_state.json").write_text(json.dumps(state))

        # Create staging copies (both source and FINAL)
        staging_dir = tmp_path / "staging"
        staging_dir.mkdir()
        (staging_dir / "TestProject.mp4").write_bytes(b"x" * 2000)  # source copy
        (staging_dir / "testproject_FINAL.mp4").write_bytes(b"x" * 1000)  # FINAL copy

        # D: drive copy (for verification gate)
        d_dir = tmp_path / "d_output" / "testproject"
        d_dir.mkdir(parents=True)
        (d_dir / "testproject_FINAL.mp4").write_bytes(b"x" * 1000)

        cleanup = PipelineCleanup(
            output_base=tmp_path / "output",
            staging_dir=staging_dir,
            d_drive_base=tmp_path / "d_output",
        )

        projects = cleanup.discover_projects()
        plan = cleanup.plan_cleanup(projects, min_age_days=0)

        cleanup_paths = [f["path"] for f in plan.files]
        # Both source and FINAL staging copies should be in plan
        assert str(staging_dir / "TestProject.mp4") in cleanup_paths
        assert str(staging_dir / "testproject_FINAL.mp4") in cleanup_paths

    def test_plan_includes_color_analysis_artifacts(self, tmp_path):
        """Color analysis JSON files in staging should be cleaned."""
        project_dir = tmp_path / "output" / "testproject"
        stages_dir = project_dir / "_stages"
        stages_dir.mkdir(parents=True)

        final = project_dir / "testproject_FINAL.mp4"
        final.write_bytes(b"x" * 1000)

        import json

        state = {
            "stages_completed": ["csv_generation"],
            "source_video": "/data/input/TestProject.mp4",
        }
        (stages_dir / "pipeline_state.json").write_text(json.dumps(state))

        staging_dir = tmp_path / "staging"
        staging_dir.mkdir()
        (staging_dir / "TestProject_color_analysis.json").write_text('{"luma": 0.5}')

        d_dir = tmp_path / "d_output" / "testproject"
        d_dir.mkdir(parents=True)
        (d_dir / "testproject_FINAL.mp4").write_bytes(b"x" * 1000)

        cleanup = PipelineCleanup(
            output_base=tmp_path / "output",
            staging_dir=staging_dir,
            d_drive_base=tmp_path / "d_output",
        )

        projects = cleanup.discover_projects()
        plan = cleanup.plan_cleanup(projects, min_age_days=0)

        cleanup_paths = [f["path"] for f in plan.files]
        assert any("color_analysis" in p for p in cleanup_paths)


class TestCleanupSafetyGates:
    """Cleanup must NOT delete files when safety gates fail."""

    def test_no_cleanup_if_pipeline_incomplete(self, tmp_path):
        """Incomplete pipeline should not trigger staging cleanup."""
        project_dir = tmp_path / "output" / "testproject"
        stages_dir = project_dir / "_stages"
        stages_dir.mkdir(parents=True)

        import json

        state = {"stages_completed": ["detect_properties", "stop_removal"]}
        (stages_dir / "pipeline_state.json").write_text(json.dumps(state))

        staging_dir = tmp_path / "staging"
        staging_dir.mkdir()
        (staging_dir / "TestProject.mp4").write_bytes(b"x" * 1000)

        cleanup = PipelineCleanup(
            output_base=tmp_path / "output",
            staging_dir=staging_dir,
        )

        projects = cleanup.discover_projects()
        plan = cleanup.plan_cleanup(projects, min_age_days=0)

        assert len(plan.files) == 0
        assert any("not complete" in s["reason"] for s in plan.skipped)


class TestCleanupFinalIntegrityGate:
    """``_stages/`` intermediates must NOT be deleted when ``_FINAL.mp4``
    is truncated-but-nonzero.

    The old gate was ``final.exists() and final.stat().st_size > 0`` only — a
    corrupt FINAL that still has bytes triggered intermediate deletion,
    destroying the only economical re-encode path. The fix ffprobe-validates
    the final first, mirroring the ``drop_day_master`` shorts-integrity guard.
    """

    def _completed_project(self, tmp_path):
        """A project whose pipeline is complete with a nonzero FINAL + stages."""
        import json

        project_dir = tmp_path / "output" / "testproject"
        stages_dir = project_dir / "_stages"
        stages_dir.mkdir(parents=True)

        final = project_dir / "testproject_FINAL.mp4"
        final.write_bytes(b"x" * 1000)  # nonzero but (per fake probe) corrupt

        state = {
            "stages_completed": ["csv_generation"],
            "source_video": "/data/input/TestProject.mp4",
        }
        (stages_dir / "pipeline_state.json").write_text(json.dumps(state))

        # Populated intermediates that the bug would delete.
        (stages_dir / "_no_stops.mp4").write_bytes(b"x" * 5000)
        (stages_dir / "_color.mp4").write_bytes(b"x" * 4000)
        (stages_dir / "_analysis.json").write_text("{}")
        return project_dir, stages_dir, final

    def test_corrupt_final_skips_intermediate_deletion(self, tmp_path):
        """Truncated-but-nonzero FINAL → intermediates land in plan.skipped."""
        project_dir, stages_dir, _ = self._completed_project(tmp_path)

        cleanup = PipelineCleanup(
            output_base=tmp_path / "output",
            staging_dir=tmp_path / "staging",
        )

        def bad_probe(_path):
            return _FakeProbeResult(valid=False, duration=None, reason="no moov atom")

        projects = cleanup.discover_projects()
        plan = cleanup.plan_cleanup(projects, min_age_days=0, ffprobe_fn=bad_probe)

        # No intermediate may appear in the deletion list.
        intermediate_deletes = [f["path"] for f in plan.files if str(stages_dir) in f["path"]]
        assert not intermediate_deletes, (
            f"Corrupt FINAL must not delete intermediates: {intermediate_deletes}"
        )
        # And the stages dir must be recorded as skipped with a probe reason.
        assert any(str(stages_dir) in s["path"] and "FINAL" in s["reason"] for s in plan.skipped)

    def test_valid_final_still_deletes_intermediates(self, tmp_path):
        """Happy path: a valid FINAL still lets intermediates be reclaimed."""
        project_dir, stages_dir, _ = self._completed_project(tmp_path)

        cleanup = PipelineCleanup(
            output_base=tmp_path / "output",
            staging_dir=tmp_path / "staging",
        )

        projects = cleanup.discover_projects()
        plan = cleanup.plan_cleanup(projects, min_age_days=0, ffprobe_fn=_ok_probe)

        intermediate_deletes = [f["path"] for f in plan.files if str(stages_dir) in f["path"]]
        # The two .mp4 + one .json intermediate (state.json kept) are deletable.
        assert any("_no_stops.mp4" in p for p in intermediate_deletes)
        assert any("_color.mp4" in p for p in intermediate_deletes)
        assert any("_analysis.json" in p for p in intermediate_deletes)
        # pipeline_state.json is never deleted.
        assert not any("pipeline_state.json" in p for p in intermediate_deletes)


class TestCleanupStateReadVisibility:
    """Corrupt state files must be visible via a WARNING log.

    A bare ``except Exception: return None`` would mask unreadable state
    files: the project would be skipped with no signal that a corrupt JSON
    file was the cause. The reader degrades gracefully (returns None, so
    the caller treats the project as incomplete) and emits a WARNING
    naming the file and the exception so triage has a breadcrumb.
    """

    def test_corrupt_state_logs_warning(self, tmp_path, caplog):
        project_dir = tmp_path / "output" / "broken"
        stages_dir = project_dir / "_stages"
        stages_dir.mkdir(parents=True)
        state_file = stages_dir / "pipeline_state.json"
        state_file.write_text("{not valid json")

        cleanup = PipelineCleanup(
            output_base=tmp_path / "output",
            staging_dir=tmp_path / "staging",
        )

        import logging

        with caplog.at_level(logging.WARNING, logger="gomoveshift.pipeline.cleanup"):
            result = cleanup._load_pipeline_state(project_dir)

        assert result is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "expected at least one WARNING log on corrupt state"
        message = warnings[-1].getMessage()
        assert str(state_file) in message
        assert "JSONDecodeError" in message or "json" in message.lower()

    def test_missing_state_does_not_warn(self, tmp_path, caplog):
        """Absent state file is the normal in-progress case — no WARNING."""
        project_dir = tmp_path / "output" / "fresh"
        (project_dir / "_stages").mkdir(parents=True)

        cleanup = PipelineCleanup(
            output_base=tmp_path / "output",
            staging_dir=tmp_path / "staging",
        )

        import logging

        with caplog.at_level(logging.WARNING, logger="gomoveshift.pipeline.cleanup"):
            result = cleanup._load_pipeline_state(project_dir)

        assert result is None
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]


class TestCleanupDriveStateRecognition:
    """Cleanup must recognize drive-orchestrator state files.

    The drive orchestrator writes ``phases_completed`` (with phase names like
    ``phase_a``, ``phase_d``) to the same ``_stages/pipeline_state.json`` path
    that the cleanup handler reads. Before the fix the cleanup handler only
    looked for ``csv_generation`` in ``stages_completed`` — so drive-pipeline
    projects were never recognized as complete and their intermediates piled
    up indefinitely. Both keys are legitimate, distinct logical buckets; the
    fix is to accept either as a completion signal, not to rename one into
    the other.
    """

    def _write_drive_state(self, state_path: Path, phases_completed: list[str]) -> None:
        """Mirror what :class:`DrivePipelineOrchestrator._save_state` writes.

        Not calling the orchestrator directly (that would pull the whole
        drive-pipeline import graph and require a full fixture tree). The
        state-write path is a single-line JSON dump through
        ``save_drive_state``; by constructing an equivalent dict we exercise
        the same cleanup-read surface that the orchestrator writes against.
        """
        import json

        state = {
            "phases_completed": phases_completed,
            "source_dir": "/data/source/TestDrive",
            "region": "usa",
            "publish_day": "wed",
        }
        state_path.write_text(json.dumps(state))

    def test_cleanup_recognizes_drive_pipeline_as_complete(self, tmp_path):
        """Drive state with ``phase_d`` in ``phases_completed`` must be complete."""
        project_dir = tmp_path / "output" / "driveproject"
        stages_dir = project_dir / "_stages"
        stages_dir.mkdir(parents=True)

        # Drive orchestrator writes phases_completed, not stages_completed.
        self._write_drive_state(
            stages_dir / "pipeline_state.json",
            phases_completed=["phase_a", "phase_bc_day_first", "phase_d"],
        )

        # Final deliverable exists (so the _FINAL.mp4 gate passes).
        final = project_dir / "driveproject_FINAL.mp4"
        final.write_bytes(b"x" * 1000)

        staging_dir = tmp_path / "staging"
        staging_dir.mkdir()

        cleanup = PipelineCleanup(
            output_base=tmp_path / "output",
            staging_dir=staging_dir,
        )

        projects = cleanup.discover_projects()
        # The bug: cleanup reads ``stages_completed`` (main-pipeline key), so
        # drive projects are always skipped as "not complete". This assertion
        # fails before the fix.
        assert cleanup._is_pipeline_complete(cleanup._load_pipeline_state(project_dir)), (
            "Cleanup must recognize drive-pipeline completion via phases_completed."
        )

        plan = cleanup.plan_cleanup(projects, min_age_days=0)
        not_complete_skips = [s for s in plan.skipped if "not complete" in s["reason"]]
        assert not not_complete_skips, (
            f"Drive project incorrectly skipped as incomplete: {not_complete_skips}"
        )

    def test_cleanup_skips_drive_pipeline_without_terminal_phase(self, tmp_path):
        """Drive state missing ``phase_d`` must still be treated as incomplete."""
        project_dir = tmp_path / "output" / "driveproject"
        stages_dir = project_dir / "_stages"
        stages_dir.mkdir(parents=True)

        # Partial drive run — phase_d never ran.
        self._write_drive_state(
            stages_dir / "pipeline_state.json",
            phases_completed=["phase_a", "phase_bc_day_first"],
        )

        cleanup = PipelineCleanup(
            output_base=tmp_path / "output",
            staging_dir=tmp_path / "staging",
        )

        assert not cleanup._is_pipeline_complete(cleanup._load_pipeline_state(project_dir)), (
            "Drive state without phase_d must not be treated as complete."
        )

    def test_cleanup_recognizes_main_pipeline_as_complete(self, tmp_path):
        """Regression: main pipeline completion path must still work.

        The main video pipeline writes ``stages_completed`` with
        ``csv_generation`` as its terminal marker. The fix must keep reading
        that key, not replace it.
        """
        import json

        project_dir = tmp_path / "output" / "mainproject"
        stages_dir = project_dir / "_stages"
        stages_dir.mkdir(parents=True)

        state = {
            "stages_completed": [
                "detect_properties",
                "stop_removal",
                "csv_generation",
            ],
            "source_video": "/data/input/MainProject.mp4",
        }
        (stages_dir / "pipeline_state.json").write_text(json.dumps(state))

        cleanup = PipelineCleanup(
            output_base=tmp_path / "output",
            staging_dir=tmp_path / "staging",
        )

        assert cleanup._is_pipeline_complete(cleanup._load_pipeline_state(project_dir)), (
            "Main-pipeline stages_completed path must remain recognized."
        )


class TestPurgeDayIntermediates:
    """Per-day intermediate cleanup for the day-first drive pipeline.

    After ``run_day_pipeline`` emits final ``lf_XX.mp4`` and ``short_XX.mp4``
    files, the LF work files (``_segments.mp4``, ``_with_montage.mp4``,
    ``.lf_XX_work/``) and short pre-OST raw files (``short_XX_raw.mp4``) are
    dead weight — they can account for 13+ GB per day. ``purge_day_intermediates``
    removes them, but only per-LF/short where the final file exists with
    non-zero size. It never touches the master or the final deliverables.
    """

    def _make_day_dir(self, tmp_path: Path) -> Path:
        day_dir = tmp_path / "day_2026-02-21"
        (day_dir / "regular_lfs").mkdir(parents=True)
        (day_dir / "shorts").mkdir(parents=True)
        return day_dir

    def test_purges_lf_intermediates_when_final_exists(self, tmp_path):
        day_dir = self._make_day_dir(tmp_path)
        lf_dir = day_dir / "regular_lfs"

        (lf_dir / "lf_02.mp4").write_bytes(b"x" * 100)  # final
        segments = lf_dir / "lf_02_segments.mp4"
        montage = lf_dir / "lf_02_with_montage.mp4"
        segments.write_bytes(b"x" * 5000)
        montage.write_bytes(b"x" * 4000)

        summary = purge_day_intermediates(day_dir)

        assert not segments.exists()
        assert not montage.exists()
        assert summary["files_deleted"] == 2
        assert summary["bytes_freed"] == 9000

    def test_purges_lf_work_dirs_when_final_exists(self, tmp_path):
        day_dir = self._make_day_dir(tmp_path)
        lf_dir = day_dir / "regular_lfs"

        (lf_dir / "lf_03.mp4").write_bytes(b"x" * 100)
        work = lf_dir / ".lf_03_work"
        work.mkdir()
        (work / "scratch.mp4").write_bytes(b"x" * 2000)

        summary = purge_day_intermediates(day_dir)

        assert not work.exists()
        assert summary["dirs_deleted"] == 1

    def test_purges_short_raw_when_final_exists(self, tmp_path):
        day_dir = self._make_day_dir(tmp_path)
        shorts_dir = day_dir / "shorts"

        (shorts_dir / "short_01.mp4").write_bytes(b"x" * 50)  # final
        raw = shorts_dir / "short_01_raw.mp4"
        raw.write_bytes(b"x" * 1500)

        summary = purge_day_intermediates(day_dir)

        assert not raw.exists()
        assert summary["files_deleted"] == 1

    def test_purges_superlong_intermediates_when_final_exists(self, tmp_path):
        day_dir = self._make_day_dir(tmp_path)
        sl_dir = day_dir / "superlong"
        sl_dir.mkdir()

        (sl_dir / "sl_01.mp4").write_bytes(b"x" * 100)  # final — must survive
        segments = sl_dir / "sl_01_segments.mp4"
        segments.write_bytes(b"x" * 3000)
        work = sl_dir / ".sl_01_work"
        work.mkdir()
        (work / "scratch.mp4").write_bytes(b"x" * 1000)

        summary = purge_day_intermediates(day_dir)

        assert not segments.exists()
        assert not work.exists()
        assert (sl_dir / "sl_01.mp4").exists()
        assert summary["files_deleted"] == 1
        assert summary["dirs_deleted"] == 1
        assert summary["bytes_freed"] == 4000

    def test_keeps_intermediates_when_final_missing(self, tmp_path):
        day_dir = self._make_day_dir(tmp_path)
        lf_dir = day_dir / "regular_lfs"
        shorts_dir = day_dir / "shorts"

        # No final lf_02.mp4 or short_01.mp4 produced — must not purge.
        segments = lf_dir / "lf_02_segments.mp4"
        montage = lf_dir / "lf_02_with_montage.mp4"
        short_raw = shorts_dir / "short_01_raw.mp4"
        segments.write_bytes(b"x" * 100)
        montage.write_bytes(b"x" * 100)
        short_raw.write_bytes(b"x" * 100)

        summary = purge_day_intermediates(day_dir)

        assert segments.exists()
        assert montage.exists()
        assert short_raw.exists()
        assert summary["files_deleted"] == 0
        assert summary["dirs_deleted"] == 0

    def test_keeps_intermediates_when_final_is_empty(self, tmp_path):
        day_dir = self._make_day_dir(tmp_path)
        lf_dir = day_dir / "regular_lfs"

        # Final exists but is zero bytes (failed encode that left a stub).
        (lf_dir / "lf_02.mp4").write_bytes(b"")
        segments = lf_dir / "lf_02_segments.mp4"
        segments.write_bytes(b"x" * 500)

        summary = purge_day_intermediates(day_dir)

        assert segments.exists()
        assert summary["files_deleted"] == 0

    def test_preserves_master_and_finals(self, tmp_path):
        day_dir = self._make_day_dir(tmp_path)
        lf_dir = day_dir / "regular_lfs"
        shorts_dir = day_dir / "shorts"

        master = day_dir / "day_2026-02-21_master.mp4"
        cache = day_dir / "day_2026-02-21_color_cache.json"
        master.write_bytes(b"x" * 10_000)
        cache.write_text("{}")
        (lf_dir / "lf_02.mp4").write_bytes(b"x" * 100)
        (lf_dir / "lf_02_segments.mp4").write_bytes(b"x" * 500)
        (shorts_dir / "short_01.mp4").write_bytes(b"x" * 100)
        (shorts_dir / "short_01_raw.mp4").write_bytes(b"x" * 500)

        purge_day_intermediates(day_dir)

        assert master.exists(), "day master must never be purged by this function"
        assert cache.exists(), "color cache must survive — needed for resume"
        assert (lf_dir / "lf_02.mp4").exists()
        assert (shorts_dir / "short_01.mp4").exists()

    def test_idempotent(self, tmp_path):
        day_dir = self._make_day_dir(tmp_path)
        lf_dir = day_dir / "regular_lfs"

        (lf_dir / "lf_02.mp4").write_bytes(b"x" * 100)
        (lf_dir / "lf_02_segments.mp4").write_bytes(b"x" * 500)

        first = purge_day_intermediates(day_dir)
        second = purge_day_intermediates(day_dir)

        assert first["files_deleted"] == 1
        assert second["files_deleted"] == 0
        assert second["bytes_freed"] == 0

    def test_missing_day_dir_is_noop(self, tmp_path):
        """Non-existent day_dir returns zero summary, does not raise."""
        summary = purge_day_intermediates(tmp_path / "does_not_exist")
        assert summary == {"files_deleted": 0, "dirs_deleted": 0, "bytes_freed": 0}

    def test_per_lf_granularity(self, tmp_path):
        """Per-LF gate: lf_02 purged (final exists), lf_03 preserved (no final)."""
        day_dir = self._make_day_dir(tmp_path)
        lf_dir = day_dir / "regular_lfs"

        # LF 02: final exists → intermediates should go
        (lf_dir / "lf_02.mp4").write_bytes(b"x" * 100)
        lf_02_segs = lf_dir / "lf_02_segments.mp4"
        lf_02_segs.write_bytes(b"x" * 500)

        # LF 03: no final → intermediates must stay
        lf_03_segs = lf_dir / "lf_03_segments.mp4"
        lf_03_work = lf_dir / ".lf_03_work"
        lf_03_segs.write_bytes(b"x" * 500)
        lf_03_work.mkdir()
        (lf_03_work / "scratch.mp4").write_bytes(b"x" * 200)

        purge_day_intermediates(day_dir)

        assert not lf_02_segs.exists()
        assert lf_03_segs.exists()
        assert lf_03_work.exists()


class _FakeProbeResult:
    """Mimics ``ffmpeg_runner.VideoValidation`` for tests."""

    def __init__(self, valid: bool, duration: float | None = 10.0, reason: str = ""):
        self.valid = valid
        self.duration = duration
        self.reason = reason


def _ok_probe(_path):
    return _FakeProbeResult(valid=True, duration=10.0)


class TestDropDayMaster:
    """Drop `day_*_master.mp4` after a day's LFs + shorts are fully emitted.

    The master is ~59 GB per day and is the single biggest file in the cache.
    It's a working artifact used for LF extraction and short extraction; once
    the finals exist on disk, the master is dead weight. Resume is still safe
    because ``_reconstruct_day_artifacts_from_disk`` can rebuild ``DayArtifacts``
    from the finals alone.

    A shorts-integrity gate guards the master drop. These tests inject ``_ok_probe`` to
    isolate the file-existence behaviour under test from real ffprobe, which
    rejects the synthetic fixture bytes used here.
    """

    def _day_dir(self, tmp_path: Path, date: str = "2026-02-21") -> Path:
        day_dir = tmp_path / f"day_{date}"
        (day_dir / "regular_lfs").mkdir(parents=True)
        (day_dir / "shorts").mkdir(parents=True)
        return day_dir

    def test_drops_master_when_lf_and_short_finals_exist(self, tmp_path):
        day_dir = self._day_dir(tmp_path)
        master = day_dir / "day_2026-02-21_master.mp4"
        master.write_bytes(b"x" * 50_000)
        (day_dir / "regular_lfs" / "lf_01.mp4").write_bytes(b"x" * 200)
        (day_dir / "shorts" / "short_01.mp4").write_bytes(b"x" * 100)

        summary = drop_day_master(day_dir, date="2026-02-21", ffprobe_fn=_ok_probe)

        assert not master.exists()
        assert summary["dropped"] is True
        assert summary["bytes_freed"] == 50_000

    def test_keeps_master_when_no_lf_finals(self, tmp_path):
        day_dir = self._day_dir(tmp_path)
        master = day_dir / "day_2026-02-21_master.mp4"
        master.write_bytes(b"x" * 50_000)
        # Shorts final exists but no LF final — day incomplete, master stays
        (day_dir / "shorts" / "short_01.mp4").write_bytes(b"x" * 100)

        summary = drop_day_master(day_dir, date="2026-02-21", ffprobe_fn=_ok_probe)

        assert master.exists()
        assert summary["dropped"] is False

    def test_keeps_master_when_no_short_finals(self, tmp_path):
        day_dir = self._day_dir(tmp_path)
        master = day_dir / "day_2026-02-21_master.mp4"
        master.write_bytes(b"x" * 50_000)
        (day_dir / "regular_lfs" / "lf_01.mp4").write_bytes(b"x" * 200)

        summary = drop_day_master(day_dir, date="2026-02-21", ffprobe_fn=_ok_probe)

        assert master.exists()
        assert summary["dropped"] is False

    def test_keeps_master_when_finals_are_empty(self, tmp_path):
        day_dir = self._day_dir(tmp_path)
        master = day_dir / "day_2026-02-21_master.mp4"
        master.write_bytes(b"x" * 50_000)
        # Zero-byte finals (mid-write stubs) should not count
        (day_dir / "regular_lfs" / "lf_01.mp4").write_bytes(b"")
        (day_dir / "shorts" / "short_01.mp4").write_bytes(b"")

        summary = drop_day_master(day_dir, date="2026-02-21", ffprobe_fn=_ok_probe)

        assert master.exists()
        assert summary["dropped"] is False

    def test_preserves_color_cache(self, tmp_path):
        """``day_*_color_cache.json`` must survive a master drop — resume
        fingerprint check reads it to avoid re-analysing colour from scratch."""
        day_dir = self._day_dir(tmp_path)
        master = day_dir / "day_2026-02-21_master.mp4"
        cache = day_dir / "day_2026-02-21_color_cache.json"
        master.write_bytes(b"x" * 50_000)
        cache.write_text('{"filter_str": "..."}')
        (day_dir / "regular_lfs" / "lf_01.mp4").write_bytes(b"x" * 100)
        (day_dir / "shorts" / "short_01.mp4").write_bytes(b"x" * 100)

        drop_day_master(day_dir, date="2026-02-21", ffprobe_fn=_ok_probe)

        assert not master.exists()
        assert cache.exists()

    def test_missing_master_is_noop(self, tmp_path):
        day_dir = self._day_dir(tmp_path)
        (day_dir / "regular_lfs" / "lf_01.mp4").write_bytes(b"x" * 100)
        (day_dir / "shorts" / "short_01.mp4").write_bytes(b"x" * 100)

        summary = drop_day_master(day_dir, date="2026-02-21", ffprobe_fn=_ok_probe)

        assert summary["dropped"] is False
        assert summary["bytes_freed"] == 0

    def test_missing_day_dir_is_noop(self, tmp_path):
        summary = drop_day_master(tmp_path / "nope", date="2026-02-21")
        # The summary contract includes the ``integrity`` key.
        assert summary["dropped"] is False
        assert summary["bytes_freed"] == 0
        assert summary["integrity"] == {}

    def test_keeps_master_until_all_superlong_parts_final(self, tmp_path):
        """Opt-in stream: master survives until every planned part is final."""
        day_dir = self._day_dir(tmp_path)
        master = day_dir / "day_2026-02-21_master.mp4"
        master.write_bytes(b"x" * 50_000)
        (day_dir / "regular_lfs" / "lf_01.mp4").write_bytes(b"x" * 200)
        (day_dir / "shorts" / "short_01.mp4").write_bytes(b"x" * 100)
        sl_dir = day_dir / "superlong"
        sl_dir.mkdir()
        (sl_dir / "sl_01.mp4").write_bytes(b"x" * 300)  # only 1 of 2 planned
        kwargs = dict(date="2026-02-21", ffprobe_fn=_ok_probe, expected_superlong_parts=2)

        pending = drop_day_master(day_dir, **kwargs)
        assert (master.exists(), pending["dropped"]) == (True, False)

        (sl_dir / "sl_02.mp4").write_bytes(b"x" * 300)
        complete = drop_day_master(day_dir, **kwargs)
        assert (master.exists(), complete["dropped"]) == (False, True)

    def test_idempotent(self, tmp_path):
        day_dir = self._day_dir(tmp_path)
        master = day_dir / "day_2026-02-21_master.mp4"
        master.write_bytes(b"x" * 10_000)
        (day_dir / "regular_lfs" / "lf_01.mp4").write_bytes(b"x" * 100)
        (day_dir / "shorts" / "short_01.mp4").write_bytes(b"x" * 100)

        first = drop_day_master(day_dir, date="2026-02-21", ffprobe_fn=_ok_probe)
        second = drop_day_master(day_dir, date="2026-02-21", ffprobe_fn=_ok_probe)

        assert first["dropped"] is True
        assert second["dropped"] is False
        assert second["bytes_freed"] == 0


class TestVerifyShortsIntegrity:
    """Integrity gate that protects ``drop_day_master``."""

    def _day_dir(self, tmp_path: Path) -> Path:
        day_dir = tmp_path / "day_2026-02-21"
        (day_dir / "shorts").mkdir(parents=True)
        return day_dir

    def test_passes_when_all_shorts_exist_and_probe_clean(self, tmp_path):
        day_dir = self._day_dir(tmp_path)
        for i in range(1, 4):
            (day_dir / "shorts" / f"short_{i:02d}.mp4").write_bytes(b"x" * 200)

        result = verify_shorts_integrity(day_dir, expected_count=3, ffprobe_fn=_ok_probe)

        assert result["ok"] is True
        assert result["missing"] == []
        assert result["bad"] == []
        assert result["count_actual"] == 3

    def test_fails_when_a_short_is_missing(self, tmp_path):
        day_dir = self._day_dir(tmp_path)
        # 2 of 3 expected
        (day_dir / "shorts" / "short_01.mp4").write_bytes(b"x" * 100)
        (day_dir / "shorts" / "short_03.mp4").write_bytes(b"x" * 100)

        result = verify_shorts_integrity(day_dir, expected_count=3, ffprobe_fn=_ok_probe)

        assert result["ok"] is False
        assert "short_02.mp4" in result["missing"]
        assert result["count_actual"] == 2
        assert "missing" in result["reason"]

    def test_fails_when_short_ffprobe_invalid(self, tmp_path):
        day_dir = self._day_dir(tmp_path)
        (day_dir / "shorts" / "short_01.mp4").write_bytes(b"x" * 100)
        (day_dir / "shorts" / "short_02.mp4").write_bytes(b"x" * 100)

        def bad_probe(path):
            if path.name == "short_02.mp4":
                return _FakeProbeResult(valid=False, duration=None, reason="no moov")
            return _FakeProbeResult(valid=True, duration=10.0)

        result = verify_shorts_integrity(day_dir, expected_count=2, ffprobe_fn=bad_probe)

        assert result["ok"] is False
        bad_names = [b["name"] for b in result["bad"]]
        assert "short_02.mp4" in bad_names
        assert "ffprobe" in result["reason"]

    def test_fails_when_short_duration_is_zero(self, tmp_path):
        day_dir = self._day_dir(tmp_path)
        (day_dir / "shorts" / "short_01.mp4").write_bytes(b"x" * 100)

        def zero_dur(_path):
            return _FakeProbeResult(valid=True, duration=0.0)

        result = verify_shorts_integrity(day_dir, expected_count=1, ffprobe_fn=zero_dur)

        assert result["ok"] is False
        assert any(b["name"] == "short_01.mp4" for b in result["bad"])

    def test_fails_on_count_mismatch_one_below_expected(self, tmp_path):
        """13 of 14 expected → fail (the canonical scenario)."""
        day_dir = self._day_dir(tmp_path)
        for i in range(1, 14):  # 1..13
            (day_dir / "shorts" / f"short_{i:02d}.mp4").write_bytes(b"x" * 100)

        result = verify_shorts_integrity(day_dir, expected_count=14, ffprobe_fn=_ok_probe)

        assert result["ok"] is False
        assert result["count_actual"] == 13
        assert result["count_expected"] == 14
        assert "short_14.mp4" in result["missing"]

    def test_default_probe_does_not_require_audio(self, tmp_path, monkeypatch):
        """Shorts are video-only by design (`-an`); the default probe must not
        flag a real audio-less short as bad (that would block `drop_day_master`
        for every day)."""
        day_dir = self._day_dir(tmp_path)
        (day_dir / "shorts" / "short_01.mp4").write_bytes(b"x" * 100)

        calls = []

        def spy_validate_video(
            video_path, expected_duration=None, duration_tolerance=0.05, require_audio=True
        ):
            calls.append(require_audio)
            return _FakeProbeResult(valid=True, duration=10.0)

        monkeypatch.setattr("gomoveshift.pipeline.ffmpeg_runner.validate_video", spy_validate_video)

        result = verify_shorts_integrity(day_dir, expected_count=1)

        assert calls == [False]
        assert result["ok"] is True

    def test_returns_named_missing_and_bad_lists(self, tmp_path):
        """Failure result must name offending shorts so the operator can act."""
        day_dir = self._day_dir(tmp_path)
        (day_dir / "shorts" / "short_01.mp4").write_bytes(b"x" * 100)
        # short_02 missing entirely; short_01 will fail probe

        def bad_probe(_path):
            return _FakeProbeResult(valid=False, duration=None, reason="corrupt")

        result = verify_shorts_integrity(day_dir, expected_count=2, ffprobe_fn=bad_probe)

        assert result["ok"] is False
        # Both kinds of failure reported by name
        assert "short_02.mp4" in result["missing"]
        # bad list documents probe-failures
        # (when missing is non-empty we short-circuit reason to missing, but
        # the bad list is still populated for the operator to see)
        assert isinstance(result["bad"], list)


class TestDropDayMasterIntegrityGate:
    """Master must NOT be deleted when shorts integrity fails."""

    def _day_dir_with_finals(self, tmp_path: Path, *, n_shorts: int = 2) -> Path:
        day_dir = tmp_path / "day_2026-02-21"
        (day_dir / "regular_lfs").mkdir(parents=True)
        (day_dir / "shorts").mkdir(parents=True)
        master = day_dir / "day_2026-02-21_master.mp4"
        master.write_bytes(b"M" * 50_000)
        (day_dir / "regular_lfs" / "lf_01.mp4").write_bytes(b"x" * 200)
        for i in range(1, n_shorts + 1):
            (day_dir / "shorts" / f"short_{i:02d}.mp4").write_bytes(b"x" * 100)
        return day_dir

    def test_drops_master_when_integrity_passes(self, tmp_path):
        day_dir = self._day_dir_with_finals(tmp_path, n_shorts=3)
        master = day_dir / "day_2026-02-21_master.mp4"

        summary = drop_day_master(
            day_dir,
            date="2026-02-21",
            expected_shorts_count=3,
            ffprobe_fn=_ok_probe,
        )

        assert not master.exists()
        assert summary["dropped"] is True
        assert summary["integrity"]["ok"] is True

    def test_preserves_master_when_a_short_is_missing(self, tmp_path, caplog):
        day_dir = self._day_dir_with_finals(tmp_path, n_shorts=2)
        master = day_dir / "day_2026-02-21_master.mp4"

        with caplog.at_level("ERROR"):
            summary = drop_day_master(
                day_dir,
                date="2026-02-21",
                expected_shorts_count=3,  # one missing
                ffprobe_fn=_ok_probe,
            )

        assert master.exists(), "master must be preserved on integrity failure"
        assert summary["dropped"] is False
        assert summary["integrity"]["ok"] is False
        # ERROR log must mention master path so operator can recover
        assert any(str(master) in rec.message for rec in caplog.records)

    def test_preserves_master_when_short_ffprobe_fails(self, tmp_path, caplog):
        day_dir = self._day_dir_with_finals(tmp_path, n_shorts=2)
        master = day_dir / "day_2026-02-21_master.mp4"

        def bad_probe(_path):
            return _FakeProbeResult(valid=False, duration=None, reason="no moov atom")

        with caplog.at_level("ERROR"):
            summary = drop_day_master(
                day_dir,
                date="2026-02-21",
                expected_shorts_count=2,
                ffprobe_fn=bad_probe,
            )

        assert master.exists()
        assert summary["dropped"] is False
        assert any(str(master) in rec.message for rec in caplog.records)

    def test_preserves_master_on_count_mismatch_thirteen_of_fourteen(self, tmp_path, caplog):
        """Canonical scenario: 13 of 14 shorts → master stays."""
        day_dir = self._day_dir_with_finals(tmp_path, n_shorts=13)
        master = day_dir / "day_2026-02-21_master.mp4"

        with caplog.at_level("ERROR"):
            summary = drop_day_master(
                day_dir,
                date="2026-02-21",
                expected_shorts_count=14,
                ffprobe_fn=_ok_probe,
            )

        assert master.exists()
        assert summary["dropped"] is False
        assert summary["integrity"]["count_actual"] == 13
        assert summary["integrity"]["count_expected"] == 14


class TestDDriveCopyVerificationTautology:
    """Live data-loss defect: without a distinct ``d_drive_base``, the D:
    drive verification gate compared ``project_dir`` against itself
    (``d_drive_base`` defaulted to ``output_base``) and always passed,
    queueing ``_FINAL.mp4`` for deletion though no separate D: copy exists.
    """

    def _completed_project_with_final(self, tmp_path, name="testproject"):
        import json

        project_dir = tmp_path / "output" / name
        stages_dir = project_dir / "_stages"
        stages_dir.mkdir(parents=True)
        final = project_dir / f"{name}_FINAL.mp4"
        final.write_bytes(b"x" * 1000)
        state = {
            "stages_completed": ["csv_generation"],
            "source_video": "/data/input/Test.mp4",
        }
        (stages_dir / "pipeline_state.json").write_text(json.dumps(state))
        return project_dir, final

    def test_no_d_drive_base_never_queues_final_for_deletion(self, tmp_path, monkeypatch):
        """Constructed exactly as ``cli/handlers/cleanup.py`` does: no
        ``d_drive_base`` is passed, so it and ``output_base`` both resolve
        via ``gms_output_dir()``. Must not queue the deliverable."""
        monkeypatch.setenv("GMS_OUTPUT_DIR", str(tmp_path / "output"))
        monkeypatch.setenv("GMS_STAGING_DIR", str(tmp_path / "staging"))
        _project_dir, final = self._completed_project_with_final(tmp_path)

        from gomoveshift.paths import gms_output_dir, gms_staging_dir

        cleanup = PipelineCleanup(
            output_base=gms_output_dir(),
            staging_dir=gms_staging_dir(),
        )

        projects = cleanup.discover_projects()
        plan = cleanup.plan_cleanup(projects, min_age_days=0, ffprobe_fn=_ok_probe)

        deleted_paths = [f["path"] for f in plan.files]
        assert str(final) not in deleted_paths, (
            "Deliverable must never be queued for deletion when no distinct "
            "D: drive location is configured — self-comparison is not verification."
        )

    def test_distinct_d_drive_with_matching_size_still_eligible(self, tmp_path):
        """A genuinely distinct D: copy with a matching size is still a valid gate pass."""
        _project_dir, final = self._completed_project_with_final(tmp_path)
        d_dir = tmp_path / "d_output" / "testproject"
        d_dir.mkdir(parents=True)
        (d_dir / "testproject_FINAL.mp4").write_bytes(b"x" * 1000)

        cleanup = PipelineCleanup(
            output_base=tmp_path / "output",
            staging_dir=tmp_path / "staging",
            d_drive_base=tmp_path / "d_output",
        )

        projects = cleanup.discover_projects()
        plan = cleanup.plan_cleanup(projects, min_age_days=0, ffprobe_fn=_ok_probe)

        deleted_paths = [f["path"] for f in plan.files]
        assert str(final) in deleted_paths

    def test_distinct_d_drive_with_different_size_not_eligible(self, tmp_path):
        """A distinct D: copy whose size differs must not verify."""
        _project_dir, final = self._completed_project_with_final(tmp_path)
        d_dir = tmp_path / "d_output" / "testproject"
        d_dir.mkdir(parents=True)
        (d_dir / "testproject_FINAL.mp4").write_bytes(b"x" * 500)  # size mismatch

        cleanup = PipelineCleanup(
            output_base=tmp_path / "output",
            staging_dir=tmp_path / "staging",
            d_drive_base=tmp_path / "d_output",
        )

        projects = cleanup.discover_projects()
        plan = cleanup.plan_cleanup(projects, min_age_days=0, ffprobe_fn=_ok_probe)

        deleted_paths = [f["path"] for f in plan.files]
        assert str(final) not in deleted_paths


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
