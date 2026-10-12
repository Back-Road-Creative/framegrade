"""Tests for photo artifact cleanup."""

import json
import time
from unittest.mock import MagicMock

import pytest
import yaml

from gomoveshift.pipeline.orchestrator_lock import (
    ConcurrentRunError,
    _acquire_run_lock,
    _release_run_lock,
)
from gomoveshift.pipeline.photos.cleanup import PhotoCleanup, PhotoCleanupPlan
from gomoveshift.pipeline.photos.orchestrator import PHOTO_RUN_LOCK_NAME, PhotoOrchestrator


class TestPhotoCleanupDiscover:
    def test_discover_finds_caption_weeks(self, tmp_path):
        caption_dir = tmp_path / "captions"
        caption_dir.mkdir()
        (caption_dir / "2026-02-09_captions.yaml").write_text("test")
        (caption_dir / "2026-02-16_captions.yaml").write_text("test")

        cleanup = PhotoCleanup(
            caption_dir=caption_dir,
            metricool_dir=tmp_path / "metricool",
            resized_dir=tmp_path / "resized",
        )
        batches = cleanup.discover_batches()
        assert batches == ["2026-02-09", "2026-02-16"]

    def test_discover_finds_csv_weeks(self, tmp_path):
        metricool_dir = tmp_path / "metricool"
        metricool_dir.mkdir()
        (metricool_dir / "photo_metricool_2026-02-09.csv").write_text("test")

        cleanup = PhotoCleanup(
            caption_dir=tmp_path / "captions",
            metricool_dir=metricool_dir,
            resized_dir=tmp_path / "resized",
        )
        batches = cleanup.discover_batches()
        assert batches == ["2026-02-09"]

    def test_discover_finds_resized_weeks(self, tmp_path):
        resized_dir = tmp_path / "resized"
        (resized_dir / "2026-02-09").mkdir(parents=True)

        cleanup = PhotoCleanup(
            caption_dir=tmp_path / "captions",
            metricool_dir=tmp_path / "metricool",
            resized_dir=resized_dir,
        )
        batches = cleanup.discover_batches()
        assert batches == ["2026-02-09"]

    def test_discover_union_of_all_sources(self, tmp_path):
        caption_dir = tmp_path / "captions"
        caption_dir.mkdir()
        (caption_dir / "2026-02-09_captions.yaml").write_text("test")

        metricool_dir = tmp_path / "metricool"
        metricool_dir.mkdir()
        (metricool_dir / "photo_metricool_2026-02-16.csv").write_text("test")

        resized_dir = tmp_path / "resized"
        (resized_dir / "2026-02-23").mkdir(parents=True)

        cleanup = PhotoCleanup(
            caption_dir=caption_dir,
            metricool_dir=metricool_dir,
            resized_dir=resized_dir,
        )
        batches = cleanup.discover_batches()
        assert batches == ["2026-02-09", "2026-02-16", "2026-02-23"]

    def test_discover_empty_dirs(self, tmp_path):
        cleanup = PhotoCleanup(
            caption_dir=tmp_path / "captions",
            metricool_dir=tmp_path / "metricool",
            resized_dir=tmp_path / "resized",
        )
        assert cleanup.discover_batches() == []


class TestPhotoCleanupPlan:
    @pytest.fixture
    def old_artifact_dirs(self, tmp_path):
        """Create artifact dirs with files old enough to clean."""
        caption_dir = tmp_path / "captions"
        caption_dir.mkdir()
        metricool_dir = tmp_path / "metricool"
        metricool_dir.mkdir()
        resized_dir = tmp_path / "resized"
        resized_dir.mkdir()

        return caption_dir, metricool_dir, resized_dir

    def _make_old(self, path, days=15):
        """Set file modification time to N days ago."""
        old_time = time.time() - (days * 86400)
        import os

        os.utime(path, (old_time, old_time))

    def test_plan_skips_recent_files(self, old_artifact_dirs):
        caption_dir, metricool_dir, resized_dir = old_artifact_dirs

        # Create a recent caption file (< 14 days)
        caption = caption_dir / "2026-02-16_captions.yaml"
        caption.write_text(yaml.dump({"reviewed": True, "captions": []}))

        cleanup = PhotoCleanup(
            caption_dir=caption_dir,
            metricool_dir=metricool_dir,
            resized_dir=resized_dir,
        )
        plan = cleanup.plan_cleanup(["2026-02-16"])
        assert len(plan.files) == 0
        assert any("Too recent" in s["reason"] for s in plan.skipped)

    def test_plan_skips_unreviewed_captions(self, old_artifact_dirs):
        caption_dir, metricool_dir, resized_dir = old_artifact_dirs

        caption = caption_dir / "2026-02-01_captions.yaml"
        caption.write_text(yaml.dump({"reviewed": False, "captions": []}))
        self._make_old(caption)

        cleanup = PhotoCleanup(
            caption_dir=caption_dir,
            metricool_dir=metricool_dir,
            resized_dir=resized_dir,
        )
        plan = cleanup.plan_cleanup(["2026-02-01"])
        assert len(plan.files) == 0
        assert any("not reviewed" in s["reason"] for s in plan.skipped)

    def test_plan_skips_reviewed_captions_without_csv(self, old_artifact_dirs):
        """A reviewed batch that never shipped a CSV is unpublished — keep it.

        Photos are marked posted at selection, so deleting the caption YAML of
        a never-published batch loses the batch unrecoverably.
        """
        caption_dir, metricool_dir, resized_dir = old_artifact_dirs

        caption = caption_dir / "2026-02-01_captions.yaml"
        caption.write_text(yaml.dump({"reviewed": True, "captions": []}))
        self._make_old(caption)

        cleanup = PhotoCleanup(
            caption_dir=caption_dir,
            metricool_dir=metricool_dir,
            resized_dir=resized_dir,
        )
        plan = cleanup.plan_cleanup(["2026-02-01"])
        assert len(plan.files) == 0
        assert any("publish not complete" in s["reason"] for s in plan.skipped)

    def test_plan_includes_reviewed_old_captions(self, old_artifact_dirs):
        caption_dir, metricool_dir, resized_dir = old_artifact_dirs

        caption = caption_dir / "2026-02-01_captions.yaml"
        caption.write_text(yaml.dump({"reviewed": True, "captions": []}))
        self._make_old(caption)
        # Publish evidence: the week shipped a Metricool CSV (recent, so the
        # CSV itself is skipped and only the caption YAML is planned).
        (metricool_dir / "photo_metricool_2026-02-01.csv").write_text("csv")

        cleanup = PhotoCleanup(
            caption_dir=caption_dir,
            metricool_dir=metricool_dir,
            resized_dir=resized_dir,
        )
        plan = cleanup.plan_cleanup(["2026-02-01"])
        assert len(plan.files) == 1
        assert "Caption YAML" in plan.files[0]["reason"]

    def test_plan_includes_old_csv(self, old_artifact_dirs):
        caption_dir, metricool_dir, resized_dir = old_artifact_dirs

        csv = metricool_dir / "photo_metricool_2026-02-01.csv"
        csv.write_text("header\ndata")
        self._make_old(csv)

        cleanup = PhotoCleanup(
            caption_dir=caption_dir,
            metricool_dir=metricool_dir,
            resized_dir=resized_dir,
        )
        plan = cleanup.plan_cleanup(["2026-02-01"])
        assert len(plan.files) == 1
        assert "Metricool CSV" in plan.files[0]["reason"]

    def test_plan_resized_requires_csv(self, old_artifact_dirs):
        caption_dir, metricool_dir, resized_dir = old_artifact_dirs

        week_dir = resized_dir / "2026-02-01"
        week_dir.mkdir()
        photo = week_dir / "photo.jpg"
        photo.write_bytes(b"fake image data")
        self._make_old(week_dir)
        self._make_old(photo)

        # No CSV exists
        cleanup = PhotoCleanup(
            caption_dir=caption_dir,
            metricool_dir=metricool_dir,
            resized_dir=resized_dir,
        )
        plan = cleanup.plan_cleanup(["2026-02-01"])
        assert len(plan.files) == 0
        assert any("No CSV" in s["reason"] for s in plan.skipped)

    def test_plan_resized_with_csv(self, old_artifact_dirs):
        caption_dir, metricool_dir, resized_dir = old_artifact_dirs

        # Create CSV so gate passes
        csv = metricool_dir / "photo_metricool_2026-02-01.csv"
        csv.write_text("data")

        week_dir = resized_dir / "2026-02-01"
        week_dir.mkdir()
        photo = week_dir / "photo.jpg"
        photo.write_bytes(b"fake image data")
        self._make_old(week_dir)
        self._make_old(photo)

        cleanup = PhotoCleanup(
            caption_dir=caption_dir,
            metricool_dir=metricool_dir,
            resized_dir=resized_dir,
        )
        plan = cleanup.plan_cleanup(["2026-02-01"])
        # Should include the resized photo (CSV also included if old enough)
        resized_files = [f for f in plan.files if "Resized" in f["reason"]]
        assert len(resized_files) == 1

    def test_plan_custom_min_age(self, old_artifact_dirs):
        caption_dir, metricool_dir, resized_dir = old_artifact_dirs

        caption = caption_dir / "2026-02-01_captions.yaml"
        caption.write_text(yaml.dump({"reviewed": True, "captions": []}))
        self._make_old(caption, days=10)
        (metricool_dir / "photo_metricool_2026-02-01.csv").write_text("csv")

        cleanup = PhotoCleanup(
            caption_dir=caption_dir,
            metricool_dir=metricool_dir,
            resized_dir=resized_dir,
        )
        # Default 14 days — should skip
        plan = cleanup.plan_cleanup(["2026-02-01"], min_age_days=14)
        assert len(plan.files) == 0

        # Custom 7 days — should include
        plan = cleanup.plan_cleanup(["2026-02-01"], min_age_days=7)
        assert len(plan.files) == 1


class TestPhotoCleanupExecute:
    def test_execute_deletes_files(self, tmp_path):
        caption_dir = tmp_path / "captions"
        caption_dir.mkdir()

        target = caption_dir / "2026-02-01_captions.yaml"
        target.write_text("test data")

        plan = PhotoCleanupPlan(
            files=[{"path": str(target), "size": target.stat().st_size, "reason": "test"}],
            skipped=[],
        )

        cleanup = PhotoCleanup(
            caption_dir=caption_dir,
            metricool_dir=tmp_path / "metricool",
            resized_dir=tmp_path / "resized",
        )
        deleted = cleanup.execute_plan(plan)
        assert deleted == 1
        assert not target.exists()

    def test_execute_removes_empty_resized_dirs(self, tmp_path):
        resized_dir = tmp_path / "resized"
        week_dir = resized_dir / "2026-02-01"
        week_dir.mkdir(parents=True)
        photo = week_dir / "photo.jpg"
        photo.write_bytes(b"data")

        plan = PhotoCleanupPlan(
            files=[{"path": str(photo), "size": photo.stat().st_size, "reason": "test"}],
            skipped=[],
        )

        cleanup = PhotoCleanup(
            caption_dir=tmp_path / "captions",
            metricool_dir=tmp_path / "metricool",
            resized_dir=resized_dir,
        )
        cleanup.execute_plan(plan)
        assert not week_dir.exists()

    def test_execute_empty_plan(self, tmp_path):
        cleanup = PhotoCleanup(
            caption_dir=tmp_path / "captions",
            metricool_dir=tmp_path / "metricool",
            resized_dir=tmp_path / "resized",
        )
        plan = PhotoCleanupPlan()
        deleted = cleanup.execute_plan(plan)
        assert deleted == 0

    def test_execute_handles_missing_file(self, tmp_path):
        plan = PhotoCleanupPlan(
            files=[{"path": str(tmp_path / "gone.txt"), "size": 100, "reason": "test"}],
            skipped=[],
        )
        cleanup = PhotoCleanup(
            caption_dir=tmp_path / "captions",
            metricool_dir=tmp_path / "metricool",
            resized_dir=tmp_path / "resized",
        )
        deleted = cleanup.execute_plan(plan)
        assert deleted == 0


class TestDriveManifestCleanup:
    def _make_old(self, path, days=15):
        old_time = time.time() - (days * 86400)
        import os

        os.utime(path, (old_time, old_time))

    def test_completed_manifest_cleaned(self, tmp_path):
        # The cleanup scans caption_dir.parent for manifests
        output_dir = tmp_path / "output"
        caption_dir = output_dir / "photo_captions"
        caption_dir.mkdir(parents=True)

        manifest = output_dir / "photo_gdrive_manifest.json"
        manifest.write_text(json.dumps({"completed": True, "files": {}}))
        self._make_old(manifest)

        cleanup = PhotoCleanup(
            caption_dir=caption_dir,
            metricool_dir=output_dir / "photo_metricool",
            resized_dir=output_dir / "photo_resized",
        )
        plan = cleanup.plan_cleanup([])
        manifest_files = [f for f in plan.files if "manifest" in f["reason"].lower()]
        assert len(manifest_files) == 1

    def test_incomplete_manifest_skipped(self, tmp_path):
        output_dir = tmp_path / "output"
        caption_dir = output_dir / "photo_captions"
        caption_dir.mkdir(parents=True)

        manifest = output_dir / "photo_gdrive_manifest.json"
        manifest.write_text(json.dumps({"completed": False, "files": {}}))
        self._make_old(manifest)

        cleanup = PhotoCleanup(
            caption_dir=caption_dir,
            metricool_dir=output_dir / "photo_metricool",
            resized_dir=output_dir / "photo_resized",
        )
        plan = cleanup.plan_cleanup([])
        assert len(plan.files) == 0
        assert any("not completed" in s["reason"].lower() for s in plan.skipped)

    def _resized_week_with_manifest(self, tmp_path, completed):
        """Old resized week dir (CSV shipped) holding a photo + Drive manifest."""
        output_dir = tmp_path / "output"
        caption_dir = output_dir / "photo_captions"
        caption_dir.mkdir(parents=True)
        metricool_dir = output_dir / "photo_metricool"
        metricool_dir.mkdir()
        (metricool_dir / "photo_metricool_2026-02-01.csv").write_text("data")

        week_dir = output_dir / "photo_resized" / "2026-02-01"
        week_dir.mkdir(parents=True)
        photo = week_dir / "photo.jpg"
        photo.write_bytes(b"img")
        manifest = week_dir / "photo_gdrive_manifest.json"
        manifest.write_text(json.dumps({"completed": completed, "files": {}}))
        for p in (week_dir, photo, manifest):
            self._make_old(p)

        cleanup = PhotoCleanup(
            caption_dir=caption_dir,
            metricool_dir=metricool_dir,
            resized_dir=output_dir / "photo_resized",
        )
        return cleanup, photo, manifest

    def test_incomplete_manifest_in_resized_dir_survives(self, tmp_path):
        """The resized-intermediates sweep must not delete an incomplete Drive
        manifest on the CSV gate alone — it is the only resume record for an
        in-progress upload; the completed gate governs it on every path."""
        cleanup, photo, manifest = self._resized_week_with_manifest(tmp_path, completed=False)

        plan = cleanup.plan_cleanup(["2026-02-01"])
        planned = [f["path"] for f in plan.files]
        assert str(manifest) not in planned
        assert str(photo) in planned  # real intermediates still clean
        assert any(
            s["path"] == str(manifest) and "not completed" in s["reason"].lower()
            for s in plan.skipped
        )

    def test_completed_manifest_in_resized_dir_planned_once(self, tmp_path):
        cleanup, photo, manifest = self._resized_week_with_manifest(tmp_path, completed=True)

        plan = cleanup.plan_cleanup(["2026-02-01"])
        manifest_entries = [f for f in plan.files if f["path"] == str(manifest)]
        assert len(manifest_entries) == 1
        assert "manifest" in manifest_entries[0]["reason"].lower()


class TestCleanupRunLock:
    """Cleanup racing a live generate/publish could delete artifacts the run
    is mid-way through producing — it must hold the same photo-run lock."""

    def test_cleanup_refused_while_a_photo_run_holds_the_lock(self, tmp_path):
        orch = PhotoOrchestrator(site_root=tmp_path, output_dir=tmp_path)
        orch.cleanup_handler = MagicMock()

        fd, lock_path = _acquire_run_lock(tmp_path, tmp_path, lock_name=PHOTO_RUN_LOCK_NAME)
        try:
            with pytest.raises(ConcurrentRunError):
                orch.cleanup()
        finally:
            _release_run_lock(fd, lock_path)
        orch.cleanup_handler.discover_batches.assert_not_called()
        orch.cleanup_handler.execute_plan.assert_not_called()

    def test_cleanup_releases_the_lock(self, tmp_path):
        orch = PhotoOrchestrator(site_root=tmp_path, output_dir=tmp_path)
        orch.cleanup_handler = MagicMock()
        orch.cleanup_handler.plan_cleanup.return_value = PhotoCleanupPlan()

        orch.cleanup()
        assert not (tmp_path / PHOTO_RUN_LOCK_NAME).exists()
        orch.cleanup()  # a sequential second run is never blocked
        assert orch.cleanup_handler.plan_cleanup.call_count == 2
