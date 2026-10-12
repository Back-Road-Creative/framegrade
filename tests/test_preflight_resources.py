"""Tests for preflight resource checks: RAM, tool presence, staging disk, cleanup.

These extend the disk/OST/input-file checks in test_preflight.py with the
resource and resilience gates added for reboot-recovery hardening.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest


_GB = 1_073_741_824


class TestCheckRam:
    def test_sufficient_ram_passes(self):
        from gomoveshift.pipeline.preflight import check_ram

        mem = type("Mem", (), {"available": 8 * _GB})()
        with patch("gomoveshift.pipeline.preflight._virtual_memory", return_value=mem):
            check_ram(min_free_gb=2)

    def test_insufficient_ram_raises(self):
        from gomoveshift.pipeline.preflight import check_ram

        mem = type("Mem", (), {"available": 512 * 1024 * 1024})()  # 0.5 GB
        with patch("gomoveshift.pipeline.preflight._virtual_memory", return_value=mem):
            with pytest.raises(RuntimeError, match="(?i)ram|memory"):
                check_ram(min_free_gb=2)


class TestCheckTools:
    def test_all_tools_present_passes(self):
        from gomoveshift.pipeline.preflight import check_tools

        with patch(
            "gomoveshift.pipeline.preflight.shutil.which",
            side_effect=lambda t: f"/usr/bin/{t}",
        ):
            check_tools(["ffmpeg", "ffprobe"])

    def test_missing_tool_raises(self):
        from gomoveshift.pipeline.preflight import check_tools

        def which(t):
            return None if t == "ffprobe" else f"/usr/bin/{t}"

        with patch("gomoveshift.pipeline.preflight.shutil.which", side_effect=which):
            with pytest.raises(RuntimeError, match="ffprobe"):
                check_tools(["ffmpeg", "ffprobe"])


class TestCheckOstFont:
    """The OST burn font must exist before any encode — a missing font
    silently ships title-less long-forms via the apply_ost stream-copy."""

    def test_present_font_passes(self, tmp_path):
        from gomoveshift.pipeline.preflight import check_ost_font

        font = tmp_path / "DejaVuSans-Bold.ttf"
        font.write_bytes(b"\x00")  # contents irrelevant; only existence is gated
        with patch("gomoveshift.pipeline.ffmpeg_filters.FONT_PATH", str(font)):
            check_ost_font()

    def test_missing_font_raises(self, tmp_path):
        from gomoveshift.pipeline.preflight import check_ost_font

        missing = tmp_path / "nope" / "DejaVuSans-Bold.ttf"
        with patch("gomoveshift.pipeline.ffmpeg_filters.FONT_PATH", str(missing)):
            with pytest.raises(RuntimeError, match="(?i)font not found"):
                check_ost_font()


class TestRunPreflightStaging:
    """run_preflight accepts staging_dir and checks its free space."""

    def test_staging_short_raises(self, tmp_path):
        from gomoveshift.pipeline.preflight import run_preflight

        src = tmp_path / "source"
        src.mkdir()
        (src / "GX010001.MP4").touch()
        os.truncate((src / "GX010001.MP4"), 200 * 1024 * 1024)  # sparse: 200 MB st_size, no disk

        def fake_usage(path):
            if "staging" in str(path):
                return type("U", (), {"free": 1 * _GB})()
            return type("U", (), {"free": 500 * _GB})()

        with patch("shutil.disk_usage", side_effect=fake_usage):
            with pytest.raises(RuntimeError, match="(?i)staging|disk"):
                run_preflight(
                    source_dir=src,
                    output_dir=tmp_path / "out",
                    staging_dir=tmp_path / "staging",
                    min_staging_gb=50,
                )

    def test_staging_dir_created_if_missing(self, tmp_path):
        from gomoveshift.pipeline.preflight import run_preflight

        src = tmp_path / "source"
        src.mkdir()
        (src / "GX010001.MP4").touch()
        os.truncate((src / "GX010001.MP4"), 200 * 1024 * 1024)  # sparse: 200 MB st_size, no disk

        staging = tmp_path / "staging-new"
        usage = type("U", (), {"free": 500 * _GB})()
        with patch("shutil.disk_usage", return_value=usage):
            run_preflight(
                source_dir=src,
                output_dir=tmp_path / "out",
                staging_dir=staging,
                min_staging_gb=1,
            )
        assert staging.is_dir()


class TestRunPreflightRamTools:
    def test_ram_check_integrated(self, tmp_path):
        from gomoveshift.pipeline.preflight import run_preflight

        src = tmp_path / "source"
        src.mkdir()
        (src / "GX010001.MP4").touch()
        os.truncate((src / "GX010001.MP4"), 200 * 1024 * 1024)  # sparse: 200 MB st_size, no disk

        usage = type("U", (), {"free": 500 * _GB})()
        mem = type("Mem", (), {"available": 256 * 1024 * 1024})()  # 0.25 GB
        with (
            patch("shutil.disk_usage", return_value=usage),
            patch("gomoveshift.pipeline.preflight._virtual_memory", return_value=mem),
        ):
            with pytest.raises(RuntimeError, match="(?i)ram|memory"):
                run_preflight(
                    source_dir=src,
                    output_dir=tmp_path / "out",
                    min_ram_gb=2,
                )

    def test_tool_check_integrated(self, tmp_path):
        from gomoveshift.pipeline.preflight import run_preflight

        src = tmp_path / "source"
        src.mkdir()
        (src / "GX010001.MP4").touch()
        os.truncate((src / "GX010001.MP4"), 200 * 1024 * 1024)  # sparse: 200 MB st_size, no disk

        usage = type("U", (), {"free": 500 * _GB})()
        with (
            patch("shutil.disk_usage", return_value=usage),
            patch("gomoveshift.pipeline.preflight.shutil.which", return_value=None),
        ):
            with pytest.raises(RuntimeError, match="ffmpeg"):
                run_preflight(
                    source_dir=src,
                    output_dir=tmp_path / "out",
                    require_tools=("ffmpeg",),
                )


class TestCleanupStaleStaging:
    def _age_dir(self, p: Path, days: int) -> None:
        ts = time.time() - (days * 86400)
        os.utime(p, (ts, ts))

    def test_removes_old_project_dirs(self, tmp_path):
        from gomoveshift.pipeline.preflight import cleanup_stale_staging

        old = tmp_path / "old_proj"
        (old / "_stages").mkdir(parents=True)
        (old / "_stages" / "file").write_text("x")
        recent = tmp_path / "recent_proj"
        (recent / "_stages").mkdir(parents=True)

        self._age_dir(old, 30)
        self._age_dir(old / "_stages", 30)

        removed = cleanup_stale_staging(tmp_path, max_age_days=14)
        assert old in removed
        assert recent not in removed
        assert not old.exists()
        assert recent.exists()

    def test_empty_base_returns_empty(self, tmp_path):
        from gomoveshift.pipeline.preflight import cleanup_stale_staging

        assert cleanup_stale_staging(tmp_path / "missing", max_age_days=14) == []

    def test_protects_active_dir(self, tmp_path):
        from gomoveshift.pipeline.preflight import cleanup_stale_staging

        active = tmp_path / "active_proj"
        (active / "_stages").mkdir(parents=True)
        self._age_dir(active, 30)
        self._age_dir(active / "_stages", 30)

        removed = cleanup_stale_staging(tmp_path, max_age_days=14, protect={active})
        assert active.exists()
        assert active not in removed

    def test_finished_output_dir_survives_sweep(self, tmp_path):
        """Shape (a): ext4 --output-dir makes the sweep base == output base, so a
        finished sibling PROJECT's output dir (FINAL.mp4 + youtube_uploads.json
        dedupe sidecar) sits directly under the sweep base. It must NOT be
        wholesale-deleted 14 days after last write."""
        from gomoveshift.pipeline.preflight import cleanup_stale_staging

        finished = tmp_path / "finished_proj"
        finished.mkdir()
        (finished / "finished_proj_FINAL.mp4").write_bytes(b"\x00" * 16)
        (finished / "youtube_uploads.json").write_text("{}")
        self._age_dir(finished, 30)

        removed = cleanup_stale_staging(tmp_path, max_age_days=14)

        assert finished not in removed
        assert finished.exists()
        assert (finished / "finished_proj_FINAL.mp4").exists()
        assert (finished / "youtube_uploads.json").exists()

    def test_paused_project_with_masters_survives_sweep(self, tmp_path):
        """Shape (b): a paused-at-gate sibling project started >14 days ago (its
        project dir mtime reflects project START, since ``_stages`` is created
        once) must NOT have its masters/finals/state destroyed by an unrelated
        drive's preflight — that would bypass the master-deletion integrity gate."""
        from gomoveshift.pipeline.preflight import cleanup_stale_staging

        paused = tmp_path / "paused_proj"
        day = paused / "_stages" / "day_2026-01-01"
        day.mkdir(parents=True)
        (day / "day_2026-01-01_master.mp4").write_bytes(b"\x00" * 16)
        (paused / "_stages" / "pipeline_state.json").write_text("{}")
        # Project START (dir mtime) is >14 days ago; the live master inside is
        # recent, but the dir-mtime age check would still flag it for deletion.
        self._age_dir(paused, 30)
        self._age_dir(paused / "_stages", 30)

        removed = cleanup_stale_staging(tmp_path, max_age_days=14)

        assert paused not in removed
        assert paused.exists()
        assert (day / "day_2026-01-01_master.mp4").exists()
        assert (paused / "_stages" / "pipeline_state.json").exists()

    def test_rmtree_failure_on_one_dir_still_removes_others_and_surfaces(
        self, tmp_path, monkeypatch, caplog
    ):
        """A persistently-stuck dir must not silently mask the others.

        When ``rmtree`` fails for one stale dir, the rest are still reclaimed
        AND the failure is surfaced (aggregated end-of-function WARNING) so a
        stuck dir is visible across runs instead of disappearing into a
        per-iteration log line.
        """
        from gomoveshift.pipeline import preflight
        from gomoveshift.pipeline.preflight import cleanup_stale_staging

        stuck = tmp_path / "stuck_proj"
        (stuck / "_stages").mkdir(parents=True)
        good = tmp_path / "good_proj"
        (good / "_stages").mkdir(parents=True)
        for d in (stuck, good):
            self._age_dir(d / "_stages", 30)
            self._age_dir(d, 30)

        real_rmtree = preflight.shutil.rmtree

        def fake_rmtree(path, *args, **kwargs):
            if Path(path).resolve() == stuck.resolve():
                raise OSError("device busy")
            return real_rmtree(path, *args, **kwargs)

        monkeypatch.setattr(preflight.shutil, "rmtree", fake_rmtree)

        with caplog.at_level(logging.WARNING):
            removed = cleanup_stale_staging(tmp_path, max_age_days=14)

        # The healthy dir is still reclaimed; the stuck one is not.
        assert good in removed
        assert stuck not in removed
        assert not good.exists()
        assert stuck.exists()

        # The failure is surfaced in an aggregated end-of-function summary.
        summaries = [
            r.getMessage()
            for r in caplog.records
            if "could not be reclaimed" in r.getMessage().lower()
            or "could not reclaim" in r.getMessage().lower()
        ]
        assert summaries, "expected an aggregated rmtree-failure summary warning"
        assert "1" in summaries[-1]
        assert str(stuck) in summaries[-1]
