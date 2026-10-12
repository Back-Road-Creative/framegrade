"""Tests for GoPro multi-clip ingestion pipeline."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from gomoveshift.pipeline.ingest import (
    Episode,
    GoProChapter,
    GoProRecording,
    IngestPlan,
    _haversine_km,
    create_ingest_plan,
    group_recordings_into_episodes,
    parse_gopro_filename,
)


def _cfg(tmp_path):
    """A real Config whose SEO storage (the content ledger's home) is under ``tmp_path``."""
    from gomoveshift.config import Config

    cfg = Config()
    cfg.seo.storage_path = str(tmp_path / "seo")
    Path(cfg.seo.storage_path).mkdir(exist_ok=True)
    return cfg


class TestParseGoProFilename:
    """Parse GoPro GXccNNNN.MP4 naming convention."""

    def test_standard_gopro_name(self):
        assert parse_gopro_filename("GX010001.MP4") == (1, 1)

    def test_multi_chapter(self):
        assert parse_gopro_filename("GX070009.MP4") == (7, 9)

    def test_hevc_prefix(self):
        """GH prefix is used by some GoPro models for HEVC."""
        assert parse_gopro_filename("GH010001.MP4") == (1, 1)

    def test_lowercase_extension(self):
        assert parse_gopro_filename("GX010001.mp4") == (1, 1)

    def test_non_gopro_file(self):
        assert parse_gopro_filename("video.mp4") is None

    def test_lrv_file(self):
        """Low-res proxy files should not match."""
        assert parse_gopro_filename("GL010001.LRV") is None

    def test_thm_file(self):
        """Thumbnail files should not match."""
        assert parse_gopro_filename("GX010001.THM") is None

    def test_empty_string(self):
        assert parse_gopro_filename("") is None

    def test_chapter_zero(self):
        """Chapter 00 is technically valid in the pattern."""
        assert parse_gopro_filename("GX000001.MP4") == (0, 1)


class TestGoProRecording:
    """GoProRecording data model."""

    def _make_chapter(self, chapter_num, rec_num, duration, ct=None):
        return GoProChapter(
            path=Path(f"/fake/GX{chapter_num:02d}{rec_num:04d}.MP4"),
            chapter_num=chapter_num,
            recording_num=rec_num,
            duration=duration,
            creation_time=ct,
        )

    def test_total_duration(self):
        rec = GoProRecording(recording_num=4)
        rec.chapters = [
            self._make_chapter(1, 4, 1025.0),
            self._make_chapter(2, 4, 1025.0),
            self._make_chapter(3, 4, 617.7),
        ]
        assert abs(rec.total_duration - 2667.7) < 0.01

    def test_creation_time(self):
        ct = datetime(2026, 2, 21, 17, 11, 14)
        rec = GoProRecording(recording_num=4)
        rec.chapters = [self._make_chapter(1, 4, 1025.0, ct)]
        assert rec.creation_time == ct

    def test_end_time(self):
        ct = datetime(2026, 2, 21, 17, 11, 14)
        rec = GoProRecording(recording_num=4)
        rec.chapters = [
            self._make_chapter(1, 4, 1025.0, ct),
            self._make_chapter(2, 4, 600.0, ct),
        ]
        expected = ct + timedelta(seconds=1625.0)
        assert rec.end_time == expected


class TestEpisode:
    """Episode data model."""

    def _make_recording(self, rec_num, duration, ct):
        rec = GoProRecording(recording_num=rec_num)
        rec.chapters = [
            GoProChapter(
                path=Path(f"/fake/GX01{rec_num:04d}.MP4"),
                chapter_num=1,
                recording_num=rec_num,
                duration=duration,
                creation_time=ct,
            )
        ]
        return rec

    def test_total_duration(self):
        ep = Episode(
            index=1,
            recordings=[
                self._make_recording(1, 1620.0, datetime(2026, 2, 20, 16, 0)),
                self._make_recording(2, 2913.0, datetime(2026, 2, 20, 17, 48)),
            ],
        )
        assert abs(ep.total_duration - 4533.0) < 0.01

    def test_total_duration_min(self):
        ep = Episode(
            index=1,
            recordings=[self._make_recording(1, 3600.0, datetime(2026, 2, 20, 16, 0))],
        )
        assert abs(ep.total_duration_min - 60.0) < 0.01

    def test_chapter_files(self):
        rec = GoProRecording(recording_num=1)
        rec.chapters = [
            GoProChapter(Path("/fake/GX020001.MP4"), 2, 1, 595.0),
            GoProChapter(Path("/fake/GX010001.MP4"), 1, 1, 1025.0),
        ]
        ep = Episode(index=1, recordings=[rec])
        files = ep.chapter_files()
        # Should be sorted by chapter number, not insertion order
        assert files[0].name == "GX010001.MP4"
        assert files[1].name == "GX020001.MP4"


class TestGroupRecordings:
    """Group recordings into episodes with min/max rules."""

    def _make_recording(self, rec_num, duration_min, ct):
        rec = GoProRecording(recording_num=rec_num)
        rec.chapters = [
            GoProChapter(
                path=Path(f"/fake/GX01{rec_num:04d}.MP4"),
                chapter_num=1,
                recording_num=rec_num,
                duration=duration_min * 60,
                creation_time=ct,
            )
        ]
        return rec

    def test_combine_close_recordings(self):
        """Two recordings within max_gap and under max_duration should combine."""
        recs = {
            1: self._make_recording(1, 27, datetime(2026, 2, 20, 16, 0)),
            2: self._make_recording(2, 40, datetime(2026, 2, 20, 17, 48)),
        }
        # 27 + 40 = 67min < 75min max
        episodes = group_recordings_into_episodes(recs, max_episode_duration=75 * 60)
        assert len(episodes) == 1
        assert len(episodes[0].recordings) == 2

    def test_split_when_over_max_duration(self):
        """Don't combine if total would exceed max_episode_duration."""
        recs = {
            1: self._make_recording(1, 50, datetime(2026, 2, 20, 16, 0)),
            2: self._make_recording(2, 50, datetime(2026, 2, 20, 17, 0)),
        }
        episodes = group_recordings_into_episodes(recs, max_episode_duration=75 * 60)
        assert len(episodes) == 2

    def test_split_when_large_gap(self):
        """Don't combine if time gap exceeds max_gap."""
        recs = {
            1: self._make_recording(1, 27, datetime(2026, 2, 20, 10, 0)),
            2: self._make_recording(2, 49, datetime(2026, 2, 20, 16, 0)),
        }
        # Gap is ~6 hours, max_gap is 2 hours
        episodes = group_recordings_into_episodes(recs, max_gap=120 * 60)
        assert len(episodes) == 2

    def test_different_days_separate(self):
        """Recordings on different days with large gap should be separate."""
        recs = {
            1: self._make_recording(1, 27, datetime(2026, 2, 20, 16, 0)),
            4: self._make_recording(4, 96, datetime(2026, 2, 21, 17, 0)),
        }
        episodes = group_recordings_into_episodes(recs)
        assert len(episodes) == 2

    def test_three_close_recordings_chain(self):
        """Three recordings that individually fit combine until max exceeded."""
        recs = {
            5: self._make_recording(5, 30, datetime(2026, 2, 21, 18, 48)),
            6: self._make_recording(6, 26, datetime(2026, 2, 21, 19, 20)),
            7: self._make_recording(7, 12, datetime(2026, 2, 21, 19, 50)),
        }
        episodes = group_recordings_into_episodes(recs, max_episode_duration=75 * 60)
        assert len(episodes) == 1
        assert len(episodes[0].recordings) == 3

    def test_empty_recordings(self):
        episodes = group_recordings_into_episodes({})
        assert episodes == []

    def test_single_recording(self):
        recs = {
            9: self._make_recording(9, 117, datetime(2026, 2, 24, 15, 0)),
        }
        episodes = group_recordings_into_episodes(recs)
        assert len(episodes) == 1

    def test_episode_numbering(self):
        recs = {
            1: self._make_recording(1, 27, datetime(2026, 2, 20, 10, 0)),
            2: self._make_recording(2, 49, datetime(2026, 2, 20, 16, 0)),
            4: self._make_recording(4, 96, datetime(2026, 2, 21, 17, 0)),
        }
        episodes = group_recordings_into_episodes(recs)
        assert [ep.index for ep in episodes] == [1, 2, 3]

    def test_filters_short_episodes(self):
        """Episodes under MIN_EPISODE_DURATION (20min) should be dropped."""
        recs = {
            8: self._make_recording(8, 0.2, datetime(2026, 2, 24, 14, 0)),
            9: self._make_recording(9, 117, datetime(2026, 2, 24, 15, 0)),
        }
        episodes = group_recordings_into_episodes(recs)
        assert len(episodes) == 1
        assert episodes[0].recordings[0].recording_num == 9
        assert episodes[0].index == 1  # Re-numbered after filtering

    def test_date_preserved_from_first_recording(self):
        ct = datetime(2026, 2, 20, 16, 26, 7)
        recs = {
            1: self._make_recording(1, 27, ct),
        }
        episodes = group_recordings_into_episodes(recs)
        assert episodes[0].date == "2026-02-20"

    def test_mixed_tz_aware_and_missing_creation_time_does_not_crash(self):
        """``extract_recording_date`` now yields tz-AWARE
        creation times. A batch that mixes a present (aware) recording with
        one whose ``creation_time`` is missing must still group without an
        aware/naive comparison ``TypeError`` in the chronological sort.
        """
        aware = datetime(2026, 2, 20, 16, 0, tzinfo=timezone.utc)
        recs = {
            1: self._make_recording(1, 27, aware),
            2: self._make_recording(2, 40, None),
        }
        episodes = group_recordings_into_episodes(recs, max_episode_duration=75 * 60)
        assert len(episodes) >= 1


class TestHaversine:
    """Haversine distance calculation."""

    def test_same_point(self):
        assert _haversine_km(51.9, -9.5, 51.9, -9.5) == 0.0

    def test_known_distance(self):
        # Dublin to Cork ~219km
        dist = _haversine_km(53.3498, -6.2603, 51.8985, -8.4756)
        assert 215 < dist < 225

    def test_short_distance(self):
        # ~500m offset
        dist = _haversine_km(51.9, -9.5, 51.9045, -9.5)
        assert 0.3 < dist < 0.7


class TestIngestPlan:
    """IngestPlan summary output."""

    def test_summary_format(self):
        ep = Episode(
            index=1,
            recordings=[
                GoProRecording(
                    recording_num=1,
                    chapters=[
                        GoProChapter(
                            Path("/f/GX010001.MP4"), 1, 1, 1620.0, datetime(2026, 2, 20, 16, 0)
                        ),
                    ],
                )
            ],
            date="2026-02-20",
            location="South Carolina, US",
        )
        plan = IngestPlan(source_dir=Path("/test"), episodes=[ep])
        summary = plan.summary()
        assert "Episode 1" in summary
        assert "South Carolina" in summary
        assert "1 episodes" in summary


class TestScanGoProDirectory:
    """scan_gopro_directory with mocked ffprobe."""

    @patch("gomoveshift.pipeline.ingest.gopro_parser.get_clip_duration")
    @patch("gomoveshift.pipeline.gopro_gps.extract_recording_date")
    def test_groups_by_recording_number(self, mock_date, mock_dur, tmp_path):
        from gomoveshift.pipeline.ingest import scan_gopro_directory

        mock_dur.return_value = 1025.0
        mock_date.return_value = datetime(2026, 2, 20, 16, 0)

        # Create fake GoPro files
        for name in ["GX010001.MP4", "GX020001.MP4", "GX010002.MP4"]:
            (tmp_path / name).touch()

        recs, filtered, unrec = scan_gopro_directory(tmp_path)

        assert 1 in recs
        assert 2 in recs
        assert len(recs[1].chapters) == 2
        assert len(recs[2].chapters) == 1

    @patch("gomoveshift.pipeline.ingest.gopro_parser.get_clip_duration")
    @patch("gomoveshift.pipeline.gopro_gps.extract_recording_date")
    def test_filters_short_clips(self, mock_date, mock_dur, tmp_path):
        from gomoveshift.pipeline.ingest import scan_gopro_directory

        mock_date.return_value = datetime(2026, 2, 20, 16, 0)

        # One long, one short
        (tmp_path / "GX010001.MP4").touch()
        (tmp_path / "GX010003.MP4").touch()

        def side_effect(path):
            if "0001" in Path(path).name:
                return 1025.0
            return 4.0  # Under MIN_CLIP_DURATION

        mock_dur.side_effect = side_effect

        recs, filtered, unrec = scan_gopro_directory(tmp_path)

        assert 1 in recs
        assert 3 not in recs
        assert len(filtered) == 1
        assert filtered[0].recording_num == 3

    @patch("gomoveshift.pipeline.ingest.gopro_parser.get_clip_duration")
    @patch("gomoveshift.pipeline.gopro_gps.extract_recording_date")
    def test_skips_non_gopro_files(self, mock_date, mock_dur, tmp_path):
        from gomoveshift.pipeline.ingest import scan_gopro_directory

        mock_dur.return_value = 1025.0
        mock_date.return_value = datetime(2026, 2, 20, 16, 0)

        (tmp_path / "GX010001.MP4").touch()
        (tmp_path / "random_video.MP4").touch()

        recs, filtered, unrec = scan_gopro_directory(tmp_path)

        assert len(recs) == 1
        assert len(unrec) == 1
        assert unrec[0].name == "random_video.MP4"

    @patch("gomoveshift.pipeline.ingest.gopro_parser.get_clip_duration")
    @patch("gomoveshift.pipeline.gopro_gps.extract_recording_date")
    def test_chapters_sorted_by_number(self, mock_date, mock_dur, tmp_path):
        from gomoveshift.pipeline.ingest import scan_gopro_directory

        mock_dur.return_value = 1025.0
        mock_date.return_value = datetime(2026, 2, 20, 16, 0)

        # Create files in reverse order
        for name in ["GX030001.MP4", "GX010001.MP4", "GX020001.MP4"]:
            (tmp_path / name).touch()

        recs, _, _ = scan_gopro_directory(tmp_path)
        chapters = recs[1].chapters
        assert [ch.chapter_num for ch in chapters] == [1, 2, 3]


class TestCreateIngestPlan:
    """create_ingest_plan integration (with mocked ffprobe)."""

    @patch("gomoveshift.pipeline.ingest.gopro_parser.get_clip_duration")
    @patch("gomoveshift.pipeline.gopro_gps.extract_recording_date")
    def test_full_plan_creation(self, mock_date, mock_dur, tmp_path):
        # Simulate two recordings on the same day
        mock_dur.return_value = 1800.0  # 30 min each

        def date_side(path):
            if "0001" in Path(path).name:
                return datetime(2026, 2, 20, 16, 0)
            return datetime(2026, 2, 20, 17, 0)

        mock_date.side_effect = date_side

        for name in ["GX010001.MP4", "GX010002.MP4"]:
            (tmp_path / name).touch()

        plan = create_ingest_plan(tmp_path)

        assert len(plan.episodes) == 1  # Combined (30+30 < 75min, gap < 2hr)
        assert plan.total_source_files == 2

    @patch("gomoveshift.pipeline.ingest.gopro_parser.get_clip_duration")
    @patch("gomoveshift.pipeline.gopro_gps.extract_recording_date")
    def test_empty_directory(self, mock_date, mock_dur, tmp_path):
        plan = create_ingest_plan(tmp_path)
        assert len(plan.episodes) == 0


class TestBatchCleanup:
    """Verify concat intermediates are cleaned up after pipeline processing."""

    def _make_episode_with_concat(self, tmp_path, index=1):
        """Create an episode with a real concat file on disk."""
        concat_file = tmp_path / f"episode_{index:02d}.mp4"
        concat_file.write_bytes(b"\x00" * 1024)  # 1KB dummy
        ct = datetime(2026, 2, 20, 16, 0)
        rec = GoProRecording(recording_num=1)
        rec.chapters = [
            GoProChapter(
                path=Path("/fake/GX010001.MP4"),
                chapter_num=1,
                recording_num=1,
                duration=1800.0,
                creation_time=ct,
            )
        ]
        ep = Episode(index=index, recordings=[rec], date="2026-02-20")
        ep.concat_path = concat_file
        ep.location = "South Carolina, United States"
        return ep

    def test_concat_deleted_after_successful_pipeline(self, tmp_path):
        """Concat file should be deleted after successful pipeline run."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output_dir = str(tmp_path / "output")
        mock_result.final_video = None
        mock_result.errors = []

        ep = self._make_episode_with_concat(tmp_path)
        plan = IngestPlan(source_dir=tmp_path, episodes=[ep])
        assert ep.concat_path.exists()

        with (
            patch("gomoveshift.pipeline.orchestrator.PipelineOrchestrator") as mock_orch_cls,
            patch("gomoveshift.config.get_config", return_value=_cfg(tmp_path)),
        ):
            mock_orch_cls.return_value.run.return_value = mock_result
            from gomoveshift.pipeline.ingest import run_batch_pipeline

            results = run_batch_pipeline(plan, output_base=tmp_path / "out")

        assert results[0]["success"] is True
        assert not ep.concat_path.exists()  # Cleaned up

    def test_concat_deleted_after_failed_pipeline(self, tmp_path):
        """Concat file should still be deleted after failed pipeline run."""
        ep = self._make_episode_with_concat(tmp_path)
        plan = IngestPlan(source_dir=tmp_path, episodes=[ep])
        assert ep.concat_path.exists()

        with (
            patch("gomoveshift.pipeline.orchestrator.PipelineOrchestrator") as mock_orch_cls,
            patch("gomoveshift.config.get_config", return_value=_cfg(tmp_path)),
        ):
            mock_orch_cls.return_value.run.side_effect = RuntimeError("encode failed")
            from gomoveshift.pipeline.ingest import run_batch_pipeline

            results = run_batch_pipeline(plan, output_base=tmp_path / "out")

        assert results[0]["success"] is False
        assert not ep.concat_path.exists()  # Still cleaned up

    def test_output_base_defaults_to_gms_output_dir(self, tmp_path, monkeypatch):
        """With no ``output_base`` the run writes under ``gms_output_dir()``
        (``$GMS_OUTPUT_DIR`` override, else ``<data home>/output``)."""
        from gomoveshift.pipeline.ingest import run_batch_pipeline

        def _run():
            ep = self._make_episode_with_concat(tmp_path)
            plan = IngestPlan(source_dir=tmp_path, episodes=[ep])
            with (
                patch("gomoveshift.pipeline.orchestrator.PipelineOrchestrator") as orch_cls,
                patch("gomoveshift.config.get_config", return_value=_cfg(tmp_path)),
            ):
                orch_cls.return_value.run.side_effect = RuntimeError("stop")
                run_batch_pipeline(plan)
            return orch_cls.return_value.run.call_args.args[0].output_base

        override = tmp_path / "override-out"
        monkeypatch.setenv("GMS_OUTPUT_DIR", str(override))
        assert _run() == override

        monkeypatch.delenv("GMS_OUTPUT_DIR")
        monkeypatch.setenv("GMS_DATA_HOME", str(tmp_path / "home"))
        assert _run() == tmp_path / "home" / "output"


class TestConcatSkipExisting:
    """Verify concat skips already-completed files on restart."""

    def test_concat_skips_existing_file(self, tmp_path):
        """If concat output already exists, skip the concat."""
        ct = datetime(2026, 2, 20, 16, 0)
        rec = GoProRecording(recording_num=1)
        chapter_path = tmp_path / "GX010001.MP4"
        chapter_path.write_bytes(b"\x00" * 100)
        rec.chapters = [
            GoProChapter(
                path=chapter_path,
                chapter_num=1,
                recording_num=1,
                duration=1800.0,
                creation_time=ct,
            )
        ]
        ep = Episode(index=1, recordings=[rec], date="2026-02-20")

        output_dir = tmp_path / "out"
        output_dir.mkdir()
        existing = output_dir / "episode_01_2026-02-20.mp4"
        existing.write_bytes(b"\x00" * 500)  # Pre-existing concat

        from gomoveshift.pipeline.ingest import concat_episode

        # Reuse requires a cache that probes to the episode's length (a killed
        # run's truncated concat is rebuilt — test_concat_cache_validated.py).
        with patch(
            "gomoveshift.pipeline.ingest.concatenator.get_clip_duration", return_value=1799.5
        ):
            result = concat_episode(ep, output_dir)

        assert result == existing
        assert existing.stat().st_size == 500  # Not overwritten


class TestBatchSkipStops:
    """Verify batch passes skip_stops for driving content."""

    def test_run_batch_passes_skip_stops(self, tmp_path):
        """run_batch_pipeline should pass skip_stops to orchestrator."""
        ct = datetime(2026, 2, 20, 16, 0)
        rec = GoProRecording(recording_num=1)
        rec.chapters = [GoProChapter(Path("/fake/GX010001.MP4"), 1, 1, 1800.0, ct)]
        ep = Episode(index=1, recordings=[rec], date="2026-02-20")
        concat_file = tmp_path / "ep.mp4"
        concat_file.write_bytes(b"\x00" * 100)
        ep.concat_path = concat_file
        ep.location = "South Carolina, United States"
        plan = IngestPlan(source_dir=tmp_path, episodes=[ep])

        with (
            patch("gomoveshift.pipeline.orchestrator.PipelineOrchestrator") as mock_cls,
            patch("gomoveshift.config.get_config", return_value=_cfg(tmp_path)),
        ):
            mock_result = MagicMock(success=True, output_dir="/out", final_video=None, errors=[])
            mock_cls.return_value.run.return_value = mock_result
            from gomoveshift.pipeline.ingest import run_batch_pipeline

            run_batch_pipeline(plan, output_base=tmp_path, skip_stops=True)
            call_args = mock_cls.return_value.run.call_args
            options = call_args[0][0]  # RunOptions is the first positional arg
            assert "stop_removal" in options.skip


class TestIngestEdgeCases:
    """Edge cases for GoPro ingestion that should be handled gracefully."""

    @patch("gomoveshift.pipeline.ingest.gopro_parser.get_clip_duration")
    @patch("gomoveshift.pipeline.gopro_gps.extract_recording_date")
    def test_missing_chapter_mid_sequence(self, mock_date, mock_dur, tmp_path):
        """GX01 exists, GX02 missing, GX03 exists — should still group correctly."""
        from gomoveshift.pipeline.ingest import scan_gopro_directory

        mock_dur.return_value = 1025.0
        mock_date.return_value = datetime(2026, 2, 20, 16, 0)

        # Chapter 1 and 3 present, chapter 2 missing
        (tmp_path / "GX010001.MP4").touch()
        (tmp_path / "GX030001.MP4").touch()

        recs, _, _ = scan_gopro_directory(tmp_path)
        assert 1 in recs
        chapters = recs[1].chapters
        # Should have 2 chapters (1 and 3), gap is handled by the user
        assert len(chapters) == 2
        assert chapters[0].chapter_num == 1
        assert chapters[1].chapter_num == 3

    @patch("gomoveshift.pipeline.ingest.gopro_parser.get_clip_duration")
    @patch("gomoveshift.pipeline.gopro_gps.extract_recording_date")
    def test_corrupt_file_zero_duration(self, mock_date, mock_dur, tmp_path):
        """A corrupt/truncated file (0 duration from ffprobe) should be filtered."""
        from gomoveshift.pipeline.ingest import scan_gopro_directory

        mock_date.return_value = datetime(2026, 2, 20, 16, 0)

        (tmp_path / "GX010001.MP4").touch()
        (tmp_path / "GX010002.MP4").touch()

        def dur_side(path):
            if "0002" in Path(path).name:
                return 0.0  # Corrupt/truncated file
            return 1800.0

        mock_dur.side_effect = dur_side

        recs, filtered, _ = scan_gopro_directory(tmp_path)
        # File with 0 duration should be filtered (< MIN_CLIP_DURATION=10)
        assert 1 in recs
        assert len(filtered) == 1
        assert filtered[0].recording_num == 2

    @patch("gomoveshift.pipeline.ingest.gopro_parser.get_clip_duration")
    @patch("gomoveshift.pipeline.gopro_gps.extract_recording_date")
    def test_duplicate_chapter_numbers_kept(self, mock_date, mock_dur, tmp_path):
        """Different recordings with same chapter num shouldn't conflict."""
        from gomoveshift.pipeline.ingest import scan_gopro_directory

        mock_dur.return_value = 1800.0
        mock_date.return_value = datetime(2026, 2, 20, 16, 0)

        # Two different recordings, both with chapter 1
        (tmp_path / "GX010001.MP4").touch()
        (tmp_path / "GX010002.MP4").touch()

        recs, _, _ = scan_gopro_directory(tmp_path)
        assert 1 in recs
        assert 2 in recs
        assert len(recs[1].chapters) == 1
        assert len(recs[2].chapters) == 1

    @patch("gomoveshift.pipeline.ingest.gopro_parser.get_clip_duration")
    @patch("gomoveshift.pipeline.gopro_gps.extract_recording_date")
    def test_ffprobe_failure_skips_file(self, mock_date, mock_dur, tmp_path):
        """If ffprobe fails on a file, skip it gracefully."""
        from gomoveshift.pipeline.ingest import scan_gopro_directory

        mock_date.return_value = datetime(2026, 2, 20, 16, 0)
        (tmp_path / "GX010001.MP4").touch()
        (tmp_path / "GX010002.MP4").touch()

        def dur_side(path):
            if "0002" in Path(path).name:
                raise RuntimeError("ffprobe failed: invalid data")
            return 1800.0

        mock_dur.side_effect = dur_side

        recs, _, _ = scan_gopro_directory(tmp_path)
        assert 1 in recs
        # Recording 2 should be absent (ffprobe failed)
        assert 2 not in recs


class TestDeriveLocationMajorityVote:
    """derive_location_from_gpx must majority-vote across multiple waypoints."""

    @patch("gps_journey.geocoding.nominatim_roads.NominatimRoadsClient")
    @patch("gps_journey.geocoding.cache.GeocodingCache")
    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_majority_vote_returns_most_common_state(self, mock_parser, mock_cache, mock_geocoder):
        """When 3/5 samples return County Clare, result should be County Clare."""
        from gomoveshift.pipeline.ingest import derive_location_from_gpx

        # Build fake journey with 10 waypoints
        waypoints = []
        for i in range(10):
            wp = MagicMock()
            wp.latitude = 52.0 + i * 0.01
            wp.longitude = -9.0 + i * 0.01
            waypoints.append(wp)

        mock_journey = MagicMock()
        mock_journey.waypoints = waypoints
        mock_parser.return_value.parse.return_value = mock_journey

        # Geocoder returns County Clare for most, County Galway for some
        # With 10 waypoints, samples are: [0, 2, 5, 7, 9]
        # wp[0] lat=52.0 -> Galway, wp[2] lat=52.02 -> Clare,
        # wp[5] lat=52.05 -> Clare, wp[7] lat=52.07 -> Clare,
        # wp[9] lat=52.09 -> Clare => 4 Clare, 1 Galway
        def geocode_side(lat, lon):
            loc = MagicMock()
            if lat < 52.01:
                loc.state = "County Galway"
                loc.county = "County Galway"
            else:
                loc.state = "County Clare"
                loc.county = "County Clare"
            loc.country = "Ireland"
            return loc

        mock_geocoder.return_value.reverse_geocode.side_effect = geocode_side

        result = derive_location_from_gpx(Path("/fake/test.gpx"))
        assert "County Clare" in result
        assert "Ireland" in result

    @patch("gps_journey.geocoding.nominatim_roads.NominatimRoadsClient")
    @patch("gps_journey.geocoding.cache.GeocodingCache")
    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_samples_five_evenly_spaced(self, mock_parser, mock_cache, mock_geocoder):
        """With 20 waypoints, should sample indices 0, 5, 10, 15, 19."""
        from gomoveshift.pipeline.ingest import derive_location_from_gpx

        waypoints = []
        for i in range(20):
            wp = MagicMock()
            wp.latitude = 52.0 + i * 0.01
            wp.longitude = -9.0 + i * 0.01
            waypoints.append(wp)

        mock_journey = MagicMock()
        mock_journey.waypoints = waypoints
        mock_parser.return_value.parse.return_value = mock_journey

        loc = MagicMock()
        loc.state = "County Cork"
        loc.county = "County Cork"
        loc.country = "Ireland"
        mock_geocoder.return_value.reverse_geocode.return_value = loc

        result = derive_location_from_gpx(Path("/fake/test.gpx"))
        assert result == "County Cork, Ireland"
        # Should have been called exactly 5 times
        assert mock_geocoder.return_value.reverse_geocode.call_count == 5

    @patch("gps_journey.geocoding.nominatim_roads.NominatimRoadsClient")
    @patch("gps_journey.geocoding.cache.GeocodingCache")
    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_endpoint_samples_dodge_raw_first_and_last_waypoints(
        self, mock_parser, mock_cache, mock_geocoder
    ):
        """Endpoint samples anchor inward (_pick_anchor_waypoint) so a stale
        previous-session GPS fix at waypoint 0 or n-1 never votes."""
        from gomoveshift.pipeline.ingest import derive_location_from_gpx

        waypoints = []
        for i in range(100):
            wp = MagicMock()
            wp.latitude = float(i)
            wp.longitude = -9.0
            waypoints.append(wp)

        mock_journey = MagicMock()
        mock_journey.waypoints = waypoints
        mock_parser.return_value.parse.return_value = mock_journey

        loc = MagicMock()
        loc.state = "County Kerry"
        loc.county = "County Kerry"
        loc.country = "Ireland"
        mock_geocoder.return_value.reverse_geocode.return_value = loc

        derive_location_from_gpx(Path("/fake/test.gpx"))

        sampled_lats = [
            call.args[0] for call in mock_geocoder.return_value.reverse_geocode.call_args_list
        ]
        # n=100 -> anchor offset min(20, 100//20)=5: endpoints sample indices
        # 5 and 94 — never the raw 0 / 99 that can carry a stale fix.
        assert sampled_lats == [5.0, 25.0, 50.0, 75.0, 94.0]

    @patch("gps_journey.geocoding.nominatim_roads.NominatimRoadsClient")
    @patch("gps_journey.geocoding.cache.GeocodingCache")
    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_empty_waypoints_returns_empty(self, mock_parser, mock_cache, mock_geocoder):
        """Empty waypoints should return empty string."""
        from gomoveshift.pipeline.ingest import derive_location_from_gpx

        mock_journey = MagicMock()
        mock_journey.waypoints = []
        mock_parser.return_value.parse.return_value = mock_journey

        result = derive_location_from_gpx(Path("/fake/test.gpx"))
        assert result == ""


def _mock_location(town=None, city=None, village=None, county=None, state=None, country=None):
    """Helper to create a mock Location object for reverse_geocode tests."""
    loc = MagicMock()
    loc.town = town
    loc.city = city
    loc.village = village
    loc.county = county
    loc.state = state
    loc.country = country
    return loc


class TestDeriveDriveOSTText:
    """derive_drive_ost_text produces 'StartTown to EndTown' from GPX endpoints.

    Drive-level OST intentionally omits "via road" (roads are episode-level
    specifics that don't generalize across a multi-region drive).
    """

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_uses_start_and_end_towns(self, mock_parser):
        """Drive-level OST reverse-geocodes first-gpx-first-waypoint and
        last-gpx-last-waypoint, formatted as 'StartTown to EndTown'."""
        from gomoveshift.pipeline.ingest import derive_drive_ost_text

        first_journey = MagicMock()
        first_journey.waypoints = [
            MagicMock(latitude=32.78, longitude=-79.93),  # Charleston
            MagicMock(latitude=32.80, longitude=-79.90),
        ]
        last_journey = MagicMock()
        last_journey.waypoints = [
            MagicMock(latitude=33.68, longitude=-78.88),
            MagicMock(latitude=33.69, longitude=-78.86),  # Myrtle Beach
        ]
        mock_parser.return_value.parse.side_effect = [first_journey, last_journey]

        def fake_geocode(lat, lon):
            loc = MagicMock()
            if lat < 33:
                loc.town = "Charleston"
                loc.city = None
                loc.village = None
                loc.state = "South Carolina"
                loc.country = "United States"
            else:
                loc.town = "Myrtle Beach"
                loc.city = None
                loc.village = None
                loc.state = "South Carolina"
                loc.country = "United States"
            return loc

        mock_geocoder = MagicMock()
        mock_geocoder.reverse_geocode.side_effect = fake_geocode
        mock_analyzer = MagicMock()  # signature-compat only, not used by new impl

        result = derive_drive_ost_text(
            [Path("/fake/ep1.gpx"), Path("/fake/ep2.gpx")],
            route_analyzer=mock_analyzer,
            geocoder=mock_geocoder,
        )
        assert "Charleston" in result
        assert "Myrtle Beach" in result
        # Drive-level omits "via road" — roads are episode specifics
        assert " via " not in result

    def test_empty_gpx_list_returns_empty(self):
        """No GPX files should return empty string."""
        from gomoveshift.pipeline.ingest import derive_drive_ost_text

        result = derive_drive_ost_text([])
        assert result == ""


class TestDeriveEpisodeOSTText:
    """derive_episode_ost_text uses RouteAnalyzer for road names + towns."""

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_uses_road_and_towns(self, mock_parser):
        """OST includes the longest road and towns from RouteAnalyzer."""
        from gomoveshift.pipeline.ingest import derive_episode_ost_text

        mock_journey = MagicMock()
        mock_journey.waypoints = [MagicMock() for _ in range(100)]
        mock_parser.return_value.parse.return_value = mock_journey

        mock_analyzer = MagicMock()
        mock_analyzer.get_road_segments.return_value = [
            {"road": "Seaside Road", "distance_km": 17.7},
            {"road": "Janette Drive", "distance_km": 2.4},
        ]
        mock_analyzer.get_towns_passed_through.return_value = [
            {"name": "Longwood", "duration_minutes": 10},
        ]

        result = derive_episode_ost_text(
            Path("/fake/ep.gpx"),
            route_analyzer=mock_analyzer,
        )
        assert "Seaside Road" in result
        assert "Longwood" in result

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_two_towns_produces_route(self, mock_parser):
        """With two towns, OST should be 'Town A to Town B via Road'."""
        from gomoveshift.pipeline.ingest import derive_episode_ost_text

        mock_journey = MagicMock()
        mock_journey.waypoints = [MagicMock() for _ in range(100)]
        mock_parser.return_value.parse.return_value = mock_journey

        mock_analyzer = MagicMock()
        mock_analyzer.get_road_segments.return_value = [
            {"road": "Trask Parkway", "distance_km": 21.0},
        ]
        mock_analyzer.get_towns_passed_through.return_value = [
            {"name": "Port Royal", "duration_minutes": 8},
            {"name": "Beaufort", "duration_minutes": 15},
        ]

        result = derive_episode_ost_text(
            Path("/fake/ep.gpx"),
            route_analyzer=mock_analyzer,
        )
        assert "Port Royal" in result
        assert "Beaufort" in result
        assert "Trask Parkway" in result

    def test_none_gpx_returns_empty(self):
        """None GPX path returns empty string."""
        from gomoveshift.pipeline.ingest import derive_episode_ost_text

        result = derive_episode_ost_text(None)
        assert result == ""

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_no_roads_falls_back_to_geocoder(self, mock_parser):
        """When RouteAnalyzer finds no roads, fall back to geocoder for town+state."""
        from gomoveshift.pipeline.ingest import derive_episode_ost_text

        mock_journey = MagicMock()
        mock_journey.waypoints = [MagicMock(latitude=32.0, longitude=-80.0)]
        mock_parser.return_value.parse.return_value = mock_journey

        mock_analyzer = MagicMock()
        mock_analyzer.get_road_segments.return_value = []
        mock_analyzer.get_towns_passed_through.return_value = []

        mock_geocoder = MagicMock()
        mock_geocoder.reverse_geocode.return_value = _mock_location(
            town="Beaufort",
            state="South Carolina",
        )

        result = derive_episode_ost_text(
            Path("/fake/ep.gpx"),
            route_analyzer=mock_analyzer,
            geocoder=mock_geocoder,
        )
        assert "Beaufort" in result
