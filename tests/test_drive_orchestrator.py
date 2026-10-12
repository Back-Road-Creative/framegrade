"""Tests for DrivePipelineOrchestrator — integration tests with mock processors."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gomoveshift.config import Config, DriveConfig
from gomoveshift.models import DrivePlan, LFSegment, RegularLFPlan
from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator

from _state_helpers import make_state


class TestDrivePipelineOrchestrator:
    """Integration tests for the drive pipeline coordination."""

    def test_instantiation(self):
        """Orchestrator can be created with default config."""
        orch = DrivePipelineOrchestrator()
        assert orch is not None
        assert orch.config is not None

    def test_instantiation_with_config(self):
        """Orchestrator accepts custom config."""
        cfg = Config()
        cfg.drive.min_segment_duration = 60.0
        orch = DrivePipelineOrchestrator(config=cfg)
        assert orch.config.drive.min_segment_duration == 60.0

    def test_run_phases_defined(self):
        """Orchestrator has the expected phase methods for the single surviving pipeline."""
        orch = DrivePipelineOrchestrator()
        assert hasattr(orch, "run")
        assert hasattr(orch, "_phase_a_ingest_and_analyze")
        assert hasattr(orch, "_phase_bc_day_first")
        assert hasattr(orch, "_phase_d_metadata_and_publish")
        # Legacy attrs must be gone
        assert not hasattr(orch, "_phase_b_process_files")
        assert not hasattr(orch, "_phase_c_assemble")
        assert not hasattr(orch, "_run_c_d_per_day")

    @patch("gomoveshift.pipeline.drive_orchestrator.ReviewGate")
    @patch(
        "gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator._phase_d_metadata_and_publish"
    )
    @patch("gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator._phase_bc_day_first")
    @patch(
        "gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator._phase_a_ingest_and_analyze"
    )
    def test_run_calls_all_phases(self, mock_a, mock_bc, mock_d, mock_rg, tmp_path):
        """run() calls A → BC-day-first → D in order (single surviving pipeline)."""

        # Populate state["episode_files"] so the phase_a coherence guard
        # passes (real phase_a sets this before returning). episode_locations
        # must be set parallel to episode_files — empty-with-episodes now
        # hard-fails the OST gate.
        def _a_side_effect(state):
            state["episode_files"] = ["/mock/ep0.mp4"]
            state["episode_locations"] = ["Mock Town, SC"]
            return {}

        mock_a.side_effect = _a_side_effect

        # Populate state["regular_lf_paths"] so Layer-1 invariants pass when
        # phase_bc_day_first gets appended to phases_completed. The real
        # phase_bc_day_first sets these fields before returning; a mock that
        # leaves state unchanged would (correctly) trip the invariants.
        # Also populate ``day_artifacts`` so the phase-bc append guard
        # passes (real BC sets this when run_all_days produces masters).
        def _bc_side_effect(state):
            state["regular_lf_paths"] = ["/mock/lf_01.mp4"]
            state["shorts_per_lf"] = {1: []}
            state["day_artifacts"] = [{"date": "2026-02-20", "master_path": "/mock/d1.mp4"}]
            return {}

        mock_bc.side_effect = _bc_side_effect
        mock_d.return_value = {}

        source = tmp_path / "source"
        source.mkdir()

        orch = DrivePipelineOrchestrator()
        result = orch.run(
            source_dir=source,
            output_base=tmp_path / "output",
            region="ireland",
        )

        mock_a.assert_called_once()
        mock_bc.assert_called_once()
        mock_d.assert_called_once()
        assert result.stages_completed == ["phase_a", "phase_bc_day_first", "phase_d"]

    def test_result_tracks_errors(self, tmp_path):
        """Result captures errors without crashing."""
        orch = DrivePipelineOrchestrator()
        # Source dir doesn't exist, should produce error result
        result = orch.run(
            source_dir=tmp_path / "nonexistent",
            output_base=tmp_path / "output",
            region="ireland",
        )
        assert result.success is False
        assert len(result.errors) > 0


class TestPhaseAWiring:
    """Tests for Phase A: Ingest & Analyze — real wiring, mocked I/O."""

    def _make_episode(self, concat_path, gpx_path=None, duration=300.0):
        """Create a mock Episode for testing."""
        ep = MagicMock()
        ep.concat_path = concat_path
        ep.gpx_path = gpx_path
        ep.total_duration = duration
        ep.date = "2026-03-24"
        ep.location = ""
        ep.index = 1
        return ep

    @patch("gomoveshift.pipeline.drive_phases.phase_a.UnifiedSegmentScorer")
    @patch("gomoveshift.pipeline.drive_phases.phase_a.execute_ingest")
    @patch("gomoveshift.pipeline.drive_phases.phase_a.create_ingest_plan")
    def test_phase_a_calls_ingest_pipeline(
        self,
        mock_create_plan,
        mock_execute,
        mock_scorer_cls,
        tmp_path,
    ):
        """Phase A calls create_ingest_plan then execute_ingest."""
        source = tmp_path / "source"
        source.mkdir()
        output = tmp_path / "output"

        plan = MagicMock()
        plan.episodes = []
        mock_create_plan.return_value = plan
        mock_execute.return_value = plan

        mock_scorer = MagicMock()
        mock_scorer.merge_file_scores.return_value = np.zeros((0, 9))
        mock_scorer_cls.return_value = mock_scorer

        orch = DrivePipelineOrchestrator()
        state = make_state(source_dir=source, output_base=output, fresh=False)
        orch._phase_a_ingest_and_analyze(state)

        # Generic video ingest: mode + extensions now plumbed
        # from config.drive into create_ingest_plan.
        mock_create_plan.assert_called_once()
        call_args = mock_create_plan.call_args
        assert call_args.args[0] == source
        assert call_args.kwargs.get("mode") == "gopro"
        assert ".mp4" in call_args.kwargs.get("extensions", [])
        mock_execute.assert_called_once()

    @patch("gomoveshift.pipeline.drive_phases.phase_a.UnifiedSegmentScorer")
    @patch("gomoveshift.pipeline.drive_phases.phase_a.execute_ingest")
    @patch("gomoveshift.pipeline.drive_phases.phase_a.create_ingest_plan")
    def test_phase_a_scores_each_episode(
        self,
        mock_create_plan,
        mock_execute,
        mock_scorer_cls,
        tmp_path,
    ):
        """Phase A runs score_file for each episode with concat_path."""
        source = tmp_path / "source"
        source.mkdir()
        output = tmp_path / "output"

        ep1 = self._make_episode(tmp_path / "ep1.mp4", duration=300.0)
        ep2 = self._make_episode(tmp_path / "ep2.mp4", duration=200.0)
        plan = MagicMock()
        plan.episodes = [ep1, ep2]
        mock_create_plan.return_value = plan
        mock_execute.return_value = plan

        mock_scorer = MagicMock()
        mock_scorer.score_file.side_effect = [
            np.ones((300, 9)),
            np.ones((200, 9)),
        ]
        mock_scorer.merge_file_scores.return_value = np.ones((500, 9))
        mock_scorer_cls.return_value = mock_scorer

        orch = DrivePipelineOrchestrator()
        state = make_state(source_dir=source, output_base=output, fresh=False)
        orch._phase_a_ingest_and_analyze(state)

        assert mock_scorer.score_file.call_count == 2
        mock_scorer.merge_file_scores.assert_called_once()

    @patch("gomoveshift.pipeline.drive_phases.phase_a.UnifiedSegmentScorer")
    @patch("gomoveshift.pipeline.drive_phases.phase_a.execute_ingest")
    @patch("gomoveshift.pipeline.drive_phases.phase_a.create_ingest_plan")
    def test_phase_a_populates_state(
        self,
        mock_create_plan,
        mock_execute,
        mock_scorer_cls,
        tmp_path,
    ):
        """Phase A sets episode_files, drive_scores, file_boundaries, file_durations."""
        source = tmp_path / "source"
        source.mkdir()
        output = tmp_path / "output"

        ep = self._make_episode(tmp_path / "ep.mp4", duration=120.0)
        plan = MagicMock()
        plan.episodes = [ep]
        mock_create_plan.return_value = plan
        mock_execute.return_value = plan

        mock_scorer = MagicMock()
        scores = np.random.rand(120, 9)
        mock_scorer.score_file.return_value = scores
        mock_scorer.merge_file_scores.return_value = scores
        mock_scorer_cls.return_value = mock_scorer

        orch = DrivePipelineOrchestrator()
        state = make_state(source_dir=source, output_base=output, fresh=False)
        orch._phase_a_ingest_and_analyze(state)

        assert len(state["episode_files"]) == 1
        assert state["drive_scores"] is not None
        assert len(state["file_durations"]) == 1
        assert state["file_durations"][0] == 120.0

    @patch("gomoveshift.pipeline.drive_phases.phase_a.UnifiedSegmentScorer")
    @patch("gomoveshift.pipeline.drive_phases.phase_a.execute_ingest")
    @patch("gomoveshift.pipeline.drive_phases.phase_a.create_ingest_plan")
    def test_phase_a_skips_episodes_without_concat(
        self,
        mock_create_plan,
        mock_execute,
        mock_scorer_cls,
        tmp_path,
    ):
        """Episodes with no concat_path are skipped during scoring."""
        source = tmp_path / "source"
        source.mkdir()
        output = tmp_path / "output"

        ep_good = self._make_episode(tmp_path / "ep1.mp4", duration=300.0)
        ep_bad = self._make_episode(None, duration=0.0)
        plan = MagicMock()
        plan.episodes = [ep_good, ep_bad]
        mock_create_plan.return_value = plan
        mock_execute.return_value = plan

        mock_scorer = MagicMock()
        mock_scorer.score_file.return_value = np.ones((300, 9))
        mock_scorer.merge_file_scores.return_value = np.ones((300, 9))
        mock_scorer_cls.return_value = mock_scorer

        orch = DrivePipelineOrchestrator()
        state = make_state(source_dir=source, output_base=output, fresh=False)
        orch._phase_a_ingest_and_analyze(state)

        # Only the episode with concat_path is scored
        assert mock_scorer.score_file.call_count == 1
        assert len(state["episode_files"]) == 1

    @patch("gomoveshift.pipeline.drive_phases.phase_a.parse_gpx_file")
    @patch("gomoveshift.pipeline.drive_phases.phase_a.UnifiedSegmentScorer")
    @patch("gomoveshift.pipeline.drive_phases.phase_a.execute_ingest")
    @patch("gomoveshift.pipeline.drive_phases.phase_a.create_ingest_plan")
    def test_phase_a_passes_gps_to_scorer(
        self,
        mock_create_plan,
        mock_execute,
        mock_scorer_cls,
        mock_parse_gpx,
        tmp_path,
    ):
        """When episode has gpx_path, GPS data is passed to score_file."""
        source = tmp_path / "source"
        source.mkdir()
        output = tmp_path / "output"

        ep = self._make_episode(
            tmp_path / "ep.mp4",
            gpx_path=tmp_path / "ep.gpx",
            duration=120.0,
        )
        plan = MagicMock()
        plan.episodes = [ep]
        mock_create_plan.return_value = plan
        mock_execute.return_value = plan

        fake_gps = MagicMock()
        fake_gps.start_time = 1000.0
        mock_parse_gpx.return_value = fake_gps

        mock_scorer = MagicMock()
        mock_scorer.score_file.return_value = np.ones((120, 9))
        mock_scorer.merge_file_scores.return_value = np.ones((120, 9))
        mock_scorer_cls.return_value = mock_scorer

        orch = DrivePipelineOrchestrator()
        state = make_state(source_dir=source, output_base=output, fresh=False)
        orch._phase_a_ingest_and_analyze(state)

        # score_file called with gps_data and gps_offset
        call_kwargs = mock_scorer.score_file.call_args
        assert call_kwargs.kwargs.get("gps_data") is fake_gps
        assert call_kwargs.kwargs.get("gps_offset") == 1000.0


class TestPhaseDWiring:
    """Tests for Phase D: Metadata & Publish — real wiring, mocked I/O."""

    @pytest.fixture(autouse=True)
    def _stub_final_validation_gate(self):
        # These wiring tests use placeholder LF paths that never
        # exist on disk. The pre-upload file_integrity gate now HARD-fails on
        # a missing/truncated FINAL (was a silent skip), which would abort
        # every test before the metadata/schedule/upload wiring under test
        # runs. They don't exercise the validator, so stub it to pass.
        with patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineValidator") as val_cls:
            val = val_cls.return_value
            val.validate.return_value = MagicMock(passed=True, summary=lambda: "ok")
            yield

    def _make_phase_d_state(self, tmp_path, num_lfs=1, num_shorts_per_lf=7):
        """Create a state dict as Phase C would produce."""
        output = tmp_path / "output"
        output.mkdir(parents=True, exist_ok=True)

        source = tmp_path / "source_drive"
        source.mkdir(parents=True, exist_ok=True)

        processed = [tmp_path / f"ep_{i:03d}_FINAL.mp4" for i in range(max(num_lfs, 1))]

        regular_lfs = []
        lf_paths = []
        shorts_per_lf = {}
        for i in range(num_lfs):
            seg = LFSegment(
                source_file_index=i,
                source_path=processed[min(i, len(processed) - 1)],
                start_time=10.0,
                end_time=610.0,
                original_start_time=10.0 + i * 600,
                original_end_time=610.0 + i * 600,
                composite_score=0.7,
                gps_center_lat=0.0,
                gps_center_lon=0.0,
                terrain_type="countryside",
                contiguous_with_previous=False,
            )
            lf = RegularLFPlan(
                index=i + 1,
                segments=[seg],
                montage_peaks=[seg],
                shorts_target=num_shorts_per_lf,
                region="ireland",
                total_duration=600.0,
            )
            regular_lfs.append(lf)
            # The pre-upload file_integrity gate HARD-fails on a
            # missing/under-1-MiB FINAL, so write a real ≥1 MiB placeholder
            # (ffprobe is stubbed via the autouse validator fixture).
            lf_path = output / "long_forms" / f"lf_{i + 1:02d}.mp4"
            lf_path.parent.mkdir(parents=True, exist_ok=True)
            lf_path.write_bytes(b"\0" * (1024 * 1024 + 1))
            lf_paths.append(lf_path)
            shorts_per_lf[i + 1] = [MagicMock() for _ in range(num_shorts_per_lf)]

        drive_plan = DrivePlan(
            source_files=processed,
            processed_files=processed,
            regular_lfs=regular_lfs,
            score_threshold=0.5,
            total_source_duration=600.0 * max(num_lfs, 1),
            total_highlight_duration=600.0 * num_lfs,
        )

        state = make_state(
            source_dir=source,
            output_base=output,
            region="ireland",
            publish_day="wed",
            location="Cork, Ireland",
            skip_upload=False,
            skip_csv=False,
            drive_plan=drive_plan,
            regular_lf_paths=lf_paths,
            shorts_per_lf=shorts_per_lf,
            file_durations=[120.0, 180.0, 90.0],
            # Phase BC invariant ``_assert_per_lf_recording_date_complete``
            # populates this map in production. Test fixtures bypass
            # Phase BC, so we thread the data the contract assertion in
            # Phase D's ``_build_recording_dates_by_short_index`` expects.
            per_lf_recording_date={i + 1: "2025-09-14" for i in range(num_lfs)},
            # BC→D contract (assert_phase_d_per_lf_inputs_present):
            # every rendered LF must have an entry in per_lf_locations and
            # per_lf_trim_ratio. BC populates both in production.
            per_lf_locations={i + 1: "Cork, Ireland" for i in range(num_lfs)},
            per_lf_trim_ratio={i + 1: 0.7 for i in range(num_lfs)},
        )
        # ShortsValidator gate (1080×1920 enforcement) ffprobes
        # every path in ``shorts_per_lf``. These orchestration-plumbing
        # tests use ``MagicMock`` placeholders for short paths — they
        # never exist on disk, so the gate aborts before the wiring
        # under test runs. The tests don't assert anything about the
        # gate; bypass it via the documented ``--allow-bad-shorts``
        # state flag so we test what we mean to test.
        state.skip_shorts_quality_gate = True
        return state

    def _setup_phase_d_mocks(
        self,
        mock_meta_cls,
        mock_sched_cls,
        mock_social_cls,
        mock_upload_cls,
        mock_csv_cls,
        num_lfs=1,
        num_shorts_per_lf=7,
    ):
        """Standard mock config for Phase D. Returns mock instances."""
        from datetime import datetime as _dt
        from datetime import timedelta as _td

        # LongFormMetadataGenerator
        # The distinct-title rule requires distinct titles per LF. side_effect
        # varies the returned title by project_name.
        mock_meta = MagicMock()

        def _meta_side_effect(**kwargs):
            project_name = kwargs.get("project_name", "lf")
            return {
                "title": f"Title for {project_name}",
                "description": "Test description",
                "tags": ["driving", "ireland"],
                "category": "Travel & Events",
            }

        mock_meta.generate.side_effect = _meta_side_effect
        mock_meta_cls.return_value = mock_meta

        # PipelineScheduler
        # lf_slot dates must land on days in
        # LONG_FORM_DAYS (Wed, weekday=2). 2026-04-01 is Wed; every
        # 7-day step yields the next Wed.
        mock_sched = MagicMock()
        wed_dates = [
            (_dt(2026, 4, 1) + _td(days=7 * i)).strftime("%Y-%m-%d") for i in range(num_lfs)
        ]
        mock_lf_slots = [
            MagicMock(
                date=wed_dates[i],
                time="11:00:00",
                datetime_est=f"{wed_dates[i]}T11:00:00-04:00",
                day_of_week="Wednesday",
            )
            for i in range(num_lfs)
        ]
        mock_shorts_slots = [
            MagicMock(
                short_index=i + 1,
                date=wed_dates[i // max(num_shorts_per_lf, 1)],
                time="16:00:00",
                datetime_est=f"{wed_dates[i // max(num_shorts_per_lf, 1)]}T16:00:00-04:00",
                day_of_week="Wednesday",
            )
            for i in range(num_lfs * num_shorts_per_lf)
        ]
        mock_drive_schedule = MagicMock()
        mock_drive_schedule.lf_slots = mock_lf_slots
        mock_drive_schedule.shorts_slots = mock_shorts_slots
        mock_drive_schedule.to_dict.return_value = {"region": "ireland"}
        mock_sched.generate_drive_schedule.return_value = mock_drive_schedule
        mock_sched_cls.return_value = mock_sched
        mock_sched_cls.from_config.return_value = mock_sched

        # SocialGenerator
        # CSV row count must equal Σ len(shorts_per_lf[i]).
        # generate_all must return ``num_shorts_per_lf`` posts per LF
        # call (not a single post).
        mock_social = MagicMock()
        posts_all: list = []
        for lf_i in range(num_lfs):
            lf_posts = []
            for sh_j in range(num_shorts_per_lf):
                idx = lf_i * num_shorts_per_lf + sh_j + 1
                p = MagicMock(short_index=idx, caption=f"caption_{idx}")
                p.to_dict.return_value = {"short_index": idx, "caption": f"caption_{idx}"}
                lf_posts.append(p)
            posts_all.append(lf_posts)
        mock_social.generate_all.side_effect = posts_all
        mock_social_cls.return_value = mock_social

        # PipelineUploader. Day-first contract: upload_shorts returns a LIST
        # of Drive links (one per short path), zipped into the per-LF
        # UploadManifest.files. A bare MagicMock return iterated empty → every
        # Picture Url 1 blank (METRICOOL-CSV-NO-MEDIA gate).
        mock_uploader = MagicMock()

        def _fake_upload_shorts(*_args, shorts_dir=None, **_kwargs):
            n = len(shorts_dir) if isinstance(shorts_dir, (list, tuple)) else 1
            return [f"https://drive.google.com/test{i}" for i in range(n)]

        mock_uploader.upload_shorts.side_effect = _fake_upload_shorts
        mock_upload_cls.return_value = mock_uploader

        # CSVGenerator
        mock_csv = MagicMock()
        mock_csv.generate.return_value = "/tmp/test.csv"
        mock_csv_cls.return_value = mock_csv

        return (
            mock_meta,
            mock_sched,
            mock_social,
            mock_uploader,
            mock_csv,
            mock_drive_schedule,
        )

    @patch("gomoveshift.pipeline.drive_phases.phase_d.CSVGenerator")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineUploader")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.SocialGenerator")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineScheduler")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.LongFormMetadataGenerator")
    def test_phase_d_generates_metadata_per_lf(
        self,
        mock_meta_cls,
        mock_sched_cls,
        mock_social_cls,
        mock_upload_cls,
        mock_csv_cls,
        tmp_path,
    ):
        """Phase D calls LongFormMetadataGenerator.generate once per LF."""
        state = self._make_phase_d_state(tmp_path, num_lfs=2)
        mock_meta, *_ = self._setup_phase_d_mocks(
            mock_meta_cls,
            mock_sched_cls,
            mock_social_cls,
            mock_upload_cls,
            mock_csv_cls,
            num_lfs=2,
        )

        DrivePipelineOrchestrator()._phase_d_metadata_and_publish(state)

        assert mock_meta.generate.call_count == 2
        # Verify location and publish_day passed
        for call in mock_meta.generate.call_args_list:
            assert call.kwargs["location"] == "Cork, Ireland"
            assert call.kwargs["publish_day"] == "Wednesday"
            assert call.kwargs["video_duration"] == 600.0

    @patch("gomoveshift.pipeline.drive_phases.phase_d.CSVGenerator")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineUploader")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.SocialGenerator")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineScheduler")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.LongFormMetadataGenerator")
    def test_phase_d_generates_drive_schedule(
        self,
        mock_meta_cls,
        mock_sched_cls,
        mock_social_cls,
        mock_upload_cls,
        mock_csv_cls,
        tmp_path,
    ):
        """Phase D calls PipelineScheduler.generate_drive_schedule."""
        state = self._make_phase_d_state(tmp_path, num_lfs=2)
        _, mock_sched, *_ = self._setup_phase_d_mocks(
            mock_meta_cls,
            mock_sched_cls,
            mock_social_cls,
            mock_upload_cls,
            mock_csv_cls,
            num_lfs=2,
        )

        DrivePipelineOrchestrator()._phase_d_metadata_and_publish(state)

        mock_sched.generate_drive_schedule.assert_called_once()
        call_kwargs = mock_sched.generate_drive_schedule.call_args.kwargs
        assert len(call_kwargs["regular_lfs"]) == 2
        assert call_kwargs["region"] == "ireland"

    @patch("gomoveshift.pipeline.drive_phases.phase_d.CSVGenerator")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineUploader")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.SocialGenerator")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineScheduler")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.LongFormMetadataGenerator")
    def test_phase_d_social_per_lf_with_series_part(
        self,
        mock_meta_cls,
        mock_sched_cls,
        mock_social_cls,
        mock_upload_cls,
        mock_csv_cls,
        tmp_path,
    ):
        """Phase D creates SocialGenerator per LF with series_part=lf.index."""
        state = self._make_phase_d_state(tmp_path, num_lfs=2, num_shorts_per_lf=7)
        self._setup_phase_d_mocks(
            mock_meta_cls,
            mock_sched_cls,
            mock_social_cls,
            mock_upload_cls,
            mock_csv_cls,
            num_lfs=2,
        )

        DrivePipelineOrchestrator()._phase_d_metadata_and_publish(state)

        # SocialGenerator instantiated twice (once per LF)
        assert mock_social_cls.call_count == 2
        # Verify series_part set per LF
        call1_kwargs = mock_social_cls.call_args_list[0].kwargs
        call2_kwargs = mock_social_cls.call_args_list[1].kwargs
        assert call1_kwargs["series_part"] == 1
        assert call2_kwargs["series_part"] == 2
        assert call1_kwargs["location"] == "Cork, Ireland"

    @patch("gomoveshift.pipeline.drive_phases.phase_d.CSVGenerator")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineUploader")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.SocialGenerator")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineScheduler")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.LongFormMetadataGenerator")
    def test_phase_d_uploads_per_lf(
        self,
        mock_meta_cls,
        mock_sched_cls,
        mock_social_cls,
        mock_upload_cls,
        mock_csv_cls,
        tmp_path,
    ):
        """Phase D calls PipelineUploader.upload_shorts per LF."""
        state = self._make_phase_d_state(tmp_path, num_lfs=2)
        _, _, _, mock_uploader, *_ = self._setup_phase_d_mocks(
            mock_meta_cls,
            mock_sched_cls,
            mock_social_cls,
            mock_upload_cls,
            mock_csv_cls,
            num_lfs=2,
        )

        DrivePipelineOrchestrator()._phase_d_metadata_and_publish(state)

        assert mock_uploader.upload_shorts.call_count == 2

    @patch("gomoveshift.pipeline.drive_phases.phase_d.CSVGenerator")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineUploader")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.SocialGenerator")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineScheduler")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.LongFormMetadataGenerator")
    def test_phase_d_skips_upload_when_flagged(
        self,
        mock_meta_cls,
        mock_sched_cls,
        mock_social_cls,
        mock_upload_cls,
        mock_csv_cls,
        tmp_path,
    ):
        """Phase D skips upload when skip_upload=True."""
        state = self._make_phase_d_state(tmp_path, num_lfs=1)
        state["skip_upload"] = True
        self._setup_phase_d_mocks(
            mock_meta_cls,
            mock_sched_cls,
            mock_social_cls,
            mock_upload_cls,
            mock_csv_cls,
        )

        DrivePipelineOrchestrator()._phase_d_metadata_and_publish(state)

        mock_upload_cls.assert_not_called()

    @patch("gomoveshift.pipeline.drive_phases.phase_d.CSVGenerator")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineUploader")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.SocialGenerator")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineScheduler")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.LongFormMetadataGenerator")
    def test_phase_d_generates_csv_per_lf(
        self,
        mock_meta_cls,
        mock_sched_cls,
        mock_social_cls,
        mock_upload_cls,
        mock_csv_cls,
        tmp_path,
    ):
        """Phase D writes ONE consolidated shorts CSV (not per-LF)."""
        state = self._make_phase_d_state(tmp_path, num_lfs=2)
        self._setup_phase_d_mocks(
            mock_meta_cls,
            mock_sched_cls,
            mock_social_cls,
            mock_upload_cls,
            mock_csv_cls,
            num_lfs=2,
        )

        DrivePipelineOrchestrator()._phase_d_metadata_and_publish(state)

        # Exactly one consolidated CSV written to disk
        csvs = list((tmp_path / "output").glob("*_shorts_metricool.csv"))
        assert len(csvs) == 1
        assert "source_drive" in csvs[0].name

    @patch("gomoveshift.pipeline.drive_phases.phase_d.CSVGenerator")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineUploader")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.SocialGenerator")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineScheduler")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.LongFormMetadataGenerator")
    def test_phase_d_skips_csv_when_flagged(
        self,
        mock_meta_cls,
        mock_sched_cls,
        mock_social_cls,
        mock_upload_cls,
        mock_csv_cls,
        tmp_path,
    ):
        """Phase D skips CSV when skip_csv=True."""
        state = self._make_phase_d_state(tmp_path, num_lfs=1)
        state["skip_csv"] = True
        self._setup_phase_d_mocks(
            mock_meta_cls,
            mock_sched_cls,
            mock_social_cls,
            mock_upload_cls,
            mock_csv_cls,
        )

        DrivePipelineOrchestrator()._phase_d_metadata_and_publish(state)

        mock_csv_cls.assert_not_called()

    @patch("gomoveshift.pipeline.drive_phases.phase_d.CSVGenerator")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineUploader")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.SocialGenerator")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineScheduler")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.LongFormMetadataGenerator")
    def test_phase_d_empty_plan_noop(
        self,
        mock_meta_cls,
        mock_sched_cls,
        mock_social_cls,
        mock_upload_cls,
        mock_csv_cls,
        tmp_path,
    ):
        """Phase D with no drive plan sets empty state."""
        state = self._make_phase_d_state(tmp_path, num_lfs=0)
        state["drive_plan"] = None

        DrivePipelineOrchestrator()._phase_d_metadata_and_publish(state)

        mock_meta_cls.assert_not_called()
        mock_sched_cls.assert_not_called()
        assert state["metadata_per_lf"] == {}
        assert state["drive_schedule"] is None

    @patch("gomoveshift.pipeline.drive_phases.phase_d.CSVGenerator")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineUploader")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.SocialGenerator")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineScheduler")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.LongFormMetadataGenerator")
    def test_phase_d_populates_state(
        self,
        mock_meta_cls,
        mock_sched_cls,
        mock_social_cls,
        mock_upload_cls,
        mock_csv_cls,
        tmp_path,
    ):
        """Phase D sets metadata_per_lf, drive_schedule, social_posts_per_lf, upload_manifests."""
        state = self._make_phase_d_state(tmp_path, num_lfs=1)
        self._setup_phase_d_mocks(
            mock_meta_cls,
            mock_sched_cls,
            mock_social_cls,
            mock_upload_cls,
            mock_csv_cls,
        )

        DrivePipelineOrchestrator()._phase_d_metadata_and_publish(state)

        assert 1 in state["metadata_per_lf"]
        assert state["drive_schedule"] is not None
        assert 1 in state["social_posts_per_lf"]
        assert 1 in state["upload_manifests"]

    @patch("gomoveshift.pipeline.drive_phases.phase_d.CSVGenerator")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineUploader")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.SocialGenerator")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineScheduler")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.LongFormMetadataGenerator")
    def test_phase_d_saves_metadata_json(
        self,
        mock_meta_cls,
        mock_sched_cls,
        mock_social_cls,
        mock_upload_cls,
        mock_csv_cls,
        tmp_path,
    ):
        """Phase D saves metadata JSON per LF."""
        state = self._make_phase_d_state(tmp_path, num_lfs=1)
        self._setup_phase_d_mocks(
            mock_meta_cls,
            mock_sched_cls,
            mock_social_cls,
            mock_upload_cls,
            mock_csv_cls,
        )

        DrivePipelineOrchestrator()._phase_d_metadata_and_publish(state)

        meta_file = tmp_path / "output" / "source_drive_lf01_youtube.json"
        assert meta_file.exists()

    # Super-LF-based tests removed (super LF step deleted entirely).
    # Kept as dummy so the test class contract documents the removal explicitly.
    @patch("gomoveshift.pipeline.drive_phases.phase_d.CSVGenerator")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineUploader")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.SocialGenerator")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineScheduler")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.LongFormMetadataGenerator")
    def test_phase_d_populates_lf_drive_links_in_state(
        self,
        mock_meta_cls,
        mock_sched_cls,
        mock_social_cls,
        mock_upload_cls,
        mock_csv_cls,
        tmp_path,
    ):
        """LFs no longer go to Drive — upload_single_file MUST NOT be called."""
        state = self._make_phase_d_state(tmp_path, num_lfs=2)
        _, _, _, mock_uploader, *_ = self._setup_phase_d_mocks(
            mock_meta_cls,
            mock_sched_cls,
            mock_social_cls,
            mock_upload_cls,
            mock_csv_cls,
            num_lfs=2,
        )

        DrivePipelineOrchestrator()._phase_d_metadata_and_publish(state)

        mock_uploader.upload_single_file.assert_not_called()
        # Drive-link state key is gone; YouTube upload state is populated instead.
        assert "lf_drive_links" not in state
        assert "youtube_uploads" in state

    @patch("gomoveshift.pipeline.drive_phases.phase_d.CSVGenerator")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineUploader")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.SocialGenerator")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineScheduler")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.LongFormMetadataGenerator")
    def test_phase_d_full_integration(
        self,
        mock_meta_cls,
        mock_sched_cls,
        mock_social_cls,
        mock_upload_cls,
        mock_csv_cls,
        tmp_path,
    ):
        """Integration: 2 regular LFs, no super LF, consolidated CSV."""
        state = self._make_phase_d_state(tmp_path, num_lfs=2, num_shorts_per_lf=5)

        mock_meta, mock_sched, mock_social, mock_uploader, mock_csv, mock_drive_schedule = (
            self._setup_phase_d_mocks(
                mock_meta_cls,
                mock_sched_cls,
                mock_social_cls,
                mock_upload_cls,
                mock_csv_cls,
                num_lfs=2,
                num_shorts_per_lf=5,
            )
        )

        DrivePipelineOrchestrator()._phase_d_metadata_and_publish(state)

        # 1. Metadata generated for each regular LF (no super LF calls)
        regular_calls = [
            c
            for c in mock_meta.generate.call_args_list
            if "super" not in str(c.kwargs.get("project_name", ""))
        ]
        assert len(regular_calls) == 2
        for c in regular_calls:
            assert c.kwargs["video_duration"] == 600.0

        # 2. Drive schedule generated
        mock_sched.generate_drive_schedule.assert_called_once()
        sched_kwargs = mock_sched.generate_drive_schedule.call_args.kwargs
        assert len(sched_kwargs["regular_lfs"]) == 2

        # 3. Social posts created per LF with series_part
        assert mock_social_cls.call_count == 2
        assert mock_social_cls.call_args_list[0].kwargs["series_part"] == 1
        assert mock_social_cls.call_args_list[1].kwargs["series_part"] == 2

        # 4. Shorts uploaded to Drive — but LFs are NOT (they go to YouTube).
        assert mock_uploader.upload_shorts.call_count == 2
        mock_uploader.upload_single_file.assert_not_called()

        # 5. State is populated with the new contract.
        assert len(state["metadata_per_lf"]) == 2
        assert state["drive_schedule"] is not None
        assert len(state["social_posts_per_lf"]) == 2
        assert "youtube_uploads" in state
        assert "lf_drive_links" not in state


# ── Snapshot path contract ────────────────────────────


class TestSnapshotPath:
    """Snapshot dir for DoD validator must be stages_base.parent, not stages_base."""

    @patch("gomoveshift.pipeline.drive_orchestrator.run_validation")
    @patch("gomoveshift.pipeline.drive_orchestrator.ReviewGate")
    @patch(
        "gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator"
        "._phase_d_metadata_and_publish"
    )
    @patch("gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator._phase_bc_day_first")
    @patch(
        "gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator"
        "._phase_a_ingest_and_analyze"
    )
    def test_snapshot_dir_is_parent_of_stages_base(
        self, mock_a, mock_bc, mock_d, mock_rg, mock_validate, tmp_path
    ):
        """snapshot_dir must be stages_base.parent / '_snapshot_pre_fix'."""
        # Mock validator returns a passing report
        report = MagicMock()
        report.passed_count = 13
        report.checks = [MagicMock()] * 13
        report.failed = []
        report.failed_count = 0
        mock_validate.return_value = report

        # Populate state["regular_lf_paths"] so Layer-1 invariants pass when
        # phase_bc_day_first completes. See state_invariants Rule 5.
        def _bc_side_effect(state):
            state["regular_lf_paths"] = ["/mock/lf_01.mp4"]
            state["shorts_per_lf"] = {1: []}
            return {}

        mock_bc.side_effect = _bc_side_effect

        source = tmp_path / "source"
        source.mkdir()

        orch = DrivePipelineOrchestrator()
        orch.run(
            source_dir=source,
            output_base=tmp_path / "output",
            region="ireland",
        )

        mock_validate.assert_called_once()
        call_kwargs = mock_validate.call_args
        snapshot_dir = call_kwargs.kwargs.get("snapshot_dir") or call_kwargs[1].get("snapshot_dir")
        if snapshot_dir is None:
            # positional arg
            snapshot_dir = call_kwargs[0][2]

        # The snapshot dir must NOT be under _stages/ — it should be one level up
        assert "_stages" not in str(snapshot_dir) or str(snapshot_dir).endswith(
            "_snapshot_pre_fix"
        ), f"snapshot_dir should not be under _stages: {snapshot_dir}"
        assert str(snapshot_dir).endswith("_snapshot_pre_fix")
        # Must be stages_base.parent / _snapshot_pre_fix
        assert snapshot_dir.parent.name != "_stages", (
            f"snapshot_dir parent should not be _stages, got {snapshot_dir.parent}"
        )


# ── Checkpoint / Resume ────────────────────────────────────────────────


class TestProjectDirectoryIsolation:
    """Output must go under output_base/project_name/ not flat in output_base."""

    @patch(
        "gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator._phase_d_metadata_and_publish"
    )
    @patch("gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator._phase_bc_day_first")
    @patch(
        "gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator._phase_a_ingest_and_analyze"
    )
    def test_output_under_project_subdir(self, mock_a, mock_bc, mock_d, tmp_path):
        """Pipeline state file should be under output_base/project_name/_stages/."""
        source = tmp_path / "Project2026"
        source.mkdir()

        orch = DrivePipelineOrchestrator()
        orch.run(source_dir=source, output_base=tmp_path / "output")

        # Output should be under output/project2026/ not output/
        project_dir = tmp_path / "output" / "project2026"
        assert project_dir.exists(), "Project dir must be created"
        state_file = project_dir / "_stages" / "pipeline_state.json"
        assert state_file.exists(), "State file must be under project dir"

    @patch(
        "gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator._phase_d_metadata_and_publish"
    )
    @patch("gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator._phase_bc_day_first")
    @patch(
        "gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator._phase_a_ingest_and_analyze"
    )
    def test_result_output_dir_is_project_dir(self, mock_a, mock_bc, mock_d, tmp_path):
        """DrivePipelineResult.output_dir should point to the project dir."""
        source = tmp_path / "MyDrive"
        source.mkdir()

        orch = DrivePipelineOrchestrator()
        result = orch.run(source_dir=source, output_base=tmp_path / "output")

        expected = tmp_path / "output" / "mydrive"
        assert result.output_dir == expected


# ── Publish-Priority Encoding ─────────────────────────────────────────


class TestSaveStateHardening:
    """_save_state must handle sets and write errors gracefully."""

    def test_save_state_serializes_set_as_sorted_list(self, tmp_path):
        """A set value must be serialized as a sorted list in the JSON output."""
        import json

        state_file = tmp_path / "pipeline_state.json"
        state = make_state(delivered_lfs={2, 0, 1})

        DrivePipelineOrchestrator()._save_state(state_file, state)

        with open(state_file) as f:
            data = json.load(f)

        assert data["delivered_lfs"] == [0, 1, 2]

    def test_delivered_lfs_restored_as_set_via_phase_c_state_get(self, tmp_path):
        """Round-trip: set saved → list in JSON → set() coercion in Phase C restore."""
        import json

        state_file = tmp_path / "pipeline_state.json"
        state = make_state(delivered_lfs={0, 1, 2})

        DrivePipelineOrchestrator()._save_state(state_file, state)

        with open(state_file) as f:
            loaded = json.load(f)

        # Simulate Phase C's state.get("delivered_lfs", set()) with set() coercion
        delivered_lfs = set(loaded.get("delivered_lfs", set()))

        assert isinstance(delivered_lfs, set)
        assert delivered_lfs == {0, 1, 2}

    def test_save_state_raises_on_write_failure(self, tmp_path):
        """Atomic write failure must propagate — caller handles errors."""
        import gomoveshift.pipeline._drive_state as ds_mod
        from unittest.mock import patch as mock_patch

        state_file = tmp_path / "pipeline_state.json"
        state = {"some_key": "some_value"}

        with mock_patch.object(ds_mod.json, "dump", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                DrivePipelineOrchestrator()._save_state(state_file, state)

    def test_save_state_preserves_int_keyed_dict_of_lists(self, tmp_path):
        """Int-keyed dict with list values serializes correctly.

        The recursive serializer preserves list values inside int-keyed
        dicts rather than str-coercing them.
        """
        import json

        state_file = tmp_path / "pipeline_state.json"
        state = make_state(shorts_per_lf={0: [1, 2, 3]})

        # Must not raise
        DrivePipelineOrchestrator()._save_state(state_file, state)

        with open(state_file) as f:
            data = json.load(f)

        # Key is present and list value is correctly preserved
        assert "shorts_per_lf" in data
        assert data["shorts_per_lf"] == {"0": [1, 2, 3]}


# ── Phase B Disk Space Preflight ─────────────────────────────────────────────


class TestLoadStateValidation:
    """_load_state must validate phases_completed to prevent corrupt state from propagating."""

    def test_phases_completed_string_treated_as_invalid(self, tmp_path):
        """If phases_completed is a string (not a list), it must be treated as invalid."""
        import json

        state_file = tmp_path / "pipeline_state.json"
        state_file.write_text(json.dumps({"phases_completed": "phase_a,phase_b"}))
        orch = DrivePipelineOrchestrator()
        result = orch._load_state(state_file)
        # Either returns None or sanitizes phases_completed to a list
        if result is not None and "phases_completed" in result:
            assert isinstance(result["phases_completed"], list), (
                "phases_completed must be a list; string value must be rejected"
            )

    def test_phases_completed_unknown_values_are_filtered(self, tmp_path):
        """Unknown phase names must be stripped — only known phases survive."""
        import json

        state_file = tmp_path / "pipeline_state.json"
        state_file.write_text(
            json.dumps({"phases_completed": ["phase_a", "phase_x_unknown", "phase_inject"]})
        )
        orch = DrivePipelineOrchestrator()
        result = orch._load_state(state_file)
        if result and "phases_completed" in result:
            known = {"phase_a", "phase_b_preview", "phase_bc_day_first", "phase_d"}
            for ph in result["phases_completed"]:
                assert ph in known, f"Unknown phase '{ph}' must be filtered from loaded state"

    def test_valid_state_is_unchanged(self, tmp_path):
        """A well-formed state file must load without modification."""
        import json

        state_file = tmp_path / "pipeline_state.json"
        state_file.write_text(
            json.dumps(
                {
                    "phases_completed": ["phase_a", "phase_bc_day_first"],
                    "location": "Test Location, France",
                }
            )
        )
        orch = DrivePipelineOrchestrator()
        result = orch._load_state(state_file)
        assert result is not None
        assert result["phases_completed"] == ["phase_a", "phase_bc_day_first"]


class TestPreviewMode:
    """Phase B preview: encode 90s clip from middle of each episode, then exit."""

    def test_preview_exits_after_phase_a(self, tmp_path):
        """With preview=True, pipeline runs phase_a then preview, not phase_bc_day_first."""
        from gomoveshift.config import Config

        config = Config()
        config.drive.preview = True
        orch = DrivePipelineOrchestrator(config=config)

        mock_a = MagicMock()
        mock_preview = MagicMock()
        mock_bc = MagicMock()

        mock_a.return_value = None
        mock_preview.return_value = None
        mock_bc.return_value = None

        with patch.object(orch, "_phase_a_ingest_and_analyze", mock_a):
            with patch.object(orch, "_phase_b_preview", mock_preview):
                with patch.object(orch, "_phase_bc_day_first", mock_bc):
                    source = tmp_path / "source"
                    source.mkdir()
                    orch.run(
                        source_dir=source,
                        output_base=tmp_path / "output",
                        region="ireland",
                    )

        mock_a.assert_called_once()
        mock_preview.assert_called_once()
        mock_bc.assert_not_called()

    def test_preview_method_exists(self):
        """DrivePipelineOrchestrator must have _phase_b_preview method."""
        orch = DrivePipelineOrchestrator()
        assert hasattr(orch, "_phase_b_preview"), "_phase_b_preview method is required"

    def test_preview_wires_review_gate(self, tmp_path):
        """gate_preview is
        wired into the --preview path."""
        from gomoveshift.config import Config

        config = Config()
        config.drive.preview = True
        orch = DrivePipelineOrchestrator(config=config)
        source = tmp_path / "source"
        source.mkdir()

        with (
            patch.object(orch, "_phase_a_ingest_and_analyze", MagicMock(return_value=None)),
            patch.object(orch, "_phase_b_preview", MagicMock(return_value=None)),
            patch("gomoveshift.pipeline.drive_orchestrator.ReviewGate") as mock_rg,
        ):
            orch.run(source_dir=source, output_base=tmp_path / "out", region="ireland")

        mock_rg.return_value.gate_preview.assert_called_once()

    def test_preview_config_field_exists(self):
        """DriveConfig must have preview: bool = False field."""

        cfg = DriveConfig()
        assert hasattr(cfg, "preview"), "DriveConfig must have preview field"
        assert cfg.preview is False, "preview must default to False"

    def test_preview_encodes_90s_from_middle(self, tmp_path, monkeypatch):
        """Preview encode targets 90s from the middle of each episode."""
        import gomoveshift.pipeline.drive_orchestrator as orch_mod
        from gomoveshift.config import Config

        config = Config()
        orch = DrivePipelineOrchestrator(config=config)

        # State with one episode file (fake path) and duration 300s
        ep_file = tmp_path / "ep_000.mp4"
        ep_file.touch()

        stages_dir = tmp_path / "_stages"
        stages_dir.mkdir(parents=True, exist_ok=True)

        state = make_state(
            episode_files=[ep_file],
            output_base=tmp_path,
        )

        captured_cmds = []

        def fake_run(cmd, *args, **kwargs):
            captured_cmds.append(list(cmd))
            result = MagicMock()
            result.returncode = 0
            result.stdout = b"300.0"
            result.stderr = b""
            return result

        import subprocess as _subprocess

        monkeypatch.setattr(_subprocess, "run", fake_run)

        orch._phase_b_preview(state)

        # Should have called ffprobe and ffmpeg
        assert captured_cmds, "No subprocess calls made"
        # Find the ffmpeg encode call (has -t 90 in it)
        encode_cmd = None
        for cmd in captured_cmds:
            cmd_str = " ".join(str(a) for a in cmd)
            if "-t" in cmd_str and "90" in cmd_str and "ffmpeg" in cmd_str:
                encode_cmd = cmd_str
                break

        assert encode_cmd is not None, f"No 90s encode call found. Commands: {captured_cmds}"
        assert "-t" in encode_cmd
        assert "90" in encode_cmd


class TestPreviewBurnsOST:
    """The --preview encode must burn the OST drawtext into its output clips
    so reviewers can visually verify the OST text before Phase B commits to
    full-length encodes. Previously the preview was a plain copy with no
    drawtext filter, making `gate_preview`'s "OST visible + correct?" prompt
    misleading."""

    def _run_preview_with_mock(
        self,
        tmp_path,
        monkeypatch,
        state: dict,
        probe_stdout: bytes = b"3840\n300.0",
    ) -> list[str]:
        """Run _phase_b_preview against a mocked subprocess.run and return
        the captured ffmpeg encode command as a list of strings."""
        import gomoveshift.pipeline.drive_orchestrator as orch_mod
        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_orchestrator import (
            DrivePipelineOrchestrator,
        )

        config = Config()
        orch = DrivePipelineOrchestrator(config=config)

        captured_cmds = []

        def fake_run(cmd, *args, **kwargs):
            captured_cmds.append(list(cmd))
            result = MagicMock()
            result.returncode = 0
            # ffprobe reads width on first line, duration on second
            result.stdout = probe_stdout
            result.stderr = b""
            return result

        import subprocess as _subprocess

        monkeypatch.setattr(_subprocess, "run", fake_run)

        # Ensure the font file the preview looks for appears to exist
        monkeypatch.setattr(
            "pathlib.Path.exists",
            lambda self: True,
        )

        orch._phase_b_preview(state)

        # Find the ffmpeg ENCODE call by its OUTPUT, not by a flag. This phase
        # now also takes a grade-window cut (carries "-t") and reaches the NVENC
        # probe (carries "-c:v"), so only the destination file is unambiguous.
        for cmd in captured_cmds:
            cmd_str_list = [str(a) for a in cmd]
            if "ffmpeg" in cmd_str_list[0] and cmd_str_list[-1].endswith("_preview.mp4"):
                return cmd_str_list
        return []

    def test_preview_command_includes_drawtext_with_episode_ost(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Preview encode for an episode with an OST should include a
        -vf filter containing the OST text via drawtext."""
        ep_file = tmp_path / "ep_000.mp4"
        ep_file.touch()
        (tmp_path / "_stages").mkdir()

        state = make_state(
            episode_files=[ep_file],
            episode_locations=["Port Royal to Beaufort via Trask Parkway"],
            output_base=tmp_path,
        )

        cmd = self._run_preview_with_mock(tmp_path, monkeypatch, state)
        assert cmd, "No ffmpeg encode command captured"

        assert "-vf" in cmd, f"Preview command missing -vf filter: {cmd}"
        vf_idx = cmd.index("-vf")
        vf_value = cmd[vf_idx + 1]
        assert "drawtext" in vf_value, f"-vf filter has no drawtext: {vf_value}"
        assert "Port Royal to Beaufort" in vf_value, f"OST text missing from drawtext: {vf_value}"

    def test_preview_falls_back_to_drive_location_when_episode_empty(
        self,
        tmp_path,
        monkeypatch,
    ):
        """When episode_locations[i] is empty/missing, preview burns the
        drive-level 'location' string instead."""
        ep_file = tmp_path / "ep_000.mp4"
        ep_file.touch()
        (tmp_path / "_stages").mkdir()

        state = make_state(
            episode_files=[ep_file],
            episode_locations=[""],
            location="Longwood to Highlands",
            output_base=tmp_path,
        )

        cmd = self._run_preview_with_mock(tmp_path, monkeypatch, state)
        assert cmd, "No ffmpeg encode command captured"

        assert "-vf" in cmd
        vf_value = cmd[cmd.index("-vf") + 1]
        assert "Longwood to Highlands" in vf_value

    def test_preview_uses_nvenc_same_as_before(self, tmp_path, monkeypatch):
        """Encoder must still be hevc_nvenc — the OST filter is additive,
        not a regression on codec choice."""
        ep_file = tmp_path / "ep_000.mp4"
        ep_file.touch()
        (tmp_path / "_stages").mkdir()

        state = make_state(
            episode_files=[ep_file],
            episode_locations=["Test OST"],
            output_base=tmp_path,
        )

        cmd = self._run_preview_with_mock(tmp_path, monkeypatch, state)
        assert "hevc_nvenc" in cmd
        assert "-t" in cmd and "90" in cmd

    def test_preview_no_location_encodes_without_drawtext(
        self,
        tmp_path,
        monkeypatch,
    ):
        """When neither episode OST nor drive location is set, the preview
        still encodes (for playback verification) but without a -vf filter."""
        ep_file = tmp_path / "ep_000.mp4"
        ep_file.touch()
        (tmp_path / "_stages").mkdir()

        state = make_state(
            episode_files=[ep_file],
            episode_locations=[],
            output_base=tmp_path,
        )

        cmd = self._run_preview_with_mock(tmp_path, monkeypatch, state)
        # Command should exist (encode still runs)
        assert cmd
        # But no -vf since there's no OST to burn
        assert "-vf" not in cmd

    def test_preview_font_scales_with_probed_width(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Preview should pass the probed video_width into _build_video_filter
        so 4K episodes get 96/72 font sizes and 1080p ones get 64/48."""
        ep_file = tmp_path / "ep_000.mp4"
        ep_file.touch()
        (tmp_path / "_stages").mkdir()

        state = make_state(
            episode_files=[ep_file],
            episode_locations=["Test OST"],
            output_base=tmp_path,
        )

        # 4K probe
        cmd_4k = self._run_preview_with_mock(
            tmp_path,
            monkeypatch,
            state,
            probe_stdout=b"3840\n300.0",
        )
        vf_4k = cmd_4k[cmd_4k.index("-vf") + 1]
        assert "fontsize=96" in vf_4k, f"Expected 4K scaled fontsize=96 in: {vf_4k}"


# ── State/resume contract completeness ───────────────────────────


class TestStateResumeContract:
    """Validate the state/resume contract — _KNOWN_PHASES,
    serialization round-trips, and variant phase handling."""

    def test_known_phases_includes_all_variants(self):
        """_KNOWN_PHASES must include every phase the surviving orchestrator can produce.

        Legacy phase_b / phase_c / phase_cd_per_day were deleted.
        """
        from gomoveshift.pipeline._drive_state import _KNOWN_PHASES

        expected = {
            "phase_a",
            "phase_b_preview",
            "phase_bc_day_first",
            "phase_d",
        }
        assert _KNOWN_PHASES == expected

    def test_phase_spec_is_single_source_of_truth(self):
        """Execution order, --phase prereq validation, and resume artifact-gates
        all derive from ONE PHASE_SPECS table — no parallel hardcoded structures.
        """
        from gomoveshift.pipeline import drive_orchestrator as do
        from gomoveshift.pipeline._drive_state import _KNOWN_PHASES

        assert hasattr(do, "PHASE_SPECS"), (
            "expected a single consolidated PHASE_SPECS table driving order, "
            "prereqs, and resume artifact-gates"
        )
        specs = do.PHASE_SPECS

        # Resume artifact-gate map derives from the table.
        assert do.PHASE_ARTIFACTS == {s.name: list(s.artifacts) for s in specs}

        orch = do.DrivePipelineOrchestrator()

        # --phase prereq validation derives from the table.
        assert orch._get_prereqs() == {s.name: list(s.prereqs) for s in specs}

        # Full-run execution order derives from the table, in table order.
        manifest_names = [name for name, _ in orch._select_phase_manifest(None)]
        assert manifest_names == [s.name for s in specs if s.in_manifest]

        # The table stays coherent with the state-layer drift guard.
        assert {s.name for s in specs} == set(_KNOWN_PHASES)

    def test_serialize_state_preserves_dicts_in_lists(self, tmp_path):
        """Lists containing dicts must survive serialization round-trip."""
        import json
        from gomoveshift.pipeline._drive_state import save_drive_state

        state_file = tmp_path / "state.json"
        state = make_state(
            day_artifacts=[
                {"date": "2025-01-01", "lf_path": "/tmp/lf1.mp4"},
                {"date": "2025-01-02", "lf_path": "/tmp/lf2.mp4"},
            ]
        )
        save_drive_state(state_file, state)

        with open(state_file) as f:
            loaded = json.load(f)

        assert "day_artifacts" in loaded
        assert len(loaded["day_artifacts"]) == 2
        assert loaded["day_artifacts"][0]["date"] == "2025-01-01"

    def test_serialize_state_preserves_mixed_list(self, tmp_path):
        """Lists with mixed types (str, dict, Path) must all serialize."""
        import json
        from gomoveshift.pipeline._drive_state import save_drive_state

        state_file = tmp_path / "state.json"
        state = {
            "mixed": ["a_string", {"key": "val"}, Path("/tmp/video.mp4")],
        }
        save_drive_state(state_file, state)

        with open(state_file) as f:
            loaded = json.load(f)

        assert loaded["mixed"] == ["a_string", {"key": "val"}, "/tmp/video.mp4"]

    def test_load_state_preserves_variant_phases(self, tmp_path):
        """Variant phase names must not be filtered out on load."""
        import json
        from gomoveshift.pipeline._drive_state import load_drive_state

        state_file = tmp_path / "state.json"
        state_data = {
            "phases_completed": ["phase_a", "phase_bc_day_first"],
            "location": "Kerry",
        }
        with open(state_file, "w") as f:
            json.dump(state_data, f)

        loaded = load_drive_state(state_file)

        assert loaded is not None
        assert loaded["phases_completed"] == ["phase_a", "phase_bc_day_first"]

    # test_load_state_preserves_per_day_phase removed — phase_cd_per_day
    # was deleted. State files carrying the legacy
    # names (phase_b / phase_c / phase_cd_per_day) used to be silently
    # remapped; that shim is gone and loading now raises instead.
    # See test_load_state_raises_on_stale_legacy_phase below.

    @pytest.mark.parametrize("legacy_phase", ["phase_b", "phase_c", "phase_cd_per_day"])
    def test_load_state_raises_on_stale_legacy_phase(self, tmp_path, legacy_phase):
        """State files with legacy phase names must raise.

        The silent ``_LEGACY_PHASE_MAP`` remap was deleted because
        the only callers of those phase names were the non-day-first
        pipeline (since deleted), and any state file old enough to
        carry them is almost certainly incompatible with the current
        encoder/schedule contracts. Force the operator to delete the
        stale state file and re-run rather than silently mapping into a
        possibly-mismatched checkpoint.
        """
        import json
        from gomoveshift.pipeline._drive_state import load_drive_state

        state_file = tmp_path / "pipeline_state.json"
        state_data = {
            "source_dir": "/data/src",
            "output_base": str(tmp_path),
            "phases_completed": ["phase_a", legacy_phase],
        }
        with open(state_file, "w") as f:
            json.dump(state_data, f)

        with pytest.raises(RuntimeError, match=r"[Ss]tale.*legacy.*state file"):
            load_drive_state(state_file)


# ── DoD validator in single-phase mode ────────────
#
# The terminal DoD validator runs on every successful exit (except an OST-audit pause, ost_audit_pending), including
# single-phase runs such as ``--phase phase_d`` (the resumable re-upload
# path); ``config.drive.skip_validation`` / ``--skip-validation`` is the
# explicit escape hatch. The in-Phase-D quality gate
# (``assert_phase_d_quality``) runs inside ``PhaseD.run``, so it also
# executes in single-phase mode. These tests pin both.


@pytest.fixture
def no_ambient_disk_gate(monkeypatch):
    """Neutralize the resume preflight's free-disk probe.

    ``_check_resume_preflight`` calls ``check_disk_space(output_base,
    min_free_gb=50)`` against the real filesystem holding ``tmp_path``. The
    tests below pin DoD-validator wiring, not capacity, so on any box whose
    tmp filesystem has under 50 GB free they died with ``RuntimeError:
    Insufficient disk space`` before reaching the behavior under test — green
    on a roomy CI runner, red on a full workstation, for reasons having
    nothing to do with the code. Same monkeypatch ``test_resume_invariants``
    already uses.
    """
    monkeypatch.setattr(
        "gomoveshift.pipeline.state_invariants.check_disk_space",
        lambda *a, **kw: None,
    )


@pytest.mark.usefixtures("no_ambient_disk_gate")
class TestDoDValidatorInSinglePhaseMode:
    """DoD validator must run when ``--phase phase_d`` succeeds.

    Sets up a resumable checkpoint (phase_a + phase_bc_day_first artifacts
    already on disk, matching ``PHASE_ARTIFACTS``) so the orchestrator's
    prereq gate lets single-phase mode start, then verifies the DoD
    validator is invoked at terminal exit.
    """

    @staticmethod
    def _seed_resumed_state(tmp_path, project_name="project2026"):
        """Pre-seed a checkpoint + required artifacts so ``phase=phase_d``
        can execute. Returns (source_dir, output_base, state_file).

        Mirrors the on-disk shape the orchestrator expects after phases A
        and BC already completed in a prior run — so this test exercises
        the single-phase "I already have day-first output, just re-upload"
        workflow.
        """
        import json

        source = tmp_path / project_name.upper()
        source.mkdir()
        output_base = tmp_path / "output"
        stages_dir = output_base / project_name / "_stages"
        stages_dir.mkdir(parents=True, exist_ok=True)

        # Phase artifacts declared in PHASE_ARTIFACTS — must exist on disk
        # or the orchestrator's artifact-drift gate will drop the phase
        # from resumed_phases and force a re-run. drive_scores.npz must
        # be a genuine numpy archive (the resume path loads it via
        # UnifiedSegmentScorer.load_scores → np.load) so write a tiny
        # valid placeholder instead of arbitrary bytes.
        np.savez(
            stages_dir / "drive_scores.npz",
            scores=np.array([[0.0, 1.0, 0.5]], dtype=np.float32),
            source_files=np.array(["ep0.mp4"]),
        )
        (stages_dir / "drive_plan.json").write_text("{}")

        # Saved state file carrying the two completed phases.
        # per_lf_recording_date + per_lf_trim_ratio
        # are now contract members; the resume invariants strict-raise on
        # empty maps under completed phase_bc_day_first. Pre-populate so
        # the fixture clears resume_invariants without touching the
        # behavior under test (DoD validator wiring).
        state_data = {
            "phases_completed": ["phase_a", "phase_bc_day_first"],
            "regular_lf_paths": ["/mock/lf_01.mp4"],
            "shorts_per_lf": {"1": []},
            "output_base": str(output_base / project_name),
            "per_lf_recording_date": {"1": "2026-02-20"},
            "per_lf_trim_ratio": {"1": 0.5},
            "per_lf_gpx_path": {"1": "/mock/g01.gpx"},
        }
        state_file = stages_dir / "pipeline_state.json"
        with open(state_file, "w") as f:
            json.dump(state_data, f)
        return source, output_base, state_file

    @patch("gomoveshift.pipeline.drive_orchestrator.run_validation")
    @patch("gomoveshift.pipeline.drive_orchestrator.ReviewGate")
    @patch(
        "gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator"
        "._phase_d_metadata_and_publish"
    )
    def test_dod_runs_on_single_phase_phase_d(self, mock_d, mock_rg, mock_validate, tmp_path):
        """Regression pin: ``--phase phase_d`` success exit
        MUST invoke the DoD validator; a ``phase is None`` guard would skip it.
        """
        # Validator returns a passing report.
        report = MagicMock()
        report.passed_count = 13
        report.checks = [MagicMock()] * 13
        report.failed = []
        report.failed_count = 0
        mock_validate.return_value = report

        source, output_base, _ = self._seed_resumed_state(tmp_path)
        mock_d.return_value = {}

        orch = DrivePipelineOrchestrator()
        result = orch.run(
            source_dir=source,
            output_base=output_base,
            region="ireland",
            phase="phase_d",
        )

        assert result.success is True, f"single-phase phase_d should succeed; got {result.errors}"
        # There is no ``phase is None`` guard: a successful phase_d exit must call it once.
        assert mock_validate.call_count == 1, (
            "DoD validator must run on single-phase phase_d success — "
            "no ``phase is None`` guard may skip it; "
            f"got call_count={mock_validate.call_count}"
        )

    @patch("gomoveshift.pipeline.drive_orchestrator.run_validation")
    @patch("gomoveshift.pipeline.drive_orchestrator.ReviewGate")
    @patch(
        "gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator"
        "._phase_d_metadata_and_publish"
    )
    def test_dod_respects_skip_validation_in_single_phase(
        self, mock_d, mock_rg, mock_validate, tmp_path
    ):
        """Escape hatch still works: ``config.drive.skip_validation=True`` skips
        the DoD validator in single-phase mode too. Keeps the emergency-only
        flag usable for operators who already know the batch is fine and need
        to push past a transient validator glitch.
        """
        mock_d.return_value = {}
        source, output_base, _ = self._seed_resumed_state(tmp_path)

        cfg = Config()
        cfg.drive.skip_validation = True
        orch = DrivePipelineOrchestrator(config=cfg)
        result = orch.run(
            source_dir=source,
            output_base=output_base,
            region="ireland",
            phase="phase_d",
        )

        assert result.success is True
        assert mock_validate.call_count == 0, (
            "skip_validation=True must bypass the DoD validator on single-phase "
            f"exits too — escape hatch (emergency only); got call_count={mock_validate.call_count}"
        )

    @patch("gomoveshift.pipeline.drive_orchestrator.run_validation")
    @patch("gomoveshift.pipeline.drive_orchestrator.ReviewGate")
    @patch(
        "gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator"
        "._phase_d_metadata_and_publish"
    )
    def test_dod_not_run_on_failure(self, mock_d, mock_rg, mock_validate, tmp_path):
        """Pre-existing behavior: the DoD validator only runs on ``success``.
        If the phase loop accumulated errors, the root cause is already
        visible and running the validator on top would just add noise.
        """
        # Phase D raises — the phase loop traps this and appends to errors.
        mock_d.side_effect = RuntimeError("simulated phase_d failure")

        source, output_base, _ = self._seed_resumed_state(tmp_path)

        orch = DrivePipelineOrchestrator()
        result = orch.run(
            source_dir=source,
            output_base=output_base,
            region="ireland",
            phase="phase_d",
        )

        assert result.success is False
        assert mock_validate.call_count == 0, (
            "DoD validator must NOT run on phase failure — pre-existing "
            f"behavior the fix must preserve; got call_count={mock_validate.call_count}"
        )


@pytest.mark.usefixtures("no_ambient_disk_gate")
class TestDoDLfDurationCapByLfMode:
    """``max_lf_duration`` passed to ``run_validation`` must reflect the
    configured ``lf_selection_mode``. ``best_moments`` (default) → 1800.0;
    ``full_drive`` → ``config.drive.full_drive_max_duration_sec`` (default
    7200). Without that, full_drive LFs > 30 min would always hard-fail the
    LF-duration DoD check even though the planner accepted the duration upstream.
    """

    @patch("gomoveshift.pipeline.drive_orchestrator.run_validation")
    @patch("gomoveshift.pipeline.drive_orchestrator.ReviewGate")
    @patch(
        "gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator"
        "._phase_d_metadata_and_publish"
    )
    def test_full_drive_lifts_max_lf_duration_to_config_ceiling(
        self, mock_d, mock_rg, mock_validate, tmp_path
    ):
        report = MagicMock()
        report.passed_count = 13
        report.checks = [MagicMock()] * 13
        report.failed = []
        report.failed_count = 0
        mock_validate.return_value = report
        mock_d.return_value = {}

        source, output_base, _ = TestDoDValidatorInSinglePhaseMode._seed_resumed_state(tmp_path)

        cfg = Config()
        cfg.drive.lf_selection_mode = "full_drive"
        cfg.drive.full_drive_max_duration_sec = 7200
        orch = DrivePipelineOrchestrator(config=cfg)
        orch.run(
            source_dir=source,
            output_base=output_base,
            region="ireland",
            phase="phase_d",
        )

        assert mock_validate.call_count == 1
        kwargs = mock_validate.call_args.kwargs
        assert kwargs["max_lf_duration"] == 7200.0, (
            f"full_drive must lift cap to full_drive_max_duration_sec; got {kwargs['max_lf_duration']}"
        )

    @patch("gomoveshift.pipeline.drive_orchestrator.run_validation")
    @patch("gomoveshift.pipeline.drive_orchestrator.ReviewGate")
    @patch(
        "gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator"
        "._phase_d_metadata_and_publish"
    )
    def test_best_moments_keeps_default_1800_cap(self, mock_d, mock_rg, mock_validate, tmp_path):
        report = MagicMock()
        report.passed_count = 13
        report.checks = [MagicMock()] * 13
        report.failed = []
        report.failed_count = 0
        mock_validate.return_value = report
        mock_d.return_value = {}

        source, output_base, _ = TestDoDValidatorInSinglePhaseMode._seed_resumed_state(tmp_path)

        cfg = Config()
        cfg.drive.lf_selection_mode = "best_moments"
        orch = DrivePipelineOrchestrator(config=cfg)
        orch.run(
            source_dir=source,
            output_base=output_base,
            region="ireland",
            phase="phase_d",
        )

        kwargs = mock_validate.call_args.kwargs
        assert kwargs["max_lf_duration"] == 1800.0


class TestPipelineSummaryLogLine:
    """Structured PIPELINE_SUMMARY exit line for grep-based observability.

    Long batches need a greppable exit summary rather than scrolling logs
    for totals, so run() ends with a single structured INFO line and
    ``grep PIPELINE_SUMMARY`` surfaces every run's outcome.

    Sentinel format:
      PIPELINE_SUMMARY source=<name> lfs=<N> shorts=<N>
        youtube_uploaded=<true|false|skipped> phases=<done>/<total>
        duration=<float>s
    """

    @patch("gomoveshift.pipeline.drive_orchestrator.ReviewGate")
    @patch(
        "gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator"
        "._phase_d_metadata_and_publish"
    )
    @patch("gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator._phase_bc_day_first")
    @patch(
        "gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator"
        "._phase_a_ingest_and_analyze"
    )
    def test_pipeline_summary_logged_with_sentinel_prefix(
        self, mock_a, mock_bc, mock_d, mock_rg, tmp_path, caplog
    ):
        """Successful run emits a single PIPELINE_SUMMARY INFO line with all 6 fields."""
        import logging as _logging

        def _bc_side_effect(state):
            state["regular_lf_paths"] = ["/mock/lf_01.mp4", "/mock/lf_02.mp4"]
            state["shorts_per_lf"] = {1: ["/mock/s1.mp4"], 2: ["/mock/s2.mp4", "/mock/s3.mp4"]}
            return {}

        mock_a.return_value = {}
        mock_bc.side_effect = _bc_side_effect
        mock_d.return_value = {}

        source = tmp_path / "source"
        source.mkdir()

        cfg = Config()
        cfg.drive.skip_validation = True
        orch = DrivePipelineOrchestrator(config=cfg)

        with caplog.at_level(_logging.INFO, logger="gomoveshift.pipeline.drive_orchestrator"):
            orch.run(
                source_dir=source,
                output_base=tmp_path / "output",
                region="ireland",
                skip_upload=True,
            )

        summary_lines = [rec.message for rec in caplog.records if "PIPELINE_SUMMARY" in rec.message]
        assert len(summary_lines) == 1, (
            f"Expected exactly one PIPELINE_SUMMARY line, got {len(summary_lines)}: {summary_lines}"
        )
        line = summary_lines[0]
        assert line.startswith("PIPELINE_SUMMARY "), f"missing sentinel prefix in: {line}"
        for field in ("source=", "lfs=", "shorts=", "youtube_uploaded=", "phases=", "duration="):
            assert field in line, f"missing field {field!r} in: {line}"

    @patch("gomoveshift.pipeline.drive_orchestrator.ReviewGate")
    @patch(
        "gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator"
        "._phase_d_metadata_and_publish"
    )
    @patch("gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator._phase_bc_day_first")
    @patch(
        "gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator"
        "._phase_a_ingest_and_analyze"
    )
    def test_pipeline_summary_counts_match_state(
        self, mock_a, mock_bc, mock_d, mock_rg, tmp_path, caplog
    ):
        """lfs= and shorts= fields reflect state counts, not hardcoded values."""
        import logging as _logging

        def _bc_side_effect(state):
            state["regular_lf_paths"] = ["/a", "/b", "/c"]
            state["shorts_per_lf"] = {
                1: ["/s1"],
                2: ["/s2a", "/s2b"],
                3: ["/s3a", "/s3b"],
            }
            return {}

        mock_a.return_value = {}
        mock_bc.side_effect = _bc_side_effect
        mock_d.return_value = {}

        source = tmp_path / "source"
        source.mkdir()

        cfg = Config()
        cfg.drive.skip_validation = True
        orch = DrivePipelineOrchestrator(config=cfg)

        with caplog.at_level(_logging.INFO, logger="gomoveshift.pipeline.drive_orchestrator"):
            orch.run(
                source_dir=source,
                output_base=tmp_path / "output",
                region="ireland",
                skip_upload=True,
            )

        summary = next(rec.message for rec in caplog.records if "PIPELINE_SUMMARY" in rec.message)
        assert "lfs=3" in summary, summary
        assert "shorts=5" in summary, summary

    @patch("gomoveshift.pipeline.drive_orchestrator.ReviewGate")
    @patch(
        "gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator"
        "._phase_d_metadata_and_publish"
    )
    @patch("gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator._phase_bc_day_first")
    @patch(
        "gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator"
        "._phase_a_ingest_and_analyze"
    )
    def test_pipeline_summary_youtube_uploaded_reports_skipped_mode(
        self, mock_a, mock_bc, mock_d, mock_rg, tmp_path, caplog
    ):
        """When skip_upload=True, the field reports ``skipped`` (not true/false)."""
        import logging as _logging

        def _bc_side_effect(state):
            state["regular_lf_paths"] = ["/a"]
            state["shorts_per_lf"] = {1: []}
            return {}

        mock_a.return_value = {}
        mock_bc.side_effect = _bc_side_effect
        mock_d.return_value = {}

        source = tmp_path / "source"
        source.mkdir()

        cfg = Config()
        cfg.drive.skip_validation = True
        orch = DrivePipelineOrchestrator(config=cfg)

        with caplog.at_level(_logging.INFO, logger="gomoveshift.pipeline.drive_orchestrator"):
            orch.run(
                source_dir=source,
                output_base=tmp_path / "output",
                region="ireland",
                skip_upload=True,
            )

        summary = next(rec.message for rec in caplog.records if "PIPELINE_SUMMARY" in rec.message)
        assert "youtube_uploaded=skipped" in summary, summary


class TestPipelineSummaryAllExitPaths:
    """The sentinel must fire
    on *every* explicit exit, not just the happy path; otherwise operators
    grepping outcomes have a silent gap. Each early-return path and the
    DoD-failure path now emits the sentinel.
    """

    def _sentinels(self, caplog):
        return [rec.message for rec in caplog.records if "PIPELINE_SUMMARY" in rec.message]

    def test_sentinel_fires_when_source_dir_missing(self, tmp_path, caplog):
        """source_dir not existing → early return + sentinel."""
        import logging as _logging

        cfg = Config()
        orch = DrivePipelineOrchestrator(config=cfg)
        with caplog.at_level(_logging.INFO, logger="gomoveshift.pipeline.drive_orchestrator"):
            result = orch.run(
                source_dir=tmp_path / "does-not-exist",
                output_base=tmp_path / "out",
                region="ireland",
                skip_upload=True,
            )
        assert result.success is False
        lines = self._sentinels(caplog)
        assert len(lines) == 1, lines
        assert "lfs=0" in lines[0] and "shorts=0" in lines[0]

    @patch("gomoveshift.pipeline.drive_orchestrator.ReviewGate")
    @patch("gomoveshift.pipeline.orchestrator_preflight.print_plan")
    def test_sentinel_fires_on_dry_run_exit(self, mock_print_plan, mock_rg, tmp_path, caplog):
        """--dry-run exits after print_plan — sentinel must still fire."""
        import logging as _logging

        source = tmp_path / "source"
        source.mkdir()
        cfg = Config()
        cfg.drive.skip_validation = True
        orch = DrivePipelineOrchestrator(config=cfg)
        with caplog.at_level(_logging.INFO, logger="gomoveshift.pipeline.drive_orchestrator"):
            result = orch.run(
                source_dir=source,
                output_base=tmp_path / "out",
                region="ireland",
                skip_upload=True,
                dry_run=True,
                force=True,
            )
        assert result.success is True
        assert result.stages_completed == []
        lines = self._sentinels(caplog)
        assert len(lines) == 1, lines

    @patch("gomoveshift.pipeline.drive_orchestrator.ReviewGate")
    @patch("gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator._phase_b_preview")
    @patch(
        "gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator"
        "._phase_a_ingest_and_analyze"
    )
    def test_sentinel_fires_on_preview_success(
        self, mock_a, mock_preview, mock_rg, tmp_path, caplog
    ):
        """Preview-mode happy exit — the exact path that silently skipped
        the sentinel."""
        import logging as _logging

        mock_a.return_value = {}
        mock_preview.return_value = None
        source = tmp_path / "source"
        source.mkdir()
        cfg = Config()
        cfg.drive.skip_validation = True
        cfg.drive.preview = True
        orch = DrivePipelineOrchestrator(config=cfg)
        with caplog.at_level(_logging.INFO, logger="gomoveshift.pipeline.drive_orchestrator"):
            result = orch.run(
                source_dir=source,
                output_base=tmp_path / "out",
                region="ireland",
                skip_upload=True,
                force=True,
            )
        assert result.success is True
        lines = self._sentinels(caplog)
        assert len(lines) == 1, lines

    @patch("gomoveshift.pipeline.drive_orchestrator.ReviewGate")
    @patch("gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator._phase_b_preview")
    @patch(
        "gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator"
        "._phase_a_ingest_and_analyze"
    )
    def test_sentinel_fires_on_preview_failure(
        self, mock_a, mock_preview, mock_rg, tmp_path, caplog
    ):
        """Preview-mode encoder failure → sentinel still fires."""
        import logging as _logging

        mock_a.return_value = {}
        mock_preview.side_effect = RuntimeError("ffmpeg boom")
        source = tmp_path / "source"
        source.mkdir()
        cfg = Config()
        cfg.drive.skip_validation = True
        cfg.drive.preview = True
        orch = DrivePipelineOrchestrator(config=cfg)
        with caplog.at_level(_logging.INFO, logger="gomoveshift.pipeline.drive_orchestrator"):
            result = orch.run(
                source_dir=source,
                output_base=tmp_path / "out",
                region="ireland",
                skip_upload=True,
                force=True,
            )
        assert result.success is False
        lines = self._sentinels(caplog)
        assert len(lines) == 1, lines
