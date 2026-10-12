"""Tests for regular LF upload to Drive + Metricool CSV rows.

TDD: failing tests first. Regular long-form videos must be:
1. Uploaded to Google Drive
2. Have drive_link stored in metadata
3. Get a Metricool CSV row with YouTube Video Type = "VIDEO"
"""

import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.models import DrivePlan, LFSegment, RegularLFPlan
from gomoveshift.pipeline.csv_generator import CSVGenerator, GOLDEN_COLUMNS
from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator
from gomoveshift.pipeline.scheduler import (
    DriveReleaseSchedule,
    ScheduledSlot,
)
from gomoveshift.pipeline.social_generator import SocialPost
from gomoveshift.pipeline.uploader import UploadManifest

from _state_helpers import make_state


def _make_lf(index, duration=900.0):
    """Create a RegularLFPlan for testing."""
    seg = LFSegment(
        source_file_index=0,
        source_path=Path("/tmp/fake.mp4"),
        start_time=0.0,
        end_time=duration,
        original_start_time=0.0,
        original_end_time=duration,
        composite_score=0.8,
        gps_center_lat=51.9,
        gps_center_lon=-8.5,
        terrain_type="countryside",
        contiguous_with_previous=False,
    )
    return RegularLFPlan(
        index=index,
        segments=[seg],
        montage_peaks=[seg],
        shorts_target=3,
        region="ireland",
        total_duration=duration,
    )


def _make_drive_plan(num_lfs=2):
    """Create a DrivePlan with regular LFs."""
    return DrivePlan(
        source_files=[],
        processed_files=[],
        regular_lfs=[_make_lf(i + 1) for i in range(num_lfs)],
        score_threshold=0.5,
        total_source_duration=3600.0,
        total_highlight_duration=1800.0,
    )


def _make_lf_slot(index, date="2026-04-01", time="11:00:00"):
    return ScheduledSlot(
        short_index=0,
        date=date,
        time=time,
        datetime_est=f"{date}T{time}-05:00",
        day_of_week="Wed",
    )


def _make_shorts_slot(index, date="2026-04-01", time="12:00:00"):
    return ScheduledSlot(
        short_index=index,
        date=date,
        time=time,
        datetime_est=f"{date}T{time}-05:00",
        day_of_week="Wed",
    )


class TestLFUploadToDrive:
    """Regression guard: LFs must NOT be uploaded to Drive anymore.

    (They go to YouTube now — see :class:`tests.test_phase_d.TestPhaseDYouTubeUpload`.)
    """

    @pytest.fixture(autouse=True)
    def _stub_final_validation_gate(self):
        # These tests use sub-1-MiB placeholder LF files. The
        # pre-upload file_integrity gate now HARD-fails on a missing/truncated
        # FINAL (was a silent skip); they exercise the upload wiring, not the
        # validator, so stub it to pass.
        with patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineValidator") as val_cls:
            val = val_cls.return_value
            val.validate.return_value = MagicMock(passed=True, summary=lambda: "ok")
            yield

    @patch("gomoveshift.pipeline.drive_phases.phase_d.CSVGenerator")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineUploader")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.SocialGenerator")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineScheduler")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.LongFormMetadataGenerator")
    def test_lf_videos_uploaded_to_drive(
        self,
        mock_meta_cls,
        mock_sched_cls,
        mock_social_cls,
        mock_uploader_cls,
        mock_csv_cls,
        tmp_path,
    ):
        """Phase D uploads each regular LF video file to Google Drive."""
        source = tmp_path / "source"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir(parents=True, exist_ok=True)
        (output / "_stages").mkdir(parents=True, exist_ok=True)

        drive_plan = _make_drive_plan(2)

        lf1_path = tmp_path / "lf1.mp4"
        lf2_path = tmp_path / "lf2.mp4"
        # ≥1 MiB so the pre-upload file_integrity gate passes.
        lf1_path.write_bytes(b"\0" * (1024 * 1024 + 1))
        lf2_path.write_bytes(b"\0" * (1024 * 1024 + 1))

        # Distinct title per LF — phase_d's quality gate Rule "LF titles all
        # distinct" rejects duplicates, so vary on call count.
        mock_meta = MagicMock()
        _meta_call_count = {"n": 0}

        def _meta_generate(*args, **kwargs):
            _meta_call_count["n"] += 1
            return {
                "title": f"Test Title {_meta_call_count['n']}",
                "description": "Test desc",
                "tags": ["tag1"],
                "category": "Travel & Events",
            }

        mock_meta.generate.side_effect = _meta_generate
        mock_meta_cls.return_value = mock_meta

        mock_sched = MagicMock()
        mock_sched.generate_drive_schedule.return_value = DriveReleaseSchedule(
            region="ireland",
            lf_slots=[
                _make_lf_slot(1, "2026-04-01"),
                _make_lf_slot(2, "2026-04-08"),
            ],
            shorts_slots=[
                _make_shorts_slot(1, "2026-04-01", "12:00:00"),
                _make_shorts_slot(2, "2026-04-02", "08:00:00"),
            ],
        )
        mock_sched_cls.return_value = mock_sched
        mock_sched_cls.from_config.return_value = mock_sched

        # The mock honors num_shorts so the consolidated CSV row count
        # invariant (Σ rows == Σ len(shorts_per_lf[i])) holds.
        # With ``shorts_per_lf={}`` we expect 0 posts per LF and 0 CSV rows.
        # Returning a fixed 1-post list regardless of num_shorts (the legacy
        # fixture shape) fed phantom rows into the consolidated CSV.
        mock_social = MagicMock()

        def _generate_all(num_shorts, *args, index_offset=0, **kwargs):
            return [
                SocialPost(
                    short_index=index_offset + i + 1,
                    caption=f"Test caption {i + 1}",
                    youtube_title=f"Test YT {i + 1}",
                    youtube_tags="tag1,tag2",
                    youtube_category="Travel & Events",
                    tiktok_title=f"Test TT {i + 1}",
                    pinterest_board="Travel",
                    first_comment="#test",
                )
                for i in range(num_shorts)
            ]

        mock_social.generate_all.side_effect = _generate_all
        mock_social_cls.return_value = mock_social

        mock_uploader = MagicMock()
        mock_uploader.upload_shorts.return_value = UploadManifest(
            project_name="test",
            files={"short_01.mp4": "https://drive.google.com/file/d/abc/view?usp=sharing"},
        )
        mock_uploader.upload_single_file.return_value = (
            "https://drive.google.com/file/d/lf_file_id/view?usp=sharing"
        )
        mock_uploader_cls.return_value = mock_uploader

        mock_csv = MagicMock()
        mock_csv.generate.return_value = output / "test_metricool.csv"
        mock_csv.generate_lf_row.return_value = output / "test_lf_metricool.csv"
        mock_csv_cls.return_value = mock_csv

        state = make_state(
            source_dir=source,
            output_base=output,
            drive_plan=drive_plan,
            regular_lf_paths=[lf1_path, lf2_path],
            # Zero-shorts day: production phase_bc emits per-LF empty lists
            # (truthy), NOT an empty dict — {} is the legacy-layout fallback shape
            # that now hard-fails at Phase D.
            shorts_per_lf={1: [], 2: []},
            region="ireland",
            publish_day="wed",
            location="Cork, Ireland",
            skip_upload=False,
            skip_csv=False,
            # BC→D boundary contract — populated by BC in production.
            per_lf_recording_date={1: "2026-02-20", 2: "2026-02-20"},
            per_lf_locations={1: "Cork, Ireland", 2: "Cork, Ireland"},
            per_lf_trim_ratio={1: 0.7, 2: 0.7},
        )

        orch = DrivePipelineOrchestrator()
        orch._phase_d_metadata_and_publish(state)

        # LFs no longer go to Drive; upload_single_file MUST NOT be called.
        mock_uploader.upload_single_file.assert_not_called()

    @patch("gomoveshift.pipeline.drive_phases.phase_d.CSVGenerator")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineUploader")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.SocialGenerator")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineScheduler")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.LongFormMetadataGenerator")
    def test_lf_metadata_contains_drive_link(
        self,
        mock_meta_cls,
        mock_sched_cls,
        mock_social_cls,
        mock_uploader_cls,
        mock_csv_cls,
        tmp_path,
    ):
        """After upload, metadata_per_lf includes a drive_link field."""
        source = tmp_path / "source"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir(parents=True, exist_ok=True)
        (output / "_stages").mkdir(parents=True, exist_ok=True)

        drive_plan = _make_drive_plan(1)

        lf_path = tmp_path / "lf1.mp4"
        lf_path.write_bytes(b"\0" * (1024 * 1024 + 1))  # ≥1 MiB integrity floor

        mock_meta = MagicMock()
        mock_meta.generate.return_value = {
            "title": "Test Title",
            "description": "Test desc",
            "tags": ["tag1"],
            "category": "Travel & Events",
        }
        mock_meta_cls.return_value = mock_meta

        mock_sched = MagicMock()
        mock_sched.generate_drive_schedule.return_value = DriveReleaseSchedule(
            region="ireland",
            lf_slots=[_make_lf_slot(1)],
            shorts_slots=[_make_shorts_slot(1)],
        )
        mock_sched_cls.return_value = mock_sched
        mock_sched_cls.from_config.return_value = mock_sched

        mock_social = MagicMock()
        mock_social.generate_all.return_value = []
        mock_social_cls.return_value = mock_social

        mock_uploader = MagicMock()
        mock_uploader.upload_shorts.return_value = UploadManifest(project_name="test")
        mock_uploader.upload_single_file.return_value = (
            "https://drive.google.com/file/d/xyz123/view?usp=sharing"
        )
        mock_uploader_cls.return_value = mock_uploader

        mock_csv = MagicMock()
        mock_csv.generate.return_value = output / "test_metricool.csv"
        mock_csv.generate_lf_row.return_value = output / "test_lf_metricool.csv"
        mock_csv_cls.return_value = mock_csv

        state = make_state(
            source_dir=source,
            output_base=output,
            drive_plan=drive_plan,
            regular_lf_paths=[lf_path],
            shorts_per_lf={1: []},  # zero-shorts LF = empty list (truthy), not {} (hard-fail)
            region="ireland",
            publish_day="wed",
            location="Cork, Ireland",
            skip_upload=False,
            skip_csv=False,
            # BC→D boundary contract — populated by BC in production.
            per_lf_recording_date={1: "2026-02-20"},
            per_lf_locations={1: "Cork, Ireland"},
            per_lf_trim_ratio={1: 0.7},
        )

        orch = DrivePipelineOrchestrator()
        orch._phase_d_metadata_and_publish(state)

        # drive_link no longer written into metadata_per_lf.
        # YouTube upload state lives in state["youtube_uploads"] instead.
        metadata = state["metadata_per_lf"][1]
        assert "drive_link" not in metadata
        assert "youtube_uploads" in state

    @patch("gomoveshift.pipeline.drive_phases.phase_d.CSVGenerator")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineUploader")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.SocialGenerator")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineScheduler")
    @patch("gomoveshift.pipeline.drive_phases.phase_d.LongFormMetadataGenerator")
    def test_lf_upload_skipped_when_skip_upload(
        self,
        mock_meta_cls,
        mock_sched_cls,
        mock_social_cls,
        mock_uploader_cls,
        mock_csv_cls,
        tmp_path,
    ):
        """LF upload is skipped when skip_upload=True."""
        source = tmp_path / "source"
        source.mkdir()
        output = tmp_path / "output"
        output.mkdir(parents=True, exist_ok=True)
        (output / "_stages").mkdir(parents=True, exist_ok=True)

        drive_plan = _make_drive_plan(1)

        lf_path = tmp_path / "lf1.mp4"
        lf_path.write_bytes(b"\0" * (1024 * 1024 + 1))  # ≥1 MiB integrity floor

        mock_meta = MagicMock()
        mock_meta.generate.return_value = {
            "title": "Test Title",
            "description": "Test desc",
            "tags": ["tag1"],
            "category": "Travel & Events",
        }
        mock_meta_cls.return_value = mock_meta

        mock_sched = MagicMock()
        mock_sched.generate_drive_schedule.return_value = DriveReleaseSchedule(
            region="ireland",
            lf_slots=[_make_lf_slot(1)],
            shorts_slots=[],
        )
        mock_sched_cls.return_value = mock_sched
        mock_sched_cls.from_config.return_value = mock_sched

        mock_social = MagicMock()
        mock_social.generate_all.return_value = []
        mock_social_cls.return_value = mock_social

        mock_uploader = MagicMock()
        mock_uploader_cls.return_value = mock_uploader

        mock_csv = MagicMock()
        mock_csv_cls.return_value = mock_csv

        state = make_state(
            source_dir=source,
            output_base=output,
            drive_plan=drive_plan,
            regular_lf_paths=[lf_path],
            shorts_per_lf={1: []},  # zero-shorts LF = empty list (truthy), not {} (hard-fail)
            region="ireland",
            publish_day="wed",
            location="Cork, Ireland",
            skip_upload=True,
            skip_csv=True,
            # BC→D boundary contract — populated by BC in production.
            per_lf_recording_date={1: "2026-02-20"},
            per_lf_locations={1: "Cork, Ireland"},
            per_lf_trim_ratio={1: 0.7},
        )

        orch = DrivePipelineOrchestrator()
        orch._phase_d_metadata_and_publish(state)

        mock_uploader.upload_single_file.assert_not_called()


class TestGenerateLFRow:
    """generate_lf_row() produces valid Metricool CSV for long-form videos."""

    def test_generate_lf_row_creates_file(self, tmp_path):
        """generate_lf_row writes a CSV file to output_path."""
        csv_gen = CSVGenerator()
        output_path = tmp_path / "lf_metricool.csv"

        result = csv_gen.generate_lf_row(
            title="POV Drive Through Cork | Scenic Road Trip",
            tags="POV driving,scenic drive,road trip",
            category="19",
            drive_link="https://drive.google.com/file/d/abc123/view?usp=sharing",
            schedule_date="2026-04-01",
            schedule_time="11:00:00",
            output_path=output_path,
        )

        assert result == output_path
        assert output_path.exists()

    def test_generate_lf_row_valid_csv_content(self, tmp_path):
        """Generated CSV has correct header and one data row."""
        csv_gen = CSVGenerator()
        output_path = tmp_path / "lf_metricool.csv"

        csv_gen.generate_lf_row(
            title="Test Drive Title",
            tags="driving,ireland",
            category="19",
            drive_link="https://drive.google.com/file/d/xyz/view?usp=sharing",
            schedule_date="2026-05-01",
            schedule_time="14:00:00",
            output_path=output_path,
        )

        with open(output_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        row = rows[0]
        assert row["YouTube Video Title"] == "Test Drive Title"
        assert row["YouTube Video Tags"] == "driving,ireland"
        assert row["Date"] == "2026-05-01"
        assert row["Time"] == "14:00:00"
        assert row["Picture Url 1"] == "https://drive.google.com/file/d/xyz/view?usp=sharing"

    def test_generate_lf_row_category_19(self, tmp_path):
        """Category field is set to '19' (Travel & Events ID)."""
        csv_gen = CSVGenerator()
        output_path = tmp_path / "lf_cat.csv"

        csv_gen.generate_lf_row(
            title="Category Test",
            tags="test",
            category="19",
            drive_link="https://drive.google.com/file/d/cat/view?usp=sharing",
            schedule_date="2026-04-01",
            schedule_time="11:00:00",
            output_path=output_path,
        )

        with open(output_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)

        assert row["YouTube Video Category"] == "19"
        assert row["YouTube Video Type"] == "VIDEO"

    def test_generate_lf_row_youtube_only_platforms(self, tmp_path):
        """LF row enables YouTube only — no Instagram, TikTok, Pinterest."""
        csv_gen = CSVGenerator()
        output_path = tmp_path / "lf_platforms.csv"

        csv_gen.generate_lf_row(
            title="Platform Test",
            tags="test",
            category="19",
            drive_link="https://drive.google.com/file/d/plat/view?usp=sharing",
            schedule_date="2026-04-01",
            schedule_time="11:00:00",
            output_path=output_path,
        )

        with open(output_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)

        assert row["YouTube"] == "TRUE"
        assert row["Instagram"] == "FALSE"
        assert row["TikTok"] == "FALSE"
        assert row["Pinterest"] == "FALSE"
        assert row["Facebook"] == "FALSE"

    def test_generate_lf_row_has_all_golden_columns(self, tmp_path):
        """LF CSV has all 31 golden columns in header."""
        csv_gen = CSVGenerator()
        output_path = tmp_path / "lf_golden.csv"

        csv_gen.generate_lf_row(
            title="Golden Test",
            tags="test",
            category="19",
            drive_link="https://drive.google.com/file/d/gold/view?usp=sharing",
            schedule_date="2026-04-01",
            schedule_time="11:00:00",
            output_path=output_path,
        )

        with open(output_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)

        for col in GOLDEN_COLUMNS:
            assert col in row, f"Missing column: {col}"

    def test_generate_lf_row_default_description(self, tmp_path):
        """When description is omitted, Text field uses title."""
        csv_gen = CSVGenerator()
        output_path = tmp_path / "lf_desc.csv"

        csv_gen.generate_lf_row(
            title="Fallback Title",
            tags="test",
            category="19",
            drive_link="https://drive.google.com/file/d/desc/view?usp=sharing",
            schedule_date="2026-04-01",
            schedule_time="11:00:00",
            output_path=output_path,
        )

        with open(output_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)

        assert row["Text"] == "Fallback Title"

    def test_generate_lf_row_with_description(self, tmp_path):
        """When description is provided, Text field uses it."""
        csv_gen = CSVGenerator()
        output_path = tmp_path / "lf_desc2.csv"

        csv_gen.generate_lf_row(
            title="Title Here",
            tags="test",
            category="19",
            drive_link="https://drive.google.com/file/d/desc2/view?usp=sharing",
            schedule_date="2026-04-01",
            schedule_time="11:00:00",
            output_path=output_path,
            description="A scenic drive through the countryside.",
        )

        with open(output_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)

        assert row["Text"] == "A scenic drive through the countryside."

    def test_generate_lf_row_creates_parent_dirs(self, tmp_path):
        """generate_lf_row creates parent directories if needed."""
        csv_gen = CSVGenerator()
        output_path = tmp_path / "nested" / "deep" / "lf.csv"

        result = csv_gen.generate_lf_row(
            title="Nested Dir Test",
            tags="test",
            category="19",
            drive_link="https://drive.google.com/file/d/nest/view?usp=sharing",
            schedule_date="2026-04-01",
            schedule_time="11:00:00",
            output_path=output_path,
        )

        assert result.exists()
