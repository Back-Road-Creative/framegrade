"""Tests for data boundary validation: a CSV with a missing slot raises, and an LF count mismatch raises."""

from unittest.mock import MagicMock

import pytest

from gomoveshift.pipeline.csv_generator import CSVGenerator
from gomoveshift.pipeline.scheduler import ReleaseSchedule, ScheduledSlot
from gomoveshift.pipeline.social_generator import SocialPost
from gomoveshift.pipeline.uploader import UploadManifest


def _make_post(short_index: int) -> SocialPost:
    return SocialPost(
        short_index=short_index,
        youtube_title=f"Drive Short #{short_index}",
        tiktok_title=f"Short #{short_index}",
        caption=f"Caption {short_index}",
        youtube_tags="#drive,#roadtrip",
        youtube_category="19",
        first_comment="#drive",
        pinterest_board="POV Driving",
    )


def _make_slot(short_index: int) -> ScheduledSlot:
    return ScheduledSlot(
        short_index=short_index,
        date="2026-05-01",
        time="10:00:00",
        datetime_est="2026-05-01T10:00:00",
        day_of_week="Friday",
    )


def _make_schedule(slots) -> ReleaseSchedule:
    sched = MagicMock(spec=ReleaseSchedule)
    sched.slots = slots
    return sched


def _make_manifest() -> UploadManifest:
    m = MagicMock(spec=UploadManifest)
    # Drive links for both shorts so the CSV actually writes: the media/schedule
    # asserts now DEFER a link-less CSV instead of shipping a dead one, so a
    # slot-focused test must supply media to isolate the slot check from that.
    m.files = {
        "short_01.mp4": "https://drive.google.com/short1",
        "short_02.mp4": "https://drive.google.com/short2",
    }
    return m


class TestCSVMissingSlotRaises:
    """generate() must raise ValueError when a social post has no schedule slot."""

    def test_raises_when_short_has_no_slot(self, tmp_path):
        """If short #2 has no slot but short #1 does, generate() must raise."""
        gen = CSVGenerator()
        posts = [_make_post(1), _make_post(2)]
        schedule = _make_schedule([_make_slot(1)])  # only slot for short #1, not #2

        with pytest.raises(ValueError, match="[Nn]o schedule slot.*2|short.*2"):
            gen.generate(
                project_name="test",
                social_posts=posts,
                schedule=schedule,
                manifest=_make_manifest(),
                output_dir=str(tmp_path),
                validate=False,
            )

    def test_no_raise_when_all_shorts_have_slots(self, tmp_path):
        """All shorts with matching slots must succeed without raising."""
        gen = CSVGenerator()
        posts = [_make_post(1), _make_post(2)]
        schedule = _make_schedule([_make_slot(1), _make_slot(2)])

        # ``_make_manifest`` carries a Drive link per short — this test pins the
        # SLOT gate, and a link-less manifest is now deferred, not written.
        result = gen.generate(
            project_name="test",
            social_posts=posts,
            schedule=schedule,
            manifest=_make_manifest(),
            output_dir=str(tmp_path),
            validate=False,
        )
        assert result.exists()


class TestLFCountMismatchRaises:
    """Phase D zip loops must raise RuntimeError when LF count doesn't match plan."""

    def _make_minimal_state(self, tmp_path, num_lf_paths: int):
        """State with num_lf_paths assembled paths."""
        from _state_helpers import make_state

        return make_state(
            source_dir=tmp_path,
            output_base=tmp_path,
            location="Test Location, France",
            region="France",
            publish_day="Saturday",
            regular_lf_paths=[tmp_path / f"lf_{i:02d}.mp4" for i in range(num_lf_paths)],
            phases_completed=["phase_a", "phase_bc_day_first"],
        )

    def test_raises_when_fewer_paths_than_plan(self, tmp_path):
        """If plan has 3 LFs but Phase C only produced 2 paths, Phase D must raise."""
        from gomoveshift.models import DrivePlan, RegularLFPlan, LFSegment
        from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator

        # Build a plan with 3 LFs
        def _make_lf(i):
            seg = LFSegment(
                source_file_index=0,
                source_path=tmp_path / "ep.mp4",
                start_time=0.0,
                end_time=300.0,
                original_start_time=0.0,
                original_end_time=300.0,
                composite_score=0.5,
                gps_center_lat=48.8,
                gps_center_lon=2.3,
                terrain_type="urban",
                contiguous_with_previous=False,
            )
            return RegularLFPlan(
                index=i,
                segments=[seg],
                montage_peaks=[],
                shorts_target=5,
                region="France",
                total_duration=300.0,
            )

        plan = DrivePlan(
            source_files=[tmp_path / "ep.mp4"],
            processed_files=[],
            regular_lfs=[_make_lf(1), _make_lf(2), _make_lf(3)],
            score_threshold=0.5,
            total_source_duration=900.0,
            total_highlight_duration=900.0,
        )

        # Only 2 paths assembled (Phase C partial failure)
        state = self._make_minimal_state(tmp_path, num_lf_paths=2)

        orch = DrivePipelineOrchestrator()
        with pytest.raises(RuntimeError, match="[Ll][Ff].*mismatch|mismatch.*[Ll][Ff]|3.*2|2.*3"):
            orch._phase_d_metadata_and_publish(state, drive_plan=plan)

    def test_no_raise_when_counts_match(self, tmp_path):
        """Matching LF count must not raise (Phase D proceeds normally)."""
        from gomoveshift.models import DrivePlan, RegularLFPlan, LFSegment
        from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator

        def _make_lf(i):
            seg = LFSegment(
                source_file_index=0,
                source_path=tmp_path / "ep.mp4",
                start_time=0.0,
                end_time=300.0,
                original_start_time=0.0,
                original_end_time=300.0,
                composite_score=0.5,
                gps_center_lat=48.8,
                gps_center_lon=2.3,
                terrain_type="urban",
                contiguous_with_previous=False,
            )
            return RegularLFPlan(
                index=i,
                segments=[seg],
                montage_peaks=[],
                shorts_target=5,
                region="France",
                total_duration=300.0,
            )

        plan = DrivePlan(
            source_files=[tmp_path / "ep.mp4"],
            processed_files=[],
            regular_lfs=[_make_lf(1), _make_lf(2)],
            score_threshold=0.5,
            total_source_duration=600.0,
            total_highlight_duration=600.0,
        )

        # 2 paths to match 2 LFs
        state = self._make_minimal_state(tmp_path, num_lf_paths=2)

        orch = DrivePipelineOrchestrator()
        # Should not raise on count check — may raise later on missing metadata etc.
        try:
            orch._phase_d_metadata_and_publish(state, drive_plan=plan)
        except RuntimeError as e:
            assert "mismatch" not in str(e).lower(), f"Unexpected mismatch error: {e}"
        except Exception:
            pass  # Other errors are fine — we only care the count check doesn't raise
