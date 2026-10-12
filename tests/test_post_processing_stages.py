"""TDD tests for post-processing pipeline stages 7-11.

Tests the orchestrator integration of:
  Stage 7:  Thumbnail generation (frames + styled overlays)
  Stage 8:  Shorts analysis (motion segment identification)
  Stage 9:  Schedule generation (posting windows)
  Stage 10: Social post generation (captions per short)
  Stage 11: Metricool CSV generation

All tests mock heavy I/O (ffmpeg, video reading) to run fast.
"""

import csv
import json
from unittest.mock import MagicMock, patch

import pytest

# These drive the full orchestrator on a dummy video; blur is on by default,
# so opt out of the real face/plate detector (see conftest ``disable_blur``).
pytestmark = pytest.mark.usefixtures("disable_blur")

# All tests mock compute_file_hash so the resume check succeeds
# without needing a real video file.
_FAKE_HASH = "abc123"


@pytest.fixture(autouse=True)
def mock_file_hash():
    """Mock file hashing so resume works with fake video files."""
    with patch(
        "gomoveshift.pipeline.orchestrator.compute_file_hash",
        return_value=_FAKE_HASH,
    ):
        yield


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline_output(tmp_path):
    """Create a minimal pipeline output directory with prior stages done.

    The orchestrator derives output_dir from ``output_base / project_name``,
    so we structure the fixture as ``tmp_path / testproject / ...`` and pass
    ``output_base=tmp_path`` to ``run()``.
    """
    project_dir = tmp_path / "testproject"
    project_dir.mkdir()
    stages_dir = project_dir / "_stages"
    stages_dir.mkdir()

    # Fake video file — named so stem.lower() == "testproject"
    final_video = project_dir / "testproject.mp4"
    final_video.write_bytes(b"\x00" * 1024)

    # Fake metadata from stage 6
    metadata = {
        "title": "POV Drive: Test Location 🚗 France",
        "description": "A scenic drive...",
        "tags": ["pov drive", "ireland", "scenic"],
        "category": "Travel & Events",
        "publish_day": "Saturday",
        "project_name": "testproject",
    }
    meta_path = project_dir / "testproject_youtube.json"
    meta_path.write_text(json.dumps(metadata))

    # Pipeline state with stages 1-6 complete
    state = {
        "project_name": "testproject",
        "source_video": str(final_video),
        "source_hash": "abc123",
        "stages_completed": [
            "detect_properties",
            "stop_removal",
            "ost_overlays",
            "audio_normalization",
            "metadata_generation",
        ],
        "current_video": str(final_video),
        "video_properties": {
            "width": 3840,
            "height": 2160,
            "fps": 59.94,
            "duration": 3900.0,
            "bitrate_kbps": 83000,
        },
        "timings": {},
        "params": {
            "location": "Test Location, France",
            "publish_day": "sat",
            "skip_color": True,
            "skip_stops": True,
        },
    }
    state_file = stages_dir / "pipeline_state.json"
    state_file.write_text(json.dumps(state))

    return tmp_path  # Return the *base* directory, not project_dir


# ===========================================================================
# Stage 7: Thumbnail Generation
# ===========================================================================


class TestThumbnailStage:
    """Stage 7: Thumbnail generation produces styled frames."""

    def test_thumbnail_stage_creates_styled_directory(self, pipeline_output):
        """Thumbnail stage should create thumbnails/styled/ directory."""
        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator, RunOptions

        orch = PipelineOrchestrator()
        thumb_dir = pipeline_output / "testproject" / "thumbnails" / "styled"

        # The directory structure should be created by the stage
        # We test via the helper method directly
        assert not thumb_dir.exists()

        with patch("gomoveshift.pipeline.orchestrator.generate_thumbnails_from_video") as mock_gen:
            mock_gen.return_value = {
                "frame1": {
                    "minimal": str(thumb_dir / "f1_minimal.jpg"),
                    "bold": str(thumb_dir / "f1_bold.jpg"),
                    "immersive": str(thumb_dir / "f1_immersive.jpg"),
                },
            }
            # Create the files the mock "generated"
            thumb_dir.mkdir(parents=True)
            for name in ["f1_minimal.jpg", "f1_bold.jpg", "f1_immersive.jpg"]:
                # >= MIN_THUMBNAIL_BYTES: the stage now hard-fails on a
                # blank/truncated thumbnail instead of warning past it.
                (thumb_dir / name).write_bytes(b"\xff\xd8" + b"\x00" * 20480)

            result = orch.run(
                RunOptions.from_legacy(
                    video_path=pipeline_output / "testproject" / "testproject.mp4",
                    location="Test Location, France",
                    publish_day="sat",
                    output_base=pipeline_output,
                    skip_color=True,
                    skip_stops=True,
                    skip_ost=True,
                    skip_audio=True,
                    skip_metadata=True,
                )
            )

        assert "thumbnail_generation" in result.stages_completed

    def test_thumbnail_stage_generates_3_styles_per_frame(self, pipeline_output):
        """Each frame should get minimal, bold, and immersive styled versions."""
        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator, RunOptions

        thumb_dir = pipeline_output / "testproject" / "thumbnails" / "styled"

        with patch("gomoveshift.pipeline.orchestrator.generate_thumbnails_from_video") as mock_gen:
            mock_gen.return_value = {
                "frame1": {"minimal": "a.jpg", "bold": "b.jpg", "immersive": "c.jpg"},
                "frame2": {"minimal": "d.jpg", "bold": "e.jpg", "immersive": "f.jpg"},
                "frame3": {"minimal": "g.jpg", "bold": "h.jpg", "immersive": "i.jpg"},
            }
            thumb_dir.mkdir(parents=True)

            orch = PipelineOrchestrator()
            _result = orch.run(
                RunOptions.from_legacy(
                    video_path=pipeline_output / "testproject" / "testproject.mp4",
                    location="Test Location, France",
                    publish_day="sat",
                    output_base=pipeline_output,
                    skip_color=True,
                    skip_stops=True,
                    skip_ost=True,
                    skip_audio=True,
                    skip_metadata=True,
                )
            )

            # Verify the generator was called with correct location info
            call_kwargs = mock_gen.call_args
            assert "Test Location, France" in str(call_kwargs)

    def test_thumbnail_stage_skippable(self, pipeline_output):
        """--skip-thumbnails should skip stage 7."""
        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator, RunOptions

        orch = PipelineOrchestrator()
        result = orch.run(
            RunOptions.from_legacy(
                video_path=pipeline_output / "testproject" / "testproject.mp4",
                location="Test Location, France",
                publish_day="sat",
                output_base=pipeline_output,
                skip_color=True,
                skip_stops=True,
                skip_ost=True,
                skip_audio=True,
                skip_metadata=True,
                skip_thumbnails=True,
            )
        )

        assert "thumbnail_generation" not in result.stages_completed

    def test_thumbnail_stage_resumes(self, pipeline_output):
        """If thumbnail_generation already in stages_completed, skip it."""
        stages_dir = pipeline_output / "testproject" / "_stages"
        state_file = stages_dir / "pipeline_state.json"
        state = json.loads(state_file.read_text())
        state["stages_completed"].append("thumbnail_generation")
        state_file.write_text(json.dumps(state))

        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator, RunOptions

        orch = PipelineOrchestrator()
        with patch("gomoveshift.pipeline.orchestrator.generate_thumbnails_from_video") as mock_gen:
            result = orch.run(
                RunOptions.from_legacy(
                    video_path=pipeline_output / "testproject" / "testproject.mp4",
                    location="Test Location, France",
                    publish_day="sat",
                    output_base=pipeline_output,
                    skip_color=True,
                    skip_stops=True,
                    skip_ost=True,
                    skip_audio=True,
                    skip_metadata=True,
                )
            )
            mock_gen.assert_not_called()

        assert "thumbnail_generation" in result.stages_completed


# ===========================================================================
# Stage 8: Shorts Analysis
# ===========================================================================


class TestShortsAnalysisStage:
    """Stage 8: Shorts analysis identifies high-motion segments."""

    def test_shorts_stage_creates_output_dir(self, pipeline_output):
        """Shorts analysis should create shorts/ directory."""
        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator, RunOptions

        shorts_dir = pipeline_output / "testproject" / "shorts"
        assert not shorts_dir.exists()

        with patch("gomoveshift.pipeline.orchestrator.PipelineShortsExtractor") as MockExtractor:  # noqa: N806
            mock_instance = MockExtractor.return_value
            mock_result = MagicMock()
            mock_result.num_rendered = 9
            mock_result.shorts = [MagicMock() for _ in range(9)]
            for i, s in enumerate(mock_result.shorts):
                s.to_dict.return_value = {
                    "index": i + 1,
                    "start_time": i * 200.0,
                    "end_time": i * 200.0 + 30.0,
                    "duration": 30.0,
                    "avg_motion_energy": 0.05,
                }
            mock_result.errors = []
            mock_instance.extract_and_render.return_value = mock_result

            orch = PipelineOrchestrator()
            result = orch.run(
                RunOptions.from_legacy(
                    video_path=pipeline_output / "testproject" / "testproject.mp4",
                    location="Test Location, France",
                    publish_day="sat",
                    output_base=pipeline_output,
                    skip_color=True,
                    skip_stops=True,
                    skip_ost=True,
                    skip_audio=True,
                    skip_metadata=True,
                    skip_thumbnails=True,
                )
            )

        assert "shorts_analysis" in result.stages_completed

    def test_shorts_stage_saves_analysis_json(self, pipeline_output):
        """Shorts analysis should write shorts_analysis.json."""
        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator, RunOptions

        with patch("gomoveshift.pipeline.orchestrator.PipelineShortsExtractor") as MockExtractor:  # noqa: N806
            mock_instance = MockExtractor.return_value
            mock_result = MagicMock()
            mock_result.num_rendered = 3
            mock_result.shorts = []
            for i in range(3):
                s = MagicMock()
                s.to_dict.return_value = {
                    "index": i + 1,
                    "start_time": i * 300.0,
                    "end_time": i * 300.0 + 30.0,
                    "duration": 30.0,
                    "avg_motion_energy": 0.04 + i * 0.01,
                }
                mock_result.shorts.append(s)
            mock_result.errors = []
            mock_instance.extract_and_render.return_value = mock_result

            orch = PipelineOrchestrator()
            _result = orch.run(
                RunOptions.from_legacy(
                    video_path=pipeline_output / "testproject" / "testproject.mp4",
                    location="Test Location, France",
                    publish_day="sat",
                    output_base=pipeline_output,
                    skip_color=True,
                    skip_stops=True,
                    skip_ost=True,
                    skip_audio=True,
                    skip_metadata=True,
                    skip_thumbnails=True,
                )
            )

        analysis_file = pipeline_output / "testproject" / "shorts" / "shorts_analysis.json"
        assert analysis_file.exists()
        data = json.loads(analysis_file.read_text())
        assert len(data["segments"]) == 3

    def test_shorts_stage_skippable(self, pipeline_output):
        """--skip-shorts should skip stage 8."""
        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator, RunOptions

        orch = PipelineOrchestrator()
        result = orch.run(
            RunOptions.from_legacy(
                video_path=pipeline_output / "testproject" / "testproject.mp4",
                location="Test Location, France",
                publish_day="sat",
                output_base=pipeline_output,
                skip_color=True,
                skip_stops=True,
                skip_ost=True,
                skip_audio=True,
                skip_metadata=True,
                skip_thumbnails=True,
                skip_shorts=True,
            )
        )

        assert "shorts_analysis" not in result.stages_completed

    def test_single_video_shorts_carry_whole_video_location(self, pipeline_output):
        """Stage call contract: whole-video --location + location_map=None.

        ``location_map=None`` means "derive
        per-clip labels from the GPX inside ``extract_and_render``, falling
        back to the operator ``--location``" — the stage itself still passes
        ``location=ctx.location`` and leaves ``location_map`` unpopulated
        (``None``); pin that call contract so it can't silently drift.
        """
        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator, RunOptions

        with patch("gomoveshift.pipeline.orchestrator.PipelineShortsExtractor") as MockExtractor:  # noqa: N806
            mock_instance = MockExtractor.return_value
            mock_result = MagicMock()
            mock_result.num_rendered = 0
            mock_result.shorts = []
            mock_result.errors = []
            mock_instance.extract_and_render.return_value = mock_result

            orch = PipelineOrchestrator()
            orch.run(
                RunOptions.from_legacy(
                    video_path=pipeline_output / "testproject" / "testproject.mp4",
                    location="Test Location, France",
                    publish_day="sat",
                    output_base=pipeline_output,
                    skip_color=True,
                    skip_stops=True,
                    skip_ost=True,
                    skip_audio=True,
                    skip_metadata=True,
                    skip_thumbnails=True,
                )
            )

        kwargs = mock_instance.extract_and_render.call_args.kwargs
        assert kwargs["location"] == "Test Location, France"
        assert kwargs["location_map"] is None


# ===========================================================================
# Stage 9: Schedule Generation
# ===========================================================================


class TestScheduleStage:
    """Stage 9: Schedule generation calculates posting windows."""

    def test_schedule_stage_creates_yaml(self, pipeline_output):
        """Schedule stage should write release_schedule.yaml."""
        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator, RunOptions

        # Need shorts analysis data first
        shorts_dir = pipeline_output / "testproject" / "shorts"
        shorts_dir.mkdir(parents=True, exist_ok=True)
        analysis = {
            "segments": [
                {
                    "index": i + 1,
                    "start_time": i * 300.0,
                    "end_time": i * 300.0 + 30.0,
                    "duration": 30.0,
                }
                for i in range(9)
            ]
        }
        (shorts_dir / "shorts_analysis.json").write_text(json.dumps(analysis))

        # Mark shorts_analysis as done in state
        stages_dir = pipeline_output / "testproject" / "_stages"
        state = json.loads((stages_dir / "pipeline_state.json").read_text())
        state["stages_completed"].append("thumbnail_generation")
        state["stages_completed"].append("shorts_analysis")
        (stages_dir / "pipeline_state.json").write_text(json.dumps(state))

        with patch("gomoveshift.pipeline.orchestrator.PipelineScheduler") as MockScheduler:  # noqa: N806
            mock_instance = MockScheduler.return_value
            mock_schedule = MagicMock()
            mock_schedule.slots = [MagicMock() for _ in range(9)]
            mock_schedule.to_dict.return_value = {"publish_date": "2026-02-07", "slots": []}
            mock_instance.generate_schedule.return_value = mock_schedule

            orch = PipelineOrchestrator()
            result = orch.run(
                RunOptions.from_legacy(
                    video_path=pipeline_output / "testproject" / "testproject.mp4",
                    location="Test Location, France",
                    publish_day="sat",
                    output_base=pipeline_output,
                    skip_color=True,
                    skip_stops=True,
                    skip_ost=True,
                    skip_audio=True,
                    skip_metadata=True,
                    skip_thumbnails=True,
                    skip_shorts=True,
                )
            )

        assert "schedule_generation" in result.stages_completed

    def test_schedule_stage_skippable(self, pipeline_output):
        """--skip-schedule should skip stage 9."""
        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator, RunOptions

        orch = PipelineOrchestrator()
        result = orch.run(
            RunOptions.from_legacy(
                video_path=pipeline_output / "testproject" / "testproject.mp4",
                location="Test Location, France",
                publish_day="sat",
                output_base=pipeline_output,
                skip_color=True,
                skip_stops=True,
                skip_ost=True,
                skip_audio=True,
                skip_metadata=True,
                skip_thumbnails=True,
                skip_shorts=True,
                skip_schedule=True,
            )
        )

        assert "schedule_generation" not in result.stages_completed


# ===========================================================================
# Stage 10: Social Post Generation
# ===========================================================================


class TestSocialPostStage:
    """Stage 10: Social post generation creates unique captions."""

    def test_social_stage_generates_posts_per_short(self, pipeline_output):
        """Should generate one social post per short segment."""
        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator, RunOptions

        # Set up prerequisites
        shorts_dir = pipeline_output / "testproject" / "shorts"
        shorts_dir.mkdir(parents=True, exist_ok=True)
        analysis = {
            "segments": [
                {
                    "index": i + 1,
                    "start_time": i * 300.0,
                    "end_time": i * 300.0 + 30.0,
                    "duration": 30.0,
                }
                for i in range(9)
            ]
        }
        (shorts_dir / "shorts_analysis.json").write_text(json.dumps(analysis))

        # Mark prior stages done
        stages_dir = pipeline_output / "testproject" / "_stages"
        state = json.loads((stages_dir / "pipeline_state.json").read_text())
        state["stages_completed"].extend(
            [
                "thumbnail_generation",
                "shorts_analysis",
                "schedule_generation",
            ]
        )
        (stages_dir / "pipeline_state.json").write_text(json.dumps(state))

        with patch("gomoveshift.pipeline.orchestrator.SocialGenerator") as MockSocial:  # noqa: N806
            mock_instance = MockSocial.return_value
            mock_posts = []
            for i in range(9):
                p = MagicMock()
                p.to_dict.return_value = {
                    "short_index": i + 1,
                    "caption": f"Scenic drive #{i + 1}",
                    "youtube_title": f"Short #{i + 1}",
                    "youtube_tags": "pov,ireland",
                    "tiktok_title": f"Short #{i + 1}",
                }
                mock_posts.append(p)
            mock_instance.generate_all.return_value = mock_posts

            orch = PipelineOrchestrator()
            result = orch.run(
                RunOptions.from_legacy(
                    video_path=pipeline_output / "testproject" / "testproject.mp4",
                    location="Test Location, France",
                    publish_day="sat",
                    output_base=pipeline_output,
                    skip_color=True,
                    skip_stops=True,
                    skip_ost=True,
                    skip_audio=True,
                    skip_metadata=True,
                    skip_thumbnails=True,
                    skip_shorts=True,
                    skip_schedule=True,
                )
            )

        assert "social_generation" in result.stages_completed
        posts_file = pipeline_output / "testproject" / "shorts" / "social_posts.json"
        assert posts_file.exists()

    def test_social_stage_skippable(self, pipeline_output):
        """--skip-social should skip stage 10."""
        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator, RunOptions

        orch = PipelineOrchestrator()
        result = orch.run(
            RunOptions.from_legacy(
                video_path=pipeline_output / "testproject" / "testproject.mp4",
                location="Test Location, France",
                publish_day="sat",
                output_base=pipeline_output,
                skip_color=True,
                skip_stops=True,
                skip_ost=True,
                skip_audio=True,
                skip_metadata=True,
                skip_thumbnails=True,
                skip_shorts=True,
                skip_schedule=True,
                skip_social=True,
            )
        )

        assert "social_generation" not in result.stages_completed

    def test_single_video_social_carries_whole_video_location(self, pipeline_output):
        """Single-video social captions all use the whole-video --location.

        Companion to the shorts-OST pin: on the operator-owned single-video path
        the social stage builds every post from ``ctx.location`` (one
        ``SocialGenerator`` for the whole run), not a per-clip label. Pin it.
        """
        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator, RunOptions

        stages_dir = pipeline_output / "testproject" / "_stages"
        state = json.loads((stages_dir / "pipeline_state.json").read_text())
        state["stages_completed"].extend(
            ["thumbnail_generation", "shorts_analysis", "schedule_generation"]
        )
        (stages_dir / "pipeline_state.json").write_text(json.dumps(state))

        with patch("gomoveshift.pipeline.orchestrator.SocialGenerator") as MockSocial:  # noqa: N806
            mock_instance = MockSocial.return_value
            mock_instance.generate_all.return_value = []

            orch = PipelineOrchestrator()
            orch.run(
                RunOptions.from_legacy(
                    video_path=pipeline_output / "testproject" / "testproject.mp4",
                    location="Test Location, France",
                    publish_day="sat",
                    output_base=pipeline_output,
                    skip_color=True,
                    skip_stops=True,
                    skip_ost=True,
                    skip_audio=True,
                    skip_metadata=True,
                    skip_thumbnails=True,
                    skip_shorts=True,
                    skip_schedule=True,
                )
            )

        assert MockSocial.call_args.kwargs["location"] == "Test Location, France"


def test_single_video_operator_owned_shorts_location_is_documented():
    """The single-video shorts-location contract is pinned in docs.

    The doc describes GPX-derived per-clip
    labels with an operator-owned ``--location`` fallback; this pin keeps the
    documented contract and the code from drifting apart silently.
    """
    from pathlib import Path

    docs = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "pipeline"
        / "60-single-video-shorts-location-gpx-derived-per-clip-operator-f.md"
    )
    text = docs.read_text(encoding="utf-8").lower()
    assert "single-video shorts location" in text
    assert "operator-owned" in text
    assert "location_map" in text


# ===========================================================================
# Stage 11: Metricool CSV Generation
# ===========================================================================


class TestMetricoolCSVStage:
    """Stage 11: Metricool CSV combines social + schedule into importable CSV."""

    def test_csv_stage_produces_valid_metricool_csv(self, pipeline_output):
        """CSV should match golden reference header format."""
        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator, RunOptions

        # Set up all prerequisites
        shorts_dir = pipeline_output / "testproject" / "shorts"
        shorts_dir.mkdir(parents=True, exist_ok=True)

        analysis = {
            "segments": [
                {
                    "index": i + 1,
                    "start_time": i * 300.0,
                    "end_time": i * 300.0 + 30.0,
                    "duration": 30.0,
                }
                for i in range(3)
            ]
        }
        (shorts_dir / "shorts_analysis.json").write_text(json.dumps(analysis))

        posts = [
            {
                "short_index": i + 1,
                "caption": f"Drive #{i + 1}",
                "youtube_title": f"Short #{i + 1}",
                "youtube_tags": "pov,ireland",
                "youtube_category": "Travel & Events",
                "tiktok_title": f"TikTok #{i + 1}",
                "pinterest_board": "Scenic Drives",
                "first_comment": "#ireland",
            }
            for i in range(3)
        ]
        (shorts_dir / "social_posts.json").write_text(json.dumps(posts))

        # Mark prior stages done
        stages_dir = pipeline_output / "testproject" / "_stages"
        state = json.loads((stages_dir / "pipeline_state.json").read_text())
        state["stages_completed"].extend(
            [
                "thumbnail_generation",
                "shorts_analysis",
                "schedule_generation",
                "social_generation",
            ]
        )
        (stages_dir / "pipeline_state.json").write_text(json.dumps(state))

        with patch("gomoveshift.pipeline.orchestrator.CSVGenerator") as MockCSV:  # noqa: N806
            mock_instance = MockCSV.return_value
            # CSVGenerator.generate returns the path to the CSV
            csv_path = pipeline_output / "testproject" / "metricool_testproject.csv"
            # Write a minimal valid CSV
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Text", "Date", "Time", "Draft", "YouTube"])
                for i in range(3):
                    writer.writerow([f"Drive #{i + 1}", "2026-02-08", "08:00:00", "FALSE", "TRUE"])
            mock_instance.generate.return_value = csv_path

            orch = PipelineOrchestrator()
            result = orch.run(
                RunOptions.from_legacy(
                    video_path=pipeline_output / "testproject" / "testproject.mp4",
                    location="Test Location, France",
                    publish_day="sat",
                    output_base=pipeline_output,
                    skip_color=True,
                    skip_stops=True,
                    skip_ost=True,
                    skip_audio=True,
                    skip_metadata=True,
                    skip_thumbnails=True,
                    skip_shorts=True,
                    skip_schedule=True,
                    skip_social=True,
                )
            )

        assert "csv_generation" in result.stages_completed

    def test_csv_stage_skippable(self, pipeline_output):
        """--skip-csv should skip stage 11."""
        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator, RunOptions

        orch = PipelineOrchestrator()
        result = orch.run(
            RunOptions.from_legacy(
                video_path=pipeline_output / "testproject" / "testproject.mp4",
                location="Test Location, France",
                publish_day="sat",
                output_base=pipeline_output,
                skip_color=True,
                skip_stops=True,
                skip_ost=True,
                skip_audio=True,
                skip_metadata=True,
                skip_thumbnails=True,
                skip_shorts=True,
                skip_schedule=True,
                skip_social=True,
                skip_csv=True,
            )
        )

        assert "csv_generation" not in result.stages_completed


# ===========================================================================
# Resume Integration
# ===========================================================================


class TestPostProcessingResume:
    """Resume logic correctly handles all 11 stages."""

    def test_resume_skips_all_completed_post_stages(self, pipeline_output):
        """Pipeline with all stages complete should skip everything."""
        stages_dir = pipeline_output / "testproject" / "_stages"
        state = json.loads((stages_dir / "pipeline_state.json").read_text())
        state["stages_completed"].extend(
            [
                "thumbnail_generation",
                "shorts_analysis",
                "schedule_generation",
                "social_generation",
                "csv_generation",
            ]
        )
        (stages_dir / "pipeline_state.json").write_text(json.dumps(state))

        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator, RunOptions

        with (
            patch("gomoveshift.pipeline.orchestrator.generate_thumbnails_from_video") as mock_thumb,
            patch("gomoveshift.pipeline.orchestrator.PipelineShortsExtractor") as mock_shorts,
            patch("gomoveshift.pipeline.orchestrator.PipelineScheduler") as mock_sched,
            patch("gomoveshift.pipeline.orchestrator.SocialGenerator") as mock_social,
            patch("gomoveshift.pipeline.orchestrator.CSVGenerator") as mock_csv,
        ):
            orch = PipelineOrchestrator()
            result = orch.run(
                RunOptions.from_legacy(
                    video_path=pipeline_output / "testproject" / "testproject.mp4",
                    location="Test Location, France",
                    publish_day="sat",
                    output_base=pipeline_output,
                    skip_color=True,
                    skip_stops=True,
                    skip_ost=True,
                    skip_audio=True,
                    skip_metadata=True,
                )
            )

            mock_thumb.assert_not_called()
            mock_shorts.assert_not_called()
            mock_sched.assert_not_called()
            mock_social.assert_not_called()
            mock_csv.assert_not_called()

        assert result.success

    def test_pipeline_state_includes_new_stages(self, pipeline_output):
        """Pipeline state file should track post-processing stage names."""
        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator, RunOptions

        with patch("gomoveshift.pipeline.orchestrator.generate_thumbnails_from_video") as mock_gen:
            mock_gen.return_value = {"frame1": {"minimal": "a.jpg"}}
            thumb_dir = pipeline_output / "testproject" / "thumbnails" / "styled"
            thumb_dir.mkdir(parents=True)
            # A real thumbnail on disk: the stage gate now requires one.
            (thumb_dir / "a.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 20480)

            orch = PipelineOrchestrator()
            _result = orch.run(
                RunOptions.from_legacy(
                    video_path=pipeline_output / "testproject" / "testproject.mp4",
                    location="Test Location, France",
                    publish_day="sat",
                    output_base=pipeline_output,
                    skip_color=True,
                    skip_stops=True,
                    skip_ost=True,
                    skip_audio=True,
                    skip_metadata=True,
                    skip_shorts=True,
                    skip_schedule=True,
                    skip_social=True,
                    skip_csv=True,
                )
            )

        # Check saved state includes thumbnail stage
        state_file = pipeline_output / "testproject" / "_stages" / "pipeline_state.json"
        state = json.loads(state_file.read_text())
        assert "thumbnail_generation" in state["stages_completed"]
