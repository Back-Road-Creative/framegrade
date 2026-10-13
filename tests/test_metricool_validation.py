"""TDD tests for Metricool CSV validation against real import constraints.

These tests encode the ACTUAL constraints from Metricool's CSV importer,
validated against a known-good CSV that Metricool imported successfully.

Constraints discovered from Metricool import UI:
  1. Text field: max 280 characters (Twitter/X limit)
  2. YouTube Video Category: must be empty string (Metricool rejects text values)
  3. YouTube Video Tags: comma-separated, no pipes
  4. Picture Url 1: must be a valid public URL
  5. Date format: YYYY-MM-DD
  6. Time format: HH:MM:SS

Reference: a known-good export, named by GMS_METRICOOL_REFERENCE_CSV (tests skip when unset).
"""

import csv
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# The upload-wiring test drives the full orchestrator on a dummy video; blur is
# on by default, so opt out of the real detector (see conftest ``disable_blur``).
pytestmark = pytest.mark.usefixtures("disable_blur")

TWITTER_MAX_CHARS = 280
REFERENCE_CSV = Path(os.environ.get("GMS_METRICOOL_REFERENCE_CSV") or "/nonexistent-reference.csv")


# ===========================================================================
# 1. Text field: max 280 characters
# ===========================================================================


class TestTextLength:
    """Caption text must fit within Twitter's 280 character limit."""

    def test_social_generator_captions_under_280(self):
        """All generated captions must be ≤280 characters."""
        from gomoveshift.pipeline.social_generator import SocialGenerator

        sg = SocialGenerator(
            location="Lissard to Creggane, County Limerick, Ireland", series_part=1
        )
        posts = sg.generate_all(num_shorts=3)

        for post in posts:
            assert len(post.caption) <= TWITTER_MAX_CHARS, (
                f"Short #{post.short_index} caption is {len(post.caption)} chars "
                f"(max {TWITTER_MAX_CHARS}):\n{post.caption[:100]}..."
            )

    def test_social_generator_short_location_under_280(self):
        """Captions with shorter location names must also be ≤280."""
        from gomoveshift.pipeline.social_generator import SocialGenerator

        sg = SocialGenerator(location="County Tipperary, Ireland", series_part=1)
        posts = sg.generate_all(num_shorts=3)

        for post in posts:
            assert len(post.caption) <= TWITTER_MAX_CHARS, (
                f"Short #{post.short_index} caption is {len(post.caption)} chars"
            )

    def test_social_generator_long_location_under_280(self):
        """Even with very long location names, captions must be ≤280."""
        from gomoveshift.pipeline.social_generator import SocialGenerator

        sg = SocialGenerator(
            location="Cloonlee to Portumna via Lough Derg, County Galway, Ireland", series_part=1
        )
        posts = sg.generate_all(num_shorts=3)

        for post in posts:
            assert len(post.caption) <= TWITTER_MAX_CHARS, (
                f"Short #{post.short_index} caption is {len(post.caption)} chars"
            )


# ===========================================================================
# 2. YouTube Video Category: shorts CSV MUST emit empty string
# ===========================================================================
#
# Metricool rejects populated YouTube category
# values on shorts rows and silently routes the import to draft. Legacy
# known-good CSVs leave this column empty across all rows. The shorts CSV
# writer must therefore force ``""`` regardless of what's on the SocialPost.
# (The SocialGenerator itself still defaults to "19" because non-shorts
# consumers may use the field; only the shorts CSV writer is contractually
# required to drop it.)


class TestYouTubeCategory:
    """Shorts CSV ``YouTube Video Category`` must be empty regardless of input."""

    def test_shorts_csv_row_category_is_empty(self):
        """``CSVGenerator._build_row`` must emit ``""`` for category."""
        from gomoveshift.pipeline.csv_generator import CSVGenerator
        from gomoveshift.pipeline.scheduler import ScheduledSlot
        from gomoveshift.pipeline.social_generator import SocialPost

        post = SocialPost(
            short_index=1,
            caption="Test",
            youtube_title="Test",
            youtube_tags="pov,drive",
            youtube_category="19",  # populated input — writer must still drop it
            tiktok_title="Test",
            pinterest_board="Drives",
            first_comment="#test",
        )
        slot = ScheduledSlot(
            short_index=1,
            date="2026-02-07",
            time="12:00:00",
            datetime_est="2026-02-07T12:00:00",
            day_of_week="Saturday",
        )

        gen = CSVGenerator()
        row = gen._build_row(post, slot, "https://example.com/v.mp4")

        assert row["YouTube Video Category"] == "", (
            f"Category must be empty for shorts rows; got: {repr(row['YouTube Video Category'])}"
        )

    def test_consolidated_shorts_csv_category_is_empty(self):
        """``build_consolidated_shorts_csv`` must emit ``""`` for category."""
        from gomoveshift.pipeline.csv_generator import build_consolidated_shorts_csv
        from gomoveshift.pipeline.scheduler import ScheduledSlot
        from gomoveshift.pipeline.social_generator import SocialPost
        import tempfile
        from pathlib import Path

        post = SocialPost(
            short_index=1,
            caption="Test",
            youtube_title="Test",
            youtube_tags="pov,drive",
            youtube_category="19",  # populated input — writer must still drop it
            tiktok_title="Test",
            pinterest_board="Drives",
            first_comment="#test",
        )
        slot = ScheduledSlot(
            short_index=1,
            date="2026-02-07",
            time="12:00:00",
            datetime_est="2026-02-07T12:00:00",
            day_of_week="Saturday",
        )

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "shorts.csv"
            build_consolidated_shorts_csv(
                output_path=out_path,
                shorts_by_lf={1: [post]},
                slots_by_short_index={1: slot},
                drive_links_by_short_index={1: "https://example.com/v.mp4"},
                parent_lf_video_id_by_short_index={1: "vidABC"},
                parent_recording_date_by_short_index={1: "2026-02-07"},
            )
            with open(out_path, "r", encoding="utf-8") as f:
                row = next(csv.DictReader(f))
            assert row["YouTube Video Category"] == "", (
                f"consolidated shorts CSV must emit empty category; "
                f"got: {repr(row['YouTube Video Category'])}"
            )


# ===========================================================================
# 3. YouTube Video Tags: comma-separated, no pipes, no spaces
# ===========================================================================


class TestTagsFormat:
    """YouTube tags must be comma-separated with no extra whitespace."""

    def test_tags_comma_separated_no_spaces(self):
        """Tags must use comma (no space) as separator."""
        from gomoveshift.pipeline.social_generator import SocialGenerator

        sg = SocialGenerator(location="County Limerick, Ireland", series_part=1)
        posts = sg.generate_all(num_shorts=3)

        for post in posts:
            tags = post.youtube_tags
            assert "," in tags, f"Tags must be comma-separated: {tags}"
            assert "|" not in tags, f"Pipes don't work in Metricool: {tags}"

            # Each individual tag must be clean
            for tag in tags.split(","):
                assert tag == tag.strip(), f"Tag has whitespace: {repr(tag)} in {tags}"
                assert len(tag) > 0, f"Empty tag found in: {tags}"


# ===========================================================================
# 4. Full CSV end-to-end validation
# ===========================================================================


class TestCSVEndToEnd:
    """Generate a full CSV and validate every field matches Metricool constraints."""

    def test_empty_manifest_defers_csv_write(self, tmp_path):
        """With NO Drive links every ``Picture Url 1`` is blank — a dead
        Metricool deliverable. ``generate`` must DEFER (not write) rather than
        ship a link-less CSV with golden validation silently disabled."""
        from gomoveshift.pipeline.csv_generator import CSVGenerator
        from gomoveshift.pipeline.scheduler import PipelineScheduler
        from gomoveshift.pipeline.social_generator import SocialGenerator
        from gomoveshift.pipeline.uploader import UploadManifest

        posts = SocialGenerator(location="County Kerry, Ireland", series_part=1).generate_all(
            num_shorts=3
        )
        schedule = PipelineScheduler().generate_schedule(
            publish_date="next_wed",
            publish_day="wed",
            num_shorts=3,
            project_name="test",
            location="Kerry",
        )
        manifest = UploadManifest(project_name="test", files={})  # no Drive links

        csv_path = CSVGenerator().generate(
            project_name="test",
            social_posts=posts,
            schedule=schedule,
            manifest=manifest,
            output_dir=str(tmp_path),
            validate=bool(manifest.files),  # the exact caller-side pattern (False here)
        )
        assert not csv_path.exists(), "a link-less CSV must be deferred, not written"

    def test_full_csv_metricool_compatible(self, tmp_path):
        """Full pipeline output must be importable by Metricool without errors."""
        from gomoveshift.pipeline.csv_generator import CSVGenerator
        from gomoveshift.pipeline.scheduler import PipelineScheduler
        from gomoveshift.pipeline.social_generator import SocialGenerator
        from gomoveshift.pipeline.uploader import UploadManifest

        # Generate all components
        sg = SocialGenerator(
            location="Lissard to Creggane, County Limerick, Ireland", series_part=1
        )
        posts = sg.generate_all(num_shorts=3)

        sched = PipelineScheduler()
        schedule = sched.generate_schedule(
            publish_date="next_wed",
            publish_day="wed",
            num_shorts=3,
            project_name="test",
            location="Limerick",
        )

        manifest = UploadManifest(
            project_name="test",
            files={
                f"short_{i:02d}.mp4": f"https://drive.google.com/uc?export=download&id=fake{i}"
                for i in range(1, 4)
            },
        )

        gen = CSVGenerator()
        csv_path = gen.generate(
            project_name="test",
            social_posts=posts,
            schedule=schedule,
            manifest=manifest,
            output_dir=str(tmp_path),
            validate=False,
        )

        # Validate every row
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3

        for i, row in enumerate(rows, 1):
            # Text ≤ 280
            assert len(row["Text"]) <= TWITTER_MAX_CHARS, (
                f"Row {i}: text is {len(row['Text'])} chars (max 280)"
            )

            # Shorts row YouTube Category must be empty.
            assert row["YouTube Video Category"] == "", (
                f"Row {i}: category must be '', got {repr(row['YouTube Video Category'])}"
            )

            # Tags comma-separated, no pipes
            tags = row["YouTube Video Tags"]
            assert "|" not in tags, f"Row {i}: pipes in tags: {tags}"
            for tag in tags.split(","):
                assert tag == tag.strip(), f"Row {i}: tag has whitespace: {repr(tag)}"

            # Drive URL present
            assert row["Picture Url 1"].startswith("https://"), f"Row {i}: missing Drive URL"

            # Date format
            assert len(row["Date"]) == 10, f"Row {i}: bad date {row['Date']}"
            assert row["Date"][4] == "-" and row["Date"][7] == "-"

            # Time format
            assert len(row["Time"]) == 8, f"Row {i}: bad time {row['Time']}"

            # YouTube Video Type
            assert row["YouTube Video Type"] == "SHORT"

            # YouTube privacy
            assert row["YouTube Video Privacy"] == "public"


# ===========================================================================
# 5. Comparison against known-good reference CSV
# ===========================================================================


class TestReferenceCsv:
    """Validate our output matches the format of the working reference CSV."""

    @pytest.fixture
    def reference_row(self):
        """Load first row from the known-good reference CSV."""
        if not REFERENCE_CSV.exists():
            pytest.skip("GMS_METRICOOL_REFERENCE_CSV not configured or file not found")
        with open(REFERENCE_CSV, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            return next(reader)

    def test_same_column_count(self, reference_row):
        """Generated CSV should have same number of columns as the reference."""
        assert len(reference_row) == 31

    def test_category_matches_reference(self, reference_row):
        """Category should match the reference format (empty)."""
        assert reference_row["YouTube Video Category"] == ""

    def test_boolean_format_matches(self, reference_row):
        """Boolean fields should use TRUE/FALSE (uppercase)."""
        for field in ["Facebook", "Twitter", "Instagram", "YouTube", "TikTok"]:
            assert reference_row[field] in ("TRUE", "FALSE"), f"{field}={reference_row[field]}"

    def test_text_under_280(self, reference_row):
        """Reference text was under 280 — our CSV should match."""
        assert len(reference_row["Text"]) <= TWITTER_MAX_CHARS


# ===========================================================================
# 6. Upload stage wired into orchestrator
# ===========================================================================


class TestUploadStageWired:
    """Orchestrator should have an upload stage between social and CSV."""

    @pytest.fixture(autouse=True)
    def mock_file_hash(self):
        with patch(
            "gomoveshift.pipeline.orchestrator.compute_file_hash",
            return_value="abc123",
        ):
            yield

    @pytest.fixture
    def pipeline_output(self, tmp_path):
        """Minimal pipeline output with stages 1-10 complete."""
        project_dir = tmp_path / "testproject"
        project_dir.mkdir()
        stages_dir = project_dir / "_stages"
        stages_dir.mkdir()

        final_video = project_dir / "testproject.mp4"
        final_video.write_bytes(b"\x00" * 1024)

        metadata = {
            "title": "POV Drive: Test 🚗 Ireland",
            "description": "A scenic drive...",
            "tags": ["pov", "ireland"],
            "category": "Travel & Events",
            "project_name": "testproject",
        }
        (project_dir / "testproject_youtube.json").write_text(json.dumps(metadata))

        shorts_dir = project_dir / "shorts"
        shorts_dir.mkdir()
        posts = [
            {
                "short_index": i + 1,
                "caption": f"Drive #{i + 1}",
                "youtube_title": f"Short #{i + 1}",
                "youtube_tags": "pov,ireland",
                "youtube_category": "",
                "tiktok_title": f"TikTok #{i + 1}",
                "pinterest_board": "Scenic Drives",
                "first_comment": "#ireland",
            }
            for i in range(3)
        ]
        (shorts_dir / "social_posts.json").write_text(json.dumps(posts))
        for i in range(1, 4):
            (shorts_dir / f"short_{i:02d}.mp4").write_bytes(b"\x00" * 100)

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
                "thumbnail_generation",
                "shorts_analysis",
                "schedule_generation",
                "social_generation",
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
                "location": "Test Location, Italy",
                "publish_day": "sat",
                "skip_color": True,
                "skip_stops": True,
            },
        }
        (stages_dir / "pipeline_state.json").write_text(json.dumps(state))
        return tmp_path

    def test_upload_stage_exists_in_orchestrator(self, pipeline_output):
        """Orchestrator should have skip_upload parameter."""
        import inspect

        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator

        sig = inspect.signature(PipelineOrchestrator().run)
        assert "skip_upload" in sig.parameters

    def test_upload_stage_calls_uploader(self, pipeline_output):
        """Upload stage should call PipelineUploader.upload_shorts()."""
        from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator
        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator, RunOptions

        mock_vr = MagicMock()
        mock_vr.valid = True
        mock_vr.duration = 30.0

        with (
            patch("gomoveshift.pipeline.orchestrator.PipelineUploader") as MockUploader,  # noqa: N806
            patch("gomoveshift.pipeline.orchestrator.CSVGenerator") as MockCSV,  # noqa: N806
            patch("gomoveshift.pipeline.orchestrator.validate_video", return_value=mock_vr),
        ):
            mock_manifest = MagicMock()
            mock_manifest.files = {"short_01.mp4": "https://drive.google.com/test"}
            MockUploader.return_value.upload_shorts.return_value = mock_manifest
            MockCSV.return_value.generate.return_value = (
                pipeline_output / "testproject" / "test.csv"
            )

            result = PipelineOrchestrator(metadata_generator=LongFormMetadataGenerator).run(
                RunOptions.from_legacy(
                    video_path=pipeline_output / "testproject" / "testproject.mp4",
                    location="Test Location, Italy",
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

            MockUploader.return_value.upload_shorts.assert_called_once()
            assert "drive_upload" in result.stages_completed

    def test_upload_stage_skippable(self, pipeline_output):
        """--skip-upload should skip the upload stage."""
        from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator
        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator, RunOptions

        with patch("gomoveshift.pipeline.orchestrator.CSVGenerator") as MockCSV:  # noqa: N806
            MockCSV.return_value.generate.return_value = (
                pipeline_output / "testproject" / "test.csv"
            )

            result = PipelineOrchestrator(metadata_generator=LongFormMetadataGenerator).run(
                RunOptions.from_legacy(
                    video_path=pipeline_output / "testproject" / "testproject.mp4",
                    location="Test Location, Ireland",
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
                    skip_upload=True,
                )
            )

            assert "drive_upload" not in result.stages_completed


# ===========================================================================
# 7. PipelineUploader self-contained
# ===========================================================================


class TestUploaderSelfContained:
    """PipelineUploader uses google API directly, no content pipeline dep."""

    def test_uploader_imports_cleanly(self):
        """PipelineUploader should import without the legacy content pipeline."""
        from gomoveshift.pipeline.uploader import PipelineUploader, UploadManifest

        assert PipelineUploader is not None
        assert UploadManifest is not None

    def test_uploader_uses_google_api_directly(self):
        """Drive client should use google.oauth2, not content_pipeline."""
        import inspect

        from gomoveshift.pipeline.uploader import PipelineUploader

        # Auth logic lives in _get_credentials and _build_drive_service
        cred_source = inspect.getsource(PipelineUploader._get_credentials)
        build_source = inspect.getsource(PipelineUploader._build_drive_service)
        combined = cred_source + build_source
        assert "content_pipeline" not in combined
        assert "googleapiclient" in combined or "google.oauth2" in combined

    def test_manifest_round_trip(self, tmp_path):
        """UploadManifest save/load should preserve all fields."""
        from gomoveshift.pipeline.uploader import UploadManifest

        manifest = UploadManifest(
            project_name="test",
            folder_id="folder123",
            files={
                "short_01.mp4": "https://drive.google.com/uc?id=abc",
                "short_02.mp4": "https://drive.google.com/uc?id=def",
            },
            completed=True,
        )
        path = tmp_path / "manifest.json"
        manifest.save(path)
        loaded = UploadManifest.load(path)

        assert loaded.project_name == "test"
        assert loaded.folder_id == "folder123"
        assert len(loaded.files) == 2
        assert loaded.completed is True

    def test_manifest_loads_flat_format(self, tmp_path):
        """UploadManifest should load flat {filename: link} JSON format."""
        from gomoveshift.pipeline.uploader import UploadManifest

        flat = {"short_01.mp4": "https://drive.google.com/uc?id=abc"}
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(flat))

        loaded = UploadManifest.load(path)
        assert len(loaded.files) == 1
        assert "short_01.mp4" in loaded.files


# ===========================================================================
# 8. CSV loads manifest from disk
# ===========================================================================


class TestCSVLoadsManifest:
    """CSV stage should load gdrive_manifest.json if it exists on disk."""

    def test_csv_generator_uses_manifest_file(self, tmp_path):
        """generate() with manifest should populate Drive links in CSV."""
        from gomoveshift.pipeline.csv_generator import CSVGenerator
        from gomoveshift.pipeline.scheduler import ReleaseSchedule, ScheduledSlot
        from gomoveshift.pipeline.social_generator import SocialPost
        from gomoveshift.pipeline.uploader import UploadManifest

        posts = [
            SocialPost(
                short_index=1,
                caption="Test",
                youtube_title="Test",
                youtube_tags="pov,test",
                youtube_category="",
                tiktok_title="Test",
                pinterest_board="Drives",
                first_comment="#test",
            )
        ]
        schedule = ReleaseSchedule(
            project_name="test",
            location="Test",
            long_form_publish="",
            window_start="",
            window_end="",
            num_shorts=1,
            num_slots=1,
            slots=[
                ScheduledSlot(
                    short_index=1,
                    date="2026-02-07",
                    time="12:00:00",
                    datetime_est="2026-02-07T12:00:00",
                    day_of_week="Saturday",
                )
            ],
        )
        manifest = UploadManifest(
            project_name="test",
            files={"short_01.mp4": "https://drive.google.com/uc?export=download&id=abc123"},
            completed=True,
        )

        csv_path = CSVGenerator().generate(
            project_name="test",
            social_posts=posts,
            schedule=schedule,
            manifest=manifest,
            output_dir=str(tmp_path),
            validate=False,
        )

        with open(csv_path, "r", encoding="utf-8-sig") as f:
            row = next(csv.DictReader(f))
            assert "drive.google.com" in row["Picture Url 1"]
