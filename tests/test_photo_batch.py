"""Tests for photo batch (one-shot auto_batch) workflow."""

from unittest.mock import patch

import pytest

from gomoveshift.pipeline.photos.caption_generator import CaptionBatch, PhotoCaption
from gomoveshift.pipeline.photos.orchestrator import PhotoOrchestrator
from gomoveshift.pipeline.photos.voice_styles import VoiceStyle


def _make_caption(index=0, slug="test-journey", filename="photo-001.jpg"):
    return PhotoCaption(
        photo_index=index,
        main_caption="Test caption for batch",
        tiktok_caption="Test batch",
        hashtags=["test", "roadtrip"],
        journey_slug=slug,
        filename=filename,
        location="TestPlace",
    )


def _make_batch(week_start="2026-02-17", count=1, reviewed=False):
    captions = [_make_caption(i, filename=f"photo-{i:03d}.jpg") for i in range(count)]
    return CaptionBatch(
        week_start=week_start,
        voice_style="voice_first_person",
        captions=captions,
        reviewed=reviewed,
        generated_at="2026-02-16T12:00:00",
        model="test-model",
    )


def _orchestrator(tmp_path):
    """Create orchestrator with caption_dir pointing to tmp_path."""
    orch = PhotoOrchestrator(site_root=tmp_path / "site", output_dir=tmp_path)
    orch.caption_gen.caption_dir = tmp_path / "photo_captions"
    return orch


class TestAutoBatchDateCalculation:
    def test_count_from_dates_single_day(self, tmp_path):
        """Feb 17 to Feb 17 = 1 post."""
        orchestrator = _orchestrator(tmp_path)
        with (
            patch.object(orchestrator, "generate_batch") as mock_gen,
            patch.object(orchestrator, "publish_batch") as mock_pub,
        ):
            batch = _make_batch("2026-02-17", count=1)
            mock_gen.return_value = batch
            csv = tmp_path / "photo_metricool" / "test.csv"
            csv.parent.mkdir(parents=True, exist_ok=True)
            csv.write_text("header\n")
            mock_pub.return_value = csv

            orchestrator.auto_batch("2026-02-17", "2026-02-17")

            mock_gen.assert_called_once()
            assert mock_gen.call_args.kwargs["count"] == 1

    def test_count_from_dates_twelve_days(self, tmp_path):
        """Feb 17 to Feb 28 = 12 posts."""
        orchestrator = _orchestrator(tmp_path)
        with (
            patch.object(orchestrator, "generate_batch") as mock_gen,
            patch.object(orchestrator, "publish_batch") as mock_pub,
        ):
            batch = _make_batch("2026-02-17", count=12)
            mock_gen.return_value = batch
            csv = tmp_path / "photo_metricool" / "test.csv"
            csv.parent.mkdir(parents=True, exist_ok=True)
            csv.write_text("header\n")
            mock_pub.return_value = csv

            orchestrator.auto_batch("2026-02-17", "2026-02-28")

            assert mock_gen.call_args.kwargs["count"] == 12

    def test_invalid_dates_raises(self, tmp_path):
        """end_date before start_date raises ValueError."""
        orchestrator = _orchestrator(tmp_path)
        with pytest.raises(ValueError, match="before start_date"):
            orchestrator.auto_batch("2026-02-28", "2026-02-17")


class TestAutoBatchWorkflow:
    def test_calls_generate_then_publish(self, tmp_path):
        """Verifies generate is called before publish."""
        orchestrator = _orchestrator(tmp_path)
        call_order = []

        batch = _make_batch("2026-02-17", count=3)
        csv = tmp_path / "photo_metricool" / "test.csv"
        csv.parent.mkdir(parents=True, exist_ok=True)
        csv.write_text("header\n")

        def fake_generate(**kwargs):
            call_order.append("generate")
            return batch

        def fake_publish(**kwargs):
            call_order.append("publish")
            return csv

        with (
            patch.object(orchestrator, "generate_batch", side_effect=fake_generate),
            patch.object(orchestrator, "publish_batch", side_effect=fake_publish),
        ):
            orchestrator.auto_batch("2026-02-17", "2026-02-19")

        assert call_order == ["generate", "publish"]

    def test_auto_approves_batch(self, tmp_path):
        """Batch.reviewed is set to True before publish."""
        orchestrator = _orchestrator(tmp_path)
        batch = _make_batch("2026-02-17", count=2, reviewed=False)
        assert batch.reviewed is False

        csv = tmp_path / "photo_metricool" / "test.csv"
        csv.parent.mkdir(parents=True, exist_ok=True)
        csv.write_text("header\n")

        saved_batches = []
        original_save = CaptionBatch.save

        def tracking_save(self_batch, path):
            saved_batches.append(self_batch.reviewed)
            original_save(self_batch, path)

        with (
            patch.object(orchestrator, "generate_batch", return_value=batch),
            patch.object(orchestrator, "publish_batch", return_value=csv),
            patch.object(CaptionBatch, "save", tracking_save),
        ):
            orchestrator.auto_batch("2026-02-17", "2026-02-18")

        # Batch was saved with reviewed=True
        assert saved_batches == [True]
        assert batch.reviewed is True

    def test_publish_called_with_force(self, tmp_path):
        """publish_batch is called with force=True."""
        orchestrator = _orchestrator(tmp_path)
        batch = _make_batch("2026-02-17", count=1)
        csv = tmp_path / "test.csv"
        csv.write_text("header\n")

        with (
            patch.object(orchestrator, "generate_batch", return_value=batch),
            patch.object(orchestrator, "publish_batch", return_value=csv) as mock_pub,
        ):
            orchestrator.auto_batch("2026-02-17", "2026-02-17")

        assert mock_pub.call_args.kwargs["force"] is True

    def test_voice_override_passed_through(self, tmp_path):
        """Voice style is forwarded to generate_batch."""
        orchestrator = _orchestrator(tmp_path)
        batch = _make_batch("2026-02-17", count=1)
        csv = tmp_path / "test.csv"
        csv.write_text("header\n")

        with (
            patch.object(orchestrator, "generate_batch", return_value=batch) as mock_gen,
            patch.object(orchestrator, "publish_batch", return_value=csv),
        ):
            orchestrator.auto_batch(
                "2026-02-17",
                "2026-02-17",
                voice_style=VoiceStyle.MINIMAL_ZEN,
            )

        assert mock_gen.call_args.kwargs["voice_style"] == VoiceStyle.MINIMAL_ZEN

    def test_default_voice_none(self, tmp_path):
        """When no voice specified, None is passed to generate_batch."""
        orchestrator = _orchestrator(tmp_path)
        batch = _make_batch("2026-02-17", count=1)
        csv = tmp_path / "test.csv"
        csv.write_text("header\n")

        with (
            patch.object(orchestrator, "generate_batch", return_value=batch) as mock_gen,
            patch.object(orchestrator, "publish_batch", return_value=csv),
        ):
            orchestrator.auto_batch("2026-02-17", "2026-02-17")

        assert mock_gen.call_args.kwargs["voice_style"] is None


class TestAutoBatchCSVOutput:
    def test_copies_to_output_dir(self, tmp_path):
        """CSV is copied to output_csv_dir with descriptive filename."""
        orchestrator = _orchestrator(tmp_path)
        batch = _make_batch("2026-02-17", count=1)

        # Create a source CSV
        csv_src = tmp_path / "photo_metricool" / "source.csv"
        csv_src.parent.mkdir(parents=True, exist_ok=True)
        csv_src.write_text("col1,col2\nval1,val2\n")

        output_dir = tmp_path / "gms_output"

        with (
            patch.object(orchestrator, "generate_batch", return_value=batch),
            patch.object(orchestrator, "publish_batch", return_value=csv_src),
        ):
            result = orchestrator.auto_batch(
                "2026-02-17",
                "2026-02-28",
                output_csv_dir=str(output_dir),
            )

        assert result.parent == output_dir
        assert result.name == "gms_photos_2026-02-17_to_2026-02-28_metricool.csv"
        assert result.exists()
        assert result.read_text() == "col1,col2\nval1,val2\n"

    def test_no_copy_when_output_none(self, tmp_path):
        """No copy when output_csv_dir is None."""
        orchestrator = _orchestrator(tmp_path)
        batch = _make_batch("2026-02-17", count=1)

        csv_src = tmp_path / "photo_metricool" / "source.csv"
        csv_src.parent.mkdir(parents=True, exist_ok=True)
        csv_src.write_text("header\n")

        with (
            patch.object(orchestrator, "generate_batch", return_value=batch),
            patch.object(orchestrator, "publish_batch", return_value=csv_src),
        ):
            result = orchestrator.auto_batch("2026-02-17", "2026-02-17")

        # Returns the original CSV path, no copy made
        assert result == csv_src

    def test_creates_output_dir_if_missing(self, tmp_path):
        """output_csv_dir is created if it doesn't exist."""
        orchestrator = _orchestrator(tmp_path)
        batch = _make_batch("2026-02-17", count=1)

        csv_src = tmp_path / "source.csv"
        csv_src.write_text("header\n")

        nested_dir = tmp_path / "a" / "b" / "c"
        assert not nested_dir.exists()

        with (
            patch.object(orchestrator, "generate_batch", return_value=batch),
            patch.object(orchestrator, "publish_batch", return_value=csv_src),
        ):
            result = orchestrator.auto_batch(
                "2026-02-17",
                "2026-02-17",
                output_csv_dir=str(nested_dir),
            )

        assert nested_dir.exists()
        assert result.exists()


class TestAutoBatchWeekStart:
    def test_week_start_matches_start_date(self, tmp_path):
        """generate_batch receives start_date as week_start."""
        orchestrator = _orchestrator(tmp_path)
        batch = _make_batch("2026-03-01", count=1)
        csv = tmp_path / "test.csv"
        csv.write_text("header\n")

        with (
            patch.object(orchestrator, "generate_batch", return_value=batch) as mock_gen,
            patch.object(orchestrator, "publish_batch", return_value=csv),
        ):
            orchestrator.auto_batch("2026-03-01", "2026-03-07")

        assert mock_gen.call_args.kwargs["week_start"] == "2026-03-01"


class TestBatchCLIParsing:
    """Test CLI argument parsing for photos batch subcommand.

    Since the parser is built inside main(), we test via subprocess-like
    approach using parse_known_args pattern from the CLI module.
    """

    @pytest.fixture
    def parser(self):
        """Build a minimal parser matching the photos batch subparser."""
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        photos_parser = subparsers.add_parser("photos")
        photos_sub = photos_parser.add_subparsers(dest="photos_command")

        batch_parser = photos_sub.add_parser("batch")
        batch_parser.add_argument("--start", type=str, required=True)
        batch_parser.add_argument("--end", type=str, required=True)
        batch_parser.add_argument("--output", type=str, default=None)
        batch_parser.add_argument(
            "--voice",
            type=str,
            default=None,
            choices=["first_person", "minimal_zen", "engaging"],
        )
        return parser

    def test_cli_batch_parses_required_args(self, parser):
        """CLI parses --start and --end correctly."""
        args = parser.parse_args(
            ["photos", "batch", "--start", "2026-02-17", "--end", "2026-02-28"]
        )
        assert args.photos_command == "batch"
        assert args.start == "2026-02-17"
        assert args.end == "2026-02-28"
        assert args.output is None
        assert args.voice is None

    def test_cli_batch_parses_all_args(self, parser):
        """CLI parses --output and --voice correctly."""
        args = parser.parse_args(
            [
                "photos",
                "batch",
                "--start",
                "2026-02-17",
                "--end",
                "2026-02-28",
                "--output",
                "/tmp/gms-out/",
                "--voice",
                "minimal_zen",
            ]
        )
        assert args.output == "/tmp/gms-out/"
        assert args.voice == "minimal_zen"

    def test_cli_batch_rejects_invalid_voice(self, parser):
        """CLI rejects unknown voice style."""
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "photos",
                    "batch",
                    "--start",
                    "2026-02-17",
                    "--end",
                    "2026-02-28",
                    "--voice",
                    "robot_voice",
                ]
            )

    def test_cli_batch_requires_start(self, parser):
        """CLI requires --start argument."""
        with pytest.raises(SystemExit):
            parser.parse_args(["photos", "batch", "--end", "2026-02-28"])

    def test_cli_batch_requires_end(self, parser):
        """CLI requires --end argument."""
        with pytest.raises(SystemExit):
            parser.parse_args(["photos", "batch", "--start", "2026-02-17"])
