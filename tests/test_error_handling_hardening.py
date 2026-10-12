"""Tests for error handling hardening.

Covers:
- GPS parser: malformed/missing lat/lon trackpoints
- Stop processor: invalid JSON from ffprobe
- Upload handler: bare except replaced with logged warnings
"""

import json
import logging
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest


# ---------------------------------------------------------------------------
# GPS parser malformed trackpoint guard
# ---------------------------------------------------------------------------


class TestGPSParserMalformedTrackpoints:
    """gps_parser.parse_gpx_file must skip trackpoints with bad lat/lon."""

    def _make_gpx(self, trkpts_xml: str) -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<gpx xmlns="http://www.topografix.com/GPX/1/1">\n'
            f"<trk><trkseg>\n{trkpts_xml}\n</trkseg></trk>\n"
            "</gpx>\n"
        )

    def test_skips_none_lat_lon(self, tmp_path, caplog):
        """Trackpoint with missing lat/lon attributes is skipped with warning."""
        gpx_content = self._make_gpx(
            '<trkpt lat="51.5" lon="-0.1"><time>2025-01-01T00:00:00Z</time></trkpt>'
            "\n<trkpt><time>2025-01-01T00:01:00Z</time></trkpt>"
            '\n<trkpt lat="51.6" lon="-0.2"><time>2025-01-01T00:02:00Z</time></trkpt>'
        )
        gpx_file = tmp_path / "test.gpx"
        gpx_file.write_text(gpx_content)

        from gomoveshift.analysis.gps_parser import parse_gpx_file

        with caplog.at_level(logging.WARNING):
            result = parse_gpx_file(gpx_file)

        assert len(result.trackpoints) == 2
        assert any("malformed trackpoint" in r.message.lower() for r in caplog.records)

    def test_skips_non_numeric_lat_lon(self, tmp_path, caplog):
        """Trackpoint with non-numeric lat/lon is skipped with warning."""
        gpx_content = self._make_gpx(
            '<trkpt lat="51.5" lon="-0.1"><time>2025-01-01T00:00:00Z</time></trkpt>'
            '\n<trkpt lat="abc" lon="def"><time>2025-01-01T00:01:00Z</time></trkpt>'
        )
        gpx_file = tmp_path / "test.gpx"
        gpx_file.write_text(gpx_content)

        from gomoveshift.analysis.gps_parser import parse_gpx_file

        with caplog.at_level(logging.WARNING):
            result = parse_gpx_file(gpx_file)

        assert len(result.trackpoints) == 1
        assert any("malformed trackpoint" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# stop_processor ffprobe JSON guard
# ---------------------------------------------------------------------------


class TestStopProcessorDurationProbe:
    """_get_duration must never fabricate a 0 duration — it recomputes from
    the video stream when the container has none, else raises ProbeFailedError."""

    def test_raises_on_invalid_json(self):
        """Invalid JSON from ffprobe raises ProbeFailedError, never 0.0."""
        from gomoveshift.pipeline.exceptions import ProbeFailedError
        from gomoveshift.pipeline.stop_processor import StopProcessor

        proc = StopProcessor.__new__(StopProcessor)

        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = "NOT VALID JSON"
        fake_result.stderr = ""

        with patch("subprocess.run", return_value=fake_result):
            with pytest.raises(ProbeFailedError):
                proc._get_duration(Path("/fake/video.mp4"))

    def test_raises_when_no_positive_duration(self):
        """Empty/zero duration in both format and streams raises ProbeFailedError."""
        from gomoveshift.pipeline.exceptions import ProbeFailedError
        from gomoveshift.pipeline.stop_processor import StopProcessor

        proc = StopProcessor.__new__(StopProcessor)

        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = json.dumps({"format": {}, "streams": [{"codec_type": "audio"}]})
        fake_result.stderr = ""

        with patch("subprocess.run", return_value=fake_result):
            with pytest.raises(ProbeFailedError):
                proc._get_duration(Path("/fake/video.mp4"))

    def test_recomputes_from_video_stream_when_format_missing(self):
        """Container without a format duration recomputes from the video stream."""
        from gomoveshift.pipeline.stop_processor import StopProcessor

        proc = StopProcessor.__new__(StopProcessor)

        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = json.dumps(
            {
                "format": {},
                "streams": [
                    {"codec_type": "audio", "duration": "30.0"},
                    {"codec_type": "video", "duration": "42.5"},
                ],
            }
        )
        fake_result.stderr = ""

        with patch("subprocess.run", return_value=fake_result):
            duration = proc._get_duration(Path("/fake/video.mp4"))

        assert duration == 42.5

    def test_raises_on_nonzero_returncode(self):
        """ffprobe non-zero exit raises ProbeFailedError, never a silent default."""
        from gomoveshift.pipeline.exceptions import ProbeFailedError
        from gomoveshift.pipeline.stop_processor import StopProcessor

        proc = StopProcessor.__new__(StopProcessor)

        fake_result = MagicMock()
        fake_result.returncode = 1
        fake_result.stdout = ""
        fake_result.stderr = "ffprobe: No such file"

        with patch("subprocess.run", return_value=fake_result):
            with pytest.raises(ProbeFailedError):
                proc._get_duration(Path("/fake/video.mp4"))


# ---------------------------------------------------------------------------
# upload handler bare except -> logged warning
# ---------------------------------------------------------------------------


class TestUploadHandlerLogging:
    """Bare except/pass in upload handler replaced with logged warnings."""

    def test_drive_upload_state_update_logs_on_error(self, tmp_path, caplog):
        """If pipeline_state.json is corrupt, warning is logged (not swallowed)."""
        import logging as _logging

        # Create a corrupt state file
        stages_dir = tmp_path / "_stages"
        stages_dir.mkdir()
        state_file = stages_dir / "pipeline_state.json"
        state_file.write_text("NOT VALID JSON")

        # Create shorts + the run's manifest so the upload gate passes
        # (this test asserts corrupt-state logging, not shorts QA).
        import json as _json

        shorts_dir = tmp_path / "shorts"
        shorts_dir.mkdir()
        (shorts_dir / "short_01.mp4").write_text("fake")
        (shorts_dir / "social_posts.json").write_text(
            _json.dumps([{"short_index": 1, "caption": "x"}])
        )

        # We need to mock the uploader to avoid real Google Drive calls
        mock_manifest = MagicMock()
        mock_manifest.files = [MagicMock()]

        mock_uploader_cls = MagicMock()
        mock_uploader_instance = MagicMock()
        mock_uploader_instance.upload_shorts.return_value = mock_manifest
        mock_uploader_cls.return_value = mock_uploader_instance

        mock_args = MagicMock()
        mock_args.project_dir = tmp_path
        mock_args.credentials = None
        mock_args.token = None
        mock_args.init_oauth = False
        mock_args.skip_csv = True

        # Pre-import so the module is cached, then patch attributes on it.
        import gomoveshift.pipeline.uploader as _uploader_mod  # noqa: F811
        from gomoveshift.pipeline.pipeline_validator import PipelineValidator

        # cmd_upload calls configure_logging() which invokes
        # logging.basicConfig(force=True) and rips out caplog's handler
        # before the corrupt-state warning is emitted. Stub it so caplog
        # captures the warning the test is actually asserting on.
        with (
            patch.object(_uploader_mod, "PipelineUploader", mock_uploader_cls),
            patch.object(
                PipelineValidator,
                "validate_shorts",
                lambda self, d: MagicMock(passed=True, summary=lambda: "ok"),
            ),
            patch.object(
                _uploader_mod,
                "DEFAULT_CREDENTIALS_ENC",
                MagicMock(exists=MagicMock(return_value=True)),
            ),
            patch.object(
                _uploader_mod,
                "DEFAULT_TOKEN_ENC",
                MagicMock(exists=MagicMock(return_value=True)),
            ),
            patch("gomoveshift.cli.handlers.upload.configure_logging", lambda *a, **kw: None),
            caplog.at_level(_logging.WARNING),
        ):
            from gomoveshift.cli.handlers.upload import cmd_upload

            result = cmd_upload(mock_args, {})

        assert result == 0
        # Mutation now flows through state_manager.append_stage_completed,
        # which emits "state file ... corrupt" on JSONDecodeError. Match
        # either wording so this test stays resilient to helper renames.
        assert any(
            "pipeline state" in r.message.lower()
            or ("state file" in r.message.lower() and "corrupt" in r.message.lower())
            for r in caplog.records
        )
