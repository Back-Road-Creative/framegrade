"""Tests for shorts quality validation in PipelineValidator."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline.pipeline_validator import (
    PipelineValidator,
)


@pytest.fixture
def validator():
    return PipelineValidator()


def _make_probe_data(
    width=1080,
    height=1920,
    duration=35.0,
    bit_rate=4000000,
    codec_type_video="video",
    include_audio=False,
):
    """Build ffprobe-style JSON data for a short.

    Near-silent policy: shorts ship MUTED, so a valid
    short carries NO audio stream by default. ``include_audio=True`` models the
    regression (a boosted audio stream) the muted gate must reject.
    """
    streams = [
        {
            "codec_type": codec_type_video,
            "codec_name": "hevc",
            "width": width,
            "height": height,
            # Encoding contract — values confirmed against a real
            # pipeline-produced 10-bit artifact.
            "profile": "Main 10",
            "pix_fmt": "yuv420p10le",
            "color_range": "tv",
            "color_space": "bt709",
            "color_primaries": "bt709",
            "color_transfer": "bt709",
        }
    ]
    if include_audio:
        streams.append({"codec_type": "audio", "codec_name": "aac"})
    return {
        "format": {
            "duration": str(duration),
            "bit_rate": str(bit_rate),
        },
        "streams": streams,
    }


def _ffprobe_mock(data: dict):
    """Return mock subprocess result with given probe data."""
    mock = MagicMock()
    mock.returncode = 0
    mock.stdout = json.dumps(data)
    mock.stderr = ""
    return mock


def _ffprobe_duration_mock(duration: float):
    """Return mock for integrity-check ffprobe (duration only)."""
    mock = MagicMock()
    mock.returncode = 0
    mock.stdout = json.dumps({"format": {"duration": str(duration)}})
    mock.stderr = ""
    return mock


def _create_short_file(tmp_path: Path, name: str = "short_01_rank1.mp4", size_kb: int = 600):
    """Create a dummy short file with given size."""
    p = tmp_path / name
    p.write_bytes(b"x" * (size_kb * 1024))
    return p


# ── File integrity checks ──────────────────────────────────────────────


class TestShortFileIntegrity:
    def test_missing_file_fails(self, validator, tmp_path):
        gate = validator._check_short_file_integrity(tmp_path / "nonexistent.mp4")
        assert not gate.passed

    def test_file_too_small_fails(self, validator, tmp_path):
        p = _create_short_file(tmp_path, size_kb=100)
        gate = validator._check_short_file_integrity(p)
        assert not gate.passed
        assert "500KB" in gate.message

    @patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")
    def test_valid_file_passes(self, mock_run, validator, tmp_path):
        mock_run.return_value = _ffprobe_duration_mock(35.0)
        p = _create_short_file(tmp_path, size_kb=600)
        gate = validator._check_short_file_integrity(p)
        assert gate.passed

    @patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")
    def test_corrupt_file_fails(self, mock_run, validator, tmp_path):
        bad = MagicMock()
        bad.returncode = 1
        bad.stderr = "Invalid data"
        mock_run.return_value = bad
        p = _create_short_file(tmp_path, size_kb=600)
        gate = validator._check_short_file_integrity(p)
        assert not gate.passed


# ── Resolution checks ──────────────────────────────────────────────────


class TestShortResolution:
    def test_correct_resolution_passes(self, validator, tmp_path):
        p = _create_short_file(tmp_path)
        data = _make_probe_data(width=1080, height=1920)
        gate = validator._check_short_resolution(p, data)
        assert gate.passed

    def test_wrong_resolution_fails(self, validator, tmp_path):
        p = _create_short_file(tmp_path)
        data = _make_probe_data(width=1920, height=1080)
        gate = validator._check_short_resolution(p, data)
        assert not gate.passed
        assert "1920x1080" in gate.message

    def test_no_video_stream_fails(self, validator, tmp_path):
        p = _create_short_file(tmp_path)
        data = {"format": {}, "streams": [{"codec_type": "audio"}]}
        gate = validator._check_short_resolution(p, data)
        assert not gate.passed


# ── Duration checks ────────────────────────────────────────────────────


class TestShortDuration:
    def test_valid_duration_passes(self, validator, tmp_path):
        p = _create_short_file(tmp_path)
        data = _make_probe_data(duration=35.0)
        gate = validator._check_short_duration(p, data)
        assert gate.passed

    def test_too_short_fails(self, validator, tmp_path):
        p = _create_short_file(tmp_path)
        data = _make_probe_data(duration=10.0)
        gate = validator._check_short_duration(p, data)
        assert not gate.passed

    def test_too_long_fails(self, validator, tmp_path):
        p = _create_short_file(tmp_path)
        data = _make_probe_data(duration=65.0)
        gate = validator._check_short_duration(p, data)
        assert not gate.passed

    def test_boundary_15s_passes(self, validator, tmp_path):
        p = _create_short_file(tmp_path)
        data = _make_probe_data(duration=15.0)
        gate = validator._check_short_duration(p, data)
        assert gate.passed

    def test_boundary_60s_passes(self, validator, tmp_path):
        p = _create_short_file(tmp_path)
        data = _make_probe_data(duration=60.0)
        gate = validator._check_short_duration(p, data)
        assert gate.passed


# ── Bitrate checks (WARNING only) ─────────────────────────────────────


class TestShortBitrate:
    def test_good_bitrate_passes(self, validator, tmp_path):
        p = _create_short_file(tmp_path)
        data = _make_probe_data(bit_rate=4000000)  # 4000 kbps
        gate = validator._check_short_bitrate(p, data)
        assert gate.passed

    def test_low_bitrate_fails(self, validator, tmp_path):
        p = _create_short_file(tmp_path)
        data = _make_probe_data(bit_rate=1500000)  # 1500 kbps
        gate = validator._check_short_bitrate(p, data)
        assert not gate.passed
        assert "warning" in gate.message

    def test_zero_bitrate_fails_closed(self, validator, tmp_path):
        """An unreadable bitrate must FAIL, never report a clean pass.

        A short whose container carries no bitrate is a short whose encode was
        never confirmed — the old ``return passed=True`` shipped it as verified.
        """
        p = _create_short_file(tmp_path)
        data = _make_probe_data(bit_rate=0)
        gate = validator._check_short_bitrate(p, data)
        assert not gate.passed, "unreadable bitrate must fail closed"
        assert "unreadable" in gate.message

    def test_unreadable_bitrate_is_fatal_not_advisory(self, validator, tmp_path, monkeypatch):
        """Below-floor bitrate stays advisory; UNREADABLE bitrate is fatal.

        ``validate_short_files`` demotes the ``short_bitrate:`` floor check to a
        warning (a 1500 kbps short still plays). An unmeasurable encode is a
        different class — it must sink the run like every other failed check.
        """
        p = _create_short_file(tmp_path)
        shorts = validator._shorts
        monkeypatch.setattr(
            shorts,
            "_validate_single_short",
            lambda path: [shorts._check_short_bitrate(path, _make_probe_data(bit_rate=1500000))],
        )
        assert shorts.validate_short_files([p]).passed, "below-floor bitrate stays advisory"

        monkeypatch.setattr(
            shorts,
            "_validate_single_short",
            lambda path: [shorts._check_short_bitrate(path, _make_probe_data(bit_rate=0))],
        )
        assert not shorts.validate_short_files([p]).passed, (
            "an unreadable bitrate must be fatal, not filtered out as advisory"
        )


# ── Encoding-contract checks (codec / profile / pix_fmt / colour) ─────


class TestShortEncodingContract:
    """A regressed 8-bit short passed EVERY shorts gate before the encoding-contract gate existed.

    ``ShortsValidator`` had zero pix_fmt / profile / codec / colour-flag
    checks, so a banded-skies regression (p5 / CQ 23 / 8-bit
    yuv420p) would have shipped clean. The expected values are the frozen
    ``encoding_constants``; the reported forms (``Main 10`` / ``yuv420p10le``
    for a ``-profile:v main10 -pix_fmt p010le`` NVENC encode) were confirmed
    against a real pipeline-produced short.
    """

    def _gate(self, validator, tmp_path, **overrides):
        p = _create_short_file(tmp_path)
        data = _make_probe_data()
        video = data["streams"][0]
        for key, value in overrides.items():
            if value is None:
                video.pop(key, None)
            else:
                video[key] = value
        return validator._shorts._check_short_encoding_contract(p, data)

    def test_in_spec_short_passes(self, validator, tmp_path):
        gate = self._gate(validator, tmp_path)
        assert gate.passed, gate.message

    def test_8bit_short_is_rejected(self, validator, tmp_path):
        """The regressed 8-bit shape: Main / yuv420p."""
        gate = self._gate(validator, tmp_path, profile="Main", pix_fmt="yuv420p")
        assert not gate.passed
        assert "pix_fmt" in gate.message
        assert "profile" in gate.message

    def test_wrong_codec_is_rejected(self, validator, tmp_path):
        gate = self._gate(validator, tmp_path, codec_name="h264")
        assert not gate.passed
        assert "codec" in gate.message

    def test_full_range_short_is_rejected(self, validator, tmp_path):
        gate = self._gate(validator, tmp_path, color_range="pc")
        assert not gate.passed
        assert "color_range" in gate.message

    def test_untagged_short_fails_closed(self, validator, tmp_path):
        gate = self._gate(validator, tmp_path, color_primaries=None)
        assert not gate.passed
        assert "unreadable" in gate.message

    def test_contract_failure_is_fatal_for_the_day(self, validator, tmp_path):
        """An off-contract short must sink the run, not warn."""
        p = _create_short_file(tmp_path)
        shorts = validator._shorts
        bad = _make_probe_data()
        bad["streams"][0]["pix_fmt"] = "yuv420p"
        shorts._validate_single_short = lambda path: [
            shorts._check_short_encoding_contract(path, bad)
        ]
        assert not shorts.validate_short_files([p]).passed


# ── Audio muted-by-design checks ──────────────────────────────────────


class TestShortAudioMuted:
    """Near-silent policy: shorts ship muted (`-an`), so
    a shipped short MUST carry no audio stream. Missing audio = PASS, present
    audio = FAIL. The gate probes stream metadata (no loudnorm / LUFS)."""

    @patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")
    def test_muted_short_passes(self, mock_run, validator, tmp_path):
        mock_run.return_value = _ffprobe_mock(_make_probe_data(include_audio=False))
        p = _create_short_file(tmp_path)
        gate = validator._check_short_audio_lufs(p)
        assert gate.passed, gate.message
        assert gate.gate == f"short_audio_muted:{p.name}"

    @patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")
    def test_audio_stream_present_fails(self, mock_run, validator, tmp_path):
        mock_run.return_value = _ffprobe_mock(_make_probe_data(include_audio=True))
        p = _create_short_file(tmp_path)
        gate = validator._check_short_audio_lufs(p)
        assert not gate.passed
        assert gate.gate == f"short_audio_muted:{p.name}"
        assert "muted" in gate.message.lower()

    @patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")
    def test_multiple_audio_streams_fail(self, mock_run, validator, tmp_path):
        data = _make_probe_data(include_audio=True)
        data["streams"].append({"codec_type": "audio", "codec_name": "aac"})
        mock_run.return_value = _ffprobe_mock(data)
        p = _create_short_file(tmp_path)
        gate = validator._check_short_audio_lufs(p)
        assert not gate.passed


# ── Full validate_shorts integration ───────────────────────────────────


class TestValidateShorts:
    def test_empty_dir_passes(self, validator, tmp_path):
        result = validator.validate_shorts(tmp_path)
        assert result.passed
        assert "skipped" in result.gates[0].message

    @patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")
    def test_valid_short_passes(self, mock_run, validator, tmp_path):
        """A fully valid short should pass all gates."""
        probe_data = _make_probe_data()

        def side_effect(cmd, **kwargs):
            if cmd[0] == "ffprobe":
                return _ffprobe_mock(probe_data)
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = side_effect

        _create_short_file(tmp_path, "short_01_rank1.mp4", size_kb=600)
        result = validator.validate_shorts(tmp_path)
        assert result.passed

    @patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")
    def test_wrong_resolution_is_fatal(self, mock_run, validator, tmp_path):
        """Wrong resolution should make the overall result fail."""
        probe_data = _make_probe_data(width=1920, height=1080)

        def side_effect(cmd, **kwargs):
            if cmd[0] == "ffprobe":
                return _ffprobe_mock(probe_data)
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = side_effect

        _create_short_file(tmp_path, "short_01_rank1.mp4", size_kb=600)
        result = validator.validate_shorts(tmp_path)
        assert not result.passed

    @patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")
    def test_low_bitrate_is_warning_only(self, mock_run, validator, tmp_path):
        """Low bitrate should NOT make the overall result fail (warning only)."""
        probe_data = _make_probe_data(bit_rate=1500000)  # 1500 kbps

        def side_effect(cmd, **kwargs):
            if cmd[0] == "ffprobe":
                return _ffprobe_mock(probe_data)
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = side_effect

        _create_short_file(tmp_path, "short_01_rank1.mp4", size_kb=600)
        result = validator.validate_shorts(tmp_path)
        # Should pass overall — bitrate is warning-only
        assert result.passed
        # But the bitrate gate itself should have failed
        bitrate_gates = [g for g in result.gates if "bitrate" in g.gate]
        assert any(not g.passed for g in bitrate_gates)

    @patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")
    def test_multiple_shorts_all_checked(self, mock_run, validator, tmp_path):
        """All short files in the directory should be validated."""
        probe_data = _make_probe_data()

        def side_effect(cmd, **kwargs):
            if cmd[0] == "ffprobe":
                return _ffprobe_mock(probe_data)
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = side_effect

        _create_short_file(tmp_path, "short_01_rank1.mp4", size_kb=600)
        _create_short_file(tmp_path, "short_02_rank2.mp4", size_kb=600)
        _create_short_file(tmp_path, "short_03_rank3.mp4", size_kb=600)

        result = validator.validate_shorts(tmp_path)
        assert result.passed
        # Each short gets 6 gates: integrity, resolution, duration, bitrate,
        # encoding_contract, audio_muted.
        assert len(result.gates) == 18
