"""Tests for WeatherDetector CUDA hwaccel + tunable timeout cap.

Root-cause fix: weather ffmpeg now matches the rest of
the pipeline (CUDA decode where available). Software decode of 4K HEVC
signalstats can't keep up with realtime, breaching the per-chunk timeout
cap regardless of source filesystem.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline import weather as weather_mod
from gomoveshift.pipeline.exceptions import WeatherMeasurementError
from gomoveshift.pipeline.weather import WeatherDetector


@pytest.fixture(autouse=True)
def _reset_capability_cache():
    """Each test starts with a clean ffmpeg capability cache so the
    fixture sets one test does not bleed into another."""
    weather_mod.reset_ffmpeg_capability_cache()
    yield
    weather_mod.reset_ffmpeg_capability_cache()


def _seed_capability(hwaccels=("cuda",), decoders=("hevc_cuvid", "h264_cuvid", "av1_cuvid")):
    """Seed the process-level ffmpeg capability cache so callers can
    exercise the cuda/auto code paths without shelling real ffmpeg."""
    weather_mod._FFMPEG_CAPABILITY_CACHE["hwaccels"] = set(hwaccels)
    weather_mod._FFMPEG_CAPABILITY_CACHE["decoders"] = set(decoders)
    weather_mod._FFMPEG_CAPABILITY_CACHE["probed"] = True


class TestWeatherDetectorHwaccel:
    def test_init_accepts_new_knobs(self):
        d = WeatherDetector(
            chunk_max_sec=180.0,
            chunk_timeout_cap_sec=240.0,
            hwaccel="cuda",
        )
        assert d.chunk_max_sec == 180.0
        assert d.chunk_timeout_cap_sec == 240.0
        assert d.hwaccel == "cuda"

    def test_init_defaults(self):
        d = WeatherDetector()
        assert d.chunk_max_sec == 120.0
        assert d.chunk_timeout_cap_sec == 180.0
        assert d.hwaccel == "auto"

    def test_init_rejects_nonpositive_cap(self):
        with pytest.raises(ValueError, match="chunk_timeout_cap_sec"):
            WeatherDetector(chunk_timeout_cap_sec=0)

    def test_cuvid_decoder_map(self):
        """The cuvid decoder map covers the codecs the pipeline ingests."""
        assert WeatherDetector._CUVID_BY_CODEC["hevc"] == "hevc_cuvid"
        assert WeatherDetector._CUVID_BY_CODEC["h264"] == "h264_cuvid"
        assert WeatherDetector._CUVID_BY_CODEC["av1"] == "av1_cuvid"

    def test_cuvid_decoder_unknown_codec_returns_none(self):
        d = WeatherDetector(hwaccel="cuda")
        fake = MagicMock(stdout="prores\n", returncode=0)
        with patch("subprocess.run", return_value=fake):
            assert d._cuvid_decoder_for(Path("/fake.mov")) is None

    def test_cuvid_decoder_ffprobe_missing_returns_none(self):
        d = WeatherDetector(hwaccel="cuda")
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert d._cuvid_decoder_for(Path("/fake.mp4")) is None

    def test_cuvid_decoder_matches_hevc(self):
        d = WeatherDetector(hwaccel="cuda")
        fake = MagicMock(stdout="hevc\n", returncode=0)
        with patch("subprocess.run", return_value=fake):
            assert d._cuvid_decoder_for(Path("/fake.mp4")) == "hevc_cuvid"

    def test_chunk_timeout_cap_uses_configured_value(self):
        """The per-chunk timeout uses self.chunk_timeout_cap_sec, not the
        old hardcoded 90s ceiling."""
        d = WeatherDetector(chunk_timeout_cap_sec=240.0, hwaccel="none")
        # A 120s chunk wants 2*120=240s; cap=240 → exactly 240.
        # Spy on _measure_chunk via subprocess.run patching.
        call_args = []

        def fake_run(cmd, **kwargs):
            call_args.append(kwargs.get("timeout"))
            return MagicMock(
                stderr="[Parsed_metadata_2 @ 0x0] lavfi.signalstats.YAVG=128.0\n", returncode=0
            )

        with patch("subprocess.run", side_effect=fake_run):
            d._get_avg_brightness_ffmpeg(Path("/fake.mp4"), 0.0, 120.0)
        assert call_args, "expected subprocess.run to be called"
        # First (and only) chunk gets timeout = min(cap=240, 2*120=240) = 240.
        assert call_args[0] == 240.0

    def test_hwaccel_none_omits_flag(self):
        """When hwaccel='none' the ffmpeg command must not include
        -hwaccel — covers operators who explicitly disable CUDA."""
        d = WeatherDetector(hwaccel="none")
        captured = []

        def fake_run(cmd, **kwargs):
            captured.append(cmd)
            return MagicMock(
                stderr="[Parsed_metadata_2 @ 0x0] lavfi.signalstats.YAVG=128.0\n", returncode=0
            )

        with patch("subprocess.run", side_effect=fake_run):
            d._measure_chunk(Path("/fake.mp4"), 0.0, 30.0, 60.0)
        assert captured
        assert "-hwaccel" not in captured[0]

    def test_hwaccel_cuda_adds_decoder_when_hevc(self):
        """hwaccel='cuda' on an HEVC source adds both -hwaccel cuda and
        -c:v hevc_cuvid — the proven fast-decode pattern matching the
        rest of the pipeline."""
        _seed_capability()
        d = WeatherDetector(hwaccel="cuda")
        probe = MagicMock(stdout="hevc\n", returncode=0)
        chunk = MagicMock(
            stderr="[Parsed_metadata_2 @ 0x0] lavfi.signalstats.YAVG=128.0\n", returncode=0
        )
        calls = {"cmds": []}

        def fake_run(cmd, **kwargs):
            if cmd[0] == "ffprobe":
                return probe
            calls["cmds"].append(cmd)
            return chunk

        with patch("subprocess.run", side_effect=fake_run):
            d._measure_chunk(Path("/fake.mp4"), 0.0, 30.0, 60.0)
        assert calls["cmds"], "expected ffmpeg invocation"
        cmd = calls["cmds"][0]
        assert "-hwaccel" in cmd
        assert "cuda" in cmd
        assert "-c:v" in cmd
        assert "hevc_cuvid" in cmd


class TestWeatherDetectorAutoMode:
    """Root-cause fix: hwaccel='auto' (the default) must
    emit the cuvid decoder for cuvid-mappable codecs on a cuda-capable
    ffmpeg. The pre-fix bug emitted bare ``-hwaccel auto`` without
    ``-c:v <codec>_cuvid``, leaving signalstats on software HEVC decode
    that times out on 4K Insta360 sources."""

    def test_auto_mode_emits_cuvid_when_codec_and_ffmpeg_support(self):
        _seed_capability()
        d = WeatherDetector(hwaccel="auto")
        probe = MagicMock(stdout="hevc\n", returncode=0)
        chunk = MagicMock(
            stderr="[Parsed_metadata_2 @ 0x0] lavfi.signalstats.YAVG=128.0\n", returncode=0
        )
        ffmpeg_cmds: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            if cmd[0] == "ffprobe":
                return probe
            ffmpeg_cmds.append(cmd)
            return chunk

        with patch("subprocess.run", side_effect=fake_run):
            d._measure_chunk(Path("/fake/v.mp4"), 0.0, 30.0, 60.0)
        assert ffmpeg_cmds
        cmd = ffmpeg_cmds[0]
        assert cmd.count("-hwaccel") == 1
        i = cmd.index("-hwaccel")
        assert cmd[i + 1] == "cuda"
        assert "-c:v" in cmd
        assert "hevc_cuvid" in cmd

    def test_auto_mode_software_fallback_when_codec_unmapped(self):
        _seed_capability()
        d = WeatherDetector(hwaccel="auto")
        probe = MagicMock(stdout="prores\n", returncode=0)
        chunk = MagicMock(
            stderr="[Parsed_metadata_2 @ 0x0] lavfi.signalstats.YAVG=128.0\n", returncode=0
        )
        ffmpeg_cmds: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            if cmd[0] == "ffprobe":
                return probe
            ffmpeg_cmds.append(cmd)
            return chunk

        with patch("subprocess.run", side_effect=fake_run):
            d._measure_chunk(Path("/fake/v.mov"), 0.0, 30.0, 60.0)
        assert ffmpeg_cmds
        assert "-hwaccel" not in ffmpeg_cmds[0]

    def test_auto_mode_software_fallback_when_ffmpeg_lacks_cuda(self):
        _seed_capability(hwaccels=(), decoders=())
        d = WeatherDetector(hwaccel="auto")
        probe = MagicMock(stdout="hevc\n", returncode=0)
        chunk = MagicMock(
            stderr="[Parsed_metadata_2 @ 0x0] lavfi.signalstats.YAVG=128.0\n", returncode=0
        )
        ffmpeg_cmds: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            if cmd[0] == "ffprobe":
                return probe
            ffmpeg_cmds.append(cmd)
            return chunk

        with patch("subprocess.run", side_effect=fake_run):
            d._measure_chunk(Path("/fake/v.mp4"), 0.0, 30.0, 60.0)
        assert ffmpeg_cmds
        assert "-hwaccel" not in ffmpeg_cmds[0]

    def test_codec_probe_cached_per_path(self):
        """The ffprobe codec probe runs at most once per source file —
        cached on the detector instance so chunked aggregation does not
        re-probe per chunk."""
        _seed_capability()
        d = WeatherDetector(hwaccel="auto", chunk_max_sec=50.0)
        probe = MagicMock(stdout="hevc\n", returncode=0)
        chunk = MagicMock(stderr="[Parsed_signalstats_1] YAVG:128.0\n", returncode=0)
        ffprobe_calls = {"n": 0}

        def fake_run(cmd, **kwargs):
            if cmd[0] == "ffprobe":
                ffprobe_calls["n"] += 1
                return probe
            return chunk

        with patch("subprocess.run", side_effect=fake_run):
            d._get_avg_brightness_ffmpeg(Path("/fake/v.mp4"), 0.0, 200.0)
        assert ffprobe_calls["n"] == 1


class TestWeatherDetectorCudaStrict:
    """hwaccel='cuda' is the explicit-opt-in path. Must raise
    WeatherMeasurementError on any failure to wire up GPU decode — never
    silent software fallback."""

    def test_cuda_raises_when_codec_unmappable(self):
        _seed_capability()
        d = WeatherDetector(hwaccel="cuda")
        probe = MagicMock(stdout="prores\n", returncode=0)

        def fake_run(cmd, **kwargs):
            if cmd[0] == "ffprobe":
                return probe
            return MagicMock(
                stderr="[Parsed_metadata_2 @ 0x0] lavfi.signalstats.YAVG=128.0\n", returncode=0
            )

        with patch("subprocess.run", side_effect=fake_run):
            with pytest.raises(WeatherMeasurementError, match="cuvid decoder"):
                d._measure_chunk(Path("/fake/v.mov"), 0.0, 30.0, 60.0)

    def test_cuda_raises_when_ffmpeg_lacks_cuda(self):
        _seed_capability(hwaccels=(), decoders=())
        d = WeatherDetector(hwaccel="cuda")
        probe = MagicMock(stdout="hevc\n", returncode=0)

        def fake_run(cmd, **kwargs):
            if cmd[0] == "ffprobe":
                return probe
            return MagicMock(
                stderr="[Parsed_metadata_2 @ 0x0] lavfi.signalstats.YAVG=128.0\n", returncode=0
            )

        with patch("subprocess.run", side_effect=fake_run):
            with pytest.raises(WeatherMeasurementError, match="hwaccel='cuda'"):
                d._measure_chunk(Path("/fake/v.mp4"), 0.0, 30.0, 60.0)


class TestFfmpegCapabilityProbe:
    def test_probe_parses_cuda_and_cuvid(self):
        hwaccel_stdout = "Hardware acceleration methods:\nvdpau\ncuda\nvaapi\n"
        decoders_stdout = (
            " V..... hevc_cuvid          Nvidia CUVID HEVC decoder (codec hevc)\n"
            " V..... h264_cuvid          Nvidia CUVID H.264 decoder (codec h264)\n"
            " V..... mjpeg               Motion JPEG\n"
        )

        def fake_run(cmd, **kwargs):
            if "-hwaccels" in cmd:
                return MagicMock(stdout=hwaccel_stdout, returncode=0)
            if "-decoders" in cmd:
                return MagicMock(stdout=decoders_stdout, returncode=0)
            raise AssertionError(f"unexpected cmd {cmd}")

        with patch("subprocess.run", side_effect=fake_run):
            assert weather_mod.ffmpeg_supports_cuda_decode("hevc_cuvid") is True
            # second call hits the cache (no further subprocess.run)
            assert weather_mod.ffmpeg_supports_cuda_decode("h264_cuvid") is True
            assert weather_mod.ffmpeg_supports_cuda_decode("av1_cuvid") is False

    def test_probe_returns_false_when_ffmpeg_missing(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert weather_mod.ffmpeg_supports_cuda_decode("hevc_cuvid") is False
