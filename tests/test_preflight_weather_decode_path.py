"""Preflight gate: verify weather_hwaccel can deliver GPU decode for
cuvid-mappable sources before Phase A burns wall-clock.

Root-cause hardening. The pre-fix failure
mode was: weather_hwaccel='auto' default + 4K HEVC Insta360 source →
bare `-hwaccel auto` ffmpeg (no -c:v hevc_cuvid) → software signalstats
below realtime → every chunk hits the timeout cap → degraded
measurement → (pre-fix) silent overcast baked into drive_scores.npz.

This gate makes that configuration unreachable: preflight raises with
explicit remediation before Phase A starts.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline import weather as weather_mod
from gomoveshift.pipeline.preflight import check_weather_decode_path


@pytest.fixture(autouse=True)
def _reset_capability_cache():
    weather_mod.reset_ffmpeg_capability_cache()
    yield
    weather_mod.reset_ffmpeg_capability_cache()


def _seed(hwaccels=("cuda",), decoders=("hevc_cuvid", "h264_cuvid", "av1_cuvid")):
    weather_mod._FFMPEG_CAPABILITY_CACHE["hwaccels"] = set(hwaccels)
    weather_mod._FFMPEG_CAPABILITY_CACHE["decoders"] = set(decoders)
    weather_mod._FFMPEG_CAPABILITY_CACHE["probed"] = True


def _patch_codec(codec: str):
    """Patch every subprocess.run ffprobe to return the given codec."""

    def fake_run(cmd, **kwargs):
        if cmd and cmd[0] == "ffprobe":
            return MagicMock(stdout=f"{codec}\n", returncode=0)
        return MagicMock(stdout="", stderr="", returncode=0)

    return patch("subprocess.run", side_effect=fake_run)


class TestWeatherDecodePathGate:
    def test_none_short_circuits(self):
        """weather_hwaccel='none' skips the gate — operator explicitly
        opted into software decode."""
        # No seeding, no patching — function must return without touching anything.
        check_weather_decode_path([Path("/fake.mp4")], "none")

    def test_empty_string_short_circuits(self):
        check_weather_decode_path([Path("/fake.mp4")], "")

    def test_no_sources_no_check(self):
        """Empty source list — preflight has already rejected this case
        elsewhere; the gate just no-ops."""
        check_weather_decode_path([], "auto")

    def test_auto_passes_when_ffmpeg_supports_cuvid(self):
        _seed()
        with _patch_codec("hevc"):
            check_weather_decode_path([Path("/fake.mp4")], "auto")

    def test_auto_passes_when_codec_unmappable(self):
        """Prores source + auto mode: cuvid unavailable for this codec
        is fine — auto falls through to software, no GPU promise to keep."""
        _seed()
        with _patch_codec("prores"):
            check_weather_decode_path([Path("/fake.mov")], "auto")

    def test_auto_raises_when_ffmpeg_lacks_cuvid_for_mappable_source(self):
        """The real-world case: HEVC source + auto mode + ffmpeg missing
        cuvid. Pre-fix silently fell through to software and timed out;
        post-fix preflight raises before Phase A."""
        _seed(hwaccels=(), decoders=())
        with _patch_codec("hevc"):
            with pytest.raises(RuntimeError, match="CUDA decode support"):
                check_weather_decode_path([Path("/fake.mp4")], "auto")

    def test_auto_raises_when_ffmpeg_has_cuda_but_missing_cuvid_decoder(self):
        """Edge case: cuda hwaccel present but the specific cuvid
        decoder for this codec is missing from the ffmpeg build."""
        _seed(hwaccels=("cuda",), decoders=("h264_cuvid",))  # no hevc_cuvid
        with _patch_codec("hevc"):
            with pytest.raises(RuntimeError, match="hevc_cuvid"):
                check_weather_decode_path([Path("/fake.mp4")], "auto")

    def test_cuda_raises_when_no_source_is_cuvid_mappable(self):
        """Operator asked for cuda but every source is prores. Surface
        here, not after stage init."""
        _seed()
        with _patch_codec("prores"):
            with pytest.raises(RuntimeError, match="cuvid-mappable"):
                check_weather_decode_path([Path("/fake.mov")], "cuda")

    def test_cuda_raises_when_ffmpeg_lacks_cuvid(self):
        _seed(hwaccels=(), decoders=())
        with _patch_codec("hevc"):
            with pytest.raises(RuntimeError, match="CUDA decode support"):
                check_weather_decode_path([Path("/fake.mp4")], "cuda")

    def test_error_message_names_remediation_options(self):
        """The error must point the operator at both fixes (rebuild
        ffmpeg or set weather_hwaccel='none')."""
        _seed(hwaccels=(), decoders=())
        with _patch_codec("hevc"):
            with pytest.raises(RuntimeError) as exc:
                check_weather_decode_path([Path("/fake.mp4")], "auto")
        msg = str(exc.value)
        assert "--enable-cuda-llvm" in msg or "cuda-llvm" in msg
        assert "weather_hwaccel='none'" in msg or "weather-hwaccel none" in msg
