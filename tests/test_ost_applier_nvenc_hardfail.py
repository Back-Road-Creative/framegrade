"""Regression tests for NVENC hard-fail.

Background: apply_ost silently fell back to libx265 when NVENC sessions
had not released yet from a just-finished master encode. A silent CPU
fallback turns a GPU encode into a many-hour libx265 encode at lower
quality.

The libx265 fallback path is deleted. detect_nvenc() raises
NVENCUnavailableError unconditionally on probe failure. The OST applier
keeps a 5s warm-up retry on the first failure when prefer_nvenc=True,
but the second failure propagates — there is no env-var or CLI override
that re-enables CPU encode.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline import ost_applier
from gomoveshift.pipeline.color_filters import NVENCUnavailableError

from ost_subprocess_harness import patch_ost_run


def _stub_probe():
    return {
        "width": 3840,
        "height": 2160,
        "fps": 29.97,
        "bitrate_kbps": 52000,
        "duration": 817.0,
    }


def _make_input(tmp_path):
    p = tmp_path / "in.mp4"
    p.write_bytes(b"\x00" * 1024)
    return p


@pytest.fixture(autouse=True)
def _accept_burned_artifact(monkeypatch):
    """apply_ost ffprobe-validates the artifact it burned; ffmpeg is mocked here.

    Post-encode validation itself is covered by tests/test_ost_output_validation.py.
    """
    from gomoveshift.pipeline.ffmpeg_runner import VideoValidation

    monkeypatch.setattr(
        "gomoveshift.pipeline.ost_applier.validate_video",
        lambda *_a, **_kw: VideoValidation(valid=True),
    )


class TestPreferNvencRetry:
    """When prefer_nvenc=True and the first detect_nvenc probe raises
    NVENCUnavailableError, retry once after a short sleep; a second failure
    propagates."""

    def test_retry_once_when_first_probe_raises(self, tmp_path, caplog):
        from gomoveshift.pipeline.ost_applier import _resolve_encoding_args

        call_count = {"n": 0}

        def flaky_detect():
            call_count["n"] += 1
            # First call: NVENC "busy" — raises (detect_nvenc raises on
            # probe failure; it never returns False). Second call: True.
            if call_count["n"] == 1:
                raise NVENCUnavailableError("simulated NVENC busy")
            return True

        info = _stub_probe()

        def fake_get_args(**kw):
            return ["-c:v", "hevc_nvenc" if kw["has_nvenc"] else "libx265"]

        with (
            patch("gomoveshift.pipeline.color.detect_nvenc", side_effect=flaky_detect),
            patch("gomoveshift.pipeline.color.get_encoding_args", side_effect=fake_get_args),
            patch("gomoveshift.pipeline.ost_applier.time.sleep") as mock_sleep,
            caplog.at_level("WARNING", logger="gomoveshift.pipeline.ost_applier"),
        ):
            args = _resolve_encoding_args(info, prefer_nvenc=True, input_name="lf_02.mp4")

        assert call_count["n"] == 2, "expected one retry after the initial failure"
        mock_sleep.assert_called_once_with(5)
        assert "hevc_nvenc" in args, f"expected NVENC after retry, got {args}"
        assert "libx265" not in args
        # Retry should log a warning so operators see the NVENC flap
        assert any("NVENC" in r.message for r in caplog.records)

    def test_no_retry_when_first_probe_true(self, tmp_path):
        from gomoveshift.pipeline.ost_applier import _resolve_encoding_args

        call_count = {"n": 0}

        def always_nvenc():
            call_count["n"] += 1
            return True

        def fake_get_args(**kw):
            return ["-c:v", "hevc_nvenc" if kw["has_nvenc"] else "libx265"]

        with (
            patch("gomoveshift.pipeline.color.detect_nvenc", side_effect=always_nvenc),
            patch("gomoveshift.pipeline.color.get_encoding_args", side_effect=fake_get_args),
            patch("gomoveshift.pipeline.ost_applier.time.sleep") as mock_sleep,
        ):
            args = _resolve_encoding_args(_stub_probe(), prefer_nvenc=True, input_name="lf_02.mp4")

        assert call_count["n"] == 1, "no retry needed when first probe True"
        mock_sleep.assert_not_called()
        assert "hevc_nvenc" in args

    def test_retry_failure_propagates_no_fallback(self, tmp_path):
        """Both probes raise → error propagates, no libx265 path."""
        from gomoveshift.pipeline.ost_applier import _resolve_encoding_args

        def always_busy():
            raise NVENCUnavailableError("simulated NVENC busy")

        def fake_get_args(**kw):
            return ["-c:v", "hevc_nvenc" if kw["has_nvenc"] else "libx265"]

        with (
            patch("gomoveshift.pipeline.color.detect_nvenc", side_effect=always_busy),
            patch("gomoveshift.pipeline.color.get_encoding_args", side_effect=fake_get_args),
            patch("gomoveshift.pipeline.ost_applier.time.sleep"),
        ):
            with pytest.raises(NVENCUnavailableError):
                _resolve_encoding_args(_stub_probe(), prefer_nvenc=True, input_name="lf_02.mp4")

    def test_prefer_false_means_no_retry(self, tmp_path):
        from gomoveshift.pipeline.ost_applier import _resolve_encoding_args

        call_count = {"n": 0}

        def flaky_detect():
            call_count["n"] += 1
            raise NVENCUnavailableError("simulated NVENC busy")

        def fake_get_args(**kw):
            return ["-c:v", "hevc_nvenc" if kw["has_nvenc"] else "libx265"]

        with (
            patch("gomoveshift.pipeline.color.detect_nvenc", side_effect=flaky_detect),
            patch("gomoveshift.pipeline.color.get_encoding_args", side_effect=fake_get_args),
            patch("gomoveshift.pipeline.ost_applier.time.sleep") as mock_sleep,
        ):
            # prefer_nvenc=False — the resolver must re-raise the first
            # NVENCUnavailableError without retrying.
            with pytest.raises(NVENCUnavailableError):
                _resolve_encoding_args(_stub_probe(), prefer_nvenc=False, input_name="short_01.mp4")

        # No retry because caller didn't opt in
        assert call_count["n"] == 1
        mock_sleep.assert_not_called()


class TestOstEncodeTimeout:
    """The subprocess.run timeout must be configurable via env var, and now
    scales with the probed input duration.

    A fixed 3600s timeout is too short for a long or slow encode, so the
    timeout scales with input duration and can be overridden by env var.
    """

    def test_default_timeout_is_7200_for_a_typical_lf(self, monkeypatch):
        """817s (~14min) is today's typical LF — must keep the same budget."""
        monkeypatch.delenv("GMS_OST_ENCODE_TIMEOUT_SEC", raising=False)
        assert ost_applier._ost_encode_timeout_sec(817.0) == 7200

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("GMS_OST_ENCODE_TIMEOUT_SEC", "900")
        assert ost_applier._ost_encode_timeout_sec(817.0) == 900

    def test_env_override_non_numeric_ignored(self, monkeypatch):
        monkeypatch.setenv("GMS_OST_ENCODE_TIMEOUT_SEC", "forever")
        assert ost_applier._ost_encode_timeout_sec(817.0) == 7200

    def test_short_duration_keeps_the_7200_floor(self, monkeypatch):
        """A short input must never get a TIGHTER timeout than before."""
        monkeypatch.delenv("GMS_OST_ENCODE_TIMEOUT_SEC", raising=False)
        assert ost_applier._ost_encode_timeout_sec(60.0) == 7200

    def test_long_duration_scales_above_the_floor(self, monkeypatch):
        """A 4h superlong part (the new planner cap) must scale above 7200s."""
        monkeypatch.delenv("GMS_OST_ENCODE_TIMEOUT_SEC", raising=False)
        duration = 14400.0
        timeout = ost_applier._ost_encode_timeout_sec(duration)
        assert timeout > 7200
        faststart_buffer = max(
            ost_applier._OST_FASTSTART_BUFFER_FLOOR_SEC,
            int(duration * ost_applier._OST_FASTSTART_BUFFER_FRAC),
        )
        expected = int(duration * ost_applier._OST_ENCODE_TIMEOUT_MULTIPLIER) + faststart_buffer
        assert timeout == expected

    def test_env_override_wins_regardless_of_duration(self, monkeypatch):
        monkeypatch.setenv("GMS_OST_ENCODE_TIMEOUT_SEC", "500")
        assert ost_applier._ost_encode_timeout_sec(14400.0) == 500

    def test_apply_ost_passes_timeout_to_subprocess(self, tmp_path, monkeypatch):
        """The timeout is read from the env at encode time, not module import."""
        monkeypatch.setenv("GMS_OST_ENCODE_TIMEOUT_SEC", "123")
        in_mp4 = _make_input(tmp_path)
        out_mp4 = tmp_path / "out.mp4"

        with (
            patch("gomoveshift.pipeline.ost_applier._probe_video_info", return_value=_stub_probe()),
            patch("gomoveshift.pipeline.ost_applier.Path.exists", return_value=True),
            patch(
                "gomoveshift.pipeline.ost_applier._resolve_encoding_args",
                return_value=["-c:v", "hevc_nvenc"],
            ),
            patch_ost_run() as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stderr=b"")
            ost_applier.apply_ost(
                input_path=in_mp4,
                output_path=out_mp4,
                location="Test",
                layout="regular_lf",
            )
        # subprocess.run was called with timeout=123
        _, kwargs = mock_run.call_args
        assert kwargs.get("timeout") == 123


class TestApplyOstPreferNvencKwarg:
    def test_apply_ost_accepts_prefer_nvenc_kwarg(self):
        import inspect

        sig = inspect.signature(ost_applier.apply_ost)
        assert "prefer_nvenc" in sig.parameters
        assert sig.parameters["prefer_nvenc"].default is True

    def test_prefer_nvenc_threads_to_resolver(self, tmp_path):
        in_mp4 = _make_input(tmp_path)
        out_mp4 = tmp_path / "out.mp4"

        with (
            patch("gomoveshift.pipeline.ost_applier._probe_video_info", return_value=_stub_probe()),
            patch("gomoveshift.pipeline.ost_applier.Path.exists", return_value=True),
            patch(
                "gomoveshift.pipeline.ost_applier._resolve_encoding_args",
                return_value=["-c:v", "hevc_nvenc"],
            ) as mock_resolve,
            patch_ost_run() as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stderr=b"")
            ost_applier.apply_ost(
                input_path=in_mp4,
                output_path=out_mp4,
                location="Test",
                layout="regular_lf",
                prefer_nvenc=True,
            )
        # _resolve_encoding_args called with prefer_nvenc=True
        _, kwargs = mock_resolve.call_args
        # signature is (info, prefer_nvenc, input_name)
        positional = mock_resolve.call_args[0]
        assert positional[1] is True, f"expected prefer_nvenc=True threaded, got args={positional}"


class TestOstBurnUsesStallDetectingRunner:
    """The OST burn must run on ffmpeg_runner.run_ffmpeg_encode (the same
    Popen-based, stall-detecting runner color_filters.encode_video already
    uses), not a raw subprocess.run guarded only by a multi-hour wall-clock
    timeout. A genuinely stalled ffmpeg (process alive, no progress) must be
    killed within the stall window, not after burning most of the day's
    budget — and the failure the caller sees must say *why* ("stall", not
    a bare non-zero returncode).
    """

    def test_stalled_encode_is_killed_via_run_ffmpeg_encode(self, tmp_path):
        from gomoveshift.pipeline.ffmpeg_runner import FFmpegResult

        input_path = _make_input(tmp_path)
        output_path = tmp_path / "out.mp4"

        stall_result = FFmpegResult(
            success=False,
            returncode=-1,
            wall_time=45.0,
            stderr_tail="no progress for 600s",
            killed_reason="stall",
        )
        mock_encode = MagicMock(return_value=stall_result)

        with (
            patch("gomoveshift.pipeline.ost_applier._probe_video_info", return_value=_stub_probe()),
            patch("gomoveshift.pipeline.ost_applier.Path.exists", return_value=True),
            patch("gomoveshift.pipeline.color.detect_nvenc", return_value=True),
            patch.object(ost_applier, "run_ffmpeg_encode", mock_encode, create=True),
        ):
            result = ost_applier.apply_ost(
                input_path=input_path,
                output_path=output_path,
                location="Test Place",
                layout="regular_lf",
            )

        assert mock_encode.called, (
            "the OST burn must call ffmpeg_runner.run_ffmpeg_encode (stall "
            "detection) instead of a raw subprocess.run with only a "
            "wall-clock timeout"
        )
        _, kwargs = mock_encode.call_args
        stall_timeout = kwargs.get("stall_timeout")
        assert stall_timeout is not None and stall_timeout <= 1800, (
            f"stall_timeout must be well inside the ~7200s+ wall-clock "
            f"budget, got {stall_timeout!r}"
        )
        assert result is None, "a killed (stalled) burn must not report a finished LF"
