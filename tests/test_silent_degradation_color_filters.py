"""Regression tests for color_filters.py silent-degradation.

Ensures previously-silent (or INFO-only) fallback paths now emit WARNING logs
with a self-signalling prefix. All tests are pure unit tests that patch
subprocess.run so they do not actually invoke ffmpeg.

Covers:
- detect_nvenc(): exception branch + nonzero-returncode branch
- get_encoding_args(): has_nvenc=False emits WARNING
- encode_video() codec info → WARNING when falling back to libx265
"""

from __future__ import annotations

import logging
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline import color_filters

LOGGER_NAME = "gomoveshift.pipeline.color_filters"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_completed(returncode: int):
    return SimpleNamespace(returncode=returncode, stdout=b"", stderr=b"")


# ---------------------------------------------------------------------------
# detect_nvenc()
# ---------------------------------------------------------------------------


class TestDetectNvencWarnings:
    """Probe-failure paths must RAISE unconditionally.

    Returning False with a WARNING would let a silent libx265 fallback run
    unnoticed during long runs, at a quality cost. The
    GMS_ALLOW_CPU_FALLBACK env-var workaround was deleted.
    """

    def test_timeout_expired_raises_when_fallback_not_authorized(self, monkeypatch):
        monkeypatch.delenv("GMS_ALLOW_CPU_FALLBACK", raising=False)
        with patch.object(
            color_filters.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=10),
        ):
            with pytest.raises(color_filters.NVENCUnavailableError, match="TimeoutExpired"):
                color_filters.detect_nvenc()

    def test_file_not_found_raises_when_fallback_not_authorized(self, monkeypatch):
        monkeypatch.delenv("GMS_ALLOW_CPU_FALLBACK", raising=False)
        with patch.object(
            color_filters.subprocess,
            "run",
            side_effect=FileNotFoundError("ffmpeg not on PATH"),
        ):
            with pytest.raises(color_filters.NVENCUnavailableError, match="FileNotFoundError"):
                color_filters.detect_nvenc()

    def test_nonzero_returncode_raises_when_fallback_not_authorized(self, monkeypatch):
        monkeypatch.delenv("GMS_ALLOW_CPU_FALLBACK", raising=False)
        with patch.object(color_filters.subprocess, "run", return_value=_fake_completed(1)):
            with pytest.raises(color_filters.NVENCUnavailableError, match="nonzero"):
                color_filters.detect_nvenc()

    def test_success_does_not_warn(self, monkeypatch, caplog):
        # Success path: env var doesn't matter, no logs at WARNING+.
        monkeypatch.delenv("GMS_ALLOW_CPU_FALLBACK", raising=False)
        with patch.object(color_filters.subprocess, "run", return_value=_fake_completed(0)):
            with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
                result = color_filters.detect_nvenc()

        assert result is True
        assert not any(rec.levelno >= logging.WARNING for rec in caplog.records), (
            f"Unexpected warnings on success: {[r.message for r in caplog.records]}"
        )


# ---------------------------------------------------------------------------
# get_encoding_args()
# ---------------------------------------------------------------------------


class TestGetEncodingArgsFallbackWarning:
    def test_has_nvenc_false_emits_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            args = color_filters.get_encoding_args(
                width=1920, height=1080, fps=29.97, has_nvenc=False
            )

        # libx265 code path must be selected
        assert "libx265" in args
        assert "hevc_nvenc" not in args

        msgs = [r.message for r in caplog.records]
        assert any("NVENC" in m and "libx265" in m for m in msgs), (
            f"Expected NVENC-unavailable WARNING, got: {msgs}"
        )
        # Self-signalling prefix
        assert any("⚠" in m for m in msgs), f"Expected ⚠ severity prefix, got: {msgs}"

    def test_has_nvenc_true_does_not_emit_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            args = color_filters.get_encoding_args(
                width=1920, height=1080, fps=29.97, has_nvenc=True
            )

        assert "hevc_nvenc" in args
        assert "libx265" not in args
        # No WARNING-level records from this module
        module_warnings = [
            r for r in caplog.records if r.name == LOGGER_NAME and r.levelno >= logging.WARNING
        ]
        assert not module_warnings, (
            f"Unexpected warnings on NVENC path: {[r.message for r in module_warnings]}"
        )


# ---------------------------------------------------------------------------
# encode_video() encoder-selection logging severity
# ---------------------------------------------------------------------------


class TestEncodeVideoCodecLogSeverity:
    def test_libx265_fallback_logs_warning(self, caplog, tmp_path):
        vid = tmp_path / "in.mp4"
        vid.write_bytes(b"\x00")
        out = tmp_path / "out.mp4"

        fake_result = SimpleNamespace(success=True, stderr_tail="")
        with patch.object(color_filters, "run_ffmpeg_encode", return_value=fake_result):
            with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
                color_filters.encode_video(
                    video_path=vid,
                    output_path=out,
                    filter_str="null",
                    width=1920,
                    height=1080,
                    fps=29.97,
                    duration=10.0,
                    has_nvenc=False,
                )

        msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("libx265" in m and "CRF 18" in m for m in msgs), (
            f"Expected libx265 CRF 18 WARNING, got: {msgs}"
        )
        assert any("⚠" in m for m in msgs), f"Expected ⚠ severity prefix, got: {msgs}"

    def test_nvenc_path_does_not_warn_about_encoder(self, caplog, tmp_path):
        vid = tmp_path / "in.mp4"
        vid.write_bytes(b"\x00")
        out = tmp_path / "out.mp4"

        fake_result = SimpleNamespace(success=True, stderr_tail="")
        with patch.object(color_filters, "run_ffmpeg_encode", return_value=fake_result):
            with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
                color_filters.encode_video(
                    video_path=vid,
                    output_path=out,
                    filter_str="null",
                    width=1920,
                    height=1080,
                    fps=29.97,
                    duration=10.0,
                    has_nvenc=True,
                )

        # The only WARNING we'd expect on the NVENC path would be from
        # get_encoding_args if has_nvenc were False — not here.
        libx265_warnings = [
            r
            for r in caplog.records
            if r.levelno >= logging.WARNING and r.name == LOGGER_NAME and "libx265" in r.message
        ]
        assert not libx265_warnings, (
            f"Unexpected libx265 WARNING on NVENC path: {[r.message for r in libx265_warnings]}"
        )


# ---------------------------------------------------------------------------
# Hard-fail contract — no env-var escape hatch
# ---------------------------------------------------------------------------


class TestNvencHardFailContract:
    """detect_nvenc raises unconditionally on probe failure.

    The legacy GMS_ALLOW_CPU_FALLBACK=1 escape hatch is deleted. CPU
    (libx265) encode is unrepresentable from the drive run path.
    Tests assert the env var is ignored AND no probe-failure path
    can return False.
    """

    def test_cpu_fallback_allowed_symbol_removed(self):
        assert not hasattr(color_filters, "_cpu_fallback_allowed"), (
            "_cpu_fallback_allowed must stay deleted; the "
            "env-var escape hatch is gone by construction"
        )

    def test_gms_allow_cpu_fallback_env_ignored_on_timeout(self, monkeypatch):
        # Env var set to the legacy "1" value — must NOT alter behaviour.
        monkeypatch.setenv("GMS_ALLOW_CPU_FALLBACK", "1")
        with patch.object(
            color_filters.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=10),
        ):
            with pytest.raises(color_filters.NVENCUnavailableError, match="TimeoutExpired"):
                color_filters.detect_nvenc()

    def test_gms_allow_cpu_fallback_env_ignored_on_nonzero(self, monkeypatch):
        monkeypatch.setenv("GMS_ALLOW_CPU_FALLBACK", "1")
        with patch.object(color_filters.subprocess, "run", return_value=_fake_completed(1)):
            with pytest.raises(color_filters.NVENCUnavailableError, match="nonzero"):
                color_filters.detect_nvenc()

    def test_gms_allow_cpu_fallback_env_ignored_on_filenotfound(self, monkeypatch):
        monkeypatch.setenv("GMS_ALLOW_CPU_FALLBACK", "1")
        with patch.object(
            color_filters.subprocess,
            "run",
            side_effect=FileNotFoundError("ffmpeg not on PATH"),
        ):
            with pytest.raises(color_filters.NVENCUnavailableError, match="FileNotFoundError"):
                color_filters.detect_nvenc()

    def test_libx265_unreachable_in_drive_path_single_pass(self, monkeypatch):
        """SinglePassProcessor.__init__ propagates NVENCUnavailableError.

        Probe failure during construction means no encoder instance is
        ever produced — the `_has_nvenc()=False` branch is unreachable.
        """
        monkeypatch.delenv("GMS_ALLOW_CPU_FALLBACK", raising=False)
        from gomoveshift.config import Config
        from gomoveshift.pipeline.single_pass import SinglePassProcessor

        with patch.object(color_filters.subprocess, "run", return_value=_fake_completed(1)):
            with pytest.raises(color_filters.NVENCUnavailableError):
                SinglePassProcessor(Config())

    def test_libx265_unreachable_in_drive_path_ost_applier(self, monkeypatch):
        """ost_applier._resolve_encoding_args propagates NVENCUnavailableError.

        prefer_nvenc=False (shorts) — first-probe failure raises with no
        retry. prefer_nvenc=True (LF) — retry then raise.
        """
        monkeypatch.delenv("GMS_ALLOW_CPU_FALLBACK", raising=False)
        from gomoveshift.pipeline.ost_applier import _resolve_encoding_args

        info = {
            "width": 3840,
            "height": 2160,
            "fps": 29.97,
            "bitrate_kbps": 52000,
            "duration": 600.0,
        }

        # prefer_nvenc=False: first failure propagates
        with patch.object(color_filters.subprocess, "run", return_value=_fake_completed(1)):
            with pytest.raises(color_filters.NVENCUnavailableError):
                _resolve_encoding_args(info, prefer_nvenc=False, input_name="short_01.mp4")

        # prefer_nvenc=True: still raises (after retry); env var ignored
        monkeypatch.setenv("GMS_ALLOW_CPU_FALLBACK", "1")
        with (
            patch.object(color_filters.subprocess, "run", return_value=_fake_completed(1)),
            patch("gomoveshift.pipeline.ost_applier.time.sleep"),
        ):
            with pytest.raises(color_filters.NVENCUnavailableError):
                _resolve_encoding_args(info, prefer_nvenc=True, input_name="lf_02.mp4")
