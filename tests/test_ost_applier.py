"""Tests for ost_applier.

The OST applier is the single authoritative place where OST text gets
burned onto a finished video artifact (day master, regular LF, or short).
It re-encodes the input with an ffmpeg drawtext filter using layout-
specific timing:

  * regular_lf: location overlay at t=2s-10s (over the montage intro),
    website CTA (gomoveshift.com) at end
  * short: location overlay at t=0s-5s, YouTube CTA mid-video,
    website CTA at end

Tests mock subprocess.run to capture and assert on the ffmpeg command
without invoking real encoding.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gomoveshift.pipeline.ffmpeg_runner import VideoValidation

from ost_subprocess_harness import monkeypatch_ost_run


@pytest.fixture(autouse=True)
def _accept_burned_artifact(monkeypatch):
    """apply_ost ffprobe-validates the artifact it burned; ffmpeg is mocked here.

    These tests assert on the emitted ffmpeg command, so the "output" is a
    touched empty file that no real validator would accept. Post-encode
    validation itself is covered by tests/test_ost_output_validation.py.
    """
    monkeypatch.setattr(
        "gomoveshift.pipeline.ost_applier.validate_video",
        lambda *_a, **_kw: VideoValidation(valid=True),
    )


def _mock_ffprobe_response(
    width: int = 3840,
    duration: float = 600.0,
    height: int = 2160,
    fps: float = 29.97,
    bitrate_kbps: int = 53000,
):
    """Return a fake ffprobe JSON stdout with video stream and format info."""
    import json

    data = {
        "streams": [{"width": width, "height": height, "r_frame_rate": f"{int(fps * 1000)}/1000"}],
        "format": {"duration": str(duration), "bit_rate": str(bitrate_kbps * 1000)},
    }
    return json.dumps(data).encode("utf-8")


class TestApplyOSTBasics:
    def test_applied_to_regular_lf_layout(self, tmp_path, monkeypatch):
        """regular_lf layout burns the location at t=2-10s (over the montage intro)."""
        from gomoveshift.pipeline.ost_applier import apply_ost

        input_path = tmp_path / "regular_lf_01.mp4"
        input_path.touch()
        output_path = tmp_path / "regular_lf_01_final.mp4"

        captured_cmds = []

        def fake_run(cmd, *args, **kwargs):
            captured_cmds.append(list(cmd))
            result = MagicMock()
            result.returncode = 0
            if "ffprobe" in str(cmd[0]):
                result.stdout = _mock_ffprobe_response(3840, 1800.0)
            else:
                result.stdout = b""
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.touch()
            result.stderr = b""
            return result

        monkeypatch_ost_run(monkeypatch, fake_run)
        monkeypatch.setattr(
            "gomoveshift.pipeline.ost_applier.Path.exists",
            lambda self: True,
        )

        result = apply_ost(
            input_path=input_path,
            output_path=output_path,
            location="Frogmore to Port Royal via Overland Trail",
            layout="regular_lf",
        )

        assert result == output_path
        encode = next(
            (c for c in captured_cmds if "ffmpeg" in str(c[0]) and "-vf" in c),
            None,
        )
        assert encode is not None
        vf = encode[encode.index("-vf") + 1]
        assert "Frogmore to Port Royal" in vf
        # regular_lf timing: location at 2-10s (overlays montage intro)
        assert "between(t,2" in vf or "between(t\\,2" in vf
        # Long-form: website CTA yes, YouTube CTA no
        assert "gomoveshift.com" in vf
        assert "Full drive" not in vf

    def test_applied_to_short_layout(self, tmp_path, monkeypatch):
        """short layout burns location at t=0-5s, YouTube CTA, and website."""
        from gomoveshift.pipeline.ost_applier import apply_ost

        input_path = tmp_path / "short_01.mp4"
        input_path.touch()
        output_path = tmp_path / "short_01_final.mp4"

        captured_cmds = []

        def fake_run(cmd, *args, **kwargs):
            captured_cmds.append(list(cmd))
            result = MagicMock()
            result.returncode = 0
            if "ffprobe" in str(cmd[0]):
                result.stdout = _mock_ffprobe_response(2160, 35.0)
            else:
                result.stdout = b""
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.touch()
            result.stderr = b""
            return result

        monkeypatch_ost_run(monkeypatch, fake_run)
        monkeypatch.setattr(
            "gomoveshift.pipeline.ost_applier.Path.exists",
            lambda self: True,
        )

        result = apply_ost(
            input_path=input_path,
            output_path=output_path,
            location="Beaufort, SC",
            layout="short",
        )
        assert result == output_path

        encode = next(
            (c for c in captured_cmds if "ffmpeg" in str(c[0]) and "-vf" in c),
            None,
        )
        assert encode is not None
        vf = encode[encode.index("-vf") + 1]
        assert "Beaufort" in vf
        # short timing: location starts at 0s (realigned to the live shorts
        # window t∈[0,5])
        assert "between(t,0.0" in vf or "between(t\\,0.0" in vf
        # No subscribe CTA — replaced by YouTube CTA + website
        assert "Subscribe" not in vf
        assert "Full drive on YouTube" in vf or "Full drive" in vf
        assert "gomoveshift.com" in vf


class TestApplyOSTEncoder:
    def test_uses_hevc_nvenc_with_quality_params(self, tmp_path, monkeypatch):
        """The re-encode must use hevc_nvenc for speed at high quality."""
        from gomoveshift.pipeline.ost_applier import apply_ost

        input_path = tmp_path / "in.mp4"
        input_path.touch()
        output_path = tmp_path / "out.mp4"

        captured_cmds = []

        def fake_run(cmd, *args, **kwargs):
            captured_cmds.append(list(cmd))
            result = MagicMock()
            result.returncode = 0
            if "ffprobe" in str(cmd[0]):
                result.stdout = _mock_ffprobe_response(3840, 600.0)
            else:
                result.stdout = b""
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.touch()
            result.stderr = b""
            return result

        monkeypatch_ost_run(monkeypatch, fake_run)
        monkeypatch.setattr(
            "gomoveshift.pipeline.ost_applier.Path.exists",
            lambda self: True,
        )

        apply_ost(
            input_path=input_path,
            output_path=output_path,
            location="Test",
            layout="regular_lf",
        )

        encode = next(
            (c for c in captured_cmds if "ffmpeg" in str(c[0]) and "-vf" in c),
            None,
        )
        assert encode is not None
        assert "hevc_nvenc" in encode

    def test_audio_is_re_encoded_for_loudnorm(self, tmp_path, monkeypatch):
        """OST burn uses AAC (not stream-copy) so the loudnorm -af filter takes effect."""
        from gomoveshift.pipeline.ost_applier import apply_ost

        input_path = tmp_path / "in.mp4"
        input_path.touch()
        output_path = tmp_path / "out.mp4"
        captured = []

        def fake_run(cmd, *args, **kwargs):
            captured.append(list(cmd))
            result = MagicMock()
            result.returncode = 0
            if "ffprobe" in str(cmd[0]):
                result.stdout = _mock_ffprobe_response(3840, 600.0)
            else:
                result.stdout = b""
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.touch()
            result.stderr = b""
            return result

        monkeypatch_ost_run(monkeypatch, fake_run)
        monkeypatch.setattr(
            "gomoveshift.pipeline.ost_applier.Path.exists",
            lambda self: True,
        )

        apply_ost(
            input_path=input_path,
            output_path=output_path,
            location="Test",
            layout="regular_lf",
        )

        encode = next(c for c in captured if "ffmpeg" in str(c[0]) and "-vf" in c)
        # OST burn now uses AAC for per-LF loudnorm; stream-copy is gone
        assert "-c:a" in encode
        ca_idx = encode.index("-c:a")
        assert encode[ca_idx + 1] == "aac"
        assert "copy" not in encode[ca_idx : ca_idx + 2]


class TestApplyOSTFontScaling:
    def test_4k_input_uses_scaled_font_sizes(self, tmp_path, monkeypatch):
        """At 3840 width, font sizes scale to 96/72/72 (1.5x cap)."""
        from gomoveshift.pipeline.ost_applier import apply_ost

        input_path = tmp_path / "in.mp4"
        input_path.touch()
        output_path = tmp_path / "out.mp4"
        captured = []

        def fake_run(cmd, *args, **kwargs):
            captured.append(list(cmd))
            result = MagicMock()
            result.returncode = 0
            if "ffprobe" in str(cmd[0]):
                result.stdout = _mock_ffprobe_response(3840, 600.0)
            else:
                result.stdout = b""
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.touch()
            result.stderr = b""
            return result

        monkeypatch_ost_run(monkeypatch, fake_run)
        monkeypatch.setattr(
            "gomoveshift.pipeline.ost_applier.Path.exists",
            lambda self: True,
        )

        apply_ost(
            input_path=input_path,
            output_path=output_path,
            location="Test",
            layout="regular_lf",
        )

        encode = next(c for c in captured if "ffmpeg" in str(c[0]) and "-vf" in c)
        vf = encode[encode.index("-vf") + 1]
        # Brand 64 * 1.5 = 96, location 48 * 1.5 = 72
        assert "fontsize=96" in vf
        assert "fontsize=72" in vf

    def test_1080p_input_uses_reference_font_sizes(self, tmp_path, monkeypatch):
        """At 1080-width playback, fonts use the reference (1.0x) sizes."""
        from gomoveshift.pipeline.ost_applier import apply_ost

        input_path = tmp_path / "in.mp4"
        input_path.touch()
        output_path = tmp_path / "out.mp4"
        captured = []

        def fake_run(cmd, *args, **kwargs):
            captured.append(list(cmd))
            result = MagicMock()
            result.returncode = 0
            if "ffprobe" in str(cmd[0]):
                result.stdout = _mock_ffprobe_response(1920, 600.0)
            else:
                result.stdout = b""
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.touch()
            result.stderr = b""
            return result

        monkeypatch_ost_run(monkeypatch, fake_run)
        monkeypatch.setattr(
            "gomoveshift.pipeline.ost_applier.Path.exists",
            lambda self: True,
        )

        apply_ost(
            input_path=input_path,
            output_path=output_path,
            location="Test",
            layout="regular_lf",
        )

        encode = next(c for c in captured if "ffmpeg" in str(c[0]) and "-vf" in c)
        vf = encode[encode.index("-vf") + 1]
        assert "fontsize=64" in vf
        assert "fontsize=48" in vf


class TestApplyOSTEdgeCases:
    def test_empty_location_hard_fails(
        self,
        tmp_path,
        monkeypatch,
    ):
        """An empty location at burn time is an upstream state defect.
        apply_ost now raises (greppable OST-EMPTY-LOCATION) instead of the
        legacy silent stream-copy SUCCESS that shipped title-less videos."""
        import pytest

        from gomoveshift.pipeline.ost_applier import apply_ost

        input_path = tmp_path / "in.mp4"
        input_path.touch()
        output_path = tmp_path / "out.mp4"

        monkeypatch.setattr(
            "gomoveshift.pipeline.ost_applier.Path.exists",
            lambda self: True,
        )

        with pytest.raises(ValueError, match="OST-EMPTY-LOCATION"):
            apply_ost(
                input_path=input_path,
                output_path=output_path,
                location="",
                layout="regular_lf",
            )

    def test_invalid_layout_raises(self, tmp_path):
        from gomoveshift.pipeline.ost_applier import apply_ost
        import pytest

        input_path = tmp_path / "in.mp4"
        input_path.touch()
        output_path = tmp_path / "out.mp4"

        with pytest.raises(ValueError, match="layout"):
            apply_ost(
                input_path=input_path,
                output_path=output_path,
                location="Test",
                layout="invalid_layout",
            )

    def test_missing_input_returns_none(self, tmp_path):
        from gomoveshift.pipeline.ost_applier import apply_ost

        input_path = tmp_path / "nonexistent.mp4"
        output_path = tmp_path / "out.mp4"

        # No monkeypatch on Path.exists — the real file doesn't exist
        result = apply_ost(
            input_path=input_path,
            output_path=output_path,
            location="Test",
            layout="regular_lf",
        )
        assert result is None

    def test_ffmpeg_failure_returns_none(self, tmp_path, monkeypatch):
        from gomoveshift.pipeline.ost_applier import apply_ost

        input_path = tmp_path / "in.mp4"
        input_path.touch()
        output_path = tmp_path / "out.mp4"

        def fake_run(cmd, *args, **kwargs):
            result = MagicMock()
            if "ffprobe" in str(cmd[0]):
                result.returncode = 0
                result.stdout = _mock_ffprobe_response(3840, 600.0)
                result.stderr = b""
            else:
                # ffmpeg fails
                result.returncode = 1
                result.stdout = b""
                result.stderr = b"ffmpeg error: something went wrong"
            return result

        monkeypatch_ost_run(monkeypatch, fake_run)
        monkeypatch.setattr(
            "gomoveshift.pipeline.ost_applier.Path.exists",
            lambda self: True,
        )
        # No GPU in test env — stub the NVENC detector so the resolver
        # doesn't try to probe ffmpeg via subprocess (that path raises
        # NVENCUnavailableError on dev/CI hosts and masks the actual
        # ffmpeg-failure scenario this test exercises).
        monkeypatch.setattr(
            "gomoveshift.pipeline.color.detect_nvenc",
            lambda: True,
        )

        result = apply_ost(
            input_path=input_path,
            output_path=output_path,
            location="Test",
            layout="regular_lf",
        )
        assert result is None


class TestLayoutConstants:
    def test_layout_regular_lf_timing(self):
        from gomoveshift.pipeline.ost_applier import LAYOUTS

        layout = LAYOUTS["regular_lf"]
        # Title overlays the montage intro — retention-friendly YouTube window
        assert layout["location_start_sec"] == 2.0
        assert layout["location_duration_sec"] == 8.0
        assert layout["show_youtube_cta"] is False
        assert layout["show_website"] is True
        # Brand logo dropped from regular_lf to avoid
        # double-logo collision with YouTube's channel watermark.
        assert layout["show_logo"] is False
        assert layout["show_persistent_url"] is True

    def test_layout_short_timing(self):
        from gomoveshift.pipeline.ost_applier import LAYOUTS

        layout = LAYOUTS["short"]
        # Aligned to the live shorts window (build_ost_filter ost_show_duration=5.0,
        # title t∈[0,5]); previously drifted to 0.5-8s.
        assert layout["location_start_sec"] == 0.0
        assert layout["location_duration_sec"] == 5.0
        assert layout["show_youtube_cta"] is True
        assert layout["show_website"] is True

    def test_layout_short_cta_upper_third(self):
        from gomoveshift.pipeline.ost_applier import LAYOUTS

        layout = LAYOUTS["short"]
        # Shorts CTAs in upper third to avoid mobile platform UI chrome
        assert layout["cta_y_frac"] == 0.25

    def test_layout_lf_cta_lower_area(self):
        from gomoveshift.pipeline.ost_applier import LAYOUTS

        layout = LAYOUTS["regular_lf"]
        # Long-form website CTA in lower area (standard for landscape)
        assert layout["cta_y_frac"] == 0.75


class TestApplyOSTEncoderQuality:
    """apply_ost must use the same encoder quality path as SinglePassProcessor.

    The previous implementation used bare `-cq 18` with no VBR floor, producing
    ~20 Mbps output from ~53 Mbps sources. This caused Gate B failures because
    the 50 Mbps bitrate floor applies to all artifacts including LFs.

    The fix: apply_ost calls `get_encoding_args()` from `color.py`, which adds
    `-b:v`/`-maxrate`/`-bufsize` VBR floor hints, multipass, lookahead, and AQ
    — matching the SinglePassProcessor quality path.
    """

    def test_encode_includes_vbr_bitrate_floor(self, tmp_path, monkeypatch):
        """The ffmpeg command must include -b:v/-maxrate/-bufsize for VBR floor."""
        from gomoveshift.pipeline.ost_applier import apply_ost

        input_path = tmp_path / "lf_01_raw.mp4"
        input_path.touch()
        output_path = tmp_path / "lf_01.mp4"
        captured = []

        def fake_run(cmd, *args, **kwargs):
            captured.append(list(cmd))
            result = MagicMock()
            result.returncode = 0
            if "ffprobe" in str(cmd[0]):
                result.stdout = _mock_ffprobe_response(3840, 1800.0)
            else:
                result.stdout = b""
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.touch()
            result.stderr = b""
            return result

        monkeypatch_ost_run(monkeypatch, fake_run)
        monkeypatch.setattr(
            "gomoveshift.pipeline.ost_applier.Path.exists",
            lambda self: True,
        )

        apply_ost(
            input_path=input_path,
            output_path=output_path,
            location="Frogmore to Port Royal, SC",
            layout="regular_lf",
        )

        encode = next(
            (c for c in captured if "ffmpeg" in str(c[0]) and "-vf" in c),
            None,
        )
        assert encode is not None
        assert "-b:v" in encode, f"Missing -b:v in encode cmd: {encode}"
        assert "-maxrate" in encode, f"Missing -maxrate in encode cmd: {encode}"
        assert "-bufsize" in encode, f"Missing -bufsize in encode cmd: {encode}"

    def test_encode_uses_full_nvenc_quality_args(self, tmp_path, monkeypatch):
        """Must use multipass, lookahead, AQ, 10-bit, p7 — not bare cq."""
        from gomoveshift.pipeline.ost_applier import apply_ost

        input_path = tmp_path / "lf_01_raw.mp4"
        input_path.touch()
        output_path = tmp_path / "lf_01.mp4"
        captured = []

        def fake_run(cmd, *args, **kwargs):
            captured.append(list(cmd))
            result = MagicMock()
            result.returncode = 0
            if "ffprobe" in str(cmd[0]):
                result.stdout = _mock_ffprobe_response(3840, 1800.0)
            else:
                result.stdout = b""
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.touch()
            result.stderr = b""
            return result

        monkeypatch_ost_run(monkeypatch, fake_run)
        monkeypatch.setattr(
            "gomoveshift.pipeline.ost_applier.Path.exists",
            lambda self: True,
        )

        apply_ost(
            input_path=input_path,
            output_path=output_path,
            location="Test",
            layout="regular_lf",
        )

        encode = next(
            (c for c in captured if "ffmpeg" in str(c[0]) and "-vf" in c),
            None,
        )
        assert encode is not None
        assert "-multipass" in encode, "Missing multipass"
        assert "-rc-lookahead" in encode, "Missing rc-lookahead"
        assert "-spatial-aq" in encode, "Missing spatial-aq"
        assert "-temporal-aq" in encode, "Missing temporal-aq"
        assert "p010le" in encode, "Missing 10-bit pixel format"

    def test_bitrate_floor_meets_gate_b_threshold(self, tmp_path, monkeypatch):
        """The -b:v value must be >= 40000k (LONGFORM_MIN_BITRATE_4K) for 4K
        input, ensuring Gate B's bitrate floor is satisfiable."""
        from gomoveshift.pipeline.ost_applier import apply_ost

        input_path = tmp_path / "lf_01_raw.mp4"
        input_path.touch()
        output_path = tmp_path / "lf_01.mp4"
        captured = []

        def fake_run(cmd, *args, **kwargs):
            captured.append(list(cmd))
            result = MagicMock()
            result.returncode = 0
            if "ffprobe" in str(cmd[0]):
                result.stdout = _mock_ffprobe_response(3840, 1800.0)
            else:
                result.stdout = b""
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.touch()
            result.stderr = b""
            return result

        monkeypatch_ost_run(monkeypatch, fake_run)
        monkeypatch.setattr(
            "gomoveshift.pipeline.ost_applier.Path.exists",
            lambda self: True,
        )

        apply_ost(
            input_path=input_path,
            output_path=output_path,
            location="Test",
            layout="regular_lf",
        )

        encode = next(
            (c for c in captured if "ffmpeg" in str(c[0]) and "-vf" in c),
            None,
        )
        assert encode is not None
        bv_idx = encode.index("-b:v")
        bv_val = encode[bv_idx + 1]  # e.g. "40000k"
        bv_kbps = int(bv_val.rstrip("k"))
        assert bv_kbps >= 40000, f"-b:v {bv_kbps}k is below 40000k minimum for 4K"


class TestOstApplierLoudnorm:
    """Per-LF audio ATTENUATION during OST burn: the burn
    re-encode carries an attenuate-only ``aresample=async=1000,volume=<=0 dB``
    filter — never a loudnorm boost."""

    def _make_fake_run(self, captured, output_path):
        def fake_run(cmd, *args, **kwargs):
            captured.append(list(cmd))
            result = MagicMock()
            result.returncode = 0
            if "ffprobe" in str(cmd[0]):
                result.stdout = _mock_ffprobe_response(3840, 600.0)
            else:
                result.stdout = b""
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.touch()
            result.stderr = b""
            return result

        return fake_run

    def test_no_logo_command_attenuates_audio(self, tmp_path, monkeypatch):
        """apply_ost (no-logo path) must pass an attenuate-only -af
        (aresample=async=1000,volume=<=0 dB) — never a loudnorm boost."""
        from gomoveshift.pipeline.ost_applier import apply_ost

        input_path = tmp_path / "lf_01.mp4"
        input_path.touch()
        output_path = tmp_path / "lf_01_final.mp4"
        captured = []

        monkeypatch_ost_run(monkeypatch, self._make_fake_run(captured, output_path))
        monkeypatch.setattr(
            "gomoveshift.pipeline.ost_applier.Path.exists",
            lambda self: True,
        )

        apply_ost(
            input_path=input_path,
            output_path=output_path,
            location="Test Location",
            layout="regular_lf",
        )

        encode = next((c for c in captured if "ffmpeg" in str(c[0]) and "-vf" in c), None)
        assert encode is not None, "No ffmpeg encode command captured"
        assert "-af" in encode, "Missing -af in ffmpeg command"
        af_idx = encode.index("-af")
        af_val = encode[af_idx + 1]
        assert af_val.startswith("aresample=async=1000,volume="), af_val
        assert "loudnorm" not in af_val, f"-af value {af_val!r} must not contain 'loudnorm'"

    def test_no_logo_command_uses_aac_not_copy(self, tmp_path, monkeypatch):
        """apply_ost (no-logo) must use -c:a aac and NOT -c:a copy."""
        from gomoveshift.pipeline.ost_applier import apply_ost

        input_path = tmp_path / "lf_01.mp4"
        input_path.touch()
        output_path = tmp_path / "lf_01_final.mp4"
        captured = []

        monkeypatch_ost_run(monkeypatch, self._make_fake_run(captured, output_path))
        monkeypatch.setattr(
            "gomoveshift.pipeline.ost_applier.Path.exists",
            lambda self: True,
        )

        apply_ost(
            input_path=input_path,
            output_path=output_path,
            location="Test Location",
            layout="regular_lf",
        )

        encode = next((c for c in captured if "ffmpeg" in str(c[0]) and "-vf" in c), None)
        assert encode is not None
        assert "-c:a" in encode
        ca_idx = encode.index("-c:a")
        assert encode[ca_idx + 1] == "aac", "Expected -c:a aac, got stream-copy"
        # Verify no standalone 'copy' value immediately after -c:a
        flat = " ".join(encode)
        assert "-c:a copy" not in flat, "-c:a copy must not appear in OST encode command"

    def test_logo_command_attenuates_audio(self, tmp_path, monkeypatch):
        """apply_ost (logo path, filter_complex) must also pass the attenuate-only -af."""
        from gomoveshift.pipeline.ost_applier import apply_ost

        input_path = tmp_path / "lf_01.mp4"
        input_path.touch()
        logo_path = tmp_path / "logo.png"
        logo_path.touch()
        output_path = tmp_path / "lf_01_final.mp4"
        captured = []

        monkeypatch_ost_run(monkeypatch, self._make_fake_run(captured, output_path))
        # Path.exists must return True so the font + logo are "found"
        monkeypatch.setattr(
            "gomoveshift.pipeline.ost_applier.Path.exists",
            lambda self: True,
        )
        # Path.is_file must return True so logo_exists=True
        monkeypatch.setattr(
            "gomoveshift.pipeline.ost_applier.Path.is_file",
            lambda self: True,
        )

        # Use a layout that opts in to show_logo (patch LAYOUTS to force it on)
        import gomoveshift.pipeline.ost_applier as oa_mod

        patched_layouts = {k: dict(v) for k, v in oa_mod.LAYOUTS.items()}
        patched_layouts["regular_lf"]["show_logo"] = True
        monkeypatch.setattr(oa_mod, "LAYOUTS", patched_layouts)

        apply_ost(
            input_path=input_path,
            output_path=output_path,
            location="Test Location",
            layout="regular_lf",
            brand_logo_path=str(logo_path),
        )

        # Logo path uses -filter_complex not -vf
        encode = next(
            (c for c in captured if "ffmpeg" in str(c[0]) and "-filter_complex" in c), None
        )
        assert encode is not None, "No filter_complex encode command captured"
        assert "-af" in encode, "Missing -af in logo-path ffmpeg command"
        af_idx = encode.index("-af")
        af_val = encode[af_idx + 1]
        assert af_val.startswith("aresample=async=1000,volume="), af_val
        assert "loudnorm" not in af_val, f"-af value {af_val!r} must not contain 'loudnorm'"

    def test_stream_copy_path_has_no_loudnorm(self, tmp_path, monkeypatch):
        """A missing font takes _stream_copy — no loudnorm in that path.

        (The empty-location branch hard-fails, so the surviving
        graceful-degradation stream-copy path is the missing-font one.)"""
        from gomoveshift.pipeline.ost_applier import apply_ost

        input_path = tmp_path / "lf_01.mp4"
        input_path.touch()
        output_path = tmp_path / "lf_01_final.mp4"
        captured = []

        def fake_run(cmd, *args, **kwargs):
            captured.append(list(cmd))
            result = MagicMock()
            result.returncode = 0
            if "ffprobe" in str(cmd[0]):
                result.stdout = _mock_ffprobe_response(3840, 600.0)
            else:
                result.stdout = b""
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.touch()
            result.stderr = b""
            return result

        monkeypatch_ost_run(monkeypatch, fake_run)

        def fake_exists(self):
            # input exists; the supplied font path does not
            return str(self) != "/nonexistent/font.ttf"

        monkeypatch.setattr(
            "gomoveshift.pipeline.ost_applier.Path.exists",
            fake_exists,
        )

        # Missing font → stream-copy fallback (real graceful degradation)
        result = apply_ost(
            input_path=input_path,
            output_path=output_path,
            location="Real Location",
            layout="regular_lf",
            font_path="/nonexistent/font.ttf",
        )

        assert result == output_path
        # All captured ffmpeg commands should be stream-copy only
        encode_cmds = [c for c in captured if "ffmpeg" in str(c[0])]
        assert encode_cmds, "No ffmpeg command captured"
        for cmd in encode_cmds:
            flat = " ".join(cmd)
            assert "loudnorm" not in flat, (
                f"loudnorm must not appear in stream-copy path, got: {flat}"
            )

    def test_audio_attenuate_filter_includes_aresample(self):
        """_audio_attenuate_filter must prepend aresample=async=1000 (suppresses
        non-monotonic DTS) then an attenuate-only volume= gain — no loudnorm."""
        from gomoveshift.pipeline.ost_applier import _audio_attenuate_filter

        f = _audio_attenuate_filter()
        assert f.startswith("aresample=async=1000,volume="), (
            f"Filter must start with aresample + volume gain, got: {f!r}"
        )
        assert "loudnorm" not in f, f"Filter must not contain loudnorm, got: {f!r}"

    def test_get_encoding_args_normalize_audio_false(self):
        """normalize_audio=False returns -c:a copy (unchanged callers)."""
        from gomoveshift.pipeline.color_filters import get_encoding_args

        args = get_encoding_args(width=3840, height=2160, fps=29.97, normalize_audio=False)
        flat = " ".join(args)
        assert "-c:a copy" in flat, f"Expected -c:a copy in args, got: {flat}"
        assert "aac" not in flat, "aac must not appear when normalize_audio=False"

    def test_get_encoding_args_normalize_audio_true(self):
        """normalize_audio=True returns -c:a aac -b:a 384k, NOT copy."""
        from gomoveshift.pipeline.color_filters import get_encoding_args

        args = get_encoding_args(width=3840, height=2160, fps=29.97, normalize_audio=True)
        flat = " ".join(args)
        assert "-c:a aac" in flat, f"Expected -c:a aac in args, got: {flat}"
        assert "-b:a" in args, f"-b:a missing from args: {args}"
        ba_idx = args.index("-b:a")
        assert args[ba_idx + 1] == "384k", f"Expected 384k, got {args[ba_idx + 1]}"
        assert "-c:a copy" not in flat, "-c:a copy must not appear when normalize_audio=True"


# Realistic loudnorm pass-1 measurement JSON, as ffmpeg emits it to stderr for
# very quiet ambient content (~-30 LUFS input → attenuated -14.58 dB down to the
# -45 LUFS near-silent ceiling).
_MEASURE_JSON = (
    "[Parsed_loudnorm_0 @ 0x55] \n"
    "{\n"
    '\t"input_i" : "-30.42",\n'
    '\t"input_tp" : "-9.88",\n'
    '\t"input_lra" : "5.10",\n'
    '\t"input_thresh" : "-40.55",\n'
    '\t"output_i" : "-20.01",\n'
    '\t"output_tp" : "-1.51",\n'
    '\t"output_lra" : "4.90",\n'
    '\t"output_thresh" : "-30.12",\n'
    '\t"normalization_type" : "dynamic",\n'
    '\t"target_offset" : "0.42"\n'
    "}\n"
)


class TestOstApplierMeasureThenAttenuate:
    """apply_ost still runs a measurement pass, then feeds
    the MEASURED stats into an ATTENUATE-ONLY volume= gain (build_attenuate_filter),
    so each LF is pulled DOWN to the near-silent ceiling — never boosted."""

    def _make_fake_run(self, captured, output_path):
        def fake_run(cmd, *args, **kwargs):
            captured.append(list(cmd))
            result = MagicMock()
            result.returncode = 0
            if "ffprobe" in str(cmd[0]):
                result.stdout = _mock_ffprobe_response(3840, 600.0)
                result.stderr = b""
            elif "-vn" in cmd and "null" in cmd:
                # Audio measurement pass — emit the loudnorm JSON on stderr.
                result.stdout = b""
                result.stderr = _MEASURE_JSON.encode("utf-8")
            else:
                result.stdout = b""
                result.stderr = b""
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.touch()
            return result

        return fake_run

    def test_measure_audio_parses_loudnorm_json(self, tmp_path, monkeypatch):
        """_measure_audio_loudnorm parses the pass-1 JSON stats from stderr."""
        from gomoveshift.pipeline.ost_applier import _measure_audio_loudnorm

        captured = []
        out = tmp_path / "x.mp4"
        monkeypatch_ost_run(monkeypatch, self._make_fake_run(captured, out))
        measured = _measure_audio_loudnorm(tmp_path / "lf_01.mp4")
        assert measured.get("input_i") == "-30.42"
        assert measured.get("target_offset") == "0.42"
        # The measurement command is audio-only (-vn) and decodes to null.
        measure_cmd = next((c for c in captured if "-vn" in c and "null" in c), None)
        assert measure_cmd is not None, "no measurement pass command captured"
        assert "-vn" in measure_cmd
        assert "print_format=json" in " ".join(measure_cmd)

    def test_audio_filter_attenuates_from_measured_stats(self):
        """_audio_attenuate_filter(measured) pulls the measured source loudness
        DOWN to the near-silent ceiling: for input_i=-30.42 the gain is
        -45-(-30.42) = -14.58 dB — a bare volume= attenuation, no loudnorm."""
        from gomoveshift.pipeline.ost_applier import _audio_attenuate_filter

        measured = {
            "input_i": "-30.42",
            "input_tp": "-9.88",
            "input_lra": "5.10",
            "input_thresh": "-40.55",
            "target_offset": "0.42",
        }
        f = _audio_attenuate_filter(measured)
        assert f == "aresample=async=1000,volume=-14.58dB", f
        assert "loudnorm" not in f
        assert "measured_I" not in f

    def test_audio_filter_empty_measured_uses_fixed_attenuation(self):
        """No measured stats → conservative fixed attenuation (still never a boost)."""
        from gomoveshift.pipeline.ost_applier import _audio_attenuate_filter

        f = _audio_attenuate_filter({})
        assert f == "aresample=async=1000,volume=-25.00dB", f
        assert "loudnorm" not in f
        assert "measured_I" not in f

    def test_apply_ost_runs_measurement_then_attenuate_af(self, tmp_path, monkeypatch):
        """apply_ost issues a measurement pass and the encode -af carries the
        attenuate-only volume gain derived from the MEASURED stats."""
        from gomoveshift.pipeline.ost_applier import apply_ost

        input_path = tmp_path / "lf_01.mp4"
        input_path.touch()
        output_path = tmp_path / "lf_01_final.mp4"
        captured = []

        monkeypatch_ost_run(monkeypatch, self._make_fake_run(captured, output_path))
        monkeypatch.setattr(
            "gomoveshift.pipeline.ost_applier.Path.exists",
            lambda self: True,
        )

        result = apply_ost(
            input_path=input_path,
            output_path=output_path,
            location="Test Location",
            layout="regular_lf",
        )
        assert result == output_path

        # A measurement pass (audio-only, null muxer) was issued.
        measure_cmd = next((c for c in captured if "-vn" in c and "null" in c), None)
        assert measure_cmd is not None, "apply_ost did not run a measurement pass"
        assert "-vn" in measure_cmd

        # The video encode -af carries the attenuate-only gain derived from the
        # measured stats (input_i=-30.42 → -45-(-30.42) = -14.58 dB), no loudnorm.
        encode = next((c for c in captured if "ffmpeg" in str(c[0]) and "-vf" in c), None)
        assert encode is not None, "no encode command captured"
        af_val = encode[encode.index("-af") + 1]
        assert af_val == "aresample=async=1000,volume=-14.58dB", (
            f"-af must be the attenuate-only measured gain, got: {af_val!r}"
        )
        assert "loudnorm" not in af_val


class TestOstApplierSilentMaster:
    """--no-audio encodes the day master -an, so LF segments stream-cut from it
    have NO audio stream. apply_ost must detect the absent stream and burn -an
    rather than attaching an -af filter to nothing (which would error or measure
    a stream that isn't there). Regression guard for the silent-LF path."""

    def _make_fake_run(self, captured, output_path, *, has_audio):
        def fake_run(cmd, *args, **kwargs):
            captured.append(list(cmd))
            result = MagicMock()
            result.returncode = 0
            if "ffprobe" in str(cmd[0]):
                # The audio-presence probe (-select_streams a): report NO audio.
                if "a" in cmd and "stream=index" in cmd:
                    result.stdout = b"" if not has_audio else b"1\n"
                else:
                    result.stdout = _mock_ffprobe_response(3840, 600.0)
            else:
                result.stdout = b""
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.touch()
            result.stderr = b""
            return result

        return fake_run

    def _run(self, tmp_path, monkeypatch, *, has_audio, logo=False):
        from gomoveshift.pipeline.ost_applier import apply_ost

        input_path = tmp_path / "lf_01.mp4"
        input_path.touch()
        output_path = tmp_path / "lf_01_final.mp4"
        captured = []
        monkeypatch_ost_run(
            monkeypatch, self._make_fake_run(captured, output_path, has_audio=has_audio)
        )
        monkeypatch.setattr("gomoveshift.pipeline.ost_applier.Path.exists", lambda self: True)
        kwargs = {}
        if logo:
            kwargs["brand_logo_path"] = str(tmp_path / "logo.png")
        apply_ost(
            input_path=input_path,
            output_path=output_path,
            location="Test Location",
            layout="regular_lf",
            **kwargs,
        )
        return captured

    def test_silent_master_no_logo_uses_an_not_af(self, tmp_path, monkeypatch):
        captured = self._run(tmp_path, monkeypatch, has_audio=False)
        encode = next((c for c in captured if "ffmpeg" in str(c[0]) and "-vf" in c), None)
        assert encode is not None, "no encode command captured"
        assert "-an" in encode, f"silent master must burn -an, got: {encode}"
        assert "-af" not in encode, f"must NOT attach -af to a silent input, got: {encode}"
        # No audio-only measurement pass should run on a silent input.
        measure = next((c for c in captured if "ffmpeg" in str(c[0]) and "-vn" in c), None)
        assert measure is None, "must not run a loudnorm measurement on a silent input"

    def test_audio_present_still_uses_af(self, tmp_path, monkeypatch):
        captured = self._run(tmp_path, monkeypatch, has_audio=True)
        encode = next((c for c in captured if "ffmpeg" in str(c[0]) and "-vf" in c), None)
        assert encode is not None
        assert "-af" in encode, "an audio-bearing LF must still be attenuated with -af"
        assert "-an" not in encode
