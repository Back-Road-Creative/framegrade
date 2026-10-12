"""Tests for the logo overlay + persistent URL additions to apply_ost().

These cover the regression where the day-first pipeline (via apply_ost) was
missing the brand logo watermark and persistent URL text that existed in the
old non-day-first path (filtergraph._apply_logo_overlay + _apply_url_text).

The fix ports both into apply_ost so the single surviving assembly path
produces the full OST package: brand + location + logo + persistent URL +
end-card website CTA.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline import ost_applier
from gomoveshift.pipeline.ffmpeg_runner import VideoValidation

from ost_subprocess_harness import patch_ost_run


@pytest.fixture(autouse=True)
def _accept_burned_artifact(monkeypatch):
    """apply_ost ffprobe-validates the artifact it burned; ffmpeg is mocked here.

    Post-encode validation itself is covered by tests/test_ost_output_validation.py.
    """
    monkeypatch.setattr(
        "gomoveshift.pipeline.ost_applier.validate_video",
        lambda *_a, **_kw: VideoValidation(valid=True),
    )


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


class TestOstTimingFix:
    """regular_lf OST timing moved from t=31s to t=2s (overlays montage).

    The old t=31s made sense when regular_lf was a secondary format
    waiting for a 30s montage to finish. Now that regular_lf is the
    published format, the title should land in the retention window.
    """

    def test_regular_lf_location_starts_at_2s(self):
        layout = ost_applier.LAYOUTS["regular_lf"]
        assert layout["location_start_sec"] == 2.0
        assert layout["location_duration_sec"] == 8.0

    def test_short_layout_unchanged(self):
        layout = ost_applier.LAYOUTS["short"]
        # Realigned to the live shorts window t∈[0,5];
        # previously 0.5-8s, which had drifted from build_ost_filter.
        assert layout["location_start_sec"] == 0.0
        assert layout["location_duration_sec"] == 5.0


class TestLogoOverlay:
    def test_apply_ost_accepts_brand_logo_path(self, tmp_path):
        """apply_ost() accepts a brand_logo_path kwarg (backward-compat optional)."""
        import inspect

        sig = inspect.signature(ost_applier.apply_ost)
        assert "brand_logo_path" in sig.parameters

    def test_regular_lf_does_not_add_logo_overlay(self, tmp_path):
        """regular_lf MUST NOT burn in the brand logo.

        Even with a valid brand_logo_path threaded through, the regular_lf
        layout suppresses the overlay (eliminates double-logo collision with YouTube's
        channel watermark). The capability lives on for opt-in layouts.
        """
        in_mp4 = _make_input(tmp_path)
        out_mp4 = tmp_path / "out.mp4"
        logo = tmp_path / "logo.png"
        logo.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        with (
            patch("gomoveshift.pipeline.ost_applier._probe_video_info", return_value=_stub_probe()),
            patch("gomoveshift.pipeline.ost_applier.Path.exists", return_value=True),
            patch_ost_run() as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stderr=b"")
            ost_applier.apply_ost(
                input_path=in_mp4,
                output_path=out_mp4,
                location="Longwood to Fort Fremont",
                layout="regular_lf",
                brand_logo_path=str(logo),
            )
        cmd = mock_run.call_args[0][0]
        # Only the source -i input; no logo input
        i_flags = [i for i, a in enumerate(cmd) if a == "-i"]
        assert len(i_flags) == 1, f"regular_lf must not add logo input, got cmd={cmd}"
        # Simple -vf path, no filter_complex (no overlay needed)
        assert "-vf" in cmd
        assert "-filter_complex" not in cmd
        # No logo overlay segment anywhere in the filter string
        vf_idx = cmd.index("-vf")
        filter_str = cmd[vf_idx + 1]
        assert "overlay=" not in filter_str, (
            f"regular_lf filter must not contain logo overlay; got: {filter_str}"
        )
        assert "scale=240" not in filter_str

    def test_regular_lf_layout_show_logo_false(self):
        """Layout config flag is False — single source of truth for the suppression."""
        assert ost_applier.LAYOUTS["regular_lf"]["show_logo"] is False

    def test_logo_overlay_capability_preserved_for_opt_in_layouts(self, tmp_path, monkeypatch):
        """The bottom-right logo overlay code path is preserved for any
        layout that opts in via show_logo=True. Regression guard so the
        ost_applier.apply_ost() filter_complex branch isn't dead code after
        the regular_lf layout no longer opts in.
        """
        # Inject a temporary opt-in layout that exercises the logo branch.
        custom_layout = dict(ost_applier.LAYOUTS["regular_lf"])
        custom_layout["show_logo"] = True
        monkeypatch.setitem(ost_applier.LAYOUTS, "_logo_optin_test", custom_layout)

        in_mp4 = _make_input(tmp_path)
        out_mp4 = tmp_path / "out.mp4"
        logo = tmp_path / "logo.png"
        logo.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        with (
            patch("gomoveshift.pipeline.ost_applier._probe_video_info", return_value=_stub_probe()),
            patch("gomoveshift.pipeline.ost_applier.Path.exists", return_value=True),
            patch_ost_run() as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stderr=b"")
            ost_applier.apply_ost(
                input_path=in_mp4,
                output_path=out_mp4,
                location="Longwood to Fort Fremont",
                layout="_logo_optin_test",
                brand_logo_path=str(logo),
            )
        cmd = mock_run.call_args[0][0]
        i_flags = [i for i, a in enumerate(cmd) if a == "-i"]
        assert len(i_flags) >= 2, f"opt-in layout must add logo input, got cmd={cmd}"
        assert "-filter_complex" in cmd
        fc_idx = cmd.index("-filter_complex")
        filter_str = cmd[fc_idx + 1]
        assert "scale=240" in filter_str
        assert "overlay=" in filter_str
        assert "W-w-40" in filter_str or "w-40" in filter_str

    def test_logo_missing_does_not_warn_for_regular_lf(self, tmp_path, caplog):
        """regular_lf no longer requests the logo, so a
        missing logo file produces NO warning (the old behavior of
        warn-and-continue applied when show_logo=True). This avoids
        spurious 'logo not found' noise on every LF encode.
        """
        in_mp4 = _make_input(tmp_path)
        out_mp4 = tmp_path / "out.mp4"
        missing_logo = tmp_path / "nope.png"  # does not exist

        def exists_side(path_obj):
            s = str(path_obj)
            if "nope.png" in s:
                return False
            return True

        with (
            patch("gomoveshift.pipeline.ost_applier._probe_video_info", return_value=_stub_probe()),
            patch.object(Path, "exists", autospec=True, side_effect=exists_side),
            patch_ost_run() as mock_run,
            caplog.at_level("WARNING", logger="gomoveshift.pipeline.ost_applier"),
        ):
            mock_run.return_value = MagicMock(returncode=0, stderr=b"")
            ost_applier.apply_ost(
                input_path=in_mp4,
                output_path=out_mp4,
                location="Longwood to Fort Fremont",
                layout="regular_lf",
                brand_logo_path=str(missing_logo),
            )
        # No "logo not found" warning — show_logo=False short-circuits the check
        assert not any(
            "brand_logo_path" in r.message.lower() and "not found" in r.message.lower()
            for r in caplog.records
        )
        cmd = mock_run.call_args[0][0]
        i_flags = [i for i, a in enumerate(cmd) if a == "-i"]
        assert len(i_flags) == 1

    def test_no_logo_path_uses_simple_vf_path(self, tmp_path):
        """brand_logo_path=None: keep the existing simple -vf drawtext path."""
        in_mp4 = _make_input(tmp_path)
        out_mp4 = tmp_path / "out.mp4"

        with (
            patch("gomoveshift.pipeline.ost_applier._probe_video_info", return_value=_stub_probe()),
            patch("gomoveshift.pipeline.ost_applier.Path.exists", return_value=True),
            patch_ost_run() as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stderr=b"")
            ost_applier.apply_ost(
                input_path=in_mp4,
                output_path=out_mp4,
                location="Longwood to Fort Fremont",
                layout="regular_lf",
                # brand_logo_path omitted
            )
        cmd = mock_run.call_args[0][0]
        # Single input, old -vf form
        assert "-vf" in cmd
        assert "-filter_complex" not in cmd


class TestPersistentUrlText:
    """Persistent gomoveshift.com drawtext in top-right for the full video."""

    def test_url_text_in_filter_chain_for_regular_lf(self):
        """Filter chain for regular_lf includes a persistent URL drawtext."""
        layout = ost_applier.LAYOUTS["regular_lf"]
        chain = ost_applier._build_drawtext_filter(
            location="Sample Location",
            layout=layout,
            font_path=ost_applier.FONT_PATH,
            video_width=3840,
            video_duration=817.0,
        )
        # Persistent URL: text is gomoveshift.com AND positioned top-right (not center-bottom end-card)
        # Heuristic: the persistent URL drawtext has no enable clause, or enables for the full duration.
        # Count "gomoveshift.com" occurrences — we now have two (persistent + end-card) OR one (persistent
        # replaces end-card). Either is acceptable but at least one must be persistent.
        assert "gomoveshift.com" in chain
        # Persistent presence: either no enable, or enable spans the whole video (0..duration)
        # The end-card CTA has enable='between(t,X,Y)' with Y near duration. A persistent drawtext either
        # has no enable clause or one like enable='gte(t,0)' or enable='between(t,0,<full_duration>)'.
        # The simplest tell: a drawtext entry for gomoveshift.com that doesn't have a "duration-" offset
        # in its alpha or enable.
        assert (
            "x=w-text_w-" in chain
            or "x=main_w-text_w-" in chain
            or "x=W-text_w-" in chain
            or "x=(w-text_w)" in chain
        )


class TestShortsLayout:
    """Shorts keep their own OST window and don't get the whole-video logo (too small for 240px BR overlay)."""

    def test_shorts_dont_get_logo(self, tmp_path):
        """Shorts output path skips logo overlay (different aspect, watermark is distracting on 9:16)."""
        in_mp4 = _make_input(tmp_path)
        out_mp4 = tmp_path / "short_out.mp4"
        logo = tmp_path / "logo.png"
        logo.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        with (
            patch(
                "gomoveshift.pipeline.ost_applier._probe_video_info",
                return_value={
                    "width": 1080,
                    "height": 1920,
                    "fps": 29.97,
                    "bitrate_kbps": 8000,
                    "duration": 45.0,
                },
            ),
            patch("gomoveshift.pipeline.ost_applier.Path.exists", return_value=True),
            patch_ost_run() as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stderr=b"")
            ost_applier.apply_ost(
                input_path=in_mp4,
                output_path=out_mp4,
                location="Sample",
                layout="short",
                brand_logo_path=str(logo),
            )
        cmd = mock_run.call_args[0][0]
        # Shorts: no logo input, no filter_complex (keep the simple -vf path)
        i_flags = [i for i, a in enumerate(cmd) if a == "-i"]
        assert len(i_flags) == 1
