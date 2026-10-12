"""Tests for the OST title full-montage crossfade and the brand-text montage gate.

Behavior changes (regular_lf only — shorts unchanged):
  * LOCATION drawtext is visible from t=0 through the end of the montage
    (``enable='between(t,0,M)'``), then crossfades linearly to 0 over
    ``ost_title_crossfade_sec`` as the main video begins.
  * BRAND drawtext shares the SAME ``enable=between(t,0,M)`` gate AND
    the same alpha crossfade-out as the LOCATION text (the earlier
    persistent-full-video brand text was dropped because it
    burned center-screen brand text on top of every video frame, a
    content-integrity regression).
  * LAYOUTS["regular_lf"] legacy ``location_start_sec`` /
    ``location_duration_sec`` fields are DEPRECATED (ignored when
    ``montage_duration`` is threaded in).

Single-source-of-truth invariant: this behaviour does
NOT change what text is rendered — ``state["per_lf_ost_text"]`` remains
authoritative. Only the timing / opacity window changes.
"""

from __future__ import annotations

import re
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


# ---------------------------------------------------------------------------
# Helpers for parsing the emitted drawtext filter string
# ---------------------------------------------------------------------------


def _find_location_entry(chain: str) -> str:
    """Return the drawtext entry rendering the LOCATION text."""
    entries = chain.split("drawtext=")
    for e in entries:
        if f"y=h*{ost_applier._LOCATION_Y_FRAC}" in e:
            return e
    raise AssertionError(f"LOCATION drawtext entry not found in chain:\n{chain}")


def _find_brand_entry(chain: str) -> str:
    """Return the drawtext entry rendering the BRAND text."""
    entries = chain.split("drawtext=")
    for e in entries:
        if f"y=h*{ost_applier._BRAND_Y_FRAC}" in e:
            return e
    raise AssertionError(f"BRAND drawtext entry not found in chain:\n{chain}")


def _extract_alpha_expr(entry: str) -> str:
    """Pull the alpha='...' value out of a drawtext entry, or '' if none."""
    m = re.search(r"alpha='([^']*)'", entry)
    return m.group(1) if m else ""


def _extract_enable_expr(entry: str) -> str:
    """Pull the enable='...' value out of a drawtext entry, or '' if none."""
    m = re.search(r"enable='([^']*)'", entry)
    return m.group(1) if m else ""


def _eval_alpha_at(expr: str, t: float) -> float:
    """Evaluate the LOCATION crossfade alpha expression at time ``t``.

    The expected expression is of the form::

        if(lt(t,M-C),1,if(lt(t,M),1-(t-(M-C))/C,0))

    We parse out M and C from the expression and evaluate directly —
    independent of any FFmpeg runtime.
    """
    m = re.search(
        r"if\(lt\(t,([\d.]+)\),1,if\(lt\(t,([\d.]+)\),1-\(t-\(([\d.]+)\)\)/([\d.]+),0\)\)",
        expr,
    )
    assert m is not None, f"crossfade alpha expression did not match expected form: {expr}"
    fade_start = float(m.group(1))
    montage_end = float(m.group(2))
    crossfade = float(m.group(4))
    if t < fade_start:
        return 1.0
    if t < montage_end:
        return 1.0 - (t - fade_start) / crossfade
    return 0.0


# ---------------------------------------------------------------------------
# LOCATION alpha / timing
# ---------------------------------------------------------------------------


class TestLocationAlphaFullMontageCrossfade:
    """LOCATION title visible at full opacity through M-C, then linear fade."""

    def _chain(self, montage_duration: float = 15.0, crossfade: float = 0.5) -> str:
        layout = ost_applier.LAYOUTS["regular_lf"]
        return ost_applier._build_drawtext_filter(
            location="Sample Location",
            layout=layout,
            font_path=ost_applier.FONT_PATH,
            video_width=3840,
            video_duration=817.0,
            montage_duration=montage_duration,
            crossfade_sec=crossfade,
        )

    def test_ost_location_alpha_full_during_montage(self):
        chain = self._chain(montage_duration=15.0, crossfade=0.5)
        loc = _find_location_entry(chain)
        alpha = _extract_alpha_expr(loc)
        assert _eval_alpha_at(alpha, 0.0) == pytest.approx(1.0)
        assert _eval_alpha_at(alpha, 7.0) == pytest.approx(1.0)
        assert _eval_alpha_at(alpha, 14.4) == pytest.approx(1.0)

    def test_ost_location_alpha_linear_crossfade(self):
        chain = self._chain(montage_duration=15.0, crossfade=0.5)
        loc = _find_location_entry(chain)
        alpha = _extract_alpha_expr(loc)
        assert _eval_alpha_at(alpha, 14.75) == pytest.approx(0.5, abs=1e-6)

    def test_ost_location_alpha_zero_after_montage(self):
        chain = self._chain(montage_duration=15.0, crossfade=0.5)
        loc = _find_location_entry(chain)
        alpha = _extract_alpha_expr(loc)
        assert _eval_alpha_at(alpha, 15.5) == pytest.approx(0.0)
        assert _eval_alpha_at(alpha, 100.0) == pytest.approx(0.0)

    def test_ost_location_timing_uses_montage_duration(self):
        """LOCATION enable window is [0, montage_duration]."""
        chain = self._chain(montage_duration=15.0, crossfade=0.5)
        loc = _find_location_entry(chain)
        enable = _extract_enable_expr(loc)
        assert enable == "between(t,0,15.0)", f"got enable={enable!r}"

    def test_ost_location_timing_honors_overridden_montage_duration(self):
        chain = self._chain(montage_duration=10.0, crossfade=0.5)
        loc = _find_location_entry(chain)
        assert _extract_enable_expr(loc) == "between(t,0,10.0)"

    def test_ost_crossfade_duration_configurable(self):
        chain = self._chain(montage_duration=15.0, crossfade=1.25)
        loc = _find_location_entry(chain)
        alpha = _extract_alpha_expr(loc)
        assert "13.75" in alpha, f"alpha missing fade-start M-C=13.75: {alpha}"
        assert "1.25" in alpha, f"alpha missing crossfade value 1.25: {alpha}"


# ---------------------------------------------------------------------------
# BRAND persistent full-video
# ---------------------------------------------------------------------------


class TestBrandRespectsMontageGate:
    """BRAND drawtext shares montage gate + crossfade with LOCATION.

    There is no persistent full-video brand text. Both texts
    visible only during montage intro; both fade together at end of M.
    """

    def _chain(self, montage_duration: float = 15.0, crossfade: float = 0.5) -> str:
        layout = ost_applier.LAYOUTS["regular_lf"]
        return ost_applier._build_drawtext_filter(
            location="Sample Location",
            layout=layout,
            font_path=ost_applier.FONT_PATH,
            video_width=3840,
            video_duration=817.0,
            montage_duration=montage_duration,
            crossfade_sec=crossfade,
        )

    def test_ost_brand_has_enable_gate_matching_location(self):
        """BRAND must have enable=between(t,0,M), the same as LOCATION."""
        chain = self._chain(montage_duration=15.0, crossfade=0.5)
        brand = _find_brand_entry(chain)
        loc = _find_location_entry(chain)
        brand_enable = _extract_enable_expr(brand)
        loc_enable = _extract_enable_expr(loc)
        assert brand_enable == "between(t,0,15.0)", (
            f"BRAND enable must gate to montage; got {brand_enable!r}"
        )
        assert brand_enable == loc_enable, (
            f"BRAND and LOCATION must share enable window; "
            f"brand={brand_enable!r} loc={loc_enable!r}"
        )

    def test_ost_brand_alpha_zero_after_montage(self):
        """For t > montage_duration, BRAND alpha must be 0 (core of the brand-gate behaviour)."""
        chain = self._chain(montage_duration=15.0, crossfade=0.5)
        brand = _find_brand_entry(chain)
        alpha = _extract_alpha_expr(brand)
        assert alpha, f"BRAND must have an alpha expression that fades to 0; got entry:\n{brand}"
        assert _eval_alpha_at(alpha, 15.5) == pytest.approx(0.0)
        assert _eval_alpha_at(alpha, 100.0) == pytest.approx(0.0)
        assert _eval_alpha_at(alpha, 817.0) == pytest.approx(0.0)

    def test_ost_brand_alpha_full_during_montage(self):
        chain = self._chain(montage_duration=15.0, crossfade=0.5)
        brand = _find_brand_entry(chain)
        alpha = _extract_alpha_expr(brand)
        assert _eval_alpha_at(alpha, 0.0) == pytest.approx(1.0)
        assert _eval_alpha_at(alpha, 7.0) == pytest.approx(1.0)
        assert _eval_alpha_at(alpha, 14.4) == pytest.approx(1.0)

    def test_ost_brand_alpha_crossfades_with_location(self):
        """BRAND and LOCATION must use the same alpha expression."""
        chain = self._chain(montage_duration=15.0, crossfade=0.5)
        brand_alpha = _extract_alpha_expr(_find_brand_entry(chain))
        loc_alpha = _extract_alpha_expr(_find_location_entry(chain))
        assert brand_alpha == loc_alpha, (
            f"BRAND alpha must match LOCATION alpha (shared crossfade); "
            f"brand={brand_alpha!r} loc={loc_alpha!r}"
        )


# ---------------------------------------------------------------------------
# Shorts layout untouched
# ---------------------------------------------------------------------------


class TestShortsLayoutUnchanged:
    """Shorts branch of _build_drawtext_filter emits identical output pre/post."""

    def test_regular_lf_shorts_unchanged(self):
        """Shorts path — no montage_duration threading — uses the live-aligned
        enable='between(t,0.0,5.0)' window for BRAND + LOCATION (the
        short layout was realigned from 0.5-8s to the live
        build_ost_filter window t∈[0,5])."""
        layout = ost_applier.LAYOUTS["short"]
        chain = ost_applier._build_drawtext_filter(
            location="Sample",
            layout=layout,
            font_path=ost_applier.FONT_PATH,
            video_width=1080,
            video_duration=45.0,
        )
        assert "between(t,0.0,5.0)" in chain, (
            f"shorts LOCATION enable changed — expected between(t,0.0,5.0) in chain:\n{chain}"
        )
        assert chain.count("between(t,0.0,5.0)") >= 2, (
            f"shorts BRAND + LOCATION should both share the live-aligned window; got:\n{chain}"
        )


# ---------------------------------------------------------------------------
# apply_ost() threads montage_duration + crossfade
# ---------------------------------------------------------------------------


class TestApplyOstThreadsMontageDuration:
    """apply_ost() accepts montage_duration + crossfade_sec kwargs and
    threads them into _build_drawtext_filter."""

    def _stub_probe(self):
        return {
            "width": 3840,
            "height": 2160,
            "fps": 29.97,
            "bitrate_kbps": 52000,
            "duration": 817.0,
        }

    def _make_input(self, tmp_path):
        p = tmp_path / "in.mp4"
        p.write_bytes(b"\x00" * 1024)
        return p

    def test_apply_ost_accepts_montage_duration_kwarg(self):
        import inspect

        sig = inspect.signature(ost_applier.apply_ost)
        assert "montage_duration" in sig.parameters
        assert "crossfade_sec" in sig.parameters

    def test_apply_ost_threads_montage_duration_into_filter(self, tmp_path):
        in_mp4 = self._make_input(tmp_path)
        out_mp4 = tmp_path / "out.mp4"

        with (
            patch(
                "gomoveshift.pipeline.ost_applier._probe_video_info",
                return_value=self._stub_probe(),
            ),
            patch("gomoveshift.pipeline.ost_applier.Path.exists", return_value=True),
            patch_ost_run() as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stderr=b"")
            ost_applier.apply_ost(
                input_path=in_mp4,
                output_path=out_mp4,
                location="Sample",
                layout="regular_lf",
                montage_duration=12.5,
                crossfade_sec=0.75,
            )

        cmd = mock_run.call_args[0][0]
        vf_idx = cmd.index("-vf")
        vfilter = cmd[vf_idx + 1]
        assert "between(t,0,12.5)" in vfilter, (
            f"expected LOCATION enable=between(t,0,12.5); got:\n{vfilter}"
        )
        assert "11.75" in vfilter and "0.75" in vfilter, (
            f"expected crossfade fade_start=11.75 and C=0.75; got:\n{vfilter}"
        )
