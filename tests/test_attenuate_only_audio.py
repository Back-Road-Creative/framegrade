"""Attenuate-only audio contract.

New invariants for the near-silent audio policy:

  * ``loudnorm_measure.build_attenuate_filter`` computes a single non-positive
    ``volume=`` gain — it can NEVER amplify. Source already below the ceiling
    is left untouched (gain 0); a louder source is pulled DOWN to the
    ``AUDIO_SILENCE_CEILING_LUFS`` integrated ceiling (tightened further by the
    ``AUDIO_TRUE_PEAK_CAP`` when needed).
  * ``single_pass._build_audio_filter`` starts with ``volume=``, contains no
    ``loudnorm``, and its defensive invariant raises if the helper ever returns
    a loudnorm (boost) filter.
  * ``shorts_extractor.render_short_ffmpeg`` renders MUTED (`-an`, no `-af`,
    no `-c:a`).
"""

from __future__ import annotations

import re

import pytest

from gomoveshift.pipeline import encoding_constants as ec
from gomoveshift.pipeline import single_pass as sp_mod
from gomoveshift.pipeline.loudnorm_measure import build_attenuate_filter
from gomoveshift.pipeline.single_pass import SinglePassProcessor

_VOLUME_RE = re.compile(r"^volume=(-?\d+(?:\.\d+)?)dB$")


def _gain_db(af: str) -> float:
    m = _VOLUME_RE.match(af)
    assert m, f"filter is not a bare volume= gain: {af!r}"
    return float(m.group(1))


# ── build_attenuate_filter: never amplifies ────────────────────────────────


def test_never_returns_positive_gain_across_sweep():
    """A wide sweep of measured integrated loudness / true peak must never
    yield an amplifying (positive) gain."""
    for input_i in range(-70, -4, 5):
        for input_tp in range(-40, 1, 5):
            af = build_attenuate_filter({"input_i": input_i, "input_tp": input_tp})
            assert _gain_db(af) <= 0.0, (
                f"positive gain at input_i={input_i}, input_tp={input_tp}: {af!r}"
            )


def test_quieter_than_ceiling_is_not_amplified():
    """Source already below the near-silent ceiling stays untouched (gain 0)."""
    af = build_attenuate_filter({"input_i": -50.0, "input_tp": -30.0})
    assert _gain_db(af) == 0.0, af


def test_louder_than_ceiling_is_attenuated_to_ceiling():
    """A source louder than the ceiling is attenuated so integrated lands at
    the ceiling — gain == ceiling - input_i, unless the TP cap tightens it."""
    input_i = -20.0
    # input_tp chosen so the integrated-reduction term dominates (TP does not
    # tighten): ceiling-input_i = -25 is more negative than tp_cap-input_tp.
    af = build_attenuate_filter({"input_i": input_i, "input_tp": -8.0})
    expected = ec.AUDIO_SILENCE_CEILING_LUFS - input_i  # -45 - (-20) = -25
    assert _gain_db(af) == pytest.approx(expected), af


def test_true_peak_tightens_the_gain_further():
    """When the true peak is hot, the gain must tighten beyond the integrated
    reduction so output TP lands at/under the cap."""
    # input_i=-30 → integrated term = -45-(-30) = -15. input_tp=-2 → tp term =
    # -10-(-2) = -8 → the integrated term (-15) is MORE negative, so gain=-15.
    af = build_attenuate_filter({"input_i": -30.0, "input_tp": -2.0})
    assert _gain_db(af) == pytest.approx(-15.0), af


def test_specific_quiet_source_value():
    af = build_attenuate_filter({"input_i": -30.0, "input_tp": -12.0})
    assert af == "volume=-15.00dB", af


def test_specific_loud_source_value():
    af = build_attenuate_filter({"input_i": -15.0, "input_tp": -2.0})
    assert af == "volume=-30.00dB", af


def test_already_below_ceiling_value():
    af = build_attenuate_filter({"input_i": -60.0, "input_tp": -40.0})
    assert af == "volume=0.00dB", af


def test_no_measured_uses_fixed_fallback():
    assert build_attenuate_filter(None) == "volume=-25.00dB"
    assert build_attenuate_filter({}) == "volume=-25.00dB"


# ── single_pass._build_audio_filter ────────────────────────────────────────


def _make_processor() -> SinglePassProcessor:
    return SinglePassProcessor.__new__(SinglePassProcessor)


def test_build_audio_filter_is_volume_only():
    proc = _make_processor()
    af = proc._build_audio_filter({"input_i": "-30.0", "input_tp": "-12.0"})
    assert af.startswith("volume=")
    assert "loudnorm" not in af


def test_build_audio_filter_raises_on_injected_loudnorm(monkeypatch):
    proc = _make_processor()
    monkeypatch.setattr(
        sp_mod,
        "build_attenuate_filter",
        lambda measured: "loudnorm=I=-20:TP=-1.5:LRA=11:measured_I=-30",
    )
    with pytest.raises(RuntimeError, match="loudnorm"):
        proc._build_audio_filter({"input_i": "-30.0"})


# ── shorts render is muted ─────────────────────────────────────────────────


def test_shorts_render_cmd_is_muted(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    from gomoveshift.pipeline import shorts_extractor

    source = tmp_path / "master.mp4"
    source.touch()
    out = tmp_path / "short_01.mp4"
    captured: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        captured.append(list(cmd))
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        out.write_bytes(b"x" * 4096)
        return result

    monkeypatch.setattr(shorts_extractor.subprocess, "run", fake_run)
    monkeypatch.setattr(
        "gomoveshift.pipeline.ffmpeg_runner.validate_video",
        lambda *a, **k: MagicMock(valid=True, reason="", duration=30.0),
    )

    ok = shorts_extractor.render_short_ffmpeg(
        source_video=source,
        output_path=out,
        start_time=42.0,
        duration=30.0,
        location="Beaufort, SC",
        src_width=3840,
        src_height=2160,
        use_nvenc=True,
    )
    assert ok is True

    encode_cmd = next((c for c in captured if "hevc_nvenc" in c or "-c:v" in c), None)
    assert encode_cmd is not None, captured
    assert "-an" in encode_cmd
    assert "-af" not in encode_cmd
    assert "-c:a" not in encode_cmd
