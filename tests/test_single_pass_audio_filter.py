"""Single-pass audio filter construction — attenuate-only contract.

``_build_audio_filter`` no longer emits a two-pass ``loudnorm=I=-20`` boost (which pulled quiet cab audio UP to a
loudness target and shipped audible speech). It now returns
``build_attenuate_filter(measured)`` — a single non-positive ``volume=`` gain
that can only pull loudness DOWN toward the near-silent ceiling.

The contract these tests pin down:

1. The returned string starts with ``volume=`` — never ``loudnorm=`` or any
   other filter.
2. The string contains no ``loudnorm`` and never a positive (amplifying) gain.
3. The defensive invariant in ``_build_audio_filter`` raises if a future
   refactor makes the helper return a ``loudnorm`` (boost) filter, or any
   non-``volume=`` filter.
"""

from __future__ import annotations

import re

import pytest

from gomoveshift.pipeline import single_pass as sp_mod
from gomoveshift.pipeline.single_pass import SinglePassProcessor

_VOLUME_RE = re.compile(r"^volume=(-?\d+(?:\.\d+)?)dB$")


def _make_processor() -> SinglePassProcessor:
    # ``_build_audio_filter`` only delegates to build_attenuate_filter (reads
    # encoding_constants, not config / NVENC), so bypass __init__ entirely.
    return SinglePassProcessor.__new__(SinglePassProcessor)


def _gain_db(af: str) -> float:
    m = _VOLUME_RE.match(af)
    assert m, f"filter is not a bare volume= gain: {af!r}"
    return float(m.group(1))


class TestAudioFilterAttenuateOnlyContract:
    """``_build_audio_filter`` must emit an attenuate-only ``volume=`` gain."""

    def test_two_pass_form_starts_with_volume(self):
        proc = _make_processor()
        measured = {
            "input_i": "-14.0",
            "input_tp": "-1.5",
            "input_lra": "9.3",
            "input_thresh": "-34.46",
            "target_offset": "-0.26",
        }
        af = proc._build_audio_filter(measured)
        assert af.startswith("volume="), af

    def test_filter_contains_no_loudnorm(self):
        """The old boost form must never reappear."""
        proc = _make_processor()
        measured = {"input_i": "-14.0", "input_tp": "-1.5"}
        af = proc._build_audio_filter(measured)
        assert "loudnorm" not in af, af

    def test_filter_never_amplifies(self):
        proc = _make_processor()
        # A loud source (louder than the ceiling) is attenuated (negative gain).
        af = proc._build_audio_filter({"input_i": "-14.0", "input_tp": "-1.5"})
        assert _gain_db(af) <= 0.0, af

    def test_quiet_source_left_untouched(self):
        """Source already below the near-silent ceiling is not boosted up."""
        proc = _make_processor()
        af = proc._build_audio_filter({"input_i": "-60.0", "input_tp": "-40.0"})
        assert _gain_db(af) == 0.0, af

    def test_empty_measured_falls_back_to_fixed_attenuation(self):
        proc = _make_processor()
        af = proc._build_audio_filter({})
        assert af.startswith("volume=")
        assert "loudnorm" not in af
        assert _gain_db(af) <= 0.0, af

    def test_partial_measured_missing_input_i_still_attenuate_only(self):
        proc = _make_processor()
        af = proc._build_audio_filter({"input_tp": "-1.5"})
        assert af.startswith("volume=")
        assert "loudnorm" not in af
        assert _gain_db(af) <= 0.0, af

    def test_no_whitespace_in_filter(self):
        proc = _make_processor()
        af = proc._build_audio_filter({"input_i": "-14.0", "input_tp": "-1.5"})
        assert " " not in af
        assert "\t" not in af
        assert "\n" not in af


class TestAudioFilterDefensiveInvariant:
    """The build-time guards must trip if the helper ever returns a boost."""

    def test_invariant_raises_if_loudnorm_injected(self, monkeypatch):
        """A future refactor that reintroduces a loudnorm boost via the helper
        must trip the ``no loudnorm`` invariant at filter-build time."""
        proc = _make_processor()
        monkeypatch.setattr(
            sp_mod,
            "build_attenuate_filter",
            lambda measured: "loudnorm=I=-20:TP=-1.5:LRA=11",
        )
        with pytest.raises(RuntimeError, match="loudnorm"):
            proc._build_audio_filter({"input_i": "-14.0"})

    def test_invariant_raises_if_non_volume_filter_injected(self, monkeypatch):
        """Any non-``volume=`` filter (that isn't loudnorm) must also trip the
        ``must be a 'volume=' gain`` invariant."""
        proc = _make_processor()
        monkeypatch.setattr(
            sp_mod,
            "build_attenuate_filter",
            lambda measured: "aresample=async=1000",
        )
        with pytest.raises(RuntimeError, match="volume="):
            proc._build_audio_filter({"input_i": "-14.0"})
