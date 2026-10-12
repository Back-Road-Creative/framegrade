"""Production day-master audio filter is attenuate-only (no loudnorm).

The day-master encode no longer runs any
``loudnorm`` normalization. The prior two-pass ``loudnorm=I=-20`` boosted quiet
cab audio UP to a loudness target, which makes speech
audible. ``_build_audio_filter`` now emits a single non-positive
``volume=`` gain (see build_attenuate_filter), so the production filter must
contain no ``loudnorm`` (and, a fortiori, no ``linear=true``) at all.

Since the day master runs ONE audio pass and every regular LF is
stream-copied out of it, this single filter governs every LF's loudness — so
"no boost here" is what keeps every LF near-silent.
"""

from __future__ import annotations

import re

from gomoveshift.pipeline.single_pass import SinglePassProcessor

_VOLUME_RE = re.compile(r"^volume=(-?\d+(?:\.\d+)?)dB$")


def _make_processor() -> SinglePassProcessor:
    # ``_build_audio_filter`` reads encoding_constants only, not config/NVENC.
    return SinglePassProcessor.__new__(SinglePassProcessor)


def test_production_filter_omits_loudnorm():
    """The production day-master filter shape must contain no ``loudnorm``
    (and no ``linear=true``) — it is a bare attenuate-only ``volume=`` gain.
    """
    proc = _make_processor()
    measured = {
        "input_i": "-13.27",
        "input_tp": "-3.24",
        "input_lra": "11.70",
        "input_thresh": "-23.60",
        "target_offset": "0.70",
    }
    af = proc._build_audio_filter(measured)
    assert "loudnorm" not in af, af
    assert "linear=" not in af, af
    assert af.startswith("volume="), af


def test_production_filter_never_boosts():
    """Across a spread of measured source loudness the production filter is
    always a non-positive gain — it can never normalize quiet audio UP.
    """
    proc = _make_processor()
    for input_i in (-8.0, -14.0, -20.0, -31.2, -50.0, -70.0):
        af = proc._build_audio_filter({"input_i": str(input_i), "input_tp": "-3.0"})
        m = _VOLUME_RE.match(af)
        assert m, af
        assert float(m.group(1)) <= 0.0, f"amplifying gain for input_i={input_i}: {af!r}"
