"""``build_loudnorm_measure_cmd`` must emit an ffmpeg-legal TP value.

ffmpeg's ``loudnorm`` filter only accepts ``TP`` in ``[-9, 0]``. The frozen
validation ceiling ``encoding_constants.LONGFORM_AUDIO_TP`` is ``-10`` (out of
that range), so splicing it straight into the filter string makes the pass-1
measurement command fail every single time, silently degrading every
long-form to single-pass loudness. The command builder must clamp the filter
argument to ffmpeg's legal range without touching the validation ceiling.
"""

from __future__ import annotations

import re
from pathlib import Path

from gomoveshift.pipeline import encoding_constants as ec
from gomoveshift.pipeline.loudnorm_measure import build_loudnorm_measure_cmd

_TP_RE = re.compile(r":TP=(-?\d+(?:\.\d+)?)")


def test_measure_cmd_tp_is_ffmpeg_legal():
    cmd = build_loudnorm_measure_cmd(Path("input.mp4"))
    af = next(arg for arg in cmd if arg.startswith("loudnorm="))
    m = _TP_RE.search(af)
    assert m, af
    tp = float(m.group(1))
    assert -9.0 <= tp <= 0.0, f"loudnorm TP={tp} is outside ffmpeg's legal [-9, 0] range: {af!r}"


def test_validation_ceiling_constant_untouched():
    # The -10 attenuation/validation ceiling used elsewhere in the pipeline
    # must stay frozen at -10 — only the filter argument in the measure
    # command builder is clamped.
    assert ec.LONGFORM_AUDIO_TP == -10.0
    assert ec.AUDIO_TRUE_PEAK_CAP == -10.0
