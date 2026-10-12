"""Shared two-pass loudnorm measurement primitives.

The pass-1 loudnorm *measurement* (audio-only decode that prints the
``input_i/input_tp/input_lra/input_thresh`` stats as JSON on stderr) was
duplicated verbatim in ``ost_applier._measure_audio_loudnorm`` and
``single_pass.SinglePassProcessor._measure_audio``: identical filter +
command construction, identical regex extraction, identical JSON-parse /
empty-dict fallback. The two sites differ ONLY in their subprocess wrapper
(raw ``subprocess.run`` + a fixed timeout vs ``run_ffmpeg_quick`` + a
computed timeout) and in their warning loggers/wording — so the subprocess
call and the warning emission stay at each call site (preserving per-module
logger names and exact messages), while the mechanical, byte-for-byte
identical parts live here.

``build_loudnorm_measure_cmd`` builds the audio-only ffmpeg command;
``parse_loudnorm_stats`` extracts the stats dict from ffmpeg's stderr.
Together they own the "what command / how to parse" contract; each caller
owns the "how to run it / how to log a degrade" contract.
"""

import json
import re
from pathlib import Path
from typing import Optional

# loudnorm pass-1 stats block as ffmpeg emits it to stderr (a JSON object whose
# first key is "input_i"). Matched non-greedily up to the first closing brace.
_LOUDNORM_STATS_RE = re.compile(r'\{\s*\n\s*"input_i".*?\}', re.DOTALL)

# ffmpeg's loudnorm filter only accepts TP in [-9, 0]; our validation ceiling
# (encoding_constants.LONGFORM_AUDIO_TP / AUDIO_TRUE_PEAK_CAP) is -10, which
# ffmpeg rejects outright. Clamp ONLY the filter argument here — the -10
# ceiling itself stays frozen and is enforced elsewhere post-measurement.
FFMPEG_LOUDNORM_TP_MIN = -9.0


def build_loudnorm_measure_cmd(input_path: Path) -> list[str]:
    """Build the audio-only loudnorm measurement command for ``input_path``.

    Decodes audio only (``-vn``), runs ``loudnorm=...:print_format=json`` so
    ffmpeg prints the pass-1 stats to stderr, and discards output to null.
    """
    from . import encoding_constants as ec

    measure_tp = max(ec.LONGFORM_AUDIO_TP, FFMPEG_LOUDNORM_TP_MIN)
    af = (
        f"loudnorm=I={ec.LONGFORM_AUDIO_LUFS}"
        f":TP={measure_tp}"
        f":LRA={ec.LONGFORM_AUDIO_LRA}"
        f":print_format=json"
    )
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-af",
        af,
        "-f",
        "null",
        "-",
    ]


# Conservative fixed attenuation applied when a source loudness measurement is
# unavailable. Typical cab audio sits at roughly -20..-35 LUFS integrated, so a
# -25 dB reduction reliably lands it below the -45 LUFS ceiling. Never a boost.
_FALLBACK_ATTENUATION_DB = -25.0


def _safe_float(measured: Optional[dict], key: str) -> Optional[float]:
    if not measured or key not in measured:
        return None
    try:
        return float(measured[key])
    except (TypeError, ValueError):
        return None


def build_attenuate_filter(measured: Optional[dict]) -> str:
    """Build an ATTENUATE-ONLY ``-af`` filter that can never amplify audio.

    Standing rule: every published video must be practically silent —
    conversation/songs must not be interpretable at max volume. The prior
    two-pass ``loudnorm=I=-20`` boosted quiet cab audio *up* to target, which
    can make speech audible. This builder instead computes a single,
    non-positive ``volume=`` gain that pulls the measured source loudness DOWN to the
    near-silent integrated ceiling (``AUDIO_SILENCE_CEILING_LUFS``) and also
    honours the true-peak cap (``AUDIO_TRUE_PEAK_CAP``). Source already quieter
    than the ceiling is left untouched (gain 0). Positive gain is impossible by
    construction and asserted below (robust by construction).

    ``measured`` is the pass-1 ``parse_loudnorm_stats`` dict (``input_i`` /
    ``input_tp``). When absent (measurement failed), a conservative fixed
    attenuation is applied instead of any boost.
    """
    from . import encoding_constants as ec

    ceiling = ec.AUDIO_SILENCE_CEILING_LUFS
    tp_cap = ec.AUDIO_TRUE_PEAK_CAP
    input_i = _safe_float(measured, "input_i")
    input_tp = _safe_float(measured, "input_tp")

    if input_i is None:
        gain_db = _FALLBACK_ATTENUATION_DB
    else:
        gain_db = min(0.0, ceiling - input_i)
        if input_tp is not None:
            # Ensure true peak also lands at/under the cap: after a uniform
            # ``volume`` gain, output TP ~= input_tp + gain. Tighten the gain if
            # the integrated-loudness reduction alone would leave TP too hot.
            gain_db = min(gain_db, tp_cap - input_tp)
        gain_db = min(0.0, gain_db)

    if gain_db > 0:
        raise RuntimeError(
            f"attenuate-only audio filter must never amplify (computed gain "
            f"{gain_db:+.2f} dB) — see build_attenuate_filter and the "
            "near-silent policy in encoding_constants.py"
        )
    return f"volume={gain_db:.2f}dB"


def parse_loudnorm_stats(stderr: str) -> Optional[dict]:
    """Extract the pass-1 loudnorm stats dict from ffmpeg ``stderr``.

    Returns the parsed ``input_i/input_tp/input_lra/input_thresh`` (+
    ``target_offset``) dict on success, or ``None`` when no JSON stats block
    is present in the stderr (caller should warn + degrade to single-pass).
    Raises ``json.JSONDecodeError`` when a block is present but malformed, so
    the caller can distinguish that failure mode in its own warning.
    """
    json_match = _LOUDNORM_STATS_RE.search(stderr)
    if not json_match:
        return None
    return json.loads(json_match.group())
