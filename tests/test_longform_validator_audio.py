"""
Longform audio-loudness validator — ceiling-only contract.

The long-form audio gate is CEILING-ONLY.
Under the near-silent policy every published LF is attenuate-only, so the only
defect is being too LOUD. The gate passes iff:

  * integrated loudness ``measured_i <= max_lufs`` (default -43 LUFS), AND
  * true peak ``measured_tp <= max_true_peak + 0.1`` (default max_true_peak -10 dBTP)

There is NO lower bound: a very QUIET output (e.g. -60 LUFS) is the GOAL and
passes. The prior two-sided window failed too-quiet audio and thereby demanded
the -20 LUFS boost that made speech audible; it is gone.
"""

from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import MagicMock

from gomoveshift.pipeline import encoding_constants as ec
from gomoveshift.pipeline.validation.longform_validator import LongformValidator


def _loudnorm_stderr(input_i: float, input_tp: float = -12.0) -> str:
    return (
        "\n"
        "{\n"
        f'        "input_i" : "{input_i}",\n'
        f'        "input_tp" : "{input_tp}",\n'
        '        "input_lra" : "9.30",\n'
        '        "input_thresh" : "-54.46",\n'
        '        "output_i" : "-45.00",\n'
        '        "output_tp" : "-12.00",\n'
        '        "output_lra" : "6.40",\n'
        '        "output_thresh" : "-55.74",\n'
        '        "normalization_type" : "dynamic",\n'
        '        "target_offset" : "-0.26"\n'
        "}\n"
    )


def _make_validator(**overrides) -> tuple[LongformValidator, MagicMock]:
    subproc_mod = MagicMock()
    ffprobe = MagicMock()
    ffprobe._subprocess = subproc_mod
    # Default probed duration for a typical short LF — keeps every existing
    # test's timeout at the flat 300s floor unless a test overrides it.
    ffprobe.probe_duration.return_value = 120.0
    return LongformValidator(ffprobe, **overrides), subproc_mod


def _set(subproc_mod, input_i, input_tp=-12.0):
    subproc_mod.run.return_value = CompletedProcess(
        args=[], returncode=0, stdout="", stderr=_loudnorm_stderr(input_i, input_tp)
    )


class TestLongformAudioLoudnessCeiling:
    """check_audio_loudness must enforce ONLY an upper ceiling (no lower bound)."""

    def test_at_ceiling_passes(self):
        v, subproc_mod = _make_validator()
        _set(subproc_mod, input_i=-43.0)  # exactly at the -43 ceiling
        result = v.check_audio_loudness(Path("/tmp/lf.mp4"))
        assert result.passed, result.message

    def test_far_below_ceiling_passes(self):
        """A very quiet output is the GOAL — it must PASS (no lower bound)."""
        v, subproc_mod = _make_validator()
        _set(subproc_mod, input_i=-60.0, input_tp=-40.0)
        result = v.check_audio_loudness(Path("/tmp/lf.mp4"))
        assert result.passed, result.message

    def test_loud_output_fails(self):
        """A loud output (e.g. -20 LUFS) must FAIL — it is LOUDER than ceiling."""
        v, subproc_mod = _make_validator()
        _set(subproc_mod, input_i=-20.0)
        result = v.check_audio_loudness(Path("/tmp/lf.mp4"))
        assert not result.passed
        assert result.message.startswith("Audio too loud"), result.message
        assert "-20" in result.message

    def test_just_above_ceiling_fails(self):
        v, subproc_mod = _make_validator()
        _set(subproc_mod, input_i=-42.9)  # louder than -43 by 0.1
        result = v.check_audio_loudness(Path("/tmp/lf.mp4"))
        assert not result.passed
        assert result.message.startswith("Audio too loud"), result.message

    def test_defaults_come_from_encoding_constants(self):
        v, _ = _make_validator()
        assert v.max_lufs == ec.LONGFORM_AUDIO_MAX_LUFS  # -43.0
        assert v.max_true_peak == ec.LONGFORM_AUDIO_TP  # -10.0

    def test_peak_cap_enforced_at_minus_ten(self):
        """Integrated may be quiet, but a hot true peak still fails at -10 dBTP."""
        v, subproc_mod = _make_validator()
        _set(subproc_mod, input_i=-50.0, input_tp=-5.0)  # -5 dBTP > -10 cap
        result = v.check_audio_loudness(Path("/tmp/lf.mp4"))
        assert not result.passed
        assert "true peak" in result.message.lower()
        assert result.message.startswith("Audio too loud"), result.message

    def test_peak_within_tolerance_passes(self):
        """0.1 dB measurement tolerance absorbs ISP/float jitter at the cap."""
        v, subproc_mod = _make_validator()
        _set(subproc_mod, input_i=-50.0, input_tp=-9.95)  # within -10 + 0.1
        result = v.check_audio_loudness(Path("/tmp/lf.mp4"))
        assert result.passed, result.message

    def test_peak_beyond_tolerance_fails(self):
        v, subproc_mod = _make_validator()
        _set(subproc_mod, input_i=-50.0, input_tp=-9.8)  # beyond -10 + 0.1
        result = v.check_audio_loudness(Path("/tmp/lf.mp4"))
        assert not result.passed
        assert "true peak" in result.message.lower()

    def test_threshold_field_describes_ceiling(self):
        v, subproc_mod = _make_validator()
        _set(subproc_mod, input_i=-45.0)
        result = v.check_audio_loudness(Path("/tmp/lf.mp4"))
        threshold = result.threshold or ""
        assert "-43" in threshold
        assert "ceiling" in threshold.lower()

    def test_gate_name_unchanged(self):
        v, subproc_mod = _make_validator()
        _set(subproc_mod, input_i=-45.0)
        result = v.check_audio_loudness(Path("/tmp/lf.mp4"))
        assert result.gate == "audio_loudness"


def _ebur128_stderr(momentary_values: list[float]) -> str:
    """Fake ffmpeg ebur128 stderr: one per-frame status line per M value, plus a
    pre-roll gate-floor line the gate must ignore."""
    lines = [
        "[Parsed_ebur128_0 @ 0x1] t: 0.0   M: -120.7 S: -120.7 I: -70.0 LUFS  LRA: 0.0 LU",
    ]
    for i, m in enumerate(momentary_values):
        lines.append(
            f"[Parsed_ebur128_0 @ 0x1] t: {i + 1}.0   "
            f"M: {m:.1f} S: -50.0 I: -55.0 LUFS  LRA: 3.0 LU  FTPK: -30.0 dBFS"
        )
    return "\n".join(lines) + "\n"


class TestLongformMomentaryLoudnessBelt:
    """check_momentary_loudness: the speech-intelligibility belt rejects a file
    whose loudest 400 ms window stays interpretable, even when the whole-file
    integrated ceiling passes."""

    def _set_ebur128(self, subproc_mod, momentary_values):
        subproc_mod.run.return_value = CompletedProcess(
            args=[], returncode=0, stdout="", stderr=_ebur128_stderr(momentary_values)
        )

    def test_near_silent_momentary_passes(self):
        v, subproc_mod = _make_validator()
        # An honestly-attenuated LF: every momentary window sits far below -30.
        self._set_ebur128(subproc_mod, [-48.0, -50.0, -46.0, -52.0])
        result = v.check_momentary_loudness(Path("/tmp/lf.mp4"))
        assert result.passed, result.message
        assert result.gate == "audio_momentary"

    def test_loud_momentary_spike_fails(self):
        v, subproc_mod = _make_validator()
        # A loud speech word spikes momentary loudness even if the average is low.
        self._set_ebur128(subproc_mod, [-48.0, -12.0, -50.0, -46.0])
        result = v.check_momentary_loudness(Path("/tmp/lf.mp4"))
        assert not result.passed
        assert "momentary" in result.message.lower()
        assert "-12" in result.message

    def test_exactly_at_ceiling_passes(self):
        v, subproc_mod = _make_validator()
        self._set_ebur128(subproc_mod, [-40.0, ec.LONGFORM_AUDIO_MAX_MOMENTARY, -45.0])
        result = v.check_momentary_loudness(Path("/tmp/lf.mp4"))
        assert result.passed, result.message

    def test_just_above_ceiling_fails(self):
        v, subproc_mod = _make_validator()
        self._set_ebur128(subproc_mod, [-40.0, ec.LONGFORM_AUDIO_MAX_MOMENTARY + 0.5, -45.0])
        result = v.check_momentary_loudness(Path("/tmp/lf.mp4"))
        assert not result.passed

    def test_gate_floor_samples_ignored(self):
        """The -120 LUFS pre-roll gate floor must not count as the max."""
        v, subproc_mod = _make_validator()
        # Only gate-floor lines → treated as no usable samples → WARN-pass.
        subproc_mod.run.return_value = CompletedProcess(
            args=[], returncode=0, stdout="", stderr=_ebur128_stderr([-130.0, -125.0])
        )
        result = v.check_momentary_loudness(Path("/tmp/lf.mp4"))
        assert result.passed, result.message

    def test_no_samples_degrades_to_pass(self):
        """A parse failure (no M: samples) WARNs and passes — the integrated
        gate is the primary guard."""
        v, subproc_mod = _make_validator()
        subproc_mod.run.return_value = CompletedProcess(
            args=[], returncode=0, stdout="", stderr="no ebur128 output here"
        )
        result = v.check_momentary_loudness(Path("/tmp/lf.mp4"))
        assert result.passed, result.message

    def test_default_ceiling_from_encoding_constants(self):
        v, _ = _make_validator()
        assert v.max_momentary_lufs == ec.LONGFORM_AUDIO_MAX_MOMENTARY


class TestAudioScanTimeoutScaling:
    """Both audio-only scans (loudnorm measure, ebur128) must scale their
    ffmpeg timeout with the probed input duration — a multi-hour FINAL's
    audio-only decode can exceed the prior flat 300s."""

    def test_short_duration_keeps_the_300_floor(self):
        v, subproc_mod = _make_validator()
        v._ffprobe.probe_duration.return_value = 60.0
        _set(subproc_mod, input_i=-45.0)
        v.check_audio_loudness(Path("/tmp/lf.mp4"))
        _, kwargs = subproc_mod.run.call_args
        assert kwargs["timeout"] == 300

    def test_long_duration_scales_above_the_floor(self):
        v, subproc_mod = _make_validator()
        v._ffprobe.probe_duration.return_value = 14400.0  # 4h superlong part
        _set(subproc_mod, input_i=-45.0)
        v.check_audio_loudness(Path("/tmp/lf.mp4"))
        _, kwargs = subproc_mod.run.call_args
        assert kwargs["timeout"] == 14400

    def test_momentary_scan_also_scales(self):
        v, subproc_mod = _make_validator()
        v._ffprobe.probe_duration.return_value = 14400.0
        subproc_mod.run.return_value = CompletedProcess(
            args=[], returncode=0, stdout="", stderr=_ebur128_stderr([-48.0])
        )
        v.check_momentary_loudness(Path("/tmp/lf.mp4"))
        _, kwargs = subproc_mod.run.call_args
        assert kwargs["timeout"] == 14400


def _run(input_i: float, decoded_s: float | None, returncode: int = 0) -> CompletedProcess:
    progress = f"size=N/A time={_hms(decoded_s)} bitrate=N/A speed=20x\n" if decoded_s else ""
    return CompletedProcess(
        args=[], returncode=returncode, stdout="", stderr=progress + _loudnorm_stderr(input_i)
    )


def _hms(seconds: float) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:05.2f}"


class TestLoudnessMeasurementCoversTheFile:
    """r5 (job 341c78b6, 2026-10-20 15:40 ET) failed this gate at -42.9 LUFS on an LF that
    re-measures -45.0 on every rerun with the same code. loudnorm prints its JSON for whatever
    it decoded, so a decode cut short reads a partial-file loudness as the file's. A
    measurement that did not cover the audio, or whose ffmpeg exited nonzero, is re-taken
    once and never judged; the -43 ceiling is unchanged."""

    def test_short_decode_is_remeasured_and_the_full_reading_judged(self):
        v, subproc_mod = _make_validator()
        v._ffprobe.probe_duration.return_value = 4329.0
        subproc_mod.run.side_effect = [_run(-42.9, 2900.0), _run(-45.0, 4328.6)]
        result = v.check_audio_loudness(Path("/tmp/lf.mp4"))
        assert result.passed, result.message
        assert subproc_mod.run.call_count == 2

    def test_full_decode_that_is_loud_still_fails_without_a_retry(self):
        v, subproc_mod = _make_validator()
        v._ffprobe.probe_duration.return_value = 4329.0
        subproc_mod.run.side_effect = [_run(-42.9, 4328.6)]
        result = v.check_audio_loudness(Path("/tmp/lf.mp4"))
        assert not result.passed
        assert "LOUDER" in result.message
        assert subproc_mod.run.call_count == 1

    def test_two_short_decodes_fail_closed_as_incomplete(self):
        v, subproc_mod = _make_validator()
        v._ffprobe.probe_duration.return_value = 4329.0
        subproc_mod.run.side_effect = [_run(-50.0, 1000.0), _run(-50.0, 1200.0)]
        result = v.check_audio_loudness(Path("/tmp/lf.mp4"))
        assert not result.passed
        assert "incomplete" in result.message
        assert "1200" in result.message

    def test_nonzero_ffmpeg_exit_is_not_judged(self):
        v, subproc_mod = _make_validator()
        v._ffprobe.probe_duration.return_value = 4329.0
        subproc_mod.run.side_effect = [_run(-50.0, 4328.6, 1), _run(-50.0, 4328.6, 1)]
        result = v.check_audio_loudness(Path("/tmp/lf.mp4"))
        assert not result.passed
        assert "incomplete" in result.message
