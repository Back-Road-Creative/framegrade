"""Long-form video validation — extracted from PipelineValidator."""

from __future__ import annotations

import json
import logging
import re
import statistics
import subprocess
from pathlib import Path

from gomoveshift.pipeline import encoding_constants as ec
from gomoveshift.pipeline import quality
from gomoveshift.pipeline.pipeline_validator import GateResult
from gomoveshift.pipeline.quality import aspect_of, rendition_for

from .ffprobe_client import FFprobeClient

logger = logging.getLogger(__name__)

# ffprobe stream key → the frozen constant it must equal. Checking these on
# every final artifact guards against the full-range-upload class (a
# `pc`-range or untagged upload that a player renders crushed). The reported
# forms were confirmed
# against a real pipeline-produced 10-bit artifact: `-color_range tv
# -colorspace bt709 -color_primaries bt709 -color_trc bt709` probes back as
# color_range=tv / color_space=bt709 / color_primaries=bt709 /
# color_transfer=bt709.
COLOR_TAG_CONTRACT: dict[str, str] = {
    "color_range": ec.LONGFORM_COLOR_RANGE,
    "color_space": ec.LONGFORM_COLORSPACE,
    "color_primaries": ec.LONGFORM_COLOR_PRIMARIES,
    "color_transfer": ec.LONGFORM_COLOR_TRC,
}

# Duration-scaled timeout for a full-file audio-only ffmpeg scan (loudnorm
# measure in check_audio_loudness, ebur128 in check_momentary_loudness). Both
# commands already pass ``-vn`` (audio-only decode, no video decode), so wall
# time tracks input duration close to 1:1 even under concurrent encode I/O
# load — mirrors single_pass._calc_audio_measurement_timeout's shape. Floored
# at the prior flat timeout so today's short LFs keep the same budget, and
# degrades to the floor on any probe failure (probe_duration returns 0.0 on
# failure) — never a TIGHTER timeout than before this change. A multi-hour
# FINAL's audio-only decode can otherwise exceed the flat 300s.
_AUDIO_SCAN_TIMEOUT_MULTIPLIER = 1.0
_AUDIO_SCAN_TIMEOUT_FLOOR_SEC = 300


def color_tag_issues(video_stream: dict) -> list[str]:
    """Return one issue per colour tag that is absent or off-contract.

    Fail-closed: an ABSENT tag is the untagged-upload case (the player guesses
    the range and can render the whole video crushed), so it is reported, never
    skipped. Shared by the long-form and shorts gates — both encodes stamp the
    same SDR BT.709 limited-range flags.
    """
    issues: list[str] = []
    for key, expected in COLOR_TAG_CONTRACT.items():
        actual = str(video_stream.get(key, "") or "").lower()
        if not actual:
            issues.append(f"{key} unreadable from ffprobe (fail-closed)")
        elif actual != expected:
            issues.append(f"{key} '{actual}' should be {expected}")
    return issues


# ── Perceptual spot-check ─────────────────────────────────────
# Every pre-upload gate read container/stream METADATA only, so a final that
# encoded to a black or frozen picture satisfied all of them and shipped. Three
# windows cover head/middle/tail for a few seconds on a multi-hour encode. A
# window is "dead" when one black run covers essentially all of it; the REPORT
# thresholds never abort a run (see check_perceptual_spot).
PERCEPTUAL_WINDOW_SEC = 10.0
PERCEPTUAL_WINDOW_FRACTIONS = (0.2, 0.5, 0.8)
PERCEPTUAL_DEAD_WINDOW_RATIO = 0.95
PERCEPTUAL_BLACK_REPORT_SEC = 2.0
PERCEPTUAL_FREEZE_REPORT_SEC = 3.0
PERCEPTUAL_PROBE_TIMEOUT_SEC = 120
PERCEPTUAL_FILTER = "blackdetect=d=0.5:pix_th=0.10,freezedetect=n=0.003:d=2"

_BLACK_DURATION_RE = re.compile(r"black_duration:\s*([0-9.]+)")
_FREEZE_DURATION_RE = re.compile(r"freeze_duration:\s*([0-9.]+)")


PICTURE_SKIPPED_MESSAGE = "skipped: no rendition step changed the picture"


def picture_quality_gate(
    video_path: Path,
    duration: float,
    *,
    source_sharpness: float | None,
    changed: bool,
) -> GateResult:
    """Fail a delivered file that is softer than its source or rings.

    Skipped (recorded as passed, with a message -- never absent) when no
    rendition step changed the picture or the source sharpness is unmeasured.
    """
    gate = "picture_quality"
    if not changed or source_sharpness is None:
        return GateResult(gate=gate, passed=True, message=PICTURE_SKIPPED_MESSAGE)
    sharp, ring = quality.sample_delivered_picture(video_path, duration)
    if not sharp:
        return GateResult(
            gate=gate, passed=False, message="no frame could be decoded for the picture check"
        )
    delivered = float(statistics.median(sharp))
    ringing = float(statistics.median(ring))
    floor = source_sharpness * (1.0 - quality.DELIVERED_SHARPNESS_TOLERANCE)
    problems = []
    if delivered < floor:
        problems.append(
            f"delivered sharpness {delivered:.1f} < {floor:.1f} "
            f"(source {source_sharpness:.1f} less {quality.DELIVERED_SHARPNESS_TOLERANCE:.0%})"
        )
    if ringing > quality.RINGING_MAX:
        problems.append(f"ringing {ringing:.4f} > max {quality.RINGING_MAX:.4f}")
    if problems:
        return GateResult(gate=gate, passed=False, message="; ".join(problems))
    return GateResult(
        gate=gate,
        passed=True,
        message=f"sharpness {delivered:.1f} (source {source_sharpness:.1f}), ringing {ringing:.4f}",
    )


class LongformValidator:
    """Audio loudness, video quality, and bitrate preservation checks.

    All methods return ``GateResult`` for consistency with the
    ``PipelineValidationResult`` aggregation in the facade.
    """

    # True-peak measurement tolerance. The measured inter-sample peak lands
    # within ~±0.1 dB of the true value due to floating-point quantisation and
    # IIR oversampling rounding, so the gate applies this slack at the true-peak
    # cap (LONGFORM_AUDIO_TP = -10 dBTP under the near-silent policy) to
    # avoid failing exactly at the cap on jitter alone. Near-silent attenuated
    # output sits far below the cap, so this tolerance is a belt only.
    TRUE_PEAK_MEASUREMENT_TOLERANCE_DB = 0.1

    def __init__(
        self,
        ffprobe: FFprobeClient,
        *,
        target_lufs: float = ec.LONGFORM_AUDIO_LUFS,
        lufs_tolerance: float = 2.0,
        max_lufs: float = ec.LONGFORM_AUDIO_MAX_LUFS,
        max_true_peak: float = ec.LONGFORM_AUDIO_TP,
        max_momentary_lufs: float = ec.LONGFORM_AUDIO_MAX_MOMENTARY,
        min_video_bitrate_kbps: int = 5000,
    ) -> None:
        self._ffprobe = ffprobe
        self.target_lufs = target_lufs
        self.lufs_tolerance = lufs_tolerance
        self.max_lufs = max_lufs
        self.max_true_peak = max_true_peak
        self.max_momentary_lufs = max_momentary_lufs
        self.min_video_bitrate_kbps = min_video_bitrate_kbps

    def _audio_scan_timeout_sec(self, video_path: Path) -> int:
        """Timeout for a full-file audio-only ffmpeg scan — see the module-
        level ``_AUDIO_SCAN_TIMEOUT_*`` constants for the scaling rationale."""
        duration = self._ffprobe.probe_duration(video_path)
        return max(_AUDIO_SCAN_TIMEOUT_FLOOR_SEC, int(duration * _AUDIO_SCAN_TIMEOUT_MULTIPLIER))

    def check_audio_loudness(self, video_path: Path) -> GateResult:
        """Measure output loudness and verify it's below threshold."""
        try:
            cmd = [
                "ffmpeg",
                "-i",
                str(video_path),
                "-vn",
                "-af",
                "loudnorm=print_format=json",
                "-f",
                "null",
                "-",
            ]
            result = self._ffprobe._subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._audio_scan_timeout_sec(video_path),
            )
            stderr = result.stderr or ""

            json_match = re.search(
                r'\{\s*\n\s*"input_i".*?\}',
                stderr,
                re.DOTALL,
            )
            if not json_match:
                return GateResult(
                    gate="audio_loudness",
                    passed=False,
                    message="Could not measure output loudness",
                )

            stats = json.loads(json_match.group())
            measured_i = float(stats.get("input_i", 0))
            measured_tp = float(stats.get("input_tp", 0))

            # Near-silent policy: the output
            # must be NO LOUDER than the near-silent ceiling. We enforce ONLY an
            # upper bound — a quieter-than-ceiling output is the GOAL, not a
            # defect. The prior two-sided window failed too-QUIET audio and
            # thereby demanded the -20 LUFS boost that shipped audible
            # speech; it is deliberately gone.
            ceiling_ok = measured_i <= self.max_lufs
            # True peak hard cap. Small tolerance absorbs ISP/float jitter at
            # the cap; see TRUE_PEAK_MEASUREMENT_TOLERANCE_DB.
            effective_tp_ceiling = self.max_true_peak + self.TRUE_PEAK_MEASUREMENT_TOLERANCE_DB
            peak_ok = measured_tp <= effective_tp_ceiling

            threshold_str = (
                f"loudness <= {self.max_lufs:.0f} LUFS near-silent ceiling, "
                f"peak <= {self.max_true_peak:.1f} dBTP "
                f"(+{self.TRUE_PEAK_MEASUREMENT_TOLERANCE_DB:.1f} dB tolerance)"
            )

            if ceiling_ok and peak_ok:
                return GateResult(
                    gate="audio_loudness",
                    passed=True,
                    message=(
                        f"Loudness {measured_i:.1f} LUFS "
                        f"(<= {self.max_lufs:.0f} ceiling), peak {measured_tp:.1f} dBTP"
                    ),
                    measured_value=f"{measured_i:.1f} LUFS",
                    threshold=threshold_str,
                )

            issues: list[str] = []
            if not ceiling_ok:
                issues.append(
                    f"loudness {measured_i:.1f} is LOUDER than the near-silent "
                    f"ceiling {self.max_lufs:.0f} LUFS"
                )
            if not peak_ok:
                issues.append(
                    f"true peak {measured_tp:.2f} exceeds "
                    f"max {self.max_true_peak:.1f} "
                    f"(+{self.TRUE_PEAK_MEASUREMENT_TOLERANCE_DB:.1f} dB tolerance)"
                )
            return GateResult(
                gate="audio_loudness",
                passed=False,
                message=f"Audio too loud: {', '.join(issues)}",
                measured_value=f"{measured_i:.1f} LUFS / {measured_tp:.1f} dBTP",
                threshold=threshold_str,
            )

        except subprocess.TimeoutExpired:
            return GateResult(
                gate="audio_loudness",
                passed=False,
                message="Audio measurement timed out",
            )
        except Exception as e:
            return GateResult(
                gate="audio_loudness",
                passed=False,
                message=f"Audio measurement error: {e}",
            )

    # ebur128 prints one status line per frame carrying the running momentary
    # ("M:"), short-term ("S:"), and integrated ("I:") loudness. We take the max
    # momentary across the file. The pre-roll gate emits a sentinel floor
    # (~-120 LUFS) before 400 ms of audio is buffered — dropped below.
    _MOMENTARY_RE = re.compile(r"\bM:\s*(-?\d+(?:\.\d+)?)")
    _EBUR128_GATE_FLOOR_LUFS = -120.0

    def check_momentary_loudness(self, video_path: Path) -> GateResult:
        """Reject an LF whose loudest 400 ms window stays interpretable.

        Belt-and-suspenders on top of the integrated-LUFS ceiling
        (:meth:`check_audio_loudness`): the attenuate-only ``volume`` gain
        pulls momentary windows down with the integrated loudness, so a
        correctly near-silent LF sits far below ``max_momentary_lufs``. A file
        that skipped (or under-applied) attenuation keeps its loud speech
        words — which barely move the whole-file integrated average but spike
        momentary loudness — and is caught here, upstream of upload. Momentary
        loudness comes from ffmpeg's ``ebur128`` per-frame ``M:`` field.

        A parse failure (no ``M:`` samples) WARNs and passes: the integrated
        gate is the primary guard and already hard-fails on a measurement
        failure, so this belt degrades rather than double-failing.
        """
        try:
            cmd = [
                "ffmpeg",
                "-i",
                str(video_path),
                "-vn",
                "-af",
                "ebur128=peak=true",
                "-f",
                "null",
                "-",
            ]
            result = self._ffprobe._subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._audio_scan_timeout_sec(video_path),
            )
            stderr = result.stderr or ""
            samples = [
                float(m)
                for m in self._MOMENTARY_RE.findall(stderr)
                if float(m) > self._EBUR128_GATE_FLOOR_LUFS
            ]
            if not samples:
                logger.warning(
                    "momentary-loudness belt: ebur128 emitted no M: samples for %s — "
                    "passing (the integrated-LUFS gate is the primary guard)",
                    video_path,
                )
                return GateResult(
                    gate="audio_momentary",
                    passed=True,
                    message="momentary loudness unmeasured (no ebur128 M: samples) — "
                    "deferring to the integrated-LUFS gate",
                )
            max_m = max(samples)
            threshold_str = f"max momentary <= {self.max_momentary_lufs:.0f} LUFS"
            if max_m <= self.max_momentary_lufs:
                return GateResult(
                    gate="audio_momentary",
                    passed=True,
                    message=f"max momentary {max_m:.1f} LUFS "
                    f"(<= {self.max_momentary_lufs:.0f} ceiling)",
                    measured_value=f"{max_m:.1f} LUFS",
                    threshold=threshold_str,
                )
            return GateResult(
                gate="audio_momentary",
                passed=False,
                message=f"max momentary loudness {max_m:.1f} LUFS exceeds the near-silent "
                f"ceiling {self.max_momentary_lufs:.0f} LUFS — brief speech may be "
                f"interpretable at max volume; the audio was not attenuated",
                measured_value=f"{max_m:.1f} LUFS",
                threshold=threshold_str,
            )
        except subprocess.TimeoutExpired:
            return GateResult(
                gate="audio_momentary",
                passed=False,
                message="momentary-loudness measurement timed out",
            )
        except Exception as e:
            return GateResult(
                gate="audio_momentary",
                passed=False,
                message=f"momentary-loudness measurement error: {e}",
            )

    def check_picture_quality(
        self,
        video_path: Path,
        *,
        source_sharpness: float | None,
        changed: bool,
    ) -> GateResult:
        """Post-rendition picture gate -- see ``picture_quality_gate``."""
        duration = self._ffprobe.probe_duration(video_path) if changed else 0.0
        return picture_quality_gate(
            video_path, duration, source_sharpness=source_sharpness, changed=changed
        )

    def check_video_quality(self, video_path: Path, *, skip_audio: bool = False) -> GateResult:
        """Verify video codec, resolution, and bitrate meet minimums."""
        try:
            data = self._ffprobe.probe(video_path, show_streams=True, show_format=True)
            fmt = data.get("format", {})
            streams = data.get("streams", [])

            video_stream = None
            audio_stream = None
            for s in streams:
                if s.get("codec_type") == "video" and video_stream is None:
                    video_stream = s
                if s.get("codec_type") == "audio" and audio_stream is None:
                    audio_stream = s

            if not video_stream:
                return GateResult(
                    gate="video_quality",
                    passed=False,
                    message="No video stream found",
                )

            if not audio_stream and not skip_audio:
                return GateResult(
                    gate="video_quality",
                    passed=False,
                    message="No audio stream found",
                )

            codec = video_stream.get("codec_name", "unknown")
            width = int(video_stream.get("width", 0))
            height = int(video_stream.get("height", 0))
            bitrate = int(fmt.get("bit_rate", 0)) // 1000
            profile = video_stream.get("profile", "").lower()
            pix_fmt = video_stream.get("pix_fmt", "")

            issues: list[str] = []
            if codec not in ("hevc", "h265"):
                issues.append(f"unexpected codec: {codec} (expected hevc)")

            # 4K-claim gate: long-form metadata unconditionally
            # asserts "4K" — in the title ("… Scenic Drive in 4K | …") and the
            # above-the-fold SEO description line ("… 4K drive …") — and
            # the long-form product is contractually 4K (encoding_constants).
            # A sub-2160p final would ship those literal 4K claims while not
            # actually being 4K, so reject a non-4K long-form here rather than
            # let it publish with a false resolution claim.
            #
            # FAIL-CLOSED: every check below used to be guarded by
            # its own field's truthiness (``if height and …``, ``if bitrate > 0
            # and …``, ``if profile and …``, ``if pix_fmt and …``), so a PARTIAL
            # ffprobe read skipped the check and reported a clean pass on an
            # unverified final. A field the probe could not read is now an
            # issue in its own right — the gate never certifies what it did not
            # measure.
            if not height:
                issues.append("resolution unreadable from ffprobe (fail-closed)")
            else:
                lf4k = rendition_for("drive_longform_4k")
                if height < lf4k.height:
                    issues.append(
                        f"resolution {width}x{height} below 4K ({lf4k.height}p) — long-form "
                        f"title/description claim 4K; a non-4K final must not ship"
                    )
                elif (width, height) != (lf4k.width, lf4k.height):
                    want_aw, want_ah = lf4k.aspect or (0, 0)
                    got_aw, got_ah = aspect_of(width, height)
                    issues.append(
                        f"resolution {width}x{height} (aspect {got_aw}:{got_ah}) is not the "
                        f"{lf4k.width}x{lf4k.height} ({want_aw}:{want_ah}) long-form frame "
                        f"the 4K claim requires"
                    )

            # Floors are constants-sourced. The former hardcoded
            # 30000/12000 sat BELOW encoding_constants' real 40000/15000, so a
            # regressed final landing between the two numbers passed here while
            # failing ``check_bitrate_preservation``, which reads ec correctly.
            if height >= rendition_for("drive_longform_4k").height:
                min_bitrate = rendition_for("drive_longform_4k").min_bitrate_kbps
            elif height >= 1080:
                min_bitrate = ec.LONGFORM_MIN_BITRATE_1080P
            else:
                min_bitrate = self.min_video_bitrate_kbps
            if bitrate <= 0:
                issues.append("bitrate unreadable from ffprobe (fail-closed)")
            elif bitrate < min_bitrate:
                issues.append(
                    f"bitrate {bitrate}kbps below minimum {min_bitrate}kbps for {height}p"
                )

            if not profile:
                issues.append("profile unreadable from ffprobe (fail-closed)")
            elif "10" not in profile:
                issues.append(f"profile '{profile}' should be main10")
            if not pix_fmt:
                issues.append("pix_fmt unreadable from ffprobe (fail-closed)")
            elif pix_fmt not in ("yuv420p10le", "p010le"):
                issues.append(f"pix_fmt '{pix_fmt}' should be yuv420p10le or p010le")
            issues.extend(color_tag_issues(video_stream))

            if issues:
                return GateResult(
                    gate="video_quality",
                    passed=False,
                    message=f"Quality issues: {', '.join(issues)}",
                    measured_value=f"{codec} {width}x{height} {bitrate}kbps",
                )

            return GateResult(
                gate="video_quality",
                passed=True,
                message=f"{codec} {width}x{height} @ {bitrate}kbps",
                measured_value=f"{codec} {width}x{height} {bitrate}kbps",
            )

        except Exception as e:
            return GateResult(
                gate="video_quality",
                passed=False,
                message=f"Quality check error: {e}",
            )

    def _sample_perceptual_window(self, video_path: Path, start: float, window: float) -> str:
        """Run one blackdetect+freezedetect pass and return ffmpeg's stderr."""
        head = f"ffmpeg -hide_banner -nostats -ss {start:.3f} -t {window:.3f} -i".split()
        result = self._ffprobe._subprocess.run(
            [*head, str(video_path), "-an", "-vf", PERCEPTUAL_FILTER, "-f", "null", "-"],
            capture_output=True,
            text=True,
            timeout=PERCEPTUAL_PROBE_TIMEOUT_SEC,
        )
        return result.stderr or ""

    def check_perceptual_spot(self, video_path: Path, *, duration: float | None = None):
        """Look at actual pixels of the final before it is uploaded.

        Asymmetric on purpose: only a final where EVERY sampled window is
        essentially all black fails (no picture — a render fault, never
        footage). Shorter black/frozen runs are reported but pass, because a
        night tunnel and a minute stopped in traffic are real footage and this
        runs immediately before an irreversible upload. An unprobeable file or
        ffmpeg failure degrades to a pass — ``file_integrity`` already
        hard-fails a corrupt final, the same don't-double-fail policy as
        :meth:`check_momentary_loudness`.
        """
        gate = "perceptual_spot"
        try:
            probe = self._ffprobe.probe_or_none(video_path, show_format=True, show_streams=True)
            if probe and (duration is None or duration <= 0):
                duration = float(probe.get("format", {}).get("duration", 0) or 0)
            if not probe or not duration or duration <= 0:
                return GateResult(
                    gate=gate,
                    passed=True,
                    message="perceptual spot-check unavailable (file did not probe, or no "
                    "readable duration) — deferring to the file_integrity gate",
                )

            window = min(PERCEPTUAL_WINDOW_SEC, duration)
            starts = sorted(
                {
                    round(max(0.0, min(duration - window, duration * f)), 3)
                    for f in PERCEPTUAL_WINDOW_FRACTIONS
                }
            )
            black_runs: list[float] = []
            freeze_runs: list[float] = []
            dead_windows = 0
            for start in starts:
                stderr = self._sample_perceptual_window(video_path, start, window)
                blacks = [float(v) for v in _BLACK_DURATION_RE.findall(stderr)]
                black_runs.extend(blacks)
                freeze_runs.extend(float(v) for v in _FREEZE_DURATION_RE.findall(stderr))
                if blacks and max(blacks) >= window * PERCEPTUAL_DEAD_WINDOW_RATIO:
                    dead_windows += 1

            longest_black = max(black_runs, default=0.0)
            longest_freeze = max(freeze_runs, default=0.0)
            measured = (
                f"black {longest_black:.1f}s / freeze {longest_freeze:.1f}s "
                f"across {len(starts)} x {window:.0f}s window(s)"
            )
            if starts and dead_windows == len(starts):
                return GateResult(
                    gate=gate,
                    passed=False,
                    message=f"every sampled window is black ({dead_windows}/{len(starts)}) — "
                    "the final has no picture; re-encode rather than upload",
                    measured_value=measured,
                    threshold="at least one sampled window carries picture",
                )
            if (
                longest_black >= PERCEPTUAL_BLACK_REPORT_SEC
                or longest_freeze >= PERCEPTUAL_FREEZE_REPORT_SEC
            ):
                logger.warning(
                    "PERCEPTUAL-SPOT: %s — %s. Legitimate on a tunnel or a stop; "
                    "check the final if neither applies.",
                    video_path.name,
                    measured,
                )
            return GateResult(
                gate=gate,
                passed=True,
                message=f"picture present in every sampled window ({measured})",
                measured_value=measured,
            )
        except subprocess.TimeoutExpired:
            return GateResult(gate=gate, passed=True, message="perceptual spot-check timed out")
        except Exception as e:
            return GateResult(gate=gate, passed=True, message=f"perceptual spot-check error: {e}")

    def check_bitrate_preservation(self, video_path: Path) -> GateResult:
        """Verify output bitrate meets the absolute resolution-based floor.

        The floor is fixed per resolution (40 Mbps for 4K, 15 Mbps for 1080p),
        so this gate needs nothing about the source: it reads the output and
        the frozen constants only.
        """
        try:
            data = self._ffprobe.probe(video_path, show_format=True, show_streams=True)
            output_bitrate_kbps = int(data.get("format", {}).get("bit_rate", 0)) // 1000

            if output_bitrate_kbps <= 0:
                return GateResult(
                    gate="bitrate_preservation",
                    passed=False,
                    message="Could not read output bitrate",
                )

            # Determine resolution-based floor
            streams = data.get("streams", [])
            lf4k = rendition_for("drive_longform_4k")
            height = lf4k.height  # default to 4K floor (strictest)
            for s in streams:
                if s.get("codec_type") == "video":
                    height = int(s.get("height", lf4k.height))
                    break

            if height >= lf4k.height:
                floor = lf4k.min_bitrate_kbps
            elif height >= 1080:
                floor = ec.LONGFORM_MIN_BITRATE_1080P
            else:
                floor = 5000

            if output_bitrate_kbps >= floor:
                return GateResult(
                    gate="bitrate_preservation",
                    passed=True,
                    message=(
                        f"Output {output_bitrate_kbps}kbps >= {floor}kbps floor for {height}p"
                    ),
                    measured_value=f"{output_bitrate_kbps}kbps",
                    threshold=f">= {floor}kbps",
                )
            else:
                return GateResult(
                    gate="bitrate_preservation",
                    passed=False,
                    message=(
                        f"Output {output_bitrate_kbps}kbps below {floor}kbps floor for {height}p"
                    ),
                    measured_value=f"{output_bitrate_kbps}kbps",
                    threshold=f">= {floor}kbps",
                )

        except Exception as e:
            return GateResult(
                gate="bitrate_preservation",
                passed=False,
                message=f"Bitrate preservation check error: {e}",
            )
