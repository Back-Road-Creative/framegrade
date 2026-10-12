"""Shorts validation — extracted from PipelineValidator."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from gomoveshift.pipeline import encoding_constants as ec
from gomoveshift.pipeline.pipeline_validator import GateResult, PipelineValidationResult

from .ffprobe_client import FFprobeClient
from .longform_validator import color_tag_issues

logger = logging.getLogger(__name__)

# Only the below-floor bitrate check is advisory: a short a little under the
# 2000 kbps floor still plays, so it warns rather than sinking the day. An
# UNREADABLE bitrate is a different class — the encode was never confirmed at
# all — and ships under its own gate name so this prefix does not swallow it.
# Every other per-short gate is fatal.
ADVISORY_GATE_PREFIX = "short_bitrate:"


def _is_advisory(gate: str) -> bool:
    """True when a failing gate is WARNING-tier rather than fatal."""
    return gate.startswith(ADVISORY_GATE_PREFIX)


class ShortsValidator:
    """Quality validation for rendered short_*.mp4 files.

    Checks per short:
    - Resolution: must be exactly 1080x1920 (FATAL)
    - Duration: must be 15-60 seconds (FATAL)
    - File integrity: ffprobe passes, file > 500KB (FATAL)
    - Audio LUFS: within target +/- 3 LUFS (FATAL)
    - Bitrate floor: must exceed 2000 kbps (WARNING only)
    - Bitrate unreadable: ffprobe reported no container bitrate (FATAL)
    - Encoding contract: HEVC main10 10-bit, bt709/tv colour tags (FATAL)
    """

    def __init__(self, ffprobe: FFprobeClient) -> None:
        self._ffprobe = ffprobe

    def validate_short_files(self, short_files: list[Path]) -> PipelineValidationResult:
        """Run quality validation on a specific list of short files.

        Used by phase_d's day-first quality gate where each LF maps to a
        list of paths under ``_stages/day_<DATE>/shorts/`` (see
        ``state["shorts_per_lf"]``). Mirrors ``validate_shorts``'s gate
        contract — including the FATAL 1080×1920 resolution check that
        catches the day-first 16:9 regression.
        """
        short_files = sorted([Path(p) for p in short_files])
        if not short_files:
            logger.warning(
                "shorts_validator: validate_short_files called with empty list — "
                "treating as skipped (no shorts to validate)"
            )
            return PipelineValidationResult(
                passed=True,
                gates=[
                    GateResult(
                        gate="shorts_validation",
                        passed=True,
                        message="No short files supplied -- skipped",
                    )
                ],
            )

        gates: list[GateResult] = []
        has_fatal = False

        for short_path in short_files:
            short_gates = self._validate_single_short(short_path)
            for g in short_gates:
                gates.append(g)
                if not g.passed and not _is_advisory(g.gate):
                    has_fatal = True

        all_passed = not has_fatal
        result = PipelineValidationResult(passed=all_passed, gates=gates)
        if all_passed:
            logger.info(
                f"Shorts validation (file list): ALL GATES PASSED ({len(short_files)} shorts)"
            )
        else:
            failed = [g for g in gates if not g.passed]
            logger.error(
                f"Shorts validation (file list): {len(failed)} gate(s) FAILED\n{result.summary()}"
            )
        return result

    def validate_shorts(self, shorts_dir: Path) -> PipelineValidationResult:
        """Run quality validation on all rendered short_*.mp4 files."""
        short_files = sorted(shorts_dir.glob("short_*.mp4"))
        if not short_files:
            logger.warning(
                "⚠ shorts_validator: no short_*.mp4 files found in %s — "
                "preferred path (validate ≥1 rendered short) unavailable; "
                "falling back to no-op (returning passed=True with 'skipped' "
                "message; upstream shorts extraction may have failed silently)",
                shorts_dir,
            )
            return PipelineValidationResult(
                passed=True,
                gates=[
                    GateResult(
                        gate="shorts_validation",
                        passed=True,
                        message="No short files found -- skipped",
                    )
                ],
            )

        gates: list[GateResult] = []
        has_fatal = False

        for short_path in short_files:
            short_gates = self._validate_single_short(short_path)
            for g in short_gates:
                gates.append(g)
                if not g.passed and not _is_advisory(g.gate):
                    has_fatal = True

        all_passed = not has_fatal
        result = PipelineValidationResult(passed=all_passed, gates=gates)

        if all_passed:
            logger.info(f"Shorts validation: ALL GATES PASSED ({len(short_files)} shorts)")
        else:
            failed = [g for g in gates if not g.passed]
            logger.error(f"Shorts validation: {len(failed)} gate(s) FAILED\n{result.summary()}")

        return result

    # -- per-short checks ---------------------------------------------------

    def _validate_single_short(self, short_path: Path) -> list[GateResult]:
        name = short_path.name
        gates: list[GateResult] = []

        gates.append(self._check_short_file_integrity(short_path))
        if not gates[-1].passed:
            return gates

        probe_data = self._ffprobe.probe_or_none(short_path, show_streams=True, show_format=True)
        if probe_data is None:
            gates.append(
                GateResult(
                    gate=f"short_probe:{name}",
                    passed=False,
                    message=f"ffprobe failed for {name}",
                )
            )
            return gates

        gates.append(self._check_short_resolution(short_path, probe_data))
        gates.append(self._check_short_duration(short_path, probe_data))
        gates.append(self._check_short_bitrate(short_path, probe_data))
        gates.append(self._check_short_encoding_contract(short_path, probe_data))
        gates.append(self._check_short_audio_lufs(short_path))

        return gates

    def _check_short_file_integrity(self, short_path: Path) -> GateResult:
        name = short_path.name

        if not short_path.exists():
            return GateResult(
                gate=f"short_integrity:{name}",
                passed=False,
                message=f"File does not exist: {name}",
            )

        size = short_path.stat().st_size
        if size < 500 * 1024:
            return GateResult(
                gate=f"short_integrity:{name}",
                passed=False,
                message=f"{name}: file too small ({size / 1024:.0f}KB < 500KB)",
                measured_value=f"{size / 1024:.0f}KB",
                threshold="500KB",
            )

        try:
            self._ffprobe.probe(
                short_path,
                verbose_level="error",
                entries="format=duration",
            )
        except Exception as e:
            return GateResult(
                gate=f"short_integrity:{name}",
                passed=False,
                message=f"{name}: corrupt or unreadable: {e}",
            )

        return GateResult(
            gate=f"short_integrity:{name}",
            passed=True,
            message=f"{name}: OK ({size / (1024 * 1024):.1f}MB)",
        )

    def _check_short_resolution(self, short_path: Path, probe_data: dict) -> GateResult:
        name = short_path.name
        streams = probe_data.get("streams", [])
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)

        if not video_stream:
            return GateResult(
                gate=f"short_resolution:{name}",
                passed=False,
                message=f"{name}: no video stream found",
            )

        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))

        if width == ec.SHORTS_WIDTH and height == ec.SHORTS_HEIGHT:
            return GateResult(
                gate=f"short_resolution:{name}",
                passed=True,
                message=f"{name}: {width}x{height}",
                measured_value=f"{width}x{height}",
                threshold=f"{ec.SHORTS_WIDTH}x{ec.SHORTS_HEIGHT}",
            )

        return GateResult(
            gate=f"short_resolution:{name}",
            passed=False,
            message=(
                f"{name}: resolution {width}x{height} != "
                f"required {ec.SHORTS_WIDTH}x{ec.SHORTS_HEIGHT}"
            ),
            measured_value=f"{width}x{height}",
            threshold=f"{ec.SHORTS_WIDTH}x{ec.SHORTS_HEIGHT}",
        )

    def _check_short_duration(self, short_path: Path, probe_data: dict) -> GateResult:
        name = short_path.name
        duration = float(probe_data.get("format", {}).get("duration", 0))

        min_dur = 15.0
        max_dur = 60.0

        if min_dur <= duration <= max_dur:
            return GateResult(
                gate=f"short_duration:{name}",
                passed=True,
                message=f"{name}: {duration:.1f}s",
                measured_value=f"{duration:.1f}s",
                threshold=f"{min_dur:.0f}-{max_dur:.0f}s",
            )

        return GateResult(
            gate=f"short_duration:{name}",
            passed=False,
            message=f"{name}: duration {duration:.1f}s outside {min_dur:.0f}-{max_dur:.0f}s",
            measured_value=f"{duration:.1f}s",
            threshold=f"{min_dur:.0f}-{max_dur:.0f}s",
        )

    def _check_short_bitrate(self, short_path: Path, probe_data: dict) -> GateResult:
        name = short_path.name
        min_bitrate_kbps = 2000
        bitrate = int(probe_data.get("format", {}).get("bit_rate", 0)) // 1000

        if bitrate <= 0:
            # Fail-closed: this used to return passed=True, so a
            # short whose encode could not be measured at all was reported as
            # verified. It ships under its own FATAL gate name rather than the
            # advisory ``short_bitrate:`` prefix — an unmeasurable encode is not
            # the same class as one a little under the floor.
            return GateResult(
                gate=f"short_bitrate_unreadable:{name}",
                passed=False,
                message=(
                    f"{name}: bitrate unreadable from ffprobe — fail-closed "
                    f"(cannot certify the encode against the {min_bitrate_kbps}kbps floor)"
                ),
                threshold=f">= {min_bitrate_kbps}kbps",
            )

        if bitrate >= min_bitrate_kbps:
            return GateResult(
                gate=f"short_bitrate:{name}",
                passed=True,
                message=f"{name}: {bitrate}kbps",
                measured_value=f"{bitrate}kbps",
                threshold=f">= {min_bitrate_kbps}kbps",
            )

        return GateResult(
            gate=f"short_bitrate:{name}",
            passed=False,
            message=(f"{name}: bitrate {bitrate}kbps below floor {min_bitrate_kbps}kbps (warning)"),
            measured_value=f"{bitrate}kbps",
            threshold=f">= {min_bitrate_kbps}kbps",
        )

    def _check_short_encoding_contract(self, short_path: Path, probe_data: dict) -> GateResult:
        """Assert the shipped short matches the frozen encode contract.

        Without these checks a short encoded 8-bit / Main profile (the
        banded-skies regression: p5 / CQ 23 / 8-bit ``yuv420p``) passed every
        shorts gate and shipped with banded skies; this gate fails pix_fmt /
        profile / codec / colour-flag drift. Shorts are cut from the graded 10-bit day master and
        encoded to match the long-form master, so the contract is
        ``SHORTS_PROFILE`` main10, 10-bit 4:2:0, HEVC, and the same SDR BT.709
        limited-range tags — the reported forms (``Main 10`` /
        ``yuv420p10le`` for a ``-pix_fmt p010le`` NVENC encode) confirmed
        against a real produced short. Fail-closed like the long-form gate: a
        field ffprobe could not read is an issue, never a skip.
        """
        name = short_path.name
        gate = f"short_encoding_contract:{name}"
        streams = probe_data.get("streams", [])
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        if not video_stream:
            return GateResult(
                gate=gate,
                passed=False,
                message=f"{name}: no video stream found",
            )

        codec = str(video_stream.get("codec_name", "") or "").lower()
        profile = str(video_stream.get("profile", "") or "").lower()
        pix_fmt = str(video_stream.get("pix_fmt", "") or "").lower()

        issues: list[str] = []
        if not codec:
            issues.append("codec unreadable from ffprobe (fail-closed)")
        elif codec not in ("hevc", "h265"):
            issues.append(f"codec '{codec}' should be hevc")
        if not profile:
            issues.append("profile unreadable from ffprobe (fail-closed)")
        elif "10" not in profile:
            issues.append(f"profile '{profile}' should be {ec.SHORTS_PROFILE}")
        if not pix_fmt:
            issues.append("pix_fmt unreadable from ffprobe (fail-closed)")
        elif pix_fmt not in ("yuv420p10le", ec.SHORTS_PIX_FMT):
            issues.append(f"pix_fmt '{pix_fmt}' should be yuv420p10le or {ec.SHORTS_PIX_FMT}")
        issues.extend(color_tag_issues(video_stream))

        if issues:
            return GateResult(
                gate=gate,
                passed=False,
                message=f"{name}: off-contract encode: {', '.join(issues)}",
                measured_value=f"{codec} {profile} {pix_fmt}",
                threshold=f"hevc {ec.SHORTS_PROFILE} 10-bit, bt709/tv",
            )
        return GateResult(
            gate=gate,
            passed=True,
            message=f"{name}: {codec} {profile} {pix_fmt}, bt709/tv",
            measured_value=f"{codec} {profile} {pix_fmt}",
        )

    def _check_short_audio_lufs(self, short_path: Path) -> GateResult:
        """Near-silent policy: shorts ship
        MUTED (`-an`), so a shipped short MUST carry no audio stream. A present
        audio stream is a regression — the old two-pass loudnorm boost that
        shipped audible speech — and is hard-failed here,
        upstream of the Metricool CSV write and any Drive upload. The intent is
        derived (shorts are mute-by-design): missing audio = PASS, present
        audio = FAIL. This is stronger than the prior loudness-window measure,
        which a re-normalizing platform could defeat.
        """
        name = short_path.name
        gate = f"short_audio_muted:{name}"
        try:
            data = self._ffprobe.probe(short_path, show_streams=True)
            streams = data.get("streams", []) if isinstance(data, dict) else []
            audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

            if not audio_streams:
                return GateResult(
                    gate=gate,
                    passed=True,
                    message=f"{name}: muted (no audio stream) — near-silent policy",
                    threshold="no audio stream (-an)",
                )

            return GateResult(
                gate=gate,
                passed=False,
                message=(
                    f"{name}: carries {len(audio_streams)} audio stream(s) but shorts "
                    "must ship muted (-an) under the near-silent policy"
                ),
                measured_value=f"{len(audio_streams)} audio stream(s)",
                threshold="no audio stream (-an)",
            )

        except subprocess.TimeoutExpired:
            return GateResult(
                gate=gate,
                passed=False,
                message=f"{name}: audio-stream probe timed out",
            )
        except Exception as e:
            return GateResult(
                gate=gate,
                passed=False,
                message=f"{name}: audio-stream probe error: {e}",
            )
