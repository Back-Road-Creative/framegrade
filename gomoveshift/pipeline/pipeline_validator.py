"""
Pipeline Validator -- final pre-publish quality gate.

Thin facade that delegates to focused modules in ``validation/``:
- ``FFprobeClient`` -- unified ffprobe subprocess wrapper
- ``LongformValidator`` -- audio loudness, video quality, bitrate preservation
- ``ShortsValidator`` -- per-short resolution, duration, LUFS checks
- ``content_gates`` -- content safety, metadata quality, YouTube best practices
- ``health_checks`` -- drive-pipeline health checks

All data classes and the public ``PipelineValidator`` API are preserved
for backward compatibility.
"""

import logging
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from . import encoding_constants as ec
from .content_validator import ContentValidator

logger = logging.getLogger(__name__)


# ── Data classes (public API) ─────────────────────────────────────────────


class ValidationSeverity(Enum):
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class HealthCheckResult:
    passed: bool
    severity: ValidationSeverity
    message: str


@dataclass
class GateResult:
    gate: str
    passed: bool
    message: str
    measured_value: Optional[str] = None
    threshold: Optional[str] = None


@dataclass
class PipelineValidationResult:
    passed: bool
    gates: list[GateResult] = field(default_factory=list)

    def summary(self) -> str:
        lines = []
        for g in self.gates:
            status = "PASS" if g.passed else "FAIL"
            lines.append(f"  [{status}] {g.gate}: {g.message}")
        return "\n".join(lines)


class PipelineValidationError(Exception):
    """Raised when the final pipeline validation gate fails."""


# ── Facade ────────────────────────────────────────────────────────────────


class PipelineValidator:
    """Final validation gate that runs after all pipeline stages."""

    def __init__(
        self,
        max_lufs: float = ec.LONGFORM_AUDIO_MAX_LUFS,
        max_true_peak: float = ec.LONGFORM_AUDIO_TP,
        min_video_bitrate_kbps: int = 5000,
        duration_tolerance: float = 0.10,
        ffprobe_timeout_sec: int = 30,
    ):
        self.max_lufs = max_lufs
        self.max_true_peak = max_true_peak
        self.min_video_bitrate_kbps = min_video_bitrate_kbps
        self.duration_tolerance = duration_tolerance
        self.content_validator = ContentValidator()

        from .validation.ffprobe_client import FFprobeClient
        from .validation.longform_validator import LongformValidator
        from .validation.shorts_validator import ShortsValidator

        # Config-driven ffprobe timeout so a slow machine under encode load
        # does not surface a transient probe timeout as a fatal corruption gate.
        self._ffprobe = FFprobeClient(subprocess, timeout=ffprobe_timeout_sec)
        self._longform = LongformValidator(
            self._ffprobe,
            max_lufs=max_lufs,
            max_true_peak=max_true_peak,
            min_video_bitrate_kbps=min_video_bitrate_kbps,
        )
        self._shorts = ShortsValidator(self._ffprobe)

    # ── Main validate (longform) ──────────────────────────────────────────

    def validate(
        self,
        video_path: Path,
        expected_duration: Optional[float] = None,
        metadata_json: Optional[dict] = None,
        social_posts: Optional[list] = None,
        skip_audio: bool = False,
        day_record: Optional[dict] = None,
        claim_native_resolution_only: bool = True,
    ) -> PipelineValidationResult:
        """Run all validation gates on the final output.

        ``day_record`` / ``claim_native_resolution_only`` feed the honest-
        resolution claim check in the video-quality gate (defaults: no check).

        Takes no source bitrate: the ``bitrate_preservation`` floor is absolute
        and resolution-based, so that gate runs on every call rather than only
        when a caller happened to have a source bitrate to pass.
        """
        from .validation import content_gates

        if metadata_json is not None:
            required = ("title", "description", "tags", "category")
            missing = [k for k in required if k not in metadata_json]
            if missing:
                raise ValueError(f"metadata_json is missing required key(s): {', '.join(missing)}")

        gates: list[GateResult] = []
        if not skip_audio:
            gates.append(self._longform.check_audio_loudness(video_path))
            # Speech-intelligibility belt on top of the integrated ceiling: a
            # loud speech spike barely moves the integrated average but shows in
            # momentary loudness. See LongformValidator.check_momentary_loudness.
            gates.append(self._longform.check_momentary_loudness(video_path))
        gates.append(
            self._longform.check_video_quality(
                video_path,
                skip_audio=skip_audio,
                day_record=day_record,
                claim_native_resolution_only=claim_native_resolution_only,
            )
        )
        gates.append(self._check_file_integrity(video_path))
        if expected_duration is not None:
            gates.append(self._check_duration(video_path, expected_duration))
        if metadata_json or social_posts:
            gates.append(
                content_gates.check_content_safety(
                    self.content_validator, metadata_json, social_posts
                )
            )
        if metadata_json:
            gates.append(content_gates.check_metadata_quality(metadata_json))
            gates.append(content_gates.check_location_invariant(metadata_json))
            gates.append(content_gates.check_label_quality(metadata_json))
        gates.append(self._longform.check_bitrate_preservation(video_path))
        if metadata_json and metadata_json.get("seo_degraded"):
            gates.append(
                GateResult(
                    gate="seo_quality",
                    passed=False,
                    message="Metadata generated in degraded mode (SEO Manager unavailable). "
                    "Fix SEO Manager init before publishing.",
                )
            )
        if metadata_json:
            gates.append(content_gates.check_youtube_best_practices(metadata_json))

        all_passed = all(g.passed for g in gates)
        result = PipelineValidationResult(passed=all_passed, gates=gates)
        if all_passed:
            logger.info("Pipeline validation: ALL GATES PASSED")
        else:
            failed = [g for g in gates if not g.passed]
            logger.error(f"Pipeline validation: {len(failed)} gate(s) FAILED\n{result.summary()}")
        return result

    # ── File / duration gates (kept here -- use ffprobe client) ───────────

    def _check_file_integrity(self, video_path: Path) -> GateResult:
        if not video_path.exists():
            logger.warning(
                "⚠ pipeline_validator: file_integrity gate FAILED for %s "
                "(file does not exist); returning GateResult(passed=False) — "
                "caller must respect the failed gate or the orchestrator will "
                "silently proceed",
                video_path,
            )
            return GateResult(
                gate="file_integrity", passed=False, message=f"File does not exist: {video_path}"
            )
        size = video_path.stat().st_size
        if size < 1024 * 1024:
            logger.warning(
                "⚠ pipeline_validator: file_integrity gate FAILED for %s "
                "(file size %d bytes < 1MB floor); returning "
                "GateResult(passed=False)",
                video_path,
                size,
            )
            return GateResult(
                gate="file_integrity",
                passed=False,
                message=f"File suspiciously small: {size} bytes",
            )
        try:
            self._ffprobe.probe(video_path, verbose_level="error", entries="format=duration")
        except Exception as e:
            logger.warning(
                "⚠ pipeline_validator: file_integrity gate FAILED for %s "
                "(ffprobe raised %s); returning GateResult(passed=False)",
                video_path,
                e,
            )
            return GateResult(
                gate="file_integrity", passed=False, message=f"File corrupt or unreadable: {e}"
            )
        return GateResult(
            gate="file_integrity", passed=True, message=f"OK ({size / (1024 * 1024):.1f} MB)"
        )

    def _check_duration(self, video_path: Path, expected_duration: float) -> GateResult:
        try:
            actual = self._ffprobe.probe_duration(video_path)
            if expected_duration <= 0:
                return GateResult(
                    gate="duration",
                    passed=True,
                    message=f"Actual: {actual:.1f}s (no expected duration to compare)",
                )
            deviation = abs(actual - expected_duration) / expected_duration
            if deviation <= self.duration_tolerance:
                return GateResult(
                    gate="duration",
                    passed=True,
                    message=f"Actual: {actual:.1f}s, expected: {expected_duration:.1f}s "
                    f"(deviation: {deviation * 100:.1f}%)",
                    measured_value=f"{actual:.1f}s",
                    threshold=f"within {self.duration_tolerance * 100:.0f}%",
                )
            return GateResult(
                gate="duration",
                passed=False,
                message=f"Duration mismatch: actual {actual:.1f}s vs expected "
                f"{expected_duration:.1f}s (deviation: {deviation * 100:.1f}%)",
                measured_value=f"{actual:.1f}s",
                threshold=f"within {self.duration_tolerance * 100:.0f}%",
            )
        except Exception as e:
            return GateResult(gate="duration", passed=False, message=f"Duration check error: {e}")

    # ── Drive-pipeline health checks (delegated) ──────────────────────────

    def validate_video_file(self, path: Path) -> HealthCheckResult:
        from .validation.health_checks import validate_video_file

        return validate_video_file(self._ffprobe, path)

    def validate_duration(
        self, path: Path, expected_seconds: float, tolerance: float = 0.05
    ) -> HealthCheckResult:
        from .validation.health_checks import validate_duration

        return validate_duration(self._ffprobe, path, expected_seconds, tolerance)

    def validate_score_distribution(self, scores) -> HealthCheckResult:
        from .validation.health_checks import validate_score_distribution

        return validate_score_distribution(scores)

    def validate_assembly_plan(self, plan) -> HealthCheckResult:
        from .validation.health_checks import validate_assembly_plan

        return validate_assembly_plan(plan)

    def validate_montage_output(self, path: Path, target_duration: float) -> HealthCheckResult:
        return self.validate_duration(path, target_duration, tolerance=0.10)

    def validate_concatenated_video(
        self, path: Path, expected_duration: float
    ) -> HealthCheckResult:
        return self.validate_duration(path, expected_duration, tolerance=0.05)

    # ── Shorts (delegated) ────────────────────────────────────────────────

    def validate_shorts(self, shorts_dir: Path) -> PipelineValidationResult:
        return self._shorts.validate_shorts(shorts_dir)

    def validate_short_files(
        self,
        short_files,
        *,
        source_sharpness=None,
        picture_changed: bool = False,
        source_ringing=None,
    ) -> PipelineValidationResult:
        """Validate a specific list of short files (day-first paths)."""
        return self._shorts.validate_short_files(
            list(short_files),
            source_sharpness=source_sharpness,
            picture_changed=picture_changed,
            source_ringing=source_ringing,
        )

    # ── Backward-compat private method delegates ──────────────────────────

    def _check_content_safety(self, metadata_json, social_posts) -> GateResult:
        from .validation.content_gates import check_content_safety

        return check_content_safety(self.content_validator, metadata_json, social_posts)

    def _check_metadata_quality(self, metadata_json: dict) -> GateResult:
        from .validation.content_gates import check_metadata_quality

        return check_metadata_quality(metadata_json)

    def _check_youtube_best_practices(self, metadata_json: dict) -> GateResult:
        from .validation.content_gates import check_youtube_best_practices

        return check_youtube_best_practices(metadata_json)

    def _check_audio_loudness(self, video_path: Path) -> GateResult:
        return self._longform.check_audio_loudness(video_path)

    def _check_video_quality(self, video_path: Path, skip_audio: bool = False) -> GateResult:
        return self._longform.check_video_quality(video_path, skip_audio=skip_audio)

    def _check_bitrate_preservation(self, video_path: Path) -> GateResult:
        return self._longform.check_bitrate_preservation(video_path)

    def _probe_duration(self, path: Path) -> float:
        return self._ffprobe.probe_duration(path)

    def _validate_single_short(self, short_path: Path) -> list[GateResult]:
        return self._shorts._validate_single_short(short_path)

    def _probe_short(self, short_path: Path) -> Optional[dict]:
        return self._ffprobe.probe_or_none(short_path, show_streams=True, show_format=True)

    def _check_short_file_integrity(self, short_path: Path) -> GateResult:
        return self._shorts._check_short_file_integrity(short_path)

    def _check_short_resolution(self, short_path: Path, probe_data: dict) -> GateResult:
        return self._shorts._check_short_resolution(short_path, probe_data)

    def _check_short_duration(self, short_path: Path, probe_data: dict) -> GateResult:
        return self._shorts._check_short_duration(short_path, probe_data)

    def _check_short_bitrate(self, short_path: Path, probe_data: dict) -> GateResult:
        return self._shorts._check_short_bitrate(short_path, probe_data)

    def _check_short_audio_lufs(self, short_path: Path) -> GateResult:
        return self._shorts._check_short_audio_lufs(short_path)
