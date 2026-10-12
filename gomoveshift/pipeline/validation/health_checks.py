"""Drive-pipeline health checks -- extracted from PipelineValidator."""

from __future__ import annotations

from pathlib import Path

from gomoveshift.pipeline.pipeline_validator import (
    HealthCheckResult,
    ValidationSeverity,
)

from .ffprobe_client import FFprobeClient


def validate_video_file(ffprobe: FFprobeClient, path: Path) -> HealthCheckResult:
    """Check file exists and has positive duration (ffprobe)."""
    if not path.exists():
        return HealthCheckResult(
            passed=False,
            severity=ValidationSeverity.CRITICAL,
            message=f"File not found: {path}",
        )
    dur = ffprobe.probe_duration(path)
    if dur <= 0:
        return HealthCheckResult(
            passed=False,
            severity=ValidationSeverity.CRITICAL,
            message=f"Zero or unreadable duration for {path.name}",
        )
    return HealthCheckResult(
        passed=True,
        severity=ValidationSeverity.OK,
        message=f"{path.name}: {dur:.1f}s",
    )


def validate_duration(
    ffprobe: FFprobeClient,
    path: Path,
    expected_seconds: float,
    tolerance: float = 0.05,
) -> HealthCheckResult:
    """Actual duration must be within tolerance of expected."""
    actual = ffprobe.probe_duration(path)
    if expected_seconds <= 0:
        return HealthCheckResult(
            passed=True,
            severity=ValidationSeverity.OK,
            message="No expected duration provided -- skipped",
        )
    deviation = abs(actual - expected_seconds) / expected_seconds
    if deviation > tolerance:
        return HealthCheckResult(
            passed=False,
            severity=ValidationSeverity.CRITICAL,
            message=(
                f"{path.name}: actual={actual:.1f}s expected={expected_seconds:.1f}s "
                f"deviation={deviation:.1%} > tolerance={tolerance:.1%}"
            ),
        )
    return HealthCheckResult(
        passed=True,
        severity=ValidationSeverity.OK,
        message=f"{path.name}: {actual:.1f}s \u2248 {expected_seconds:.1f}s "
        f"(\u00b1{deviation:.1%})",
    )


def validate_score_distribution(scores) -> HealthCheckResult:
    """Sanity check score array: no NaN/Inf, not all identical."""
    import numpy as np

    arr = np.asarray(scores, dtype=float)
    if not np.isfinite(arr).all():
        return HealthCheckResult(
            passed=False,
            severity=ValidationSeverity.CRITICAL,
            message="Score array contains NaN or Inf values",
        )
    if arr.size > 1 and float(np.std(arr)) == 0.0:
        return HealthCheckResult(
            passed=False,
            severity=ValidationSeverity.CRITICAL,
            message="Score array is all-identical -- scoring may have failed",
        )
    return HealthCheckResult(
        passed=True,
        severity=ValidationSeverity.OK,
        message=f"Scores OK: {len(arr)} samples, "
        f"mean={float(np.mean(arr)):.3f} std={float(np.std(arr)):.3f}",
    )


def validate_assembly_plan(plan) -> HealthCheckResult:
    """Plan sanity: LFs in 480-1800s range, at least one LF present."""
    if not plan.regular_lfs:
        return HealthCheckResult(
            passed=False,
            severity=ValidationSeverity.CRITICAL,
            message="Drive plan has no regular long-forms",
        )
    lf_min, lf_max = 480.0, 1800.0
    for lf in plan.regular_lfs:
        if lf.total_duration < lf_min:
            return HealthCheckResult(
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"LF {lf.index} duration {lf.total_duration:.0f}s < {lf_min:.0f}s minimum",
            )
        if lf.total_duration > lf_max:
            return HealthCheckResult(
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"LF {lf.index} duration {lf.total_duration:.0f}s > {lf_max:.0f}s maximum",
            )
    return HealthCheckResult(
        passed=True,
        severity=ValidationSeverity.OK,
        message=f"Plan OK: {len(plan.regular_lfs)} LFs",
    )
