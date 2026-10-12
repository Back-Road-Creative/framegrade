"""Drift harness — quantify chapter-timestamp placement error.

On the DRIVE path the chapter generator places every chapter with ONE uniform
linear scale (`MarkerMerger.scale_gps_to_video`: ``gps_time * video_duration /
gps_duration``) and passes ``keep_segments=None``, so the piecewise
gap-removal remap (`map_to_edited_time`) never runs. The uniform scale assumes
removed-stop time is spread evenly across the drive; when stops are CLUSTERED
the placement drifts from where the town actually appears on screen.

This harness drives BOTH real production functions (no re-implementation) over
realistic synthetic stop patterns and measures, for towns sampled across the
drive, ``drift = uniform_placement - true_edited_placement``. It answers the
question — "is the uniform scale good enough?" — with numbers instead of
prose, and locks the conclusion as a regression guard.

Run the readout with:
    .venv/bin/python -m pytest tests/test_chapter_drift_harness.py -s -o addopts=""
"""

from __future__ import annotations

from gomoveshift.pipeline.chapters.marker_merger import MarkerMerger

# ── Harness ────────────────────────────────────────────────────────────────


def _keep_to_video_duration(keep_segments: list[tuple[float, float]]) -> float:
    """Edited video duration = total kept (moving) footage."""
    return sum(end - start for start, end in keep_segments)


def _drift_profile(
    gps_duration: float,
    keep_segments: list[tuple[float, float]],
    n_samples: int = 24,
) -> dict:
    """Drift of the uniform scale vs the true piecewise map across the drive.

    For each sampled GPS time ``g`` (a town the car passes), compares the
    DRIVE-path placement (uniform scale, the only transform the drive path
    applies) against the true edited position (piecewise map over the moving
    segments). Both come from the real ``MarkerMerger`` — the harness models
    inputs only, never the math.
    """
    video_duration = _keep_to_video_duration(keep_segments)
    drifts: list[float] = []
    for i in range(1, n_samples):
        g = gps_duration * i / n_samples
        uniform = MarkerMerger.scale_gps_to_video(g, gps_duration, video_duration)
        true_pos = MarkerMerger.map_to_edited_time(g, keep_segments)
        drifts.append(uniform - true_pos)
    abs_drifts = [abs(d) for d in drifts]
    return {
        "video_duration": video_duration,
        "removed": gps_duration - video_duration,
        "max_abs": max(abs_drifts),
        "mean_abs": sum(abs_drifts) / len(abs_drifts),
        "drifts": drifts,
    }


# ── Scenarios (GPS span 3600s = 60 min) ─────────────────────────────────────

_GPS = 3600.0
_SCENARIOS: dict[str, list[tuple[float, float]]] = {
    # 15 min of stops, all up front → towns after the stop are pushed late.
    "stops_clustered_early": [(900.0, 3600.0)],
    # 15 min of stops, all at the end → towns pulled early.
    "stops_clustered_late": [(0.0, 2700.0)],
    # 15 min of stops as one mid-drive block.
    "stops_clustered_middle": [(0.0, 1350.0), (2250.0, 3600.0)],
    # 15 min of stops spread as nine evenly-distributed ~100s pauses
    # (moving 300s, pause 100s, repeated): 9*300=2700s moving, 900s removed.
    "stops_distributed_even": [(i * 400.0, i * 400.0 + 300.0) for i in range(9)],
    # Realistic best-moments LF: only ~60s of stops within the highlight,
    # as one short clustered pause partway through.
    "realistic_lf_60s_clustered": [(0.0, 1800.0), (1860.0, 3600.0)],
    # Realistic LF with NO within-LF stops (the uniform scale is exact).
    "realistic_lf_no_stops": [(0.0, 3600.0)],
}


def _report() -> dict[str, dict]:
    return {name: _drift_profile(_GPS, keep) for name, keep in _SCENARIOS.items()}


# ── Readout (printed under -s) ───────────────────────────────────────────────


def test_print_drift_readout() -> None:
    rows = _report()
    print("\n\n=== chapter-timestamp drift harness (GPS span 60:00) ===")
    print(f"{'scenario':<32}{'removed':>9}{'video':>8}{'max drift':>11}{'mean':>8}")
    for name, r in rows.items():
        print(
            f"{name:<32}{r['removed'] / 60:>7.1f}m{r['video_duration'] / 60:>7.1f}m"
            f"{r['max_abs']:>9.0f}s{r['mean_abs']:>7.0f}s"
        )
    print(
        "\nReading: drift = uniform placement (the drive path) - true edited\n"
        "position. With stops removed, clustered stops drive large drift;\n"
        "distributed stops largely cancel; no within-LF stops = exact.\n"
    )


# ── Regression guards (lock the conclusion) ─────────────────────────────────


def test_no_stops_is_exact() -> None:
    """When no footage is removed, the uniform scale is exact (drift 0)."""
    r = _drift_profile(_GPS, _SCENARIOS["realistic_lf_no_stops"])
    assert r["max_abs"] < 1e-6


def test_clustered_stops_drift_is_large() -> None:
    """15 min of clustered stops → multi-minute mid-drive drift.

    This is the failure mode this harness exists to characterise: the uniform scale
    is NOT safe when a large block of footage is removed in one place.
    """
    early = _drift_profile(_GPS, _SCENARIOS["stops_clustered_early"])
    assert early["max_abs"] > 180.0  # > 3 minutes


def test_distributed_stops_drift_is_small() -> None:
    """Evenly-distributed removal stays close to the uniform assumption."""
    even = _drift_profile(_GPS, _SCENARIOS["stops_distributed_even"])
    # Same 15 min removed as the clustered cases, but spread out.
    assert even["max_abs"] < 200.0
    assert even["max_abs"] < _drift_profile(_GPS, _SCENARIOS["stops_clustered_early"])["max_abs"]


def test_realistic_small_removal_drift_is_bounded() -> None:
    """A best-moments LF with only ~60s of within-LF stops drifts < 60s.

    This is the regime GMS actually ships (best-moments scoring favours
    motion, so little stopped footage survives inside an LF). The drift here
    is below YouTube's coarse chapter granularity — evidence that the
    uniform scale is good enough IN THIS REGIME.
    """
    r = _drift_profile(_GPS, _SCENARIOS["realistic_lf_60s_clustered"])
    assert r["removed"] <= 60.0
    assert r["max_abs"] < 60.0
