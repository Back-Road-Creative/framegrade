"""Stop-free window gate (``check_stop_free_windows``).

Cross-checks a shipped LF/short's RETAINED source-time window against motion
evidence so a majority-parked clip hard-fails UPSTREAM of the irreversible
encode/upload — the late-gate analog of the per-day stop mask.

GPS points are ``(t_master, lat, lon)`` tuples in day-master seconds; speed is
derived from coordinate deltas (GPX carries no speed channel), so a parked span
repeats a coordinate and a moving span advances latitude per second.
"""

from __future__ import annotations

import numpy as np

from gomoveshift.pipeline.validation.content_gates import check_stop_free_windows

_M_PER_DEG_LAT = 111_320.0


def _parked(n: int, base_lat: float = 40.0, base_lon: float = -80.0):
    return [(float(t), base_lat, base_lon) for t in range(n)]


def _moving(n: int, mps: float = 15.0, base_lat: float = 40.0, base_lon: float = -80.0):
    step = mps / _M_PER_DEG_LAT  # m/s -> deg lat per 1 s
    return [(float(t), base_lat + t * step, base_lon) for t in range(n)]


def test_majority_parked_window_fails():
    """A window whose GPS is majority stationary HARD FAILS."""
    result = check_stop_free_windows([(0.0, 20.0)], _parked(21))
    assert result.passed is False
    assert "stationary" in result.message


def test_partial_majority_parked_window_fails():
    """>50% parked (30 stopped + 9 moving seconds) still fails on the fraction."""
    pts = _parked(30) + [(float(t), 40.0 + (t - 29) * 0.001, -80.0) for t in range(30, 40)]
    result = check_stop_free_windows([(0.0, 39.0)], pts)
    assert result.passed is False


def test_moving_window_passes():
    """A moving window (median well above the parked floor) passes."""
    result = check_stop_free_windows([(0.0, 20.0)], _moving(21))
    assert result.passed is True


def test_empty_gps_window_falls_back_to_mask_not_auto_pass():
    """A GPS-bearing day whose window has NO in-window fixes must consult the
    stop mask, not auto-pass. With the mask marking the window stopped → FAIL;
    with no mask there is simply no evidence → PASS (never a false positive)."""
    # Day HAS GPS (>=2 fixes) but all fixes fall OUTSIDE the [0,10] window.
    gps_outside = [(float(t), 41.0 + t * 0.001, -80.0) for t in range(100, 111)]
    mask = np.ones(20, dtype=bool)  # every second stopped

    failed = check_stop_free_windows([(0.0, 10.0)], gps_outside, stop_mask=mask)
    assert failed.passed is False
    assert "masked" in failed.message

    # No mask → no evidence for the empty window → cannot fail (but did NOT
    # auto-pass on a GPS None; it simply has nothing to judge).
    passed = check_stop_free_windows([(0.0, 10.0)], gps_outside, stop_mask=None)
    assert passed.passed is True


def test_gps_less_and_mask_absent_noops():
    """Genuinely GPS-less + mask-absent legacy inputs no-op (pass)."""
    assert check_stop_free_windows([(0.0, 20.0)], None).passed is True
    assert check_stop_free_windows([(0.0, 20.0)], []).passed is True
    assert check_stop_free_windows([(0.0, 20.0)], None, stop_mask=None).passed is True


def test_no_windows_noops():
    assert check_stop_free_windows([], _parked(20)).passed is True
