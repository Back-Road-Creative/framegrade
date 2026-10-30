"""Tests for the review-set scan (synthetic score arrays only)."""

import json

import numpy as np
import pytest

from gomoveshift.pipeline import review_set_scan as rs
from gomoveshift.pipeline.stop_mask import MotionState


def _scores(n=40, bright=0.6, vib=0.5, sky=0.7, golden=0.0, motion=0.5):
    a = np.zeros((n, rs.NUM_COLUMNS))
    a[:, 0] = np.arange(n)
    a[:, rs.COL_BRIGHTNESS] = bright
    a[:, rs.COL_COLOR_VIBRANCY] = vib
    a[:, rs.COL_SKY_RATIO] = sky
    a[:, rs.COL_GOLDEN_HOUR] = golden
    a[:, rs.COL_MOTION_ENERGY] = motion
    return a


def test_every_condition_present_and_no_input_when_optionals_absent():
    out = rs.scan_review_set(_scores())
    assert tuple(out["conditions"]) == rs.CONDITIONS
    for c in ("creeping", "parked_traffic", "hdr_gradients", "gps_gap"):
        assert out["conditions"][c]["status"] == "no_input"
        assert out["conditions"][c]["windows"] == []
    assert out["conditions"]["daylight"]["status"] == "ok"


@pytest.mark.parametrize("c", ["backlight", "foliage", "wet_windshield"])
def test_undetectable_conditions_are_unsupported(c):
    s = _scores()
    out = rs.scan_review_set(
        s, states=np.zeros(40, np.int8), gps_points=[(0, 1, 1), (39, 1, 1)], tonemap={}
    )
    assert out["conditions"][c]["status"] == "unsupported"
    assert out["conditions"][c]["windows"] == []


def test_lighting_windows_carry_rule_and_thresholds():
    s = _scores()
    s[10:20, rs.COL_BRIGHTNESS] = 0.1
    out = rs.scan_review_set(s)
    (w,) = out["conditions"]["dusk_night"]["windows"]
    assert (w["start_s"], w["end_s"]) == (10.0, 20.0)
    assert w["rule"] == "brightness_low"
    assert w["thresholds"] == {"max_brightness": 0.25}
    assert w["inputs"]["brightness"] == pytest.approx(0.1)
    assert out["conditions"]["daylight"]["windows"][0]["rule"]


def test_overcast_and_golden_hour():
    s = _scores(vib=0.1)
    s[5:12, rs.COL_GOLDEN_HOUR] = 0.9
    out = rs.scan_review_set(s)
    assert out["conditions"]["overcast"]["windows"]
    (g,) = out["conditions"]["golden_hour"]["windows"]
    assert (g["start_s"], g["end_s"]) == (5.0, 12.0)


def test_tunnel_needs_dark_sky_and_brightness_drop():
    s = _scores()
    s[15:20, rs.COL_SKY_RATIO] = 0.05
    s[15:20, rs.COL_BRIGHTNESS] = 0.2
    out = rs.scan_review_set(s)
    t = out["conditions"]["tunnel"]
    assert t["heuristic"] is True
    (w,) = t["windows"]
    assert (w["start_s"], w["end_s"]) == (15.0, 20.0)
    assert "baseline_brightness_median" in w["thresholds"]
    # dark sky alone (no brightness drop) is not a tunnel
    s2 = _scores()
    s2[15:20, rs.COL_SKY_RATIO] = 0.05
    assert rs.scan_review_set(s2)["conditions"]["tunnel"]["windows"] == []


def test_creeping_and_parked_traffic_from_states():
    s = _scores(motion=0.1)
    st = np.zeros(40, np.int8)
    st[5:10] = MotionState.CREEPING
    st[20:30] = MotionState.STOPPED
    s[20:25, rs.COL_MOTION_ENERGY] = 0.8
    out = rs.scan_review_set(s, states=st)
    (c,) = out["conditions"]["creeping"]["windows"]
    assert (c["start_s"], c["end_s"]) == (5.0, 10.0)
    p = out["conditions"]["parked_traffic"]
    assert p["heuristic"] is True
    assert [(w["start_s"], w["end_s"]) for w in p["windows"]] == [(20.0, 25.0)]


def test_states_length_mismatch_raises():
    with pytest.raises(ValueError):
        rs.scan_review_set(_scores(), states=np.zeros(3, np.int8))


def test_hdr_from_tonemap_sidecar():
    out = rs.scan_review_set(
        _scores(), tonemap={"tonemap": "hable", "source_color_trc": "smpte2084"}
    )
    (w,) = out["conditions"]["hdr_gradients"]["windows"]
    assert (w["start_s"], w["end_s"]) == (0.0, 40.0)
    assert w["inputs"]["tonemap"] == "hable"


def test_gps_gap_windows():
    pts = [(float(t), 40.0, -100.0 + t * 1e-4) for t in range(0, 10)]
    pts += [(float(t), 40.0, -100.0 + t * 1e-4) for t in range(30, 40)]
    out = rs.scan_review_set(_scores(), gps_points=pts)
    g = out["conditions"]["gps_gap"]
    assert g["status"] == "ok" and g["windows"]
    assert all(w["rule"] == "gps_fix_gap_exceeds_max" for w in g["windows"])
    assert (
        rs.scan_review_set(_scores(), gps_points=[])["conditions"]["gps_gap"]["status"]
        == "no_input"
    )


def test_deterministic_and_json_serialisable():
    s = _scores()
    s[10:20, rs.COL_BRIGHTNESS] = 0.1
    a = rs.scan_review_set(s, states=np.zeros(40, np.int8), tonemap={"tonemap": "x"})
    b = rs.scan_review_set(s.copy(), states=np.zeros(40, np.int8), tonemap={"tonemap": "x"})
    assert json.dumps(a, sort_keys=False) == json.dumps(b, sort_keys=False)


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        rs.scan_review_set(np.zeros((5, 3)))


def test_default_path_unchanged_module_is_opt_in():
    """Importing the scan changes no existing behaviour: stop_mask exports intact."""
    from gomoveshift.pipeline import stop_mask

    assert int(stop_mask.MotionState.CREEPING) == 2


def test_column_indices_match_the_scorer():
    # The scan mirrors the columns to avoid the scorer's OpenCV import; a drift
    # here would read the wrong signal without any error.
    from gomoveshift.analysis import unified_scorer as us

    for name in ("COL_TIMESTAMP", "COL_COLOR_VIBRANCY", "COL_SKY_RATIO", "COL_GOLDEN_HOUR",
                 "COL_BRIGHTNESS", "NUM_COLUMNS"):  # fmt: skip
        assert getattr(rs, name) == getattr(us, name), name
    assert rs.COL_MOTION_ENERGY == us.COL_MOTION_ENERGY
