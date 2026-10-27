"""Motion states behind the per-day STOP mask: creep policy, UNCERTAIN,
hysteresis, and the persisted ``states`` array. Defaults must reproduce the
pre-states boolean mask exactly."""

import numpy as np
import pytest

from gomoveshift.config import Config, StopDetectionConfig
from gomoveshift.pipeline.stop_mask import (
    MotionState,
    build_stop_mask,
    classify_motion_states,
    states_to_stop_mask,
    stop_mask_kwargs,
)
from tests.test_stop_mask import _make_scores, _moving_gps, _parked_gps

N = 40
MID = slice(10, 30)  # clear of the centered-window edges


@pytest.mark.parametrize(
    "kmh,remove,keep",
    [
        (0.0, MotionState.STOPPED, MotionState.STOPPED),
        (2.5, MotionState.STOPPED, MotionState.CREEPING),
        (5.0, MotionState.STOPPED, MotionState.CREEPING),
        (9.0, MotionState.STOPPED, MotionState.CREEPING),
        (11.0, MotionState.MOVING, MotionState.MOVING),
    ],
)
def test_speed_bands_under_both_policies(kmh, remove, keep):
    scores, gps = _make_scores(N, motion=0.9), _moving_gps(N, kmh)
    assert (classify_motion_states(scores, gps)[MID] == remove).all()
    keep_states = classify_motion_states(scores, gps, creep_policy="keep")
    assert (keep_states[MID] == keep).all()


def test_two_kmh_is_creeping_boundary_and_below_stays_stopped():
    scores = _make_scores(N, motion=0.9)
    at = classify_motion_states(scores, _moving_gps(N, 2.2), creep_policy="keep")
    below = classify_motion_states(scores, _moving_gps(N, 1.8), creep_policy="keep")
    assert (at[MID] == MotionState.CREEPING).all()
    assert (below[MID] == MotionState.STOPPED).all()


def test_creep_never_true_in_mask_and_remove_matches_build_stop_mask():
    scores, gps = _make_scores(N, motion=0.9), _moving_gps(N, 5.0)
    states = classify_motion_states(scores, gps, creep_policy="keep")
    assert not states_to_stop_mask(states).any()
    assert build_stop_mask(scores, gps)[MID].all()
    np.testing.assert_array_equal(
        build_stop_mask(scores, gps), states_to_stop_mask(classify_motion_states(scores, gps))
    )


def test_invalid_creep_policy_rejected():
    with pytest.raises(ValueError):
        classify_motion_states(_make_scores(5), None, creep_policy="drop")


def test_passing_traffic_at_zero_kmh_stays_stopped():
    states = classify_motion_states(
        _make_scores(N, motion=0.95), _parked_gps(N), creep_policy="keep"
    )
    assert (states == MotionState.STOPPED).all()


def test_optical_only_input():
    motion = np.full(N, 0.9)
    motion[10:30] = 0.001
    states = classify_motion_states(_make_scores(N, motion=motion), None)
    assert (states[10:30] == MotionState.STOPPED).all()
    assert (states[:10] == MotionState.MOVING).all()
    assert (states[30:] == MotionState.MOVING).all()


def test_gps_gap_falls_back_to_optical_and_nan_is_uncertain():
    motion = np.full(N, 0.9)
    motion[20:] = np.nan
    states = classify_motion_states(_make_scores(N, motion=motion), _moving_gps(2, 40.0))
    assert (states[10:20] == MotionState.MOVING).all()
    assert (states[22:] == MotionState.UNCERTAIN).all()
    assert not states_to_stop_mask(states).any()


def test_uncertain_not_masked_even_beside_a_stop():
    motion = np.full(N, 0.001)
    motion[20:] = np.nan
    states = classify_motion_states(_make_scores(N, motion=motion), None)
    assert (states[:20] == MotionState.STOPPED).all()
    assert (states[20:] == MotionState.UNCERTAIN).all()


def test_short_blip_is_not_a_stop():
    motion = np.full(N, 0.9)
    motion[10:13] = 0.001
    assert (
        classify_motion_states(_make_scores(N, motion=motion), None) == MotionState.MOVING
    ).all()


# --- hysteresis -------------------------------------------------------------

_FLAP = np.array([0.001] * 10 + [0.009, 0.001] * 10 + [0.5] * 10)  # thr 0.008


def test_hysteresis_off_flapping_splits_the_stop():
    off = classify_motion_states(_make_scores(40, motion=_FLAP), None)
    assert not (off[10:30] == MotionState.STOPPED).all()


def test_hysteresis_suppresses_flapping_and_exits_above_band():
    on = classify_motion_states(
        _make_scores(40, motion=_FLAP),
        None,
        hysteresis_enabled=True,
        hysteresis_exit_multiplier=2.0,
    )
    assert (on[:30] == MotionState.STOPPED).all()  # 0.009 <= 0.016: stays stopped
    assert (on[30:] == MotionState.MOVING).all()  # 0.5 clears the exit band


def test_hysteresis_enter_unchanged():
    motion = np.full(20, 0.012)  # between enter (0.008) and exit (0.016)
    on = classify_motion_states(
        _make_scores(20, motion=motion),
        None,
        hysteresis_enabled=True,
        hysteresis_exit_multiplier=2.0,
    )
    assert (on == MotionState.MOVING).all()


def test_hysteresis_disabled_or_unit_multiplier_matches_default():
    scores = _make_scores(40, motion=_FLAP)
    base = classify_motion_states(scores, None)
    np.testing.assert_array_equal(
        base, classify_motion_states(scores, None, hysteresis_exit_multiplier=3.0)
    )
    np.testing.assert_array_equal(
        base,
        classify_motion_states(
            scores, None, hysteresis_enabled=True, hysteresis_exit_multiplier=1.0
        ),
    )


def test_hysteresis_on_gps_speed():
    kmh = np.array([5.0] * 12 + [12.0] * 6 + [5.0] * 12 + [30.0] * 10)
    t = np.arange(40.0)
    lat = 40.0 + np.concatenate([[0.0], np.cumsum(kmh[1:] / 3.6)]) / 111_320.0
    gps = [(float(a), float(b), -80.0) for a, b in zip(t, lat)]
    scores = _make_scores(40, motion=0.9)
    on = classify_motion_states(
        scores, gps, hysteresis_enabled=True, hysteresis_exit_multiplier=2.0
    )
    off = classify_motion_states(scores, gps)
    assert (on == MotionState.STOPPED).sum() > (off == MotionState.STOPPED).sum()


# --- config / kwargs / persistence -----------------------------------------


def test_kwargs_default_config_is_the_three_base_keys():
    assert set(stop_mask_kwargs(Config())) == {
        "stop_speed_kmh",
        "min_stop_duration_s",
        "optical_motion_threshold",
    }


def test_kwargs_wire_creep_and_hysteresis():
    cfg = Config()
    cfg.stop_detection = StopDetectionConfig(creep_policy="keep", creep_min_speed_kmh=3.0)
    cfg.stop_detection.hysteresis.enabled = True
    cfg.stop_detection.hysteresis.exit_multiplier = 1.5
    kw = stop_mask_kwargs(cfg)
    assert kw["creep_policy"] == "keep"
    assert kw["creep_min_speed_kmh"] == 3.0
    assert kw["hysteresis_enabled"] is True
    assert kw["hysteresis_exit_multiplier"] == 1.5
    classify_motion_states(_make_scores(10), None, **kw)  # accepted by the builder


def test_config_defaults():
    sd = StopDetectionConfig()
    assert sd.creep_policy == "remove"
    assert sd.creep_min_speed_kmh == 2.0


def test_npz_has_mask_and_states(tmp_path):
    scores, gps = _make_scores(N, motion=0.9), _moving_gps(N, 5.0)
    states = classify_motion_states(scores, gps, creep_policy="keep")
    mask = states_to_stop_mask(states)
    path = tmp_path / "stop_mask_2026-01-01.npz"
    np.savez_compressed(path, mask=mask, states=states)
    with np.load(path) as z:
        assert z["mask"].dtype == bool and z["states"].dtype == np.int8
        assert not z["mask"][z["states"] == MotionState.CREEPING].any()


def _legacy_mask(scores, gps):
    """The pre-states algorithm, verbatim, as the byte-identity reference."""
    from gomoveshift.pipeline.gps_quality import window_median_speed_mps
    from gomoveshift.pipeline.stop_mask import _sustained_runs

    n = len(scores)
    cand = np.zeros(n, dtype=bool)
    have = gps is not None and len(gps) >= 2
    for i in range(n):
        sp = None
        if have:
            sp = window_median_speed_mps(gps, max(0.0, i - 2.5), min(float(n), i + 2.5))
        cand[i] = sp < 10.0 / 3.6 if sp is not None else float(scores[i, 2]) < 0.008
    return _sustained_runs(cand, 5.0)


@pytest.mark.parametrize("seed", range(6))
def test_defaults_reproduce_legacy_mask(seed):
    rng = np.random.default_rng(seed)
    n = 120
    motion = rng.choice([0.0, 0.004, 0.02, 0.9, np.nan], size=n)
    kmh = np.repeat(rng.choice([0.0, 1.0, 5.0, 9.5, 12.0, 40.0], size=n // 10), 10)
    lat = 40.0 + np.cumsum(kmh / 3.6) / 111_320.0
    gps_all = [(float(t), float(lat[t]), -80.0) for t in range(n)]
    gps = [p for p in gps_all if not (50 <= p[0] < 70)]  # a GPS gap
    scores = _make_scores(n, motion=motion)
    for g in (gps, None, gps[:2]):
        np.testing.assert_array_equal(build_stop_mask(scores, g), _legacy_mask(scores, g))
