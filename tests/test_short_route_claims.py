"""A Short's burned place is checked against its OWN clip's GPS.

The long-form rule (route_claims) at Short scale, with no new
number: one sample per ``GAP_S`` bin, any interval over
``GAP_S`` refuses, any unsettled bin refuses, every place any bin resolves to
must be named, and a region stands in only at full coverage (unchanged). Every
refusal is reported together, before any Short burns.

Synthetic places only ("Harrow", "Pellmere", "Ostry", "Ashby County",
"Westmark" and "Vell" are invented); no production trace, label or coordinate
appears here.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

import gomoveshift.pipeline.day_pipeline as dp
import gomoveshift.pipeline.route_claims as rc
from gomoveshift.pipeline.drive_phases import phase_bc_day_first
from gomoveshift.pipeline.route_claims import (
    RouteClaimRefusedError,
    make_short_route_fn,
    short_route_window,
)

from test_retained_gps_claims import OSTRY, _BandGeocoder

DATE = "2026-01-01"


class _VellGeocoder(_BandGeocoder):
    """Ostry's band lies in another state, so no region holds the whole clip."""

    def reverse_geocode_detailed(self, lat, lon):
        hit = super().reverse_geocode_detailed(lat, lon)
        return {**hit, "state": "Vell"} if hit["town"] == OSTRY["town"] else hit


class _UnsettledGeocoder(_BandGeocoder):
    """The Pellmere band resolves to nothing at all (no town, county or state)."""

    def reverse_geocode_detailed(self, lat, lon):
        hit = super().reverse_geocode_detailed(lat, lon)
        return {} if 10.0 <= lat < 10.5 else hit


def _fixes(end=38.0, *, skip=(), lat_of_t=lambda t: 9.5, extra=()):
    # One fix a second, a moving trace (~22 m/s north) so the stop-free gate
    # never trips. ``skip`` (a, b) drops integer t with a <= t < b.
    ts = [float(t) for t in range(int(end)) if not any(a <= t < b for a, b in skip)]
    ts = sorted(ts + [float(t) for t in extra])
    return [(t, lat_of_t(t), 20.0 + t * 0.0002) for t in ts]


def _run(
    monkeypatch, tmp_path, points, *, text="Harrow", ranks=((1, 0.0, 38.0),), geo=None, state=None
):
    rendered = []
    monkeypatch.setattr(dp, "_render_short_ffmpeg", lambda **kw: rendered.append(kw) or True)
    monkeypatch.setattr(dp, "compute_day_file_boundaries", lambda b: [])
    monkeypatch.setattr(
        dp, "assign_shorts_to_lfs_by_source_window", lambda **k: {1: range(len(ranks))}
    )
    _, _, entries = dp._render_day_shorts(
        day_dir=tmp_path,
        date=DATE,
        day_batch=SimpleNamespace(episodes=[]),
        short_ranges=[SimpleNamespace(rank=r, start_sec=a, end_sec=b) for r, a, b in ranks],
        allow_short_count=True,
        gps_points_by_date={DATE: points},
        lf_plans=[SimpleNamespace(index=1)],
        per_lf_ost_text={1: "Harrow to Pellmere"},
        per_lf_locations={1: "Ashby County"},
        location="Drive Wide",
        short_ost_text_fn=lambda pts, sr: text,
        short_route_fn=make_short_route_fn(geo or _BandGeocoder()),
        master_source_for_extract=Path("/dev/null"),
        state=state
        if state is not None
        else {"ost_audit_confirmed": True, "ost_audit_confirmed_this_run": True},
    )
    return entries, rendered


def _pellmere_5s(t):
    return 10.2 if 20 <= t < 25 else 9.5


def test_a_short_inside_its_named_place_burns_unchanged(monkeypatch, tmp_path):
    entries, rendered = _run(monkeypatch, tmp_path, _fixes())
    assert [e["ost_text"] for e in entries] == ["Harrow"]
    assert [r["location"] for r in rendered] == ["Harrow"]


# S1
def test_s1_a_12s_dropout_refuses_the_short_naming_the_gap(monkeypatch, tmp_path):
    with pytest.raises(RouteClaimRefusedError, match=r"rank 1:(.|\n)*gap_seconds 12\.0"):
        _run(monkeypatch, tmp_path, _fixes(skip=[(10, 21)]))


# S2
def test_s2_a_4s_dropout_passes(monkeypatch, tmp_path):
    entries, _ = _run(monkeypatch, tmp_path, _fixes(skip=[(10, 13)]))
    assert entries[0]["ost_text"] == "Harrow"


# S3
def test_s3_exactly_gap_s_passes(monkeypatch, tmp_path):
    entries, _ = _run(monkeypatch, tmp_path, _fixes(skip=[(10, 14)]))
    assert entries[0]["ost_text"] == "Harrow"


def test_s3_just_over_gap_s_refuses(monkeypatch, tmp_path):
    with pytest.raises(RouteClaimRefusedError, match=r"rank 1:(.|\n)*gap_seconds 5\.0"):
        _run(monkeypatch, tmp_path, _fixes(skip=[(10, 15)], extra=[14.01]))


# S4
def test_s4_one_5s_bin_in_a_second_place_refuses_unless_named(monkeypatch, tmp_path):
    pts = _fixes(lat_of_t=lambda t: 10.7 if 20 <= t < 25 else 9.5)
    with pytest.raises(RouteClaimRefusedError, match=r"rank 1:(.|\n)*omits 'Ostry'"):
        _run(monkeypatch, tmp_path, pts, geo=_VellGeocoder())
    entries, _ = _run(monkeypatch, tmp_path, pts, text="Harrow to Ostry", geo=_VellGeocoder())
    assert entries[0]["ost_text"] == "Harrow to Ostry"


def test_s4_a_5s_second_place_inside_the_county_falls_back_to_the_county(monkeypatch, tmp_path):
    # Region fallback unchanged: full coverage, nothing outside the county.
    entries, rendered = _run(monkeypatch, tmp_path, _fixes(lat_of_t=_pellmere_5s))
    assert entries[0]["ost_text"] == "Ashby County, Westmark"
    assert [r["location"] for r in rendered] == ["Ashby County, Westmark"]


# S5
def test_s5_one_unsettled_bin_refuses(monkeypatch, tmp_path):
    with pytest.raises(RouteClaimRefusedError, match=r"rank 1:(.|\n)*unsettled_seconds 5\.0"):
        _run(monkeypatch, tmp_path, _fixes(lat_of_t=_pellmere_5s), geo=_UnsettledGeocoder())


# S6
def test_s6_long_form_constants_and_defaults_are_unchanged():
    assert (rc.GAP_S, rc.BIN_S, rc.MIN_PLACE_S, rc.REGION_MIN_COVERAGE) == (5.0, 30.0, 60.0, 1.0)
    params = inspect.signature(rc.measure_route).parameters
    assert params["bin_s"].default == rc.BIN_S
    assert params["gap_s"].default == rc.GAP_S
    assert params["min_place_s"].default == rc.MIN_PLACE_S
    for fn in (rc.label_issues, rc.select_supported_label):
        assert inspect.signature(fn).parameters["unknown_floor_s"].default == rc.MIN_PLACE_S


def test_a_gap_over_gap_s_with_under_gap_s_inside_the_clip_still_refuses(monkeypatch, tmp_path):
    # A 7 s fix-to-fix dropout (34 -> 41) that the clip's end (38 s) cuts at 4 s
    # of retained overlap. A gap over GAP_S refuses however little of it the clip
    # keeps, so a floor of GAP_S (4 s < 5 s would pass) is not "no floor".
    pts = _fixes(end=45.0, skip=[(35, 41)])
    with pytest.raises(RouteClaimRefusedError, match=r"rank 1:(.|\n)*gap_seconds 4\.0"):
        _run(monkeypatch, tmp_path, pts)


def test_the_short_scale_uses_no_new_number():
    assert rc._SHORT_FLOOR_S == 0.0


def test_a_short_no_region_contains_is_refused(monkeypatch, tmp_path):
    pts = _fixes(lat_of_t=lambda t: 9.5 if t < 20 else 10.7)
    with pytest.raises(RouteClaimRefusedError, match=r"rank 1:(.|\n)*omits 'Ostry'"):
        _run(monkeypatch, tmp_path, pts, geo=_VellGeocoder())


def test_every_refused_short_is_reported_together_and_nothing_burns(monkeypatch, tmp_path):
    rendered = []
    monkeypatch.setattr(dp, "_render_short_ffmpeg", lambda **kw: rendered.append(kw) or True)
    pts = _fixes(end=120.0, skip=[(10, 22), (90, 104)])
    with pytest.raises(RouteClaimRefusedError) as exc:
        _run(monkeypatch, tmp_path, pts, ranks=((1, 0.0, 38.0), (2, 80.0, 118.0)))
    assert "rank 1:" in str(exc.value) and "rank 2:" in str(exc.value)
    assert rendered == []


def test_the_short_window_sees_the_dropout_a_clip_edge_falls_inside():
    # Fixes stop 6 s before the clip starts: the gap reaches into the clip.
    w = short_route_window(1, 30.0, 60.0, _fixes(end=70.0, skip=[(24, 36)]))
    assert (w.start_s, w.end_s, w.prev_fix_s, w.next_fix_s) == (30.0, 60.0, 23.0, 60.0)
    assert w.fixes[0][0] == 36.0


def test_phase_bc_wires_the_short_route_check_into_the_day_pipeline():
    for fn in (dp.run_day_pipeline, dp.run_all_days, dp._render_day_shorts):
        assert "short_route_fn" in inspect.signature(fn).parameters
    assert "make_short_route_fn(geocoder)" in inspect.getsource(phase_bc_day_first)
