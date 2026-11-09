"""A day-GPX-sliced track is judged at the day-track fix gap, not the GPMF one.

``route_claims.GAP_S`` (5 s) models a GoPro's GPMF: one fix a second, so a
longer silence is lock loss and nothing may be claimed across it. A sidecar the
day-GPX slicer wrote (``creator="gms-day-gpx-slicer"``) is a phone logger:
bursty, with holes of a few seconds while the phone sleeps (Clonakilty day 12:
886 fixes, median 1 s apart, 58 holes of 5-20 s summing 439 s; the longest hole
in the whole day track is 20 s). Judged at 5 s the planner cut 1085 s and the
day fell under the 3600 s part floor. ``DAY_TRACK_GAP_S`` is one evidence bin
(``BIN_S``): a hole no longer than the bin adds no uncertainty the sampled
method already accepts. The tolerance is read from the track's ``creator``, so
a GPMF track, or any track of unknown origin, keeps 5 s.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gomoveshift.analysis.gps_parser import parse_gpx_file
from gomoveshift.analysis.shorts_models import GPSData, GPSTrackpoint
from gomoveshift.pipeline import route_claims
from gomoveshift.pipeline.gps_enrichment import _build_route_windows_by_lf
from gomoveshift.pipeline.ingest.day_gpx_slicer import MARKER
from gomoveshift.pipeline.route_claims import (
    BIN_S,
    DAY_TRACK_GAP_S,
    GAP_S,
    RouteWindow,
    fix_gap_s,
    measure_route,
)

HARROW = {"town": "Harrow", "county": "Ashby County", "state": "Westmark"}


class _OnePlace:
    def reverse_geocode_detailed(self, lat, lon):
        return dict(HARROW)


def _gpx(path, creator):
    attr = f' creator="{creator}"' if creator is not None else ""
    pts = "".join(
        f'<trkpt lat="9.{i:03d}" lon="20.0"><time>2026-09-17T10:00:{i:02d}Z</time></trkpt>'
        for i in range(3)
    )
    path.write_text(
        f'<?xml version="1.0"?><gpx version="1.1"{attr}><trk><trkseg>{pts}</trkseg></trk></gpx>'
    )
    return path


def _window(hole_at, hole_len, end=300.0, **kw):
    fixes = [
        (float(t), 9.5, 20.0)
        for t in range(0, int(end) + 1)
        if not hole_at < t < hole_at + hole_len
    ]
    return RouteWindow("lf01-s0", 0.0, end, fixes, **kw)


# --- the tolerance -----------------------------------------------------------


def test_day_track_gap_is_one_evidence_bin():
    assert DAY_TRACK_GAP_S == BIN_S == 30.0
    assert GAP_S == 5.0


def test_parser_keeps_the_gpx_creator(tmp_path):
    assert parse_gpx_file(_gpx(tmp_path / "a.gpx", MARKER)).creator == MARKER
    assert parse_gpx_file(_gpx(tmp_path / "b.gpx", "Some Phone App")).creator == "Some Phone App"
    assert parse_gpx_file(_gpx(tmp_path / "c.gpx", None)).creator is None


@pytest.mark.parametrize(
    "gps_data, expected",
    [
        (GPSData(trackpoints=[], creator=MARKER), DAY_TRACK_GAP_S),
        (GPSData(trackpoints=[]), GAP_S),
        (GPSData(trackpoints=[], creator="Some Phone App"), GAP_S),
        ({"trackpoints": [], "creator": MARKER}, DAY_TRACK_GAP_S),
        ({"trackpoints": []}, GAP_S),
        (None, GAP_S),
    ],
)
def test_fix_gap_s_reads_the_track_origin(gps_data, expected):
    assert fix_gap_s(gps_data) == expected


# --- the claim gate ----------------------------------------------------------


def test_a_window_judged_at_the_day_track_gap_accepts_a_phone_hole():
    ev = measure_route([_window(100, 20, gap_s=DAY_TRACK_GAP_S)], _OnePlace())
    assert ev.gaps == []
    assert ev.coverage == 1.0
    assert ev.record()["gap_threshold_s"] == 30.0


def test_a_window_judged_at_the_day_track_gap_still_refuses_a_longer_hole():
    ev = measure_route([_window(100, 40, gap_s=DAY_TRACK_GAP_S)], _OnePlace())
    assert len(ev.gaps) == 1
    assert ev.gap_seconds == pytest.approx(40.0)


def test_a_window_without_its_own_gap_keeps_the_gpmf_rule():
    ev = measure_route([_window(100, 20)], _OnePlace())
    assert len(ev.gaps) == 1
    assert ev.record()["gap_threshold_s"] == 5.0


def test_mixed_windows_record_every_threshold_used():
    ev = measure_route([_window(100, 2), _window(100, 2, gap_s=DAY_TRACK_GAP_S)], _OnePlace())
    assert ev.record()["gap_threshold_s"] == [5.0, 30.0]


# --- the windows carry the origin of their track -----------------------------


def _plan():
    seg0 = SimpleNamespace(start_time=0.0, end_time=100.0, source_file_index=0)
    seg1 = SimpleNamespace(start_time=0.0, end_time=100.0, source_file_index=1)
    return SimpleNamespace(
        regular_lfs=[
            SimpleNamespace(index=1, segments=[seg0]),
            SimpleNamespace(index=2, segments=[seg1]),
        ]
    )


def _track(creator):
    tps = [GPSTrackpoint(9.5, 20.0, None, 1_000_000.0 + i) for i in range(101)]
    return GPSData(trackpoints=tps, creator=creator)


def test_route_windows_take_their_gap_from_the_episode_track():
    windows = _build_route_windows_by_lf(
        _plan(), [(_track(MARKER), 1_000_000.0), (_track(None), 1_000_000.0)]
    )
    assert windows[1][0].gap_s == DAY_TRACK_GAP_S
    assert windows[2][0].gap_s == GAP_S


def test_route_windows_read_the_origin_of_a_resumed_dict_track():
    tps = [
        {"lat": 9.5, "lon": 20.0, "elevation": None, "timestamp": 1_000_000.0 + i}
        for i in range(101)
    ]
    entry = ({"trackpoints": tps, "creator": MARKER}, 1_000_000.0)
    windows = _build_route_windows_by_lf(_plan(), [entry, entry])
    assert windows[1][0].gap_s == DAY_TRACK_GAP_S


def test_day_fix_gap_is_uniform_or_fails_closed():
    marked, gpmf = (_track(MARKER), 0.0), (_track(None), 0.0)
    assert route_claims.day_fix_gap_s([marked, marked, (None, 0.0)]) == DAY_TRACK_GAP_S
    assert route_claims.day_fix_gap_s([gpmf]) == GAP_S
    # Mixed origins on one day: the stricter rule, never the looser.
    assert route_claims.day_fix_gap_s([marked, gpmf]) == GAP_S
    assert route_claims.day_fix_gap_s([(None, 0.0)]) == GAP_S
