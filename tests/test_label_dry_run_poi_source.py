"""The label dry run records whether the destination POI source answered.

Synthetic places and coordinates only. A full Overpass outage must leave a
per-LF record that says so, with the same label a working source produces.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from gomoveshift.pipeline import destination_resolver as dr
from gomoveshift.pipeline import gps_enrichment as ge
from gomoveshift.pipeline.drive_phases import lf_listing

from test_retained_gps_claims import _BandGeocoder, _episode, _lf

_PEAK = SimpleNamespace(gps_center_lat=9.5, gps_center_lon=20.0, composite_score=0.9)
_RESERVE = {
    "type": "relation",
    "bounds": {"minlat": 9.0, "minlon": 19.0, "maxlat": 10.0, "maxlon": 21.0},
    "tags": {"leisure": "nature_reserve", "name": "Harrow Reserve"},
}


def _record(tmp_path, source, *, peaks=True):
    lf = _lf(1, (0.0, 280.0))
    lf.montage_peaks = [_PEAK] if peaks else []
    path = tmp_path / "stages" / "route_claims_dry_run.json"
    state = SimpleNamespace(gps_per_episode=_episode(), per_lf_overrides={})
    with (
        patch.object(dr, "_default_poi_source", source),
        patch.object(ge, "derive_per_lf_locations", return_value={}),
        patch.object(ge, "_differentiate_and_revalidate_siblings", return_value=None),
        patch.object(ge, "canonical_lf_ost_text", return_value="Harrow, Westmark"),
    ):
        lf_listing._render_ost_text_by_lf([lf], {}, _BandGeocoder(), state, report={"path": path})
    (rec,) = json.loads(path.read_text())["lfs"]
    return rec


def _down(_bbox):
    raise RuntimeError("every Overpass endpoint failed")


def test_outage_is_recorded_and_the_label_is_unchanged(tmp_path):
    down = _record(tmp_path / "a", _down)
    up = _record(tmp_path / "b", lambda _b: [_RESERVE])
    assert down["destination_source"] == "failed"
    assert up["destination_source"] == "overpass"
    assert down["label"] == up["label"] == "Harrow, Westmark"
    assert down["containment_proven"] is False


def test_no_peaks_or_no_candidate_is_none_not_failed(tmp_path):
    assert _record(tmp_path / "a", _down, peaks=False)["destination_source"] == "none"
    assert _record(tmp_path / "b", lambda _b: [])["destination_source"] == "none"


def test_resolve_destination_return_type_is_unchanged():
    assert dr.resolve_destination([_PEAK], poi_source=_down) is None
    assert dr.resolve_destination([_PEAK], poi_source=lambda _b: [_RESERVE]) == "Harrow Reserve"
