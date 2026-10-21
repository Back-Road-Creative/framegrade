"""Zero-tolerance place gate: a place a text names holds every retained second.

Synthetic places only (Harrow, Pellmere, Ostry, Ashby/Brell County, Westmark);
the proof tests use a real state/county name because proofs pin Census FIPS.
"""

from __future__ import annotations

import pytest

from gomoveshift.pipeline.boundary_proof import _SOURCES, pinned_boundary_proof
from gomoveshift.pipeline.final_text_record import (
    DESCRIPTION_LAYOUT,
    FinalTextReviewError,
    check_final_text,
)
from gomoveshift.pipeline.route_claims import (
    RouteClaimRefusedError,
    RouteEvidence,
    RouteWindow,
    _Bin,
    _region_ok,
    label_issues,
    measure_route,
    place_text_issues,
    recheck_place_text,
    retained_windows_key,
    select_supported_label,
)
from gomoveshift.pipeline.title_decision import _retained_title_issues

BANDS = {
    "H": {"town": "Harrow", "county": "Ashby County", "state": "Westmark"},
    "P": {"town": "Pellmere", "county": "Ashby County", "state": "Westmark"},
    "O": {"town": "Ostry", "county": "Brell County", "state": "Westmark"},
}


class Geo:
    def __init__(self, fail_above=None):
        self.fail_above = fail_above

    def reverse_geocode_detailed(self, lat, lon):
        if self.fail_above is not None and lat >= self.fail_above:
            raise RuntimeError("down")
        return dict(BANDS["H" if lat < 10.0 else "P" if lat < 10.5 else "O"])


def _win(seg, start, end, lat_of_t, skip=()):
    fixes = [
        (float(t), lat_of_t(t), 20.0)
        for t in range(int(start), int(end))
        if not any(a <= t < b for a, b in skip)
    ]
    return RouteWindow(segment_id=seg, start_s=start, end_s=end, fixes=fixes)


def _harrow_with_leak(leak_lat=10.2, leak_s=20):
    # 300 s in Harrow, then `leak_s` s elsewhere (below the old 60 s floor).
    return _win("lf1", 0.0, 300.0 + leak_s, lambda t: 9.5 if t < 300 else leak_lat)


def _proof(level, name, key, state_fips="45"):
    return {
        "verified": True,
        "basis": "pinned_tiger_polygon_bounded_speed_and_error",
        "source": f"TIGER2026/{level.upper()}/{_SOURCES[level][0]}",
        "sha256": _SOURCES[level][1],
        "region": name,
        "statefp": state_fips,
        "recording_date": "2026-02-20",
        "max_speed_m_s": 70.0,
        "gps_error_m": 100.0,
        "max_gap_s": 30.0,
        "windows_key": key,
    }


def _ev(bins, key="k", proofs=None, gaps=()):
    return RouteEvidence(
        retained_seconds=sum(b.seconds for b in bins) + sum(g.seconds for g in gaps),
        gaps=list(gaps),
        places=[],
        unsettled_seconds=0.0,
        bins=bins,
        boundary_proofs=proofs or {},
        windows_key=key,
    )


# -- 1. proofs are bound to the exact retained seconds ---------------------------
def test_windows_key_is_deterministic_and_sensitive_to_one_second():
    a = [_win("lf1", 0.0, 100.0, lambda t: 9.5)]
    b = [_win("lf1", 0.0, 101.0, lambda t: 9.5)]
    assert retained_windows_key(a) == retained_windows_key(list(a))
    assert retained_windows_key(a) != retained_windows_key(b)
    assert measure_route(a, Geo()).windows_key == retained_windows_key(a)


def test_proof_over_other_windows_does_not_cover_unresolved_bins():
    a = [_win("lf1", 0.0, 100.0, lambda t: 9.5 if t < 60 else 10.7)]
    b = [_win("lf1", 0.0, 101.0, lambda t: 9.5 if t < 60 else 10.7)]
    ev = measure_route(b, Geo(fail_above=10.5))
    ev.boundary_proofs["county"] = _proof("county", "Ashby County", retained_windows_key(a))
    claim = ev.region_claim("county")
    assert claim["samples_outside"] > 0
    assert "boundary_proof" not in claim
    ev.boundary_proofs["county"]["windows_key"] = retained_windows_key(b)
    assert ev.region_claim("county")["samples_outside"] == 0


def test_pinned_proof_without_a_key_is_refused():
    claim = {"level": "state", "name": "South Carolina", "windows_key": "k"}
    assert pinned_boundary_proof(_proof("state", "South Carolina", "k"), claim)
    no_key = _proof("state", "South Carolina", "k")
    del no_key["windows_key"]
    assert not pinned_boundary_proof(no_key, claim)
    assert not pinned_boundary_proof(_proof("state", "South Carolina", "other"), claim)


# -- 2. zero tolerance even with a valid proof ------------------------------------
def test_valid_proof_does_not_excuse_a_sample_geocoded_elsewhere():
    bins = [_Bin(60.0, "Ash County", "South Carolina")] * 2 + [
        _Bin(10.0, "Pine County", "South Carolina")
    ]
    ev = _ev(bins, proofs={"county": _proof("county", "Ash County", "k")})
    claim = ev.region_claim("county")
    assert claim["boundary_proof"]
    assert _region_ok(claim, ev, 1.0)


# -- 3. one shared function ---------------------------------------------------------
def _measured(leak_lat=10.2, leak_s=20, **kw):
    return measure_route([_harrow_with_leak(leak_lat, leak_s)], Geo(**kw))


def test_leak_below_old_floor_fails_the_named_place():
    issues = place_text_issues("Harrow", _measured())
    assert issues and any("Pellmere" in i and "20" in i for i in issues)


def test_text_naming_both_places_or_their_county_or_state_is_covered():
    ev = _measured()
    assert place_text_issues("Harrow and Pellmere", ev) == []
    assert place_text_issues("Ashby County", ev) == []
    assert place_text_issues("Ashby", ev) == []
    assert place_text_issues("Harrow, Westmark", ev) == []


def test_a_state_abbreviation_inside_a_url_is_not_a_named_state():
    # A preview link like youtu.be/PREVIEW-NO-LF-ID-1 must not read as "Idaho"
    # and void the county the text does name.
    ev = _measured()
    assert place_text_issues("Ashby County https://youtu.be/PREVIEW-NO-LF-ID-1", ev) == []
    assert place_text_issues("Ashby County youtu.be/OK-IN-ME", ev) == []


def test_unresolved_seconds_outside_a_proof_fail():
    ev = _measured(leak_lat=10.7, fail_above=10.5)
    assert any("unresolved" in i for i in place_text_issues("Harrow", ev))
    # even naming the county fails without a proof covering those seconds
    assert place_text_issues("Ashby County", ev)


def test_gap_seconds_fail_unless_a_named_region_has_a_matching_proof():
    w = _win("lf1", 0.0, 300.0, lambda t: 9.5, skip=[(100, 200)])
    ev = measure_route([w], Geo())
    assert ev.gap_seconds > 0
    assert any("gap" in i for i in place_text_issues("Harrow", ev))
    assert place_text_issues("Ashby County", ev)
    ev.boundary_proofs["county"] = _proof("county", "Ashby County", ev.windows_key)
    ev.boundary_proofs["county"]["statefp"] = None
    # Proof metadata for a synthetic state cannot validate, so still fails closed.
    assert place_text_issues("Ashby County", ev)


def test_proof_covers_unresolved_seconds_for_the_named_region():
    bins = [_Bin(60.0, "Ash County", "South Carolina", "Alder")] * 2 + [_Bin(10.0, None, None)]
    ev = _ev(bins, proofs={"county": _proof("county", "Ash County", "k")})
    assert place_text_issues("Ash County", ev) == []
    assert place_text_issues("Alder", ev)  # a settlement name alone does not get the proof


def test_text_naming_nothing_known_has_no_place_issue():
    assert place_text_issues("A nice drive", _measured()) == []


# -- 4. every label surface ------------------------------------------------------------
def test_lf_label_with_small_leak_is_refused_by_label_issues():
    assert any("Pellmere" in i for i in label_issues("Harrow", _measured()))


def test_region_fallback_must_itself_pass_zero_tolerance():
    ev = _measured(leak_lat=10.7, fail_above=10.5)
    with pytest.raises(RouteClaimRefusedError):
        select_supported_label("Harrow", ev)


def test_title_with_small_leak_is_refused():
    issues = _retained_title_issues("Harrow drive in 4K", _measured())
    assert any("Pellmere" in i for i in issues)
    assert _retained_title_issues("Harrow and Pellmere drive in 4K", _measured()) == []


# -- description + tags before upload ------------------------------------------------------
@pytest.fixture(autouse=True)
def _review_accepts(monkeypatch):
    # These tests exercise the place gate, which runs after the review/byte-bind.
    monkeypatch.setattr(
        "gomoveshift.pipeline.final_text_record.verify_review", lambda *a, **k: None
    )


_REVIEW = {"description_layout": DESCRIPTION_LAYOUT, "surfaces": []}


def _ctx(ev):
    return {
        "lf_index": 1,
        "visible": {},
        "source_fingerprint": "s",
        "plan_fingerprint": "p",
        "policy_version": "v",
        "route_record": ev.record(),
    }


def test_description_naming_a_place_with_leaked_seconds_is_refused_before_upload():
    ctx = _ctx(_measured())
    with pytest.raises(FinalTextReviewError) as exc:
        check_final_text(_REVIEW, ctx, "t", description="Driving Harrow today", tags=["Harrow"])
    assert "place_outside_text" in exc.value.reasons


def test_tags_are_one_text_so_a_union_of_tags_covers():
    ctx = _ctx(_measured())
    check_final_text(_REVIEW, ctx, "t", description="drive", tags=["Harrow", "Pellmere"])


# -- offline recheck of a finished render ----------------------------------------------------
def test_recheck_place_text_from_a_persisted_record():
    rec = _measured().record()
    assert recheck_place_text(rec, "Harrow")
    assert recheck_place_text(rec, "Harrow and Pellmere") == []


def test_recheck_refuses_a_legacy_record_with_no_bins():
    rec = _measured().record()
    del rec["bins"]
    assert any("bins" in i for i in recheck_place_text(rec, "Harrow"))
