"""Retained-footage GPS claims: a label may claim only what the
retained footage's own GPS proves, at matching specificity.

Synthetic places only — "Harrow", "Pellmere" and "Ashby County, Westmark" are
invented; no production trace, title or coordinate appears here.
"""

from __future__ import annotations

import pytest

from gomoveshift.pipeline.route_claims import (
    RouteClaimRefusedError,
    RouteWindow,
    label_issues,
    measure_route,
    select_supported_label,
)

# Latitude bands → place. Harrow lat < 10.0, Pellmere 10.0..10.5, Ostry (other
# county) >= 10.5. Bands are far from any real coordinate.
HARROW = {"town": "Harrow", "county": "Ashby County", "state": "Westmark"}
PELLMERE = {"town": "Pellmere", "county": "Ashby County", "state": "Westmark"}
OSTRY = {"town": "Ostry", "county": "Brell County", "state": "Westmark"}


class _BandGeocoder:
    def __init__(self, *, fail_above=None, no_county=False):
        self.calls = 0
        self.fail_above = fail_above
        self.no_county = no_county

    def reverse_geocode_detailed(self, lat, lon):
        self.calls += 1
        if self.fail_above is not None and lat >= self.fail_above:
            raise RuntimeError("geocoder down")
        hit = dict(HARROW if lat < 10.0 else PELLMERE if lat < 10.5 else OSTRY)
        if self.no_county:
            hit["county"] = None
        return hit


def _window(seg, start, end, lat_of_t, *, step=1.0, skip=()):
    """Fixes every ``step`` s over [start, end), latitude from ``lat_of_t``."""
    fixes = []
    t = start
    while t < end:
        if not any(a <= t < b for a, b in skip):
            fixes.append((t, lat_of_t(t), 20.0))
        t += step
    return RouteWindow(segment_id=seg, start_s=start, end_s=end, fixes=fixes)


def _harrow_pellmere_harrow(seg="lf01", start=0.0, end=900.0):
    # 0-300 Harrow, 300-600 Pellmere, 600-900 back in Harrow (route returns).
    return _window(seg, start, end, lambda t: 10.2 if 300 <= t - start < 600 else 9.5)


def test_multi_place_route_refuses_single_town_label():
    ev = measure_route([_harrow_pellmere_harrow()], _BandGeocoder())
    assert [(p.name, p.required) for p in ev.places] == [("Harrow", True), ("Pellmere", True)]
    issues = label_issues("Harrow, Westmark", ev)
    assert any("Pellmere" in i for i in issues)


def test_route_returning_to_start_is_not_an_a_to_b_endpoint():
    ev = measure_route([_harrow_pellmere_harrow()], _BandGeocoder())
    # Harrow is both start and end; its retained time is both visits summed.
    harrow = next(p for p in ev.places if p.name == "Harrow")
    assert harrow.retained_seconds == pytest.approx(600.0)
    assert label_issues("Harrow to Pellmere, Westmark", ev) == []
    assert label_issues("Harrow and Pellmere, Westmark", ev) == []


def test_multi_place_route_falls_back_to_contained_county():
    ev = measure_route([_harrow_pellmere_harrow()], _BandGeocoder())
    label, note = select_supported_label("Harrow, Westmark", ev)
    assert label == "Ashby County, Westmark"
    assert "Pellmere" in note
    claim = ev.region_claim("county")
    assert claim["coverage"] == pytest.approx(1.0)
    assert claim["gaps"] == []
    assert claim["samples_outside"] == 0


def test_valid_single_place_keeps_its_label():
    ev = measure_route([_window("lf01", 0, 600, lambda t: 9.5)], _BandGeocoder())
    label, note = select_supported_label("Harrow, Westmark", ev)
    assert (label, note) == ("Harrow, Westmark", "")


def test_excluded_footage_supplies_no_claim():
    # Two retained windows in Harrow; the Pellmere stretch between them was cut.
    windows = [
        _window("lf01-s0", 0, 300, lambda t: 9.5),
        _window("lf01-s1", 900, 1200, lambda t: 9.5),
    ]
    ev = measure_route(windows, _BandGeocoder())
    assert [p.name for p in ev.places] == ["Harrow"]
    assert ev.retained_seconds == pytest.approx(600.0)
    assert select_supported_label("Harrow, Westmark", ev)[0] == "Harrow, Westmark"


def test_brief_pass_through_is_recorded_but_not_required():
    # 20 s in Pellmere, below the place floor: recorded, not required.
    w = _window("lf01", 0, 600, lambda t: 10.2 if 290 <= t < 310 else 9.5)
    ev = measure_route([w], _BandGeocoder(), bin_s=10.0)
    pellmere = next(p for p in ev.places if p.name == "Pellmere")
    assert pellmere.required is False
    assert label_issues("Harrow, Westmark", ev) == []


def test_gap_blocks_region_fallback_with_one_actionable_refusal():
    w = _window("lf01", 0, 900, lambda t: 10.2 if 300 <= t < 600 else 9.5, skip=[(400, 436)])
    ev = measure_route([w], _BandGeocoder())
    assert _gap_tuples(ev) == [("lf01", 399.0, 436.0)]
    assert ev.coverage < 1.0
    with pytest.raises(RouteClaimRefusedError) as exc:
        select_supported_label("Harrow, Westmark", ev)
    msg = str(exc.value)
    assert "Pellmere" in msg and "gap" in msg and "--per-lf-location" in msg


def test_missing_endpoint_is_a_gap_never_a_fabricated_fix():
    # GPS starts 60 s late: the head is a gap, not an interpolated Harrow fix.
    w = _window("lf01", 0, 600, lambda t: 9.5, skip=[(0, 60)])
    ev = measure_route([w], _BandGeocoder())
    assert _gap_tuples(ev) == [("lf01", 0.0, 60.0)]
    harrow = ev.places[0]
    assert harrow.retained_seconds == pytest.approx(540.0)


def test_bad_fixes_are_dropped_not_geocoded():
    w = _window("lf01", 0, 300, lambda t: 9.5)
    w.fixes[100:140] = [(t, 0.0, 0.0) for t, _la, _lo in w.fixes[100:140]]
    geo = _BandGeocoder()
    ev = measure_route([w], geo)
    assert _gap_tuples(ev) == [("lf01", 99.0, 140.0)]
    assert all(p.name == "Harrow" for p in ev.places)


def test_route_leaving_the_region_gets_no_region_fallback():
    # Both named places are in Ashby County, but a retained stretch is in Brell.
    w = _window(
        "lf01",
        0,
        900,
        lambda t: 10.2 if 300 <= t < 600 else 10.7 if 600 <= t < 640 else 9.5,
    )
    ev = measure_route([w], _BandGeocoder(), bin_s=10.0)
    assert ev.region_claim("county")["samples_outside"] > 0
    label, note = select_supported_label("Harrow, Westmark", ev)
    assert label == "Westmark"  # falls back further, to the state that contains it all


def test_unavailable_boundary_refuses_instead_of_guessing():
    ev = measure_route([_harrow_pellmere_harrow()], _BandGeocoder(no_county=True))
    claim = ev.region_claim("county")
    assert claim is None or claim["samples_outside"] > 0
    label, _ = select_supported_label("Harrow, Westmark", ev)
    assert label == "Westmark"


def test_geocoder_failure_is_unresolved_time_not_a_place():
    ev = measure_route([_harrow_pellmere_harrow()], _BandGeocoder(fail_above=10.0))
    assert [p.name for p in ev.places] == ["Harrow"]
    assert ev.unsettled_seconds == pytest.approx(300.0)
    with pytest.raises(RouteClaimRefusedError):
        select_supported_label("Harrow, Westmark", ev)


def test_operator_label_is_checked_not_rescued():
    ev = measure_route([_harrow_pellmere_harrow()], _BandGeocoder())
    with pytest.raises(RouteClaimRefusedError) as exc:
        select_supported_label("Harrow, Westmark", ev, operator=True)
    assert "operator" in str(exc.value)


# --- through the real pre-burn entry point (gps_enrichment._enrich_per_lf_gps) ---

from types import SimpleNamespace  # noqa: E402
from unittest.mock import patch  # noqa: E402

from gomoveshift.analysis.shorts_models import GPSData, GPSTrackpoint  # noqa: E402
from gomoveshift.pipeline import gps_enrichment as ge  # noqa: E402
from gomoveshift.pipeline.drive_phases import lf_listing  # noqa: E402

T0 = 1_000_000.0


def _episode(seconds=1200, lat_of_t=None, origin=T0):
    """One episode: 0-300 Harrow, 300-600 Pellmere, 600-1200 Ostry (other county)."""
    lat_of_t = lat_of_t or (lambda t: 9.5 if t < 300 else 10.2 if t < 600 else 10.8)
    tps = [
        GPSTrackpoint(
            lat=lat_of_t(t),
            lon=0.0 if lat_of_t(t) == 0.0 else 20.0,
            elevation=None,
            timestamp=origin + t,
            media_s=float(t),
        )
        for t in range(seconds)
    ]
    return [(GPSData(trackpoints=tps), origin)]


def _lf(index, *spans):
    """Segments ``(start, end)`` in episode 0, or ``(episode, start, end)``."""
    segs = [
        SimpleNamespace(source_file_index=s[0], start_time=s[1], end_time=s[2])
        if len(s) == 3
        else SimpleNamespace(source_file_index=0, start_time=s[0], end_time=s[1])
        for s in spans
    ]
    return SimpleNamespace(index=index, segments=segs)


def _enrich(lfs, labels, overrides=None, episodes=None):
    state = SimpleNamespace(
        gps_per_episode=episodes or _episode(),
        gpx_files=[],
        episode_dates=[],
        per_lf_overrides=overrides or {},
    )
    with (
        patch.object(ge, "_build_per_lf_gpx_path_map", return_value={}),
        patch.object(ge, "_build_per_lf_gpx_episode_map", return_value={}),
        patch.object(ge, "_journey_endpoint_loss_by_lf", return_value={}),
        patch.object(ge, "_build_shared_geocoder", return_value=_BandGeocoder()),
        patch.object(ge, "_partition_lf_plans_by_date", return_value={"2026-01-02": lfs}),
        patch.object(ge, "_assert_per_lf_locations_resolvable", return_value=None),
        patch.object(ge, "derive_per_lf_locations", return_value={}),
        patch.object(ge, "_resolve_lf_destination", return_value=""),
        patch.object(ge, "canonical_lf_ost_text", side_effect=lambda lf, **k: labels[lf.index]),
        patch.object(ge, "_differentiate_and_revalidate_siblings", return_value=None),
    ):
        out = ge._enrich_per_lf_gps(
            state=state,
            real_plan=SimpleNamespace(regular_lfs=lfs),
            day_batches=[],
            episode_files=[],
            per_day_locations={},
            drive_wide_location="",
        )
    out.state = state
    return out


def test_entry_point_retained_window_supports_its_own_place():
    # LF 1 keeps only Harrow footage; the episode's Pellmere/Ostry time is excluded.
    out = _enrich([_lf(1, (0.0, 280.0))], {1: "Harrow, Westmark"})
    assert out.per_lf_ost_text_by_date["2026-01-02"][1] == "Harrow, Westmark"
    assert out.route_claims_by_lf[1].retained_seconds == 280.0


def test_entry_point_persists_the_route_record_for_the_final_text_review():
    # Phase BC stores what Phase D reviews and upload re-verifies — names,
    # seconds and gaps only, never a fix.
    out = _enrich([_lf(1, (0.0, 280.0))], {1: "Harrow, Westmark"})
    entry = out.state.per_lf_route_record[1]
    assert [p["name"] for p in entry["record"]["places"]] == ["Harrow"]
    assert entry["record"]["retained_seconds"] == 280.0
    assert {c["level"] for c in entry["region_claims"]} == {"county", "state"}
    blob = json.dumps(entry)
    assert '"lat"' not in blob and '"lon"' not in blob and "9.5" not in blob


def test_entry_point_multi_place_falls_back_before_the_burn():
    out = _enrich([_lf(1, (0.0, 280.0), (320.0, 580.0))], {1: "Harrow, Westmark"})
    assert out.per_lf_ost_text_by_date["2026-01-02"][1] == "Ashby County, Westmark"


def test_entry_point_aggregates_refusals_and_checks_operator_labels():
    # Both labels are operator overrides (checked, never replaced), and both
    # omit a place with real retained time: ONE refusal names both LFs.
    lfs = [_lf(1, (0.0, 280.0), (620.0, 900.0)), _lf(2, (0.0, 280.0), (320.0, 580.0))]
    labels = {1: "Harrow, Westmark", 2: "Harrow, Westmark"}
    with pytest.raises(RouteClaimRefusedError) as exc:
        _enrich(lfs, labels, overrides=dict(labels))
    msg = str(exc.value)
    assert msg.startswith("2 long-form label(s)")
    assert "LF 1: operator label 'Harrow, Westmark'" in msg and "'Ostry'" in msg
    assert "LF 2: operator label 'Harrow, Westmark'" in msg and "'Pellmere'" in msg


def test_entry_point_two_counties_fall_back_to_state():
    out = _enrich([_lf(1, (0.0, 280.0), (620.0, 900.0))], {1: "Harrow, Westmark"})
    assert out.per_lf_ost_text_by_date["2026-01-02"][1] == "Westmark"


def test_dry_run_prints_the_refusal_before_any_encode(capsys):
    lf = _lf(2, (0.0, 280.0), (320.0, 580.0))
    state = SimpleNamespace(gps_per_episode=_episode(), per_lf_overrides={2: "Harrow, Westmark"})
    with (
        patch.object(ge, "derive_per_lf_locations", return_value={}),
        patch.object(ge, "_resolve_lf_destination", return_value=""),
        patch.object(ge, "_differentiate_and_revalidate_siblings", return_value=None),
        patch.object(ge, "validate_burned_ost_text", side_effect=lambda c, loc, idx: c),
    ):
        text = lf_listing._render_ost_text_by_lf([lf], {}, _BandGeocoder(), state)
    assert text[2] == "Harrow, Westmark"
    assert "ROUTE-CLAIM REFUSED" in capsys.readouterr().out


def test_label_naming_a_place_the_retained_footage_never_reaches_is_refused():
    # The Longwood shape: the label names a town with ZERO retained fixes (no
    # verified place) beside one it does reach. Naming a place is a claim.
    lfs = [_lf(1, (0.0, 280.0))]
    labels = {1: "Pellmere & Harrow, Westmark"}
    out = _enrich(lfs, labels)
    assert out.per_lf_ost_text_by_date["2026-01-02"][1] == "Ashby County, Westmark"
    with pytest.raises(RouteClaimRefusedError) as exc:
        _enrich(lfs, labels, overrides=dict(labels))
    assert "names 'Pellmere'" in str(exc.value)


def test_both_gpx_files_feed_the_retained_mapping():
    # Two episodes (two GPX files): the retained Ostry seconds are in episode 0,
    # the Harrow seconds in episode 1. Both must count, from their own file.
    ostry = _episode(300, lambda t: 10.8)
    harrow = _episode(300, lambda t: 9.5, origin=T0 + 10_000)
    lf = _lf(1, (1, 0.0, 280.0), (0, 0.0, 120.0))
    with pytest.raises(RouteClaimRefusedError) as exc:
        _enrich([lf], {1: "Harrow, Westmark"}, {1: "Harrow, Westmark"}, ostry + harrow)
    assert "omits 'Ostry' (120 s" in str(exc.value)
    out = _enrich([lf], {1: "Harrow, Westmark"}, episodes=ostry + harrow)
    assert out.route_claims_by_lf[1].retained_seconds == 400.0


def test_null_island_fixes_are_never_samples():
    # Fixes at 0,0 are no fix at all: that stretch is a gap, never a place, and
    # a gap blocks every region fallback, so the run refuses before the burn.
    ep = _episode(300, lambda t: 0.0 if 100 <= t < 200 else 9.5)
    with pytest.raises(RouteClaimRefusedError) as exc:
        _enrich([_lf(1, (0.0, 280.0))], {1: "Harrow, Westmark"}, episodes=ep)
    msg = str(exc.value)
    assert "101 s of retained footage has no resolved place (gap_seconds 101.0" in msg
    assert "lf01-s0 ep0 @99.0 s +101.0 s" in msg


def test_village_and_city_geocodes_are_places_not_unresolved_time():
    # The detailed geocode carries a settlement as city/town/village/hamlet;
    # reading only "town" made every rural sample unresolved (run e shape).
    class _Village(_BandGeocoder):
        def reverse_geocode_detailed(self, lat, lon):
            hit = super().reverse_geocode_detailed(lat, lon)
            return {"village": hit.pop("town"), **hit}

    ev = measure_route([_window("lf01", 0.0, 280.0, lambda t: 9.5)], _Village())
    assert [p.name for p in ev.places] == ["Harrow"]
    assert ev.unsettled_seconds == 0.0


# --- Gap evidence: the straddle cases ---

import json  # noqa: E402

from gomoveshift.pipeline.route_claims import (  # noqa: E402
    GAP_S,
    TIMESTAMP_SOURCE,
    containment_wording_issues,
)


def _gap_tuples(ev):
    return [(g.segment_id, g.start_s, g.end_s) for g in ev.gaps]


def test_d4_1_retained_gap_blocks_fallback_and_lists_each_gap():
    # Two places, so the label needs the county fallback; a 36 s dropout blocks it.
    ep = _episode(900, lambda t: 9.5 if t < 300 else 10.2)
    del ep[0][0].trackpoints[400:436]
    with pytest.raises(RouteClaimRefusedError) as exc:
        _enrich([_lf(1, (0.0, 600.0))], {1: "Harrow, Westmark"}, episodes=ep)
    msg = str(exc.value)
    # file, media start, length
    assert "lf01-s0 ep0 @399.0 s +37.0 s" in msg


def test_d4_2_gap_only_in_excluded_footage_does_not_block():
    ep = _episode(900, lambda t: 9.5 if t < 300 else 10.2)
    del ep[0][0].trackpoints[700:760]  # dropout in footage the LF does not keep
    out = _enrich([_lf(1, (0.0, 280.0), (320.0, 600.0))], {1: "Harrow, Westmark"}, episodes=ep)
    assert out.route_claims_by_lf[1].gaps == []
    assert out.per_lf_ost_text_by_date["2026-01-02"][1] == "Ashby County, Westmark"


# The two D4 gap-measurement tests below check what a gap counts as, not whether a
# label may ship over it (the zero-tolerance place gate now refuses any gap).
_gap_measure_only = patch("gomoveshift.pipeline.route_claims.label_issues", new=lambda *a, **k: [])


@_gap_measure_only
def test_d4_3_straddle_dropout_counts_only_its_retained_overlap():
    # Fixes at ...,358, then 364,...: a 6 s dropout. The segment starts at 360,
    # so 4 s of it is retained: one gap of 4 s, exact coverage.
    ep = _episode(900, lambda t: 9.5)
    ep[0][0].trackpoints[359:364] = []
    out = _enrich([_lf(1, (360.0, 760.0))], {1: "Harrow, Westmark"}, episodes=ep)
    ev = out.route_claims_by_lf[1]
    assert _gap_tuples(ev) == [("lf01-s0", 360.0, 364.0)]
    assert ev.gaps[0].interval_s == pytest.approx(6.0)
    assert ev.gap_seconds == pytest.approx(4.0)
    assert ev.coverage == pytest.approx(1 - 4.0 / 400.0)


def test_d4_3b_window_local_measure_would_drop_the_straddle():
    # Without the fix before the window the same 4 s span is only 4 s long and
    # would not qualify; the window carries the outside fix so it does.
    w = _window("lf01", 360.0, 760.0, lambda t: 9.5, skip=[(360.0, 364.0)])
    assert measure_route([w], _BandGeocoder()).gaps == []
    w.prev_fix_s = 358.0
    assert _gap_tuples(measure_route([w], _BandGeocoder())) == [("lf01", 360.0, 364.0)]


@pytest.mark.parametrize("dropout, is_gap", [(5.0, False), (5.01, True)])
def test_d4_4_exactly_five_seconds_is_not_a_gap(dropout, is_gap):
    fixes = [(float(t), 9.5, 20.0) for t in range(0, 101)]
    fixes += [(100.0 + dropout + t, 9.5, 20.0) for t in range(0, 100)]
    w = RouteWindow("lf01", 0.0, 199.0, fixes)
    ev = measure_route([w], _BandGeocoder())
    assert (len(ev.gaps) == 1) is is_gap
    rec = ev.record()
    assert rec["timestamp_source"] == TIMESTAMP_SOURCE
    assert rec["tick_s"] == pytest.approx(1.0)
    assert rec["gap_threshold_s"] == GAP_S == 5.0
    # Within one tick of the boundary: flagged either way, for the verifier.
    assert len(rec["boundary_notes"]) == 1


@_gap_measure_only
def test_d4_5_leading_no_fix_at_segment_start_counts():
    ep = _episode(600, lambda t: 9.5)
    ep[0][0].trackpoints[:30] = []  # GPS locks at 30 s; the segment starts at 0
    out = _enrich([_lf(1, (0.0, 500.0))], {1: "Harrow, Westmark"}, episodes=ep)
    assert _gap_tuples(out.route_claims_by_lf[1]) == [("lf01-s0", 0.0, 30.0)]


def test_d4_6_coverage_definition_has_a_known_value():
    # 1000 s retained, one 50 s gap (fixes at 499 and 550 are 51 s apart; the
    # gap is the 51 s fix-to-fix interval) -> pin exactly.
    w = _window("lf01", 0.0, 1000.0, lambda t: 9.5, skip=[(500.0, 550.0)])
    ev = measure_route([w], _BandGeocoder())
    assert ev.gap_seconds == pytest.approx(51.0)
    assert ev.record()["coverage_pct"] == pytest.approx(94.9)
    w2 = _window("lf01", 0.0, 1000.0, lambda t: 9.5, skip=[(0.0, 50.0)])
    assert measure_route([w2], _BandGeocoder()).record()["coverage_pct"] == 95.0


def test_d4_7_one_sample_outside_blocks_the_region_at_full_coverage():
    # Harrow + Pellmere (both Ashby) with ONE bin in Brell County: coverage
    # 100%, still no county claim.
    w = _window("lf01", 0, 900, lambda t: 10.2 if 300 <= t < 600 else 10.7 if t == 600 else 9.5)
    ev = measure_route([w], _BandGeocoder(), bin_s=10.0)
    claim = ev.region_claim("county")
    assert ev.coverage == 1.0
    assert claim["samples_outside"] == 1
    label, _ = select_supported_label("Harrow, Westmark", ev)
    assert label != "Ashby County, Westmark"


def test_d4_8_invalid_and_duplicate_fixes_are_counted_and_excluded():
    w = _window("lf01", 0, 300, lambda t: 9.5)
    w.fixes[100:120] = [(t, 0.0, 0.0) for t, _la, _lo in w.fixes[100:120]]
    w.fixes[200] = (200.0, float("nan"), 20.0)
    w.fixes.append((50.0, 9.5, 20.0))  # duplicate timestamp
    ev = measure_route([w], _BandGeocoder())
    assert _gap_tuples(ev) == [("lf01", 99.0, 120.0)]
    rec = ev.record()
    assert rec["invalid_fixes"] == 21
    assert rec["duplicate_fixes"] == 1


def test_d4_9_no_output_asserts_containment_while_gaps_exist():
    w = _window("lf01", 0, 900, lambda t: 10.2 if 300 <= t < 600 else 9.5, skip=[(400, 436)])
    ev = measure_route([w], _BandGeocoder())
    rec = ev.record()
    assert rec["containment_proven"] is False
    assert rec["containment_basis"] == "sampled_reverse_geocode"
    with pytest.raises(RouteClaimRefusedError) as exc:
        select_supported_label("Harrow, Westmark", ev)
    for text in (str(exc.value), json.dumps(rec), "Harrow, Westmark"):
        assert containment_wording_issues(text) == [], text
    assert containment_wording_issues("Verified in Ashby County")
    assert containment_wording_issues("route contained in Westmark")
    claim = ev.region_claim("county")
    assert claim["containment_proven"] is False
    assert "contains" not in claim and "required_place_ids" in claim


def test_d4_refusal_names_gap_and_unsettled_seconds_separately():
    ev = measure_route([_harrow_pellmere_harrow()], _BandGeocoder(fail_above=10.0))
    assert ev.unsettled_seconds == pytest.approx(300.0)
    with pytest.raises(RouteClaimRefusedError) as exc:
        select_supported_label("Harrow, Westmark", ev)
    msg = str(exc.value)
    assert "gap_seconds 0.0" in msg and "unsettled_seconds 300.0" in msg


# E1: --label-dry-run writes the per-LF route-claim record -- RouteEvidence.record()
# verbatim plus the label decision -- under the run's stages dir, before the refusal
# prints, so the gate is graded from a file rather than scrollback.


def _dry_run_record(tmp_path, lf, overrides, capsys):
    path = tmp_path / "_stages" / "route_claims_dry_run.json"
    state = SimpleNamespace(gps_per_episode=_episode(), per_lf_overrides=overrides)
    report = {"path": path}
    with (
        patch.object(ge, "derive_per_lf_locations", return_value={}),
        patch.object(ge, "_resolve_lf_destination", return_value=""),
        patch.object(ge, "_differentiate_and_revalidate_siblings", return_value=None),
        patch.object(ge, "canonical_lf_ost_text", return_value="Harrow, Westmark"),
    ):
        lf_listing._render_ost_text_by_lf([lf], {}, _BandGeocoder(), state, report=report)
    return report, json.loads(path.read_text()), capsys.readouterr().out


_RECORD_FIELDS = {"coverage_pct", "gap_seconds", "retained_seconds", "places", "gaps"}


def test_dry_run_writes_the_refused_lf_record_before_the_refusal_prints(tmp_path, capsys):
    lf = _lf(2, (0.0, 280.0), (320.0, 580.0))
    report, blob, out = _dry_run_record(tmp_path, lf, {2: "Harrow, Westmark"}, capsys)
    assert report["refused"] is True and blob["refused"] is True
    (rec,) = blob["lfs"]
    assert rec["lf_index"] == 2 and rec["refused"] is True and rec["label"] is None
    assert rec["label_requested"] == "Harrow, Westmark"
    assert "not supported by the retained GPS" in rec["reason"]
    assert rec["containment_proven"] is False
    assert rec["containment_basis"] == "sampled_reverse_geocode"
    assert _RECORD_FIELDS <= set(rec)
    assert out.index(str(report["path"])) < out.index("ROUTE-CLAIM REFUSED")
    assert "verified in" not in json.dumps(blob) and "contained" not in out
    assert '"lat"' not in json.dumps(blob) and '"lon"' not in json.dumps(blob)


def test_dry_run_record_shows_the_broader_label_a_fallback_burns(tmp_path, capsys):
    lf = _lf(1, (0.0, 280.0), (320.0, 580.0))
    report, blob, out = _dry_run_record(tmp_path, lf, {}, capsys)
    assert report["refused"] is False and blob["refused"] is False
    (rec,) = blob["lfs"]
    assert rec["refused"] is False and rec["label"] == "Ashby County, Westmark"
    assert rec["label_requested"] == "Harrow, Westmark" and "using" in rec["reason"]
    assert rec["containment_proven"] is False and _RECORD_FIELDS <= set(rec)
    assert "ROUTE-CLAIM REFUSED" not in out


def test_print_label_dry_run_returns_refused_and_writes_under_stages_base(
    tmp_path, monkeypatch, capsys
):
    from gomoveshift.pipeline.drive_phases import phase_bc_day_first as phase_bc

    lf = _lf(2, (0.0, 280.0), (320.0, 580.0))
    state = SimpleNamespace(
        gps_per_episode=_episode(),
        per_lf_overrides={2: "Harrow, Westmark"},
        stages_base=tmp_path / "stages",
    )
    monkeypatch.setattr(phase_bc, "_build_drive_plan_from_state", lambda c, s: SimpleNamespace(
        regular_lfs=[lf]))  # fmt: skip
    monkeypatch.setattr(ge, "_build_gps_points_by_lf", lambda p, g: {})
    monkeypatch.setattr(ge, "_build_shared_geocoder", lambda *a, **k: _BandGeocoder())
    monkeypatch.setattr(lf_listing, "_live_title_ids", lambda: {})
    with (
        patch.object(ge, "derive_per_lf_locations", return_value={}),
        patch.object(ge, "_resolve_lf_destination", return_value=""),
        patch.object(ge, "_differentiate_and_revalidate_siblings", return_value=None),
        patch.object(ge, "canonical_lf_ost_text", return_value="Harrow, Westmark"),
    ):
        refused = lf_listing.print_label_dry_run(config=object(), state=state)
    assert refused is True
    path = tmp_path / "stages" / "route_claims_dry_run.json"
    assert json.loads(path.read_text())["lfs"][0]["refused"] is True
    assert f"route-claim record: {path}" in capsys.readouterr().out


def test_dry_run_without_a_run_dir_never_writes_into_the_cwd(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    lf = _lf(1, (0.0, 280.0))
    report = {"path": None}
    with (
        patch.object(ge, "derive_per_lf_locations", return_value={}),
        patch.object(ge, "_resolve_lf_destination", return_value=""),
        patch.object(ge, "_differentiate_and_revalidate_siblings", return_value=None),
        patch.object(ge, "canonical_lf_ost_text", return_value="Harrow, Westmark"),
    ):
        lf_listing._render_ost_text_by_lf(
            [lf], {}, _BandGeocoder(), SimpleNamespace(gps_per_episode=_episode()), report=report
        )
    assert report["refused"] is False and list(tmp_path.iterdir()) == []


def test_dry_run_record_keeps_a_backed_label_and_the_run_succeeds(tmp_path, monkeypatch, capsys):
    from gomoveshift.pipeline.drive_phases import phase_bc_day_first as phase_bc

    lf = _lf(1, (0.0, 280.0))  # Harrow-only retained footage backs "Harrow, Westmark"
    state = SimpleNamespace(
        gps_per_episode=_episode(), per_lf_overrides={}, stages_base=tmp_path / "stages"
    )
    monkeypatch.setattr(phase_bc, "_build_drive_plan_from_state", lambda c, s: SimpleNamespace(
        regular_lfs=[lf]))  # fmt: skip
    monkeypatch.setattr(ge, "_build_gps_points_by_lf", lambda p, g: {})
    monkeypatch.setattr(ge, "_build_shared_geocoder", lambda *a, **k: _BandGeocoder())
    monkeypatch.setattr(lf_listing, "_live_title_ids", lambda: {})
    with (
        patch.object(ge, "derive_per_lf_locations", return_value={}),
        patch.object(ge, "_resolve_lf_destination", return_value=""),
        patch.object(ge, "_differentiate_and_revalidate_siblings", return_value=None),
        patch.object(ge, "canonical_lf_ost_text", return_value="Harrow, Westmark"),
    ):
        refused = lf_listing.print_label_dry_run(config=object(), state=state)
    assert refused is False
    blob = json.loads((tmp_path / "stages" / "route_claims_dry_run.json").read_text())
    (rec,) = blob["lfs"]
    assert blob["refused"] is False and rec["refused"] is False
    assert rec["label"] == rec["label_requested"] == "Harrow, Westmark" and rec["reason"] == ""
    assert rec["containment_proven"] is False
    assert rec["containment_basis"] == "sampled_reverse_geocode"
    assert _RECORD_FIELDS <= set(rec) and rec["retained_seconds"] == 280.0
    assert "ROUTE-CLAIM REFUSED" not in capsys.readouterr().out


def test_2c_record_names_the_gps_timeline_it_measured_on():
    # The record measures on state.gps_per_episode (the
    # phase-A parse of each episode's .gpx, minus its effective offset), not a
    # raw-GPX read. A verifier re-measuring from raw GPX sees sub-second
    # per-fix shifts, so the record must say which timeline it used.
    import gomoveshift.pipeline.route_claims as rc

    w = RouteWindow("lf01", 0.0, 10.0, [(float(t), 9.5, 20.0) for t in range(0, 11)])
    rec = measure_route([w], _BandGeocoder()).record()
    assert rec["timeline"] == rc.TIMELINE == "state.gps_per_episode"
    assert "gps_per_episode" in rc.TIMESTAMP_SOURCE
    assert "effective offset" in rc.TIMESTAMP_SOURCE


# -- unresolved bins vs a pinned boundary proof --------------------------------
from gomoveshift.pipeline.route_claims import RouteEvidence, _Bin  # noqa: E402

_PROOF = {"verified": True, "region": "Ashby County", "windows_key": "k"}


def _evidence(bins, proofs=None):
    return RouteEvidence(
        retained_seconds=30.0,
        gaps=[],
        places=[],
        unsettled_seconds=0.0,
        bins=bins,
        boundary_proofs=proofs or {},
        windows_key="k",
    )


def test_unresolved_bin_with_proof_is_not_outside():
    bins = [_Bin(10.0, "Ashby County", "Westmark")] * 2 + [_Bin(10.0, None, None)]
    claim = _evidence(bins, {"county": _PROOF}).region_claim("county")
    assert claim["samples_outside"] == 0
    assert claim["samples_unresolved"] == 1
    assert claim["samples_inside"] == 2
    assert claim["boundary_proof"] == _PROOF


def test_unresolved_bin_without_proof_still_counts_outside():
    bins = [_Bin(10.0, "Ashby County", "Westmark")] * 2 + [_Bin(10.0, None, None)]
    claim = _evidence(bins).region_claim("county")
    assert claim["samples_outside"] == 1
    assert claim["samples_unresolved"] == 1
    assert claim["samples_inside"] == 2


@pytest.mark.parametrize("proofs", [{}, {"county": _PROOF}])
def test_named_other_county_is_outside_with_or_without_proof(proofs):
    bins = [_Bin(10.0, "Ashby County", "Westmark")] * 2 + [_Bin(10.0, "Brell County", "Westmark")]
    claim = _evidence(bins, proofs).region_claim("county")
    assert claim["samples_outside"] == 1
    assert claim["samples_unresolved"] == 0
