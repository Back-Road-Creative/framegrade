"""Road-sign GPS verification: bucketing + advisory comparison.

Hermetic — no network. The autouse ``hermetic_forward_geocode`` fixture in
``tests/conftest.py`` already stubs live Nominatim; every ``geocode_fn`` used
here is an explicit local stub, so nothing in this file can reach out.

The load-bearing assertions are the CONSERVATIVE ones: one noisy sign must
never contradict GPS, town/route claims must never conflict at all, and
"no signal" must stay distinct from "corroborates" (a silent OCR failure read
as agreement is the failure mode this whole feature exists to avoid).
"""

from pathlib import Path

import pytest

from gomoveshift.analysis.sign_evidence import (
    STATUS_CONFLICTS,
    STATUS_CORROBORATES,
    STATUS_NO_SIGNAL,
    SignHit,
)
from gomoveshift.analysis.sign_verify import (
    bucket_hits_by_lf,
    resolved_from_dominant,
    verify_all,
    verify_lf,
)
from gomoveshift.models import DrivePlan, LFSegment, RegularLFPlan, ResolvedLocation
from gomoveshift.pipeline.title_builder import DominantLocation

# ── Builders ──────────────────────────────────────────────────────────


def _seg(episode: int, start: float, end: float) -> LFSegment:
    return LFSegment(
        source_file_index=episode,
        source_path=Path("/nonexistent/ep.mp4"),
        start_time=start,
        end_time=end,
        original_start_time=start,
        original_end_time=end,
        composite_score=0.7,
        gps_center_lat=32.9,
        gps_center_lon=-80.6,
        terrain_type="countryside",
        contiguous_with_previous=False,
    )


def _lf(index: int, segments: list) -> RegularLFPlan:
    return RegularLFPlan(
        index=index,
        segments=segments,
        montage_peaks=segments[:1],
        shorts_target=3,
        region="usa",
        total_duration=sum(s.end_time - s.start_time for s in segments),
    )


def _plan(*lfs: RegularLFPlan) -> DrivePlan:
    return DrivePlan(
        source_files=[],
        processed_files=[],
        regular_lfs=list(lfs),
        score_threshold=0.5,
        total_source_duration=3600.0,
        total_highlight_duration=1800.0,
    )


def _hit(episode: int, t_local: float, *claims, raw_text: str = "", date: str = "2026-07-01"):
    """One SignHit. ``claims`` are ``(kind, value)`` pairs; ``kind`` holds a
    ``LocationType`` VALUE as a plain string (see ``SignClaim.kind``)."""
    text = raw_text or " / ".join(v for _, v in claims) or "no place named"
    return SignHit(
        episode_index=episode,
        t_local=t_local,
        ts_day=t_local,
        date=date,
        raw_text=text,
        ocr_conf=0.9,
        lat=32.9,
        lon=-80.6,
        claims=[{"kind": k, "value": v, "raw_text": text} for k, v in claims],
    )


def _resolved_sc() -> ResolvedLocation:
    """A healthy SC model: Colleton voted, with a via-road concurrency."""
    return ResolvedLocation(
        dominant_county="Colleton County",
        dominant_state="South Carolina",
        country_code="us",
        is_us=True,
        ordered_place_sequence=["Walterboro", "Jacksonboro"],
        regions=["Colleton County, South Carolina"],
        vote={"Colleton County, South Carolina": 1.0},
        start_label="Walterboro, South Carolina",
        end_label="Jacksonboro, South Carolina",
        road="US-17",
        road_refs=["I 81;US 11"],
    )


# ── bucket_hits_by_lf ─────────────────────────────────────────────────


def test_hit_lands_in_the_lf_whose_segment_span_contains_it():
    plan = _plan(_lf(1, [_seg(0, 0.0, 100.0)]), _lf(2, [_seg(1, 0.0, 100.0)]))
    hit_a = _hit(0, 50.0, ("county_boundary", "Colleton"))
    hit_b = _hit(1, 50.0, ("county_boundary", "Fableport"))

    buckets = bucket_hits_by_lf({"2026-07-01": [hit_a, hit_b]}, plan)

    assert [h.raw_text for h in buckets.get(1, [])] == ["Colleton"]
    assert [h.raw_text for h in buckets.get(2, [])] == ["Fableport"]


def test_hit_on_an_episode_no_segment_uses_is_dropped():
    plan = _plan(_lf(1, [_seg(0, 0.0, 100.0)]))
    buckets = bucket_hits_by_lf({"2026-07-01": [_hit(7, 50.0, ("county_boundary", "Ghost"))]}, plan)
    assert buckets == {}


def test_hit_with_matching_episode_but_out_of_span_is_dropped():
    plan = _plan(_lf(1, [_seg(0, 0.0, 100.0)]))
    inside = _hit(0, 99.0, ("county_boundary", "Colleton"))
    after = _hit(0, 101.0, ("county_boundary", "Colleton"))

    buckets = bucket_hits_by_lf({"2026-07-01": [inside, after]}, plan)

    assert [h.t_local for h in buckets.get(1, [])] == [99.0]


def test_raw_json_hits_from_state_are_normalised():
    plan = _plan(_lf(1, [_seg(0, 0.0, 100.0)]))
    raw = {
        "2026-07-01": [
            {
                "episode_index": 0,
                "t_local": 10.0,
                "ts_day": 10.0,
                "date": "2026-07-01",
                "raw_text": "COLLETON COUNTY",
                "claims": [{"kind": "county_boundary", "value": "Colleton"}],
            }
        ]
    }
    buckets = bucket_hits_by_lf(raw, plan)
    assert isinstance(buckets[1][0], SignHit)
    assert buckets[1][0].parsed_claims()[0].value == "Colleton"


def test_missing_plan_buckets_nothing():
    assert bucket_hits_by_lf({"2026-07-01": [_hit(0, 1.0)]}, None) == {}
    assert bucket_hits_by_lf({"2026-07-01": [_hit(0, 1.0)]}, _plan()) == {}


# ── county comparison: the conservative conflict floor ────────────────


def test_one_hit_claiming_an_unsampled_county_does_not_conflict():
    hits = [_hit(0, 10.0, ("county_boundary", "Fableport"), raw_text="FABLEPORT COUNTY")]
    verdict = verify_lf(1, hits, _resolved_sc(), min_corroborating_hits=2)

    assert verdict.status != STATUS_CONFLICTS
    assert verdict.status == STATUS_NO_SIGNAL
    assert verdict.conflicting_counties == []


def test_two_hits_claiming_the_same_unsampled_county_conflict():
    hits = [
        _hit(0, 10.0, ("county_boundary", "Fableport"), raw_text="FABLEPORT COUNTY"),
        _hit(0, 90.0, ("county_boundary", "Fableport"), raw_text="ENTERING FABLEPORT CO"),
    ]
    verdict = verify_lf(1, hits, _resolved_sc(), min_corroborating_hits=2)

    assert verdict.status == STATUS_CONFLICTS
    assert verdict.conflicts is True
    assert [c.lower() for c in verdict.conflicting_counties] == ["fableport"]
    # Every flagged conflict must be judgeable from the reasons alone.
    joined = " ".join(verdict.reasons)
    assert "FABLEPORT COUNTY" in joined and "ENTERING FABLEPORT CO" in joined
    assert 0.0 < verdict.confidence <= 1.0


def test_claimed_county_present_in_regions_corroborates():
    hits = [_hit(0, 10.0, ("county_boundary", "Colleton"), raw_text="COLLETON COUNTY")]
    verdict = verify_lf(1, hits, _resolved_sc())

    assert verdict.status == STATUS_CORROBORATES
    assert [c.lower() for c in verdict.corroborating_counties] == ["colleton"]
    assert verdict.confidence == pytest.approx(1.0)
    assert "COLLETON COUNTY" in " ".join(verdict.reasons)


def test_conflict_wins_over_corroboration_in_status_precedence():
    hits = [
        _hit(0, 5.0, ("county_boundary", "Colleton"), raw_text="COLLETON COUNTY"),
        _hit(0, 10.0, ("county_boundary", "Fableport"), raw_text="FABLEPORT COUNTY"),
        _hit(0, 20.0, ("county_boundary", "Fableport"), raw_text="FABLEPORT CO LINE"),
    ]
    verdict = verify_lf(1, hits, _resolved_sc(), min_corroborating_hits=2)

    assert verdict.status == STATUS_CONFLICTS
    assert [c.lower() for c in verdict.corroborating_counties] == ["colleton"]


# ── state comparison: same floor ──────────────────────────────────────


def test_state_claim_needs_the_same_floor_before_it_can_conflict():
    one = [_hit(0, 10.0, ("state_boundary", "Georgia"), raw_text="WELCOME TO GEORGIA")]
    assert verify_lf(1, one, _resolved_sc(), min_corroborating_hits=2).status == STATUS_NO_SIGNAL

    two = one + [_hit(0, 40.0, ("state_boundary", "Georgia"), raw_text="GEORGIA STATE LINE")]
    verdict = verify_lf(1, two, _resolved_sc(), min_corroborating_hits=2)

    assert verdict.status == STATUS_CONFLICTS
    assert "Georgia" in " ".join(verdict.reasons)


def test_state_claim_matching_the_model_corroborates():
    hits = [_hit(0, 10.0, ("state_boundary", "South Carolina"), raw_text="WELCOME TO SC")]
    assert verify_lf(1, hits, _resolved_sc()).status == STATUS_CORROBORATES


# ── town + route: corroboration-only, never a conflict ────────────────


def test_town_claims_never_conflict_even_when_they_match_nothing():
    hits = [
        _hit(0, 10.0, ("town_entry", "Nowheresville"), raw_text="WELCOME TO NOWHERESVILLE"),
        _hit(0, 20.0, ("town_entry", "Nowheresville"), raw_text="NOWHERESVILLE 2 MI"),
        _hit(0, 30.0, ("town_entry", "Nowheresville"), raw_text="NOWHERESVILLE CITY LIMIT"),
    ]
    verdict = verify_lf(1, hits, _resolved_sc(), min_corroborating_hits=2)

    assert verdict.status != STATUS_CONFLICTS
    assert verdict.conflicting_counties == []


def test_town_claim_on_the_route_corroborates():
    hits = [_hit(0, 10.0, ("town_entry", "Walterboro"), raw_text="WALTERBORO CITY LIMIT")]
    assert verify_lf(1, hits, _resolved_sc()).status == STATUS_CORROBORATES


def test_unmatched_town_with_geocoder_records_distance_but_never_conflicts():
    calls = []

    def _geocode(name):
        calls.append(name)
        return (33.9, -81.6)  # ~145 km from the hit's (32.9, -80.6)

    hits = [
        _hit(0, 10.0, ("town_entry", "Nowheresville"), raw_text="WELCOME TO NOWHERESVILLE"),
        _hit(0, 20.0, ("town_entry", "Nowheresville"), raw_text="NOWHERESVILLE 2 MI"),
    ]
    verdict = verify_lf(1, hits, _resolved_sc(), min_corroborating_hits=2, geocode_fn=_geocode)

    assert calls == ["Nowheresville"]
    assert verdict.status != STATUS_CONFLICTS
    assert any("km" in r for r in verdict.reasons)


def test_degraded_geocoder_returning_none_is_a_skip_not_a_failure():
    hits = [_hit(0, 10.0, ("town_entry", "Nowheresville"), raw_text="NOWHERESVILLE")]
    verdict = verify_lf(1, hits, _resolved_sc(), geocode_fn=lambda _n: None)
    assert verdict.status == STATUS_NO_SIGNAL


def test_route_claims_never_conflict_and_match_concurrency_members():
    unmatched = [
        _hit(0, 10.0, ("route_marker", "SR-999"), raw_text="SR 999"),
        _hit(0, 20.0, ("route_marker", "SR-999"), raw_text="SR 999 NORTH"),
    ]
    unmatched_verdict = verify_lf(1, unmatched, _resolved_sc(), min_corroborating_hits=2)
    assert unmatched_verdict.status == STATUS_NO_SIGNAL
    assert unmatched_verdict.conflicting_counties == []

    # "I 81;US 11" is one road_refs entry — a member-wise compare must credit I-81.
    member = [_hit(0, 10.0, ("route_marker", "I-81"), raw_text="INTERSTATE 81 NORTH")]
    assert verify_lf(1, member, _resolved_sc()).status == STATUS_CORROBORATES

    primary = [_hit(0, 10.0, ("route_marker", "US-17"), raw_text="US 17 SOUTH")]
    assert verify_lf(1, primary, _resolved_sc()).status == STATUS_CORROBORATES


# ── silence is not agreement ──────────────────────────────────────────


def test_no_claims_is_no_signal_and_is_distinct_from_corroboration():
    no_claims = verify_lf(1, [_hit(0, 10.0, raw_text="BOB'S BAIT SHOP")], _resolved_sc())
    assert no_claims.status == STATUS_NO_SIGNAL
    assert no_claims.status != STATUS_CORROBORATES
    assert no_claims.confidence == 0.0
    assert no_claims.reasons  # an operator must be told WHY there was no signal

    no_hits_at_all = verify_lf(1, [], _resolved_sc())
    assert no_hits_at_all.status == STATUS_NO_SIGNAL
    assert no_hits_at_all.hit_count == 0


def test_legacy_or_gps_less_resolved_location_is_a_noop():
    legacy = ResolvedLocation()  # no regions, no vote — quorum-degraded / GPS-less
    hits = [
        _hit(0, 10.0, ("county_boundary", "Fableport"), raw_text="FABLEPORT COUNTY"),
        _hit(0, 20.0, ("county_boundary", "Fableport"), raw_text="FABLEPORT CO LINE"),
    ]
    verdict = verify_lf(1, hits, legacy, min_corroborating_hits=2)

    assert verdict.status == STATUS_NO_SIGNAL
    assert verdict.conflicting_counties == []
    assert verdict.reasons  # says there was nothing to verify against
    assert verify_lf(1, hits, None, min_corroborating_hits=2).status == STATUS_NO_SIGNAL


# ── resolved_from_dominant ────────────────────────────────────────────


def test_resolved_from_dominant_maps_every_field():
    dom = DominantLocation(
        county="Colleton County",
        state="South Carolina",
        name="Walterboro",
        votes={"Colleton County|South Carolina": 5, "Fableport County|South Carolina": 1},
        road="US-17",
        road_refs=["I 81;US 11"],
    )
    rl = resolved_from_dominant(dom)

    assert isinstance(rl, ResolvedLocation)
    assert rl.dominant_county == "Colleton County"
    assert rl.dominant_state == "South Carolina"
    assert rl.vote == dom.votes
    assert rl.regions == list(dom.votes.keys())
    assert rl.road == "US-17"
    assert rl.road_refs == ["I 81;US 11"]


def test_resolved_from_dominant_tolerates_none():
    rl = resolved_from_dominant(None)
    assert isinstance(rl, ResolvedLocation)
    assert rl.regions == [] and rl.vote == {}


def test_dominant_adapter_feeds_the_same_comparator_as_the_encode_path():
    """The --label-dry-run path has a ``dom``, not a persisted model — the
    adapter's output must compare identically (pipe-form vote keys included)."""
    dom = DominantLocation(
        county="Colleton County",
        state="South Carolina",
        votes={"Colleton County|South Carolina": 5},
    )
    rl = resolved_from_dominant(dom)

    hits = [_hit(0, 10.0, ("county_boundary", "Colleton"), raw_text="COLLETON COUNTY")]
    assert verify_lf(1, hits, rl).status == STATUS_CORROBORATES

    far = [
        _hit(0, 10.0, ("state_boundary", "Georgia"), raw_text="WELCOME TO GEORGIA"),
        _hit(0, 20.0, ("state_boundary", "Georgia"), raw_text="GEORGIA STATE LINE"),
    ]
    assert verify_lf(1, far, rl, min_corroborating_hits=2).status == STATUS_CONFLICTS


# ── verify_all ────────────────────────────────────────────────────────


def test_verify_all_returns_one_verdict_per_lf_with_no_cross_lf_leak():
    plan = _plan(_lf(1, [_seg(0, 0.0, 100.0)]), _lf(2, [_seg(1, 0.0, 100.0)]))
    lf2_resolved = ResolvedLocation(
        dominant_county="Fableport County",
        dominant_state="South Carolina",
        regions=["Fableport County, South Carolina"],
        vote={"Fableport County, South Carolina": 1.0},
    )
    hits_by_date = {
        "2026-07-01": [
            _hit(0, 10.0, ("county_boundary", "Colleton"), raw_text="COLLETON COUNTY"),
            # LF 2's two Colleton signs would CONFLICT if they leaked into LF 1's
            # bucket they'd instead corroborate — and vice versa.
            _hit(1, 10.0, ("county_boundary", "Colleton"), raw_text="COLLETON CO A"),
            _hit(1, 20.0, ("county_boundary", "Colleton"), raw_text="COLLETON CO B"),
        ]
    }

    verdicts = verify_all(
        hits_by_date, plan, {1: _resolved_sc(), 2: lf2_resolved}, min_corroborating_hits=2
    )

    assert sorted(verdicts) == [1, 2]
    assert verdicts[1].status == STATUS_CORROBORATES
    assert verdicts[1].hit_count == 1
    assert verdicts[2].status == STATUS_CONFLICTS
    assert verdicts[2].hit_count == 2
    # LF 1's verdict must not quote LF 2's signs.
    assert "COLLETON CO A" not in " ".join(verdicts[1].reasons)


def test_verify_all_rehydrates_raw_dict_resolved_locations():
    plan = _plan(_lf(1, [_seg(0, 0.0, 100.0)]))
    raw_resolved = {
        1: {
            "dominant_county": "Colleton County",
            "dominant_state": "South Carolina",
            "regions": ["Colleton County, South Carolina"],
            "vote": {"Colleton County, South Carolina": 1.0},
        }
    }
    hits_by_date = {
        "2026-07-01": [_hit(0, 10.0, ("county_boundary", "Colleton"), raw_text="COLLETON COUNTY")]
    }

    verdicts = verify_all(hits_by_date, plan, raw_resolved)
    assert verdicts[1].status == STATUS_CORROBORATES


def test_verify_all_skips_lfs_with_no_resolved_model():
    plan = _plan(_lf(1, [_seg(0, 0.0, 100.0)]), _lf(2, [_seg(1, 0.0, 100.0)]))
    hits_by_date = {
        "2026-07-01": [
            _hit(0, 10.0, ("county_boundary", "Colleton"), raw_text="COLLETON COUNTY"),
            _hit(1, 10.0, ("county_boundary", "Fableport"), raw_text="FABLEPORT COUNTY"),
        ]
    }
    verdicts = verify_all(hits_by_date, plan, {1: _resolved_sc()})
    assert sorted(verdicts) == [1]
