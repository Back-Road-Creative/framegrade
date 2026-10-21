"""Specificity-ladder evidence checks in the pre-encode location gate.

The correct-LF fixture models a late-stage LF ("Wytheville to Fort Chiswell via
I-81, VA" — Wythe County, moving): it MUST pass; only an unsupported label fails.
"""

from __future__ import annotations

import json

import pytest

from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
    _assert_pre_encode_location_invariant,
    _degrade_labels_to_awarded_tier,
)
from gomoveshift.pipeline.validation.content_gates import check_specificity_ladder

from _state_helpers import make_state

_LAT0, _LON0 = 36.95, -81.08  # Wythe County trace anchor
_LF5 = "Wytheville to Fort Chiswell via I-81, VA"
_W = (0.0, 100.0)


def _moving(n=101, dlat=0.0001):  # dense 1 Hz, moving north ~11 m/s
    return [{"lat": _LAT0 + i * dlat, "lon": _LON0, "time": float(i)} for i in range(n)]


def _sparse():  # 3 fixes over a 100 s window → ~0.03 coverage (dead GPS)
    return [{"lat": _LAT0 + i * 0.0001, "lon": _LON0, "time": float(i)} for i in range(3)]


def _parked(n=101):  # dense but stationary (~0 m/s)
    return [{"lat": _LAT0, "lon": _LON0, "time": float(i)} for i in range(n)]


def _resolved():
    from gomoveshift.models import ResolvedLocation

    rl = ResolvedLocation()
    rl.dominant_county, rl.dominant_state = "Wythe County", "Virginia"
    rl.regions = ["Wythe County, Virginia"]
    rl.vote = {"Wythe County, Virginia": 1.0}
    rl.ordered_place_sequence = ["Wytheville", "Fort Chiswell"]
    return rl


class TestSpecificityLadder:
    def test_correct_high_coverage_lf_passes(self):
        # All four checks active AND supporting the label → passes unchanged.
        ev = {
            "gps_points": _moving(),
            "window": _W,
            "road_refs": ["I-81"] * 8 + ["US-11"] * 2,
            "geocode_fn": lambda _t: (_LAT0, _LON0),
        }
        assert check_specificity_ladder({"location": _LF5}, ev).passed is True

    @pytest.mark.parametrize(
        "label,ev,needle",
        [
            (_LF5, {"gps_points": _sparse(), "window": _W}, "demote"),  # dead GPS
            (
                _LF5,
                {"gps_points": _moving(), "window": _W, "road_refs": ["US-11"] * 10},
                "via-road",
            ),
            ("POV Driving Beaufort, SC", {"gps_points": _parked(), "window": _W}, "stationary"),
            (
                _LF5,
                {"gps_points": _moving(), "window": _W, "geocode_fn": lambda _t: (40.0, -75.0)},
                "km from the footage",
            ),
        ],
    )
    def test_unsupported_label_fails_or_demotes(self, label, ev, needle):
        res = check_specificity_ladder({"location": label}, ev)
        assert res.passed is False
        assert needle in res.message

    def test_authoritative_flag_does_not_bypass_ladder(self):
        ev = {"gps_points": _sparse(), "window": _W, "authoritative": True}
        res = check_specificity_ladder({"location": _LF5}, ev)
        assert res.passed is False
        assert "LADDER-OVERRIDE" not in res.message

    def test_no_evidence_fails_closed(self):
        # U1: a label with no evidence record is an unsupported claim, not a skip.
        assert check_specificity_ladder({"location": _LF5}, None).passed is False

    def test_county_form_label_not_flagged_on_low_coverage(self):
        # Already-demoted county form makes no tier-1 claim → low coverage is fine.
        ev = {"gps_points": _sparse(), "window": _W}
        res = check_specificity_ladder({"location": "Colleton County, South Carolina"}, ev)
        assert res.passed is True


_EPOCH = 1_700_000_000.0  # gps_per_episode-path fixes carry epoch timestamps


def _moving_epoch(t0=0, t1=100):  # dense 1 Hz epoch-clock fixes, moving
    return [
        {"lat": _LAT0 + i * 0.0001, "lon": _LON0, "time": _EPOCH + float(t)}
        for i, t in enumerate(range(int(t0), int(t1) + 1))
    ]


def _lf_plan(idx=5, start=0.0, end=100.0):
    from types import SimpleNamespace

    return SimpleNamespace(
        index=idx, segments=[SimpleNamespace(start_time=float(start), end_time=float(end))]
    )


class TestLadderEvidenceWiring:
    """``_build_ladder_evidence`` must wire footage duration, road
    refs (normalized), and a forward geocoder, not only {gps_points,
    authoritative}. Points are epoch-clock (the production shape); the footage
    duration comes from the LF plan segments, which is what makes a dead
    PREFIX visible — the fixes' own min/max span can't see it."""

    def _evidence(self, pts, lf, resolved=None, geocode_fn=None):
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import _build_ladder_evidence

        return _build_ladder_evidence(
            {5: _LF5},
            {5: pts},
            None,
            lf_plans=[lf],
            resolved_by_lf=resolved,
            geocode_fn=geocode_fn,
        )[5]

    def test_dead_prefix_footage_demotes_tier1(self):
        # Footage [0, 3600], GPS locks only at 2040 s.
        # Coverage over the fixes' own span is ~1.0 (the bug); over the
        # footage duration it is ~0.43 → tier demotion failure fires.
        ev = self._evidence(_moving_epoch(2040, 3600), _lf_plan(end=3600.0))
        res = check_specificity_ladder({"location": _LF5}, ev)
        assert res.passed is False
        assert "demote" in res.message
        assert "0.43" in res.message

    def test_full_coverage_footage_still_passes(self):
        ev = self._evidence(_moving_epoch(0, 3600), _lf_plan(end=3600.0))
        assert check_specificity_ladder({"location": _LF5}, ev).passed is True

    def test_raw_osm_refs_normalized_so_correct_via_road_passes(self):
        # Raw OSM concurrency refs are space-separated ";"-joined; the
        # label's "I-81" would substring-match 0% of them → a CORRECT
        # via-road must normalize to 100% support, not false hard-fail.
        rl = _resolved()
        rl.road_refs = ["I 81;US 11", "I 81"]
        ev = self._evidence(_moving_epoch(), _lf_plan(), resolved={5: rl})
        assert ev["road_refs"] == ["I-81;US-11", "I-81"]
        assert check_specificity_ladder({"location": _LF5}, ev).passed is True

    def test_unsupported_via_road_fails_with_wired_refs(self):
        # Companion negative: the refs check is ACTIVE (not skipped) — a
        # via-road in zero refs hard-fails.
        rl = _resolved()
        rl.road_refs = ["I 81;US 11", "I 81"]
        ev = self._evidence(_moving_epoch(), _lf_plan(), resolved={5: rl})
        res = check_specificity_ladder({"location": "Wytheville to Fort Chiswell via I-77, VA"}, ev)
        assert res.passed is False
        assert "via-road" in res.message

    def test_fabricated_endpoint_fails_via_wired_geocoder(self):
        # A stubbed forward geocoder placing an endpoint ~600 km
        # off-trace must fire the proximity failure.
        ev = self._evidence(_moving_epoch(), _lf_plan(), geocode_fn=lambda _t: (40.0, -75.0))
        res = check_specificity_ladder({"location": _LF5}, ev)
        assert res.passed is False
        assert "km from the footage" in res.message

    def test_on_trace_endpoint_passes_via_wired_geocoder(self):
        ev = self._evidence(_moving_epoch(), _lf_plan(), geocode_fn=lambda _t: (_LAT0, _LON0))
        assert check_specificity_ladder({"location": _LF5}, ev).passed is True

    def test_forward_geocode_wrapper_swallows_network_failure(self, monkeypatch):
        from gomoveshift.pipeline.chapters import nominatim_source
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import _ladder_forward_geocode

        def _boom(_q):
            raise OSError("network down")

        monkeypatch.setattr(nominatim_source, "_nominatim_search", _boom)
        assert _ladder_forward_geocode("Wytheville") is None


class TestPreEncodeLadderIntegration:
    def test_raises_on_unsupported_tier1(self):
        with pytest.raises(RuntimeError) as exc:
            _assert_pre_encode_location_invariant(
                {5: _LF5},
                {5: _resolved()},
                evidence_by_lf={5: {"gps_points": _sparse(), "window": _W}},
            )
        msg = str(exc.value)
        assert "Pre-encode location invariant HARD FAIL" in msg
        assert "specificity ladder" in msg and "LF 5" in msg

    def test_passes_correct_lf(self):
        # Correct label + supporting evidence → no raise (rerun encode NOT blocked).
        _assert_pre_encode_location_invariant(
            {5: _LF5}, {5: _resolved()}, evidence_by_lf={5: {"gps_points": _moving(), "window": _W}}
        )

    def test_missing_evidence_refuses_the_encode(self):
        with pytest.raises(RuntimeError, match="no location evidence"):
            _assert_pre_encode_location_invariant({5: _LF5}, {5: _resolved()}, evidence_by_lf={})


def _rl_with_road(road="I-81"):
    rl = _resolved()
    rl.road = road
    return rl


class TestDegradeLabelsToAwardedTier:
    """A below-tier-1 (but renderable) LF re-renders its burned OST at the
    AWARDED tier instead of a "must demote" hard-fail. Tier 1/5 unchanged."""

    def test_two_corroborations_below_tier1_renders_tier2(self):
        ev = {"gps_points": _sparse(), "window": _W, "corroborations": 2}
        out = _degrade_labels_to_awarded_tier({5: _LF5}, {5: _rl_with_road()}, {5: ev})
        assert out == {5: "I-81, VA"}

    def test_one_corroboration_renders_tier3_county(self):
        ev = {"gps_points": _sparse(), "window": _W, "corroborations": 1}
        out = _degrade_labels_to_awarded_tier({5: _LF5}, {5: _rl_with_road()}, {5: ev})
        assert out == {5: "Wythe County, VA"}

    def test_bracket_route_alone_renders_tier4_state(self):
        ev = {"gps_points": _sparse(), "window": _W, "has_bracket_route": True}
        out = _degrade_labels_to_awarded_tier({5: _LF5}, {5: _rl_with_road()}, {5: ev})
        assert out == {5: "Virginia"}

    def test_tier1_and_tier5_are_not_demoted(self):
        # tier 1 keeps its label; tier 5 (sparse, no corroboration/bracket)
        # renders nothing → the gate hard-fails it.
        for ev in (
            {"gps_points": _moving(), "window": _W},
            {"gps_points": _sparse(), "window": _W},
        ):
            assert _degrade_labels_to_awarded_tier({5: _LF5}, {5: _rl_with_road()}, {5: ev}) == {}

    def test_authoritative_override_is_not_demoted(self):
        ev = {"gps_points": _sparse(), "window": _W, "corroborations": 2, "authoritative": True}
        out = _degrade_labels_to_awarded_tier({5: _LF5}, {5: _rl_with_road()}, {5: ev})
        assert out == {}


class TestDemotedLabelPassesPreEncodeGate:
    """The re-rendered label is no longer tier-1-shaped, so the pre-encode
    invariant PASSES it — whereas a genuine no-claim (tier 5) still raises."""

    def test_demoted_tier2_label_passes(self):
        rl = _rl_with_road()
        ev = {"gps_points": _sparse(), "window": _W, "corroborations": 2}
        demoted = _degrade_labels_to_awarded_tier({5: _LF5}, {5: rl}, {5: ev})
        assert demoted == {5: "I-81, VA"}
        _assert_pre_encode_location_invariant(demoted, {5: rl}, evidence_by_lf={5: ev})

    def test_tier5_no_claim_still_raises(self):
        rl = _rl_with_road()
        ev = {"gps_points": _sparse(), "window": _W}
        assert _degrade_labels_to_awarded_tier({5: _LF5}, {5: rl}, {5: ev}) == {}
        with pytest.raises(RuntimeError):
            _assert_pre_encode_location_invariant({5: _LF5}, {5: rl}, evidence_by_lf={5: ev})


class TestSparseGpsResolvedFeed:
    """A sparse-GPS LF that voted its OWN dominant county/state supplies a
    below-tier-1 corroboration from that model, so award_tier demotes it to tier
    3/4 and it burns "{county}, {region}" / "{state}" instead of the specificity gate
    hard-failing a tier-1 label nothing could demote. Region source is the LF's
    own ResolvedLocation ONLY — no bracket routing / drive-wide / adjacent-LF
    fallback (that fallback stays deferred)."""

    def _ev(self, pts, lf, resolved):
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import _build_ladder_evidence

        return _build_ladder_evidence(
            {5: _LF5}, {5: pts}, None, lf_plans=[lf], resolved_by_lf=resolved
        )[5]

    def test_sparse_resolved_county_feeds_tier3_and_demotes(self):
        # coverage ~0.43 (below tier 1) + a voted county = 1 corroboration → tier 3.
        rl = _rl_with_road()
        ev = self._ev(_moving_epoch(2040, 3600), _lf_plan(end=3600.0), {5: rl})
        assert ev["corroborations"] == 1
        assert _degrade_labels_to_awarded_tier({5: _LF5}, {5: rl}, {5: ev}) == {
            5: "Wythe County, VA"
        }

    def test_sparse_resolved_state_only_feeds_tier4_state(self):
        rl = _resolved()
        rl.dominant_county = None  # state voted, no county → bracket → tier 4
        ev = self._ev(_moving_epoch(2040, 3600), _lf_plan(end=3600.0), {5: rl})
        assert ev.get("has_bracket_route") is True
        assert "corroborations" not in ev
        assert _degrade_labels_to_awarded_tier({5: _LF5}, {5: rl}, {5: ev}) == {5: "Virginia"}

    def test_high_coverage_resolved_lf_stays_tier1(self):
        # The feed only bites below tier 1 — coverage >= min wins in award_tier.
        rl = _rl_with_road()
        ev = self._ev(_moving_epoch(0, 3600), _lf_plan(end=3600.0), {5: rl})
        assert ev["corroborations"] == 1
        assert _degrade_labels_to_awarded_tier({5: _LF5}, {5: rl}, {5: ev}) == {}

    def test_fully_gps_dead_lf_still_hard_fails_batch(self):
        # Deferred (locked): an LF with NO GPS/model still halts the batch. The
        # feed only reaches LFs that resolved their own model — it must not soften this.
        from types import SimpleNamespace

        from gomoveshift.pipeline.gps_enrichment import _assert_per_lf_locations_resolvable

        dead = SimpleNamespace(
            index=9, segments=[SimpleNamespace(gps_center_lat=0.0, gps_center_lon=0.0)]
        )
        with pytest.raises(ValueError):
            _assert_per_lf_locations_resolvable(
                {"2026-01-01": [dead]}, gps_points_by_lf={}, overrides={}
            )


class TestAwardedTierRecordedOnState:
    """The award the demotion pass already computes is RECORDED on state
    (coverage / tier / evidence) so the provenance sidecar can prove which tier
    each burned label was awarded. Recording only: no gating decision moves."""

    def test_award_recorded_for_demoted_lf_without_changing_the_decision(self):
        import dataclasses

        from gomoveshift.pipeline.drive_state_model import PipelineState

        declared = {f.name for f in dataclasses.fields(PipelineState)}
        assert {
            "per_lf_gps_coverage_pct",
            "per_lf_awarded_tier",
            "per_lf_tier_evidence",
        } <= declared, "ladder award fields must be DECLARED on PipelineState"

        state = make_state()
        ev = {"gps_points": _sparse(), "window": _W, "corroborations": 2}
        out = _degrade_labels_to_awarded_tier({5: _LF5}, {5: _rl_with_road()}, {5: ev}, state=state)
        assert out == {5: "I-81, VA"}  # unchanged gating decision
        assert state.per_lf_awarded_tier == {5: 2}
        assert state.per_lf_gps_coverage_pct == {5: 3.0}  # percent, not a fraction
        assert state.per_lf_tier_evidence[5]["corroborations"] == 2

    def test_tier1_lf_is_recorded_even_though_it_is_never_demoted(self):
        state = make_state()
        ev = {"gps_points": _moving(), "window": _W}
        assert (
            _degrade_labels_to_awarded_tier({5: _LF5}, {5: _rl_with_road()}, {5: ev}, state=state)
            == {}
        )
        assert state.per_lf_awarded_tier == {5: 1}
        assert state.per_lf_gps_coverage_pct == {5: 100.0}

    def test_recorded_evidence_is_json_serialisable(self):
        # The live evidence dict carries a raw GPS trace and a geocode callable;
        # neither can reach pipeline_state.json, so only a scalar projection is
        # recorded — unserialisable state is impossible by construction.
        state = make_state()
        ev = {
            "gps_points": _moving(),
            "window": _W,
            "corroborations": 2,
            "road_refs": ["I-81"],
            "duration": 900.0,
            "geocode_fn": lambda _t: (_LAT0, _LON0),
        }
        _degrade_labels_to_awarded_tier({5: _LF5}, {5: _rl_with_road()}, {5: ev}, state=state)
        recorded = state.per_lf_tier_evidence[5]
        json.dumps(recorded)
        assert "geocode_fn" not in recorded and "gps_points" not in recorded
        assert recorded["gps_point_count"] == 101
        assert recorded["road_refs"] == ["I-81"]
