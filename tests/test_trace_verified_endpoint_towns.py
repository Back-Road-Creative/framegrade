"""Endpoint towns verified against the LF's own GPS trace.

The location invariant reconciles an ``"A to B"`` label's endpoint towns with
the reverse geocoder's place sequence. Irish reverse geocodes bucket every
sample under the "West Cork" municipal district (``city`` outranks
``village``), so the GPS-true label "Goleen to Crookhaven, County Cork"
would fail as wrong-town.
These pin the geometric second opinion: a settlement of exactly the endpoint's
name within 2 km of the LF's own fixes is town-level truth; anything less
fails closed, exactly as before.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from gomoveshift.models import ResolvedLocation
from gomoveshift.pipeline import forward_geocode
from gomoveshift.pipeline.chapters import nominatim_source
from gomoveshift.pipeline.chapters.nominatim_source import (
    _nominatim_places as _real_nominatim_places,
)
from gomoveshift.pipeline.drive_phases import phase_bc_day_first
from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
    _assert_pre_encode_location_invariant,
    _populate_resolved_locations,
)
from gomoveshift.pipeline.validation.content_gates import (
    check_location_invariant,
    resolved_location_gate_dict,
    trace_verified_endpoint_towns,
)

from _state_helpers import make_state

LABEL = "Goleen to Crookhaven, County Cork"
# A corroborated evidence record: these tests exercise the county/endpoint
# invariant, so the ladder needs a claim basis but no trace checks (U1).
_EV = {1: {"corroborations": 1}}

# Nominatim-shaped hits for the scoped queries. The top hit
# for "Goleen, County Cork" is the electoral division, 2.7 km from the village;
# it is listed first here so the gate-level tests prove nearest-wins even if a
# non-settlement hit ever reached them.
_GOLEEN_ED = ("Goleen", 51.4999, -9.7498)
_GOLEEN_VILLAGE = ("Goleen", 51.4950, -9.7113)
_GOLEEN_LOCALITY = ("Goleen", 51.4933, -9.7082)
_CROOKHAVEN_VILLAGE = ("Crookhaven", 51.4684, -9.7257)
_CROOKHAVEN_LOCALITY = ("Crookhaven", 51.4668, -9.7270)
_HITS = {
    "Goleen, County Cork": [_GOLEEN_ED, _GOLEEN_VILLAGE, _GOLEEN_LOCALITY],
    "Crookhaven, County Cork": [_CROOKHAVEN_VILLAGE, _CROOKHAVEN_LOCALITY],
}


def _trace(n=13):
    """Goleen (0.12 km off the village centre) to Crookhaven, one fix a minute.

    The electoral-division centroid sits 2.7 km from the nearest fix."""
    (a_lat, a_lon), (b_lat, b_lon) = (51.4990, -9.7090), (51.4690, -9.7250)
    return [
        {
            "lat": a_lat + (b_lat - a_lat) * i / (n - 1),
            "lon": a_lon + (b_lon - a_lon) * i / (n - 1),
            "time": 1758100000.0 + 60.0 * i,
        }
        for i in range(n)
    ]


def _places(hits=None, calls=None):
    hits = _HITS if hits is None else hits

    def _fn(query, **_kw):
        if calls is not None:
            calls.append(query)
        return list(hits.get(query, []))

    return _fn


def _west_cork_geo():
    """Every sample reverse-geocodes to the municipal district, as Irish reverse geocodes can."""

    class _Geo:
        def reverse_geocode_detailed(self, lat, lon):
            return {
                "city": "West Cork",
                "county": "County Cork",
                "state": None,
                "country": "Ireland",
            }

    return _Geo()


def _resolved(**over):
    """The gate dict of the West Cork label."""
    d = {
        "dominant_county": "County Cork",
        "dominant_state": None,
        "regions": ["County Cork"],
        "vote": {"County Cork": 1.0},
        "ordered_place_sequence": ["West Cork"],
        "destination": None,
    }
    d.update(over)
    return d


def _gate(label, resolved):
    return check_location_invariant(
        {"title": label, "location": label, "resolved_location": resolved}
    )


# ── the gate reads trace-verified places as town truth ──────────────────


class TestGateReadsTraceVerifiedPlaces:
    def test_goleen_label_fails_without_verified_places(self):
        gate = _gate(LABEL, _resolved())
        assert not gate.passed
        assert "'Goleen'" in gate.message and "'Crookhaven'" in gate.message

    def test_trace_verified_endpoints_are_town_truth(self):
        gate = _gate(LABEL, _resolved(trace_verified_places=["Crookhaven", "Goleen"]))
        assert gate.passed, gate.message

    def test_one_verified_endpoint_does_not_rescue_the_other(self):
        gate = _gate(LABEL, _resolved(trace_verified_places=["Goleen"]))
        assert not gate.passed
        assert "'Crookhaven'" in gate.message and "'Goleen'" not in gate.message

    def test_failure_says_the_trace_check_ran(self):
        gate = _gate(LABEL, _resolved(trace_verified_places=[]))
        assert not gate.passed
        assert "within 2 km of the GPS trace" in gate.message
        # Not computed (legacy model): the message must not claim a check ran.
        assert "GPS trace" not in _gate(LABEL, _resolved()).message

    def test_title_keyword_check_reads_trace_verified_places(self):
        from gomoveshift.pipeline.validation.content_gates import check_metadata_quality

        meta = {
            "title": "Goleen to Crookhaven in 4K",
            "description": "POV drive along the Mizen peninsula.",
            "location": "West Cork, County Cork",
            "video_duration": 300,  # < 600 s keeps the chapter-density gate inert
            "tags": ["POV driving", "drive", "road trip", "4K driving"],
            "has_gps": True,
            "resolved_location": _resolved(trace_verified_places=["Crookhaven", "Goleen"]),
        }
        assert "no location keywords" not in check_metadata_quality(meta).message
        meta["resolved_location"] = _resolved()
        assert "no location keywords" in check_metadata_quality(meta).message

    def test_verified_places_never_open_the_town_check_on_a_model_without_town_truth(self):
        # Rescue-only: a model with no place sequence/destination skipped the
        # town check before and still does — verified places never make it stricter.
        gate = _gate(
            LABEL,
            _resolved(ordered_place_sequence=[], trace_verified_places=["Goleen"]),
        )
        assert gate.passed, gate.message


# ── the geometric verifier ──────────────────────────────────────────────


class TestTraceVerifiedEndpointTowns:
    def test_both_label_endpoints_verify(self):
        assert trace_verified_endpoint_towns(LABEL, _resolved(), _trace(), _places()) == [
            "Crookhaven",
            "Goleen",
        ]

    def test_nearest_same_name_hit_wins_over_the_top_hit(self):
        hits = {"Goleen, County Cork": [_GOLEEN_ED, _GOLEEN_VILLAGE]}
        assert trace_verified_endpoint_towns(LABEL, _resolved(), _trace(), _places(hits)) == [
            "Goleen"
        ]

    def test_the_far_top_hit_alone_does_not_verify(self):
        hits = {"Goleen, County Cork": [_GOLEEN_ED], "Crookhaven, County Cork": []}
        assert trace_verified_endpoint_towns(LABEL, _resolved(), _trace(), _places(hits)) == []

    def test_same_name_place_far_from_the_trace_does_not_verify(self):
        hits = {
            "Goleen, County Cork": [_GOLEEN_VILLAGE],
            "Crookhaven, County Cork": [("Crookhaven", 51.75, -9.55)],  # ~30 km off
        }
        assert trace_verified_endpoint_towns(LABEL, _resolved(), _trace(), _places(hits)) == [
            "Goleen"
        ]

    def test_a_different_name_beside_the_trace_does_not_verify(self):
        hits = {
            "Goleen, County Cork": [("Goleen Pier", 51.4950, -9.7110)],
            "Crookhaven, County Cork": [("crookhaven", 51.4684, -9.7257)],
        }
        assert trace_verified_endpoint_towns(LABEL, _resolved(), _trace(), _places(hits)) == [
            "Crookhaven"
        ]

    @pytest.mark.parametrize(
        "places_fn",
        [
            lambda _q: [],
            lambda _q: None,
            lambda _q: (_ for _ in ()).throw(RuntimeError("nominatim down")),
            lambda _q: [("Goleen",)],  # malformed hit
        ],
        ids=["no-hits", "none", "raises", "malformed"],
    )
    def test_no_or_broken_geocode_fails_closed(self, places_fn):
        assert trace_verified_endpoint_towns(LABEL, _resolved(), _trace(), places_fn) == []

    def test_query_is_scoped_to_the_resolved_county_and_state(self):
        calls: list[str] = []
        trace_verified_endpoint_towns(LABEL, _resolved(), _trace(), _places(calls=calls))
        assert sorted(calls) == ["Crookhaven, County Cork", "Goleen, County Cork"]
        calls.clear()
        trace_verified_endpoint_towns(
            "Radford to Salem",
            _resolved(
                dominant_county="Montgomery County",
                dominant_state="Virginia",
                regions=["Montgomery County, Virginia"],
                ordered_place_sequence=["Christiansburg"],
            ),
            _trace(),
            _places({}, calls),
        )
        assert sorted(calls) == [
            "Radford, Montgomery County, Virginia",
            "Salem, Montgomery County, Virginia",
        ]

    def test_an_endpoint_over_the_county_line_is_found_in_its_sampled_region(self):
        calls: list[str] = []
        hits = {"Crookhaven, County Kerry": [_CROOKHAVEN_VILLAGE]}
        out = trace_verified_endpoint_towns(
            LABEL,
            _resolved(regions=["County Cork", "County Kerry"]),
            _trace(),
            _places(hits, calls),
        )
        assert out == ["Crookhaven"]
        assert calls == [
            "Crookhaven, County Cork",
            "Crookhaven, County Kerry",  # stops here: verified
            "Goleen, County Cork",
            "Goleen, County Kerry",
        ]

    def test_endpoint_already_in_truth_is_never_geocoded(self):
        calls: list[str] = []
        out = trace_verified_endpoint_towns(
            "Schull to Goleen, County Cork",
            _resolved(ordered_place_sequence=["Schull", "West Cork"]),
            _trace(),
            _places(calls=calls),
        )
        assert calls == ["Goleen, County Cork"]
        assert out == ["Goleen"]

    def test_non_arc_label_never_geocodes(self):
        calls: list[str] = []
        out = trace_verified_endpoint_towns(
            "West Cork, County Cork", _resolved(), _trace(), _places(calls=calls)
        )
        assert out == [] and calls == []

    def test_untimed_trace_verifies_nothing(self):
        untimed = [{"lat": p["lat"], "lon": p["lon"]} for p in _trace()]
        assert trace_verified_endpoint_towns(LABEL, _resolved(), untimed, _places()) == []


# ── populate stamps it; the pre-encode gate reads it ────────────────────


class TestPopulateStampsTraceVerifiedPlaces:
    def _populate(self, state=None, **kw):
        state = state or SimpleNamespace(per_lf_resolved_location=None)
        _populate_resolved_locations(
            state,
            [SimpleNamespace(index=1)],
            {1: _trace()},
            _west_cork_geo(),
            per_lf_ost_text={1: LABEL},
            **kw,
        )
        return state

    def test_model_is_the_west_cork_shape_and_unverified_without_a_geocoder(self):
        rl = self._populate(places_fn=None).per_lf_resolved_location[1]
        assert rl.ordered_place_sequence == ["West Cork"]
        assert rl.regions == ["County Cork"]
        assert rl.trace_verified_places is None
        with pytest.raises(RuntimeError, match="Goleen"):
            _assert_pre_encode_location_invariant({1: LABEL}, {1: rl}, evidence_by_lf=_EV)

    def test_populate_verifies_and_the_pre_encode_gate_passes(self):
        state = self._populate(places_fn=_places())
        rl = state.per_lf_resolved_location[1]
        assert rl.trace_verified_places == ["Crookhaven", "Goleen"]
        _assert_pre_encode_location_invariant(
            {1: LABEL}, state.per_lf_resolved_location, evidence_by_lf=_EV
        )

    def test_unverifiable_endpoint_still_refuses_the_encode(self):
        state = self._populate(places_fn=_places({"Goleen, County Cork": [_GOLEEN_VILLAGE]}))
        assert state.per_lf_resolved_location[1].trace_verified_places == ["Goleen"]
        with pytest.raises(RuntimeError, match="Crookhaven"):
            _assert_pre_encode_location_invariant(
                {1: LABEL}, state.per_lf_resolved_location, evidence_by_lf=_EV
            )

    def test_production_forward_geocoder_is_the_default(self, monkeypatch):
        calls: list[str] = []
        monkeypatch.setattr(forward_geocode, "_nominatim_places", _places(calls=calls))
        rl = self._populate().per_lf_resolved_location[1]
        assert rl.trace_verified_places == ["Crookhaven", "Goleen"]
        assert sorted(calls) == ["Crookhaven, County Cork", "Goleen, County Cork"]

    def test_production_geocoder_failure_verifies_nothing(self, monkeypatch, caplog):
        def _boom(*_a, **_k):
            raise OSError("network down")

        monkeypatch.setattr(forward_geocode, "_nominatim_places", _boom)
        with caplog.at_level("WARNING", logger=phase_bc_day_first.__name__):
            assert phase_bc_day_first._forward_geocode_places("Goleen, County Cork") == []
        # The failure is named, not mistaken for "no settlement of that name".
        assert "'Goleen, County Cork' — geocoder failed (network down)" in caplog.text
        assert self._populate().per_lf_resolved_location[1].trace_verified_places == []

    def test_persisted_unverified_model_is_verified_on_the_fast_path(self):
        state = self._populate(places_fn=None)
        rl = state.per_lf_resolved_location[1]
        self._populate(state, places_fn=_places())
        assert state.per_lf_resolved_location[1] is rl  # reused, not re-derived
        assert rl.trace_verified_places == ["Crookhaven", "Goleen"]

    def test_verified_model_is_not_geocoded_again(self):
        state = self._populate(places_fn=_places())
        calls: list[str] = []
        self._populate(state, places_fn=_places(calls=calls))
        assert calls == []


# ── every gate surface carries it ───────────────────────────────────────


def _evidence(towns):
    """A location-evidence record (dict form) entering ``towns`` in order."""
    return {
        "segment_id": "lf01",
        "provenance": "test",
        "method": "reverse_geocode",
        "places": [
            {
                "name": t,
                "county": "County Cork",
                "state": "Munster",
                "scope": "in",
                "method": "reverse_geocode",
                "provenance": "test",
                "segment_id": "lf01",
                "evidence_id": f"ev-{t}",
            }
            for t in towns
        ],
        "conflicts": [],
    }


class TestGateDictCarriesTraceVerifiedPlaces:
    def _rl(self, **over):
        fields = {
            "dominant_county": "County Cork",
            "country_code": "Ireland",
            "regions": ["County Cork"],
            "vote": {"County Cork": 1.0},
            "ordered_place_sequence": ["West Cork"],
            "trace_verified_places": ["Crookhaven", "Goleen"],
        }
        fields.update(over)
        return ResolvedLocation(**fields)

    def test_gate_dict_carries_the_key(self):
        d = resolved_location_gate_dict(self._rl())
        assert d["trace_verified_places"] == ["Crookhaven", "Goleen"]
        assert (
            resolved_location_gate_dict(self._rl(trace_verified_places=None))[
                "trace_verified_places"
            ]
            is None
        )
        assert _gate(LABEL, d).passed

    def test_phase_d_metadata_threads_it_to_the_post_encode_gate(self, tmp_path):
        from gomoveshift.pipeline.drive_phases.phase_d import _build_metadata_per_lf

        gen = MagicMock()
        gen.generate.side_effect = lambda **_kw: {"title": LABEL, "description": "d"}
        with patch(
            "gomoveshift.pipeline.validation.content_gates.check_description_redundancy",
            return_value=SimpleNamespace(passed=True, message=""),
        ):
            out = _build_metadata_per_lf(
                meta_gen=gen,
                drive_plan=SimpleNamespace(
                    regular_lfs=[SimpleNamespace(index=1, total_duration=1800.0, segments=[])]
                ),
                regular_lf_paths=["/v/lf.mp4"],
                state=make_state(
                    per_lf_resolved_location={1: self._rl()},
                    # GMS-17b: Phase D refuses an A-to-B title with no record.
                    per_lf_location_evidence={1: _evidence(["Goleen", "Crookhaven"])},
                ),
                location=LABEL,
                publish_day_full="Sunday",
                project_base="p",
                output_base=tmp_path,
            )
        resolved = out[1]["resolved_location"]
        assert resolved["trace_verified_places"] == ["Crookhaven", "Goleen"]
        assert _gate(LABEL, resolved).passed

    def test_model_round_trips_state_serialisation(self):
        rl = self._rl()
        assert ResolvedLocation(**json.loads(json.dumps(asdict(rl)))) == rl
        legacy = asdict(rl)
        legacy.pop("trace_verified_places")
        assert ResolvedLocation(**legacy).trace_verified_places is None


# ── the settlement-candidate forward geocoder ───────────────────────────


def _urlopen_returning(payload, captured):
    def _stub(req, *args, **kwargs):
        captured["url"] = req.full_url
        resp = MagicMock()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        resp.read.return_value = json.dumps(payload).encode("utf-8")
        return resp

    return _stub


class TestNominatimPlaces:
    _PAYLOAD = [
        {
            "lat": "51.4999",
            "lon": "-9.7498",
            "class": "boundary",
            "type": "administrative",
            "addresstype": "city_district",
            "name": "Goleen",
            "display_name": "Goleen, West Cork, County Cork, Munster, Ireland",
        },
        {
            "lat": "51.4950",
            "lon": "-9.7113",
            "class": "place",
            "type": "village",
            "addresstype": "village",
            "name": "Goleen",
            "display_name": "Goleen, West Cork, County Cork, Munster, Ireland",
        },
        {
            "lat": "51.4933",
            "lon": "-9.7082",
            "class": "place",
            "type": "locality",
            "addresstype": "locality",
            "name": "",
            "display_name": "Goleen, West Cork, County Cork, Munster, Ireland",
        },
    ]

    def test_returns_named_settlement_hits_only(self):
        captured: dict[str, str] = {}
        with patch("urllib.request.urlopen", _urlopen_returning(self._PAYLOAD, captured)):
            with patch("time.sleep"):
                out = _real_nominatim_places("Goleen, County Cork")
        assert out == [("Goleen", 51.4950, -9.7113), ("Goleen", 51.4933, -9.7082)]
        query = captured["url"].split("?", 1)[1]
        assert "limit=5" in query and query.count("&") == 2

    def test_town_modelled_only_as_a_boundary_is_a_candidate(self):
        """A US census-designated town Nominatim returns only as an
        administrative boundary whose ``addresstype`` is ``town`` is still the
        settlement; dropping it left a GPS-true "Exampleville, TN to ..." label
        with no candidate at all ("nearest: none") and the pre-encode invariant
        failed. The 2 km trace check still
        decides; a non-settlement boundary (a county) and a non-boundary hit
        that merely carries a town addresstype (an amenity) stay out."""
        payload = [
            {
                "lat": "40.10000",
                "lon": "-80.20000",
                "class": "boundary",
                "type": "administrative",
                "addresstype": "town",
                "name": "Exampleville",
                "display_name": "Exampleville, Sample County, Tennessee, United States",
            },
            {
                "lat": "40.30000",
                "lon": "-80.40000",
                "class": "boundary",
                "type": "administrative",
                "addresstype": "county",
                "name": "Exampleville",
                "display_name": "Exampleville, Tennessee, United States",
            },
            {
                "lat": "40.10100",
                "lon": "-80.20100",
                "class": "amenity",
                "type": "townhall",
                "addresstype": "town",
                "name": "Exampleville",
                "display_name": "Exampleville, Sample County, Tennessee, United States",
            },
        ]
        with patch("urllib.request.urlopen", _urlopen_returning(payload, {})):
            with patch("time.sleep"):
                out = _real_nominatim_places("Exampleville, Sample County, Tennessee")
        assert out == [("Exampleville", 40.1, -80.2)]

    def test_no_hits_is_empty(self):
        with patch("urllib.request.urlopen", _urlopen_returning([], {})):
            with patch("time.sleep"):
                assert _real_nominatim_places("Nowhere, County Cork") == []

    def test_hermetic_fixture_stubs_it(self):
        assert nominatim_source._nominatim_places("Goleen, County Cork") == []


# ── state-qualified endpoints (cross-state USA arc) ─────────────────────
# The builder's cross-state form
# "Blountville, TN to McMullin, VA via I-81" never matched the endpoint regex
# (its name groups refuse a comma), so no endpoint was trace-checked and the
# whole-label fallback would hard-fail a GPS-true label; in this fixture
# Blountville lies ~1.5 km and McMullin ~0.5 km from the trace.

_USA_LABEL = "Blountville, TN to McMullin, VA via I-81"
_USA_HITS = {
    "Blountville, Sullivan County, Tennessee": [("Blountville", 36.5332, -82.3268)],
    "McMullin, Smyth County, Virginia": [("McMullin", 36.8139, -81.5717)],
}


def _usa_resolved(**over):
    """The gate dict of a cross-state label."""
    d = {
        "dominant_county": "Washington County",
        "dominant_state": "Virginia",
        "regions": [
            "Sullivan County, Tennessee",
            "None, Virginia",
            "Washington County, Virginia",
            "Smyth County, Virginia",
        ],
        "vote": {
            "Sullivan County, Tennessee": 0.15,
            "None, Virginia": 0.1,
            "Washington County, Virginia": 0.5,
            "Smyth County, Virginia": 0.15,
        },
        "ordered_place_sequence": [
            "Kingsport",
            "Walnut Hill",
            "Bristol",
            "Washington County",
            "Meadowview",
            "Old Glade Spring",
            "Seven Mile Ford",
        ],
        "destination": None,
    }
    d.update(over)
    return d


def _usa_trace(n=13):
    """First fix ~1.5 km from Blountville, last ~0.5 km from McMullin."""
    (a_lat, a_lon), (b_lat, b_lon) = (36.5391, -82.3420), (36.8175, -81.5744)
    return [
        {
            "lat": a_lat + (b_lat - a_lat) * i / (n - 1),
            "lon": a_lon + (b_lon - a_lon) * i / (n - 1),
            "time": 1772600000.0 + 60.0 * i,
        }
        for i in range(n)
    ]


class TestStateQualifiedEndpoints:
    def test_state_suffixed_endpoints_are_parsed(self):
        from gomoveshift.pipeline.validation.content_gates import _endpoint_town_phrases

        assert _endpoint_town_phrases(_USA_LABEL) == {"Blountville", "McMullin"}

    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            ("Goleen to Crookhaven, County Cork", {"Goleen", "Crookhaven"}),
            ("Kingsport to Bristol via I-81, Tennessee", {"Kingsport", "Bristol"}),
            ("Kingsport to Bristol via I-81, TN", {"Kingsport", "Bristol"}),
            ("Kingsport, TN to Bristol, TN", {"Kingsport", "Bristol"}),
        ],
    )
    def test_existing_arc_forms_parse_unchanged(self, label, expected):
        from gomoveshift.pipeline.validation.content_gates import _endpoint_town_phrases

        assert _endpoint_town_phrases(label) == expected

    def test_unverified_endpoints_fail_by_name(self):
        res = _gate(_USA_LABEL, _usa_resolved(trace_verified_places=[]))
        assert not res.passed
        assert "title endpoint town 'Blountville'" in res.message
        assert "title endpoint town 'McMullin'" in res.message

    def test_trace_verified_state_suffixed_endpoints_pass(self):
        res = _gate(_USA_LABEL, _usa_resolved(trace_verified_places=["Blountville", "McMullin"]))
        assert res.passed, res.message

    def test_both_endpoints_verify_against_the_trace(self):
        got = trace_verified_endpoint_towns(
            _USA_LABEL, _usa_resolved(), _usa_trace(), _places(_USA_HITS)
        )
        assert got == ["Blountville", "McMullin"]
