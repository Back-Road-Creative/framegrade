"""``analysis.sign_label``: derive an LF's label from its road signs, for an LF
the camera never located.

The deriver is the ONE place signs may become a location authority, and only
for an LF with zero GPS fixes. Every rule below is a fail-closed refusal: the
label is produced only when at least ``min_corroborating_hits`` sign hits from
DISTINCT seconds name the SAME county, no other county is named, exactly one
state is named, and the LF has no GPS fix at all. Anything else is ``None``
with a reason the dry-run can print.
"""

from __future__ import annotations

import pytest

from gomoveshift.analysis.sign_evidence import SignHit
from gomoveshift.analysis.sign_label import (
    SIGN_LABEL_SOURCE,
    SignLabel,
    derive_sign_label,
    derive_sign_labels,
)
from gomoveshift.models import LocationType

COUNTY = LocationType.COUNTY_BOUNDARY.value
STATE = LocationType.STATE_BOUNDARY.value
TOWN = LocationType.TOWN_ENTRY.value
ROUTE = LocationType.ROUTE_MARKER.value


def _hit(t_local: float, *claims, raw_text: str = "", episode: int = 0):
    text = raw_text or " / ".join(v for _, v in claims) or "no place named"
    return SignHit(
        episode_index=episode,
        t_local=t_local,
        ts_day=t_local,
        date="2026-03-05",
        raw_text=text,
        ocr_conf=0.9,
        lat=None,
        lon=None,
        claims=[{"kind": k, "value": v, "raw_text": text} for k, v in claims],
    )


def _colleton(t: float) -> SignHit:
    return _hit(t, (COUNTY, "Colleton"), (STATE, "South Carolina"), raw_text="Colleton County SC")


# ── the label that is produced ─────────────────────────────────────────


def test_two_hits_from_distinct_seconds_on_one_county_and_state_yield_the_county_label():
    result = derive_sign_label(3, [_colleton(100.0), _colleton(340.0)])
    assert isinstance(result, SignLabel)
    assert result.label == "Colleton County, SC"
    assert result.county == "Colleton"
    assert result.state == "South Carolina"
    assert result.lf_index == 3
    assert result.hit_count == 2
    assert result.hit_seconds == [100.0, 340.0]
    assert result.texts == ["Colleton County SC"]
    assert result.source == SIGN_LABEL_SOURCE == "sign_label"


def test_the_state_may_come_from_a_different_sign_than_the_county():
    # "Entering Colleton County" twice, "Welcome to South Carolina" once.
    hits = [
        _hit(50.0, (COUNTY, "Colleton"), raw_text="Entering Colleton County"),
        _hit(51.0, (COUNTY, "Colleton"), raw_text="Entering Colleton County"),
        _hit(900.0, (STATE, "South Carolina"), raw_text="Welcome to South Carolina"),
    ]
    result = derive_sign_label(1, hits)
    assert result is not None
    assert result.label == "Colleton County, SC"
    # the state-only sign is evidence for the state, not a county hit
    assert result.hit_count == 2


def test_town_and_route_claims_never_make_a_label_on_their_own():
    hits = [
        _hit(10.0, (TOWN, "Walterboro"), (ROUTE, "I-95")),
        _hit(20.0, (TOWN, "Walterboro"), (STATE, "South Carolina")),
    ]
    assert derive_sign_label(1, hits) is None


def test_the_label_is_the_county_shape_only_never_a_to_b():
    hits = [_colleton(1.0), _colleton(2.0), _hit(3.0, (TOWN, "Walterboro"))]
    result = derive_sign_label(1, hits)
    assert result is not None
    assert result.label == "Colleton County, SC"
    assert "Walterboro" not in result.label
    assert " to " not in result.label


def test_a_dict_hit_straight_off_state_is_accepted():
    hits = [_colleton(1.0).to_dict(), _colleton(5.0).to_dict()]
    result = derive_sign_label(1, hits)
    assert result is not None and result.label == "Colleton County, SC"


def test_to_dict_round_trips_the_evidence_record():
    result = derive_sign_label(2, [_colleton(1.0), _colleton(5.0)])
    assert result is not None
    d = result.to_dict()
    assert d["label"] == "Colleton County, SC"
    assert d["source"] == "sign_label"
    assert d["hit_count"] == 2
    assert d["hit_seconds"] == [1.0, 5.0]
    assert d["texts"] == ["Colleton County SC"]


# ── refusals: each rule, one test ──────────────────────────────────────


def test_one_hit_is_below_the_floor():
    assert derive_sign_label(1, [_colleton(100.0)]) is None


def test_two_hits_in_the_same_second_count_once():
    assert derive_sign_label(1, [_colleton(100.2), _colleton(100.8)]) is None


def test_the_floor_follows_min_corroborating_hits():
    hits = [_colleton(1.0), _colleton(2.0)]
    assert derive_sign_label(1, hits, min_corroborating_hits=3) is None
    assert derive_sign_label(1, hits + [_colleton(3.0)], min_corroborating_hits=3) is not None


def test_a_second_county_anywhere_in_the_lf_refuses_the_label():
    hits = [_colleton(1.0), _colleton(2.0), _hit(3.0, (COUNTY, "Dorchester"))]
    assert derive_sign_label(1, hits) is None


def test_no_state_refuses_the_label():
    hits = [
        _hit(1.0, (COUNTY, "Colleton"), raw_text="Entering Colleton County"),
        _hit(2.0, (COUNTY, "Colleton"), raw_text="Entering Colleton County"),
    ]
    assert derive_sign_label(1, hits) is None


def test_two_states_refuse_the_label():
    hits = [_colleton(1.0), _colleton(2.0), _hit(3.0, (STATE, "Georgia"))]
    assert derive_sign_label(1, hits) is None


def test_an_lf_with_any_gps_fix_never_takes_a_sign_label():
    hits = [_colleton(1.0), _colleton(2.0)]
    assert derive_sign_label(1, hits, gps_fix_count=1) is None
    assert derive_sign_label(1, hits, gps_fix_count=0) is not None


def test_no_hits_is_none():
    assert derive_sign_label(1, []) is None
    assert derive_sign_label(1, None) is None


# ── the reasons the dry-run prints ─────────────────────────────────────


@pytest.mark.parametrize(
    "hits, kwargs, fragment",
    [
        ([_colleton(1.0)], {}, "1 distinct second(s)"),
        ([_colleton(1.0), _colleton(2.0), _hit(3.0, (COUNTY, "Dorchester"))], {}, "Dorchester"),
        (
            [
                _hit(1.0, (COUNTY, "Colleton"), raw_text="Entering Colleton County"),
                _hit(2.0, (COUNTY, "Colleton"), raw_text="Entering Colleton County"),
            ],
            {},
            "no state",
        ),
        ([_colleton(1.0), _colleton(2.0), _hit(3.0, (STATE, "Georgia"))], {}, "Georgia"),
        ([_colleton(1.0), _colleton(2.0)], {"gps_fix_count": 4}, "4 GPS fix"),
        ([_hit(1.0, (TOWN, "Walterboro"))], {}, "no county"),
    ],
)
def test_refusal_reasons_name_the_rule(hits, kwargs, fragment):
    from gomoveshift.analysis.sign_label import explain_sign_label

    label, reasons = explain_sign_label(7, hits, **kwargs)
    assert label is None
    assert any(fragment in r for r in reasons), reasons
    assert all(r.startswith("LF 7:") for r in reasons)


def test_explain_returns_the_label_and_an_agree_reason_when_it_passes():
    from gomoveshift.analysis.sign_label import explain_sign_label

    label, reasons = explain_sign_label(7, [_colleton(1.0), _colleton(9.0)])
    assert label is not None and label.label == "Colleton County, SC"
    assert any("Colleton County, SC" in r and "2 distinct second(s)" in r for r in reasons)


# ── the per-LF convenience ─────────────────────────────────────────────


def test_derive_sign_labels_covers_only_zero_fix_lfs_that_pass():
    hits_by_lf = {
        1: [_colleton(1.0), _colleton(2.0)],  # passes
        2: [_colleton(1.0)],  # below floor
        3: [_colleton(1.0), _colleton(2.0)],  # has a GPS fix
    }
    gps_points_by_lf = {3: [object()]}
    out = derive_sign_labels(hits_by_lf, gps_points_by_lf=gps_points_by_lf)
    assert set(out) == {1}
    assert out[1].label == "Colleton County, SC"


def test_derive_sign_labels_with_nothing_is_empty():
    assert derive_sign_labels({}) == {}
    assert derive_sign_labels(None) == {}
