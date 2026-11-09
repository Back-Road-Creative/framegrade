"""Guide/destination signs: an ADVISORY claim kind, never a label, never a conflict.

Hermetic: every ``geocode_fn`` is a local dict-backed stub.
"""

from gomoveshift.analysis.ocr import parse_sign_claims
from gomoveshift.analysis.sign_evidence import (
    STATUS_CONFLICTS,
    STATUS_CORROBORATES,
    STATUS_NO_SIGNAL,
    SignClaim,
    SignHit,
)
from gomoveshift.analysis.sign_label import explain_sign_label
from gomoveshift.analysis.sign_verify import verify_all, verify_lf
from gomoveshift.models import LocationType, ResolvedLocation

DEST = LocationType.DESTINATION.value


def _dest(text):
    return [c for c in parse_sign_claims(text) if c.kind == DEST]


# ── parser ──


def test_noisy_exit_sign_reads_the_place_as_seen_and_keeps_raw_text():
    raw = "EExIT 283 42 Noodstock 1 MILE"
    claims = _dest(raw)
    assert [(c.value, c.distance_mi, c.raw_text) for c in claims] == [("Noodstock", 1.0, raw)]


def test_exit_sign_and_bare_place_sign():
    assert [(c.value, c.distance_mi) for c in _dest("EXIT 283 Woodstock 1 MILE")] == [
        ("Woodstock", 1.0)
    ]
    assert [c.value for c in _dest("Woodstock 1 MILE")] == ["Woodstock"]


def test_plural_miles_and_fraction():
    assert _dest("Charlottesville 12 MILES")[0].distance_mi == 12.0
    assert _dest("Woodstock 1/2 MILE")[0].distance_mi == 0.5


def test_bare_exit_number_is_no_claim():
    assert parse_sign_claims("EXIT 283") == []


def test_route_token_is_not_a_destination_but_still_a_route():
    claims = parse_sign_claims("I 81 Woodstock 1 MILE")
    assert [c.value for c in claims if c.kind == DEST] == ["Woodstock"]
    assert any(c.kind == LocationType.ROUTE_MARKER.value and c.value == "I-81" for c in claims)


def test_state_name_is_not_a_destination():
    assert _dest("Virginia 5 MILES") == []


def test_existing_welcome_sign_unchanged():
    claims = parse_sign_claims("Welcome to Woodstock")
    assert [(c.kind, c.value) for c in claims] == [(LocationType.TOWN_ENTRY.value, "Woodstock")]


# ── SignHit round trip ──


def test_legacy_claim_dict_without_distance_loads():
    hit = SignHit.from_dict(
        {"episode_index": 0, "claims": [{"kind": DEST, "value": "Woodstock", "raw_text": "x"}]}
    )
    assert hit.parsed_claims()[0].distance_mi is None


def test_distance_round_trips():
    hit = SignHit(
        episode_index=0,
        t_local=1.0,
        ts_day=1.0,
        date="2026-07-01",
        raw_text="x",
        claims=[SignClaim(kind=DEST, value="Woodstock", raw_text="x", distance_mi=1.5)],
    )
    back = SignHit.from_dict(hit.to_dict())
    assert back.parsed_claims()[0].distance_mi == 1.5


# ── verify_lf ──

ANCHOR = (38.88, -78.50)


def _model():
    return ResolvedLocation(
        dominant_county="Shenandoah County",
        dominant_state="Virginia",
        regions=["Shenandoah County, Virginia"],
        vote={"Shenandoah County, Virginia": 1.0},
    )


def _hit(text="EXIT 283 Woodstock 1 MILE", value="Woodstock", dist=1.0, anchored=True):
    return SignHit(
        episode_index=0,
        t_local=5.0,
        ts_day=5.0,
        date="2026-07-01",
        raw_text=text,
        lat=ANCHOR[0] if anchored else None,
        lon=ANCHOR[1] if anchored else None,
        claims=[{"kind": DEST, "value": value, "raw_text": text, "distance_mi": dist}],
    )


def _geo(table):
    return lambda name: table.get(name)


def _reasons(v):
    return " | ".join(v.reasons)


def test_corroborates_when_place_is_near_the_anchor():
    v = verify_lf(1, [_hit()], _model(), geocode_fn=_geo({"Woodstock": (38.885, -78.51)}))
    assert v.status == STATUS_CORROBORATES
    assert "corroborates" in _reasons(v)


def test_advisory_when_place_is_far_from_where_read():
    v = verify_lf(1, [_hit()], _model(), geocode_fn=_geo({"Woodstock": (39.42, -78.50)}))
    assert v.status != STATUS_CONFLICTS
    assert "advisory" in _reasons(v)
    assert "corroborates" not in _reasons(v)


def test_nw_variant_resolves_a_misread_initial():
    h = _hit(text="Noodstock 1 MILE", value="Noodstock")
    v = verify_lf(1, [h], _model(), geocode_fn=_geo({"Woodstock": (38.885, -78.51)}))
    assert v.status == STATUS_CORROBORATES


def test_ambiguous_when_both_variants_are_in_range():
    h = _hit(text="Noodstock 1 MILE", value="Noodstock")
    table = {"Noodstock": (38.89, -78.52), "Woodstock": (38.885, -78.51)}
    v = verify_lf(1, [h], _model(), geocode_fn=_geo(table))
    assert "ambiguous" in _reasons(v)
    assert v.status != STATUS_CORROBORATES


def test_unconfirmed_when_nothing_geocodes():
    v = verify_lf(1, [_hit()], _model(), geocode_fn=_geo({}))
    assert "unconfirmed" in _reasons(v)
    assert v.status == STATUS_NO_SIGNAL


def test_no_gazetteer_is_advisory_only():
    v = verify_lf(1, [_hit()], _model(), geocode_fn=None)
    assert "no gazetteer" in _reasons(v)
    assert v.status == STATUS_NO_SIGNAL


def test_destination_alone_never_conflicts():
    for table in ({}, {"Woodstock": (10.0, 10.0)}, {"Woodstock": (38.885, -78.51)}):
        v = verify_lf(1, [_hit()], _model(), geocode_fn=_geo(table))
        assert v.status != STATUS_CONFLICTS


def test_gpsless_model_still_lists_the_observation():
    v = verify_lf(1, [_hit()], ResolvedLocation(), geocode_fn=None)
    assert v.status == STATUS_NO_SIGNAL
    text = _reasons(v)
    assert "destination sign" in text and "Woodstock" in text and "never a label" in text


def test_destination_max_km_is_a_knob():
    far = {"Woodstock": (39.42, -78.50)}
    v = verify_lf(1, [_hit()], _model(), geocode_fn=_geo(far), destination_max_km=10.0)
    assert "unconfirmed" in _reasons(v)


def test_verify_all_accepts_destination_max_km():
    assert verify_all({}, None, {}, destination_max_km=10.0) == {}


# ── sign_label ──


def test_destination_only_hits_yield_no_county_label():
    label, reasons = explain_sign_label(1, [_hit(), _hit()], gps_fix_count=0)
    none_label, none_reasons = explain_sign_label(1, [], gps_fix_count=0)
    assert label is None and none_label is None
    assert "no county" in " ".join(reasons)
