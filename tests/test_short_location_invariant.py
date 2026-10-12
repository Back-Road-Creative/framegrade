"""Cross-surface location invariant for SHORTS (the shorts analog of
``check_location_invariant``).

Pins ``check_short_location_invariant`` (HARD-fail only when the burned county
is in ZERO of the clip's own sampled regions; no-op on GPS-less/empty clips),
the ``_make_short_regions_fn`` closure, and that the shorts encode path RAISES
upstream of the irreversible 9:16 encode rather than burning a desynced county.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gomoveshift.pipeline.validation.content_gates import (
    _make_short_regions_fn,
    check_short_location_invariant,
)

# Real cached coords (mirrors tests/test_drive_label_dominant.py).
_FRIPP = (32.3095, -80.6332)  # Beaufort County, SC
_EDISTO = (32.6591, -80.6646)  # Colleton County, SC

_BEAUFORT = {"county": "Beaufort County", "state": "South Carolina", "hamlet": "Fort Fremont"}
_COLLETON = {"county": "Colleton County", "state": "South Carolina", "town": "Edisto Beach"}


class _Geo:
    """Mock geocoder keyed by nearest cached point."""

    def reverse_geocode_detailed(self, lat, lon):
        return dict(_BEAUFORT) if abs(lat - _FRIPP[0]) < 0.2 else dict(_COLLETON)


def _clip_pts(latlon, n):
    """Shorts ``(t, lat, lon)`` positional tuples — the trim output shape."""
    return [(float(i), latlon[0], latlon[1]) for i in range(n)]


def _moving_clip_pts(latlon, n, *, step_sec=5.0, step_deg=0.0005):
    """``_clip_pts`` that actually drives (~40 km/h), so a clip clears the
    stop-free gate and reaches the location invariant. Drift stays well
    inside the 0.2 deg the mock geocoder keys on, so the county vote is stable."""
    return [(i * step_sec, latlon[0] + i * step_deg, latlon[1]) for i in range(n)]


# ── check_short_location_invariant ──────────────────────────────────────────


class TestCheckShortLocationInvariant:
    def test_burned_county_outside_clip_regions_hard_fails(self):
        # The far-corner class: a short cut from a Beaufort clip burned the
        # LF-wide Colleton county its own GPS never visited.
        res = check_short_location_invariant(
            "Edisto Beach, Colleton County, South Carolina",
            ["Beaufort County, South Carolina"],
        )
        assert res.gate == "short_location_invariant"
        assert res.passed is False
        assert "HARD FAIL" in res.message

    def test_burned_county_in_clip_regions_passes(self):
        # Burned text names a county the clip actually sampled.
        res = check_short_location_invariant(
            "Fort Fremont, Beaufort County, South Carolina",
            ["Beaufort County, South Carolina", "Jasper County, South Carolina"],
        )
        assert res.passed is True

    def test_town_form_burn_reconciles_on_place_token(self):
        # Town-only burn (no County token) → place-token intersection passes on
        # the genuine TOWN/COUNTY match (Beaufort town inside Beaufort County).
        res = check_short_location_invariant(
            "Beaufort, South Carolina", ["Beaufort County, South Carolina"]
        )
        assert res.passed is True

    def test_town_form_burn_shared_state_only_hard_fails(self):
        # A town-form burn must NOT pass on the shared STATE alone. Beaufort
        # (a town in Beaufort County) vs a Colleton County clip corroborate only
        # on {south, carolina} — a different county, same state. The destination
        # burn was never actually corroborated → HARD FAIL.
        res = check_short_location_invariant(
            "Beaufort, South Carolina", ["Colleton County, South Carolina"]
        )
        assert res.passed is False
        assert "HARD FAIL" in res.message

    def test_town_form_burn_reconciles_on_irish_place_token(self):
        # Irish town-form genuine match: Kilkenny (city) inside County Kilkenny,
        # Leinster → the non-province place token corroborates → passes.
        res = check_short_location_invariant("Kilkenny, Leinster", ["County Kilkenny, Leinster"])
        assert res.passed is True

    def test_town_form_burn_shared_province_only_hard_fails(self):
        # Irish analogue: a town-form burn must NOT pass on the shared
        # PROVINCE alone. Kilkenny vs a County Wexford clip corroborate only on
        # {leinster} — a different county, same province → HARD FAIL.
        res = check_short_location_invariant("Kilkenny, Leinster", ["County Wexford, Leinster"])
        assert res.passed is False
        assert "HARD FAIL" in res.message

    def test_irish_county_form_desync_shares_province_hard_fails(self):
        # Ireland far-corner desync: the short burned the Irish "County Kerry"
        # form but its own GPS votes County Cork — different county, same
        # Munster province. The Irish county label must be discriminated AS a
        # county, not fall through to a province-token overlap that fails OPEN.
        res = check_short_location_invariant("County Kerry, Munster", ["County Cork, Munster"])
        assert res.passed is False
        assert "HARD FAIL" in res.message

    def test_irish_county_form_same_county_passes(self):
        # Same Irish county on both surfaces → no desync.
        res = check_short_location_invariant("County Kerry, Munster", ["County Kerry, Munster"])
        assert res.passed is True

    def test_us_county_form_still_discriminated(self):
        # US "<Name> County" form keeps discriminating a same-state, different
        # county burn (no regression from the Irish-form addition).
        res = check_short_location_invariant(
            "Colleton County, South Carolina", ["Beaufort County, South Carolina"]
        )
        assert res.passed is False

    @pytest.mark.parametrize(
        "burn, regions", [("Anywhere, Nowhere", []), ("", ["Beaufort County, South Carolina"])]
    )  # GPS-less/empty clip or empty burn → skip (the LF gate's legacy no-op)
    def test_noop_on_missing_data(self, burn, regions):
        res = check_short_location_invariant(burn, regions)
        assert res.passed is True
        assert "skipped" in res.message

    def test_clip_derived_town_burn_inside_a_non_eponymous_county_passes(self):
        # The town-form check reconciles on a shared NAME token, so it only ever
        # passed for a town EPONYMOUS with its county (Beaufort/Beaufort County,
        # Kilkenny/County Kilkenny — the only two cases pinned above). Kingsport
        # sits in Sullivan County and shares no token with it, yet both strings
        # come from ONE dominant_location vote over ONE clip's GPS: the OST fn
        # renders the vote's town (gps_enrichment: ``dom.name or dom.county``)
        # and the regions fn renders the SAME vote's county. Checking one against
        # the other is vacuous, exactly as reading ``dominant_county`` back out of
        # ``regions`` would be — so clip-derived text is corroborated by
        # construction and must not hard-fail.
        res = check_short_location_invariant(
            "Kingsport, Tennessee", ["Sullivan County, Tennessee"], clip_derived=True
        )
        assert res.passed is True

    def test_clip_derived_destination_burn_passes(self):
        # Same construction, destination-first branch (destination routing): the burn is a
        # POI resolved from the clip's own points, which shares no token with any
        # county name at all.
        res = check_short_location_invariant(
            "Bays Mountain Park, Tennessee", ["Sullivan County, Tennessee"], clip_derived=True
        )
        assert res.passed is True

    def test_inherited_town_burn_outside_the_clip_county_still_hard_fails(self):
        # The town-form rule must survive the clip-derived carve-out: INHERITED text (parent-LF
        # or drive-wide) is not corroborated by this clip's GPS, so a town in a
        # county the clip never sampled still hard-fails. Default is strict.
        res = check_short_location_invariant(
            "Beaufort, South Carolina", ["Colleton County, South Carolina"]
        )
        assert res.passed is False
        assert "HARD FAIL" in res.message

    def test_clip_derived_flag_never_excuses_a_county_form_desync(self):
        # Defense in depth: even flagged clip-derived, a COUNTY-form burn naming a
        # county outside the sampled set means the two surfaces disagree about the
        # same vote — that is a real contradiction, not a granularity gap.
        res = check_short_location_invariant(
            "Colleton County, South Carolina",
            ["Beaufort County, South Carolina"],
            clip_derived=True,
        )
        assert res.passed is False
        assert "HARD FAIL" in res.message

    def test_gate_has_no_operator_exemption(self):
        # A past regression: the operator LF label
        # "Gettysburg Battlefield, PA" shipped on a GPS-bearing short whose own
        # clip voted Adams County, under an ``operator_override`` pass. An
        # operator label names the whole LF, never one clip, so the exemption is
        # gone — the keyword no longer exists.
        with pytest.raises(TypeError):
            check_short_location_invariant(
                "Gettysburg Battlefield, PA",
                ["Adams County, Pennsylvania"],
                operator_override=True,
            )

    def test_uncorroborated_operator_label_hard_fails(self):
        # The clip's own GPS does not corroborate the label: the short would name
        # a place it was not filmed in, so it never reaches the encode.
        res = check_short_location_invariant(
            "Colleton County, South Carolina",
            ["Beaufort County, South Carolina"],
        )
        assert res.passed is False
        assert "HARD FAIL" in res.message

    def test_corroborated_label_passes_with_the_plain_message(self):
        res = check_short_location_invariant(
            "Beaufort County, South Carolina",
            ["Beaufort County, South Carolina"],
        )
        assert res.passed is True
        assert res.message == "Short location invariant checks passed"

    def test_gettysburg_operator_label_on_an_adams_county_clip_hard_fails(self):
        # The exact shipped shape: a town-form label sharing no token with the
        # clip's own county vote keeps the strict town-form check.
        res = check_short_location_invariant(
            "Gettysburg Battlefield, PA", ["Adams County, Pennsylvania"]
        )
        assert res.passed is False
        assert "HARD FAIL" in res.message


# ── _make_short_regions_fn closure ──────────────────────────────────────────


class TestMakeShortRegionsFn:
    def test_votes_clip_regions_as_county_state_labels(self):
        fn = _make_short_regions_fn(_Geo())
        regions = fn(_clip_pts(_FRIPP, 8), SimpleNamespace(rank=1))
        assert "Beaufort County, South Carolina" in regions

    @pytest.mark.parametrize(
        "geo, pts", [(_Geo(), []), (None, _clip_pts(_FRIPP, 8))]
    )  # empty clip / no geocoder → no regions → gate no-ops
    def test_no_regions_when_underived(self, geo, pts):
        assert _make_short_regions_fn(geo)(pts, SimpleNamespace(rank=1)) == []

    def test_closure_then_gate_catches_desync(self):
        # End-to-end: a Beaufort clip whose burned text says Colleton hard-fails.
        regions = _make_short_regions_fn(_Geo())(_clip_pts(_FRIPP, 8), SimpleNamespace(rank=1))
        res = check_short_location_invariant("Colleton County, South Carolina", regions)
        assert res.passed is False

    def test_single_geocoded_sample_below_quorum_returns_empty(self):
        # The region vote enforces the same GEOCODE_QUORUM_MIN floor as the
        # OST closure — the gate must not "pass" a short on a single-sample
        # region set the OST fn would refuse to label.
        fn = _make_short_regions_fn(_Geo())
        assert fn(_clip_pts(_FRIPP, 1), SimpleNamespace(rank=1)) == []

    def test_derivation_exception_warns_not_debug(self, caplog):
        # An exception degrade makes the downstream gate fail OPEN — that
        # must be LOUD (WARN naming the rank), never DEBUG-only.
        with patch(
            "gomoveshift.pipeline.title_builder.dominant_location",
            side_effect=RuntimeError("geocoder down"),
        ):
            fn = _make_short_regions_fn(_Geo())
            with caplog.at_level(
                logging.WARNING, logger="gomoveshift.pipeline.validation.content_gates"
            ):
                assert fn(_clip_pts(_FRIPP, 8), SimpleNamespace(rank=2)) == []
        assert any(
            "rank 2" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records
        )


# ── a stateless winning county (state is None) ─────────────────────────

_DINGLE = (52.1409, -10.2683)  # County Kerry, Ireland — no state/province


class _StatelessGeo:
    """Winning county with NO state/province (state Python None) — Irish
    counties, US independent cities (Washington DC)."""

    def reverse_geocode_detailed(self, lat, lon):
        return {"county": "County Kerry", "state": None}


class TestStatelessShortRegion:
    def test_none_state_yields_bare_county_region_not_county_none(self):
        # A stateless clip's region set must be the bare county, never the
        # malformed "County Kerry, None" that false-fails the short invariant.
        regions = _make_short_regions_fn(_StatelessGeo())(
            _clip_pts(_DINGLE, 8), SimpleNamespace(rank=1)
        )
        assert regions == ["County Kerry"]
        assert "County Kerry, None" not in regions


# ── encode-path escalation (upstream of the irreversible 9:16 encode) ───────


class TestShortLocationGateRaisesUpstream:
    """The gate fires BEFORE _render_short_ffmpeg in _render_day_shorts."""

    def test_desync_raises_runtime_error_before_encode(self, monkeypatch, tmp_path):
        import gomoveshift.pipeline.day_pipeline as dp

        rendered: list = []

        def _no_render(*args, **kwargs):  # would be the irreversible encode
            rendered.append(kwargs.get("output_path"))
            return True

        # Burned text falls back to the drive-wide Colleton county; the clip
        # GPS votes Beaufort → desync → must raise before any render.
        monkeypatch.setattr(dp, "_render_short_ffmpeg", _no_render)
        monkeypatch.setattr(dp, "trim_gps_to_clip_timeline", lambda pts, ranges: pts)
        monkeypatch.setattr(
            dp, "_resolve_short_ost_text", lambda c, p, r, d, drive_location=None: (c, 0)
        )
        monkeypatch.setattr(dp, "compute_day_file_boundaries", lambda b: [])
        monkeypatch.setattr(dp, "assign_shorts_to_lfs_by_source_window", lambda **k: {})

        short_range = SimpleNamespace(rank=1, start_sec=0.0, end_sec=38.0)
        regions_fn = _make_short_regions_fn(_Geo())

        with pytest.raises(RuntimeError, match="Short location invariant FAILED"):
            dp._render_day_shorts(
                day_dir=tmp_path,
                date="2026-01-01",
                day_batch=SimpleNamespace(episodes=[]),
                short_ranges=[short_range],
                allow_short_count=True,
                gps_points_by_date={"2026-01-01": _clip_pts(_FRIPP, 8)},
                lf_plans=[],
                per_lf_ost_text=None,
                per_lf_locations=None,
                location="Colleton County, South Carolina",
                short_ost_text_fn=None,  # forces fallback to drive-wide location
                master_source_for_extract=Path("/dev/null"),
                short_regions_fn=regions_fn,
            )
        assert rendered == []  # the encode never happened

    def test_clip_derived_non_eponymous_town_renders(self, monkeypatch, tmp_path):
        # End-to-end wiring: a short whose OST text came from its OWN clip vote
        # (a hamlet inside the voted county, sharing no token with the county
        # name) must reach the encode. This is a past production
        # abort — 'Kingsport, Tennessee' against a Sullivan County vote — which
        # killed the run after all four LF burns had already landed.
        import gomoveshift.pipeline.day_pipeline as dp

        rendered: list = []

        def _record_render(*args, **kwargs):
            rendered.append(kwargs.get("output_path"))
            return True

        monkeypatch.setattr(dp, "_render_short_ffmpeg", _record_render)
        monkeypatch.setattr(dp, "trim_gps_to_clip_timeline", lambda pts, ranges: pts)
        monkeypatch.setattr(dp, "compute_day_file_boundaries", lambda b: [])
        monkeypatch.setattr(dp, "assign_shorts_to_lfs_by_source_window", lambda **k: {})
        # The pre-burn OST review gate is a separate contract with its own tests;
        # stub it so this test isolates the location invariant.
        monkeypatch.setattr(dp, "shorts_gate_or_raise", lambda *a, **k: None)

        short_range = SimpleNamespace(rank=1, start_sec=0.0, end_sec=38.0)

        dp._render_day_shorts(
            day_dir=tmp_path,
            date="2026-01-01",
            day_batch=SimpleNamespace(episodes=[]),
            short_ranges=[short_range],
            allow_short_count=True,
            gps_points_by_date={"2026-01-01": _moving_clip_pts(_FRIPP, 8)},
            lf_plans=[],
            per_lf_ost_text=None,
            per_lf_locations=None,
            location="Beaufort County, South Carolina",
            # The clip's own vote renders the hamlet; the region vote over the
            # SAME points renders "Beaufort County, South Carolina".
            short_ost_text_fn=lambda pts, rng: "Fort Fremont, South Carolina",
            master_source_for_extract=Path("/dev/null"),
            short_regions_fn=_make_short_regions_fn(_Geo()),
        )
        assert rendered, "clip-derived town burn must reach the encode, not hard-fail"

    def test_operator_per_lf_override_is_never_burned_on_a_gps_bearing_short(
        self, monkeypatch, tmp_path, caplog
    ):
        # A past regression: the operator
        # ``--per-lf-location "1=Gettysburg Battlefield, PA"`` was burned on a
        # GPS-bearing short — the clip vote was suppressed and the region gate
        # exempted it — so the short named a place it was not filmed in. The
        # override labels the LF only: this short burns its own clip vote.
        import gomoveshift.pipeline.day_pipeline as dp

        burned: list = []

        def _record_render(*args, **kwargs):
            burned.append(kwargs.get("location"))
            return True

        monkeypatch.setattr(dp, "_render_short_ffmpeg", _record_render)
        monkeypatch.setattr(dp, "trim_gps_to_clip_timeline", lambda pts, ranges: pts)
        monkeypatch.setattr(dp, "compute_day_file_boundaries", lambda b: [])
        # One LF, one short → the short's parent LF is index 1.
        monkeypatch.setattr(dp, "assign_shorts_to_lfs_by_source_window", lambda **k: {1: [0]})
        monkeypatch.setattr(dp, "shorts_gate_or_raise", lambda *a, **k: None)

        short_range = SimpleNamespace(rank=1, start_sec=0.0, end_sec=38.0)

        with caplog.at_level(logging.WARNING, logger="gomoveshift.pipeline.day_pipeline"):
            dp._render_day_shorts(
                day_dir=tmp_path,
                date="2026-03-05",
                day_batch=SimpleNamespace(episodes=[]),
                short_ranges=[short_range],
                allow_short_count=True,
                gps_points_by_date={"2026-03-05": _moving_clip_pts(_FRIPP, 8)},
                lf_plans=[SimpleNamespace(index=1)],
                per_lf_ost_text={1: "Gettysburg Battlefield, PA"},
                per_lf_locations={1: "Adams County, Pennsylvania"},
                per_lf_overrides={1: "Gettysburg Battlefield, PA"},
                location="Adams County, Pennsylvania",
                # The clip's own vote — consulted even under the override.
                short_ost_text_fn=lambda pts, rng: "Fort Fremont, South Carolina",
                master_source_for_extract=Path("/dev/null"),
                # The mock geocoder votes Beaufort County for these points: a
                # county-only region set that shares no token with the label.
                short_regions_fn=_make_short_regions_fn(_Geo()),
            )
        assert len(burned) == 1
        assert "Fort Fremont" in burned[0]
        assert "Gettysburg" not in burned[0]
        assert not any("UNCORROBORATED" in r.getMessage() for r in caplog.records)
