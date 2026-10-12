"""``state["per_lf_resolved_location"]`` population wiring.

``derive_per_lf_locations`` and the title/OST surfaces re-derive location
independently; ``_populate_resolved_locations`` wires the canonical
``ResolvedLocation`` builder into Phase BC: every LF with a usable GPS trace
gets one resolved model persisted into
``state.per_lf_resolved_location[lf_index]``.

Tested against the pure helper ``_populate_resolved_locations`` so the
wiring is verifiable without a full 6-hour-encode phase run. Back-compat:
LFs with no usable trace are skipped (not crashed); a None state field is
initialised lazily.
"""

from types import SimpleNamespace

import pytest

from gomoveshift.models import ResolvedLocation
from gomoveshift.pipeline import gps_enrichment
from gomoveshift.pipeline.drive_phases import phase_bc_day_first
from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
    _populate_resolved_locations,
)

_FRIPP = (32.3095, -80.6332)  # Beaufort County, SC
_EDISTO = (32.6591, -80.6646)  # Colleton County, SC
_BEAUFORT = {
    "county": "Beaufort County",
    "state": "South Carolina",
    "town": "Port Royal",
    "road": "Sea Island Parkway",
}
_COLLETON = {"county": "Colleton County", "state": "South Carolina", "town": "Edisto Beach"}


def _geo():
    class _Geo:
        def reverse_geocode_detailed(self, lat, lon):
            df = abs(lat - _FRIPP[0]) + abs(lon - _FRIPP[1])
            de = abs(lat - _EDISTO[0]) + abs(lon - _EDISTO[1])
            return dict(_BEAUFORT if df <= de else _COLLETON)

    return _Geo()


def _lf(index):
    return SimpleNamespace(index=index)


def _pts(point, n):
    return [{"lat": point[0], "lon": point[1]} for _ in range(n)]


class TestPopulateResolvedLocations:
    def test_populates_one_resolved_per_lf_with_trace(self):
        state = SimpleNamespace(per_lf_resolved_location=None)
        gps = {1: _pts(_FRIPP, 8) + _pts(_EDISTO, 2)}
        _populate_resolved_locations(state, [_lf(1)], gps, _geo())
        rl = state.per_lf_resolved_location[1]
        assert isinstance(rl, ResolvedLocation)
        assert (rl.dominant_county, rl.dominant_state) == (
            "Beaufort County",
            "South Carolina",
        )
        assert rl.is_us is True and rl.country_code == "US"
        # Location-invariant substrate carried through.
        assert "Beaufort County, South Carolina" in rl.regions
        assert rl.vote["Beaufort County, South Carolina"] > 0
        # Road comes from the builder, no second geocode pass.
        assert rl.road == "Sea Island Parkway"

    def test_no_redundant_dominant_location_call(self, monkeypatch):
        """The builder already exposes ``rl.road`` from the winning-county
        vote, so populate must NOT re-call ``dominant_location`` to recover it.
        One call per LF (inside the builder), never two."""
        calls = {"n": 0}
        real = gps_enrichment.dominant_location

        def _counting(points, geocoder):
            calls["n"] += 1
            return real(points, geocoder)

        monkeypatch.setattr(gps_enrichment, "dominant_location", _counting)
        # If a redundant call survives in phase_bc, it would route through that
        # module's own binding — spy it too (raising=False: phase_bc no longer
        # binds dominant_location, so its absence is itself the proof).
        monkeypatch.setattr(phase_bc_day_first, "dominant_location", _counting, raising=False)

        state = SimpleNamespace(per_lf_resolved_location=None)
        gps = {1: _pts(_FRIPP, 8) + _pts(_EDISTO, 2)}
        _populate_resolved_locations(state, [_lf(1)], gps, _geo())
        assert calls["n"] == 1  # one LF → exactly one dominant_location call

    def test_lf_without_trace_is_skipped_not_crashed(self):
        state = SimpleNamespace(per_lf_resolved_location=None)
        _populate_resolved_locations(state, [_lf(1), _lf(2)], {1: _pts(_FRIPP, 6)}, _geo())
        assert 1 in state.per_lf_resolved_location
        assert 2 not in state.per_lf_resolved_location

    def test_initialises_none_field_lazily(self):
        state = SimpleNamespace(per_lf_resolved_location=None)
        _populate_resolved_locations(state, [], {}, _geo())
        assert state.per_lf_resolved_location == {}

    def test_preserves_existing_entries_backcompat(self):
        """On resume an OLD entry may already be present — recompute fills the
        rest without clobbering."""
        existing = ResolvedLocation(dominant_county="Pre-existing County")
        state = SimpleNamespace(per_lf_resolved_location={0: existing})
        _populate_resolved_locations(state, [_lf(1)], {1: _pts(_EDISTO, 6)}, _geo())
        assert state.per_lf_resolved_location[0] is existing
        assert state.per_lf_resolved_location[1].dominant_county == "Colleton County"

    def test_no_geocoder_is_noop_safe(self):
        state = SimpleNamespace(per_lf_resolved_location=None)
        _populate_resolved_locations(state, [_lf(1)], {1: _pts(_FRIPP, 6)}, None)
        # No geocoder → builder returns an empty ResolvedLocation; we skip
        # storing empties so resume can recompute later.
        assert 1 not in state.per_lf_resolved_location

    def test_quorum_degraded_model_not_stored(self):
        """A quorum-miss trace (one flaky geocode on a long blank
        trace) must NOT persist a model; a degraded model (regions=
        [flake county], 1-sample road) would feed the title fallback and
        location_invariant gates as truth."""

        class _MissGeo:  # geocodes ONLY near Fripp; everything else misses
            def reverse_geocode_detailed(self, lat, lon):
                d = abs(lat - _FRIPP[0]) + abs(lon - _FRIPP[1])
                return dict(_BEAUFORT) if d < 0.15 else {"county": None, "state": None}

        state = SimpleNamespace(per_lf_resolved_location=None)
        pts = [{"lat": 32.10, "lon": -80.40} for _ in range(45)]  # geocode misses
        pts[22] = {"lat": _FRIPP[0], "lon": _FRIPP[1]}  # the single flake
        _populate_resolved_locations(state, [_lf(1)], {1: pts}, _MissGeo())
        assert 1 not in state.per_lf_resolved_location


class TestDerivationFailureHardFail:
    """A ``build_resolved_location`` FAILURE (an exception with GPS
    PRESENT) is NOT the same terminal state as a GPS-less LF. Warning+continue
    left the LF model-less, which silently SKIPS the pre-encode location_invariant far-road
    gate exactly when the geocode/GPX layer is most likely to have produced a
    WRONG label. A genuine derivation failure must HARD FAIL — unless the
    operator supplied an authoritative ``--per-lf-location`` override (which
    bypasses derivation for that LF)."""

    def _boom(self, points, *, geocoder):
        raise RuntimeError("geocoder exploded mid-derivation")

    def test_derivation_failure_with_gps_hard_fails(self, monkeypatch):
        """(a) GPS present + ``build_resolved_location`` raises → RAISE, not
        warn+continue+skip-gate."""
        monkeypatch.setattr(phase_bc_day_first, "build_resolved_location", self._boom)
        state = SimpleNamespace(per_lf_resolved_location=None, per_lf_overrides=None)
        gps = {1: _pts(_FRIPP, 8)}
        with pytest.raises(RuntimeError, match="build_resolved_location"):
            _populate_resolved_locations(state, [_lf(1)], gps, _geo())

    def test_gps_less_lf_still_no_ops_quietly(self):
        """(b) a genuinely GPS-less LF (no trace) stays a silent no-op — no
        regression of the documented legitimate skip. Never raises."""
        state = SimpleNamespace(per_lf_resolved_location=None, per_lf_overrides=None)
        # LF 2 has no GPS points → GPS-less → skipped; LF 1 resolves normally.
        _populate_resolved_locations(state, [_lf(1), _lf(2)], {1: _pts(_FRIPP, 6)}, _geo())
        assert 1 in state.per_lf_resolved_location
        assert 2 not in state.per_lf_resolved_location

    def test_operator_override_suppresses_hard_fail(self, monkeypatch):
        """The documented escape: an authoritative ``--per-lf-location``
        override means the label is supplied verbatim, so the model isn't
        needed — a derivation failure for THAT LF degrades quietly, no raise."""
        monkeypatch.setattr(phase_bc_day_first, "build_resolved_location", self._boom)
        state = SimpleNamespace(per_lf_resolved_location=None, per_lf_overrides={1: "Beaufort, SC"})
        _populate_resolved_locations(state, [_lf(1)], {1: _pts(_FRIPP, 8)}, _geo())
        assert 1 not in state.per_lf_resolved_location


class TestResumeSignatureInvalidation:
    """The persisted resolved model is stamped with the OST/title text it
    was derived against; on resume ``_populate_resolved_locations`` re-derives
    (never reuses) an entry whose signature no longer matches the freshly-
    recomputed ``per_lf_ost_text``, so the shipped title and the model feeding
    units/playlist/tags + the content gates can never diverge silently."""

    def test_signature_drift_re_derives_stale_model(self):
        """Seed a model whose stamped signature differs from THIS run's fresh OST
        text → the stale model is dropped and re-derived from the live trace."""
        stale = ResolvedLocation(dominant_county="Stale County", ost_signature="OLD TITLE")
        state = SimpleNamespace(per_lf_resolved_location={1: stale})
        gps = {1: _pts(_EDISTO, 6)}
        _populate_resolved_locations(
            state, [_lf(1)], gps, _geo(), per_lf_ost_text={1: "Edisto to Fripp"}
        )
        rl = state.per_lf_resolved_location[1]
        # Re-derived from the live GPS trace, not the seeded stale model.
        assert rl.dominant_county == "Colleton County"
        # Re-stamped so a subsequent matching-signature resume fast-paths.
        assert rl.ost_signature == "Edisto to Fripp"

    def test_legacy_unversioned_model_re_derives(self):
        """A legacy entry carries no signature (``None``) — an unversioned model
        can't be trusted, so it re-derives once (mirrors the color cache's
        invalidate-v1-on-read)."""
        legacy = ResolvedLocation(dominant_county="Stale County")  # ost_signature=None
        state = SimpleNamespace(per_lf_resolved_location={1: legacy})
        _populate_resolved_locations(
            state, [_lf(1)], {1: _pts(_EDISTO, 6)}, _geo(), per_lf_ost_text={1: "Edisto"}
        )
        assert state.per_lf_resolved_location[1].dominant_county == "Colleton County"
        assert state.per_lf_resolved_location[1].ost_signature == "Edisto"

    def test_matching_signature_fast_path_skips_re_derive(self):
        """A resume where the fresh OST text matches the stamped signature keeps
        the persisted model untouched — no needless recompute."""
        pinned = ResolvedLocation(dominant_county="Sentinel County", ost_signature="Same Title")
        state = SimpleNamespace(per_lf_resolved_location={1: pinned})
        _populate_resolved_locations(
            state, [_lf(1)], {1: _pts(_EDISTO, 6)}, _geo(), per_lf_ost_text={1: "Same Title"}
        )
        # Same object, untouched — the live trace would have voted Colleton.
        assert state.per_lf_resolved_location[1] is pinned
        assert state.per_lf_resolved_location[1].dominant_county == "Sentinel County"

    def test_fresh_run_stamps_signature(self):
        """A first (non-resume) run populates AND stamps the OST signature so the
        next resume can compare against it."""
        state = SimpleNamespace(per_lf_resolved_location=None)
        _populate_resolved_locations(
            state, [_lf(1)], {1: _pts(_FRIPP, 8)}, _geo(), per_lf_ost_text={1: "Fripp Island"}
        )
        rl = state.per_lf_resolved_location[1]
        assert rl.dominant_county == "Beaufort County"
        assert rl.ost_signature == "Fripp Island"

    def test_missing_fresh_ost_text_preserves_entry(self):
        """When THIS run has no fresh OST text for the LF (defensive/GPS-less
        path) there's nothing to compare against — the persisted entry is
        preserved rather than churned."""
        pinned = ResolvedLocation(dominant_county="Sentinel County", ost_signature="Whatever")
        state = SimpleNamespace(per_lf_resolved_location={1: pinned})
        _populate_resolved_locations(
            state, [_lf(1)], {1: _pts(_EDISTO, 6)}, _geo(), per_lf_ost_text={}
        )
        assert state.per_lf_resolved_location[1] is pinned


class TestLegacyDegradedModelBackcompat:
    """Older state files can carry a
    quorum-degraded model (``dominant_* None`` but regions/vote/road/sequence
    kept — the only shape the ``or rl.regions`` storage branch ever stored
    without a dominant county). Resume must DROP it and re-derive, and plain
    dict rehydration of legacy models must keep working unchanged."""

    def _legacy_degraded(self, **kw):
        return ResolvedLocation(
            regions=["Beaufort County, South Carolina"],
            vote={"Beaufort County, South Carolina": 1.0},
            road="Sea Island Parkway",
            ordered_place_sequence=["St. Helena"],
            **kw,
        )

    def test_legacy_persisted_dict_rehydrates(self):
        """Rehydration contract: ``_deserialize_dataclass`` builds the model via
        ``cls(**value)`` — a legacy persisted dict (same fields as the current
        model) must round-trip; anything absent defaults per the dataclass."""
        from dataclasses import asdict

        from gomoveshift.pipeline.drive_state_model import _deserialize_dataclass

        legacy = asdict(self._legacy_degraded())
        rl = _deserialize_dataclass(legacy, ResolvedLocation)
        assert isinstance(rl, ResolvedLocation)
        assert rl.road == "Sea Island Parkway"
        # A partial (older-still) dict also rehydrates — missing keys default.
        partial = {"dominant_county": "Colleton County"}
        rl2 = _deserialize_dataclass(partial, ResolvedLocation)
        assert rl2.dominant_county == "Colleton County" and rl2.regions == []

    def test_legacy_degraded_entry_dropped_and_rederived_on_resume(self):
        """A rehydrated quorum-degraded legacy entry must not survive the resume
        fast-path: its sub-floor road/regions would keep feeding the title
        fallback and location_invariant gates. It is dropped and re-derived from the live
        trace (here quorum-met → a real dominant county)."""
        legacy = self._legacy_degraded()  # ost_signature=None
        state = SimpleNamespace(per_lf_resolved_location={1: legacy})
        # per_lf_ost_text omitted → signature None, the case where the fast path would otherwise keep it.
        _populate_resolved_locations(state, [_lf(1)], {1: _pts(_EDISTO, 6)}, _geo())
        rl = state.per_lf_resolved_location[1]
        assert rl is not legacy
        assert rl.dominant_county == "Colleton County"

    def test_legacy_degraded_entry_dropped_even_on_matching_signature(self):
        """Signature match is the fast path — but a degraded model is never
        trustworthy regardless of signature; it must still be dropped."""
        legacy = self._legacy_degraded(ost_signature="Same Title")
        state = SimpleNamespace(per_lf_resolved_location={1: legacy})
        _populate_resolved_locations(
            state, [_lf(1)], {1: _pts(_EDISTO, 6)}, _geo(), per_lf_ost_text={1: "Same Title"}
        )
        rl = state.per_lf_resolved_location[1]
        assert rl is not legacy
        assert rl.dominant_county == "Colleton County"


class TestPopulateDestination:
    """The precomputed per-LF destination POI is anchored onto the
    persisted ResolvedLocation (computed once in ``_enrich_per_lf_gps``,
    reused here — no second Overpass query)."""

    def test_destination_anchored_from_map(self):
        state = SimpleNamespace(per_lf_resolved_location=None)
        gps = {1: _pts(_FRIPP, 8)}
        _populate_resolved_locations(
            state,
            [_lf(1)],
            gps,
            _geo(),
            {1: "Donnelley Wildlife Management Area"},
        )
        rl = state.per_lf_resolved_location[1]
        assert rl.destination == "Donnelley Wildlife Management Area"
        # The county vote is untouched — the location_invariant gate still reads it, not the title.
        assert rl.dominant_county == "Beaufort County"

    def test_no_map_entry_leaves_destination_none(self):
        state = SimpleNamespace(per_lf_resolved_location=None)
        _populate_resolved_locations(state, [_lf(1)], {1: _pts(_FRIPP, 6)}, _geo(), {})
        assert state.per_lf_resolved_location[1].destination is None

    def test_default_destination_arg_is_backcompat(self):
        """The 4-arg call (legacy callers / older tests) still works."""
        state = SimpleNamespace(per_lf_resolved_location=None)
        _populate_resolved_locations(state, [_lf(1)], {1: _pts(_FRIPP, 6)}, _geo())
        assert state.per_lf_resolved_location[1].destination is None


class TestResolveLfDestination:
    """``_resolve_lf_destination`` — the offline-safe per-LF resolver wrapper."""

    def test_no_montage_peaks_returns_none(self):
        assert gps_enrichment._resolve_lf_destination(SimpleNamespace(index=1)) is None
        assert (
            gps_enrichment._resolve_lf_destination(SimpleNamespace(index=1, montage_peaks=[]))
            is None
        )

    def test_resolver_failure_degrades_to_none(self, monkeypatch):
        """A resolver exception is swallowed — the destination never raises into
        the pipeline."""
        import gomoveshift.pipeline.destination_resolver as dr

        def _boom(_peaks, **_kw):
            raise RuntimeError("overpass down")

        monkeypatch.setattr(dr, "resolve_destination", _boom)
        peak = SimpleNamespace(gps_center_lat=32.6, gps_center_lon=-80.5, composite_score=0.9)
        lf = SimpleNamespace(index=3, montage_peaks=[peak])
        assert gps_enrichment._resolve_lf_destination(lf) is None
