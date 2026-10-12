"""Flake-demotion midpoint VOTE.

The cross-boundary "A to B" flake demotion validates the on-route midpoint
to decide which endpoint is a geocoder flake. It used to trust a SINGLE
midpoint sample, so one flaky mid-route geocode made the demotion fail open
(kept BOTH endpoints, re-fabricating the bogus route). These tests pin the
fix: the midpoint region is VOTED across several mid-route samples (modal
``(county, state)``) so a single flake can't defeat the demotion. The shared
helper backs both the ``title_builder`` and ``ingest/ost_formatter`` sites.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from gomoveshift.pipeline.title_builder import _vote_mid_region


def _detailed(county, state):
    return {"county": county, "state": state}


class TestVoteMidRegion:
    """``_vote_mid_region`` returns the modal ``(county, state)`` of several
    central samples — a single flake among them does not win."""

    def test_single_flake_does_not_win_the_midpoint(self):
        # Five central samples: four resolve to (Buncombe, NC), one flakes to
        # (Colleton, SC). The honest plurality must win.
        coords = [(35.6, -82.5)] * 4 + [(32.7, -80.5)]

        def side_effect(lat, lon):
            if (round(lat, 1), round(lon, 1)) == (32.7, -80.5):
                return _detailed("Colleton", "SC")
            return _detailed("Buncombe", "NC")

        gc = MagicMock()
        gc.reverse_geocode_detailed.side_effect = lambda lat, lon: side_effect(lat, lon)

        assert _vote_mid_region(gc, coords) == ("Buncombe", "NC")
        # More than one sample must have been consulted — the whole point of
        # the vote is that it is not a single-sample read.
        assert gc.reverse_geocode_detailed.call_count >= 3

    def test_unanimous_region(self):
        coords = [(35.6, -82.5)] * 5
        gc = MagicMock()
        gc.reverse_geocode_detailed.return_value = _detailed("Buncombe", "NC")
        assert _vote_mid_region(gc, coords) == ("Buncombe", "NC")

    def test_no_usable_samples_returns_none(self):
        gc = MagicMock()
        gc.reverse_geocode_detailed.return_value = {}
        assert _vote_mid_region(gc, [(35.6, -82.5)]) == (None, None)

    def test_empty_coords_returns_none(self):
        gc = MagicMock()
        assert _vote_mid_region(gc, []) == (None, None)

    def test_no_geocoder_returns_none(self):
        assert _vote_mid_region(None, [(35.6, -82.5)]) == (None, None)


class TestTiedMidVoteIsInconclusive:
    """An exact top tie is inconclusive (None, None), never alphabet-picked:
    a real crossing near the midpoint splits the samples evenly."""

    @staticmethod
    def _gc():
        gc = MagicMock()

        def side(lat, lon):
            if lat > 39.5:
                return _detailed("Adams", "PA")
            if lat > 39.0:
                return _detailed("Frederick", "MD")
            if lat > 38.0:
                return _detailed("Zed", "MD")
            return {}

        gc.reverse_geocode_detailed.side_effect = side
        return gc

    def test_cross_state_tie_inconclusive_both_orders(self):
        coords = [(39.8, -77.0)] * 2 + [(37.0, -77.0)] + [(39.2, -77.4)] * 2
        assert _vote_mid_region(self._gc(), coords) == (None, None)
        assert _vote_mid_region(self._gc(), list(reversed(coords))) == (None, None)

    def test_same_state_two_county_tie_inconclusive(self):
        coords = [(39.2, -77.4)] * 2 + [(37.0, -77.0)] + [(38.5, -77.4)] * 2
        assert _vote_mid_region(self._gc(), coords) == (None, None)

    def test_validate_route_endpoints_keeps_both_on_tie(self):
        from gomoveshift.pipeline.title_builder import _validate_route_endpoints

        mid = [(39.8, -77.0)] * 2 + [(37.0, -77.0)] + [(39.2, -77.4)] * 2
        assert _validate_route_endpoints(self._gc(), (39.8, -77.0), (39.2, -77.4), mid) == (
            True,
            True,
        )


class TestValidateRouteEndpointsUsesVote:
    """``_validate_route_endpoints`` consumes the VOTED mid region, so a
    single flaky mid sample can no longer fail the demotion open."""

    def test_flaky_mid_sample_outvoted_demotes_end(self):
        from gomoveshift.pipeline.title_builder import _validate_route_endpoints

        # Start = Buncombe/NC (the true area). End = Colleton/SC is the flake.
        # Mid trace: four genuine Buncombe points + one Colleton flake.
        first = (35.60, -82.50)
        last = (32.70, -80.50)  # flake endpoint
        mid_points = [(35.6, -82.5)] * 4 + [(32.7, -80.5)]

        def side_effect(lat, lon):
            key = (round(lat, 1), round(lon, 1))
            if key == (32.7, -80.5):
                return _detailed("Colleton", "SC")
            return _detailed("Buncombe", "NC")

        gc = MagicMock()
        gc.reverse_geocode_detailed.side_effect = lambda lat, lon: side_effect(lat, lon)

        keep_start, keep_end = _validate_route_endpoints(gc, first, last, mid_points)
        # Vote lands the midpoint in Buncombe (start), so the SC end is the
        # flake and must be dropped — NOT fail-open (True, True).
        assert (keep_start, keep_end) == (True, False)
