"""The via-road must be the MODAL road across the winning county's
samples, not the first quality-passing one, and route refs must be display-
normalized.

Root cause: a drive that is ~95% on ``I 81`` through
Smyth County, VA stamped ``via South Main Street`` because ``dominant_location``
took the FIRST winning-county sample whose road passed the road-quality filter.
``South Main Street`` is OSM-classed as a through ``highway`` (it carries
through traffic in Marion), so it passes the filter by class — but a single off-road
sample must not outvote the interstate the drive actually used. The fix votes
the road (modal, deterministic tie-break) and normalizes the ref
(``I 81;US 58`` → ``I-81``).
"""

from gomoveshift.pipeline.title_builder import dominant_location

# Smyth County, VA — the winning county. Points clustered tightly so the
# mock's nearest-point match is unambiguous.
_P1 = (36.84, -81.52)
_P2 = (36.86, -81.48)
_P3 = (36.88, -81.44)
_P4 = (36.90, -81.40)
_P5 = (36.85, -81.50)  # the lone "South Main Street" sample (encountered FIRST)


def _make_geocoder(by_point):
    class _Geo:
        def reverse_geocode_detailed(self, lat, lon):
            best, best_d = None, 0.05
            for (clat, clon), data in by_point.items():
                d = abs(lat - clat) + abs(lon - clon)
                if d < best_d:
                    best_d, best = d, data
            return dict(best) if best else {"county": None, "state": None}

    return _Geo()


def _smyth(road):
    # Both the interstate and the local "Street" are OSM through-class
    # highways, so both clear the road-quality filter — the only discriminator
    # left is the vote.
    return {
        "county": "Smyth County",
        "state": "Virginia",
        "town": "Marion",
        "road": road,
        "place_class": "highway",
        "match_type": "secondary",
    }


def _wythe(road):
    # Wythe County, VA — the winning county. I-81 runs the full length
    # of the window; I-77 only overlaps part of it (Wytheville→Fort Chiswell)
    # before branching north, joined in OSM's ``ref`` as "I 77;I 81;US 52".
    return {
        "county": "Wythe County",
        "state": "Virginia",
        "town": "Wytheville",
        "road": road,
        "place_class": "highway",
        "match_type": "motorway",
    }


class TestRoadModalVote:
    def test_modal_road_beats_first_quality_match(self):
        # _P5 (South Main Street) is listed first; I 81 holds the plurality.
        geo = _make_geocoder(
            {
                _P5: _smyth("South Main Street"),
                _P1: _smyth("I 81"),
                _P2: _smyth("I 81"),
                _P3: _smyth("I 81"),
                _P4: _smyth("I 81"),
            }
        )
        result = dominant_location([_P5, _P1, _P2, _P3, _P4], geo)
        assert result is not None
        assert result.road == "I-81"  # modal + normalized, NOT "South Main Street"

    def test_lone_local_road_still_loses_to_interstate_plurality(self):
        geo = _make_geocoder(
            {_P1: _smyth("I 81"), _P2: _smyth("I 81"), _P5: _smyth("South Main Street")}
        )
        result = dominant_location([_P1, _P2, _P5], geo)
        assert result is not None
        assert result.road == "I-81"

    def test_full_road_vote_tally_persisted_for_provenance(self):
        # Not just the winner — the whole tally is carried on the result so the
        # provenance sidecar can persist what the road vote actually looked like.
        geo = _make_geocoder(
            {_P1: _smyth("I 81"), _P2: _smyth("I 81"), _P5: _smyth("South Main Street")}
        )
        result = dominant_location([_P1, _P2, _P5], geo)
        assert result is not None
        assert result.road_votes == {"I-81": 2, "South Main Street": 1}


class TestConcurrencyRoadVote:
    """A concurrency ``ref`` must credit EVERY carried route, not just
    its primary. A drive (Wytheville→Fort Chiswell→…→Radford) runs eastbound on
    I-81; I-77 shares only the overlap and branches north at Fort Chiswell.
    Collapsing "I 77;I 81;US 52" to "I-77" let the overlap outvote I-81 and
    stamped the title "via I-77" — the wrong through-route."""

    # 8 distinct points in the winning county's window: 3 on the I-81-only
    # stretches, 5 inside the I-77/I-81/US-52 overlap.
    _I81_ONLY = [(37.00, -81.20), (37.02, -81.17), (37.14, -80.99)]
    _OVERLAP = [
        (37.06, -81.11),
        (37.07, -81.09),
        (37.08, -81.08),
        (37.09, -81.06),
        (37.10, -81.05),
    ]

    def test_concurrency_credits_every_member_so_i81_wins(self):
        by_point = {p: _wythe("I 81") for p in self._I81_ONLY}
        by_point.update({p: _wythe("I 77;I 81;US 52") for p in self._OVERLAP})
        result = dominant_location(self._I81_ONLY + self._OVERLAP, _make_geocoder(by_point))
        assert result is not None
        # Pre-fix: overlap collapses to primary "I-77" → 5 > I-81-alone 3 → "I-77".
        # Fixed: I-81 credited on all 8 samples → 8 > I-77's overlap-only 5.
        assert result.road == "I-81"
        # The raw per-sample refs are carried for the ladder's support check.
        assert set(result.road_refs) == {"I 81", "I 77;I 81;US 52"}

    def test_pure_overlap_window_drops_road_rather_than_guess(self):
        # Every sample sits inside the SAME concurrency → every member ties.
        # Prefer NO via-road (safe) over guessing a member (possibly wrong).
        by_point = {p: _wythe("I 77;I 81;US 52") for p in self._OVERLAP}
        result = dominant_location(self._OVERLAP, _make_geocoder(by_point))
        assert result is not None
        assert result.road is None


class TestClassAwareRoadTiePolicy:
    """The via-road tie policy is VOTE-based and class-aware, replacing
    the old SIGNATURE-based pure-concurrency guard.

    (a) A window spanning TWO overlap signatures that share the same
    interstates (US-11 joins the I-77/I-81/US-52 overlap mid-window, as in real
    Wytheville refs) escaped the signature-based guard, and the alphabetical
    tie-break stamped "via I-77" on an I-81 drive — a shipped defect
    class. An interstate-vs-interstate tie must emit NO road.
    (b) An all-"I 81;US 11" window used to drop the road entirely; the lone
    interstate in the tie is unambiguous and must win.
    (c) A tie with no interstate in it stays conservative: NO road, never an
    alphabetical pick."""

    _SIG_A = [(37.06, -81.11), (37.07, -81.09), (37.08, -81.08)]
    _SIG_B = [(37.09, -81.06), (37.10, -81.05)]

    def test_multi_signature_overlap_interstate_tie_emits_no_road(self):
        by_point = {p: _wythe("I 77;I 81;US 52") for p in self._SIG_A}
        by_point.update({p: _wythe("I 77;I 81;US 11;US 52") for p in self._SIG_B})
        result = dominant_location(self._SIG_A + self._SIG_B, _make_geocoder(by_point))
        assert result is not None
        # I-77, I-81 and US-52 each tally 5 votes across BOTH signatures; the
        # old guard saw two distinct signatures, skipped, and the alphabetical
        # tie-break picked "I-77". Two interstates tie → undistinguishable →
        # no via-road.
        assert result.road is None

    def test_lone_interstate_wins_concurrency_tie(self):
        by_point = {p: _wythe("I 81;US 11") for p in self._SIG_A}
        result = dominant_location(self._SIG_A, _make_geocoder(by_point))
        assert result is not None
        # I-81 and US-11 tie at 3 votes each; exactly ONE interstate is in the
        # tie, so it is the unambiguous through-route. The old guard dropped
        # the road entirely here.
        assert result.road == "I-81"

    def test_named_road_tie_without_interstate_emits_no_road(self):
        by_point = {
            self._SIG_A[0]: _wythe("Blue Ridge Parkway"),
            self._SIG_A[1]: _wythe("Skyline Drive"),
        }
        result = dominant_location(self._SIG_A[:2], _make_geocoder(by_point))
        assert result is not None
        # No interstate in the tie → stay conservative, never alphabetical-pick.
        assert result.road is None


def test_normalize_road_ref_retired():
    """``_normalize_road_ref`` is retired — ``_road_ref_members`` is
    the live per-member splitter at the vote site. Guard against reintroduction:
    no importable symbol and zero references anywhere in the shipped package.
    """
    import pathlib

    import gomoveshift
    import gomoveshift.pipeline.title_builder as tb

    assert not hasattr(tb, "_normalize_road_ref")

    pkg_root = pathlib.Path(gomoveshift.__file__).parent
    offenders = [
        str(p)
        for p in pkg_root.rglob("*.py")
        if "_normalize_road_ref" in p.read_text(encoding="utf-8")
    ]
    assert offenders == [], f"retired helper still referenced: {offenders}"
