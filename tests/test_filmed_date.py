"""Resume filmed-date anchoring and the shared date formatter.

Anchoring: the resume recording-date derivation must read the ``day_<YYYY-MM-DD>``
    dir that is the IMMEDIATE parent of ``regular_lfs`` and key the result by
    the LF's drive-global index — a relocated/nested path whose ancestors
    carry a different ``day_*`` token must not skew the date.

Formatter: one SHARED ``format_recording_date`` across long-form descriptions and
    child shorts (canonical ``📅 Filmed: <date>`` line, ``Mon YYYY`` shape);
    an unparseable value is REJECTED (dropped), not passed through.
"""

from pathlib import Path
from types import SimpleNamespace

from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
    _day_date_from_lf_path,
    _derive_per_lf_recording_date_from_paths,
)
from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator
from gomoveshift.pipeline.social_generator import format_recording_date


def _lf(index):
    return SimpleNamespace(index=index)


# ── anchored day_<DATE> extraction ─────────────────────────────


class TestDayDateAnchoring:
    def test_immediate_regular_lfs_parent_is_the_anchor(self):
        p = Path("/stages/proj/day_2025-09-15/regular_lfs/lf_00.mp4")
        assert _day_date_from_lf_path(p) == "2025-09-15"

    def test_ancestor_day_token_does_not_skew(self):
        # A relocated path nested under a DIFFERENT day_* ancestor must
        # still resolve to the day dir anchoring regular_lfs, NOT the
        # first day_* token in the string (the old `.search` bug).
        p = Path("/archive/day_2024-01-01/restored/day_2025-09-15/regular_lfs/lf_03.mp4")
        assert _day_date_from_lf_path(p) == "2025-09-15"

    def test_no_anchored_match_returns_none(self):
        assert _day_date_from_lf_path(Path("/stages/proj/regular_lfs/lf_00.mp4")) is None

    def test_ambiguous_multiple_day_dirs_without_regular_lfs_returns_none(self):
        p = Path("/a/day_2024-01-01/b/day_2025-09-15/lf_00.mp4")
        assert _day_date_from_lf_path(p) is None


class TestDeriveKeyedByIndex:
    def test_keyed_by_lf_index_not_position(self):
        # Drive-global indices are non-contiguous; the dict must key by
        # the skeleton LF's own index, each date read from its own path.
        skeleton = [_lf(5), _lf(9)]
        paths = [
            Path("/s/day_2025-09-15/regular_lfs/lf_00.mp4"),
            Path("/s/day_2025-09-16/regular_lfs/lf_01.mp4"),
        ]
        out = _derive_per_lf_recording_date_from_paths(paths, skeleton)
        assert out == {5: "2025-09-15", 9: "2025-09-16"}

    def test_relocated_path_does_not_skew_date(self):
        skeleton = [_lf(0)]
        paths = [Path("/old/day_2020-01-01/x/day_2025-09-15/regular_lfs/lf_00.mp4")]
        out = _derive_per_lf_recording_date_from_paths(paths, skeleton)
        assert out == {0: "2025-09-15"}

    def test_unparseable_path_skipped(self):
        skeleton = [_lf(0), _lf(1)]
        paths = [
            Path("/s/day_2025-09-15/regular_lfs/lf_00.mp4"),
            Path("/s/no_day_here/regular_lfs/lf_01.mp4"),
        ]
        out = _derive_per_lf_recording_date_from_paths(paths, skeleton)
        assert out == {0: "2025-09-15"}


# ── shared formatter + reject-not-passthrough ──────────────────


class TestSharedFormatter:
    def test_iso_to_mon_year(self):
        assert format_recording_date("2025-09-15") == "Sep 2025"

    def test_full_date_to_mon_year(self):
        assert format_recording_date("September 10, 2025") == "Sep 2025"

    def test_passthrough_mon_year(self):
        assert format_recording_date("Sep 2025") == "Sep 2025"

    def test_empty_and_none(self):
        assert format_recording_date("") == ""
        assert format_recording_date(None) == ""

    def test_unparseable_is_rejected_not_passed_through(self):
        # Old behavior returned the garbage string as-is; now dropped.
        assert format_recording_date("not a date at all") == ""
        assert format_recording_date("13/40/9999") == ""


class TestLongFormUsesSharedFormatter:
    def _gen(self):
        g = LongFormMetadataGenerator.__new__(LongFormMetadataGenerator)
        g.storage_path = Path("/tmp/test-seo")
        g.credentials_dir = Path("/tmp/test-creds")
        g._manager = None
        g._optimizer = None
        g._ledger = None
        return g

    def test_iso_recording_date_rendered_as_mon_year(self):
        desc = self._gen()._build_gms_description(
            location="County Kerry, Ireland",
            seo_description="A beautiful drive through Kerry!",
            duration_str="1h 23m",
            publish_day="Saturday",
            publish_date="2026-02-28",
            recording_date="2025-09-15",
        )
        assert "📅 Filmed: Sep 2025" in desc
        assert "2025-09-15" not in desc

    def test_unparseable_recording_date_drops_the_line(self):
        desc = self._gen()._build_gms_description(
            location="County Kerry, Ireland",
            seo_description="A beautiful drive through Kerry!",
            duration_str="1h 23m",
            publish_day="Saturday",
            publish_date="2026-02-28",
            recording_date="garbage-date",
        )
        assert "📅 Filmed:" not in desc
