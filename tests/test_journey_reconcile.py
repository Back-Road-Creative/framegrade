"""Tests for the channel→journey reconciler (sync-site --journeys)."""

from __future__ import annotations

import io
import textwrap
from contextlib import contextmanager

import pytest

from gomoveshift.pipeline.journal_publisher import load_journey
from gomoveshift.pipeline.orchestrator_lock import _acquire_run_lock, _release_run_lock
from gomoveshift.pipeline.site_sync.journey_reconcile import (
    JourneyRegionMismatchError,
    assert_region_consistency,
    load_routing,
    merge_routing,
    reconcile_journeys,
    routing_from_state,
)

_JOURNEY_MD = textwrap.dedent(
    """\
    ---
    title: "Test Journey"
    days:
    - day_number: 1
      date: '2026-03-04'
      videos:
      - youtube_id: existing1
        title: Already Here
        duration: '10:00'
    video_count: 1
    ---
    Body text.
    """
)

_MANIFEST = {
    "videos": [
        {"id": "lf_new", "title": "New LF", "duration_seconds": 780, "category": "real_lf"},
        {"id": "existing1", "title": "X", "duration_seconds": 600, "category": "real_lf"},
        {"id": "held1", "title": "Held One", "duration_seconds": 700, "category": "held"},
        {"id": "badday", "title": "Bad Day", "duration_seconds": 700, "category": "real_lf"},
    ]
}

# A public LF (lf_new), an already-embedded id (existing1), a held id, a route
# whose day doesn't exist (badday), and a malformed row (no_date) that
# load_routing must silently drop.
_ROUTES = textwrap.dedent(
    """\
    routes:
      lf_new: {journey: test-journey, date: '2026-03-04'}
      existing1: {journey: test-journey, date: '2026-03-04'}
      held1: {journey: test-journey, date: '2026-03-04'}
      badday: {journey: test-journey, date: '2099-01-01'}
      no_date: {journey: test-journey}
    """
)


def _make_site(tmp_path, routes=_ROUTES, journey_md=_JOURNEY_MD):
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "content" / "journeys").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "video_routing.yaml").write_text(routes)
    (tmp_path / "content" / "journeys" / "test-journey.md").write_text(journey_md)
    return tmp_path


@contextmanager
def _another_writer_holds(journey):
    """Hold this journey's lock the way another writer does, through the shared
    machinery directly — the test must not prove the lock with the very call it
    is exercising."""
    fd, lock_path = _acquire_run_lock(
        journey.parent, journey.parent, lock_name=journey.name + ".lock"
    )
    try:
        yield
    finally:
        _release_run_lock(fd, lock_path)


def test_reconcile_skips_and_warns_while_another_writer_holds_the_journey(tmp_path):
    """Another writer's load -> mutate -> write window is open, so any doc this
    reconcile loads is already stale: writing it back is a last-writer-wins
    clobber. Warn loudly and skip — never raise; the next sync re-embeds."""
    site = _make_site(tmp_path)
    journey = site / "content" / "journeys" / "test-journey.md"
    before = journey.read_text()

    with _another_writer_holds(journey):
        rec = reconcile_journeys(_MANIFEST, site, apply=True)

    assert journey.read_text() == before, "a held journey must not be clobbered"
    assert rec.embedded == []
    assert any("lock" in w for w in rec.warnings), rec.warnings


def test_reconcile_embeds_routes_dedups_defers_and_warns(tmp_path):
    site = _make_site(tmp_path)
    rec = reconcile_journeys(_MANIFEST, site, apply=True)

    assert rec.embedded == [("test-journey", "lf_new")]  # only the new public LF
    assert rec.skipped_present == [("test-journey", "existing1")]  # dedup by id
    assert rec.deferred_held == ["held1"]  # routed but not a public real_lf (no_date dropped)
    assert len(rec.warnings) == 1 and "badday" in rec.warnings[0]  # no matching day

    doc, _ = load_journey(site / "content" / "journeys" / "test-journey.md")
    day = doc["days"][0]
    assert [v["youtube_id"] for v in day["videos"]] == ["existing1", "lf_new"]  # appended once
    new = next(v for v in day["videos"] if v["youtube_id"] == "lf_new")
    assert new["title"] == "New LF" and str(new["duration"]) == "13:00"  # format_duration(780)
    assert doc["video_count"] == 2  # maintained by apply_merge


def test_reconcile_dry_run_writes_nothing(tmp_path):
    site = _make_site(tmp_path)
    md = site / "content" / "journeys" / "test-journey.md"
    before = md.read_text()
    rec = reconcile_journeys(_MANIFEST, site, apply=False)
    assert ("test-journey", "lf_new") in rec.embedded  # planned
    assert md.read_text() == before  # but unwritten


def test_reconcile_missing_routing_is_noop(tmp_path):
    (tmp_path / "data").mkdir()
    rec = reconcile_journeys(_MANIFEST, tmp_path, apply=True)
    assert rec.embedded == [] and rec.warnings == []


def test_reconcile_missing_journey_warns_not_raises(tmp_path):
    site = _make_site(tmp_path, routes="routes:\n  lf_new: {journey: ghost, date: '2026-03-04'}\n")
    rec = reconcile_journeys(_MANIFEST, site, apply=True)
    assert rec.embedded == []
    assert len(rec.warnings) == 1 and "ghost" in rec.warnings[0]


# --- Atomic journey write ---------------------------------------------------

_COMMENTED_JOURNEY_MD = _JOURNEY_MD.replace(
    "days:", "# hand-authored note the round-trip must keep\ndays:"
)


def _crash_on_file_writes(monkeypatch):
    """Kill every write-mode file handle on its first ``write`` call.

    Faithful to a real interruption: a bare ``Path.write_text`` truncates the
    target the moment it opens it, so dying before the bytes land leaves an
    empty page on the live site. An atomic write stages the bytes elsewhere
    and only swaps them in at the end, so the published page is untouched.
    """
    real_open = io.open

    def guarded_open(file, mode="r", *args, **kwargs):
        handle = real_open(file, mode, *args, **kwargs)
        if "w" in mode:

            def boom(*_a, **_k):
                raise RuntimeError("simulated crash mid-write")

            handle.write = boom
        return handle

    monkeypatch.setattr(io, "open", guarded_open)


def test_reconcile_journey_write_survives_crash_mid_write(tmp_path, monkeypatch):
    """A torn write must never truncate a published journey page.

    A missing merge is replayable; a truncated page is a corrupt artifact that
    blocks the site deploy (which hard-fails on a ``video_count`` mismatch).
    """
    site = _make_site(tmp_path, journey_md=_COMMENTED_JOURNEY_MD)
    md = site / "content" / "journeys" / "test-journey.md"
    before = md.read_text()

    _crash_on_file_writes(monkeypatch)
    with pytest.raises(RuntimeError, match="simulated crash mid-write"):
        reconcile_journeys(_MANIFEST, site, apply=True)
    monkeypatch.undo()

    assert md.read_text() == before  # not truncated, not emptied
    assert [p.name for p in (site / "content" / "journeys").iterdir()] == ["test-journey.md"]


def test_reconcile_journey_write_lands_full_content_and_keeps_comments(tmp_path):
    """The successful path still writes everything and round-trips comments."""
    site = _make_site(tmp_path, journey_md=_COMMENTED_JOURNEY_MD)
    md = site / "content" / "journeys" / "test-journey.md"
    rec = reconcile_journeys(_MANIFEST, site, apply=True)

    assert rec.embedded == [("test-journey", "lf_new")]
    text = md.read_text()
    assert "# hand-authored note the round-trip must keep" in text
    assert "lf_new" in text and "existing1" in text
    assert text.startswith("---\n") and text.endswith("Body text.\n")
    assert [p.name for p in (site / "content" / "journeys").iterdir()] == ["test-journey.md"]


# --- drive auto-route append ------------------------------------------

_STATE = {
    "youtube_uploads": {
        "1": {"video_id": "vid1"},
        "2": {"video_id": "vid2"},
        "3": {"video_id": "vid3"},  # no recording date -> skipped
        "4": {"title": "missing id"},  # no video_id -> skipped
    },
    "per_lf_recording_date": {"1": "2026-03-04", "2": "2026-03-05", "9": "2026-03-09"},
}


def test_routing_from_state_pairs_id_and_date():
    routes = routing_from_state(_STATE, "trip")
    assert routes == {
        "vid1": {"journey": "trip", "date": "2026-03-04"},
        "vid2": {"journey": "trip", "date": "2026-03-05"},
    }


def test_routing_from_state_handles_missing_or_malformed_state():
    assert routing_from_state({}, "trip") == {}
    assert routing_from_state({"youtube_uploads": []}, "trip") == {}


def test_merge_routing_upserts_and_preserves(tmp_path):
    p = tmp_path / "video_routing.yaml"
    merge_routing(p, {"vidA": {"journey": "j1", "date": "2026-01-01"}})
    merge_routing(p, {"vidB": {"journey": "j2", "date": "2026-02-02"}})  # preserves vidA
    merge_routing(p, {"vidA": {"journey": "j1", "date": "2026-09-09"}})  # overwrites vidA's route
    routing = load_routing(p)
    assert routing == {
        "vidA": {"journey": "j1", "date": "2026-09-09"},
        "vidB": {"journey": "j2", "date": "2026-02-02"},
    }


def test_merge_routing_empty_is_noop(tmp_path):
    p = tmp_path / "video_routing.yaml"
    merge_routing(p, {})
    assert not p.exists()


# --- per-LF region cross-check before durable routing -----------------


def test_assert_region_consistency_refuses_cross_region_lf():
    # A USA LF (ResolvedLocation -> usa) inside a --region ireland run must
    # loud-refuse: routing it into the Ireland journey is authoritative forever.
    state = {
        "per_lf_resolved_location": {
            "1": {"is_us": False, "country_code": "Ireland"},
            "2": {"is_us": True, "country_code": "US"},  # USA LF, wrong region
        }
    }
    with pytest.raises(JourneyRegionMismatchError) as exc:
        assert_region_consistency(state, "ireland")
    assert "usa" in str(exc.value).lower()
    assert "2" in str(exc.value)  # names the offending LF


def test_assert_region_consistency_allows_matching_region():
    state = {
        "per_lf_resolved_location": {
            "1": {"is_us": False, "country_code": "Ireland"},
            "2": {"is_us": False, "country_code": "Ireland"},
        }
    }
    assert_region_consistency(state, "ireland")  # proceeds, no raise


def test_assert_region_consistency_noop_on_indeterminate_or_no_region():
    # GPS-less / legacy LF (indeterminate model) never refuses.
    assert_region_consistency(
        {"per_lf_resolved_location": {"1": {"is_us": False, "country_code": None}}}, "usa"
    )
    # No run region -> nothing to cross-check.
    assert_region_consistency({"per_lf_resolved_location": {"1": {"is_us": True}}}, None)
    # No resolved-location map -> no-op.
    assert_region_consistency({}, "ireland")


_CORRUPT_ROUTING = "routes:\n  lf_new: {journey: [unterminated\n"

_CORRUPT_JOURNEY = textwrap.dedent(
    """\
    ---
    title: "Broken"
    days: [unterminated
    ---
    Body text.
    """
)


def test_a_corrupt_routing_file_warns_instead_of_aborting_the_sync(tmp_path):
    """``reconcile_journeys`` promises it "never raises on a routing/data
    inconsistency" -- but ``load_routing`` reads and parses a file, and both
    of those fail. A truncated routing file took down the whole sync-site
    pass with a traceback instead of the warning the contract names."""
    site = _make_site(tmp_path, routes=_CORRUPT_ROUTING)
    result = reconcile_journeys(_MANIFEST, site, apply=True)
    assert result.warnings, "a corrupt routing file produced no warning"
    assert any("routing" in w.lower() for w in result.warnings)
    assert result.embedded == []


def test_a_corrupt_journey_warns_and_the_other_journeys_still_reconcile(tmp_path):
    """One unreadable journey markdown used to abort the pass part-way --
    journeys sorted before it were already written, journeys after it were
    silently never reached, and the caller got an exception rather than a
    result naming which. Degrade per file, like the catalog reader does."""
    site = _make_site(tmp_path)
    (site / "content" / "journeys" / "aaa-broken.md").write_text(_CORRUPT_JOURNEY)
    (site / "data" / "video_routing.yaml").write_text(
        _ROUTES + "  lf_broken: {journey: aaa-broken, date: '2026-03-04'}\n"
    )
    manifest = {
        "videos": [
            *_MANIFEST["videos"],
            {"id": "lf_broken", "title": "B", "duration_seconds": 700, "category": "real_lf"},
        ]
    }

    result = reconcile_journeys(manifest, site, apply=True)

    assert any("aaa-broken" in w for w in result.warnings), (
        f"the corrupt journey is not named in the warnings: {result.warnings}"
    )
    # "aaa-broken" sorts BEFORE "test-journey": the healthy journey is only
    # reached at all if the corrupt one stopped being fatal.
    assert ("test-journey", "lf_new") in result.embedded, (
        f"a journey after the corrupt one never reconciled: {result.embedded}"
    )


def test_a_healthy_sync_reports_no_damage_warnings(tmp_path):
    """The accept-case: the new guards must not start warning about files
    that are perfectly readable."""
    result = reconcile_journeys(_MANIFEST, _make_site(tmp_path), apply=True)
    assert not [w for w in result.warnings if "could not" in w.lower()]
    assert ("test-journey", "lf_new") in result.embedded
