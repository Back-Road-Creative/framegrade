"""Unit tests for journal_publisher (pure merge logic).

CLI-level integration is covered separately in test_cli_publish_journal.py.
These tests inject a fake ffprobe duration callable so no video files are
needed; they exercise plan_merge / apply_merge directly against the
fixture YAML at tests/fixtures/journals/sample_trip.md.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from gomoveshift.pipeline.journal_publisher import (
    MergePlan,
    VideoEntry,
    apply_merge,
    format_duration,
    load_journey,
    plan_merge,
    serialize_journey,
)
from gomoveshift.pipeline.site_sync.videos_yaml import VideoLibrary

from _state_helpers import make_state

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "journals"


def _dry_library() -> VideoLibrary:
    """A dry-run library — these tests assert the journey-doc mutation only; the
    ``data/videos.yaml`` coupling lives in test_journey_embed_library.py."""
    return VideoLibrary.for_site(Path("/nonexistent-site"), writable=False)


def _fake_ffprobe(durations_by_path: dict[str, float]):
    def _probe(path):
        return durations_by_path[str(path)]

    return _probe


def _state_for_three_days():
    """State with three uploaded LFs:

    - LF 1: brand-new, matches day 1 (videos: [])           → insert
    - LF 2: video_id collides with existing day 2 entry     → skip
    - LF 3: video_id collides with human-shortened day 3    → skip
    """
    return {
        "youtube_uploads": {
            1: {
                "video_id": "FRESH_VID_1",
                "url": "https://youtu.be/FRESH_VID_1",
                "title": "Fresh Pipeline Title for Day 1",
                "uploaded_at": "2025-09-03T12:00:00Z",
            },
            2: {
                "video_id": "EXISTING_VID_2",
                "url": "https://youtu.be/EXISTING_VID_2",
                "title": "Pipeline Would Have Titled This Differently",
                "uploaded_at": "2025-09-04T12:00:00Z",
            },
            3: {
                "video_id": "HUMAN_SHORTENED_VID_3",
                "url": "https://youtu.be/HUMAN_SHORTENED_VID_3",
                "title": "Long Generated Title That Human Already Shortened In Site Repo",
                "uploaded_at": "2025-09-05T12:00:00Z",
            },
        },
        "per_lf_recording_date": {
            1: "2025-09-03",
            2: "2025-09-04",
            3: "2025-09-05",
        },
        "per_lf_titles": {
            1: "Fresh Pipeline Title for Day 1",
            2: "Pipeline Would Have Titled This Differently",
            3: "Long Generated Title That Human Already Shortened In Site Repo",
        },
        "regular_lf_paths": {
            1: "/tmp/lf_01.mp4",
            2: "/tmp/lf_02.mp4",
            3: "/tmp/lf_03.mp4",
        },
    }


# ---------------------------------------------------------------------------
# format_duration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0, "00:00"),
        (7, "00:07"),
        (67, "01:07"),
        (2527, "42:07"),
        (3600, "1:00:00"),
        (5561, "1:32:41"),
        (36000, "10:00:00"),
    ],
)
def test_format_duration_matches_site_convention(seconds, expected):
    assert format_duration(seconds) == expected


# ---------------------------------------------------------------------------
# plan_merge — happy paths
# ---------------------------------------------------------------------------


def test_plan_merge_classifies_fresh_skip_and_human_shortened(tmp_path):
    journey_path = FIXTURE_DIR / "sample_trip.md"
    doc, body = load_journey(journey_path)
    state = _state_for_three_days()
    ffprobe = _fake_ffprobe(
        {
            "/tmp/lf_01.mp4": 2527.0,  # 42:07
            "/tmp/lf_02.mp4": 686.0,  # 11:26
            "/tmp/lf_03.mp4": 5561.0,  # 1:32:41
        }
    )

    plan = plan_merge(state, doc, ffprobe_duration=ffprobe)

    inserts = {entry.lf_index: entry for entry in plan.inserts}
    assert set(inserts.keys()) == {1}, "only LF 1 should be a fresh insert"
    assert inserts[1].youtube_id == "FRESH_VID_1"
    assert inserts[1].title == "Fresh Pipeline Title for Day 1"
    assert inserts[1].duration == "42:07"
    assert inserts[1].day_date == "2025-09-03"

    skipped_indices = {idx for idx, _vid, _reason in plan.skips}
    assert skipped_indices == {2, 3}, "LF 2 and LF 3 should skip as already present"
    assert plan.errors == []


def test_plan_merge_prefers_youtube_duration_over_deleted_local_encode():
    """The upload already happened by the time plan_merge runs — the local
    encode may already be deleted. Duration must resolve from the video_id's
    YouTube-reported duration, never from ffprobing a gone file."""
    journey_path = FIXTURE_DIR / "sample_trip.md"
    doc, body = load_journey(journey_path)
    state = _state_for_three_days()
    state["regular_lf_paths"] = {
        1: "/nonexistent/lf_01.mp4",
        2: "/nonexistent/lf_02.mp4",
        3: "/nonexistent/lf_03.mp4",
    }

    def _ffprobe_must_not_be_called(path):
        raise AssertionError(f"ffprobe should not be called for {path}")

    yt_durations = {
        "FRESH_VID_1": 2527.0,  # 42:07
        "EXISTING_VID_2": 686.0,
        "HUMAN_SHORTENED_VID_3": 5561.0,
    }

    plan = plan_merge(
        state,
        doc,
        ffprobe_duration=_ffprobe_must_not_be_called,
        youtube_duration_lookup=lambda ids: yt_durations,
    )

    inserts = {entry.lf_index: entry for entry in plan.inserts}
    assert set(inserts.keys()) == {1}
    assert inserts[1].duration == "42:07"
    assert plan.errors == []


def test_plan_merge_falls_back_to_ffprobe_when_youtube_has_no_duration():
    """When the YouTube lookup has no duration for a video_id (not yet
    resolvable there), ffprobe is still consulted for that LF."""
    journey_path = FIXTURE_DIR / "sample_trip.md"
    doc, body = load_journey(journey_path)
    state = _state_for_three_days()
    ffprobe = _fake_ffprobe({"/tmp/lf_01.mp4": 2527.0})

    plan = plan_merge(
        state,
        doc,
        ffprobe_duration=ffprobe,
        youtube_duration_lookup=lambda ids: {},
    )

    inserts = {entry.lf_index: entry for entry in plan.inserts}
    assert inserts[1].duration == "42:07"
    assert plan.errors == []


def test_apply_merge_writes_new_entry_and_preserves_existing(tmp_path):
    src = FIXTURE_DIR / "sample_trip.md"
    work = tmp_path / "sample_trip.md"
    shutil.copy(src, work)

    doc, body = load_journey(work)
    state = _state_for_three_days()
    ffprobe = _fake_ffprobe(
        {
            "/tmp/lf_01.mp4": 2527.0,
            "/tmp/lf_02.mp4": 686.0,
            "/tmp/lf_03.mp4": 5561.0,
        }
    )

    plan = plan_merge(state, doc, ffprobe_duration=ffprobe)
    apply_merge(doc, plan, library=_dry_library())
    rendered = serialize_journey(doc, body)
    work.write_text(rendered)

    doc2, body2 = load_journey(work)
    day1 = next(d for d in doc2["days"] if str(d["date"]) == "2025-09-03")
    day2 = next(d for d in doc2["days"] if str(d["date"]) == "2025-09-04")
    day3 = next(d for d in doc2["days"] if str(d["date"]) == "2025-09-05")

    # Day 1: brand-new entry appended in upload order.
    assert len(day1["videos"]) == 1
    assert day1["videos"][0]["youtube_id"] == "FRESH_VID_1"
    assert day1["videos"][0]["title"] == "Fresh Pipeline Title for Day 1"
    assert day1["videos"][0]["duration"] == "42:07"

    # Day 2: existing entry untouched (no duplicate insert).
    assert len(day2["videos"]) == 1
    assert day2["videos"][0]["youtube_id"] == "EXISTING_VID_2"
    assert day2["videos"][0]["title"] == "Existing Day 2 Title"

    # Day 3: human-shortened title MUST NOT be overwritten with state's longer title.
    assert len(day3["videos"]) == 1
    assert day3["videos"][0]["title"] == "Human Shortened"
    assert day3["videos"][0]["duration"] == "42:07"

    # Markdown body preserved.
    assert "## About This Trip" in body2
    assert "Sample body content that must survive round-trip unchanged." in body2


def test_apply_merge_keeps_video_count_in_sync():
    """The site deploy CI fails when video_count != actual; apply_merge must
    recompute it (covers both immediate insert and the replay/cron flush)."""
    doc = {
        "video_count": 1,
        "days": [
            {"day_number": 1, "date": "2025-09-03", "videos": []},
            {
                "day_number": 2,
                "date": "2025-09-04",
                "videos": [{"youtube_id": "OLD", "title": "x", "duration": "1:00"}],
            },
        ],
    }
    plan = MergePlan(
        inserts=[
            VideoEntry(1, "NEW1", "n1", "2:00", "2025-09-03"),
            VideoEntry(2, "NEW2", "n2", "3:00", "2025-09-04"),
        ],
        skips=[],
        errors=[],
    )
    apply_merge(doc, plan, library=_dry_library())
    assert doc["video_count"] == 3  # 1 existing + 2 inserted


def test_apply_merge_leaves_video_count_absent_when_doc_omits_it():
    """A journey that omits video_count opts out of the check — don't impose it."""
    doc = {"days": [{"day_number": 1, "date": "2025-09-03", "videos": []}]}
    plan = MergePlan(
        inserts=[VideoEntry(1, "NEW", "n", "2:00", "2025-09-03")],
        skips=[],
        errors=[],
    )
    apply_merge(doc, plan, library=_dry_library())
    assert "video_count" not in doc


# ---------------------------------------------------------------------------
# Idempotency — re-running a planned merge after apply is a no-op
# ---------------------------------------------------------------------------


def test_re_run_after_apply_is_idempotent(tmp_path):
    src = FIXTURE_DIR / "sample_trip.md"
    work = tmp_path / "sample_trip.md"
    shutil.copy(src, work)

    doc, body = load_journey(work)
    state = _state_for_three_days()
    ffprobe = _fake_ffprobe(
        {
            "/tmp/lf_01.mp4": 2527.0,
            "/tmp/lf_02.mp4": 686.0,
            "/tmp/lf_03.mp4": 5561.0,
        }
    )
    plan = plan_merge(state, doc, ffprobe_duration=ffprobe)
    apply_merge(doc, plan, library=_dry_library())
    work.write_text(serialize_journey(doc, body))

    # Second pass — nothing new to insert.
    doc2, _body2 = load_journey(work)
    plan2 = plan_merge(state, doc2, ffprobe_duration=ffprobe)
    assert plan2.inserts == []
    assert {idx for idx, _v, _r in plan2.skips} == {1, 2, 3}
    assert plan2.errors == []


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_missing_recording_date_is_an_error_without_allow_day_create():
    journey_path = FIXTURE_DIR / "sample_trip.md"
    doc, _body = load_journey(journey_path)
    state = make_state(
        youtube_uploads={
            1: {"video_id": "ORPHAN_VID", "title": "Orphan", "url": "u"},
        },
        per_lf_recording_date={1: "2025-12-31"},  # not in fixture
        regular_lf_paths={1: "/tmp/lf_01.mp4"},
    )
    ffprobe = _fake_ffprobe({"/tmp/lf_01.mp4": 600.0})

    plan = plan_merge(state, doc, ffprobe_duration=ffprobe, allow_day_create=False)

    assert plan.inserts == []
    assert plan.errors, "unmatched recording date must surface as an error"
    msg = "\n".join(plan.errors)
    assert "2025-12-31" in msg
    # Existing dates listed for operator readability.
    assert "2025-09-03" in msg


def test_missing_recording_date_with_allow_day_create_stubs_day(tmp_path):
    src = FIXTURE_DIR / "sample_trip.md"
    work = tmp_path / "sample_trip.md"
    shutil.copy(src, work)
    doc, body = load_journey(work)

    state = make_state(
        youtube_uploads={
            1: {"video_id": "NEW_DAY_VID", "title": "On New Day", "url": "u"},
        },
        per_lf_recording_date={1: "2025-09-07"},  # not in fixture (next day after 09-06)
        regular_lf_paths={1: "/tmp/lf_01.mp4"},
    )
    ffprobe = _fake_ffprobe({"/tmp/lf_01.mp4": 600.0})

    plan = plan_merge(state, doc, ffprobe_duration=ffprobe, allow_day_create=True)
    assert plan.errors == []
    assert len(plan.inserts) == 1
    apply_merge(doc, plan, library=_dry_library())
    work.write_text(serialize_journey(doc, body))

    doc2, _ = load_journey(work)
    new_day = next(d for d in doc2["days"] if str(d["date"]) == "2025-09-07")
    assert new_day["day_number"] == 5
    assert len(new_day["videos"]) == 1
    assert new_day["videos"][0]["youtube_id"] == "NEW_DAY_VID"


def test_backfill_earlier_day_inserted_by_date_rank_not_appended():
    """A backfilled day stub dated BEFORE an existing day must take its
    chronological number and slot into date order — never max()+1 tacked on
    the end. Here day 2 is missing (a numbering gap), so numbering the
    2025-09-02 stub as day 2 needs no renumber of the published days.
    """
    doc = {
        "video_count": 1,
        "days": [
            {
                "day_number": 1,
                "date": "2025-09-01",
                "day_of_week": "",
                "heading": "",
                "narrative": "",
                "videos": [],
            },
            {
                "day_number": 3,
                "date": "2025-09-03",
                "day_of_week": "",
                "heading": "",
                "narrative": "",
                "videos": [{"youtube_id": "V3", "title": "d3", "duration": "1:00"}],
            },
        ],
    }
    plan = MergePlan(
        inserts=[VideoEntry(1, "V2", "d2", "2:00", "2025-09-02")],
        skips=[],
        errors=[],
        days_to_create=["2025-09-02"],
    )
    apply_merge(doc, plan, library=_dry_library())

    dates = [str(d["date"]) for d in doc["days"]]
    numbers = [d["day_number"] for d in doc["days"]]
    # Inserted in date-sorted position, numbered by chronological rank.
    assert dates == ["2025-09-01", "2025-09-02", "2025-09-03"]
    assert numbers == [1, 2, 3]
    new_day = doc["days"][1]
    assert new_day["day_number"] == 2  # NOT max()+1 == 4 appended at end
    assert new_day["videos"][0]["youtube_id"] == "V2"


def test_backfill_that_would_renumber_published_days_raises():
    """When numbering a backfilled day by date would push an existing published
    day_number up, refuse — don't silently reorder already-published numbers.
    Days 1 and 2 are contiguous, so a stub dated between them
    (2025-09-02) has no free number and apply_merge must raise.
    """
    doc = {
        "days": [
            {
                "day_number": 1,
                "date": "2025-09-01",
                "day_of_week": "",
                "heading": "",
                "narrative": "",
                "videos": [],
            },
            {
                "day_number": 2,
                "date": "2025-09-03",
                "day_of_week": "",
                "heading": "",
                "narrative": "",
                "videos": [],
            },
        ],
    }
    plan = MergePlan(
        inserts=[VideoEntry(1, "V2", "d2", "2:00", "2025-09-02")],
        skips=[],
        errors=[],
        days_to_create=["2025-09-02"],
    )
    with pytest.raises(ValueError, match="renumber"):
        apply_merge(doc, plan, library=_dry_library())


def test_cross_day_youtube_id_collision_is_an_error():
    """If a state video_id already lives in a DIFFERENT day than we'd insert,
    refuse rather than silently duplicate or move."""
    journey_path = FIXTURE_DIR / "sample_trip.md"
    doc, _body = load_journey(journey_path)
    state = make_state(
        youtube_uploads={
            # Collides with day 2's EXISTING_VID_2, but state says it was filmed on day 4.
            1: {"video_id": "EXISTING_VID_2", "title": "..."},
        },
        per_lf_recording_date={1: "2025-09-06"},
        regular_lf_paths={1: "/tmp/lf_01.mp4"},
    )
    ffprobe = _fake_ffprobe({"/tmp/lf_01.mp4": 600.0})

    plan = plan_merge(state, doc, ffprobe_duration=ffprobe)
    assert plan.inserts == []
    assert plan.errors
    assert "EXISTING_VID_2" in "\n".join(plan.errors)


def test_title_truncated_at_max_length():
    journey_path = FIXTURE_DIR / "sample_trip.md"
    doc, _body = load_journey(journey_path)
    long_title = "A" * 250
    state = make_state(
        youtube_uploads={
            1: {"video_id": "LONG_TITLE_VID", "title": long_title},
        },
        per_lf_recording_date={1: "2025-09-03"},
        regular_lf_paths={1: "/tmp/lf_01.mp4"},
    )
    ffprobe = _fake_ffprobe({"/tmp/lf_01.mp4": 60.0})
    plan = plan_merge(state, doc, ffprobe_duration=ffprobe)
    assert len(plan.inserts) == 1
    assert len(plan.inserts[0].title) == 100


def test_title_falls_back_to_per_lf_titles_when_uploads_lack_title():
    journey_path = FIXTURE_DIR / "sample_trip.md"
    doc, _body = load_journey(journey_path)
    state = make_state(
        youtube_uploads={
            1: {"video_id": "VID_NO_TITLE"},  # no title key
        },
        per_lf_titles={1: "Title From State Fallback"},
        per_lf_recording_date={1: "2025-09-03"},
        regular_lf_paths={1: "/tmp/lf_01.mp4"},
    )
    ffprobe = _fake_ffprobe({"/tmp/lf_01.mp4": 60.0})
    plan = plan_merge(state, doc, ffprobe_duration=ffprobe)
    assert plan.inserts[0].title == "Title From State Fallback"


def test_missing_video_id_raises_in_plan():
    journey_path = FIXTURE_DIR / "sample_trip.md"
    doc, _body = load_journey(journey_path)
    state = make_state(
        youtube_uploads={
            1: {"title": "no video id here"},
        },
        per_lf_recording_date={1: "2025-09-03"},
        regular_lf_paths={1: "/tmp/lf_01.mp4"},
    )
    ffprobe = _fake_ffprobe({"/tmp/lf_01.mp4": 60.0})
    plan = plan_merge(state, doc, ffprobe_duration=ffprobe)
    assert plan.inserts == []
    assert plan.errors
    assert any("video_id" in e for e in plan.errors)


# ---------------------------------------------------------------------------
# Preview text
# ---------------------------------------------------------------------------


def test_preview_text_lists_inserts_and_skips_human_readably():
    journey_path = FIXTURE_DIR / "sample_trip.md"
    doc, _body = load_journey(journey_path)
    state = _state_for_three_days()
    ffprobe = _fake_ffprobe(
        {
            "/tmp/lf_01.mp4": 2527.0,
            "/tmp/lf_02.mp4": 686.0,
            "/tmp/lf_03.mp4": 5561.0,
        }
    )
    plan = plan_merge(state, doc, ffprobe_duration=ffprobe)
    text = plan.preview_text()
    assert "FRESH_VID_1" in text
    assert "2025-09-03" in text
    assert "EXISTING_VID_2" in text
    assert "skip" in text.lower()


# ---------------------------------------------------------------------------
# MergePlan public surface — sanity
# ---------------------------------------------------------------------------


def test_regular_lf_paths_as_list_shape_is_supported():
    """Real pipeline_state.json stores regular_lf_paths as a 0-indexed list,
    not a dict. The publisher must coerce list → 1-based dict so ffprobe
    lookup by lf_index works on actual on-disk state.

    Regression: a smoke test against a real on-disk
    pipeline_state.json crashed because list.items() doesn't exist."""
    journey_path = FIXTURE_DIR / "sample_trip.md"
    doc, _body = load_journey(journey_path)
    state = make_state(
        youtube_uploads={"1": {"video_id": "REAL_SHAPE", "title": "t"}},
        per_lf_recording_date={"1": "2025-09-03"},
        # On-disk shape: list, not dict.
        regular_lf_paths=["/tmp/lf_01.mp4"],
    )
    ffprobe = _fake_ffprobe({"/tmp/lf_01.mp4": 600.0})
    plan = plan_merge(state, doc, ffprobe_duration=ffprobe)
    assert plan.errors == []
    assert len(plan.inserts) == 1
    assert plan.inserts[0].youtube_id == "REAL_SHAPE"


def test_merge_plan_has_errors_property():
    plan = MergePlan(inserts=[], skips=[], errors=["x"])
    assert plan.has_errors is True
    plan2 = MergePlan(inserts=[], skips=[], errors=[])
    assert plan2.has_errors is False


# ---------------------------------------------------------------------------
# Date-gate — entries with future publish_at land in deferrals, not inserts
# ---------------------------------------------------------------------------


from datetime import datetime, timezone  # noqa: E402


def _state_with_publish_at(publish_at_value):
    """Minimal state shape for date-gate exercises: one fresh LF on day 1."""
    record = {
        "video_id": "DATEGATE_VID",
        "title": "Date Gate Test",
        "uploaded_at": "2025-09-03T12:00:00Z",
    }
    if publish_at_value is not None:
        record["publish_at"] = publish_at_value
    return {
        "youtube_uploads": {1: record},
        "per_lf_recording_date": {1: "2025-09-03"},
        "regular_lf_paths": {1: "/tmp/lf_01.mp4"},
    }


def test_plan_merge_future_publish_at_lands_in_deferrals(tmp_path):
    doc, _body = load_journey(FIXTURE_DIR / "sample_trip.md")
    state = _state_with_publish_at("2099-01-01T00:00:00Z")
    ffprobe = _fake_ffprobe({"/tmp/lf_01.mp4": 600.0})
    now = datetime(2026, 5, 26, 12, 0, tzinfo=timezone.utc)

    plan = plan_merge(state, doc, ffprobe_duration=ffprobe, now=now)

    assert plan.errors == []
    assert plan.inserts == []
    assert len(plan.deferrals) == 1
    assert plan.deferrals[0].youtube_id == "DATEGATE_VID"
    assert plan.deferrals[0].publish_at == "2099-01-01T00:00:00Z"


def test_plan_merge_past_publish_at_lands_in_inserts(tmp_path):
    doc, _body = load_journey(FIXTURE_DIR / "sample_trip.md")
    state = _state_with_publish_at("2025-01-01T00:00:00Z")
    ffprobe = _fake_ffprobe({"/tmp/lf_01.mp4": 600.0})
    now = datetime(2026, 5, 26, 12, 0, tzinfo=timezone.utc)

    plan = plan_merge(state, doc, ffprobe_duration=ffprobe, now=now)

    assert plan.errors == []
    assert plan.deferrals == []
    assert len(plan.inserts) == 1
    assert plan.inserts[0].publish_at == "2025-01-01T00:00:00Z"


def test_plan_merge_missing_publish_at_treated_as_immediate(tmp_path):
    """Back-compat: legacy uploads without publish_at must still insert."""
    doc, _body = load_journey(FIXTURE_DIR / "sample_trip.md")
    state = _state_with_publish_at(None)
    ffprobe = _fake_ffprobe({"/tmp/lf_01.mp4": 600.0})

    plan = plan_merge(state, doc, ffprobe_duration=ffprobe)

    assert plan.errors == []
    assert plan.deferrals == []
    assert len(plan.inserts) == 1
    assert plan.inserts[0].publish_at is None


def test_plan_merge_unparseable_publish_at_inserts_with_raw_string(tmp_path):
    """Unparseable publish_at can't be date-gated, so we insert (not defer).
    The raw string is preserved on the entry so an operator can audit it."""
    doc, _body = load_journey(FIXTURE_DIR / "sample_trip.md")
    state = _state_with_publish_at("junk-not-iso")
    ffprobe = _fake_ffprobe({"/tmp/lf_01.mp4": 600.0})

    plan = plan_merge(state, doc, ffprobe_duration=ffprobe)

    assert plan.errors == []
    assert plan.deferrals == []
    assert len(plan.inserts) == 1
    assert plan.inserts[0].publish_at == "junk-not-iso"


def test_plan_merge_publish_at_exactly_now_is_immediate(tmp_path):
    doc, _body = load_journey(FIXTURE_DIR / "sample_trip.md")
    now = datetime(2026, 5, 26, 12, 0, tzinfo=timezone.utc)
    state = _state_with_publish_at(now.isoformat().replace("+00:00", "Z"))
    ffprobe = _fake_ffprobe({"/tmp/lf_01.mp4": 600.0})

    plan = plan_merge(state, doc, ffprobe_duration=ffprobe, now=now)

    # publish_at == now → not strictly future → insert
    assert plan.deferrals == []
    assert len(plan.inserts) == 1


def test_preview_text_renders_deferrals():
    plan = MergePlan(
        inserts=[],
        skips=[],
        errors=[],
        deferrals=[
            VideoEntry(
                lf_index=1,
                youtube_id="vid1",
                title="t",
                duration="10:00",
                day_date="2025-09-14",
                publish_at="2099-01-01T00:00:00Z",
            )
        ],
    )
    text = plan.preview_text()
    assert "Deferred until publish_at fires" in text
    assert "vid1" in text
    assert "2099-01-01T00:00:00Z" in text
