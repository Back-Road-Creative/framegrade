"""Unit tests for the superlong-day journal merge (``plan_superlong_merge``).

Superlong parts get ONE merge event per DAY, listing every
part in order, gated on the LAST part's ``publish_at`` — a 4h encode cap
must not fragment one day into weeks-apart reader-facing entries. Mirrors
the conventions in test_journal_publisher.py: fixture at
tests/fixtures/journals/sample_trip.md, raw-dict state/sidecar shapes
matching the real on-disk JSON (``state["superlong_parts"]`` and the
standalone ``superlong_uploads.json`` sidecar).
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from gomoveshift.cli.handlers.publish_journal import run_publish_journal
from gomoveshift.pipeline.journal_publisher import (
    load_journey,
    plan_merge,
    plan_superlong_merge,
)
from gomoveshift.pipeline.youtube_sidecar import save_superlong_sidecar

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "journals"
FIXTURE = FIXTURE_DIR / "sample_trip.md"
NOW = datetime(2026, 5, 26, 12, 0, tzinfo=timezone.utc)


def _superlong_state(date="2025-09-03", n=2):
    records = [
        {
            "part_index": i,
            "part_count": n,
            "duration_sec": 3600.0 * i,
            "path": f"/tmp/sl_{i:02d}.mp4",
            "location": "Test County",
            "title": f"Test Drive — Part {i} of {n}",
        }
        for i in range(1, n + 1)
    ]
    return {"superlong_parts": {date: {"part_count": n, "records": records}}}


def _uploads(date="2025-09-03", n=2, publish_ats=None):
    publish_ats = publish_ats or [None] * n
    return {
        f"{date}:{i:02d}": {
            "video_id": f"SL_{date}_{i:02d}",
            "title": f"Test Drive — Part {i} of {n}",
            "url": f"https://youtu.be/SL_{date}_{i:02d}",
            "publish_at": publish_ats[i - 1],
        }
        for i in range(1, n + 1)
    }


def test_all_parts_public_produces_one_entry_listing_every_part_in_order():
    doc, _body = load_journey(FIXTURE_DIR / "sample_trip.md")
    state = _superlong_state(n=3)
    uploads = _uploads(n=3, publish_ats=["2025-08-01T00:00:00Z"] * 3)

    plan = plan_superlong_merge(state, uploads, doc, now=NOW)

    assert plan.errors == []
    assert plan.deferrals == []
    assert [e.youtube_id for e in plan.inserts] == [
        "SL_2025-09-03_01",
        "SL_2025-09-03_02",
        "SL_2025-09-03_03",
    ]
    assert all(e.day_date == "2025-09-03" for e in plan.inserts)


def test_any_part_not_yet_public_defers_the_whole_day():
    doc, _body = load_journey(FIXTURE_DIR / "sample_trip.md")
    state = _superlong_state(n=3)
    uploads = _uploads(
        n=3,
        publish_ats=[
            "2025-08-01T00:00:00Z",
            "2025-08-08T00:00:00Z",
            "2099-01-01T00:00:00Z",  # last part still scheduled
        ],
    )

    plan = plan_superlong_merge(state, uploads, doc, now=NOW)

    assert plan.inserts == []
    assert plan.errors == []
    assert len(plan.deferrals) == 3
    # All three share the SAME (latest-part) gate so a later replay flushes
    # them together, not one at a time as each part's own date passes.
    assert {e.publish_at for e in plan.deferrals} == {"2099-01-01T00:00:00Z"}


def test_day_with_unuploaded_part_produces_no_entry_and_no_deferral():
    """A part that hasn't even been uploaded yet (no sidecar record) means
    the day can't honestly be described — no insert, no deferral."""
    doc, _body = load_journey(FIXTURE_DIR / "sample_trip.md")
    state = _superlong_state(n=2)
    uploads = _uploads(n=1)  # only part 1 uploaded so far

    plan = plan_superlong_merge(state, uploads, doc, now=NOW)

    assert plan.inserts == []
    assert plan.deferrals == []
    assert plan.errors == []


def test_regular_lf_merge_unaffected_by_superlong_parts_on_same_state():
    """plan_merge (the regular-LF path) must ignore state['superlong_parts']
    entirely — the two mechanisms never cross-contaminate."""
    doc, _body = load_journey(FIXTURE_DIR / "sample_trip.md")
    state = _superlong_state(date="2025-09-04", n=2)
    state.update(
        {
            "youtube_uploads": {1: {"video_id": "FRESH_VID_1", "title": "Regular LF"}},
            "per_lf_recording_date": {1: "2025-09-03"},
            "regular_lf_paths": {1: "/tmp/lf_01.mp4"},
        }
    )

    def ffprobe(_path):
        return 600.0

    plan = plan_merge(state, doc, ffprobe_duration=ffprobe, now=NOW)

    assert plan.errors == []
    assert len(plan.inserts) == 1
    assert plan.inserts[0].youtube_id == "FRESH_VID_1"


# ---------------------------------------------------------------------------
# CLI wiring — through run_publish_journal, not the planner directly. A test
# that only calls plan_superlong_merge cannot catch an unwired planner.
# ---------------------------------------------------------------------------


def test_run_publish_journal_writes_superlong_day_when_all_parts_public(tmp_path):
    site = tmp_path / "site"
    (site / "content" / "journeys").mkdir(parents=True)
    shutil.copy(FIXTURE, site / "content" / "journeys" / "sample-trip.md")
    journey = site / "content" / "journeys" / "sample-trip.md"

    state_path = tmp_path / "pipeline_state.json"
    state_path.write_text(
        json.dumps(
            {
                "superlong_parts": {
                    "2025-09-03": {
                        "part_count": 2,
                        "records": [
                            {"part_index": 1, "duration_sec": 3600.0, "path": "/tmp/sl_01.mp4"},
                            {"part_index": 2, "duration_sec": 7200.0, "path": "/tmp/sl_02.mp4"},
                        ],
                    }
                }
            }
        )
    )
    # Sidecar lives beside the state file here (no pending_dir passed —
    # standalone-CLI convention, same pending_base used for the sidecar).
    save_superlong_sidecar(
        tmp_path,
        {
            "2025-09-03:01": {
                "video_id": "SL_CLI_01",
                "title": "Superlong Part 1",
                "publish_at": "2025-08-01T00:00:00Z",
            },
            "2025-09-03:02": {
                "video_id": "SL_CLI_02",
                "title": "Superlong Part 2",
                "publish_at": "2025-08-08T00:00:00Z",
            },
        },
    )

    rc = run_publish_journal(
        state_path=state_path,
        site_path=site,
        journey_slug="sample-trip",
        apply=True,
        allow_day_create=False,
        now=NOW,
    )

    assert rc == 0
    text = journey.read_text()
    assert "SL_CLI_01" in text
    assert "SL_CLI_02" in text


# ---------------------------------------------------------------------------
# state_path.parent != output_base (drive-run layout) — regression
# test for that layout: ``pipeline_state.json`` lives under
# ``<output_base>/_stages/`` (an ext4-backed intermediate cache that can even
# be a wholly separate mount), while ``superlong_uploads.json`` is written
# next to the drive's durable output at ``<output_base>/``. Resolving the
# sidecar from ``state_path.parent`` looks one level below where it
# actually lives and silently drops the superlong upload from the journey.
# ---------------------------------------------------------------------------


def test_run_publish_journal_finds_superlong_sidecar_at_output_root(tmp_path):
    site = tmp_path / "site"
    (site / "content" / "journeys").mkdir(parents=True)
    shutil.copy(FIXTURE, site / "content" / "journeys" / "sample-trip.md")
    journey = site / "content" / "journeys" / "sample-trip.md"

    output_base = tmp_path / "output" / "sample-day1"
    stages_dir = output_base / "_stages"
    stages_dir.mkdir(parents=True)
    state_path = stages_dir / "pipeline_state.json"
    state_path.write_text(
        json.dumps(
            {
                "output_base": str(output_base),
                "superlong_parts": {
                    "2025-09-03": {
                        "part_count": 1,
                        "records": [
                            {"part_index": 1, "duration_sec": 3600.0, "path": "/tmp/sl_01.mp4"},
                        ],
                    }
                },
            }
        )
    )
    # Sidecar lives at the OUTPUT ROOT — sibling of _stages/, not inside it.
    save_superlong_sidecar(
        output_base,
        {
            "2025-09-03:01": {
                "video_id": "OUTPUTROOTVID",
                "title": "Superlong Part 1",
                "publish_at": "2025-08-01T00:00:00Z",
            },
        },
    )
    assert (output_base / "superlong_uploads.json").exists()
    assert not (stages_dir / "superlong_uploads.json").exists()

    rc = run_publish_journal(
        state_path=state_path,
        site_path=site,
        journey_slug="sample-trip",
        apply=True,
        allow_day_create=False,
        now=NOW,
    )

    assert rc == 0
    assert "OUTPUTROOTVID" in journey.read_text()


def test_explicit_pending_dir_still_wins_for_superlong_sidecar(tmp_path):
    """The drive auto-hook's explicit ``pending_dir`` seam must keep
    resolving the superlong sidecar, even though it differs from both
    ``state["output_base"]`` and the state file's own directory."""
    site = tmp_path / "site"
    (site / "content" / "journeys").mkdir(parents=True)
    shutil.copy(FIXTURE, site / "content" / "journeys" / "sample-trip.md")
    journey = site / "content" / "journeys" / "sample-trip.md"

    output_base = tmp_path / "output" / "sample-day1"
    stages_dir = output_base / "_stages"
    stages_dir.mkdir(parents=True)
    state_path = stages_dir / "pipeline_state.json"
    explicit_pending = tmp_path / "durable-pending-dir"
    explicit_pending.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "output_base": str(output_base),
                "superlong_parts": {
                    "2025-09-03": {
                        "part_count": 1,
                        "records": [
                            {"part_index": 1, "duration_sec": 3600.0, "path": "/tmp/sl_01.mp4"},
                        ],
                    }
                },
            }
        )
    )
    save_superlong_sidecar(
        explicit_pending,
        {
            "2025-09-03:01": {
                "video_id": "EXPLICITWINS",
                "title": "Superlong Part 1",
                "publish_at": "2025-08-01T00:00:00Z",
            },
        },
    )

    rc = run_publish_journal(
        state_path=state_path,
        site_path=site,
        journey_slug="sample-trip",
        apply=True,
        allow_day_create=False,
        now=NOW,
        pending_dir=explicit_pending,
    )

    assert rc == 0
    assert "EXPLICITWINS" in journey.read_text()
