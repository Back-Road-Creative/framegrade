"""Tests for reading a publish-plan sidecar into the delivery queue.

See docs/social-publishing-queue.md.
"""

from __future__ import annotations

import json

import pytest

from gomoveshift.publishing.enqueue_from_plan import (
    DeliveryPlanConflictError,
    PublishPlanSidecarError,
    enqueue_from_plan_sidecar,
)
from gomoveshift.publishing.queue import open_queue_db


@pytest.fixture
def conn(tmp_path):
    c = open_queue_db(tmp_path / "seo")
    try:
        yield c
    finally:
        c.close()


def _record(
    *,
    project_name="p",
    short_index=10,
    selected=("youtube", "instagram"),
    plan_hash="hash-a",
    scheduled_at="2026-09-07T16:00:00+00:00",
):
    record = {
        "project_name": project_name,
        "lf_index": 1,
        "short_index": short_index,
        "reviewed_plan_hash": plan_hash,
        "routes": {"selected": list(selected)},
    }
    if scheduled_at is not None:
        # Every real sidecar carries this (phase_d serializes
        # ``schedule.scheduled_at_utc``); the row's publish gate reads it.
        record["schedule"] = {
            "scheduled_at_utc": scheduled_at,
            "display_timezone": "America/New_York",
        }
    return record


def _write_sidecar(tmp_path, records, name="p_publish_plan.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"shorts": records}), encoding="utf-8")
    return path


class TestFreshEnqueue:
    def test_enqueues_one_row_per_selected_platform(self, tmp_path, conn):
        sidecar = _write_sidecar(tmp_path, [_record()])
        result = enqueue_from_plan_sidecar(conn, sidecar)
        assert result.enqueued_count == 2
        assert result.already_present_count == 0
        rows = conn.execute("SELECT COUNT(*) FROM deliveries").fetchone()[0]
        assert rows == 2

    def test_enqueued_rows_carry_the_sidecar_path_they_came_from(self, tmp_path, conn):
        sidecar = _write_sidecar(tmp_path, [_record()])
        enqueue_from_plan_sidecar(conn, sidecar)
        rows = conn.execute("SELECT DISTINCT sidecar_path FROM deliveries").fetchall()
        assert [r[0] for r in rows] == [str(sidecar)]

    def test_platform_absent_from_selected_gets_no_row(self, tmp_path, conn):
        sidecar = _write_sidecar(tmp_path, [_record(selected=("youtube",))])
        enqueue_from_plan_sidecar(conn, sidecar)
        rows = conn.execute(
            "SELECT platform FROM deliveries WHERE project_name='p' AND short_index=10"
        ).fetchall()
        assert [r[0] for r in rows] == ["youtube"]


class TestIdempotentRerun:
    def test_second_run_over_same_sidecar_adds_no_rows(self, tmp_path, conn):
        sidecar = _write_sidecar(tmp_path, [_record()])
        enqueue_from_plan_sidecar(conn, sidecar)
        result_2 = enqueue_from_plan_sidecar(conn, sidecar)
        assert result_2.enqueued_count == 0
        assert result_2.already_present_count == 2
        rows = conn.execute("SELECT COUNT(*) FROM deliveries").fetchone()[0]
        assert rows == 2


class TestConflict:
    def test_changed_plan_hash_raises_actionable_error(self, tmp_path, conn):
        sidecar_1 = _write_sidecar(
            tmp_path, [_record(selected=("youtube",), plan_hash="hash-a")], name="p1.json"
        )
        enqueue_from_plan_sidecar(conn, sidecar_1)
        sidecar_2 = _write_sidecar(
            tmp_path, [_record(selected=("youtube",), plan_hash="hash-b")], name="p2.json"
        )
        with pytest.raises(DeliveryPlanConflictError) as exc_info:
            enqueue_from_plan_sidecar(conn, sidecar_2)
        msg = str(exc_info.value)
        assert "p" in msg
        assert "10" in msg
        assert "youtube" in msg
        assert "hash-a" in msg
        assert "hash-b" in msg


class TestUnknownPlatform:
    def test_unknown_platform_raises_loudly(self, tmp_path, conn):
        sidecar = _write_sidecar(tmp_path, [_record(selected=("snapchat",))])
        with pytest.raises(PublishPlanSidecarError, match="snapchat"):
            enqueue_from_plan_sidecar(conn, sidecar)


class TestMalformedSidecar:
    def test_invalid_json_names_the_path(self, tmp_path, conn):
        sidecar = tmp_path / "broken.json"
        sidecar.write_text("{not json", encoding="utf-8")
        with pytest.raises(PublishPlanSidecarError, match=str(sidecar)):
            enqueue_from_plan_sidecar(conn, sidecar)

    def test_missing_key_names_the_path(self, tmp_path, conn):
        sidecar = tmp_path / "missing_key.json"
        bad = _record()
        del bad["reviewed_plan_hash"]
        sidecar.write_text(json.dumps({"shorts": [bad]}), encoding="utf-8")
        with pytest.raises(PublishPlanSidecarError, match=str(sidecar)):
            enqueue_from_plan_sidecar(conn, sidecar)

    def test_wrong_typed_field_names_the_path(self, tmp_path, conn):
        sidecar = tmp_path / "wrong_type.json"
        bad = _record()
        bad["short_index"] = "not-an-int"
        sidecar.write_text(json.dumps({"shorts": [bad]}), encoding="utf-8")
        with pytest.raises(PublishPlanSidecarError, match=str(sidecar)):
            enqueue_from_plan_sidecar(conn, sidecar)

    def test_missing_shorts_list_names_the_path(self, tmp_path, conn):
        sidecar = tmp_path / "no_shorts.json"
        sidecar.write_text(json.dumps({"not_shorts": []}), encoding="utf-8")
        with pytest.raises(PublishPlanSidecarError, match=str(sidecar)):
            enqueue_from_plan_sidecar(conn, sidecar)


def test_reads_a_sidecar_written_by_the_real_phase_d_serializer(conn, tmp_path):
    """The other tests hand-build the sidecar dict, which cannot catch a
    reader/writer shape mismatch — hand-built inner dicts are exactly why
    four live defects shipped past green tests in this module's sibling
    adapters. This one round-trips through the ACTUAL producer,
    ``phase_d._serialize_publish_plan``, so a change to either side breaks
    here rather than on the first real publish."""
    from datetime import datetime, timezone

    from gomoveshift.pipeline.drive_phases.phase_d import _serialize_publish_plan
    from gomoveshift.publishing.models import (
        MediaFingerprint,
        PlatformPayloads,
        PublishPlan,
        RouteSelection,
        ScheduleWindow,
        ShortPublishRecord,
        YouTubePayload,
    )

    scheduled = datetime(2026, 9, 1, tzinfo=timezone.utc)
    record = ShortPublishRecord(
        project_name="proj",
        lf_index=0,
        short_index=3,
        media=MediaFingerprint("/x.mp4", 1, 2.0, 1080, 1920, "h264", "a" * 64),
        schedule=ScheduleWindow(scheduled, "America/Chicago"),
        recording_date="2026-08-01",
        sampled_regions=("a",),
        parent_lf_video_id="vid",
        parent_lf_publish_at=scheduled,
        payloads=PlatformPayloads(youtube=YouTubePayload("t", "d")),
        consent_hash="c",
        reviewed_plan_hash="HASH1",
        routes=RouteSelection(selected=("youtube", "tiktok")),
    )
    sidecar = tmp_path / "proj_publish_plan.json"
    sidecar.write_text(_serialize_publish_plan(PublishPlan(shorts=(record,))), encoding="utf-8")

    first = enqueue_from_plan_sidecar(conn, sidecar)
    assert first.enqueued_count == 2, "both routed platforms should enqueue"
    assert first.already_present_count == 0

    # Re-running the same sidecar is a no-op, not an error.
    second = enqueue_from_plan_sidecar(conn, sidecar)
    assert second.enqueued_count == 0
    assert second.already_present_count == 2

    rows = sorted(
        tuple(r) for r in conn.execute("SELECT short_index, platform, plan_hash FROM deliveries")
    )
    assert rows == [(3, "tiktok", "HASH1"), (3, "youtube", "HASH1")]


def test_sidecar_rows_are_live_by_default(tmp_path, conn):
    sidecar = _write_sidecar(tmp_path, [_record()])
    enqueue_from_plan_sidecar(conn, sidecar)
    rows = conn.execute("SELECT shadow FROM deliveries").fetchall()
    assert rows and all(r["shadow"] == 0 for r in rows)


def test_shadow_run_spools_rows_that_cannot_publish(tmp_path, conn):
    sidecar = _write_sidecar(tmp_path, [_record()])
    enqueue_from_plan_sidecar(conn, sidecar, shadow=True)
    rows = conn.execute("SELECT shadow FROM deliveries").fetchall()
    assert rows and all(r["shadow"] == 1 for r in rows)


def test_a_record_without_a_schedule_is_refused(tmp_path, conn):
    """Spooling it would mean publish-immediately on three of four
    platforms -- refuse rather than guess a time."""
    sidecar = _write_sidecar(tmp_path, [_record(scheduled_at=None)])
    with pytest.raises(PublishPlanSidecarError) as exc:
        enqueue_from_plan_sidecar(conn, sidecar)
    assert "scheduled_at_utc" in str(exc.value)
    assert conn.execute("SELECT COUNT(*) FROM deliveries").fetchone()[0] == 0


def test_enqueued_rows_carry_the_plan_schedule(tmp_path, conn):
    sidecar = _write_sidecar(tmp_path, [_record(scheduled_at="2026-09-08T12:00:00+00:00")])
    enqueue_from_plan_sidecar(conn, sidecar)
    times = {r["scheduled_at"] for r in conn.execute("SELECT scheduled_at FROM deliveries")}
    assert times == {"2026-09-08T12:00:00+00:00"}


def test_max_shorts_caps_by_lowest_short_index(tmp_path, conn):
    """``max_shorts`` keeps the N lowest ``short_index`` records -- planner
    rank order -- and ignores the rest without touching the file."""
    sidecar = _write_sidecar(
        tmp_path,
        [_record(short_index=3), _record(short_index=1), _record(short_index=2)],
    )
    result = enqueue_from_plan_sidecar(conn, sidecar, max_shorts=2)
    rows = sorted({r["short_index"] for r in conn.execute("SELECT short_index FROM deliveries")})
    assert rows == [1, 2]
    assert result.enqueued_count == 4  # two platforms per kept short
