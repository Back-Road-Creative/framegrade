"""One slot, four platforms, four publish times.

The best-time resolver picks the shorts window per (region, platform) — Instagram's best
hour is not YouTube's. Until now the plan carried ONE
``schedule.scheduled_at_utc`` per short and every delivery row copied it,
so a per-platform window had nowhere to land. The plan now carries a
per-platform map beside it, and ``enqueue_from_plan`` reads the platform's
own time when one is present.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

from gomoveshift.publishing.enqueue_from_plan import enqueue_from_plan_sidecar
from gomoveshift.publishing.models import MediaFingerprint
from gomoveshift.publishing.plan_builder import build_publish_plan
from gomoveshift.publishing.plan_io import load_publish_plan
from gomoveshift.publishing.queue import open_queue_db

_BASE_WINDOW = ["12:00", "16:00"]


def _media(short_index: int) -> MediaFingerprint:
    return MediaFingerprint(
        path=f"/output/lf1/short_{short_index:03d}.mp4",
        size_bytes=12_345,
        duration_seconds=42.5,
        width=1080,
        height=1920,
        codec="h264",
        sha256=str(short_index) * 64,
    )


def _post(short_index: int):
    return SimpleNamespace(
        short_index=short_index,
        caption="Rolling through County Kerry today",
        youtube_title=f"County Kerry Drive #{short_index}",
        youtube_category="24",
        youtube_tags="County Kerry,road trip",
        youtube_description="A short drive through the hills.",
        tiktok_title=f"tt {short_index}",
        sampled_regions=["County Kerry, Munster"],
    )


def _slot(short_index: int, time: str, date: str = "2026-04-01"):
    return SimpleNamespace(short_index=short_index, date=date, time=time)


def _build(**overrides):
    posts = [_post(1), _post(2)]
    kwargs = dict(
        project_name="proj",
        shorts_by_lf={1: posts},
        slots_by_short_index={1: _slot(1, "12:00:00"), 2: _slot(2, "16:00:00")},
        media_by_short_index={p.short_index: _media(p.short_index) for p in posts},
        parent_lf_video_id_by_short_index={1: "vidparent", 2: "vidparent"},
        parent_recording_date_by_short_index={1: "2026-04-01", 2: "2026-04-01"},
        parent_publish_at_by_short_index={
            1: "2026-03-31T18:00:00+00:00",
            2: "2026-03-31T18:00:00+00:00",
        },
        route_selected=("youtube", "instagram"),
    )
    kwargs.update(overrides)
    return build_publish_plan(**kwargs)


def test_without_a_per_platform_window_every_platform_keeps_the_one_slot():
    plan = _build()
    for record in plan.shorts:
        assert record.schedule.scheduled_at_utc_by_platform == {}


def test_a_platform_window_moves_that_platform_only():
    # The window moved 16:00 to the front for Instagram: its first short of the day
    # goes at 16:00, its second at 12:00. YouTube was not resolved, so it is
    # absent from the map and keeps the slot the scheduler gave it.
    plan = _build(shorts_times_by_platform={"instagram": ["16:00", "12:00"]})
    by_index = {r.short_index: r for r in plan.shorts}

    first, second = by_index[1], by_index[2]
    assert first.schedule.scheduled_at_utc.hour == 16  # 12:00 EST -> 16:00Z
    assert first.schedule.scheduled_at_utc_by_platform["instagram"].hour == 20
    assert second.schedule.scheduled_at_utc.hour == 20  # 16:00 EST -> 20:00Z
    assert second.schedule.scheduled_at_utc_by_platform["instagram"].hour == 16
    assert "youtube" not in first.schedule.scheduled_at_utc_by_platform


def test_a_platform_window_equal_to_the_slots_records_nothing():
    """No move is not a move: an unchanged window must not write a map entry,
    or every drive's sidecar grows four redundant copies of its own slot."""
    plan = _build(shorts_times_by_platform={"instagram": list(_BASE_WINDOW)})
    for record in plan.shorts:
        assert record.schedule.scheduled_at_utc_by_platform == {}


def test_the_map_survives_the_sidecar_round_trip(tmp_path):
    from gomoveshift.pipeline.drive_phases.phase_d import _serialize_publish_plan

    plan = _build(shorts_times_by_platform={"instagram": ["16:00", "12:00"]})
    path = tmp_path / "proj_publish_plan.json"
    path.write_text(_serialize_publish_plan(plan), encoding="utf-8")

    loaded = load_publish_plan(path)
    assert loaded.shorts[0].schedule.scheduled_at_utc_by_platform["instagram"] == datetime(
        2026, 4, 1, 20, 0, tzinfo=timezone.utc
    )


def test_a_sidecar_without_per_platform_times_still_loads(tmp_path):
    from gomoveshift.pipeline.drive_phases.phase_d import _serialize_publish_plan

    path = tmp_path / "proj_publish_plan.json"
    raw = json.loads(_serialize_publish_plan(_build()))
    for record in raw["shorts"]:
        record["schedule"].pop("scheduled_at_utc_by_platform")
    path.write_text(json.dumps(raw), encoding="utf-8")

    loaded = load_publish_plan(path)
    assert loaded.shorts[0].schedule.scheduled_at_utc_by_platform == {}


def test_the_delivery_row_carries_its_own_platform_time(tmp_path):
    from gomoveshift.pipeline.drive_phases.phase_d import _serialize_publish_plan

    plan = _build(shorts_times_by_platform={"instagram": ["16:00", "12:00"]})
    path = tmp_path / "proj_publish_plan.json"
    path.write_text(_serialize_publish_plan(plan), encoding="utf-8")

    conn = open_queue_db(tmp_path / "queue.db")
    enqueue_from_plan_sidecar(conn, path)

    rows = {
        (r["platform"], r["short_index"]): r["scheduled_at"]
        for r in conn.execute(
            "SELECT platform, short_index, scheduled_at FROM deliveries"
        ).fetchall()
    }
    assert rows[("instagram", 1)].startswith("2026-04-01T20:00")
    assert rows[("youtube", 1)].startswith("2026-04-01T16:00")
    assert rows[("instagram", 2)].startswith("2026-04-01T16:00")
    assert rows[("youtube", 2)].startswith("2026-04-01T20:00")
