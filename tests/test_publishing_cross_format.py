"""The publish plan carries a format, not an assumed short.

Covers only the data model (``models.py``) and its sidecar round trip
(``plan_io.py``) -- no photo/carousel adapter exists yet.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from gomoveshift.publishing.models import (
    FORMAT_CAROUSEL,
    FORMAT_PHOTO,
    FORMAT_SHORT,
    InstagramPayload,
    MediaFingerprint,
    PlatformPayloads,
    PublishPlan,
    RouteSelection,
    ScheduleWindow,
    ShortPublishRecord,
)
from gomoveshift.publishing.plan_io import load_publish_plan

SCHEDULED = datetime(2026, 9, 1, 19, 0, tzinfo=timezone.utc)


def _media(path: str = "/v/x.mp4", sha: str = "a" * 64) -> MediaFingerprint:
    return MediaFingerprint(path, 1234, 21.5, 1080, 1920, "h264", sha)


def _record(**over) -> ShortPublishRecord:
    kwargs = dict(
        project_name="proj",
        lf_index=2,
        short_index=3,
        media=_media(),
        schedule=ScheduleWindow(SCHEDULED, "America/Chicago"),
        recording_date="2026-08-01",
        sampled_regions=("driftless",),
        parent_lf_video_id="vid1",
        parent_lf_publish_at=SCHEDULED,
        payloads=PlatformPayloads(instagram=InstagramPayload("cap")),
        consent_hash="c",
        reviewed_plan_hash="HASH1",
        routes=RouteSelection(selected=("instagram",)),
    )
    kwargs.update(over)
    return ShortPublishRecord(**kwargs)


def _write(tmp_path, plan, name="proj_publish_plan.json"):
    from gomoveshift.pipeline.drive_phases.phase_d import _serialize_publish_plan

    path = tmp_path / name
    path.write_text(_serialize_publish_plan(plan), encoding="utf-8")
    return path


# --- construction ------------------------------------------------------------


def test_photo_record_is_constructible():
    record = _record(format=FORMAT_PHOTO)
    assert record.format == FORMAT_PHOTO
    assert record.additional_media == ()


def test_carousel_record_is_constructible_with_multiple_items():
    record = _record(
        format=FORMAT_CAROUSEL,
        additional_media=(_media(path="/v/y.mp4", sha="b" * 64),),
    )
    assert record.format == FORMAT_CAROUSEL
    assert len(record.additional_media) == 1


def test_carousel_with_zero_additional_items_raises():
    with pytest.raises(ValueError, match="carousel"):
        _record(format=FORMAT_CAROUSEL, additional_media=())


def test_short_with_two_media_items_raises():
    with pytest.raises(ValueError, match="additional_media"):
        _record(format=FORMAT_SHORT, additional_media=(_media(path="/v/y.mp4", sha="b" * 64),))


def test_unstated_format_with_extra_media_raises():
    with pytest.raises(ValueError, match="additional_media"):
        _record(additional_media=(_media(path="/v/y.mp4", sha="b" * 64),))


def test_unknown_format_string_raises():
    with pytest.raises(ValueError, match="format"):
        _record(format="tweet")


# --- plan_io round trip -------------------------------------------------------


def test_photo_record_round_trips_through_plan_io(tmp_path):
    plan = PublishPlan(shorts=(_record(format=FORMAT_PHOTO),))
    loaded = load_publish_plan(_write(tmp_path, plan)).shorts[0]
    assert loaded.format == FORMAT_PHOTO


def test_carousel_record_round_trips_through_plan_io(tmp_path):
    extra = _media(path="/v/y.mp4", sha="b" * 64)
    plan = PublishPlan(shorts=(_record(format=FORMAT_CAROUSEL, additional_media=(extra,)),))
    loaded = load_publish_plan(_write(tmp_path, plan)).shorts[0]
    assert loaded.format == FORMAT_CAROUSEL
    assert loaded.additional_media == (extra,)


def test_explicit_null_format_round_trips_as_none(tmp_path):
    plan = PublishPlan(shorts=(_record(),))  # format defaults to None
    loaded = load_publish_plan(_write(tmp_path, plan)).shorts[0]
    assert loaded.format is None


def test_old_sidecar_with_no_format_key_loads_as_short(tmp_path):
    """A file that predates the ``format`` field entirely -- this package
    only ever produced shorts before it existed, so the absent key is
    documented, deliberate behaviour, not a silent default."""
    plan = PublishPlan(shorts=(_record(),))
    path = _write(tmp_path, plan)
    raw = json.loads(path.read_text())
    del raw["shorts"][0]["format"]
    path.write_text(json.dumps(raw), encoding="utf-8")

    loaded = load_publish_plan(path).shorts[0]
    assert loaded.format == FORMAT_SHORT


def test_old_sidecar_with_no_additional_media_key_loads_empty(tmp_path):
    plan = PublishPlan(shorts=(_record(),))
    path = _write(tmp_path, plan)
    raw = json.loads(path.read_text())
    del raw["shorts"][0]["additional_media"]
    path.write_text(json.dumps(raw), encoding="utf-8")

    loaded = load_publish_plan(path).shorts[0]
    assert loaded.additional_media == ()
