"""Tests for rehydrating a publish-plan sidecar (``publishing/plan_io.py``).

The load-bearing test here is the round-trip against the REAL producer,
``phase_d._serialize_publish_plan``. A reader tested only against
hand-built dicts asserts that it agrees with the test author, not with the
writer — which is exactly how four live defects shipped past green suites
in this module's sibling adapters.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone

import pytest

from gomoveshift.pipeline.drive_phases.phase_d import _serialize_publish_plan
from gomoveshift.publishing.models import (
    InstagramPayload,
    MediaFingerprint,
    PlatformPayloads,
    PublishPlan,
    RouteSelection,
    ScheduleWindow,
    ShortPublishRecord,
    TikTokPayload,
    YouTubePayload,
)
from gomoveshift.publishing.plan_io import PublishPlanFormatError, load_publish_plan

SCHEDULED = datetime(2026, 9, 1, 19, 0, tzinfo=timezone.utc)


def _record(**over) -> ShortPublishRecord:
    kwargs = dict(
        project_name="proj",
        lf_index=2,
        short_index=3,
        media=MediaFingerprint("/v/x.mp4", 1234, 21.5, 1080, 1920, "h264", "a" * 64),
        schedule=ScheduleWindow(SCHEDULED, "America/Chicago"),
        recording_date="2026-08-01",
        sampled_regions=("driftless", "coulee"),
        parent_lf_video_id="vid1",
        parent_lf_publish_at=SCHEDULED,
        payloads=PlatformPayloads(
            youtube=YouTubePayload("t", "d", ("a", "b"), "2"),
            instagram=InstagramPayload("cap"),
            tiktok=TikTokPayload("tt", "SELF_ONLY"),
        ),
        consent_hash="c",
        reviewed_plan_hash="HASH1",
        routes=RouteSelection(selected=("youtube", "tiktok")),
    )
    kwargs.update(over)
    return ShortPublishRecord(**kwargs)


def _write(tmp_path, plan, name="proj_publish_plan.json"):
    path = tmp_path / name
    path.write_text(_serialize_publish_plan(plan), encoding="utf-8")
    return path


def test_round_trips_through_the_real_phase_d_serializer(tmp_path):
    original = _record()
    loaded = load_publish_plan(_write(tmp_path, PublishPlan(shorts=(original,)))).shorts[0]

    before = dataclasses.asdict(original)
    after = dataclasses.asdict(loaded)
    # generated_at is stripped by the writer on purpose (gate C/D hash
    # stability), so the rehydrated record takes the model default.
    before.pop("generated_at")
    after.pop("generated_at")
    assert before == after


def test_tuple_fields_come_back_as_tuples_not_lists(tmp_path):
    """asdict() writes tuples as JSON arrays. The models are frozen with
    tuple[...] fields, and RouteSelection is hashable-by-value, so lists
    would silently change equality and hashing semantics."""
    loaded = load_publish_plan(_write(tmp_path, PublishPlan(shorts=(_record(),)))).shorts[0]
    assert isinstance(loaded.sampled_regions, tuple)
    assert isinstance(loaded.routes.selected, tuple)
    assert isinstance(loaded.payloads.youtube.tags, tuple)


def test_datetimes_come_back_as_aware_datetimes(tmp_path):
    loaded = load_publish_plan(_write(tmp_path, PublishPlan(shorts=(_record(),)))).shorts[0]
    assert loaded.schedule.scheduled_at_utc == SCHEDULED
    assert loaded.schedule.scheduled_at_utc.tzinfo is not None
    assert loaded.parent_lf_publish_at == SCHEDULED


def test_unset_payload_stays_none(tmp_path):
    """asdict() writes an unset payload as null, not as an absent key."""
    loaded = load_publish_plan(_write(tmp_path, PublishPlan(shorts=(_record(),)))).shorts[0]
    assert loaded.payloads.facebook is None
    assert loaded.payloads.instagram is not None


def test_empty_plan_loads_as_empty(tmp_path):
    assert load_publish_plan(_write(tmp_path, PublishPlan(shorts=()))).shorts == ()


def test_multiple_records_keep_their_order(tmp_path):
    plan = PublishPlan(shorts=(_record(short_index=1), _record(short_index=2)))
    loaded = load_publish_plan(_write(tmp_path, plan))
    assert [s.short_index for s in loaded.shorts] == [1, 2]


# --- untrusted input: every failure names the file --------------------------


def test_missing_file_names_the_path(tmp_path):
    missing = tmp_path / "nope_publish_plan.json"
    with pytest.raises(PublishPlanFormatError) as exc:
        load_publish_plan(missing)
    assert str(missing) in str(exc.value)


def test_malformed_json_names_the_path(tmp_path):
    path = tmp_path / "p_publish_plan.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(PublishPlanFormatError) as exc:
        load_publish_plan(path)
    assert str(path) in str(exc.value)


def test_missing_shorts_list_names_the_path(tmp_path):
    path = tmp_path / "p_publish_plan.json"
    path.write_text(json.dumps({"not_shorts": []}), encoding="utf-8")
    with pytest.raises(PublishPlanFormatError) as exc:
        load_publish_plan(path)
    assert "shorts" in str(exc.value)


def test_missing_required_field_names_the_field_and_record(tmp_path):
    path = _write(tmp_path, PublishPlan(shorts=(_record(),)))
    raw = json.loads(path.read_text())
    del raw["shorts"][0]["consent_hash"]
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(PublishPlanFormatError) as exc:
        load_publish_plan(path)
    assert "consent_hash" in str(exc.value)
    assert "shorts[0]" in str(exc.value)


def test_bad_timestamp_names_the_field(tmp_path):
    path = _write(tmp_path, PublishPlan(shorts=(_record(),)))
    raw = json.loads(path.read_text())
    raw["shorts"][0]["schedule"]["scheduled_at_utc"] = "not-a-date"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(PublishPlanFormatError) as exc:
        load_publish_plan(path)
    assert "scheduled_at_utc" in str(exc.value)


def test_unknown_platform_is_rejected_by_the_model(tmp_path):
    """RouteSelection.__post_init__ validates platform names, so a corrupt
    sidecar is refused here rather than producing a queue row no handler
    will ever claim."""
    path = _write(tmp_path, PublishPlan(shorts=(_record(),)))
    raw = json.loads(path.read_text())
    raw["shorts"][0]["routes"]["selected"] = ["myspace"]
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(PublishPlanFormatError) as exc:
        load_publish_plan(path)
    assert "myspace" in str(exc.value)


def test_legacy_rollback_key_is_discarded(tmp_path):
    """A pre-removal sidecar (an older on-disk shape) still carries
    ``routes.rollback`` -- it must load cleanly, with the key simply
    dropped, rather than rejecting a real on-disk sidecar."""
    path = _write(tmp_path, PublishPlan(shorts=(_record(),)))
    raw = json.loads(path.read_text())
    raw["shorts"][0]["routes"] = {"rollback": "metricool", "selected": ["youtube", "tiktok"]}
    path.write_text(json.dumps(raw), encoding="utf-8")
    loaded = load_publish_plan(path).shorts[0]
    assert loaded.routes.selected == ("youtube", "tiktok")
    assert not hasattr(loaded.routes, "rollback")


def test_wrong_typed_tuple_field_names_the_field(tmp_path):
    path = _write(tmp_path, PublishPlan(shorts=(_record(),)))
    raw = json.loads(path.read_text())
    raw["shorts"][0]["sampled_regions"] = "driftless"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(PublishPlanFormatError) as exc:
        load_publish_plan(path)
    assert "sampled_regions" in str(exc.value)


# ---------------------------------------------------------------------------
# GMS-17c: consistency gate on load (upstream of enqueue / dispatch)
# ---------------------------------------------------------------------------


def _yt(title, description, tags=("a", "b")):
    return PlatformPayloads(
        youtube=YouTubePayload(title, description, tags, "2"),
        instagram=InstagramPayload("cap"),
        tiktok=TikTokPayload("tt", "SELF_ONLY"),
    )


def test_consistent_title_hashtags_load(tmp_path):
    rec = _record(
        payloads=_yt("Coast road #Goleen #POVDriving", "Coast road.\n\n#POVDriving #Goleen")
    )
    loaded = load_publish_plan(_write(tmp_path, PublishPlan(shorts=(rec,)))).shorts[0]
    assert loaded.payloads == rec.payloads


def test_title_hashtag_absent_from_description_is_refused(tmp_path):
    """The builder takes a Short's title hashtags from its own description;
    a title that carries a hashtag its description lacks was edited apart
    from it, so title and description no longer agree."""
    rec = _record(payloads=_yt("Coast road #Kenmare", "Coast road.\n\n#POVDriving #Goleen"))
    path = _write(tmp_path, PublishPlan(shorts=(rec,)))
    with pytest.raises(PublishPlanFormatError, match="#Kenmare") as exc:
        load_publish_plan(path)
    assert str(path) in str(exc.value)


def test_place_claim_outside_the_clips_regions_is_refused(tmp_path):
    """A county the short's own sampled regions never visited, named in its
    tags and title, is refused on load as it is at build time."""
    rec = _record(
        sampled_regions=("Goleen, County Cork",),
        payloads=_yt("Roads of County Kerry", "Coast road.", tags=("County Kerry",)),
    )
    with pytest.raises(PublishPlanFormatError, match="CROSS-REGION"):
        load_publish_plan(_write(tmp_path, PublishPlan(shorts=(rec,))))


def test_spliced_text_is_refused(tmp_path):
    rec = _record(payloads=_yt("t", "A drive through To Clonakilty."))
    with pytest.raises(PublishPlanFormatError, match="GRAMMAR"):
        load_publish_plan(_write(tmp_path, PublishPlan(shorts=(rec,))))
