"""Console Publishing tab over the delivery queue (``console/pubqueue.py``).

Queue view is read-only over ``publishing/queue.py``; enqueue/drain launch the
real ``enqueue-plan``/``dispatch-queue`` CLI subcommands via the job runner,
argv matching the parser contract exactly.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from gomoveshift.console import jobs, pubqueue
from gomoveshift.console.app import create_app
from gomoveshift.pipeline.drive_phases.phase_d import _serialize_publish_plan
from gomoveshift.publishing import dispatch
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
from gomoveshift.publishing.queue import enqueue, open_queue_db, pause_platform, transition

_SCHEDULED = datetime(2026, 9, 5, 19, 0, tzinfo=timezone.utc)


def _short_record(**over) -> ShortPublishRecord:
    kwargs = dict(
        project_name="proj",
        lf_index=0,
        short_index=1,
        media=MediaFingerprint("/v/x.mp4", 1234, 21.5, 1080, 1920, "h264", "a" * 64),
        schedule=ScheduleWindow(_SCHEDULED, "America/Chicago"),
        recording_date="2026-08-01",
        sampled_regions=("driftless",),
        parent_lf_video_id="vid1",
        parent_lf_publish_at=_SCHEDULED,
        payloads=PlatformPayloads(youtube=YouTubePayload("The Title", "desc")),
        consent_hash="c",
        reviewed_plan_hash="HASH1",
        routes=RouteSelection(selected=("youtube",)),
    )
    kwargs.update(over)
    return ShortPublishRecord(**kwargs)


def _write_sidecar(tmp_path, *shorts, name="proj_publish_plan.json"):
    path = tmp_path / name
    path.write_text(_serialize_publish_plan(PublishPlan(shorts=tuple(shorts))), encoding="utf-8")
    return path


def _client() -> TestClient:
    return TestClient(create_app())


def _config(tmp_path):
    return SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))


def test_queue_rows_missing_db_is_empty_list(tmp_path):
    assert pubqueue.queue_rows(_config(tmp_path)) == []
    assert pubqueue.queue_db_path(_config(tmp_path)) == tmp_path / "social_queue.sqlite3"


def test_queue_rows_reads_fixture_rows(tmp_path):
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 1, "youtube", "h1")
    enqueue(conn, "proj", 2, "instagram", "h1")
    conn.close()

    rows = pubqueue.queue_rows(_config(tmp_path))

    assert {(r["project_name"], r["short_index"], r["platform"]) for r in rows} == {
        ("proj", 1, "youtube"),
        ("proj", 2, "instagram"),
    }
    assert all(r["status"] == "queued" for r in rows)


def test_rows_by_state_groups_and_orders_lifecycle_first():
    rows = [
        {"status": "failed_permanent", "id": 1},
        {"status": "queued", "id": 2},
        {"status": "some_future_state", "id": 3},
        {"status": "queued", "id": 4},
    ]
    grouped = pubqueue.rows_by_state(rows)
    assert [status for status, _rows in grouped] == [
        "queued",
        "failed_permanent",
        "some_future_state",
    ]
    assert [r["id"] for r in dict(grouped)["queued"]] == [2, 4]


def test_rows_by_state_labels_publish_confirmation():
    rows = [
        {"status": "published", "id": 1, "published_at_source": "operator_confirmed"},
        {"status": "published", "id": 2, "published_at_source": "platform_reconciled"},
        {"status": "published", "id": 3, "published_at_source": None},
        {"status": "published", "id": 4, "published_at_source": "platform_confirmed"},
    ]
    grouped = dict(pubqueue.rows_by_state(rows))
    labels = {r["id"]: r["publish_confirmation"] for r in grouped["published"]}
    assert labels == {
        1: "operator confirmed",
        2: "platform reconciled (delayed)",
        3: "unknown (legacy)",
        4: "platform confirmed",
    }


def test_build_argv_matches_the_cli_contract():
    full = pubqueue.build_enqueue_argv("/runs/state.json", output_dir="/runs/out", shadow=True)
    assert full[3:] == [
        "enqueue-plan",
        "/runs/state.json",
        "--apply",
        "--output-dir",
        "/runs/out",
        "--shadow",
    ]
    bare = pubqueue.build_enqueue_argv("/runs/state.json")
    assert bare[3:] == ["enqueue-plan", "/runs/state.json", "--apply"]
    drain = pubqueue.build_drain_argv(sidecar_dir="/runs/out")
    assert drain[3:] == ["dispatch-queue", "--apply", "--sidecar-dir", "/runs/out"]
    assert pubqueue.build_drain_argv()[3:] == ["dispatch-queue", "--apply"]


def test_allowlist_covers_the_two_publishing_subcommands():
    assert {"enqueue-plan", "dispatch-queue"} <= jobs.ALLOWED_SUBCOMMANDS


def test_publishing_get_renders_fixture_rows_by_state(tmp_path, monkeypatch):
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 1, "youtube", "h1")
    conn.close()

    response = _client().get("/publishing")

    assert response.status_code == 200
    assert "proj" in response.text
    assert "queued" in response.text


def test_publishing_get_missing_db_is_empty_state(tmp_path, monkeypatch):
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))

    response = _client().get("/publishing")

    assert response.status_code == 200
    assert "empty" in response.text.lower()


def test_publishing_enqueue_and_drain_launch_the_real_subcommands(tmp_path, monkeypatch):
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    # The state path has to sit inside a run this console launched, or the
    # containment guard refuses it before any argv is built.
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    state = output_dir / "pipeline_state.json"
    _register_output_dir(tmp_path, output_dir)
    launched = _launch_recorder(monkeypatch)

    enqueue_resp = _client().post(
        "/publishing/enqueue", data={"state_path": str(state)}, follow_redirects=False
    )
    drain_resp = _client().post("/publishing/drain", data={}, follow_redirects=False)

    assert enqueue_resp.status_code == 303 and drain_resp.status_code == 303
    assert launched["enqueue-plan"][3:] == ["enqueue-plan", str(state.resolve()), "--apply"]
    assert launched["dispatch-queue"][3:] == ["dispatch-queue", "--apply"]


def test_publishing_drain_form_uses_plain_language_with_flag_kept_as_diagnostic(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))

    response = _client().get("/publishing")

    assert response.status_code == 200
    assert "Publish everything that's due" in response.text
    assert "Publish now" in response.text
    assert "Publish plan file location (optional)" in response.text
    # jargon gone from the primary label/button, kept as a diagnostic hint
    assert "Drain the queue" not in response.text
    assert "Drain now" not in response.text
    assert "Sidecar dir (optional)" not in response.text
    assert "--sidecar-dir" in response.text
    assert "dispatch-queue" in response.text


def test_publishing_tab_visible_for_a_brand_that_only_disables_drives_and_shorts(
    tmp_path, monkeypatch
):
    # Isolate the queue like every sibling: without this the tab opens the
    # live SEO-service queue and its schema lock, which a different OS user
    # cannot write.
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))
    monkeypatch.setenv("GMS_CONSOLE_BRAND", "examplebrand")
    assert TestClient(create_app()).get("/publishing").status_code == 200


def test_mark_published_transitions_and_is_never_reselected_by_dispatch(tmp_path):
    conn = open_queue_db(tmp_path)
    delivery_id = enqueue(conn, "proj", 0, "tiktok", "hash-a")
    transition(conn, delivery_id, "queued", "awaiting_manual_publish")
    conn.close()

    db_path = tmp_path / "social_queue.sqlite3"
    pubqueue.mark_published(db_path, delivery_id)

    conn = open_queue_db(tmp_path)
    row = conn.execute("SELECT status FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
    assert row["status"] == "published"
    # published is one of dispatch's own two terminal statuses -- confirms
    # this can never be re-selected for drain, not just that the status
    # string happens to read "published".
    due = dispatch._select_due(conn, "9999-12-31T00:00:00+00:00")
    conn.close()
    assert delivery_id not in {r["id"] for r in due}


def test_mark_published_accepts_a_row_the_tiktok_pack_parked(tmp_path):
    """The pack parks a row with NO platform_upload_id. The owner's
    mark-published click must still close it."""
    from gomoveshift.publishing.tiktok_pack import pack_and_park

    rec = _short_record(
        payloads=PlatformPayloads(
            tiktok=TikTokPayload(title="Moab: red rock", privacy_level="PUBLIC_TO_EVERYONE")
        )
    )
    conn = open_queue_db(tmp_path)
    delivery_id = enqueue(conn, "proj", 1, "tiktok", "hash-a", scheduled_at=_SCHEDULED.isoformat())
    src = tmp_path / "short.mp4"
    src.write_bytes(b"h264")
    pack_and_park(conn, delivery_id, rec, src, pack_dir=tmp_path / "pack")
    parked = conn.execute(
        "SELECT status, platform_upload_id FROM deliveries WHERE id=?", (delivery_id,)
    ).fetchone()
    assert (parked["status"], parked["platform_upload_id"]) == ("awaiting_manual_publish", None)
    conn.close()

    pubqueue.mark_published(tmp_path / "social_queue.sqlite3", delivery_id)

    conn = open_queue_db(tmp_path)
    row = conn.execute(
        "SELECT status, published_at_source FROM deliveries WHERE id=?", (delivery_id,)
    ).fetchone()
    conn.close()
    assert (row["status"], row["published_at_source"]) == ("published", "operator_confirmed")


def test_mark_published_records_operator_confirmed_source(tmp_path):
    """A console mark-published confirmation is provably a human click, never
    confusable with the automated TikTok reconcile poll."""
    conn = open_queue_db(tmp_path)
    delivery_id = enqueue(conn, "proj", 0, "tiktok", "hash-a")
    transition(conn, delivery_id, "queued", "awaiting_manual_publish")
    conn.close()

    db_path = tmp_path / "social_queue.sqlite3"
    pubqueue.mark_published(db_path, delivery_id)

    conn = open_queue_db(tmp_path)
    row = conn.execute(
        "SELECT published_at_source FROM deliveries WHERE id=?", (delivery_id,)
    ).fetchone()
    conn.close()
    assert row["published_at_source"] == "operator_confirmed"


def test_mark_published_refuses_wrong_starting_state(tmp_path):
    conn = open_queue_db(tmp_path)
    delivery_id = enqueue(conn, "proj", 0, "tiktok", "hash-a")
    conn.close()

    db_path = tmp_path / "social_queue.sqlite3"
    with pytest.raises(ValueError):
        pubqueue.mark_published(db_path, delivery_id)

    conn = open_queue_db(tmp_path)
    row = conn.execute("SELECT status FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
    conn.close()
    assert row["status"] == "queued"


def test_mark_published_lost_race_is_reported_as_conflict_not_success(tmp_path, monkeypatch):
    """If the row moves out from under ``mark_published`` between its read
    and the ``transition`` call (a lost compare-and-set, ``transition``
    returns ``False``), that must surface as a conflict -- never be logged
    as a successful publish confirmation."""
    conn = open_queue_db(tmp_path)
    delivery_id = enqueue(conn, "proj", 0, "tiktok", "hash-a")
    transition(conn, delivery_id, "queued", "awaiting_manual_publish")
    conn.close()
    db_path = tmp_path / "social_queue.sqlite3"

    def _lost_race(conn, delivery_id, expected_status, new_status, **fields):
        # Simulate another actor moving the row to a different, non-terminal
        # state between mark_published's read and this call.
        conn.execute("UPDATE deliveries SET status='queued' WHERE id=?", (delivery_id,))
        conn.commit()
        return False

    monkeypatch.setattr(pubqueue, "transition", _lost_race)

    with pytest.raises(ValueError):
        pubqueue.mark_published(db_path, delivery_id)

    conn = open_queue_db(tmp_path)
    row = conn.execute("SELECT status FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
    conn.close()
    assert row["status"] == "queued"


def test_mark_published_repeat_confirmation_is_idempotent(tmp_path):
    """A second confirmation of an already-published row is a no-op, never
    an error and never a second publish."""
    conn = open_queue_db(tmp_path)
    delivery_id = enqueue(conn, "proj", 0, "tiktok", "hash-a")
    transition(conn, delivery_id, "queued", "awaiting_manual_publish")
    conn.close()
    db_path = tmp_path / "social_queue.sqlite3"

    pubqueue.mark_published(db_path, delivery_id)
    pubqueue.mark_published(db_path, delivery_id)  # must not raise

    conn = open_queue_db(tmp_path)
    row = conn.execute("SELECT status FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
    conn.close()
    assert row["status"] == "published"


def test_mark_published_writes_permalink_and_post_id_to_existing_columns(tmp_path):
    conn = open_queue_db(tmp_path)
    delivery_id = enqueue(conn, "proj", 0, "tiktok", "hash-a")
    transition(conn, delivery_id, "queued", "awaiting_manual_publish")
    conn.close()
    db_path = tmp_path / "social_queue.sqlite3"

    pubqueue.mark_published(
        db_path,
        delivery_id,
        permalink="https://www.tiktok.com/@x/video/1",
        platform_post_id="1",
    )

    conn = open_queue_db(tmp_path)
    row = conn.execute(
        "SELECT permalink, platform_post_id FROM deliveries WHERE id=?", (delivery_id,)
    ).fetchone()
    conn.close()
    assert row["permalink"] == "https://www.tiktok.com/@x/video/1"
    assert row["platform_post_id"] == "1"


def test_publishing_mark_published_route_accepts_permalink_and_post_id(tmp_path, monkeypatch):
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))
    conn = open_queue_db(tmp_path)
    delivery_id = enqueue(conn, "proj", 0, "tiktok", "hash-a")
    transition(conn, delivery_id, "queued", "awaiting_manual_publish")
    conn.close()

    response = _client().post(
        f"/publishing/mark-published/{delivery_id}",
        data={"permalink": "https://www.tiktok.com/@x/video/1", "platform_post_id": "1"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    conn = open_queue_db(tmp_path)
    row = conn.execute(
        "SELECT status, permalink, platform_post_id FROM deliveries WHERE id=?", (delivery_id,)
    ).fetchone()
    conn.close()
    assert row["status"] == "published"
    assert row["permalink"] == "https://www.tiktok.com/@x/video/1"
    assert row["platform_post_id"] == "1"


def test_publishing_get_never_claims_api_verified_for_a_manually_confirmed_row(
    tmp_path, monkeypatch
):
    """Regression for a real bug: the TikTok inbox handler sets
    ``platform_upload_id`` on EVERY row before it ever waits on a human
    (``handlers.py``'s ``make_tiktok_handler``, the ``inbox`` branch) --
    so every row ``mark_published`` ever touches carries it. A row's
    confirmation route cannot be told apart from an automated publish using
    columns the schema already has; the console must never claim
    "API verified" for a row a human confirmed by hand."""
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))
    conn = open_queue_db(tmp_path)
    delivery_id = enqueue(conn, "proj", 0, "tiktok", "hash-a")
    transition(conn, delivery_id, "queued", "submitting")
    # Mirrors handlers.py's real inbox-mode transition: platform_upload_id
    # is set here, before the row ever reaches awaiting_manual_publish.
    transition(conn, delivery_id, "submitting", "uploaded", platform_upload_id="tiktok-publish-1")
    transition(
        conn,
        delivery_id,
        "uploaded",
        "awaiting_manual_publish",
        platform_upload_id="tiktok-publish-1",
    )
    conn.close()
    pubqueue.mark_published(tmp_path / "social_queue.sqlite3", delivery_id)

    response = _client().get("/publishing")

    assert "api verified" not in response.text.lower()


def test_mark_published_unknown_delivery_id_raises_keyerror(tmp_path):
    conn = open_queue_db(tmp_path)
    conn.close()
    db_path = tmp_path / "social_queue.sqlite3"
    with pytest.raises(KeyError):
        pubqueue.mark_published(db_path, 999)


def test_mark_published_missing_db_raises_keyerror(tmp_path):
    with pytest.raises(KeyError):
        pubqueue.mark_published(tmp_path / "social_queue.sqlite3", 1)


def test_publishing_mark_published_route_transitions_and_redirects(tmp_path, monkeypatch):
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))
    conn = open_queue_db(tmp_path)
    delivery_id = enqueue(conn, "proj", 0, "tiktok", "hash-a")
    transition(conn, delivery_id, "queued", "awaiting_manual_publish")
    conn.close()

    response = _client().post(f"/publishing/mark-published/{delivery_id}", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/publishing"
    conn = open_queue_db(tmp_path)
    row = conn.execute("SELECT status FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
    conn.close()
    assert row["status"] == "published"


def test_publishing_mark_published_route_404_unknown_id(tmp_path, monkeypatch):
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))
    open_queue_db(tmp_path).close()

    response = _client().post("/publishing/mark-published/999", follow_redirects=False)

    assert response.status_code == 404


def test_publishing_mark_published_route_409_wrong_state(tmp_path, monkeypatch):
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))
    conn = open_queue_db(tmp_path)
    delivery_id = enqueue(conn, "proj", 0, "tiktok", "hash-a")
    conn.close()

    response = _client().post(f"/publishing/mark-published/{delivery_id}", follow_redirects=False)

    assert response.status_code == 409


def test_enrich_rows_with_sidecar_happy_path(tmp_path):
    sidecar_path = _write_sidecar(tmp_path, _short_record())
    rows = [
        {
            "project_name": "proj",
            "short_index": 1,
            "platform": "youtube",
            "sidecar_path": str(sidecar_path),
        }
    ]

    enriched = pubqueue.enrich_rows_with_sidecar(rows)

    assert enriched[0]["title"] == "The Title"
    assert enriched[0]["scheduled_platforms"] == "youtube"
    assert enriched[0]["scheduled_at_utc"] == _SCHEDULED.isoformat()
    assert enriched[0]["scheduled_at_local"] == "2026-09-05 14:00 America/Chicago"


def test_enrich_rows_with_sidecar_sets_format_from_sidecar(tmp_path):
    sidecar_path = _write_sidecar(tmp_path, _short_record(format="short"))
    rows = [
        {
            "project_name": "proj",
            "short_index": 1,
            "platform": "youtube",
            "sidecar_path": str(sidecar_path),
        }
    ]

    enriched = pubqueue.enrich_rows_with_sidecar(rows)

    assert enriched[0]["format"] == "short"


def test_enrich_rows_with_sidecar_sets_privacy_status_from_sidecar(tmp_path):
    # Default record format is None -- the video "not tracked per
    # item" reading, reused from console.library rather than re-derived.
    sidecar_path = _write_sidecar(tmp_path, _short_record())
    rows = [
        {
            "project_name": "proj",
            "short_index": 1,
            "platform": "youtube",
            "sidecar_path": str(sidecar_path),
        }
    ]

    enriched = pubqueue.enrich_rows_with_sidecar(rows)

    from gomoveshift.console.library import _PRIVACY_STATUS_VIDEO_NOT_TRACKED

    assert enriched[0]["privacy_status"] == _PRIVACY_STATUS_VIDEO_NOT_TRACKED


def test_enrich_rows_with_sidecar_missing_sidecar_path_is_none(tmp_path):
    rows = [{"project_name": "proj", "short_index": 1, "platform": "youtube"}]
    enriched = pubqueue.enrich_rows_with_sidecar(rows)
    assert enriched[0]["title"] is None
    assert enriched[0]["scheduled_at_local"] is None
    assert enriched[0]["scheduled_platforms"] is None
    assert enriched[0]["format"] is None
    assert enriched[0]["privacy_status"] is None


def test_enrich_rows_with_sidecar_missing_file_is_none(tmp_path):
    rows = [
        {
            "project_name": "proj",
            "short_index": 1,
            "platform": "youtube",
            "sidecar_path": str(tmp_path / "nope.json"),
        }
    ]
    enriched = pubqueue.enrich_rows_with_sidecar(rows)
    assert enriched[0]["title"] is None


def test_enrich_rows_with_sidecar_malformed_json_is_none(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    rows = [
        {
            "project_name": "proj",
            "short_index": 1,
            "platform": "youtube",
            "sidecar_path": str(bad),
        }
    ]
    enriched = pubqueue.enrich_rows_with_sidecar(rows)
    assert enriched[0]["title"] is None


def test_enrich_rows_with_sidecar_no_matching_short_is_none(tmp_path):
    sidecar_path = _write_sidecar(tmp_path, _short_record(short_index=1))
    rows = [
        {
            "project_name": "proj",
            "short_index": 99,
            "platform": "youtube",
            "sidecar_path": str(sidecar_path),
        }
    ]
    enriched = pubqueue.enrich_rows_with_sidecar(rows)
    assert enriched[0]["title"] is None
    assert enriched[0]["scheduled_platforms"] is None


def test_enrich_rows_with_sidecar_caches_one_read_per_file(tmp_path, monkeypatch):
    sidecar_path = _write_sidecar(
        tmp_path, _short_record(short_index=1), _short_record(short_index=2)
    )
    rows = [
        {
            "project_name": "proj",
            "short_index": 1,
            "platform": "youtube",
            "sidecar_path": str(sidecar_path),
        },
        {
            "project_name": "proj",
            "short_index": 2,
            "platform": "youtube",
            "sidecar_path": str(sidecar_path),
        },
    ]
    calls = []
    real_load = pubqueue.load_publish_plan

    def _counting_load(path):
        calls.append(path)
        return real_load(path)

    monkeypatch.setattr(pubqueue, "load_publish_plan", _counting_load)

    enriched = pubqueue.enrich_rows_with_sidecar(rows)

    assert len(calls) == 1
    assert enriched[0]["title"] == "The Title"
    assert enriched[1]["title"] == "The Title"


def test_publishing_get_renders_scheduled_date_and_title(tmp_path, monkeypatch):
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))
    conn = open_queue_db(tmp_path)
    sidecar_path = _write_sidecar(tmp_path, _short_record())
    from gomoveshift.publishing.queue import _now_iso

    with conn:
        cursor = conn.execute(
            "INSERT INTO deliveries "
            "(project_name, short_index, platform, plan_hash, status, sidecar_path, "
            "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
            ("proj", 1, "youtube", "h1", "queued", str(sidecar_path), _now_iso(), _now_iso()),
        )
        cursor.close()
    conn.close()

    response = _client().get("/publishing")

    assert response.status_code == 200
    assert "The Title" in response.text
    assert "America/Chicago" in response.text


def _parked_tiktok_row(tmp_path, monkeypatch):
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))
    conn = open_queue_db(tmp_path)
    delivery_id = enqueue(conn, "proj", 0, "tiktok", "hash-a")
    transition(conn, delivery_id, "queued", "awaiting_manual_publish")
    conn.close()
    return delivery_id


def test_publishing_get_renders_awaiting_manual_publish_hint_and_button(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_TIKTOK_MODE", "inbox")
    delivery_id = _parked_tiktok_row(tmp_path, monkeypatch)

    response = _client().get("/publishing")

    assert response.status_code == 200
    assert "publish from the tiktok app inbox" in response.text.lower()
    assert f"/publishing/mark-published/{delivery_id}" in response.text


def test_publishing_hint_names_the_pack_folder_in_pack_mode(tmp_path, monkeypatch):
    monkeypatch.delenv("GMS_TIKTOK_MODE", raising=False)
    monkeypatch.setenv("GMS_TIKTOK_PACK_DIR", "/mnt/e/ClipPack")
    _parked_tiktok_row(tmp_path, monkeypatch)

    text = _client().get("/publishing").text

    assert "Post the file from E:\\ClipPack (captions.txt), then confirm here." in text
    assert "app inbox" not in text


def test_publishing_hint_survives_an_unknown_tiktok_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_TIKTOK_MODE", "bogus")
    _parked_tiktok_row(tmp_path, monkeypatch)

    response = _client().get("/publishing")

    assert response.status_code == 200
    assert "then confirm here" in response.text


# --- The row's own scheduled_at is authoritative --------------------


def test_enrich_rows_with_sidecar_effective_schedule_from_row_when_no_sidecar():
    """Regression: a real ``scheduled_at``, no sidecar, must not render unscheduled."""
    rows = [
        {
            "project_name": "proj",
            "short_index": 1,
            "platform": "youtube",
            "scheduled_at": "2026-09-05T19:00:00+00:00",
        }
    ]

    enriched = pubqueue.enrich_rows_with_sidecar(rows)

    assert enriched[0]["scheduled_at_effective_local"] is not None
    assert "2026-09-05" in enriched[0]["scheduled_at_effective_local"]
    assert "19:00" in enriched[0]["scheduled_at_effective_local"]
    assert enriched[0]["schedule_discrepancy"] is None
    assert enriched[0]["sidecar_read_error"] is False


# --- held_reason ("blackout") ------------------------------------------


def test_enrich_rows_with_sidecar_held_reason_none_without_blackout_policy():
    rows = [
        {
            "project_name": "proj",
            "short_index": 1,
            "platform": "youtube",
            "status": "queued",
            "scheduled_at": "2026-09-05T19:00:00+00:00",
        }
    ]

    enriched = pubqueue.enrich_rows_with_sidecar(rows)

    assert enriched[0]["held_reason"] is None


def test_enrich_rows_with_sidecar_held_reason_blackout_on_a_queued_row(monkeypatch):
    from datetime import date

    from gomoveshift.publishing.brand_policy import BrandPolicy

    monkeypatch.setattr(
        pubqueue,
        "load_brand_policy",
        lambda slug: BrandPolicy(blackout=(date(2026, 9, 5),)),
    )
    rows = [
        {
            "project_name": "proj",
            "short_index": 1,
            "platform": "youtube",
            "status": "queued",
            "scheduled_at": "2026-09-05T19:00:00+00:00",
        }
    ]

    enriched = pubqueue.enrich_rows_with_sidecar(rows)

    assert enriched[0]["held_reason"] == "blackout"


def test_enrich_rows_with_sidecar_held_reason_off_blackout_date_is_none(monkeypatch):
    from datetime import date

    from gomoveshift.publishing.brand_policy import BrandPolicy

    monkeypatch.setattr(
        pubqueue,
        "load_brand_policy",
        lambda slug: BrandPolicy(blackout=(date(2026, 9, 6),)),
    )
    rows = [
        {
            "project_name": "proj",
            "short_index": 1,
            "platform": "youtube",
            "status": "queued",
            "scheduled_at": "2026-09-05T19:00:00+00:00",
        }
    ]

    enriched = pubqueue.enrich_rows_with_sidecar(rows)

    assert enriched[0]["held_reason"] is None


# --- held_reason ("quiet") ----------------------------------------------


def _patch_quiet_policy(monkeypatch, hours):
    from gomoveshift.publishing.brand_policy import BrandPolicy

    monkeypatch.setattr(
        pubqueue,
        "load_brand_policy",
        lambda slug: BrandPolicy(quiet_after_alert_hours=hours),
    )


def test_enrich_rows_with_sidecar_held_reason_quiet_inside_the_window(tmp_path, monkeypatch):
    from gomoveshift.publishing.queue import _now_iso, log_response

    _patch_quiet_policy(monkeypatch, 6)
    conn = open_queue_db(tmp_path)
    log_response(
        conn, "youtube", state="SUPPRESSED", cap=0, action="cap_set", mode="live", now=_now_iso()
    )
    conn.close()
    rows = [
        {
            "project_name": "proj",
            "short_index": 1,
            "platform": "youtube",
            "status": "queued",
            "scheduled_at": None,
        }
    ]

    enriched = pubqueue.enrich_rows_with_sidecar(rows, config=_config(tmp_path))

    assert enriched[0]["held_reason"] == "quiet"


def test_enrich_rows_with_sidecar_held_reason_none_after_the_quiet_window(tmp_path, monkeypatch):
    from gomoveshift.publishing.queue import log_response

    _patch_quiet_policy(monkeypatch, 6)
    conn = open_queue_db(tmp_path)
    log_response(
        conn,
        "youtube",
        state="SUPPRESSED",
        cap=0,
        action="cap_set",
        mode="live",
        now="2020-01-01T00:00:00+00:00",
    )
    conn.close()
    rows = [
        {
            "project_name": "proj",
            "short_index": 1,
            "platform": "youtube",
            "status": "queued",
            "scheduled_at": None,
        }
    ]

    enriched = pubqueue.enrich_rows_with_sidecar(rows, config=_config(tmp_path))

    assert enriched[0]["held_reason"] is None


def test_enrich_rows_with_sidecar_held_reason_blackout_wins_over_quiet(tmp_path, monkeypatch):
    from datetime import date

    from gomoveshift.publishing.brand_policy import BrandPolicy
    from gomoveshift.publishing.queue import _now_iso, log_response

    monkeypatch.setattr(
        pubqueue,
        "load_brand_policy",
        lambda slug: BrandPolicy(blackout=(date(2026, 9, 5),), quiet_after_alert_hours=6),
    )
    conn = open_queue_db(tmp_path)
    log_response(
        conn, "youtube", state="SUPPRESSED", cap=0, action="cap_set", mode="live", now=_now_iso()
    )
    conn.close()
    rows = [
        {
            "project_name": "proj",
            "short_index": 1,
            "platform": "youtube",
            "status": "queued",
            "scheduled_at": "2026-09-05T19:00:00+00:00",
        }
    ]

    enriched = pubqueue.enrich_rows_with_sidecar(rows, config=_config(tmp_path))

    assert enriched[0]["held_reason"] == "blackout"


def test_enrich_rows_with_sidecar_held_reason_quiet_none_without_response_log_db(monkeypatch):
    _patch_quiet_policy(monkeypatch, 6)
    rows = [
        {
            "project_name": "proj",
            "short_index": 1,
            "platform": "youtube",
            "status": "queued",
            "scheduled_at": None,
        }
    ]

    enriched = pubqueue.enrich_rows_with_sidecar(rows)

    assert enriched[0]["held_reason"] is None


def test_enrich_rows_with_sidecar_held_reason_never_set_on_a_terminal_row(monkeypatch):
    from datetime import date

    from gomoveshift.publishing.brand_policy import BrandPolicy

    monkeypatch.setattr(
        pubqueue,
        "load_brand_policy",
        lambda slug: BrandPolicy(blackout=(date(2026, 9, 5),)),
    )
    rows = [
        {
            "project_name": "proj",
            "short_index": 1,
            "platform": "youtube",
            "status": "published",
            "scheduled_at": "2026-09-05T19:00:00+00:00",
        }
    ]

    enriched = pubqueue.enrich_rows_with_sidecar(rows)

    assert enriched[0]["held_reason"] is None


def test_enrich_rows_with_sidecar_matching_time_shows_no_discrepancy(tmp_path):
    sidecar_path = _write_sidecar(tmp_path, _short_record())
    rows = [
        {
            "project_name": "proj",
            "short_index": 1,
            "platform": "youtube",
            "sidecar_path": str(sidecar_path),
            "scheduled_at": _SCHEDULED.isoformat(),
        }
    ]

    enriched = pubqueue.enrich_rows_with_sidecar(rows)

    assert enriched[0]["scheduled_at_effective_local"] is not None
    assert enriched[0]["schedule_discrepancy"] is None


def test_enrich_rows_with_sidecar_disagreement_shows_row_time_and_discrepancy(tmp_path):
    sidecar_path = _write_sidecar(tmp_path, _short_record())
    row_time = datetime(2026, 9, 5, 20, 0, tzinfo=timezone.utc)
    rows = [
        {
            "project_name": "proj",
            "short_index": 1,
            "platform": "youtube",
            "sidecar_path": str(sidecar_path),
            "scheduled_at": row_time.isoformat(),
        }
    ]

    enriched = pubqueue.enrich_rows_with_sidecar(rows)

    assert "20:00" in enriched[0]["scheduled_at_effective_local"]
    disc = enriched[0]["schedule_discrepancy"]
    assert disc and ("19:00" in disc or "14:00" in disc)


def test_enrich_rows_with_sidecar_malformed_sets_read_error_not_no_schedule(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    rows = [
        {
            "project_name": "proj",
            "short_index": 1,
            "platform": "youtube",
            "sidecar_path": str(bad),
        }
    ]

    enriched = pubqueue.enrich_rows_with_sidecar(rows)

    assert enriched[0]["sidecar_read_error"] is True
    assert enriched[0]["scheduled_at_effective_local"] is None


def test_enrich_rows_with_sidecar_no_schedule_and_no_sidecar_is_distinct():
    rows = [{"project_name": "proj", "short_index": 1, "platform": "youtube"}]

    enriched = pubqueue.enrich_rows_with_sidecar(rows)

    assert enriched[0]["scheduled_at_effective_local"] is None
    assert enriched[0]["sidecar_read_error"] is False


def _insert_delivery(conn, *, scheduled_at=None, sidecar_path=None):
    from gomoveshift.publishing.queue import _now_iso

    cols = ["project_name", "short_index", "platform", "plan_hash", "status"]
    vals = ["proj", 1, "youtube", "h1", "queued"]
    if scheduled_at is not None:
        cols.append("scheduled_at")
        vals.append(scheduled_at)
    if sidecar_path is not None:
        cols.append("sidecar_path")
        vals.append(sidecar_path)
    cols += ["created_at", "updated_at"]
    vals += [_now_iso(), _now_iso()]
    with conn:
        conn.execute(
            f"INSERT INTO deliveries ({','.join(cols)}) VALUES ({','.join('?' * len(vals))})",  # noqa: S608
            vals,
        )


def test_publishing_get_renders_row_schedule_without_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))
    conn = open_queue_db(tmp_path)
    _insert_delivery(conn, scheduled_at="2026-09-05T19:00:00+00:00")
    conn.close()

    response = _client().get("/publishing")

    assert response.status_code == 200
    assert "2026-09-05" in response.text
    assert "no schedule recorded" not in response.text.lower()


def test_publishing_get_shows_could_not_read_sidecar_message(tmp_path, monkeypatch):
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))
    conn = open_queue_db(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    _insert_delivery(conn, sidecar_path=str(bad))
    conn.close()

    response = _client().get("/publishing")

    assert response.status_code == 200
    assert "could not read" in response.text.lower()
    assert "no schedule recorded" not in response.text.lower()


def test_publishing_get_shows_no_schedule_recorded_when_neither(tmp_path, monkeypatch):
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 1, "youtube", "h1")
    conn.close()

    response = _client().get("/publishing")

    assert response.status_code == 200
    assert "no schedule recorded" in response.text.lower()


def test_awaiting_manual_publish_sorts_first():
    states = ["published", "failed_permanent", "queued", "awaiting_manual_publish"]
    rows = [{"status": s, "id": i} for i, s in enumerate(states)]

    grouped = pubqueue.rows_by_state(rows)

    assert [status for status, _rows in grouped] == [
        "awaiting_manual_publish",
        "queued",
        "published",
        "failed_permanent",
    ]


# --- Per-delivery detail page ----------------------------------------

_TOKEN = "distinctivemarker_should_never_render_9f3a2"  # noqa: S105 - fixture token, not a secret


def _insert_full_delivery(conn, **over):
    from gomoveshift.publishing.queue import _now_iso

    fields = {
        "project_name": "proj",
        "short_index": 1,
        "platform": "youtube",
        "plan_hash": "h1",
        "status": "queued",
        "shadow": 0,
        **over,
    }
    now = _now_iso()
    cols = [*fields, "created_at", "updated_at"]
    vals = [*fields.values(), now, now]
    with conn:
        cursor = conn.execute(
            f"INSERT INTO deliveries ({','.join(cols)}) VALUES ({','.join('?' * len(vals))})",  # noqa: S608
            vals,
        )
        delivery_id = cursor.lastrowid
        cursor.close()
    return delivery_id


def _detail(tmp_path, monkeypatch, **over):
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))
    conn = open_queue_db(tmp_path)
    delivery_id = _insert_full_delivery(conn, **over)
    conn.close()
    return _client().get(f"/publishing/{delivery_id}"), delivery_id


def test_publishing_detail_route_404_unknown_and_200_known(tmp_path, monkeypatch):
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))
    open_queue_db(tmp_path).close()
    assert _client().get("/publishing/999").status_code == 404
    response, _ = _detail(tmp_path, monkeypatch)
    assert response.status_code == 200
    assert "proj" in response.text


def test_detail_shows_privacy_report_with_no_sidecar(tmp_path, monkeypatch):
    # No sidecar to read from -- the page must say so, never render
    # a blank cell an operator could mistake for "nothing to worry about".
    response, _ = _detail(tmp_path, monkeypatch)
    assert response.status_code == 200
    assert "privacy report" in response.text.lower()
    assert "no sidecar" in response.text.lower()


def test_detail_shows_privacy_report_from_sidecar(tmp_path, monkeypatch):
    sidecar_path = _write_sidecar(tmp_path, _short_record())
    response, _ = _detail(tmp_path, monkeypatch, sidecar_path=str(sidecar_path))
    assert response.status_code == 200
    from gomoveshift.console.library import _PRIVACY_STATUS_VIDEO_NOT_TRACKED

    assert _PRIVACY_STATUS_VIDEO_NOT_TRACKED in response.text


def test_detail_mapped_error_class_shows_sentence_never_raw_detail(tmp_path, monkeypatch):
    response, _ = _detail(
        tmp_path,
        monkeypatch,
        status="failed_permanent",
        error_class="PermanentDeliveryError",
        error_detail=_TOKEN,
    )
    assert response.status_code == 200
    assert _TOKEN not in response.text
    assert "rejected" in response.text.lower()


def test_detail_unmapped_error_class_is_honest_and_logs_raw_detail(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))
    conn = open_queue_db(tmp_path)
    delivery_id = _insert_full_delivery(
        conn, status="failed_permanent", error_class="SomeBrandNewSDKError", error_detail=_TOKEN
    )
    conn.close()
    with caplog.at_level("WARNING"):
        response = _client().get(f"/publishing/{delivery_id}")
    assert response.status_code == 200
    assert _TOKEN not in response.text
    assert "no description yet" in response.text.lower()
    assert _TOKEN in caplog.text


_PUBLISHED_AT = "2026-09-05T19:00:00+00:00"


def test_detail_published_row_shows_source_label_never_the_word_verified(tmp_path, monkeypatch):
    response, _ = _detail(
        tmp_path,
        monkeypatch,
        status="published",
        published_at=_PUBLISHED_AT,
        published_at_source="platform_confirmed",
    )
    assert "platform confirmed" in response.text.lower()
    assert "verified" not in response.text.lower()


def test_detail_legacy_published_row_reads_unknown(tmp_path, monkeypatch):
    response, _ = _detail(tmp_path, monkeypatch, status="published", published_at=_PUBLISHED_AT)
    assert "unknown (legacy)" in response.text.lower()
    # "unverified" is the honest Status label for this row; a bare "verified" would be a claim.
    assert not re.search(r"(?<!un)verified", response.text.lower())


def test_detail_shadow_row_is_marked_not_a_real_publication(tmp_path, monkeypatch):
    response, _ = _detail(tmp_path, monkeypatch, shadow=1)
    assert "not a real publication" in response.text.lower()


def test_detail_lease_states_render_differently(tmp_path, monkeypatch):
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))
    conn = open_queue_db(tmp_path)
    no_lease_id = _insert_full_delivery(conn, short_index=1)
    live_id = _insert_full_delivery(
        conn, short_index=2, lease_owner="worker-a", lease_expires_at="2099-01-01T00:00:00+00:00"
    )
    expired_id = _insert_full_delivery(
        conn, short_index=3, lease_owner="worker-b", lease_expires_at="2000-01-01T00:00:00+00:00"
    )
    conn.close()

    no_lease = _client().get(f"/publishing/{no_lease_id}").text.lower()
    live = _client().get(f"/publishing/{live_id}").text.lower()
    expired = _client().get(f"/publishing/{expired_id}").text.lower()

    assert "no lease" in no_lease
    assert "currently holds" in live and "worker-a" in live
    assert "expired" in expired
    assert live != no_lease and expired != no_lease and expired != live


def test_detail_permalink_present_links_it(tmp_path, monkeypatch):
    response, _ = _detail(
        tmp_path,
        monkeypatch,
        status="published",
        permalink="https://example.com/post/1",
        platform_post_id="1",
    )
    assert 'href="https://example.com/post/1"' in response.text


def test_detail_no_permalink_no_post_id_says_no_confirmation(tmp_path, monkeypatch):
    response, _ = _detail(tmp_path, monkeypatch)
    assert "no platform confirmation" in response.text.lower()


def test_publishing_list_links_each_row_to_its_detail_page(tmp_path, monkeypatch):
    _, delivery_id = _detail(tmp_path, monkeypatch)
    response = _client().get("/publishing")
    assert response.status_code == 200
    assert f"/publishing/{delivery_id}" in response.text


def test_publish_now_does_not_promise_a_confirmation_step_it_does_not_have(tmp_path, monkeypatch):
    """``POST /publishing/drain`` launches the job on submit -- there is no
    second step. A hint claiming otherwise would tell an operator the button
    is safe when it publishes irreversibly to a platform."""
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))

    body = _client().get("/publishing").text

    assert "there is no confirmation" in body
    assert "without confirming below" not in body


def _row(tmp_path, delivery_id):
    conn = open_queue_db(tmp_path)
    try:
        return dict(conn.execute("SELECT * FROM deliveries WHERE id=?", (delivery_id,)).fetchone())
    finally:
        conn.close()


def test_cancelling_a_queued_delivery_makes_it_cancelled(tmp_path, monkeypatch):
    """The queue has `cancel_delivery`; the console had no way
    to reach it, so an operator watching a delivery they no longer want had
    nothing to do but wait for it to publish."""
    _, delivery_id = _detail(tmp_path, monkeypatch)

    response = _client().post(f"/publishing/{delivery_id}/cancel", follow_redirects=False)

    assert response.status_code == 303
    assert _row(tmp_path, delivery_id)["status"] == "cancelled"


def test_cancelling_an_already_terminal_delivery_is_refused_not_silent(tmp_path, monkeypatch):
    """A redirect back to a page that looks unchanged is indistinguishable
    from a cancel that worked. The refusal has to be visible."""
    _, delivery_id = _detail(tmp_path, monkeypatch, status="published")

    response = _client().post(f"/publishing/{delivery_id}/cancel", follow_redirects=False)

    assert response.status_code == 409
    assert _row(tmp_path, delivery_id)["status"] == "published"


def test_cancelling_a_delivery_a_live_worker_holds_is_refused(tmp_path, monkeypatch):
    """A live lease may already be mid platform-API call. Cancelling under it
    would leave the queue saying "cancelled" about a post that went out."""
    _, delivery_id = _detail(
        tmp_path,
        monkeypatch,
        lease_owner="worker-a:1:x",
        lease_expires_at="2999-01-01T00:00:00+00:00",
    )

    response = _client().post(f"/publishing/{delivery_id}/cancel", follow_redirects=False)

    assert response.status_code == 409
    assert _row(tmp_path, delivery_id)["status"] == "queued"


def test_cancelling_an_unknown_delivery_is_a_404_not_a_409(tmp_path, monkeypatch):
    """A 409 would tell an operator the delivery exists and is in some
    unsuitable state; it does not exist at all."""
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))
    open_queue_db(tmp_path).close()
    assert _client().post("/publishing/999/cancel", follow_redirects=False).status_code == 404


def test_retrying_a_backing_off_delivery_clears_its_next_attempt(tmp_path, monkeypatch):
    _, delivery_id = _detail(
        tmp_path,
        monkeypatch,
        status="failed_transient",
        next_attempt_at="2999-01-01T00:00:00+00:00",
    )

    response = _client().post(f"/publishing/{delivery_id}/retry", follow_redirects=False)

    assert response.status_code == 303
    assert _row(tmp_path, delivery_id)["next_attempt_at"] is None


def test_retry_never_moves_the_attempt_count(tmp_path, monkeypatch):
    """Resetting it would let a delivery dodge the max-attempts ceiling
    forever through repeated manual retries; incrementing it would burn an
    attempt against a call that has not happened. The queue owns that rule --
    this pins that the console did not quietly add its own."""
    _, delivery_id = _detail(
        tmp_path,
        monkeypatch,
        status="failed_transient",
        attempt_count=3,
        next_attempt_at="2999-01-01T00:00:00+00:00",
    )

    _client().post(f"/publishing/{delivery_id}/retry", follow_redirects=False)

    assert _row(tmp_path, delivery_id)["attempt_count"] == 3


def test_retrying_a_terminal_delivery_is_refused(tmp_path, monkeypatch):
    _, delivery_id = _detail(tmp_path, monkeypatch, status="cancelled")
    assert (
        _client().post(f"/publishing/{delivery_id}/retry", follow_redirects=False).status_code
        == 409
    )


def test_the_detail_page_offers_the_controls_only_where_they_apply(tmp_path, monkeypatch):
    """A button that always 409s is worse than no button: it tells an
    operator an action is available and then refuses it."""
    live, _ = _detail(tmp_path, monkeypatch)
    terminal, _ = _detail(tmp_path, monkeypatch, project_name="other", status="published")

    assert "/cancel" in live.text
    assert "/cancel" not in terminal.text
    assert "/retry" not in terminal.text


def test_a_javascript_permalink_is_shown_but_never_made_clickable(tmp_path, monkeypatch):
    """``permalink`` is written by the platform adapters from API responses and
    by an operator typing into the manual-confirmation form. Rendering it
    straight into an ``href`` means a ``javascript:`` value executes in the
    console's own origin on one click. Jinja's autoescaping does not help: it
    escapes the quotes, and ``javascript:alert(1)`` needs none."""
    response, _ = _detail(
        tmp_path,
        monkeypatch,
        status="published",
        permalink="javascript:alert(document.cookie)",
        platform_post_id="1",
    )
    assert 'href="javascript:' not in response.text.lower(), (
        "a javascript: permalink was rendered as a live link"
    )
    # Shown, not hidden: an operator has to be able to SEE what was recorded in
    # order to fix it, and silently dropping it would hide a hostile value.
    assert "alert(document.cookie)" in response.text


def test_a_data_url_permalink_is_not_made_clickable(tmp_path, monkeypatch):
    response, _ = _detail(
        tmp_path,
        monkeypatch,
        status="published",
        permalink="data:text/html;base64,PHNjcmlwdD4=",
        platform_post_id="1",
    )
    assert 'href="data:' not in response.text.lower()


def test_a_schemeless_permalink_is_not_made_clickable(tmp_path, monkeypatch):
    """``example.com/post/1`` in an href is a RELATIVE path -- it would point
    back into the console, not at the platform. A link that goes somewhere
    other than where it reads is worse than no link."""
    response, _ = _detail(tmp_path, monkeypatch, status="published", permalink="example.com/post/1")
    assert 'href="example.com' not in response.text.lower()
    assert "example.com/post/1" in response.text


def test_the_queue_list_applies_the_same_rule_as_the_detail_page(tmp_path, monkeypatch):
    """Both templates render the same column from the same enriched rows, so
    the guard belongs in the enrichment, not in one template."""
    _detail(
        tmp_path,
        monkeypatch,
        status="published",
        permalink="javascript:alert(1)",
        platform_post_id="1",
    )
    response = _client().get("/publishing")
    assert 'href="javascript:' not in response.text.lower()


def test_an_ordinary_https_permalink_is_still_a_link(tmp_path, monkeypatch):
    """The accept-case. A guard with no accept-case is a guard that breaks the
    feature it guards."""
    response, _ = _detail(
        tmp_path,
        monkeypatch,
        status="published",
        permalink="https://www.tiktok.com/@a/video/123",
        platform_post_id="1",
    )
    assert 'href="https://www.tiktok.com/@a/video/123"' in response.text


def test_an_http_permalink_is_still_a_link(tmp_path, monkeypatch):
    response, _ = _detail(
        tmp_path, monkeypatch, status="published", permalink="http://example.com/p/1"
    )
    assert 'href="http://example.com/p/1"' in response.text


def _register_output_dir(tmp_path, output_dir) -> None:
    """A job row is the console's record that it launched a run there --
    the same evidence ``_console_output_dirs`` reads for Drives and Shorts."""
    store = jobs.JobStore(tmp_path / "jobs.sqlite3")
    store.create("drive", ["gms", "drive", "d", "--output-dir", str(output_dir)], log_path="")


def _launch_recorder(monkeypatch):
    launched: dict = {}
    monkeypatch.setattr(
        jobs,
        "launch",
        lambda store, name, argv, log_dir, env=None: launched.setdefault(name, argv) or 1,
    )
    return launched


def test_enqueue_accepts_a_state_path_inside_a_run_this_console_launched(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))
    launched = _launch_recorder(monkeypatch)
    output_dir = tmp_path / "out"
    (output_dir / "proj").mkdir(parents=True)
    state = output_dir / "proj" / "pipeline_state.json"
    state.write_text("{}", encoding="utf-8")
    _register_output_dir(tmp_path, output_dir)

    resp = _client().post(
        "/publishing/enqueue",
        data={"state_path": str(state), "output_dir": str(output_dir)},
        follow_redirects=False,
    )

    assert resp.status_code == 303
    assert launched["enqueue-plan"][3:] == [
        "enqueue-plan",
        str(state.resolve()),
        "--apply",
        "--output-dir",
        str(output_dir.resolve()),
    ]


def test_enqueue_refuses_a_state_path_from_another_tenants_run(tmp_path, monkeypatch):
    """The plan file genuinely exists and parses -- it simply belongs to a run
    this console never launched. Typing its path must not enqueue someone
    else's content into this queue, and the refusal must not echo the path
    back (that would make the 404 a filesystem oracle)."""
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))
    launched = _launch_recorder(monkeypatch)
    mine = tmp_path / "out"
    mine.mkdir()
    _register_output_dir(tmp_path, mine)
    foreign = tmp_path / "other-tenant-out"
    foreign.mkdir()
    foreign_state = foreign / "pipeline_state.json"
    foreign_state.write_text("{}", encoding="utf-8")

    resp = _client().post(
        "/publishing/enqueue", data={"state_path": str(foreign_state)}, follow_redirects=False
    )

    assert resp.status_code == 404
    assert str(foreign_state) not in resp.text
    assert launched == {}


def test_enqueue_refuses_a_dotdot_escape_out_of_an_allowed_run(tmp_path, monkeypatch):
    """Containment compares RESOLVED paths, so a path that merely starts with
    an allowed directory does not pass."""
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))
    launched = _launch_recorder(monkeypatch)
    mine = tmp_path / "out"
    mine.mkdir()
    _register_output_dir(tmp_path, mine)
    foreign = tmp_path / "other-tenant-out"
    foreign.mkdir()
    (foreign / "pipeline_state.json").write_text("{}", encoding="utf-8")

    resp = _client().post(
        "/publishing/enqueue",
        data={"state_path": f"{mine}/../other-tenant-out/pipeline_state.json"},
        follow_redirects=False,
    )

    assert resp.status_code == 404
    assert launched == {}


def test_enqueue_refuses_a_foreign_output_dir_even_with_an_allowed_state_path(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))
    launched = _launch_recorder(monkeypatch)
    mine = tmp_path / "out"
    mine.mkdir()
    state = mine / "pipeline_state.json"
    state.write_text("{}", encoding="utf-8")
    _register_output_dir(tmp_path, mine)
    foreign = tmp_path / "other-tenant-out"
    foreign.mkdir()

    resp = _client().post(
        "/publishing/enqueue",
        data={"state_path": str(state), "output_dir": str(foreign)},
        follow_redirects=False,
    )

    assert resp.status_code == 404
    assert launched == {}


def test_drain_refuses_a_sidecar_dir_from_another_tenants_run(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))
    launched = _launch_recorder(monkeypatch)
    mine = tmp_path / "out"
    mine.mkdir()
    _register_output_dir(tmp_path, mine)
    foreign = tmp_path / "other-tenant-out"
    foreign.mkdir()

    resp = _client().post(
        "/publishing/drain", data={"sidecar_dir": str(foreign)}, follow_redirects=False
    )

    assert resp.status_code == 404
    assert str(foreign) not in resp.text
    assert launched == {}


def test_drain_accepts_a_sidecar_dir_this_console_launched(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))
    launched = _launch_recorder(monkeypatch)
    mine = tmp_path / "out"
    mine.mkdir()
    _register_output_dir(tmp_path, mine)

    resp = _client().post(
        "/publishing/drain", data={"sidecar_dir": str(mine)}, follow_redirects=False
    )

    assert resp.status_code == 303
    assert launched["dispatch-queue"][3:] == [
        "dispatch-queue",
        "--apply",
        "--sidecar-dir",
        str(mine.resolve()),
    ]


# --- The detail page tells the truth about a paused platform ----------


def test_detail_interrupted_attempt_reads_honestly_and_logs_no_warning(
    tmp_path, monkeypatch, caplog
):
    """``InterruptedAttempt`` is a class this queue writes itself
    (``publishing/dispatch.py``), so it must carry a sentence -- not the
    unmapped fallback plus a WARNING on every single page view."""
    with caplog.at_level("WARNING"):
        response, _ = _detail(
            tmp_path, monkeypatch, status="queued", error_class="InterruptedAttempt"
        )

    assert response.status_code == 200
    assert "no description yet" not in response.text.lower()
    assert "InterruptedAttempt" not in caplog.text


def _paused_detail(tmp_path, monkeypatch, platform, **over):
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))
    conn = open_queue_db(tmp_path)
    delivery_id = _insert_full_delivery(conn, platform=platform, **over)
    pause_platform(conn, platform)
    conn.close()
    return _client().get(f"/publishing/{delivery_id}"), delivery_id


def test_delivery_by_id_reports_its_platform_pause_state(tmp_path, monkeypatch):
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))
    conn = open_queue_db(tmp_path)
    paused_id = _insert_full_delivery(conn, platform="youtube")
    running_id = _insert_full_delivery(conn, platform="instagram")
    pause_platform(conn, "youtube")
    conn.close()

    assert pubqueue.delivery_by_id(paused_id)["platform_paused"] is True
    assert pubqueue.delivery_by_id(running_id)["platform_paused"] is False


def test_detail_paused_platform_does_not_promise_an_automatic_retry(tmp_path, monkeypatch):
    response, _ = _paused_detail(tmp_path, monkeypatch, "youtube", status="queued")

    assert response.status_code == 200
    assert "expected to retry automatically" not in response.text
    assert "next drain picks this up" not in response.text
    assert "paused" in response.text.lower()


def test_detail_running_platform_still_promises_the_retry(tmp_path, monkeypatch):
    response, _ = _detail(tmp_path, monkeypatch, platform="youtube", status="queued")

    assert response.status_code == 200
    assert "expected to retry automatically" in response.text
    assert "next drain picks this up" in response.text


# --- pack mode: a packed TikTok short's time lives in its file name ---------


def _parked_tiktok(tmp_path):
    conn = open_queue_db(tmp_path)
    delivery_id = enqueue(
        conn, "proj", 3, "tiktok", "HASH", scheduled_at="2026-10-03T18:00:00+00:00"
    )
    transition(conn, delivery_id, "queued", "awaiting_manual_publish")
    row = dict(conn.execute("SELECT * FROM deliveries WHERE id=?", (delivery_id,)).fetchone())
    conn.close()
    return delivery_id, row


def test_pack_mode_offers_no_reschedule_for_a_packed_tiktok_row(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_TIKTOK_MODE", "pack")
    _, row = _parked_tiktok(tmp_path)
    assert "reschedule" not in pubqueue.operator_actions_for(row)


def test_inbox_mode_still_offers_reschedule_for_a_parked_tiktok_row(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_TIKTOK_MODE", "inbox")
    _, row = _parked_tiktok(tmp_path)
    assert "reschedule" in pubqueue.operator_actions_for(row)


def test_pack_mode_refuses_to_reschedule_a_packed_tiktok_row(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_TIKTOK_MODE", "pack")
    delivery_id, _ = _parked_tiktok(tmp_path)
    db_path = tmp_path / "social_queue.sqlite3"

    with pytest.raises(ValueError, match="TikTok Studio"):
        pubqueue.reschedule(db_path, delivery_id, "2026-10-04T18:00:00+00:00")

    conn = open_queue_db(tmp_path)
    after = conn.execute(
        "SELECT scheduled_at FROM deliveries WHERE id=?", (delivery_id,)
    ).fetchone()
    conn.close()
    assert after[0] == "2026-10-03T18:00:00+00:00"
