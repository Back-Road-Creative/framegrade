"""Tests for ``gomoveshift publishing-status`` (cli/handlers/pub_status.py).

Ground-truth status command: credential-file PRESENCE only, never
contents, plus the delivery queue's row counts by state.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from gomoveshift.cli.handlers.pub_status import (
    cmd_publishing_status,
    credential_presence,
    queue_state_counts,
    validate_restored_state,
)
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
from gomoveshift.publishing.queue import enqueue, open_queue_db, transition


def _config(tmp_path):
    creds_dir = tmp_path / "credentials"
    creds_dir.mkdir()
    return SimpleNamespace(
        seo=SimpleNamespace(credentials_dir=str(creds_dir), storage_path=str(tmp_path))
    )


def test_credential_presence_all_absent(tmp_path):
    creds_dir = tmp_path / "credentials"
    creds_dir.mkdir()
    presence = credential_presence(str(creds_dir))
    assert presence == {
        "youtube": False,
        "instagram": False,
        "facebook": False,
        "tiktok": False,
    }


def test_credential_presence_detects_existing_files_never_reads_them(tmp_path):
    creds_dir = tmp_path / "credentials"
    creds_dir.mkdir()
    # A file whose CONTENT is garbage — presence must not require decrypting
    # or parsing it, only that the path exists.
    (tmp_path / "youtube_upload_token.json.enc").write_text("not sops output at all")
    (tmp_path / "meta_publishing.json.enc").write_text("garbage")

    presence = credential_presence(str(creds_dir))
    assert presence == {
        "youtube": True,
        "instagram": True,  # shares meta_publishing.json.enc with facebook
        "facebook": True,
        "tiktok": False,
    }


def test_queue_state_counts_empty_queue(tmp_path):
    conn = open_queue_db(tmp_path)
    try:
        assert queue_state_counts(conn) == {}
    finally:
        conn.close()


def test_queue_state_counts_groups_by_status(tmp_path):
    conn = open_queue_db(tmp_path)
    try:
        enqueue(conn, "proj", 1, "youtube", "h1")
        enqueue(conn, "proj", 2, "youtube", "h1")
        enqueue(conn, "proj", 3, "instagram", "h1")
        assert queue_state_counts(conn) == {"queued": 3}
    finally:
        conn.close()


def test_cmd_publishing_status_prints_presence_and_counts_and_returns_0(tmp_path, capsys):
    config = _config(tmp_path)
    (tmp_path / "tiktok_publishing.json.enc").write_text("garbage")
    conn = open_queue_db(config.seo.storage_path)
    enqueue(conn, "proj", 1, "tiktok", "h1")
    conn.close()

    rc = cmd_publishing_status(SimpleNamespace(), config)

    assert rc == 0
    out = capsys.readouterr().out
    assert "tiktok: present" in out
    assert "youtube: ABSENT" in out
    assert "queued: 1" in out


def test_cmd_publishing_status_reports_empty_queue(tmp_path, capsys):
    config = _config(tmp_path)

    rc = cmd_publishing_status(SimpleNamespace(), config)

    assert rc == 0
    assert "(empty)" in capsys.readouterr().out


_SCHEDULED = datetime(2026, 9, 5, 19, 0, tzinfo=timezone.utc)


def _sidecar(tmp_path, *, project_name="proj", short_index=1) -> Path:
    """A real publish-plan sidecar on disk, written by the same serializer
    a drive run uses -- a hand-rolled dict would validate against a shape
    nothing produces."""
    record = ShortPublishRecord(
        project_name=project_name,
        lf_index=0,
        short_index=short_index,
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
    path = tmp_path / f"{project_name}_publish_plan.json"
    path.write_text(_serialize_publish_plan(PublishPlan(shorts=(record,))), encoding="utf-8")
    return path


def _restored(tmp_path, **enqueue_kwargs) -> tuple:
    """A queue holding one non-terminal youtube delivery, plus the YouTube
    credential file present -- the shape a healthy restore has."""
    (tmp_path / "youtube_upload_token.json.enc").write_text("garbage", encoding="utf-8")
    conn = open_queue_db(tmp_path)
    delivery_id = enqueue(conn, "proj", 1, "youtube", "HASH1", **enqueue_kwargs)
    return conn, delivery_id


def _kinds(problems) -> list:
    return [p.kind for p in problems]


def test_a_clean_restore_reports_no_problems(tmp_path):
    conn, _ = _restored(tmp_path, sidecar_path=str(_sidecar(tmp_path)))
    try:
        assert validate_restored_state(conn, str(tmp_path / "credentials")) == []
    finally:
        conn.close()


def test_a_delivery_whose_sidecar_was_not_restored_is_named(tmp_path):
    """The queue DB and the sidecars are separate files. Restore one without
    the other and the row can never publish -- but nothing says so until a
    drain pass fails on it, long after the restore was called good."""
    conn, delivery_id = _restored(tmp_path, sidecar_path=str(tmp_path / "gone_publish_plan.json"))
    try:
        problems = validate_restored_state(conn, str(tmp_path / "credentials"))
    finally:
        conn.close()
    assert _kinds(problems) == ["sidecar_unreadable"]
    assert problems[0].delivery_id == delivery_id


def test_a_delivery_with_no_sidecar_path_at_all_is_named_separately(tmp_path):
    conn, _ = _restored(tmp_path)
    try:
        problems = validate_restored_state(conn, str(tmp_path / "credentials"))
    finally:
        conn.close()
    assert _kinds(problems) == ["sidecar_path_unset"]


def test_a_sidecar_that_does_not_carry_this_short_is_a_problem_not_a_pass(tmp_path):
    """A readable sidecar is not the same as the RIGHT sidecar: restoring a
    different run's plan file leaves every row pointing at a document that
    does not describe it."""
    conn, _ = _restored(tmp_path, sidecar_path=str(_sidecar(tmp_path, short_index=7)))
    try:
        problems = validate_restored_state(conn, str(tmp_path / "credentials"))
    finally:
        conn.close()
    assert _kinds(problems) == ["short_not_in_sidecar"]


def test_a_platform_with_no_credential_file_is_named_before_the_first_drain(tmp_path):
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 1, "tiktok", "HASH1", sidecar_path=str(_sidecar(tmp_path)))
    try:
        problems = validate_restored_state(conn, str(tmp_path / "credentials"))
    finally:
        conn.close()
    assert "credential_absent" in _kinds(problems)


def test_a_lease_from_the_instance_that_died_is_reported(tmp_path):
    """A restored row can carry a lease held by a worker that no longer
    exists. Until it expires, ``_has_live_foreign_lease`` refuses the
    operator's cancel and reschedule, and the row reads as busy."""
    conn, delivery_id = _restored(tmp_path, sidecar_path=str(_sidecar(tmp_path)))
    conn.execute(
        "UPDATE deliveries SET lease_owner=?, lease_expires_at=? WHERE id=?",
        ("dead-host:123:abc", "2999-01-01T00:00:00+00:00", delivery_id),
    )
    conn.commit()
    try:
        problems = validate_restored_state(conn, str(tmp_path / "credentials"))
    finally:
        conn.close()
    assert _kinds(problems) == ["stale_lease"]


def test_terminal_deliveries_are_not_validated(tmp_path):
    """A published or cancelled delivery owes no further work, so a sidecar
    it no longer needs is not a restore defect. Reporting it would bury the
    rows that DO still need one."""
    conn, delivery_id = _restored(tmp_path)
    # "published" demands a provenance for published_at -- the queue refuses
    # an unlabelled receipt, so the terminal row has to be made the real way.
    transition(conn, delivery_id, "queued", "published", published_at_source="platform_confirmed")
    try:
        assert validate_restored_state(conn, str(tmp_path / "credentials")) == []
    finally:
        conn.close()


def test_validation_changes_nothing_in_the_queue(tmp_path):
    """ "With dispatch disabled" is only a claim if the validator can write.
    Compare every column of every row before and after."""
    conn, _ = _restored(tmp_path, sidecar_path=str(tmp_path / "gone_publish_plan.json"))
    try:
        before = [dict(r) for r in conn.execute("SELECT * FROM deliveries").fetchall()]
        validate_restored_state(conn, str(tmp_path / "credentials"))
        after = [dict(r) for r in conn.execute("SELECT * FROM deliveries").fetchall()]
    finally:
        conn.close()
    assert before == after


def test_validate_flag_exits_nonzero_on_problems_and_zero_when_clean(tmp_path, capsys):
    config = _config(tmp_path)
    (tmp_path / "youtube_upload_token.json.enc").write_text("garbage", encoding="utf-8")
    conn = open_queue_db(config.seo.storage_path)
    delivery_id = enqueue(conn, "proj", 1, "youtube", "HASH1")
    conn.close()

    assert cmd_publishing_status(SimpleNamespace(validate=True), config) == 1
    out = capsys.readouterr().out
    assert "sidecar_path_unset" in out
    assert str(delivery_id) in out

    conn = open_queue_db(config.seo.storage_path)
    conn.execute(
        "UPDATE deliveries SET sidecar_path=? WHERE id=?", (str(_sidecar(tmp_path)), delivery_id)
    )
    conn.commit()
    conn.close()

    assert cmd_publishing_status(SimpleNamespace(validate=True), config) == 0
    assert "no problems" in capsys.readouterr().out.lower()


def test_without_the_flag_the_command_is_the_unchanged_report(tmp_path, capsys):
    """The plain command stays a report: a broken restore must not start
    failing a status check that never claimed to validate anything."""
    config = _config(tmp_path)
    conn = open_queue_db(config.seo.storage_path)
    enqueue(conn, "proj", 1, "youtube", "HASH1")
    conn.close()

    assert cmd_publishing_status(SimpleNamespace(), config) == 0
    assert "queued: 1" in capsys.readouterr().out
