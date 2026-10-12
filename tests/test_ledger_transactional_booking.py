"""Transactional booking: ledger entry exists ⟺ upload confirmed."""

import json
import logging
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from gomoveshift.config import Config
from gomoveshift.pipeline import content_ledger as cl
from gomoveshift.pipeline.drive_phases import external_systems_coordinator as esc
from gomoveshift.pipeline.drive_phases import phase_d

from _state_helpers import make_state

PUBLISH_AT = "2026-06-14T18:00:00+00:00"
RECORD = {"video_id": "vid123", "publish_at": PUBLISH_AT, "title": "Driving the Conor Pass"}
# A REAL PipelineState — it declares neither project_base nor project_name,
# so booking a ledger entry from it must go through the caller-supplied
# project_base argument, never a getattr guess on state.
STATE = make_state(region="ireland")
LEDGER = ".content_ledger.json"


def _cfg(path):
    return SimpleNamespace(
        seo=SimpleNamespace(storage_path=str(path)),
        drive=SimpleNamespace(publish_tiktok=True),
    )


def _post(i):
    blank = dict.fromkeys(["first_comment", "youtube_tags", "pinterest_board"], "")
    return SimpleNamespace(
        short_index=i, caption=f"S{i}", youtube_title=f"S{i}", tiktok_title=f"S{i}", **blank
    )


def _slot(i):  # distinct slots: 08:00, 16:00
    return SimpleNamespace(short_index=i, date="2026-06-21", time=f"{8 * i:02d}:00:00")


def _emit_csv(tmp_path, *, upload_manifests, skip_drive_upload):
    return phase_d._emit_consolidated_shorts_csv(
        output_base=tmp_path,
        project_base="sc",
        drive_plan=SimpleNamespace(regular_lfs=[SimpleNamespace(index=1)]),
        social_posts_per_lf={1: [_post(1), _post(2)]},
        upload_manifests=upload_manifests,
        drive_schedule=SimpleNamespace(shorts_slots=[_slot(1), _slot(2)]),
        youtube_uploads={1: {"video_id": "vid1"}},
        per_lf_trim_ratio={1: 1.0},
        per_lf_recording_date={1: "2025-09-14"},
        skip_drive_upload=skip_drive_upload,
        config=_cfg(tmp_path),
        region="ireland",
    )


def test_skip_upload_books_nothing(tmp_path):
    esc._upload_lfs_to_youtube(
        drive_plan=SimpleNamespace(regular_lfs=[SimpleNamespace(index=1)]),
        regular_lf_paths=["lf.mp4"],
        metadata_per_lf={},
        drive_schedule=None,
        youtube_uploads={},
        state=STATE,
        output_base=tmp_path,
        config=_cfg(tmp_path),
        save_state=lambda s: None,
        skip_youtube_upload=True,
        youtube_upload_enabled=True,
    )
    assert _emit_csv(tmp_path, upload_manifests={}, skip_drive_upload=True) is None  # deferred
    assert not (tmp_path / LEDGER).exists()


def test_confirmed_upload_books_identified_entry_with_real_slot(tmp_path):
    esc._book_lf_upload_in_ledger(record=RECORD, state=STATE, config=_cfg(tmp_path))

    entries = cl.ContentLedger.load(tmp_path / LEDGER).entries
    assert len(entries) == 1
    e = entries[0]
    expected = datetime.fromisoformat(PUBLISH_AT).astimezone(cl.EST)
    assert (e.date, e.time) == (expected.strftime("%Y-%m-%d"), expected.strftime("%H:%M:%S"))
    assert (e.video_id, e.region, e.content_type) == ("vid123", "ireland", "longform")
    assert e.title == "Driving the Conor Pass"


def test_book_lf_upload_uses_project_base(tmp_path):
    esc._book_lf_upload_in_ledger(
        record=RECORD, state=STATE, config=_cfg(tmp_path), project_base="sample-day1"
    )
    e = cl.ContentLedger.load(tmp_path / LEDGER).entries[0]
    assert e.project_name == "sample-day1"


def test_book_lf_upload_missing_project_base_warns_but_still_books(tmp_path, caplog):
    """Regression pin: a REAL PipelineState carries neither project_base nor
    project_name (see STATE above), so a caller that forgets to pass
    project_base must not silently ship an anonymous ledger row — it warns,
    naming the video_id, and the (irreversible) upload is still booked."""
    with caplog.at_level(logging.WARNING):
        esc._book_lf_upload_in_ledger(record=RECORD, state=STATE, config=_cfg(tmp_path))

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("vid123" in r.getMessage() for r in warnings), (
        f"expected a WARNING naming video_id=vid123, got: {[r.getMessage() for r in warnings]}"
    )
    entries = cl.ContentLedger.load(tmp_path / LEDGER).entries
    assert len(entries) == 1
    assert entries[0].project_name == ""


def test_upload_lfs_forwards_project_base_to_both_ledger_bookings(tmp_path):
    """project_base threads to BOTH booking call sites the LF loop drives:
    the regular-LF booking and the superlong-parts booking it runs after."""
    part_path = tmp_path / "sl_01.mp4"
    part_path.write_bytes(b"x")
    state = make_state(
        region="usa",
        superlong_parts={
            "2026-03-01": {
                "part_count": 1,
                "records": [
                    {
                        "path": str(part_path),
                        "part_index": 1,
                        "title": "Superlong Part",
                        "description": "d",
                        "location": "Asheville, North Carolina",
                    }
                ],
            }
        },
    )
    drive_plan = SimpleNamespace(regular_lfs=[SimpleNamespace(index=1, total_duration=1800.0)])
    metadata = {1: {"title": "T1", "description": "d", "tags": []}}
    sched = SimpleNamespace(lf_slots=[SimpleNamespace(datetime_est="2026-03-15T11:00:00")])

    def fake_upload_lf(*, yt, state, lf, **kwargs):
        return {"video_id": "vid_1", "publish_at": "2030-06-14T18:00:00+00:00", "title": "T1"}

    yt_instance = MagicMock()
    yt_instance.upload_if_needed.return_value = {
        "video_id": "sl_vid_1",
        "publish_at": "2030-06-20T18:00:00+00:00",
    }

    booked = []

    def fake_book(*, record, state, config, project_base=None, lf_index=None):
        booked.append((record.get("video_id"), project_base))

    config = Config()
    config.seo.storage_path = str(tmp_path)

    with (
        patch.object(
            esc,
            "_resolve_youtube_uploader_class",
            return_value=MagicMock(return_value=yt_instance),
        ),
        patch.object(esc, "_upload_lf_to_youtube", side_effect=fake_upload_lf),
        patch.object(esc, "_book_lf_upload_in_ledger", side_effect=fake_book),
    ):
        esc._upload_lfs_to_youtube(
            drive_plan=drive_plan,
            regular_lf_paths=["/v/lf1.mp4"],
            metadata_per_lf=metadata,
            drive_schedule=sched,
            youtube_uploads={},
            state=state,
            output_base=tmp_path,
            config=config,
            save_state=lambda s: None,
            skip_youtube_upload=False,
            youtube_upload_enabled=True,
            project_base="sample-day1",
        )

    assert booked == [("vid_1", "sample-day1"), ("sl_vid_1", "sample-day1")]


def test_resume_does_not_double_book(tmp_path):
    for _ in range(2):  # second call = resume replaying the same record
        esc._book_lf_upload_in_ledger(record=RECORD, state=STATE, config=_cfg(tmp_path))
    assert len(cl.ContentLedger.load(tmp_path / LEDGER).entries) == 1


def test_old_format_ledger_without_identity_fields_loads(tmp_path):
    path = tmp_path / LEDGER
    old = {"entries": [{"date": "2026-05-03", "time": "14:00:00", "content_type": "longform"}]}
    path.write_text(json.dumps(old))
    e = cl.ContentLedger.load(path).entries[0]
    assert (e.video_id, e.region, e.date) == (None, None, "2026-05-03")


def test_lf_booking_miss_logs_loud_error_naming_video_id(tmp_path, caplog):
    """A LIVE upload is never SILENTLY unbooked: when record_confirmed_upload
    misses (ledger booking fails after the upload already shipped), the
    caller must surface it loudly rather than swallow the False return."""
    with patch.object(cl, "record_confirmed_upload", return_value=False):
        with caplog.at_level(logging.ERROR):
            esc._book_lf_upload_in_ledger(record=RECORD, state=STATE, config=_cfg(tmp_path))
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("vid123" in r.getMessage() for r in errors), (
        f"expected an ERROR naming video_id=vid123, got: {[r.getMessage() for r in errors]}"
    )


def test_shorts_booking_miss_logs_loud_error(tmp_path, caplog):
    """Same invariant for the shorts CSV-gate booking site in phase_d."""
    with (
        patch.object(cl, "record_confirmed_upload", return_value=False),
        caplog.at_level(logging.ERROR),
    ):
        _emit_csv(
            tmp_path,
            upload_manifests={
                1: SimpleNamespace(
                    files={
                        "s1.mp4": "https://drive.example/1",
                        "s2.mp4": "https://drive.example/2",
                    }
                )
            },
            skip_drive_upload=False,
        )
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("sc" in r.getMessage() for r in errors), (
        f"expected an ERROR naming the project (sc), got: {[r.getMessage() for r in errors]}"
    )


def test_orphaned_lf_upload_books_ledger_despite_none_return(tmp_path):
    """videos.insert already succeeded and wrote its record into
    ``state.youtube_uploads`` before the wrapper's except-clause returned
    None (the failing step was the playlist insert) — the ledger booking
    must still happen, derived from the persisted record, never the None
    return. Otherwise the video is live, holds a publish slot, and
    ``reconcile_ledger_with_youtube`` never self-heals a missing entry."""
    state = make_state(region="usa")
    drive_plan = SimpleNamespace(regular_lfs=[SimpleNamespace(index=1)])
    config = Config()
    config.seo.storage_path = str(tmp_path)
    sched = SimpleNamespace(lf_slots=[SimpleNamespace(datetime_est="2026-03-15T11:00:00")])

    def fake_upload_lf(*, yt, state, lf, **kwargs):
        state.youtube_uploads[lf.index] = {
            "video_id": "ORPHAN_LF",
            "publish_at": "2030-06-14T18:00:00+00:00",
            "title": "T1",
        }
        return None  # mirrors _upload_lf_to_youtube's except-clause return

    yt_instance = MagicMock()
    with (
        patch.object(
            esc,
            "_resolve_youtube_uploader_class",
            return_value=MagicMock(return_value=yt_instance),
        ),
        patch.object(esc, "_upload_lf_to_youtube", side_effect=fake_upload_lf),
    ):
        esc._upload_lfs_to_youtube(
            drive_plan=drive_plan,
            regular_lf_paths=["/v/lf1.mp4"],
            metadata_per_lf={1: {"title": "T1", "description": "d", "tags": []}},
            drive_schedule=sched,
            youtube_uploads={},
            state=state,
            output_base=tmp_path,
            config=config,
            save_state=lambda s: None,
            skip_youtube_upload=False,
            youtube_upload_enabled=True,
        )

    entries = cl.ContentLedger.load(tmp_path / LEDGER).entries
    assert any(e.video_id == "ORPHAN_LF" for e in entries), (
        f"expected a booked entry for ORPHAN_LF, got: {[e.video_id for e in entries]}"
    )


def test_shorts_booked_only_at_csv_gate_pass(tmp_path):
    files = {"s1.mp4": "https://drive.example/1", "s2.mp4": "https://drive.example/2"}
    out = _emit_csv(
        tmp_path, upload_manifests={1: SimpleNamespace(files=files)}, skip_drive_upload=False
    )
    assert out is not None and out.exists()

    entries = cl.ContentLedger.load(tmp_path / LEDGER).entries
    assert {(e.date, e.time, e.content_type, e.region) for e in entries} == {
        ("2026-06-21", "08:00:00", "shorts", "ireland"),
        ("2026-06-21", "16:00:00", "shorts", "ireland"),
    }
