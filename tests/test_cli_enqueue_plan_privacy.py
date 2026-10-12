"""enqueue-plan refuses to spool a delivery whose media privacy
status is unrecorded or unrecognised -- upstream of any queue write, same
place ``_media_verification_errors`` already refuses the whole run. Video
records (what the drive pipeline actually builds) are never blocked: see
``console.library.privacy_status_for_media``.
"""

from __future__ import annotations

import hashlib
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from gomoveshift.cli.handlers.enqueue_plan import cmd_enqueue_plan
from gomoveshift.pipeline.drive_phases.phase_d import _serialize_publish_plan
from gomoveshift.pipeline.review_gate import compute_gate_cd_hash
from gomoveshift.publishing.models import (
    FORMAT_PHOTO,
    MediaFingerprint,
    PlatformPayloads,
    PublishPlan,
    RouteSelection,
    ScheduleWindow,
    ShortPublishRecord,
    YouTubePayload,
)
from gomoveshift.publishing.queue import open_queue_db

SCHEDULED = datetime(2026, 9, 1, 19, 0, tzinfo=timezone.utc)

_MEDIA_PATH = Path(tempfile.mkdtemp(prefix="gms-enqueue-plan-privacy-test-")) / "photo-001.jpg"
_MEDIA_PATH.write_bytes(b"gms enqueue-plan privacy test media fixture bytes")
_REAL_MEDIA = MediaFingerprint(
    str(_MEDIA_PATH),
    _MEDIA_PATH.stat().st_size,
    2.0,
    1080,
    1920,
    "h264",
    hashlib.sha256(_MEDIA_PATH.read_bytes()).hexdigest(),
)


def _record(format_=None):
    return ShortPublishRecord(
        project_name="proj",
        lf_index=0,
        short_index=3,
        media=_REAL_MEDIA,
        schedule=ScheduleWindow(SCHEDULED, "America/Chicago"),
        recording_date="2026-08-01",
        sampled_regions=("a",),
        parent_lf_video_id="vid",
        parent_lf_publish_at=SCHEDULED,
        payloads=PlatformPayloads(youtube=YouTubePayload("t", "d")),
        consent_hash="c",
        reviewed_plan_hash="HASH1",
        routes=RouteSelection(selected=("youtube",)),
        format=format_,
    )


def _sidecar(out_dir, *records):
    path = out_dir / "proj_publish_plan.json"
    path.write_text(_serialize_publish_plan(PublishPlan(shorts=tuple(records))), encoding="utf-8")
    return path


def _approved_state(out_dir, **over):
    state = SimpleNamespace(
        regular_lf_paths=[],
        gate_cd_confirmed=True,
        gate_cd_confirmation_hash=None,
        **over,
    )
    state.gate_cd_confirmation_hash = compute_gate_cd_hash(state=state, output_dir=out_dir)
    return state


def _config(tmp_path):
    return SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))


def _args(state_path, **over):
    return SimpleNamespace(
        **{
            "state_path": str(state_path),
            "apply": True,
            "output_dir": None,
            "shadow": False,
            **over,
        }
    )


def test_unrecorded_photo_privacy_status_refuses_the_whole_run(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("GMS_SITE_PATH", raising=False)
    out = tmp_path / "out"
    out.mkdir()
    _sidecar(out, _record(format_=FORMAT_PHOTO))
    state_path = out / "pipeline_state.json"
    state_path.write_text("{}", encoding="utf-8")

    with mock.patch(
        "gomoveshift.cli.handlers.enqueue_plan._load_state",
        return_value=_approved_state(out),
    ):
        rc = cmd_enqueue_plan(_args(state_path), _config(tmp_path))

    assert rc == 1
    out_text = capsys.readouterr().out
    assert "privacy" in out_text.lower()
    assert "nothing was written to the queue" in out_text

    conn = open_queue_db(tmp_path)
    rows = conn.execute("SELECT COUNT(*) AS n FROM deliveries").fetchone()["n"]
    conn.close()
    assert rows == 0


def test_video_format_record_is_never_blocked_by_the_privacy_gate(tmp_path, monkeypatch):
    monkeypatch.delenv("GMS_SITE_PATH", raising=False)
    out = tmp_path / "out"
    out.mkdir()
    _sidecar(out, _record(format_=None))
    state_path = out / "pipeline_state.json"
    state_path.write_text("{}", encoding="utf-8")

    with mock.patch(
        "gomoveshift.cli.handlers.enqueue_plan._load_state",
        return_value=_approved_state(out),
    ):
        rc = cmd_enqueue_plan(_args(state_path), _config(tmp_path))

    assert rc == 0
