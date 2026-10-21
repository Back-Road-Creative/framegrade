"""The two places that rewrite a title or description
outside the reviewed upload path cannot start a YouTube insert.

* The legacy single-video stage ``stages/chapters.py`` (and every stage
  beside it) never reaches the YouTube upload surface.
* ``plan_title_sync`` rewrites a Short's sidecar title only for a delivery
  that is already published; an unpublished Short's sidecar is never edited, and
  publishing the rewritten, published Short sends no insert.

Synthetic data only; only the YouTube client is a stub.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

import pytest
from test_cli_enqueue_plan import _record, _sidecar
from test_publishing_youtube_adapter import mock_service  # noqa: F401 -- fixture

from gomoveshift.pipeline import stages as stages_pkg
from gomoveshift.pipeline.drive_phases import youtube_uploader as yu
from gomoveshift.publishing.plan_io import load_publish_plan
from gomoveshift.publishing.plan_title_sync import sync_plan_title
from gomoveshift.publishing.queue import enqueue, open_queue_db
from gomoveshift.publishing.youtube_adapter import YouTubeShortsAdapter

_INSERT_SURFACE = re.compile(
    r"YouTubeUploader|youtube_uploader|youtube_adapter|upload_video|upload_if_needed"
    r"|external_systems_coordinator|videos\(\)\s*\.\s*insert|googleapiclient"
)


def test_t5_no_pipeline_stage_reaches_the_youtube_insert():
    stage_files = sorted(Path(stages_pkg.__file__).parent.glob("*.py"))
    assert any(p.name == "chapters.py" for p in stage_files)
    hits = {
        p.name: _INSERT_SURFACE.findall(p.read_text(encoding="utf-8"))
        for p in stage_files
        if _INSERT_SURFACE.search(p.read_text(encoding="utf-8"))
    }
    assert hits == {}


def test_t5_the_stage_upload_goes_to_drive_not_youtube():
    from gomoveshift.pipeline import uploader

    assert "youtube" not in Path(uploader.__file__).read_text(encoding="utf-8").lower()


@pytest.fixture
def conn(tmp_path):
    c = open_queue_db(tmp_path)
    yield c
    c.close()


def _enqueue(conn, tmp_path, *, published):
    out = tmp_path / "out"
    out.mkdir()
    sidecar = _sidecar(out, _record())
    did = enqueue(conn, "proj", 3, "youtube", "HASH1", sidecar_path=str(sidecar))
    if published:
        conn.execute(
            "UPDATE deliveries SET status='published', platform_upload_id='SYNTHvid01' WHERE id=?",
            (did,),
        )
        conn.commit()
    return sidecar, did


def test_t6_an_unpublished_shorts_sidecar_title_is_never_rewritten(conn, tmp_path):
    sidecar, _ = _enqueue(conn, tmp_path, published=False)
    before = sidecar.read_bytes()

    changes = sync_plan_title(conn, "SYNTHvid01", "A different title", apply=True)

    assert changes == []
    assert sidecar.read_bytes() == before


def test_t6_publishing_a_rewritten_published_short_sends_no_insert(
    conn,
    tmp_path,
    mock_service,  # noqa: F811
):
    sidecar, did = _enqueue(conn, tmp_path, published=True)
    assert sync_plan_title(conn, "SYNTHvid01", "A different title", apply=True)
    record = load_publish_plan(sidecar).shorts[0]
    assert record.payloads.youtube.title == "A different title"
    video = tmp_path / "short_00.mp4"
    video.write_bytes(b"\x00\x00\x00\x20ftypiso5")
    creds = {
        "refresh_token": "1//fake",
        "client_id": "x",
        "client_secret": "y",
    }  # pragma: allowlist secret
    with (
        patch.object(yu, "load_youtube_credentials", return_value=creds),
        patch.object(yu, "_build_credentials_obj", return_value=object()),
        patch.object(yu, "build", return_value=mock_service),
        patch.object(yu, "MediaFileUpload", return_value=object()),
        patch.object(yu, "FINAL_TEXT_BIND_ON", True, create=True),
    ):
        uploader = yu.YouTubeUploader(credentials_dir=Path("/tmp/fake-creds"))
        uploader._build_service()
        result = YouTubeShortsAdapter(uploader, object()).publish(
            record, conn=conn, delivery_id=did, video_path=video
        )

    assert result.video_id == "SYNTHvid01"
    assert mock_service.videos.return_value.insert.call_count == 0
    assert not list(tmp_path.rglob("*.confirmed"))
