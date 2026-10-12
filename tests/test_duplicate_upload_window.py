"""A successful ``videos.insert`` must be DURABLY persisted before the
(fallible, recoverable) ``playlistItems.insert`` that follows it.

The trigger has already fired in production: a long-form upload took an
``HttpError 409 … playlistItems … SERVICE_UNAVAILABLE`` after its
``videos.insert`` had already succeeded. ``_upload_lf_to_youtube`` swallows
that error and returns ``None``, and the caller loop's
``if record is None: continue`` skipped ``save_state`` — so the live video's
id existed only in memory for the rest of the run (remaining LFs, superlong
parts, thumbnails, CSV, validation gates). A crash or SIGTERM in that window
(``_handle_sigterm`` raises ``SystemExit``, which the phase-level
``except Exception`` does not catch) drops the record, and resume re-runs
``videos.insert`` for content that is already public on YouTube. Uploads are
irreversible; the cleanup is manual.

The persist must FEED the orphan-recovery path — a record with
``video_id`` present and ``playlist_id`` missing re-runs only
``playlistItems.insert`` — never bypass it, and a failed playlist insert must
stay recoverable rather than failing the run.
"""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from googleapiclient.errors import HttpError

from gomoveshift.config import Config
from gomoveshift.pipeline.drive_phases import external_systems_coordinator as esc
from gomoveshift.pipeline.drive_phases import youtube_uploader as yu

from _state_helpers import make_state

FAKE_CREDS = {
    "token": "ya29.fake",
    "refresh_token": "1//fake",
    "client_id": "fake.apps.googleusercontent.com",
    "client_secret": "fakesecret",  # pragma: allowlist secret
    "token_uri": "https://oauth2.googleapis.com/token",
    "scopes": ["https://www.googleapis.com/auth/youtube.upload"],
}


def _409() -> HttpError:
    """The exact production failure class: a retryable playlist-insert 409."""
    resp = MagicMock()
    resp.status = 409
    resp.reason = "SERVICE_UNAVAILABLE"
    return HttpError(resp, b'{"error": {"message": "backendError"}}')


def _mock_service(*, playlist_insert_raises: Exception | None = None) -> MagicMock:
    svc = MagicMock()

    insert_request = MagicMock()
    insert_request.next_chunk.side_effect = [(None, {"id": "VID_1"}), (None, {"id": "VID_2"})]
    svc.videos.return_value.insert.return_value = insert_request

    list_request = MagicMock()
    list_request.execute.return_value = {"items": []}
    svc.playlists.return_value.list.return_value = list_request

    insert_pl = MagicMock()
    insert_pl.execute.return_value = {"id": "PL_ID", "snippet": {"title": "All Drives"}}
    svc.playlists.return_value.insert.return_value = insert_pl

    insert_item = MagicMock()
    if playlist_insert_raises is not None:
        insert_item.execute.side_effect = playlist_insert_raises
    else:
        insert_item.execute.return_value = {"id": "PLITEM_ID"}
    svc.playlistItems.return_value.insert.return_value = insert_item
    return svc


def _uploader(svc: MagicMock):
    with (
        patch.object(yu, "load_youtube_credentials", return_value=FAKE_CREDS),
        patch.object(
            yu, "_build_credentials_obj", return_value=MagicMock(valid=True, expired=False)
        ),
        patch.object(yu, "build", return_value=svc),
        patch.object(yu, "MediaFileUpload", return_value=MagicMock()),
    ):
        up = yu.YouTubeUploader(credentials_dir=Path("/tmp/fake-creds"))
        up._build_service()
    return up


def _cfg(tmp_path: Path) -> Config:
    return Config.model_validate(
        {
            "drive": {"youtube_playlists": {"rules": [], "default": "All Drives"}},
            "seo": {"storage_path": str(tmp_path)},
        }
    )


def _lf_file(tmp_path: Path, idx: int) -> Path:
    p = tmp_path / f"lf_{idx:02d}.mp4"
    p.write_bytes(b"\x00\x00\x00\x20ftypiso5")
    return p


def _run_loop(*, svc, state, tmp_path, save_state, n_lfs=1):
    """Drive the production per-LF upload loop against a mocked API."""
    up = _uploader(svc)
    lfs = [SimpleNamespace(index=i + 1, total_duration=1800.0) for i in range(n_lfs)]
    meta = {
        lf.index: {"title": f"T{lf.index}", "description": "d", "tags": [], "location": "Cork"}
        for lf in lfs
    }
    sched = SimpleNamespace(
        lf_slots=[SimpleNamespace(datetime_est="2026-09-15T11:00:00") for _ in lfs]
    )
    with (
        patch.object(esc, "_resolve_youtube_uploader_class", return_value=lambda: up),
        patch.object(yu, "MediaFileUpload", return_value=MagicMock()),
    ):
        esc._upload_lfs_to_youtube(
            drive_plan=SimpleNamespace(regular_lfs=lfs),
            regular_lf_paths=[str(_lf_file(tmp_path, lf.index)) for lf in lfs],
            metadata_per_lf=meta,
            drive_schedule=sched,
            youtube_uploads=state.youtube_uploads or {},
            state=state,
            output_base=tmp_path,
            config=_cfg(tmp_path),
            save_state=save_state,
            skip_youtube_upload=False,
            youtube_upload_enabled=True,
        )


def _recording_saver(saves: list):
    """``save_state`` stand-in capturing what each flush WOULD have written."""

    def _save(s):
        saves.append(copy.deepcopy(dict(s.youtube_uploads or {})))

    return _save


# ---------------------------------------------------------------------------


def test_playlist_failure_persists_the_live_upload(tmp_path):
    """videos.insert succeeded, playlistItems.insert 409s — the record must
    already be in a flushed checkpoint. RED before the fix: the loop's
    ``continue`` skipped ``save_state`` entirely, so a crash before the next
    phase-level flush lost a video that is live on YouTube."""
    svc = _mock_service(playlist_insert_raises=_409())
    state = make_state()
    saves: list = []

    _run_loop(svc=svc, state=state, tmp_path=tmp_path, save_state=_recording_saver(saves))

    assert svc.videos.return_value.insert.call_count == 1
    persisted = [s for s in saves if s.get(1, {}).get("video_id")]
    assert persisted, (
        "no save_state flush carried LF1's video_id — the live upload exists only "
        f"in memory and a crash here re-uploads it on resume (flushes seen: {saves})"
    )
    assert persisted[-1][1]["video_id"] == "VID_1"
    # Orphan signature preserved for orphan recovery: video_id without playlist_id.
    assert persisted[-1][1].get("playlist_id") is None


def test_resume_after_playlist_failure_does_not_reupload(tmp_path):
    """The harm this pins: resume from ONLY what was durably flushed must take
    the orphan-recovery path (playlist insert alone), never a second
    videos.insert for content already public."""
    svc = _mock_service(playlist_insert_raises=_409())
    saves: list = []
    _run_loop(
        svc=svc,
        state=make_state(),
        tmp_path=tmp_path,
        save_state=_recording_saver(saves),
    )
    flushed = [s for s in saves if s.get(1, {}).get("video_id")]
    assert flushed, "nothing durable to resume from — the video_id was never flushed"

    # Simulated resume: fresh process, state rebuilt from the last checkpoint.
    resumed_state = make_state(youtube_uploads=copy.deepcopy(flushed[-1]))
    svc2 = _mock_service()
    _run_loop(svc=svc2, state=resumed_state, tmp_path=tmp_path, save_state=lambda s: None)

    svc2.videos.return_value.insert.assert_not_called()
    assert svc2.playlistItems.return_value.insert.call_count == 1
    assert resumed_state.youtube_uploads[1]["video_id"] == "VID_1"
    assert resumed_state.youtube_uploads[1]["playlist_id"] == "PL_ID"


def test_successful_upload_still_persists_both_ids(tmp_path):
    """Happy path unchanged: a flush carries video_id AND playlist_id."""
    svc = _mock_service()
    state = make_state()
    saves: list = []

    _run_loop(svc=svc, state=state, tmp_path=tmp_path, save_state=_recording_saver(saves))

    assert svc.videos.return_value.insert.call_count == 1
    assert svc.playlistItems.return_value.insert.call_count == 1
    assert saves[-1][1]["video_id"] == "VID_1"
    assert saves[-1][1]["playlist_id"] == "PL_ID"
    assert state.youtube_uploads[1]["playlist_id"] == "PL_ID"


def test_playlist_failure_does_not_abort_the_run(tmp_path):
    """A failed playlist insert stays recoverable, not fatal: the loop moves
    on to the next LF and uploads it."""
    svc = _mock_service(playlist_insert_raises=_409())
    state = make_state()
    saves: list = []

    _run_loop(svc=svc, state=state, tmp_path=tmp_path, save_state=_recording_saver(saves), n_lfs=2)

    assert svc.videos.return_value.insert.call_count == 2
    assert state.youtube_uploads[1]["video_id"] == "VID_1"
    assert state.youtube_uploads[2]["video_id"] == "VID_2"
