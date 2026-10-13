"""Superlong parts must reach ``yt.upload_if_needed``; drives the real
``_upload_lfs_to_youtube`` entry point phase_d.py calls in production."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from gomoveshift.config import Config
from gomoveshift.pipeline import youtube_sidecar
from gomoveshift.pipeline.drive_phases import external_systems_coordinator as esc
from gomoveshift.pipeline.schedule_constants import EST

from _state_helpers import make_state


# The superlong booking calls ``next_free_superlong_slot(..., after=datetime.now(EST))``,
# so without a frozen clock this module's slot assertions depend on the day the
# suite happens to run. Freeze it: with an EMPTY ledger and a project identity
# supplied, ``resolve_superlong_publish_day`` finds no related history, returns
# ``None`` ("search ANY weekday" — the rule is to be aware of the next open day when
# there is no previous related video), and the slot is simply the NEXT OPEN
# SLOT at Waypost's long-form time (the seed's 16:00). It is not pinned to Saturday.
_FROZEN_NOW = datetime(2026, 3, 10, 9, 0, tzinfo=EST)  # a Tuesday
_EXPECTED_SLOT = datetime(2026, 3, 10, 16, 0, tzinfo=EST)  # the same day, at Waypost's seed time


class _FrozenDatetime(datetime):
    """``datetime`` with a pinned ``now()``; everything else behaves normally."""

    @classmethod
    def now(cls, tz=None):
        return _FROZEN_NOW.astimezone(tz) if tz is not None else _FROZEN_NOW.replace(tzinfo=None)


def _part_record(tmp_path):
    path = tmp_path / "sl_01.mp4"
    path.write_bytes(b"x")
    return {"path": str(path), "part_index": 1, "title": "Superlong Part", "description": "d"}


def _state(tmp_path):
    return make_state(
        region="usa",
        superlong_parts={"2026-03-01": {"part_count": 1, "records": [_part_record(tmp_path)]}},
    )


def _run(tmp_path, state, *, yt_mock, lf_record=None):
    config = Config()
    config.seo.storage_path = str(tmp_path)
    with (
        patch.object(
            esc, "_resolve_youtube_uploader_class", return_value=MagicMock(return_value=yt_mock)
        ),
        patch.object(esc, "_upload_lf_to_youtube", return_value=lf_record or {"video_id": "abc"}),
        patch.object(esc, "datetime", _FrozenDatetime),
    ):
        return esc._upload_lfs_to_youtube(
            drive_plan=SimpleNamespace(regular_lfs=[SimpleNamespace(index=1)]),
            regular_lf_paths=["/v/lf1.mp4"],
            metadata_per_lf={1: {"title": "T1", "description": "d", "tags": []}},
            drive_schedule=SimpleNamespace(
                lf_slots=[SimpleNamespace(datetime_est="2026-03-15T11:00:00")]
            ),
            youtube_uploads={},
            state=state,
            output_base=tmp_path,
            config=config,
            save_state=lambda s: None,
            skip_youtube_upload=False,
            youtube_upload_enabled=True,
        )


def test_superlong_part_uploaded_after_lf_loop_via_own_sidecar(tmp_path):
    """Load-bearing (fails on current code): part reaches the uploader
    with a real slot, lands in superlong_uploads.json, never state.youtube_uploads."""
    state = _state(tmp_path)
    yt_mock = MagicMock()
    yt_mock.upload_if_needed.return_value = {"video_id": "PARTID"}

    _, youtube_uploads = _run(tmp_path, state, yt_mock=yt_mock)

    assert yt_mock.upload_if_needed.call_count == 1
    _, kwargs = yt_mock.upload_if_needed.call_args
    assert kwargs["lf_idx"] == "2026-03-01:01"
    assert str(kwargs["video_path"]) == str(tmp_path / "sl_01.mp4")
    # Exact slot, not a weekday: the old `weekday() == 5` only held when the
    # suite ran on a Friday (or a Saturday before the 11:00 slot time), so it
    # started failing mid-morning with no code change.
    assert kwargs["publish_at"] == _EXPECTED_SLOT
    assert set(youtube_uploads.keys()) == set(state.youtube_uploads.keys()) == {1}  # LF only
    sidecar = youtube_sidecar.load_superlong_sidecar(tmp_path)
    assert sidecar["2026-03-01:01"]["video_id"] == "PARTID"


def test_resume_skips_already_uploaded_part(tmp_path):
    state = _state(tmp_path)
    youtube_sidecar.save_superlong_sidecar(
        tmp_path, {"2026-03-01:01": {"video_id": "EXISTING", "playlist_id": "PL_DONE"}}
    )
    yt_mock = MagicMock()

    _run(tmp_path, state, yt_mock=yt_mock)

    yt_mock.upload_if_needed.assert_not_called()


def test_orphan_playlist_recovers_on_resume(tmp_path):
    """A part with video_id but no playlist_id (a 409 mid-run) must still
    reach upload_if_needed so the playlist-only recovery branch can run — the pre-skip must not shadow it."""
    state = _state(tmp_path)
    youtube_sidecar.save_superlong_sidecar(
        tmp_path,
        {"2026-03-01:01": {"video_id": "ORPHAN", "publish_at": "2030-01-01T12:00:00-05:00"}},
    )
    yt_mock = MagicMock()
    yt_mock.upload_if_needed.return_value = {
        "video_id": "ORPHAN",
        "publish_at": "2030-01-01T12:00:00-05:00",
        "playlist_id": "PL_RECOVERED",
    }

    _run(tmp_path, state, yt_mock=yt_mock)

    assert yt_mock.upload_if_needed.call_count == 1
    sidecar = youtube_sidecar.load_superlong_sidecar(tmp_path)
    assert sidecar["2026-03-01:01"]["playlist_id"] == "PL_RECOVERED"


def test_orphan_recovery_never_reruns_overrun_recheck(tmp_path):
    """Regression: an orphan whose publish_at has already passed is LIVE
    on YouTube (privacy flips at publish_at, independent of this run) —
    the overrun-recheck must not flip it back to private with a new
    future slot. Its slot was never at risk from THIS run; only a
    genuine upload performed in this iteration can have overrun it."""
    state = _state(tmp_path)
    past_publish_at = "2020-01-01T11:00:00-05:00"
    youtube_sidecar.save_superlong_sidecar(
        tmp_path,
        {"2026-03-01:01": {"video_id": "ORPHAN", "publish_at": past_publish_at}},
    )
    yt_mock = MagicMock()
    yt_mock.upload_if_needed.return_value = {
        "video_id": "ORPHAN",
        "publish_at": past_publish_at,
        "playlist_id": "PL_RECOVERED",
    }

    _run(tmp_path, state, yt_mock=yt_mock)

    yt_mock.update_privacy.assert_not_called()
    sidecar = youtube_sidecar.load_superlong_sidecar(tmp_path)
    assert sidecar["2026-03-01:01"]["publish_at"] == past_publish_at


def test_fresh_upload_overrun_recheck_still_runs(tmp_path):
    """Control: a genuine fresh upload (no prior sidecar record) that
    overran its own slot must still get the overrun recheck — only
    orphan-playlist recovery skips it."""
    state = _state(tmp_path)
    past_publish_at = "2020-01-01T11:00:00-05:00"
    yt_mock = MagicMock()
    yt_mock.upload_if_needed.return_value = {
        "video_id": "PARTID",
        "publish_at": past_publish_at,
        "playlist_id": "PL_NEW",
    }

    _run(tmp_path, state, yt_mock=yt_mock)

    yt_mock.update_privacy.assert_called_once()


def test_orphan_playlist_recovery_failure_does_not_raise(tmp_path, caplog):
    """A retried playlist assignment can itself fail (another 409) — the
    coordinator must log and move on, never crash the LF batch."""
    from googleapiclient.errors import HttpError

    state = _state(tmp_path)
    youtube_sidecar.save_superlong_sidecar(
        tmp_path,
        {"2026-03-01:01": {"video_id": "ORPHAN", "publish_at": "2030-01-01T12:00:00-05:00"}},
    )
    yt_mock = MagicMock()
    resp = MagicMock(status=503, reason="Service Unavailable")
    yt_mock.upload_if_needed.side_effect = HttpError(resp, b"", uri="fake://")

    with caplog.at_level("WARNING"):
        _, youtube_uploads = _run(tmp_path, state, yt_mock=yt_mock, lf_record={"video_id": "lf-ok"})

    assert yt_mock.upload_if_needed.call_count == 1
    assert "HTTP error" in caplog.text
    assert youtube_uploads == {1: {"video_id": "lf-ok"}}
    sidecar = youtube_sidecar.load_superlong_sidecar(tmp_path)
    assert sidecar["2026-03-01:01"] == {
        "video_id": "ORPHAN",
        "publish_at": "2030-01-01T12:00:00-05:00",
    }


def test_superlong_orphan_flushes_sidecar_before_playlist_crash(tmp_path):
    """videos.insert succeeds, then the process is killed mid-playlist-assign
    (modeled as a BaseException that the per-part except-Exception handler
    does NOT catch) — the sidecar on DISK must already hold the orphaned
    video_id, exactly like the regular-LF ``persist=`` checkpoint at
    ``upload_if_needed``'s videos.insert/playlist-insert boundary. Assert
    the FILE, not the in-memory dict: that is the whole defect."""
    state = _state(tmp_path)
    yt_mock = MagicMock()

    def fake_upload_if_needed(*, state, lf_idx, persist=None, **kwargs):
        state.youtube_uploads[lf_idx] = {
            "video_id": "ORPHANED",
            "publish_at": "2030-01-01T12:00:00-05:00",
        }
        if persist is not None:
            persist()
        raise KeyboardInterrupt  # models a crash after persist(), before return

    yt_mock.upload_if_needed.side_effect = fake_upload_if_needed

    state_for_run = state
    try:
        _run(tmp_path, state_for_run, yt_mock=yt_mock)
    except KeyboardInterrupt:
        pass

    sidecar = youtube_sidecar.load_superlong_sidecar(tmp_path)
    assert sidecar.get("2026-03-01:01", {}).get("video_id") == "ORPHANED", (
        f"expected the orphaned video_id flushed to disk, got: {sidecar}"
    )


def test_part_upload_error_logs_and_skips_lf_batch_intact(tmp_path):
    """A part-upload error never propagates; the already-uploaded LF batch is unaffected."""
    state = _state(tmp_path)
    yt_mock = MagicMock()
    yt_mock.upload_if_needed.side_effect = RuntimeError("boom")

    _, youtube_uploads = _run(tmp_path, state, yt_mock=yt_mock, lf_record={"video_id": "lf-ok"})

    assert youtube_uploads == {1: {"video_id": "lf-ok"}}
    assert youtube_sidecar.load_superlong_sidecar(tmp_path) == {}
