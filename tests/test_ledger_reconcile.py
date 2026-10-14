"""Startup YouTube↔ledger reconciliation.
Policy table in ``reconcile_ledger_with_youtube``. All clients are fakes — no network."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline.content_ledger import (
    EST,
    ContentEntry,
    ContentLedger,
    reconcile_ledger_with_youtube,
)
from gomoveshift.pipeline.drive_phases.phase_d_scheduling import (
    _build_and_persist_drive_schedule,
)

from _state_helpers import make_state

NOW = datetime(2026, 6, 9, 12, 0, tzinfo=EST)


def _fake_yt(videos=None, channel_video_ids=None, fail=False):
    """Read-only Data API stub. ``videos``: id → publishAt ISO (UTC Z)."""
    videos = videos or {}

    def req(result):
        key = "side_effect" if isinstance(result, Exception) else "return_value"
        return MagicMock(**{"execute." + key: result})

    def vids_list(**kw):
        if fail:
            return req(RuntimeError("API down"))
        ids = kw["id"].split(",")
        items = [
            {"id": v, "status": {"publishAt": videos[v]}, "snippet": {}} for v in ids if v in videos
        ]
        return req({"items": items})

    yt = MagicMock()
    yt.videos.return_value.list.side_effect = vids_list
    yt.channels.return_value.list.return_value = req(
        {"items": [{"contentDetails": {"relatedPlaylists": {"uploads": "UU1"}}}]}
    )
    yt.playlistItems.return_value.list.return_value = req(
        {"items": [{"contentDetails": {"videoId": v}} for v in (channel_video_ids or [])]}
    )
    return yt


def _entry(days_ahead, *, video_id=None, content_type="longform"):
    return ContentEntry(
        date=(NOW + timedelta(days=days_ahead)).strftime("%Y-%m-%d"),
        time="14:00:00",
        content_type=content_type,
        status="scheduled",
        video_id=video_id,
        source="pipeline",
    )


def _publish_at(entry, offset_sec=0.0):
    dt = entry.datetime_est + timedelta(seconds=offset_sec)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _ledger(tmp_path, entries):
    led = ContentLedger.load(tmp_path / "ledger.json")
    led.entries = list(entries)
    return led


class TestReconcilePolicy:
    def test_videoid_entries_kept_dropped_and_slot_updated(self, tmp_path):
        kept, moved, gone = (_entry(7, video_id=v) for v in ("vid1", "vid2", "gone"))
        led = _ledger(tmp_path, [kept, moved, gone])
        yt = _fake_yt({"vid1": _publish_at(kept), "vid2": _publish_at(moved, 86400)})
        out = reconcile_ledger_with_youtube(led, yt, now=NOW)
        assert out == {"kept": 2, "dropped": 1, "updated": 1}
        assert led.entries == [kept, moved]  # gone's video is deleted → dropped
        assert moved.date == (NOW + timedelta(days=8)).strftime("%Y-%m-%d")
        assert (tmp_path / "ledger.json").exists()  # changes persisted

    def test_anonymous_phantom_dropped_others_kept(self, tmp_path):
        phantom, matched = _entry(7), _entry(9)  # matched: channel video within ±1h
        past, short = _entry(-7), _entry(7, content_type="shorts")
        in_flight = [_entry(0), _entry(0, video_id="gone")]  # 14:00 today: inside 24h grace
        led = _ledger(tmp_path, [phantom, matched, past, short, *in_flight])
        yt = _fake_yt({"chan1": _publish_at(matched, 1800)}, channel_video_ids=["chan1"])
        out = reconcile_ledger_with_youtube(led, yt, now=NOW)
        assert out["dropped"] == 1 and led.entries == [matched, past, short, *in_flight]

    def test_api_failure_raises_before_any_mutation(self, tmp_path):
        entries = [_entry(7, video_id="vid1"), _entry(7)]
        led = _ledger(tmp_path, entries)
        with pytest.raises(RuntimeError, match="API down"):
            reconcile_ledger_with_youtube(led, _fake_yt(fail=True), now=NOW)
        assert led.entries == entries  # untouched
        assert not (tmp_path / "ledger.json").exists()  # never saved


class TestReconcileCallSite:
    """The Phase D schedule build is the single reconcile site."""

    def _call(self, tmp_path, state):
        scheduler = MagicMock()
        scheduler.generate_drive_schedule.return_value = SimpleNamespace(to_dict=lambda: {})
        scheduler_cls = MagicMock()
        scheduler_cls.from_config.return_value = scheduler
        config = SimpleNamespace(
            drive=SimpleNamespace(
                shorts_publish_times=["15:00"],
                long_form_publish_days=[1],
                long_form_publish_time="16:00",
            ),
            seo=SimpleNamespace(storage_path=str(tmp_path)),
        )
        return _build_and_persist_drive_schedule(
            scheduler_cls=scheduler_cls,
            config=config,
            drive_plan=SimpleNamespace(regular_lfs=[]),
            state=state,
            region="usa",
            stages_dir=tmp_path,
        )

    def test_failure_warns_once_proceeds_and_escape_flag_skips(self, tmp_path, caplog):
        with (
            patch(
                "gomoveshift.pipeline.drive_phases.phase_d_scheduling."
                "_build_readonly_youtube_service",
                side_effect=RuntimeError("no creds"),
            ) as svc,
            caplog.at_level(logging.WARNING),
        ):
            # Creds/API failure → one WARN, run proceeds with the ledger as-is.
            assert self._call(tmp_path, make_state()) is not None
            warns = [r for r in caplog.records if "LEDGER-RECONCILE skipped" in r.getMessage()]
            assert len(warns) == 1 and "no creds" in warns[0].getMessage()
            # --no-ledger-reconcile → reconcile path never entered at all.
            self._call(tmp_path, make_state(no_ledger_reconcile=True))
            svc.assert_called_once()  # only the first (unflagged) call built a client
