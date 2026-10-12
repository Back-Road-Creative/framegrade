"""Proactive YouTube Data API v3 upload quota accounting.

Previously ``_upload_lfs_to_youtube`` was purely reactive: quota
exhaustion was only discovered as an HTTP 403 mid-batch
(``YouTubeQuotaError`` from ``youtube_uploader._execute_resumable``).
These tests pin the new upstream gate: a planned run whose insert cost
(``videos.insert`` + ``playlistItems.insert`` per upload) exceeds the
configured daily budget is refused BEFORE any upload call is made, and a
run comfortably inside budget is unaffected.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline.drive_phases import external_systems_coordinator as esc
from gomoveshift.pipeline.drive_phases.external_systems_coordinator import (
    _upload_lfs_to_youtube,
)
from gomoveshift.pipeline.exceptions import YouTubeQuotaError

from _state_helpers import make_state


def _lf(idx):
    return SimpleNamespace(index=idx, total_duration=1800.0)


def _config(*, budget=10000, video_cost=1600, playlist_cost=50, enforce=True):
    return SimpleNamespace(
        drive=SimpleNamespace(
            youtube_quota_daily_budget=budget,
            youtube_video_insert_quota_cost=video_cost,
            youtube_playlist_insert_quota_cost=playlist_cost,
            youtube_quota_enforce=enforce,
        )
    )


def _mk_inputs(lf_count):
    drive_plan = SimpleNamespace(regular_lfs=[_lf(i) for i in range(1, lf_count + 1)])
    regular_lf_paths = [f"/v/lf{i}.mp4" for i in range(1, lf_count + 1)]
    metadata_per_lf = {
        i: {"title": f"T{i}", "description": "d", "tags": []} for i in range(1, lf_count + 1)
    }
    drive_schedule = SimpleNamespace(
        lf_slots=[SimpleNamespace(datetime_est="2026-03-15T11:00:00") for _ in range(lf_count)]
    )
    return drive_plan, regular_lf_paths, metadata_per_lf, drive_schedule


class TestYoutubeQuotaBudgetPreflight:
    def test_over_budget_batch_refused_before_any_upload(self, tmp_path):
        # 7 LFs * (1600 + 50) = 11550 > the 10000 default budget.
        drive_plan, paths, meta, sched = _mk_inputs(7)
        with (
            patch.object(esc, "_resolve_youtube_uploader_class") as resolve_cls,
            patch.object(esc, "_upload_lf_to_youtube") as upload_one,
        ):
            with pytest.raises(YouTubeQuotaError, match="quota"):
                _upload_lfs_to_youtube(
                    drive_plan=drive_plan,
                    regular_lf_paths=paths,
                    metadata_per_lf=meta,
                    drive_schedule=sched,
                    youtube_uploads={},
                    state=make_state(),
                    output_base=tmp_path,
                    config=_config(),
                    save_state=lambda s: None,
                    skip_youtube_upload=False,
                    youtube_upload_enabled=True,
                )
        # The refusal fires before the uploader class is even resolved —
        # no upload call is made.
        resolve_cls.assert_not_called()
        upload_one.assert_not_called()

    def test_over_budget_warns_but_proceeds_by_default(self, tmp_path, caplog):
        """Warn-only is the DEFAULT. The budget is Google's number, not ours, and
        a batch that exceeds it still uploads what fits — the reactive 403 pause
        plus skip-on-resume recovers the rest. Refusing by default would strip
        that recovery from a merely-large drive (and did break a 9-LF resume
        case in the existing suite)."""
        drive_plan, paths, meta, sched = _mk_inputs(7)  # 11550 > 10000
        yt_mock = MagicMock()
        with (
            patch.object(
                esc,
                "_resolve_youtube_uploader_class",
                return_value=MagicMock(return_value=yt_mock),
            ),
            patch.object(esc, "_upload_lf_to_youtube", return_value={"video_id": "abc"}),
        ):
            _yt, uploads = _upload_lfs_to_youtube(
                drive_plan=drive_plan,
                regular_lf_paths=paths,
                metadata_per_lf=meta,
                drive_schedule=sched,
                youtube_uploads={},
                state=make_state(),
                output_base=tmp_path,
                config=_config(enforce=False),
                save_state=lambda s: None,
                skip_youtube_upload=False,
                youtube_upload_enabled=True,
            )
        assert len(uploads) == 7, "warn-only must not block the batch"
        assert any(
            "quota projection" in r.message.lower() or "quota projection" in r.getMessage().lower()
            for r in caplog.records
        ), "the operator must still be told the run is projected over budget"

    def test_within_budget_batch_unaffected(self, tmp_path):
        # 2 LFs * (1600 + 50) = 3300, comfortably under the 10000 budget.
        drive_plan, paths, meta, sched = _mk_inputs(2)
        yt_mock = MagicMock()
        with (
            patch.object(
                esc,
                "_resolve_youtube_uploader_class",
                return_value=MagicMock(return_value=yt_mock),
            ),
            patch.object(esc, "_upload_lf_to_youtube", return_value={"video_id": "abc"}),
        ):
            yt, uploads = _upload_lfs_to_youtube(
                drive_plan=drive_plan,
                regular_lf_paths=paths,
                metadata_per_lf=meta,
                drive_schedule=sched,
                youtube_uploads={},
                state=make_state(),
                output_base=tmp_path,
                config=_config(),
                save_state=lambda s: None,
                skip_youtube_upload=False,
                youtube_upload_enabled=True,
            )
        assert yt is yt_mock
        assert len(uploads) == 2

    def test_unconfigured_test_double_config_skips_check(self, tmp_path):
        """A ``MagicMock()`` config (as existing tests pass) has no real
        numeric quota fields — the preflight must skip rather than raise
        on a non-numeric comparison, so it never breaks pre-existing
        callers that don't wire the new config fields."""
        drive_plan, paths, meta, sched = _mk_inputs(7)
        yt_mock = MagicMock()
        with (
            patch.object(
                esc,
                "_resolve_youtube_uploader_class",
                return_value=MagicMock(return_value=yt_mock),
            ),
            patch.object(esc, "_upload_lf_to_youtube", return_value={"video_id": "abc"}),
        ):
            yt, uploads = _upload_lfs_to_youtube(
                drive_plan=drive_plan,
                regular_lf_paths=paths,
                metadata_per_lf=meta,
                drive_schedule=sched,
                youtube_uploads={},
                state=make_state(),
                output_base=tmp_path,
                config=MagicMock(),
                save_state=lambda s: None,
                skip_youtube_upload=False,
                youtube_upload_enabled=True,
            )
        assert yt is yt_mock
        assert len(uploads) == 7
