"""
TDD tests: the YouTube analytics sync saves its events in one batch and runs
once per generator (one Phase D run), not once per long-form.

Incident 2026-11-14: 1126 channel videos -> 1238 full event-store index
rebuilds (one ``upsert_event`` each, repeated for every LF), ~23 min inside a
single sync before any upload.
"""

from unittest.mock import MagicMock, patch

from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator


def _response(video_ids):
    return {
        "items": [
            {
                "id": vid,
                "snippet": {"title": f"Video {vid}", "publishedAt": "2026-01-01T00:00:00Z"},
                "statistics": {
                    "viewCount": "100",
                    "likeCount": "5",
                    "commentCount": "2",
                    "favoriteCount": "0",
                },
            }
            for vid in video_ids
        ]
    }


def _gen(store):
    with patch.object(LongFormMetadataGenerator, "_init_seo_manager"):
        with patch.object(LongFormMetadataGenerator, "_init_ledger"):
            gen = LongFormMetadataGenerator()
    gen._manager = MagicMock()
    gen._manager.store = store
    gen._optimizer = None
    return gen


def _sync_with(batches):
    """YouTube client whose successive videos().list().execute() give ``batches``."""
    sync = MagicMock()
    sync.youtube_data.videos.return_value.list.return_value.execute.side_effect = batches
    sync.youtube_analytics.reports.return_value.query.return_value.execute.return_value = {}
    return sync


def _mapping(n):
    return {f"v{i}": f"c{i}" for i in range(n)}


class TestBatchSave:
    def test_batch_method_called_once_with_all_events(self):
        store = MagicMock(spec=["upsert_events", "upsert_event"])
        gen = _gen(store)
        mapping = _mapping(120)  # three offsets of 50/50/20
        ids = list(mapping)
        sync = _sync_with([_response(ids[0:50]), _response(ids[50:100]), _response(ids[100:])])

        gen._batch_sync_stats(sync, mapping)

        store.upsert_events.assert_called_once()
        (events,) = store.upsert_events.call_args.args
        assert [e.event_id for e in events] == [f"yt_sync_{v}" for v in ids]
        store.upsert_event.assert_not_called()

    def test_fallback_upserts_each_event_without_batch_method(self):
        store = MagicMock(spec=["upsert_event"])
        gen = _gen(store)
        mapping = _mapping(3)
        sync = _sync_with([_response(list(mapping))])

        gen._batch_sync_stats(sync, mapping)

        assert [c.args[0].event_id for c in store.upsert_event.call_args_list] == [
            "yt_sync_v0",
            "yt_sync_v1",
            "yt_sync_v2",
        ]

    def test_api_error_on_one_offset_keeps_other_offsets(self):
        store = MagicMock(spec=["upsert_events", "upsert_event"])
        gen = _gen(store)
        mapping = _mapping(120)
        ids = list(mapping)
        sync = _sync_with([_response(ids[0:50]), RuntimeError("quota"), _response(ids[100:])])

        gen._batch_sync_stats(sync, mapping)

        store.upsert_events.assert_called_once()
        (events,) = store.upsert_events.call_args.args
        assert [e.event_id for e in events] == [f"yt_sync_{v}" for v in ids[0:50] + ids[100:]]

    def test_store_save_error_only_logs(self):
        # Before batching, each upsert sat inside the per-offset try, so a store
        # error never stopped metadata generation. The one batched save keeps that.
        store = MagicMock(spec=["upsert_events", "upsert_event"])
        store.upsert_events.side_effect = OSError("disk full")
        gen = _gen(store)
        mapping = _mapping(3)
        sync = _sync_with([_response(list(mapping))])

        gen._batch_sync_stats(sync, mapping)  # must not raise

        store.upsert_events.assert_called_once()


class TestSyncOncePerRun:
    def test_second_generation_does_not_sync_again(self):
        gen = _gen(MagicMock())
        with patch.object(gen, "_sync_youtube_analytics") as sync:
            for _ in range(2):
                try:
                    gen._generate_with_seo_manager("Somewhere, VT", "mon", 3600.0, "proj")
                except Exception:
                    pass  # only the sync gate is under test, not metadata rendering

        assert sync.call_count == 1
