"""Publish preparation journeys: what is enqueued is what is posted.

Real queue + real ``drain_queue``; adapters are recording fakes injected via
``build_handlers``. Manifest: tests/fixtures/acceptance/journeys.json."""

from __future__ import annotations

import pytest
from .acceptance_harness import (
    ACCOUNT,
    NOW,
    PROJECT,
    RecordingInstagramAdapter,
    asset_loader,
    loader_for,
    make_record,
    row_of,
)

from gomoveshift.publishing.dispatch import drain_queue, reconcile_awaiting_manual
from gomoveshift.publishing.handlers import TikTokCredentials, build_handlers
from gomoveshift.publishing.plan_builder import _with_filmed
from gomoveshift.publishing.plan_io import PublishPlanFormatError, load_publish_plan
from gomoveshift.publishing.queue import enqueue
from gomoveshift.publishing.tiktok_adapter import ChunkPlan


def test_j07_01_adapter_receives_exactly_what_was_enqueued(conn, asset):
    record = make_record(asset, caption="Cresting the ridge at dawn. #POVDriving")
    adapter = RecordingInstagramAdapter()
    handlers = build_handlers(
        conn, loader_for({(PROJECT, 0): record}), instagram=(adapter, asset_loader)
    )
    delivery_id = enqueue(conn, PROJECT, 0, "instagram", "hash-1")

    result = drain_queue(conn, handlers, now=NOW)

    assert result.succeeded == 1
    assert len(adapter.calls) == 1
    call = adapter.calls[0]
    assert call["asset_path"] == str(asset)
    assert call["caption"] == "Cresting the ridge at dawn. #POVDriving"
    assert call["video_bytes"] == b"video-bytes"
    row = row_of(conn, delivery_id)
    assert row["status"] == "published"
    assert row["destination_account"] == ACCOUNT == call["account"]
    assert row["platform_post_id"] == adapter.remote_posts[0]["id"]


def test_j07_02_caption_keeps_full_stop_before_filmed(conn, asset):
    caption = _with_filmed(
        "Just driving through Example Town #POVDriving #PA", "Filmed March 2026."
    )
    assert "Example Town. Filmed March 2026." in caption
    record = make_record(asset, caption=caption)
    adapter = RecordingInstagramAdapter()
    handlers = build_handlers(
        conn, loader_for({(PROJECT, 0): record}), instagram=(adapter, asset_loader)
    )
    enqueue(conn, PROJECT, 0, "instagram", "hash-1")

    drain_queue(conn, handlers, now=NOW)

    sent = adapter.calls[0]["caption"]
    assert sent == caption
    assert "Example Town Filmed" not in sent
    assert ". Filmed March 2026." in sent
    assert sent.rstrip().endswith("#POVDriving #PA")


def test_j07_03_missing_sidecar_is_refused_not_published(conn, tmp_path):
    sidecar = tmp_path / "gone_publish_plan.json"
    with pytest.raises(PublishPlanFormatError):
        load_publish_plan(sidecar)

    def loader(project, idx):
        return load_publish_plan(sidecar).shorts[idx]

    adapter = RecordingInstagramAdapter()
    handlers = build_handlers(conn, loader, instagram=(adapter, asset_loader))
    delivery_id = enqueue(conn, PROJECT, 0, "instagram", "hash-1", sidecar_path=str(sidecar))

    result = drain_queue(conn, handlers, now=NOW)

    assert adapter.calls == []
    assert adapter.remote_posts == []
    assert result.succeeded == 0
    row = row_of(conn, delivery_id)
    assert row["status"] != "published"
    assert row["platform_post_id"] is None
    assert "PublishPlanFormatError" in (row["error_class"] or "")


def test_j07_04_missing_asset_is_refused_not_published(conn, asset):
    record = make_record(asset)
    asset.unlink()
    adapter = RecordingInstagramAdapter()
    handlers = build_handlers(
        conn, loader_for({(PROJECT, 0): record}), instagram=(adapter, asset_loader)
    )
    delivery_id = enqueue(conn, PROJECT, 0, "instagram", "hash-1")

    result = drain_queue(conn, handlers, now=NOW)

    assert adapter.calls == []
    assert result.succeeded == 0
    row = row_of(conn, delivery_id)
    assert row["status"] != "published"
    assert row["platform_post_id"] is None
    assert row["error_class"] == "FileNotFoundError"


def test_j07_05_parked_tiktok_row_is_never_reported_published(conn, asset, monkeypatch):
    import gomoveshift.publishing.handlers as handlers_mod

    # The inbox upload path; the default is pack mode.
    monkeypatch.setenv("GMS_TIKTOK_MODE", "inbox")
    inits = []

    def fake_init(*_a, **kwargs):
        inits.append(kwargs)
        return {
            "publish_id": "pub-1",
            "upload_url": "https://upload.invalid/x",
            "chunk_plan": ChunkPlan(video_size=11, chunk_size=11, total_chunk_count=1),
        }

    monkeypatch.setattr(handlers_mod, "init_direct_post", fake_init)
    monkeypatch.setattr(handlers_mod, "upload_video_chunks", lambda *a, **k: None)
    record = make_record(asset, caption="Ridge at dawn. Filmed Sept 2026.")
    handlers = build_handlers(
        conn,
        loader_for({(PROJECT, 0): record}),
        tiktok=(TikTokCredentials(access_token="act.fake"), asset_loader),  # noqa: S106
        worker_id="worker-1",  # the CLI passes the same id as drain_queue's owner
    )
    delivery_id = enqueue(conn, PROJECT, 0, "tiktok", "hash-1")

    result = drain_queue(conn, handlers, owner="worker-1", now=NOW)
    again = drain_queue(conn, handlers, owner="worker-1", now="2026-10-03T13:00:00+00:00")

    assert result.succeeded == 0 and again.succeeded == 0
    assert len(inits) == 1  # never uploaded twice
    assert inits[0]["title"] == "Ridge at dawn. Filmed Sept 2026."
    row = row_of(conn, delivery_id)
    assert row["status"] == "awaiting_manual_publish"
    assert row["platform_post_id"] is None and row["published_at"] is None

    # PUBLISH_COMPLETE without a post id is "not yet", never "published".
    rec = reconcile_awaiting_manual(
        conn,
        poll=lambda pid: {"status": "PUBLISH_COMPLETE", "publicaly_available_post_id": []},
        permalink_for=lambda pid: f"https://tiktok.invalid/{pid}",
    )
    assert rec.published == 0
    assert row_of(conn, delivery_id)["status"] == "awaiting_manual_publish"


def test_j07_06_default_pack_mode_parks_without_calling_tiktok(conn, asset, monkeypatch, tmp_path):
    import gomoveshift.publishing.handlers as handlers_mod

    def no_api(*_a, **_k):
        raise AssertionError("pack mode must make no TikTok API call")

    monkeypatch.delenv("GMS_TIKTOK_MODE", raising=False)
    monkeypatch.setenv("GMS_TIKTOK_PACK_DIR", str(tmp_path / "TikTok"))
    monkeypatch.setattr(handlers_mod, "init_direct_post", no_api)
    monkeypatch.setattr(handlers_mod, "upload_video_chunks", no_api)
    record = make_record(asset, caption="Ridge at dawn. Filmed Sept 2026.")
    handlers = build_handlers(
        conn,
        loader_for({(PROJECT, 0): record}),
        tiktok=(TikTokCredentials(access_token="act.fake"), asset_loader),  # noqa: S106
        tiktok_rendition_path_loader=lambda rec: asset,
        worker_id="worker-1",
    )
    delivery_id = enqueue(conn, PROJECT, 0, "tiktok", "hash-1")

    result = drain_queue(conn, handlers, owner="worker-1", now=NOW)
    again = drain_queue(conn, handlers, owner="worker-1", now="2026-10-03T13:00:00+00:00")

    assert result.succeeded == 0 and again.succeeded == 0
    row = row_of(conn, delivery_id)
    assert row["status"] == "awaiting_manual_publish"
    assert row["platform_post_id"] is None and row["published_at"] is None
    assert [p.suffix for p in (tmp_path / "TikTok").iterdir() if p.suffix == ".mp4"] == [".mp4"]
