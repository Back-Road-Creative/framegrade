"""Smoke test for the sync-site handler (no live creds — service injected)."""

from __future__ import annotations

import io
import json
import textwrap

import yaml

import gomoveshift.cli.handlers.sync_site as sync_site
from gomoveshift.pipeline.journal_publisher import load_journey
from gomoveshift.pipeline.orchestrator_lock import _acquire_run_lock, _release_run_lock

_FAKE_MANIFEST = {
    "generated_at": "2026-06-27T00:00:00+00:00",
    "source": "youtube-data-api/uploads-playlist",
    "counts": {"total": 2, "short": 1, "stream": 0, "real_lf": 1, "held": 0},
    "videos": [
        {
            "id": "lf1",
            "title": "A Real Long Form",
            "published_at": "2026-06-01T12:00:00Z",
            "duration_seconds": 780,
            "category": "real_lf",
        },
        {
            "id": "sh1",
            "title": "A Short",
            "published_at": "2026-06-02T12:00:00Z",
            "duration_seconds": 45,
            "category": "short",
        },
    ],
}


_ALL_CATEGORY_MANIFEST = {
    "generated_at": "2026-07-31T08:17:06.164454+00:00",
    "source": "youtube-data-api/uploads-playlist",
    "counts": {"total": 4, "short": 1, "stream": 1, "real_lf": 1, "held": 1},
    # Deliberately NOT in sorted id order — the slim manifest sorts its own id set.
    "videos": [
        {
            "id": "zz_real_lf",
            "title": "A Real Long Form",
            "published_at": "2026-07-04T12:00:00Z",
            "duration_seconds": 780,
            "category": "real_lf",
        },
        {
            "id": "mm_stream",
            "title": "Dash Cam Live",
            "published_at": "2026-07-03T12:00:00Z",
            "duration_seconds": 30000,
            "category": "stream",
        },
        {
            "id": "aa_short",
            "title": "A Short",
            "published_at": "2026-07-02T12:00:00Z",
            "duration_seconds": 30,
            "category": "short",
        },
        {
            "id": "bb_held",
            "title": "Scheduled, Not Public Yet",
            "published_at": "2026-07-01T12:00:00Z",
            "duration_seconds": 900,
            "category": "held",
        },
    ],
}


def _patch_manifest(monkeypatch):
    monkeypatch.setattr(sync_site, "build_channel_manifest", lambda service, *, now: _FAKE_MANIFEST)


def test_dry_run_writes_nothing(monkeypatch, tmp_path):
    _patch_manifest(monkeypatch)
    rc = sync_site.run_sync_site(site_path=tmp_path, apply=False, service=object())
    assert rc == 0
    assert not (tmp_path / "data" / "videos.yaml").exists()
    assert not (tmp_path / "data" / "channel_manifest.json").exists()


def test_apply_writes_library_and_manifest(monkeypatch, tmp_path):
    _patch_manifest(monkeypatch)
    rc = sync_site.run_sync_site(site_path=tmp_path, apply=True, service=object())
    assert rc == 0
    library = yaml.safe_load((tmp_path / "data" / "videos.yaml").read_text())
    assert [v["id"] for v in library["videos"]] == ["lf1"]  # short excluded
    assert library["videos"][0]["title"] == "A Real Long Form"
    manifest = json.loads((tmp_path / "data" / "channel_manifest.json").read_text())
    assert manifest["counts"]["real_lf"] == 1
    assert manifest["real_lf_ids"] == ["lf1"]  # slim manifest carries the id set, not full videos
    assert "videos" not in manifest  # slim — full uploads list is not committed


def test_slim_manifest_carries_every_known_channel_id(monkeypatch, tmp_path):
    """The library is long-form-only, so the site's coverage gate can only tell a
    legitimately embedded Short/stream/held clip from an id that exists nowhere (a
    typo'd or deleted embed) if the slim manifest projects the WHOLE channel's ids."""
    monkeypatch.setattr(
        sync_site, "build_channel_manifest", lambda service, *, now: _ALL_CATEGORY_MANIFEST
    )
    rc = sync_site.run_sync_site(site_path=tmp_path, apply=True, service=object())
    assert rc == 0
    manifest = json.loads((tmp_path / "data" / "channel_manifest.json").read_text())
    # One id of EVERY category lands in known_ids, sorted for a stable daily diff.
    assert manifest["known_ids"] == ["aa_short", "bb_held", "mm_stream", "zz_real_lf"]
    # ... and the long-form-only key keeps its existing shape (the checker reads it).
    assert manifest["real_lf_ids"] == ["zz_real_lf"]
    assert manifest["generated_at"] == _ALL_CATEGORY_MANIFEST["generated_at"]
    assert "videos" not in manifest  # still slim


def test_apply_serializes_against_concurrent_writer(monkeypatch, tmp_path):
    """A second writer (cron site-sync overlapping a live drive run) must never
    lose the first writer's row: the read (existing videos.yaml) and the write
    are one critical section. Simulate the overlap by holding the same lock a
    concurrent process would hold, spanning this process's whole
    load->modify->save window, and assert the second run refuses loudly
    (non-zero exit, no write) instead of silently clobbering the first run's
    already-written row with a stale snapshot.
    """
    _patch_manifest(monkeypatch)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    videos_path = data_dir / "videos.yaml"
    # First writer already landed its row (e.g. a live drive run's own sync).
    videos_path.write_text("videos:\n- id: from_other_writer\n  title: Other Writer's LF\n")

    # Hold the exclusive lock a concurrent writer would hold across its own
    # load->modify->save window.
    fd, lock_path = _acquire_run_lock(data_dir, data_dir, lock_name="videos.yaml.lock")
    try:
        buf = io.StringIO()
        rc = sync_site.run_sync_site(site_path=tmp_path, apply=True, service=object(), err=buf)
    finally:
        _release_run_lock(fd, lock_path)

    assert rc != 0, "a busy writer must be a loud refusal, not a silent success"
    library = yaml.safe_load(videos_path.read_text())
    assert [v["id"] for v in library["videos"]] == ["from_other_writer"], (
        "the second writer's stale snapshot must never overwrite the first writer's row"
    )
    assert "lock" in buf.getvalue().lower()


def test_journeys_flag_embeds_routed_lf(monkeypatch, tmp_path):
    _patch_manifest(monkeypatch)
    (tmp_path / "data").mkdir()
    journeys = tmp_path / "content" / "journeys"
    journeys.mkdir(parents=True)
    (tmp_path / "data" / "video_routing.yaml").write_text(
        textwrap.dedent(
            """\
            routes:
              lf1:
                journey: trip
                date: '2026-06-01'
            """
        )
    )
    (journeys / "trip.md").write_text(
        textwrap.dedent(
            """\
            ---
            title: "Trip"
            days:
            - day_number: 1
              date: '2026-06-01'
              videos: []
            video_count: 0
            ---
            Body.
            """
        )
    )
    rc = sync_site.run_sync_site(site_path=tmp_path, apply=True, journeys=True, service=object())
    assert rc == 0
    doc, _ = load_journey(journeys / "trip.md")
    assert [v["youtube_id"] for v in doc["days"][0]["videos"]] == ["lf1"]
    assert doc["video_count"] == 1
