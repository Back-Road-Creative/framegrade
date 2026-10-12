"""Tests for the videos.yaml UPSERT generator (no live creds)."""

from __future__ import annotations

import pytest

from gomoveshift.pipeline import _atomic_write
from gomoveshift.pipeline.site_sync.videos_yaml import (
    dump_videos_yaml,
    load_videos_yaml,
    upsert_videos_yaml,
)


def _manifest(*videos):
    return {"videos": list(videos)}


def _v(vid, category, title="", published_at="2026-01-01T00:00:00Z", seconds=780):
    return {
        "id": vid,
        "title": title,
        "category": category,
        "published_at": published_at,
        "duration_seconds": seconds,
    }


def test_preserve_existing_title_and_summary():
    existing = {
        "videos": [
            {
                "id": "a",
                "title": "Human Title",
                "date": "2026-04-29",
                "duration_min": 13,
                "summary": "Human summary.",
            }
        ]
    }
    manifest = _manifest(_v("a", "real_lf", title="API Title", seconds=900))
    entry = upsert_videos_yaml(manifest, existing)["videos"][0]
    # Human-authored fields are kept verbatim; present date/duration not clobbered.
    assert entry["title"] == "Human Title"
    assert entry["summary"] == "Human summary."
    assert entry["date"] == "2026-04-29"
    assert entry["duration_min"] == 13


def test_fill_missing_date_and_duration():
    existing = {"videos": [{"id": "a", "title": "T", "summary": "S"}]}
    manifest = _manifest(_v("a", "real_lf", published_at="2026-05-10T00:00:00Z", seconds=1860))
    entry = upsert_videos_yaml(manifest, existing)["videos"][0]
    assert entry["date"] == "2026-05-10"
    assert entry["duration_min"] == 31


def test_add_missing_real_lf():
    manifest = _manifest(
        _v("new", "real_lf", title="New LF", published_at="2026-06-01T00:00:00Z", seconds=600)
    )
    assert upsert_videos_yaml(manifest, None)["videos"] == [
        {"id": "new", "title": "New LF", "date": "2026-06-01", "duration_min": 10, "summary": ""}
    ]


def test_exclude_shorts_streams_held():
    manifest = _manifest(
        _v("lf", "real_lf", title="LF"),
        _v("s", "short", title="Short"),
        _v("st", "stream", title="Stream"),
        _v("h", "held", title="Held"),
    )
    assert [v["id"] for v in upsert_videos_yaml(manifest, None)["videos"]] == ["lf"]


def test_keep_existing_absent_from_manifest():
    existing = {
        "videos": [
            {"id": "old", "title": "Old", "date": "2026-03-01", "duration_min": 5, "summary": "x"}
        ]
    }
    manifest = _manifest(_v("new", "real_lf", published_at="2026-06-01T00:00:00Z"))
    ids = {v["id"] for v in upsert_videos_yaml(manifest, existing)["videos"]}
    assert ids == {"old", "new"}


def test_dedup_by_id():
    existing = {
        "videos": [
            {"id": "a", "title": "First", "date": "2026-01-01", "duration_min": 1, "summary": "1"},
            {"id": "a", "title": "Dupe", "date": "2026-01-02", "duration_min": 2, "summary": "2"},
        ]
    }
    out = upsert_videos_yaml(_manifest(), existing)["videos"]
    assert len(out) == 1
    assert out[0]["title"] == "First"


def test_descending_date_order():
    manifest = _manifest(
        _v("old", "real_lf", published_at="2026-01-01T00:00:00Z"),
        _v("new", "real_lf", published_at="2026-12-01T00:00:00Z"),
        _v("mid", "real_lf", published_at="2026-06-01T00:00:00Z"),
    )
    assert [v["id"] for v in upsert_videos_yaml(manifest, None)["videos"]] == ["new", "mid", "old"]


def test_load_and_dump_roundtrip(tmp_path):
    path = tmp_path / "data" / "videos.yaml"
    data = {
        "videos": [
            {"id": "a", "title": "T", "date": "2026-04-29", "duration_min": 13, "summary": "S"}
        ]
    }
    dump_videos_yaml(data, path)
    assert load_videos_yaml(path) == data
    assert load_videos_yaml(tmp_path / "nope.yaml") is None


def test_dump_survives_crash_mid_write(tmp_path, monkeypatch):
    """An interrupted rewrite leaves the previous library on disk, not a stub."""
    path = tmp_path / "videos.yaml"
    old = {"videos": [{"id": "a", "title": "Old"}]}
    dump_videos_yaml(old, path)
    before = path.read_text()

    monkeypatch.setattr(
        _atomic_write.os,
        "replace",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("simulated crash mid-write")),
    )
    with pytest.raises(RuntimeError, match="simulated crash mid-write"):
        dump_videos_yaml({"videos": [{"id": "b", "title": "New"}]}, path)

    assert path.read_text() == before
    assert [p.name for p in tmp_path.iterdir()] == ["videos.yaml"]  # no orphan tmp
