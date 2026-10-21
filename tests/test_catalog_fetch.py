"""``gomoveshift catalog-fetch``: read-only live pull feeding catalog-inventory. Synthetic ids."""

from __future__ import annotations

import json
from types import SimpleNamespace

from gomoveshift.catalog.fetch import fetch_catalog
from gomoveshift.cli.parser import _build_parser


class _Req:
    def __init__(self, payload):
        self.payload = payload

    def execute(self):
        return self.payload


class _FakeVideos:
    def __init__(self, calls):
        self.calls = calls

    def list(self, *, part, id, maxResults):  # noqa: N803 -- mirrors the API kwarg
        ids = id.split(",")
        self.calls.append(ids)
        items = [
            {
                "id": i,
                "snippet": {"title": f"t {i}", "publishedAt": "2025-01-01T00:00:00Z"},
                "contentDetails": {"duration": "PT1M"},
                "status": {"privacyStatus": "public"},
                "recordingDetails": {"recordingDate": "2024-12-31T00:00:00Z"},
            }
            for i in ids
        ]
        return _Req({"items": items})


class _FakeService:
    def __init__(self):
        self.calls = []

    def videos(self):
        return _FakeVideos(self.calls)


def test_fetch_batches_in_fifties_and_shapes_rows():
    svc = _FakeService()
    ids = [f"vid{n:04d}" for n in range(120)]
    cat = fetch_catalog(svc, ids)
    assert [len(c) for c in svc.calls] == [50, 50, 20]
    assert cat["enumerated"] == 120 and cat["listed"] == 120
    row = cat["videos"]["vid0000"]
    assert row["duration_s"] == 60
    assert row["privacy"] == "public"
    assert row["recording_date"] == "2024-12-31T00:00:00Z"
    # the keys catalog-inventory reads
    assert {
        "title",
        "privacy",
        "publish_at",
        "published_at",
        "duration_s",
        "recording_date",
    } <= set(row)


def test_cli_writes_catalog_without_touching_youtube(tmp_path, monkeypatch):
    from gomoveshift.cli.handlers import catalog_fetch as h

    svc = _FakeService()
    seen = {}
    monkeypatch.setattr(h, "_build_readonly_youtube_service", lambda: svc)

    def fake_enum(service, *, known_ids=()):
        seen["known"] = list(known_ids)
        return {"vid0001": {}, "vid0002": {}}

    monkeypatch.setattr(h, "enumerate_uploads", fake_enum)
    known = tmp_path / "known.json"
    known.write_text(json.dumps({"known_ids": ["vid0009"]}))
    out = tmp_path / "sub" / "live_catalog.json"
    args = _build_parser().parse_args(
        ["catalog-fetch", "--out", str(out), "--known-ids", str(known)]
    )
    assert h.cmd_catalog_fetch(args, SimpleNamespace()) == 0
    assert seen["known"] == ["vid0009"]
    data = json.loads(out.read_text())
    assert sorted(data["videos"]) == ["vid0001", "vid0002"]
    assert svc.calls == [["vid0001", "vid0002"]]
