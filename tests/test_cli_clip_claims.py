"""``gomoveshift clip-claims backfill``: rebuild the registry from past outputs."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from gomoveshift.cli.handlers.clip_claims import run_clip_claims_backfill
from gomoveshift.pipeline import clip_claims
from gomoveshift.pipeline.clip_claims import load_claims


@pytest.fixture
def registry(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "seo" / "gomoveshift" / "clip_claims.jsonl"
    monkeypatch.setattr(clip_claims, "claims_path", lambda: path)
    return path


def _clip(name, size):
    return {"name": name, "path": f"/drive/{name}", "size_bytes": size}


def _project(root: Path, name, uploads=None, provenance=True):
    base = root / name
    base.mkdir(parents=True)
    if provenance:
        lfs = [
            {"lf_index": 1, "source": {"clips": [_clip("VID_1.mp4", 10), _clip("VID_2.mp4", 20)]}},
            {"lf_index": 2, "source": {"clips": [_clip("VID_3.mp4", 30)]}},
        ]
        (base / "lf_provenance.json").write_text(json.dumps({"lfs": lfs}))
    if uploads is not None:
        (base / "youtube_uploads.json").write_text(json.dumps(uploads))
    return base


def _run(root, apply=False):
    out, err = io.StringIO(), io.StringIO()
    rc = run_clip_claims_backfill(roots=[root], apply=apply, out=out, err=err)
    return rc, out.getvalue(), err.getvalue()


def test_dry_run_reports_and_writes_nothing(tmp_path, registry):
    root = tmp_path / "out"
    _project(root, "pub", uploads={"1": {"video_id": "yt1"}})
    rc, out, _ = _run(root)
    assert rc == 0
    assert "would add pub lf1 VID_1.mp4 yt1" in out
    assert "dry-run" in out
    assert not registry.exists()


def test_apply_writes_published_lfs_only_and_is_idempotent(tmp_path, registry):
    root = tmp_path / "out"
    _project(root, "pub", uploads={"1": {"video_id": "yt1"}})
    rc, _, _ = _run(root, apply=True)
    assert rc == 0
    claims = load_claims()
    assert sorted(claims) == ["VID_1.mp4", "VID_2.mp4"]  # lf 2 has no video_id
    assert claims["VID_1.mp4"][0]["project"] == "pub"
    assert claims["VID_1.mp4"][0]["clip_size_bytes"] == 10
    before = registry.read_text()
    rc, out, _ = _run(root, apply=True)
    assert rc == 0 and registry.read_text() == before
    assert "added 0" in out


def test_unpublished_and_no_provenance_folders_are_skipped_with_reason(tmp_path, registry):
    root = tmp_path / "out"
    _project(root, "unpub")  # provenance, no youtube_uploads.json
    _project(root, "legacy", uploads={"1": {"video_id": "yt9"}}, provenance=False)
    rc, out, _ = _run(root, apply=True)
    assert rc == 0
    assert "skip" in out and "legacy: no lf_provenance.json" in out
    assert "unpub: nothing new" in out
    assert not registry.exists()


def test_malformed_provenance_is_skipped_not_fatal(tmp_path, registry):
    root = tmp_path / "out"
    base = _project(root, "bad", uploads={"1": {"video_id": "yt1"}})
    (base / "lf_provenance.json").write_text("{not json")
    rc, out, _ = _run(root, apply=True)
    assert rc == 0 and "bad: unreadable" in out


def test_missing_root_exits_2(tmp_path, registry):
    rc, _, err = _run(tmp_path / "nope")
    assert rc == 2 and "not a directory" in err


def test_subparser_is_registered():
    from gomoveshift.cli import _COMMAND_MAP, _build_parser

    assert "clip-claims" in _COMMAND_MAP
    args = _build_parser().parse_args(["clip-claims", "backfill", "/a", "/b"])
    assert args.command == "clip-claims"
    assert args.clip_claims_action == "backfill"
    assert args.roots == [Path("/a"), Path("/b")]
    assert args.apply is False
    assert _build_parser().parse_args(["clip-claims", "backfill", "/a", "--apply"]).apply is True
