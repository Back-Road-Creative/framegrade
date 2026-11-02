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
    assert "skip" in out and "legacy: no lf_provenance.json and no camera clips" in out
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


def _drive_log(base: Path, *names):
    lines = [f"19:14:07 [INFO]   Extracted GoPro GPS: {n}_gopro.gpx" for n in names]
    (base / "drive.log").write_text("\n".join(lines) + "\nVID_20250915_091226_00_030 again\n")


def test_old_provenance_without_clips_falls_back_to_drive_log(tmp_path, registry):
    root = tmp_path / "out"
    base = root / "old"
    base.mkdir(parents=True)
    (base / "lf_provenance.json").write_text(
        json.dumps({"lfs": [{"lf_index": 1, "source": {"episode_paths": ["/cache/e.mp4"]}}]})
    )
    (base / "youtube_uploads.json").write_text(json.dumps({"1": {"video_id": "yt1"}}))
    _drive_log(base, "GX010064", "GX020064")
    rc, out, _ = _run(root, apply=True)
    assert rc == 0
    assert "old lf1 GX010064 yt1 (drive_log_project)" in out
    rows = load_claims()
    assert sorted(rows) == ["GX010064", "GX020064", "VID_20250915_091226_00_030"]
    assert rows["GX010064"][0]["basis"] == "drive_log_project"
    assert rows["GX010064"][0]["clip_size_bytes"] is None
    before = registry.read_text()
    _run(root, apply=True)
    assert registry.read_text() == before


def test_no_provenance_claims_project_clips_for_every_upload(tmp_path, registry):
    root = tmp_path / "out"
    base = _project(
        root, "umbrella", uploads={"1": {"video_id": "a"}, "2": {"video_id": "b"}}, provenance=False
    )
    _drive_log(base, "GX010060")
    rc, out, _ = _run(root, apply=True)
    assert rc == 0
    assert {(r["lf_index"], r["video_id"]) for r in load_claims()["GX010060"]} == {
        (1, "a"),
        (2, "b"),
    }


def test_provenance_rows_are_tagged_and_win_over_drive_log(tmp_path, registry):
    root = tmp_path / "out"
    base = _project(root, "pub", uploads={"1": {"video_id": "yt1"}})
    _drive_log(base, "GX999999")
    _run(root, apply=True)
    claims = load_claims()
    assert "GX999999" not in claims
    assert claims["VID_1.mp4"][0]["basis"] == "lf_provenance"


def test_gate_matches_extensionless_claim_case_insensitively(tmp_path, registry):
    from gomoveshift.pipeline.clip_claims import find_reused_clips

    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps(
            {
                "project": "old",
                "lf_index": 1,
                "video_id": "yt1",
                "clip_name": "GX010064",
                "clip_size_bytes": None,
                "basis": "drive_log_project",
            }
        )
        + "\n"
    )
    plan = {"episodes": [{"recordings": [{"chapters": [{"path": "/x/GX010064.MP4"}]}]}]}
    found = find_reused_clips(plan, "new")
    assert [c["claimed_by"] for c in found] == ["old"]
    assert found[0]["basis"] == "drive_log_project"
    assert find_reused_clips(plan, "old") == []
