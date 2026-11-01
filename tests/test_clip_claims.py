"""Clip-claims registry and the Phase A refuse-on-reuse gate."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from test_phase_a_score_cache import _episode, _make_episode_files, _patch_phase_a, _state

from _state_helpers import make_state
from gomoveshift.config import Config
from gomoveshift.pipeline import clip_claims
from gomoveshift.pipeline.clip_claims import (
    ALLOW_CLIP_REUSE_ENV,
    ClipClaimsUnreadable,
    find_reused_clips,
    load_claims,
    record_claims,
)
from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator
from gomoveshift.pipeline.drive_phases import phase_a as phase_a_mod
from gomoveshift.pipeline.drive_phases.phase_a import PhaseA


@pytest.fixture
def registry(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "seo" / "gomoveshift" / "clip_claims.jsonl"
    monkeypatch.setattr(clip_claims, "claims_path", lambda: path)
    return path


def _project(tmp_path: Path, name="proj_a", lfs=None) -> Path:
    base = tmp_path / name
    base.mkdir()
    lfs = lfs or [
        {"lf_index": 1, "source": {"clips": [_clip("VID_1.mp4", 10), _clip("VID_2.mp4", 20)]}},
        {"lf_index": 2, "source": {"clips": [_clip("VID_3.mp4", 30)]}},
    ]
    (base / "lf_provenance.json").write_text(json.dumps({"lfs": lfs}))
    return base


def _clip(name, size):
    return {"name": name, "path": f"/drive/{name}", "size_bytes": size}


def _rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()]


# --- record_claims ------------------------------------------------------------


def test_record_claims_one_row_per_clip_for_uploaded_lfs_only(tmp_path, registry):
    base = _project(tmp_path)
    assert record_claims(base, {1: {"video_id": "abc"}, 2: {"video_id": None}}) == 2
    rows = _rows(registry)
    assert [r["clip_name"] for r in rows] == ["VID_1.mp4", "VID_2.mp4"]
    assert rows[0]["project"] == "proj_a" and rows[0]["lf_index"] == 1
    assert rows[0]["video_id"] == "abc" and rows[0]["clip_size_bytes"] == 10
    assert rows[0]["clip_path"] == "/drive/VID_1.mp4" and rows[0]["claimed_at"]


def test_record_claims_is_idempotent(tmp_path, registry):
    base = _project(tmp_path)
    assert record_claims(base, {1: {"video_id": "abc"}}) == 2
    assert record_claims(base, {1: {"video_id": "abc"}}) == 0
    assert len(_rows(registry)) == 2


def test_record_claims_accepts_string_keys(tmp_path, registry):
    base = _project(tmp_path)
    assert record_claims(base, {"2": {"video_id": "zzz"}}) == 1
    assert _rows(registry)[0]["clip_name"] == "VID_3.mp4"


def test_record_claims_missing_sidecar_or_clips_returns_zero(tmp_path, registry):
    empty = tmp_path / "nosidecar"
    empty.mkdir()
    assert record_claims(empty, {1: {"video_id": "abc"}}) == 0
    noclips = _project(tmp_path, "noclips", lfs=[{"lf_index": 1, "source": {"clips": []}}])
    assert record_claims(noclips, {1: {"video_id": "abc"}}) == 0
    assert not registry.exists()


def test_record_claims_never_raises_on_unreadable_registry(tmp_path, registry):
    registry.mkdir(parents=True)  # a directory where the file should be
    assert record_claims(_project(tmp_path), {1: {"video_id": "abc"}}) == 0


# --- load_claims --------------------------------------------------------------


def test_load_claims_missing_file_is_empty(registry):
    assert load_claims() == {}


def test_load_claims_skips_malformed_line(registry):
    registry.parent.mkdir(parents=True)
    good = {"clip_name": "VID_1.mp4", "project": "p"}
    registry.write_text("not json\n" + json.dumps(good) + "\n")
    assert load_claims() == {"VID_1.mp4": [good]}


def test_load_claims_raises_when_registry_is_a_directory(registry):
    registry.mkdir(parents=True)
    with pytest.raises(ClipClaimsUnreadable):
        load_claims()


def test_load_claims_raises_on_non_utf8(registry):
    registry.parent.mkdir(parents=True)
    registry.write_bytes(b"\xff\xfe\x00bad\n")
    with pytest.raises(ClipClaimsUnreadable):
        load_claims()


# --- find_reused_clips --------------------------------------------------------


def _plan(tmp_path: Path, name: str, size: int | None):
    p = tmp_path / "drive" / name
    if size is not None:
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(b"x" * size)
    ch = SimpleNamespace(path=p)
    rec = SimpleNamespace(chapters=[ch])
    return SimpleNamespace(episodes=[SimpleNamespace(recordings=[rec])], filtered_clips=[])


def _claim(registry: Path, project="other", name="VID_1.mp4", size=10):
    registry.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "project": project,
        "lf_index": 3,
        "video_id": "vid9",
        "clip_name": name,
        "clip_size_bytes": size,
    }
    with open(registry, "a") as fh:
        fh.write(json.dumps(row) + "\n")


def test_same_project_claim_is_ignored(tmp_path, registry):
    _claim(registry, project="mine")
    assert find_reused_clips(_plan(tmp_path, "VID_1.mp4", 10), "mine") == []


def test_other_project_equal_size_conflicts(tmp_path, registry):
    _claim(registry)
    assert find_reused_clips(_plan(tmp_path, "VID_1.mp4", 10), "mine") == [
        {"clip_name": "VID_1.mp4", "claimed_by": "other", "lf_index": 3, "video_id": "vid9"}
    ]


def test_same_name_different_size_is_a_different_file(tmp_path, registry):
    _claim(registry, size=10)
    assert find_reused_clips(_plan(tmp_path, "VID_1.mp4", 11), "mine") == []


def test_unknown_size_on_either_side_conflicts(tmp_path, registry):
    _claim(registry, size=None)
    assert len(find_reused_clips(_plan(tmp_path, "VID_1.mp4", 11), "mine")) == 1
    _claim(registry, name="VID_2.mp4", size=5)
    assert len(find_reused_clips(_plan(tmp_path, "VID_2.mp4", None), "mine")) == 1


# --- Phase A gate -------------------------------------------------------------

CONFLICT = {"clip_name": "VID_9.mp4", "claimed_by": "oldproj", "lf_index": 2, "video_id": "yt123"}


def _gate_setup(tmp_path, monkeypatch, conflicts):
    stages = tmp_path / "stages"
    stages.mkdir()
    files = _make_episode_files(tmp_path, n=1)
    _patch_phase_a(monkeypatch, [_episode(0, files[0])])
    monkeypatch.setattr(phase_a_mod, "find_reused_clips", lambda plan, project: conflicts)

    def _boom(*a, **k):
        raise AssertionError("execute_ingest must not run after a refused clip")

    return stages, _boom


def test_phase_a_refuses_reused_clip_before_ingest(tmp_path, monkeypatch):
    monkeypatch.delenv(ALLOW_CLIP_REUSE_ENV, raising=False)
    stages, boom = _gate_setup(tmp_path, monkeypatch, [CONFLICT])
    monkeypatch.setattr(phase_a_mod, "execute_ingest", boom)
    with pytest.raises(RuntimeError) as exc:
        PhaseA(Config()).run(_state(tmp_path, stages_base=stages))
    msg = str(exc.value)
    assert "VID_9.mp4" in msg and "oldproj" in msg and "yt123" in msg
    assert ALLOW_CLIP_REUSE_ENV in msg


def test_phase_a_override_warns_and_continues(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv(ALLOW_CLIP_REUSE_ENV, "1")
    stages, _ = _gate_setup(tmp_path, monkeypatch, [CONFLICT])
    with caplog.at_level(logging.WARNING):
        PhaseA(Config()).run(_state(tmp_path, stages_base=stages))
    assert any("VID_9.mp4" in r.getMessage() for r in caplog.records)


def test_phase_a_unreadable_registry_refuses(tmp_path, monkeypatch):
    monkeypatch.delenv(ALLOW_CLIP_REUSE_ENV, raising=False)
    stages, boom = _gate_setup(tmp_path, monkeypatch, [])

    def _unreadable(plan, project):
        raise ClipClaimsUnreadable("clip claims registry unreadable at /x: boom")

    monkeypatch.setattr(phase_a_mod, "find_reused_clips", _unreadable)
    monkeypatch.setattr(phase_a_mod, "execute_ingest", boom)
    with pytest.raises(RuntimeError, match="unreadable at /x"):
        PhaseA(Config()).run(_state(tmp_path, stages_base=stages))


# --- _save_state writer hook ----------------------------------------------------


def test_save_state_records_claims_after_sidecar_write(tmp_path):
    orch = DrivePipelineOrchestrator(config=Config())
    out = tmp_path / "out"
    out.mkdir()
    uploads = {1: {"video_id": "abc"}}
    state = make_state(phases_completed=[], output_base=out, youtube_uploads=uploads)
    with (
        patch("gomoveshift.pipeline.drive_orchestrator.save_drive_state"),
        patch("gomoveshift.pipeline.drive_orchestrator.save_sidecar", return_value="sha"),
        patch("gomoveshift.pipeline.drive_orchestrator.record_claims") as rec,
    ):
        orch._save_state(tmp_path / "state.json", state)
    rec.assert_called_once_with(out, uploads)


def test_save_state_survives_a_raising_record_claims(tmp_path):
    orch = DrivePipelineOrchestrator(config=Config())
    out = tmp_path / "out"
    out.mkdir()
    state = make_state(phases_completed=[], output_base=out, youtube_uploads={1: {}})
    with (
        patch("gomoveshift.pipeline.drive_orchestrator.save_drive_state"),
        patch("gomoveshift.pipeline.drive_orchestrator.save_sidecar", return_value="sha"),
        patch(
            "gomoveshift.pipeline.drive_orchestrator.record_claims",
            side_effect=RuntimeError("boom"),
        ),
    ):
        orch._save_state(tmp_path / "state.json", state)
    assert state.youtube_uploads_sha == "sha"
