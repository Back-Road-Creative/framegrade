"""Phase D under a ``--replace-scheduled`` plan: no seeded manifests needed, thumbnails only on the
new video, and no clean finish while a plan LF is unfinished."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from gomoveshift.pipeline import replace_scheduled as rs

from test_phase_d_split_skip_upload import _build_state, _run_phase_d
from test_replace_durable_finish import _incidents

AT2 = datetime(2026, 11, 27, 19, 0, tzinfo=timezone.utc).isoformat()


def _plan(state, lf2_state, rec_id):
    out = Path(state["output_base"])
    kept = {"video_id": "pub1", "title": "Kept", "publish_at": "2026-11-20T19:00:00+00:00"}
    old2 = {"video_id": "old2", "title": "Old 2", "publish_at": AT2, "playlist_id": "PL1"}
    rs.write_plan(out, {"schema": rs.REPLACE_PLAN_SCHEMA, "kept": {"1": kept}, "replace": {"2": old2}})  # fmt: skip
    rs.init_progress(out)
    rs.set_progress(out, 2, state=lf2_state, new_id="new2")
    state["youtube_uploads"] = {2: {"video_id": rec_id, "title": "Old 2", "publish_at": AT2}}
    state["skip_csv"] = False
    (out / "thumbnails").mkdir()
    for n in (1, 2):
        (out / "thumbnails" / f"lf_{n:02d}.jpg").write_bytes(b"jpg")


def _run(state, monkeypatch):
    with (
        patch("gomoveshift.pipeline.drive_phases.phase_d._emit_publish_plan_sidecar"),
        patch("gomoveshift.pipeline.drive_phases.phase_d._emit_consolidated_shorts_csv"),
    ):
        _run_phase_d(state, num_lfs=2, monkeypatch=monkeypatch)
    return sys.modules["gomoveshift.pipeline.drive_phases.youtube_uploader"].YouTubeUploader()


def _thumbs(yt):
    return [c.args[0] for c in yt.set_thumbnail.call_args_list]


def test_a_plan_run_needs_no_seeded_manifests(tmp_path, monkeypatch):
    state = _build_state(tmp_path, num_lfs=2)
    _plan(state, rs.DONE, "new2")
    assert not state.get("upload_manifests")
    _run(state, monkeypatch)  # Rule 9 would raise on 2/2 LFs missing manifests


def test_rule_9_still_binds_a_non_plan_run(tmp_path):
    from types import SimpleNamespace

    state = {"upload_manifests": {}, "shorts_per_lf": {}, "skip_upload": False}
    plan = SimpleNamespace(regular_lfs=[SimpleNamespace(index=1)])
    from gomoveshift.pipeline.quality_gates import check_shorts_upload_complete

    assert check_shorts_upload_complete(state, plan)


def test_an_abandoned_lf_old_video_gets_no_thumbnail(tmp_path, monkeypatch):
    state = _build_state(tmp_path, num_lfs=2)
    _plan(state, rs.ABANDONED, "old2")
    assert _thumbs(_run(state, monkeypatch)) == []


def test_the_new_video_is_thumbnailed_once(tmp_path, monkeypatch):
    state = _build_state(tmp_path, num_lfs=2)
    _plan(state, rs.DONE, "new2")
    assert _thumbs(_run(state, monkeypatch)) == ["new2"]
    (tmp_path / "again").mkdir()
    state2 = _build_state(tmp_path / "again", num_lfs=2)
    _plan(state2, rs.DONE, "new2")
    state2["youtube_uploads"][2]["thumbnail_path"] = "x.jpg"  # already attached by an earlier entry
    assert _thumbs(_run(state2, monkeypatch)) == []


@pytest.mark.parametrize("lf_state", [rs.PENDING, rs.UPLOADED, rs.OLD_DELETED, rs.SLOT_PASSED])
def test_phase_d_does_not_succeed_while_a_plan_lf_is_unfinished(tmp_path, monkeypatch, lf_state):
    state = _build_state(tmp_path, num_lfs=2)
    _plan(state, lf_state, "old2")
    from gomoveshift.pipeline.drive_phases.phase_d import _assert_replace_finished

    with pytest.raises(RuntimeError, match="REPLACE-UNFINISHED"):
        _assert_replace_finished(Path(state["output_base"]))
    (inc,) = _incidents("unfinished")
    assert f"LF 2 {lf_state}" in inc["error"]


@pytest.mark.parametrize("lf_state", [rs.DONE, rs.ABANDONED])
def test_a_finished_or_abandoned_plan_is_a_clean_phase(tmp_path, lf_state):
    from gomoveshift.pipeline.drive_phases.phase_d import _assert_replace_finished

    state = _build_state(tmp_path, num_lfs=2)
    _plan(state, lf_state, "new2")
    _assert_replace_finished(Path(state["output_base"]))
    assert not _incidents("unfinished")


def test_phase_d_run_raises_and_opens_an_incident_for_a_stopped_lf(tmp_path, monkeypatch):
    state = _build_state(tmp_path, num_lfs=2)
    _plan(state, rs.SLOT_PASSED, "new2")
    with pytest.raises(RuntimeError, match="REPLACE-UNFINISHED"):
        _run(state, monkeypatch)
    assert _incidents("unfinished")
