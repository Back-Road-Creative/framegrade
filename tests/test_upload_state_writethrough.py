"""Write-through of upload bookkeeping into ``state``.

Two resume-safety bugs where a per-phase-D working dict diverged from the
``state`` object the persisted checkpoint is built from:

Aliasing: ``youtube_uploads`` aliasing clobber (duplicate-upload risk).
     ``upload_if_needed`` writes each LF record straight into
     ``state.youtube_uploads``. When Phase D's per-LF loop worked a
     *separate* local dict, an LF whose ``videos.insert`` succeeded but
     whose ``playlistItems.insert`` then failed (wrapper returns None) had
     its record land only in ``state``; the next successful LF's
     ``state.youtube_uploads = <local>`` reassignment clobbered it, so
     resume re-ran ``videos.insert`` and produced a duplicate scheduled
     video. Fix: the loop operates on ``state``'s own dict.

Manifests: ``upload_manifests`` persisted only at end-of-phase. A mid-loop raise
     (a Drive failure here, or the recoverable ``YouTubeQuotaError`` exit
     later in Phase D that saves state) discarded every accumulated
     manifest, so skip-when-complete never fired on resume and ALL
     shorts re-uploaded to Drive. Fix: write manifests through to ``state``
     and persist per-LF before the next iteration.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.models import DrivePlan, LFSegment, RegularLFPlan
from gomoveshift.pipeline.drive_phases import external_systems_coordinator as esc
from gomoveshift.pipeline.drive_phases import phase_d as phase_d_mod
from gomoveshift.pipeline.drive_phases.phase_d import (
    _upload_all_lfs_shorts_to_drive,
    _upload_lfs_to_youtube,
)
from gomoveshift.pipeline.uploader import UploadManifest

from _state_helpers import make_state


# ─── youtube_uploads write-through ─────────────────────────────────


def _lf(idx):
    return SimpleNamespace(index=idx, total_duration=1800.0)


def test_youtube_record_survives_playlist_failure_then_next_lf(tmp_path):
    """LF1's video record (playlist insert failed → wrapper returns None)
    must remain in ``state`` after LF2 succeeds — otherwise resume re-runs
    ``videos.insert`` for LF1 and duplicates the scheduled video."""
    drive_plan = SimpleNamespace(regular_lfs=[_lf(1), _lf(2)])
    metadata = {i: {"title": f"T{i}", "description": "d", "tags": []} for i in (1, 2)}
    sched = SimpleNamespace(
        lf_slots=[SimpleNamespace(datetime_est="2026-03-15T11:00:00") for _ in range(2)]
    )
    state = make_state()

    def fake_upload_lf(*, yt, state, lf, **kwargs):
        # Mimic upload_if_needed: the record is written into state's OWN
        # youtube_uploads dict BEFORE the playlist insert (``youtube_uploader.upload_if_needed``).
        # For LF1 the playlist insert then fails, so the
        # wrapper returns None while the record is already persisted in state.
        uploads = state.youtube_uploads
        if uploads is None:
            uploads = {}
            state.youtube_uploads = uploads
        uploads[lf.index] = {"video_id": f"vid_{lf.index}"}
        return None if lf.index == 1 else {"video_id": f"vid_{lf.index}"}

    # Replicate Phase D's caller-side working dict so the loop is proven
    # robust to a divergent seed.
    youtube_uploads_arg = state.youtube_uploads or {}

    with (
        patch.object(
            esc,
            "_resolve_youtube_uploader_class",
            return_value=MagicMock(return_value=MagicMock()),
        ),
        patch.object(esc, "_upload_lf_to_youtube", side_effect=fake_upload_lf),
    ):
        _upload_lfs_to_youtube(
            drive_plan=drive_plan,
            regular_lf_paths=["/v/lf1.mp4", "/v/lf2.mp4"],
            metadata_per_lf=metadata,
            drive_schedule=sched,
            youtube_uploads=youtube_uploads_arg,
            state=state,
            output_base=tmp_path,
            config=MagicMock(),
            save_state=lambda s: None,
            skip_youtube_upload=False,
            youtube_upload_enabled=True,
        )

    assert 1 in state["youtube_uploads"], (
        "LF1 record was clobbered — resume would re-run videos.insert (duplicate)"
    )
    assert state["youtube_uploads"][1]["video_id"] == "vid_1"
    assert 2 in state["youtube_uploads"]


# ─── upload_manifests incremental persist ──────────────────────────


def _make_lf(index: int) -> RegularLFPlan:
    seg = LFSegment(
        source_file_index=index,
        source_path=Path(f"/tmp/ep_{index:03d}_FINAL.mp4"),
        start_time=0.0,
        end_time=600.0,
        original_start_time=0.0,
        original_end_time=600.0,
        composite_score=0.7,
        gps_center_lat=0.0,
        gps_center_lon=0.0,
        terrain_type="countryside",
        contiguous_with_previous=False,
    )
    return RegularLFPlan(
        index=index,
        segments=[seg],
        montage_peaks=[seg],
        shorts_target=2,
        region="usa",
        total_duration=600.0,
    )


def _build_state_and_plan(tmp_path: Path, num_lfs: int = 2):
    output_base = tmp_path / "output"
    output_base.mkdir()
    stages_dir = tmp_path / "cache" / "_stages"
    day_shorts_dir = stages_dir / "day_2026-04-29" / "shorts"
    day_shorts_dir.mkdir(parents=True)

    shorts_per_lf: dict[int, list[Path]] = {}
    short_counter = 0
    regular_lfs = []
    for i in range(num_lfs):
        regular_lfs.append(_make_lf(i + 1))
        per_lf_shorts: list[Path] = []
        for _j in range(2):
            short_counter += 1
            sp = day_shorts_dir / f"short_{short_counter:02d}.mp4"
            sp.write_bytes(b"y" * 64)
            per_lf_shorts.append(sp)
        shorts_per_lf[i + 1] = per_lf_shorts

    drive_plan = DrivePlan(
        source_files=[],
        processed_files=[],
        regular_lfs=regular_lfs,
        score_threshold=0.5,
        total_source_duration=600.0 * num_lfs,
        total_highlight_duration=600.0 * num_lfs,
    )
    state = make_state(shorts_per_lf=shorts_per_lf)
    return state, drive_plan, output_base


def test_manifest_persisted_to_state_before_midloop_raise(tmp_path):
    """LF1's manifest must be on ``state`` (and save_state must have fired)
    before LF2's Drive failure aborts the loop — otherwise resume loses it
    and re-uploads every short."""
    state, drive_plan, output_base = _build_state_and_plan(tmp_path, num_lfs=2)
    save_snapshots: list[dict] = []

    def fake_inner_upload(*, uploader, lf, project_name, shorts_paths, output_base, use_dayfirst):
        if lf.index == 2:
            raise RuntimeError("drive failure on LF2")
        return UploadManifest(
            project_name=project_name,
            folder_id="folder_lf1",
            files={Path(p).name: f"https://drive/{i}" for i, p in enumerate(shorts_paths)},
            completed=True,
        )

    def _save(s):
        save_snapshots.append(dict(s.upload_manifests or {}))

    with patch.object(phase_d_mod, "_upload_lf_shorts_to_drive", side_effect=fake_inner_upload):
        with pytest.raises(RuntimeError, match="drive failure on LF2"):
            _upload_all_lfs_shorts_to_drive(
                drive_plan=drive_plan,
                state=state,
                project_base="proj",
                output_base=output_base,
                uploader_factory=lambda: MagicMock(),
                skip_drive_upload=False,
                save_state=_save,
            )

    assert 1 in (state["upload_manifests"] or {}), (
        "LF1 manifest was discarded by the mid-loop raise — resume re-uploads all shorts"
    )
    assert save_snapshots and 1 in save_snapshots[-1], (
        "save_state was not called with LF1's manifest before the LF2 raise"
    )


def test_manifest_writethrough_without_save_callback(tmp_path):
    """Back-compat: called without ``save_state`` (existing callers), the loop
    still writes manifests through to ``state`` and does not crash."""
    state, drive_plan, output_base = _build_state_and_plan(tmp_path, num_lfs=2)

    def fake_inner_upload(*, uploader, lf, project_name, shorts_paths, output_base, use_dayfirst):
        return UploadManifest(
            project_name=project_name,
            folder_id=f"folder_lf{lf.index:02d}",
            files={
                Path(p).name: f"https://drive/{lf.index}/{i}" for i, p in enumerate(shorts_paths)
            },
            completed=True,
        )

    with patch.object(phase_d_mod, "_upload_lf_shorts_to_drive", side_effect=fake_inner_upload):
        manifests, _ = _upload_all_lfs_shorts_to_drive(
            drive_plan=drive_plan,
            state=state,
            project_base="proj",
            output_base=output_base,
            uploader_factory=lambda: MagicMock(),
            skip_drive_upload=False,
        )

    assert set(manifests.keys()) == {1, 2}
    assert state["upload_manifests"] is manifests
