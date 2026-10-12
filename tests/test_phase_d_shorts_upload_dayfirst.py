"""Phase D consumes day-first shorts paths for Drive upload.

Regression: ``phase_d.py`` previously looked at
``<output_base>/shorts/lf_<NN>/`` for shorts to upload to Drive, but the
day-first architecture writes shorts to
``<cache>/_stages/day_<YYYY-MM-DD>/shorts/short_<NN>.mp4``. The legacy
directory was always empty, so every Drive upload silently failed with
``FileNotFoundError: No .mp4 files found in …/shorts/lf_NN`` — leaving
the consolidated Metricool CSV with empty ``Picture Url 1`` for every
short.

These tests pin the new contract:

  1. When ``state["shorts_per_lf"]`` is populated with day-first paths,
     ``PipelineUploader.upload_shorts`` is invoked with those paths
     (NOT with ``output_base/shorts/lf_NN``).
  2. When the field is missing, phase_d falls back to the legacy
     directory lookup with a WARN log (forward-compat for older state
     files / pipelines that pre-date day-first paths).
  3. ``state["upload_manifests"]`` is populated with one entry per LF.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.config import get_config
from gomoveshift.models import DrivePlan, LFSegment, RegularLFPlan
from gomoveshift.pipeline.drive_phases.phase_d import PhaseD
from gomoveshift.pipeline.uploader import UploadManifest


def _install_fake_youtube_uploader(monkeypatch):
    fake_mod = ModuleType("gomoveshift.pipeline.drive_phases.youtube_uploader")
    inst = MagicMock()
    counter = {"n": 0}

    def _upload_if_needed(*, state, lf_idx, video_path, **kwargs):
        counter["n"] += 1
        rec = {
            "video_id": f"vid_{counter['n']:02d}",
            "url": f"https://youtu.be/vid_{counter['n']:02d}",
            "publish_at": kwargs.get("publish_at"),
            "title": kwargs.get("title", ""),
            "uploaded_at": "2026-01-01T00:00:00+00:00",
        }
        state.setdefault("youtube_uploads", {})[lf_idx] = rec
        return rec

    inst.upload_if_needed.side_effect = _upload_if_needed
    inst.set_thumbnail.return_value = True

    class YouTubeUploader:
        def __new__(cls, *a, **kw):
            return inst

    fake_mod.YouTubeUploader = YouTubeUploader
    monkeypatch.setitem(sys.modules, "gomoveshift.pipeline.drive_phases.youtube_uploader", fake_mod)
    return inst


def _build_state(tmp_path, num_lfs=3, shorts_per_lf_paths=None, populate_state_field=True):
    """Build a Phase-D state dict with N LFs.

    If ``populate_state_field`` is True, ``state["shorts_per_lf"]`` is set
    to ``shorts_per_lf_paths`` (or auto-generated day-first cache paths).
    Otherwise the field is omitted (legacy / forward-compat path).
    """
    output = tmp_path / "output"
    output.mkdir()
    source = tmp_path / "source_drive"
    source.mkdir()
    cache = tmp_path / "cache"
    stages_dir = cache / "_stages"
    stages_dir.mkdir(parents=True)

    regular_lfs = []
    lf_paths = []
    auto_paths: dict[int, list[Path]] = {}
    day = "2026-04-29"
    day_shorts_dir = stages_dir / f"day_{day}" / "shorts"
    day_shorts_dir.mkdir(parents=True)

    short_counter = 0
    for i in range(num_lfs):
        seg = LFSegment(
            source_file_index=i,
            source_path=tmp_path / f"ep_{i:03d}_FINAL.mp4",
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
        regular_lfs.append(
            RegularLFPlan(
                index=i + 1,
                segments=[seg],
                montage_peaks=[seg],
                shorts_target=2,
                region="usa",
                total_duration=600.0,
            )
        )
        lf_path = output / f"lf_{i + 1:02d}.mp4"
        lf_path.write_bytes(b"x" * (2 * 1024 * 1024))
        lf_paths.append(lf_path)

        # Day-first cache paths — two shorts per LF, all sharing one day dir.
        lf_short_paths = []
        for _j in range(2):
            short_counter += 1
            sp = day_shorts_dir / f"short_{short_counter:02d}.mp4"
            sp.write_bytes(b"y" * 1024)
            lf_short_paths.append(sp)
        auto_paths[i + 1] = lf_short_paths

    drive_plan = DrivePlan(
        source_files=[],
        processed_files=[],
        regular_lfs=regular_lfs,
        score_threshold=0.5,
        total_source_duration=600.0 * num_lfs,
        total_highlight_duration=600.0 * num_lfs,
    )

    from _state_helpers import make_state

    state = make_state(
        source_dir=source,
        output_base=output,
        stages_base=stages_dir,
        region="usa",
        publish_day="wed",
        location="Asheville, NC",
        skip_upload=False,
        skip_csv=True,  # CSV gen is out of scope for these tests
        youtube_upload=False,  # YouTube gate is out of scope
        drive_plan=drive_plan,
        regular_lf_paths=lf_paths,
        # BC→D boundary contract — every rendered LF must appear in
        # these maps. BC populates them in production; fixtures bypass BC.
        per_lf_recording_date={i + 1: day for i in range(num_lfs)},
        per_lf_locations={i + 1: "Asheville, NC" for i in range(num_lfs)},
        per_lf_trim_ratio={i + 1: 0.7 for i in range(num_lfs)},
    )
    if populate_state_field:
        state.shorts_per_lf = shorts_per_lf_paths or auto_paths
    return state, auto_paths


def _phase_d_patches():
    return [
        patch("gomoveshift.pipeline.drive_phases.phase_d.LongFormMetadataGenerator"),
        patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineScheduler"),
        patch("gomoveshift.pipeline.drive_phases.phase_d.ReleaseSchedule"),
        patch("gomoveshift.pipeline.drive_phases.phase_d.SocialGenerator"),
        patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineUploader"),
        patch("gomoveshift.pipeline.drive_phases.phase_d.UploadManifest"),
        patch("gomoveshift.pipeline.drive_phases.phase_d.CSVGenerator"),
        patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineValidator"),
    ]


def _setup_mocks(mocks, num_lfs):
    (meta_cls, sched_cls, _rs_cls, social_cls, upload_cls, _manifest_cls, csv_cls, val_cls) = mocks

    meta = MagicMock()

    def _meta_side_effect(**kwargs):
        project_name = kwargs.get("project_name", "lf")
        return {
            "title": f"Title for {project_name}",
            "description": "desc",
            "tags": ["driving"],
            "category": "Travel",
        }

    meta.generate.side_effect = _meta_side_effect
    meta_cls.return_value = meta

    wed_dates = [
        (datetime(2026, 4, 29) + timedelta(days=7 * i)).strftime("%Y-%m-%d") for i in range(num_lfs)
    ]
    lf_slots = [
        MagicMock(
            date=wed_dates[i],
            time="11:00:00",
            datetime_est=f"{wed_dates[i]}T11:00:00-04:00",
            day_of_week="Wednesday",
        )
        for i in range(num_lfs)
    ]
    drive_schedule = MagicMock()
    drive_schedule.lf_slots = lf_slots
    drive_schedule.shorts_slots = []
    drive_schedule.to_dict.return_value = {"region": "usa"}
    sched = MagicMock()
    sched.generate_drive_schedule.return_value = drive_schedule
    sched_cls.return_value = sched
    sched_cls.from_config.return_value = sched

    social = MagicMock()
    social.generate_all.side_effect = [
        [MagicMock(short_index=i * 2 + j + 1, caption=f"c_{i}_{j}") for j in range(2)]
        for i in range(num_lfs)
    ]
    social_cls.return_value = social

    uploader = MagicMock()

    def _upload_shorts(*args, **kwargs):
        # Mirror the real list-form return: List[str] of Drive links.
        sd = kwargs.get("shorts_dir", args[0] if args else None)
        if isinstance(sd, list):
            return [f"https://drive.google.com/file/d/abc{i}/view" for i in range(len(sd))]
        # Legacy directory form: return a manifest-like mock.
        manifest = MagicMock()
        manifest.files = {"short_01.mp4": "https://drive.google.com/file/d/legacy/view"}
        return manifest

    uploader.upload_shorts.side_effect = _upload_shorts
    # Surface the helper used by phase_d to resolve a per-LF folder ID.
    uploader._resolve_drive_subfolder_id.side_effect = lambda subfolder: f"folder_id_{subfolder}"
    upload_cls.return_value = uploader

    csv = MagicMock()
    csv_cls.return_value = csv

    val = MagicMock()
    val.validate.return_value = MagicMock(passed=True, summary=lambda: "ok")
    val_cls.return_value = val

    return uploader


class TestPhaseDDayFirstUpload:
    def test_uses_day_first_paths_when_state_field_populated(self, tmp_path, monkeypatch):
        """state['shorts_per_lf'] populated → uploader called with day-first paths."""
        _install_fake_youtube_uploader(monkeypatch)
        state, auto_paths = _build_state(tmp_path, num_lfs=3)

        with (
            _phase_d_patches()[0] as m0,
            _phase_d_patches()[1] as m1,
            _phase_d_patches()[2] as m2,
            _phase_d_patches()[3] as m3,
            _phase_d_patches()[4] as m4,
            _phase_d_patches()[5] as m5,
            _phase_d_patches()[6] as m6,
            _phase_d_patches()[7] as m7,
        ):
            uploader = _setup_mocks((m0, m1, m2, m3, m4, m5, m6, m7), num_lfs=3)
            PhaseD(get_config()).run(state)

        # Three LFs → three upload_shorts calls.
        assert uploader.upload_shorts.call_count == 3, (
            f"expected 3 upload calls, got {uploader.upload_shorts.call_count}"
        )

        # Each call must pass a LIST of day-first paths (NOT a directory string
        # under output_base/shorts/lf_NN).
        legacy_dir_substring = str(state["output_base"] / "shorts")
        for idx, call in enumerate(uploader.upload_shorts.call_args_list, start=1):
            kw = call.kwargs
            shorts_dir_arg = kw.get("shorts_dir")
            assert isinstance(shorts_dir_arg, list), (
                f"LF {idx}: shorts_dir must be a list of paths, got "
                f"{type(shorts_dir_arg).__name__}: {shorts_dir_arg!r}"
            )
            # All paths point inside the day-first cache, not the legacy dir.
            for p in shorts_dir_arg:
                p_str = str(p)
                assert "_stages/day_" in p_str, f"LF {idx}: path {p_str!r} not in day-first cache"
                assert legacy_dir_substring not in p_str, (
                    f"LF {idx}: path {p_str!r} still references legacy "
                    f"output_base/shorts/lf_NN layout"
                )
            # folder_id required for list-form upload.
            assert kw.get("folder_id"), (
                f"LF {idx}: list-form upload_shorts requires folder_id, got {kw}"
            )

        # Manifest is populated for every LF.
        assert set(state["upload_manifests"].keys()) == {1, 2, 3}
        for lf_idx in (1, 2, 3):
            manifest = state["upload_manifests"][lf_idx]
            # Manifest is an UploadManifest with the LF's shorts.
            assert hasattr(manifest, "files"), (
                f"LF {lf_idx}: manifest missing .files, got {manifest!r}"
            )
            assert len(manifest.files) == 2, (
                f"LF {lf_idx}: expected 2 files in manifest, got {manifest.files}"
            )
            for link in manifest.files.values():
                assert link.startswith("https://drive.google.com/"), (
                    f"LF {lf_idx}: bad Drive link {link!r}"
                )

    def test_absent_state_field_hard_fails_not_glob_ship(self, tmp_path, monkeypatch):
        """An absent/empty state['shorts_per_lf'] while regular LFs exist is
        a HARD FAIL — the legacy glob-ship of output_base/shorts/lf_NN/ (ungated
        shorts, zero LUFS/geometry/OST gating) is killed. The run must raise and
        upload_shorts must NEVER be called."""
        _install_fake_youtube_uploader(monkeypatch)
        state, _ = _build_state(tmp_path, num_lfs=2, populate_state_field=False)
        # Materialise legacy dirs with stale shorts — exactly what the killed
        # fallback would have glob-shipped ungated.
        for i in (1, 2):
            d = state["output_base"] / "shorts" / f"lf_{i:02d}"
            d.mkdir(parents=True)
            (d / "short_01.mp4").write_bytes(b"x" * 512)

        # Ensure shorts_per_lf is genuinely absent (not just empty).
        assert "shorts_per_lf" not in state

        with (
            _phase_d_patches()[0] as m0,
            _phase_d_patches()[1] as m1,
            _phase_d_patches()[2] as m2,
            _phase_d_patches()[3] as m3,
            _phase_d_patches()[4] as m4,
            _phase_d_patches()[5] as m5,
            _phase_d_patches()[6] as m6,
            _phase_d_patches()[7] as m7,
        ):
            uploader = _setup_mocks((m0, m1, m2, m3, m4, m5, m6, m7), num_lfs=2)
            with pytest.raises(RuntimeError, match="shorts_per_lf"):
                PhaseD(get_config()).run(state)

        # No ungated glob-ship: upload_shorts must not have been called at all.
        uploader.upload_shorts.assert_not_called()

    def test_all_lfs_have_paths_populates_manifests(self, tmp_path, monkeypatch):
        """7 LFs → upload_manifests has all 7 entries."""
        _install_fake_youtube_uploader(monkeypatch)
        state, _ = _build_state(tmp_path, num_lfs=7)

        with (
            _phase_d_patches()[0] as m0,
            _phase_d_patches()[1] as m1,
            _phase_d_patches()[2] as m2,
            _phase_d_patches()[3] as m3,
            _phase_d_patches()[4] as m4,
            _phase_d_patches()[5] as m5,
            _phase_d_patches()[6] as m6,
            _phase_d_patches()[7] as m7,
        ):
            uploader = _setup_mocks((m0, m1, m2, m3, m4, m5, m6, m7), num_lfs=7)
            PhaseD(get_config()).run(state)

        assert uploader.upload_shorts.call_count == 7
        assert set(state["upload_manifests"].keys()) == {1, 2, 3, 4, 5, 6, 7}
        for lf_idx in range(1, 8):
            m = state["upload_manifests"][lf_idx]
            assert isinstance(m, UploadManifest) or hasattr(m, "files")
            assert len(m.files) == 2
