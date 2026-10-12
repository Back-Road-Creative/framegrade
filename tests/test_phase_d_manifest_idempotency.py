"""Phase D preserves ``upload_manifests`` across resume.

Earlier behaviour: ``phase_d.py`` initialised ``upload_manifests = {}``
at the top of the upload section. A regen run with prior Drive uploads
already in ``state["upload_manifests"]`` would blank the dict and
re-upload everything (or, with ``--skip-upload``, blank the dict and
emit a CSV with empty Drive-link columns).

Contract (with a "complete-predicate" correction):

  1. Initialise from ``state.get("upload_manifests", {}) or {}``. ``None``
     in state must not crash.
  2. Skip a re-upload for an LF ONLY IF the prior manifest is
     COMPLETE — ``len(manifest.files) == len(state["shorts_per_lf"][lf])``.
  3. Partial manifests trigger a fresh upload (do NOT silently keep the
     gap).
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

from gomoveshift.config import get_config
from gomoveshift.models import DrivePlan, LFSegment, RegularLFPlan
from gomoveshift.pipeline.drive_phases.phase_d import PhaseD
from gomoveshift.pipeline.uploader import UploadManifest

from _state_helpers import make_state


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


def _build_state(tmp_path, num_lfs=3):
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

    state = make_state(
        source_dir=source,
        output_base=output,
        stages_base=stages_dir,
        region="usa",
        publish_day="wed",
        location="Asheville, NC",
        skip_upload=False,
        skip_csv=True,
        youtube_upload=False,
        drive_plan=drive_plan,
        regular_lf_paths=lf_paths,
        shorts_per_lf=auto_paths,
        # BC→D boundary contract (state_invariants.assert_phase_d_per_lf_inputs_present)
        # requires every rendered LF to have an entry in these maps. BC populates
        # them in production; test fixtures bypass BC and must thread them.
        per_lf_recording_date={i + 1: day for i in range(num_lfs)},
        per_lf_locations={i + 1: "Asheville, NC" for i in range(num_lfs)},
        per_lf_trim_ratio={i + 1: 0.7 for i in range(num_lfs)},
    )
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
        sd = kwargs.get("shorts_dir", args[0] if args else None)
        if isinstance(sd, list):
            return [f"https://drive.google.com/file/d/new{i}/view" for i in range(len(sd))]
        manifest = MagicMock()
        manifest.files = {"short_01.mp4": "https://drive.google.com/file/d/legacy/view"}
        return manifest

    uploader.upload_shorts.side_effect = _upload_shorts
    uploader._resolve_drive_subfolder_id.side_effect = lambda subfolder: f"folder_id_{subfolder}"
    upload_cls.return_value = uploader

    csv = MagicMock()
    csv_cls.return_value = csv

    val = MagicMock()
    val.validate.return_value = MagicMock(passed=True, summary=lambda: "ok")
    val_cls.return_value = val

    return uploader


def _make_complete_manifest(lf_idx, expected_paths):
    """Build an UploadManifest with one entry per expected short path."""
    return UploadManifest(
        project_name=f"prior_lf{lf_idx:02d}",
        folder_id=f"folder_prior_{lf_idx}",
        files={
            Path(p).name: f"https://drive.google.com/file/d/prior_lf{lf_idx}_{i}/view"
            for i, p in enumerate(expected_paths)
        },
        completed=True,
    )


def _make_partial_manifest(lf_idx, expected_paths):
    """Build an UploadManifest that is short by one file (partial)."""
    return UploadManifest(
        project_name=f"prior_lf{lf_idx:02d}",
        folder_id=f"folder_prior_{lf_idx}",
        files={
            Path(expected_paths[0]).name: (
                f"https://drive.google.com/file/d/prior_lf{lf_idx}_partial/view"
            )
        },
        completed=False,
    )


class TestPhaseDManifestIdempotency:
    def test_complete_prior_manifest_skips_reupload(self, tmp_path, monkeypatch):
        """LF1 has complete manifest → no re-upload; LF2 partial → re-upload."""
        _install_fake_youtube_uploader(monkeypatch)
        state, auto_paths = _build_state(tmp_path, num_lfs=3)

        prior = {
            1: _make_complete_manifest(1, auto_paths[1]),  # complete → skip
            2: _make_partial_manifest(2, auto_paths[2]),  # partial → re-upload
            # LF 3 absent → re-upload
        }
        state["upload_manifests"] = prior

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

        # Only LF2 + LF3 re-upload → 2 calls.
        assert uploader.upload_shorts.call_count == 2, (
            f"expected 2 re-uploads (LF2 partial + LF3 missing), got "
            f"{uploader.upload_shorts.call_count}"
        )

        # LF1 manifest preserved with the prior Drive links.
        manifest_lf1 = state["upload_manifests"][1]
        for link in manifest_lf1.files.values():
            assert "prior_lf1" in link, (
                f"LF1 manifest was overwritten — expected prior links, got {link!r}"
            )

        # LF2 manifest replaced (partial → re-upload produced fresh links).
        manifest_lf2 = state["upload_manifests"][2]
        assert len(manifest_lf2.files) == 2
        for link in manifest_lf2.files.values():
            assert "/new" in link, f"LF2 should have fresh links from re-upload, got {link!r}"

        # LF3 has fresh manifest.
        assert 3 in state["upload_manifests"]
        assert len(state["upload_manifests"][3].files) == 2

    def test_no_prior_manifest_uploads_all(self, tmp_path, monkeypatch):
        """state has no upload_manifests → upload every LF (current behavior)."""
        _install_fake_youtube_uploader(monkeypatch)
        state, _ = _build_state(tmp_path, num_lfs=3)
        assert "upload_manifests" not in state

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

        assert uploader.upload_shorts.call_count == 3
        assert set(state["upload_manifests"].keys()) == {1, 2, 3}

    def test_upload_manifests_none_in_state_does_not_crash(self, tmp_path, monkeypatch):
        """state['upload_manifests'] = None → coerce to {}, upload all LFs."""
        _install_fake_youtube_uploader(monkeypatch)
        state, _ = _build_state(tmp_path, num_lfs=2)
        state["upload_manifests"] = None  # explicit None, not just missing

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
            PhaseD(get_config()).run(state)

        # No crash; both LFs uploaded.
        assert uploader.upload_shorts.call_count == 2
        assert set(state["upload_manifests"].keys()) == {1, 2}
