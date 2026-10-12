"""Per-LF recording_date is sourced from the day-batch.

Bug shape: most LFs of a multi-day drive had a wrong
"📹 Recorded: <date>" line in their YouTube descriptions — every LF
showed the first day's date even though the actual recording days
spanned several days.

Root cause: ``phase_bc_day_first._partition_lf_plans_by_date`` remaps
each segment's ``source_file_index`` from GLOBAL to DAY-LOCAL before
``phase_d`` runs, but ``phase_d`` then does
``episode_dates_list[int(first_idx)]`` which is keyed by GLOBAL index.
Day-local index 0 always pointed at the first global episode's date.

Fix shape: ``phase_bc_day_first`` writes ``state["per_lf_recording_date"]``
keyed by ``lf.index`` (which is NOT remapped). Phase D reads that map
instead of indexing into ``episode_dates``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.config import get_config
from gomoveshift.models import DrivePlan, LFSegment, RegularLFPlan
from gomoveshift.pipeline.drive_phases.phase_d import PhaseD


# ── Helpers (lifted/trimmed from test_phase_d_lf_outputs.py) ─────────────


def _install_fake_youtube_uploader(monkeypatch):
    fake_mod = ModuleType("gomoveshift.pipeline.drive_phases.youtube_uploader")
    uploader_instance = MagicMock()
    uploader_instance.upload_video.return_value = "vid_xx"
    uploader_instance.set_thumbnail.return_value = True

    def _upload_if_needed(*, state, lf_idx, video_path, **kwargs):
        uploads = state.setdefault("youtube_uploads", {})
        record = {
            "video_id": f"vid_{lf_idx:02d}",
            "url": f"https://youtu.be/vid_{lf_idx:02d}",
            "publish_at": kwargs.get("publish_at"),
            "title": kwargs.get("title", ""),
            "uploaded_at": "2026-01-01T00:00:00+00:00",
        }
        uploads[lf_idx] = record
        return record

    uploader_instance.upload_if_needed.side_effect = _upload_if_needed

    class YouTubeUploader:
        def __new__(cls, *a, **kw):
            return uploader_instance

    fake_mod.YouTubeUploader = YouTubeUploader
    fake_mod._instance = uploader_instance
    monkeypatch.setitem(sys.modules, "gomoveshift.pipeline.drive_phases.youtube_uploader", fake_mod)
    return uploader_instance


def _build_state(tmp_path, num_lfs, *, per_lf_recording_date=None, episode_dates=None):
    output = tmp_path / "output"
    output.mkdir()
    source = tmp_path / "source_drive"
    source.mkdir()

    regular_lfs = []
    lf_paths = []
    shorts_per_lf: dict[int, list] = {}
    for i in range(num_lfs):
        # NOTE: source_file_index here is DAY-LOCAL (always 0 for the
        # first segment of each LF after _partition_lf_plans_by_date),
        # which is exactly the resume-state shape phase_d sees in prod.
        seg = LFSegment(
            source_file_index=0,
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
                region="ireland",
                total_duration=600.0,
            )
        )
        lf_path = output / f"lf_{i + 1:02d}.mp4"
        lf_path.write_bytes(b"x" * (2 * 1024 * 1024))
        lf_paths.append(lf_path)
        shorts_per_lf[i + 1] = [MagicMock() for _ in range(2)]

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
        region="ireland",
        publish_day="wed",
        location="Cork, Ireland",
        skip_upload=True,
        skip_csv=True,
        youtube_upload=False,
        drive_plan=drive_plan,
        regular_lf_paths=lf_paths,
        shorts_per_lf=shorts_per_lf,
        # BC→D boundary contract — every rendered LF must appear in
        # these maps. BC populates them in production; fixtures bypass BC.
        per_lf_locations={i + 1: "Cork, Ireland" for i in range(num_lfs)},
        per_lf_trim_ratio={i + 1: 0.7 for i in range(num_lfs)},
    )
    if per_lf_recording_date is not None:
        state.per_lf_recording_date = per_lf_recording_date
    if episode_dates is not None:
        state.episode_dates = episode_dates
    return state


def _phase_d_patches():
    return [
        patch("gomoveshift.pipeline.drive_phases.phase_d.LongFormMetadataGenerator"),
        patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineScheduler"),
        patch("gomoveshift.pipeline.drive_phases.phase_d.ReleaseSchedule"),
        patch("gomoveshift.pipeline.drive_phases.phase_d.SocialGenerator"),
        patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineUploader"),
        patch("gomoveshift.pipeline.drive_phases.phase_d.UploadManifest"),
        patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineValidator"),
    ]


def _setup_mocks(mocks, num_lfs):
    (meta_cls, sched_cls, _rs_cls, social_cls, upload_cls, _manifest_cls, val_cls) = mocks

    meta = MagicMock()
    # Capture every kwargs dict passed to .generate()
    captured_calls: list[dict] = []

    def _meta_side_effect(**kwargs):
        captured_calls.append(dict(kwargs))
        project_name = kwargs.get("project_name", "lf")
        return {
            "title": f"LF Title {project_name}",
            "description": "LF description",
            "tags": ["driving", "ireland"],
            "category": "Travel",
        }

    meta.generate.side_effect = _meta_side_effect
    meta.captured_calls = captured_calls
    meta_cls.return_value = meta

    from datetime import datetime as _dt
    from datetime import timedelta as _td

    wed_dates = [(_dt(2026, 4, 1) + _td(days=7 * i)).strftime("%Y-%m-%d") for i in range(num_lfs)]
    lf_slots = [
        MagicMock(
            date=wed_dates[i],
            time="11:00:00",
            datetime_est=f"{wed_dates[i]}T11:00:00-04:00",
            day_of_week="Wednesday",
        )
        for i in range(num_lfs)
    ]
    shorts_slots = []
    for i in range(num_lfs):
        for j in range(2):
            shorts_slots.append(
                MagicMock(
                    short_index=i * 2 + j + 1,
                    date=wed_dates[i],
                    time=f"16:0{j}:00",
                    datetime_est=f"{wed_dates[i]}T16:0{j}:00-04:00",
                    day_of_week="Wednesday",
                )
            )
    drive_schedule = MagicMock()
    drive_schedule.lf_slots = lf_slots
    drive_schedule.shorts_slots = shorts_slots
    drive_schedule.to_dict.return_value = {"region": "ireland"}
    sched = MagicMock()
    sched.generate_drive_schedule.return_value = drive_schedule
    sched_cls.return_value = sched
    sched_cls.from_config.return_value = sched

    social = MagicMock()
    posts = []
    for idx in range(1, num_lfs * 2 + 1):
        p = MagicMock(short_index=idx, caption=f"caption_{idx}")
        posts.append(p)
    social.generate_all.side_effect = [posts[i * 2 : (i + 1) * 2] for i in range(num_lfs)]
    social_cls.return_value = social

    uploader = MagicMock()
    shorts_manifest = MagicMock()
    shorts_manifest.files = {
        f"short_{i:02d}.mp4": f"https://drive.google.com/s{i}" for i in range(1, num_lfs * 2 + 1)
    }
    uploader.upload_shorts.return_value = shorts_manifest
    upload_cls.return_value = uploader

    val = MagicMock()
    val.validate.return_value = MagicMock(passed=True, summary=lambda: "ok")
    val_cls.return_value = val

    return meta


def _run_phase_d(state, num_lfs):
    patches = _phase_d_patches()
    with (
        patches[0] as m0,
        patches[1] as m1,
        patches[2] as m2,
        patches[3] as m3,
        patches[4] as m4,
        patches[5] as m5,
        patches[6] as m6,
    ):
        meta = _setup_mocks((m0, m1, m2, m3, m4, m5, m6), num_lfs)
        PhaseD(get_config()).run(state)
        return meta.captured_calls


# ─────────────────────────────────────────────────────────────────────────
# Phase D side
# ─────────────────────────────────────────────────────────────────────────


class TestPhaseDRecordingDateFromState:
    def test_per_lf_recording_date_state_field_drives_metadata(self, tmp_path, monkeypatch):
        """Multi-day shape: 7 LFs across 5 days, day-local indices,
        each LF must receive ITS OWN recording_date — not the first
        global episode's date."""
        _install_fake_youtube_uploader(monkeypatch)
        per_lf_recording_date = {
            1: "2026-02-20",
            2: "2026-02-21",
            3: "2026-02-21",
            4: "2026-02-22",
            5: "2026-02-24",
            6: "2026-02-24",
            7: "2026-02-24",
        }
        # episode_dates ALSO present, with the wrong-default 2026-02-20
        # at index 0 — proves the new state field wins over the old
        # index-into-episode_dates lookup that the bug was hitting.
        state = _build_state(
            tmp_path,
            num_lfs=7,
            per_lf_recording_date=per_lf_recording_date,
            episode_dates=["2026-02-20"] * 7,
        )
        calls = _run_phase_d(state, num_lfs=7)

        # One generate() call per LF, in LF-index order.
        by_lf = {}
        for c in calls:
            pn = c.get("project_name", "")
            # project_name = "<base>_lfNN"
            lf_num = int(pn.rsplit("_lf", 1)[-1])
            by_lf[lf_num] = c

        for lf_idx, expected_date in per_lf_recording_date.items():
            assert lf_idx in by_lf, f"missing generate() call for LF {lf_idx}"
            assert by_lf[lf_idx]["recording_date"] == expected_date, (
                f"LF {lf_idx} got recording_date="
                f"{by_lf[lf_idx]['recording_date']!r}, expected {expected_date!r}"
            )

    def test_missing_state_field_raises_d_entry_contract(self, tmp_path, monkeypatch):
        """per_lf_recording_date is a required Phase D entry input. Legacy
        state files that lack the field used to fall back to
        ``recording_date=None``; under the BC→D boundary contract, Phase D
        now raises PhaseDEntryContractError at entry instead.
        """
        from gomoveshift.pipeline.exceptions import PhaseDEntryContractError

        _install_fake_youtube_uploader(monkeypatch)
        state = _build_state(tmp_path, num_lfs=2)
        # Deliberately no per_lf_recording_date and no episode_dates.
        with pytest.raises(PhaseDEntryContractError, match="per_lf_recording_date"):
            _run_phase_d(state, num_lfs=2)


# ─────────────────────────────────────────────────────────────────────────
# Phase BC side — state field population
# ─────────────────────────────────────────────────────────────────────────


class TestPhaseBCPopulatesRecordingDateMap:
    def test_partition_result_translated_to_state_map(self):
        """Given an ``lf_plans_by_date`` dict (the output of
        ``_partition_lf_plans_by_date``), the helper that derives the
        state field must produce ``{lf.index: date}`` for every LF."""
        # Import the helper we'll add to phase_bc_day_first.
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            _build_per_lf_recording_date_map,
        )

        class _LF:
            def __init__(self, index):
                self.index = index

        lf_plans_by_date = {
            "2026-02-20": [_LF(1)],
            "2026-02-21": [_LF(2), _LF(3)],
            "2026-02-22": [_LF(4)],
            "2026-02-24": [_LF(5), _LF(6), _LF(7)],
        }
        result = _build_per_lf_recording_date_map(lf_plans_by_date)
        assert result == {
            1: "2026-02-20",
            2: "2026-02-21",
            3: "2026-02-21",
            4: "2026-02-22",
            5: "2026-02-24",
            6: "2026-02-24",
            7: "2026-02-24",
        }

    def test_partition_result_with_date_objects_normalized(self):
        """Dates may arrive as ``datetime.date`` (from episode_dates).
        The helper must coerce them to YYYY-MM-DD strings so phase_d
        can drop the value straight into the description line."""
        import datetime as _dt

        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            _build_per_lf_recording_date_map,
        )

        class _LF:
            def __init__(self, index):
                self.index = index

        lf_plans_by_date = {
            _dt.date(2026, 2, 21): [_LF(2), _LF(3)],
        }
        result = _build_per_lf_recording_date_map(lf_plans_by_date)
        assert result == {2: "2026-02-21", 3: "2026-02-21"}

    def test_empty_partition_yields_empty_map(self):
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            _build_per_lf_recording_date_map,
        )

        assert _build_per_lf_recording_date_map({}) == {}
        assert _build_per_lf_recording_date_map(None) == {}
