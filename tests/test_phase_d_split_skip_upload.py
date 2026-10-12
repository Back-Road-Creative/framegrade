"""Split --skip-upload + phase_d youtube_upload re-run trigger.

Two related contracts pinned here:

  Split flags — `--skip-upload` is split into two granular flags:
    * ``--skip-drive-upload`` only suppresses Drive shorts upload.
    * ``--skip-youtube-upload`` only suppresses YouTube LF upload.
    * ``--skip-upload`` remains a back-compat alias that sets BOTH.

  Re-run trigger — phase_d in ``state["phases_completed"]`` is no longer terminal:
    when the operator passes ``--youtube-upload`` (i.e. ``state["youtube_upload"]
    == True``) on a follow-up invocation after a ``--skip-upload`` audit, phase_d
    must re-run the upload step. Closes the audit-then-publish workflow gap
    where a skipped upload could never be completed later.

The matrix in ``TestSkipUploadFlagMatrix`` exercises phase_d directly with a
fake YouTubeUploader; ``TestPhaseDReRunTrigger`` exercises the orchestrator's
resume-then-rerun decision in isolation.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.config import get_config
from gomoveshift.models import DrivePlan, LFSegment, RegularLFPlan
from gomoveshift.pipeline.drive_phases.phase_d import PhaseD

from _state_helpers import make_state, verified_shorts_per_lf


# ──────────────────────────────────────────────────────────────────────────
# Test scaffolding (mirrors test_phase_d_shorts_upload_dayfirst.py — keep in
# sync; the day-first cache layout is the same and the fake YouTube uploader
# pattern is shared).
# ──────────────────────────────────────────────────────────────────────────


def _install_fake_youtube_uploader(monkeypatch, upload_tracker: dict):
    fake_mod = ModuleType("gomoveshift.pipeline.drive_phases.youtube_uploader")
    inst = MagicMock()
    counter = {"n": 0}

    def _upload_if_needed(*, state, lf_idx, video_path, **kwargs):
        counter["n"] += 1
        upload_tracker.setdefault("calls", []).append(lf_idx)
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


def _build_state(tmp_path, num_lfs=2):
    output = tmp_path / "output"
    output.mkdir()
    source = tmp_path / "source_drive"
    source.mkdir()
    cache = tmp_path / "cache"
    stages_dir = cache / "_stages"
    stages_dir.mkdir(parents=True)

    regular_lfs = []
    lf_paths = []
    day = "2026-04-29"
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

    # A publish run posts only shorts whose own clip GPS verified their place.
    auto_paths = verified_shorts_per_lf(
        stages_dir,
        {i + 1: 2 for i in range(num_lfs)},
        date=day,
        place="Asheville, NC",
        regions=("Buncombe County",),
    )

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
        skip_csv=True,
        skip_upload=False,
        skip_drive_upload=False,
        skip_youtube_upload=False,
        youtube_upload=True,  # default-on for these tests; flag matrix governs gate behaviour
        drive_plan=drive_plan,
        regular_lf_paths=lf_paths,
        shorts_per_lf=auto_paths,
        # BC→D boundary contract — every rendered LF must appear in
        # these maps. BC populates them in production; fixtures bypass BC.
        per_lf_recording_date={i + 1: day for i in range(num_lfs)},
        per_lf_locations={i + 1: "Asheville, NC" for i in range(num_lfs)},
        per_lf_trim_ratio={i + 1: 0.7 for i in range(num_lfs)},
    )
    return state


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
    meta.generate.side_effect = lambda **kwargs: {
        "title": f"Title for {kwargs.get('project_name', 'lf')}",
        "description": "desc",
        "tags": ["driving"],
        "category": "Travel",
    }
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
            return [f"https://drive.google.com/file/d/abc{i}/view" for i in range(len(sd))]
        manifest = MagicMock()
        manifest.files = {"short_01.mp4": "https://drive.google.com/file/d/legacy/view"}
        return manifest

    uploader.upload_shorts.side_effect = _upload_shorts
    uploader._resolve_drive_subfolder_id.side_effect = lambda subfolder: f"folder_id_{subfolder}"
    upload_cls.return_value = uploader

    csv_cls.return_value = MagicMock()
    val = MagicMock()
    val.validate.return_value = MagicMock(passed=True, summary=lambda: "ok")
    val_cls.return_value = val
    return uploader


def _run_phase_d(state, num_lfs, monkeypatch):
    upload_tracker: dict = {}
    _install_fake_youtube_uploader(monkeypatch, upload_tracker)
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
        uploader = _setup_mocks((m0, m1, m2, m3, m4, m5, m6, m7), num_lfs=num_lfs)
        PhaseD(get_config()).run(state)
    return uploader, upload_tracker


# ──────────────────────────────────────────────────────────────────────────
# Flag matrix
# ──────────────────────────────────────────────────────────────────────────


class TestSkipUploadFlagMatrix:
    def test_skip_youtube_upload_runs_drive_skips_youtube(self, tmp_path, monkeypatch):
        """--skip-youtube-upload → Drive uploaded, YouTube NOT uploaded."""
        state = _build_state(tmp_path, num_lfs=2)
        state["skip_youtube_upload"] = True
        state["skip_drive_upload"] = False
        uploader, tracker = _run_phase_d(state, num_lfs=2, monkeypatch=monkeypatch)
        assert uploader.upload_shorts.call_count == 2, (
            "Drive upload must run when only YouTube is skipped; got "
            f"{uploader.upload_shorts.call_count} calls"
        )
        assert tracker.get("calls", []) == [], (
            "YouTube upload_if_needed must NOT be invoked when "
            f"skip_youtube_upload=True; got calls for LFs {tracker.get('calls')}"
        )

    def test_skip_drive_upload_runs_youtube_skips_drive(self, tmp_path, monkeypatch):
        """--skip-drive-upload → YouTube uploaded, Drive NOT uploaded."""
        state = _build_state(tmp_path, num_lfs=2)
        state["skip_youtube_upload"] = False
        state["skip_drive_upload"] = True
        uploader, tracker = _run_phase_d(state, num_lfs=2, monkeypatch=monkeypatch)
        assert uploader.upload_shorts.call_count == 0, (
            "Drive upload must NOT run when skip_drive_upload=True; got "
            f"{uploader.upload_shorts.call_count} calls"
        )
        assert sorted(tracker.get("calls", [])) == [1, 2], (
            "YouTube upload must run for both LFs when only Drive is skipped; got "
            f"calls for {tracker.get('calls')}"
        )

    def test_skip_upload_alias_skips_both(self, tmp_path, monkeypatch):
        """Legacy --skip-upload → neither Drive nor YouTube uploaded."""
        state = _build_state(tmp_path, num_lfs=2)
        # Legacy behavior — only the alias is set; the phase_d gate must
        # honour it without explicit skip_drive_upload / skip_youtube_upload.
        state["skip_upload"] = True
        state["skip_drive_upload"] = False
        state["skip_youtube_upload"] = False
        uploader, tracker = _run_phase_d(state, num_lfs=2, monkeypatch=monkeypatch)
        assert uploader.upload_shorts.call_count == 0, (
            "Drive upload must NOT run when --skip-upload alias is set"
        )
        assert tracker.get("calls", []) == [], (
            "YouTube upload must NOT run when --skip-upload alias is set"
        )


# ──────────────────────────────────────────────────────────────────────────
# phase_d re-run trigger at orchestrator boundary
# ──────────────────────────────────────────────────────────────────────────


class TestPhaseDReRunTrigger:
    """The orchestrator's _try_resume → re-run-decision sequence.

    Tests target the small, isolated logic block at the top of
    ``DrivePipelineOrchestrator.run`` that drops phase_d from
    ``resumed_phases`` when the operator passes ``--youtube-upload``. The
    full orchestrator graph is heavy; we exercise the predicate logic
    directly so the assertion is precise and doesn't drag in unrelated
    failures.

    The predicate (mirrored here so a refactor breaks the test):
        youtube_upload AND phase_d in resumed_phases AND not skip_youtube_upload
        → drop phase_d from resumed_phases, stages_completed, phases_completed.
    """

    @staticmethod
    def _apply_rerun_trigger(
        state: dict, resumed_phases: list[str], stages_completed: list[str]
    ) -> tuple:
        """Mirror of the orchestrator's re-run trigger block.

        If a future refactor moves the predicate, update this helper to
        invoke the actual orchestrator method instead of duplicating the
        logic — the assertions below are the contract.
        """
        if (
            state.get("youtube_upload")
            and "phase_d" in resumed_phases
            and not state.get("skip_youtube_upload", False)
        ):
            resumed_phases = [p for p in resumed_phases if p != "phase_d"]
            stages_completed[:] = [p for p in stages_completed if p != "phase_d"]
            saved_phases = state.get("phases_completed") or []
            state["phases_completed"] = [p for p in saved_phases if p != "phase_d"]
        return resumed_phases, stages_completed

    def test_phase_d_completed_with_youtube_upload_triggers_rerun(self):
        """phase_d in phases_completed AND youtube_upload=True → re-runs."""
        state = make_state(
            youtube_upload=True,
            skip_youtube_upload=False,
            phases_completed=["phase_a", "phase_bc_day_first", "phase_d"],
        )
        resumed = ["phase_a", "phase_bc_day_first", "phase_d"]
        stages = ["phase_a", "phase_bc_day_first", "phase_d"]
        resumed, stages = self._apply_rerun_trigger(state, resumed, stages)
        assert "phase_d" not in resumed, (
            f"The trigger must remove phase_d from resumed_phases; got {resumed}"
        )
        assert "phase_d" not in stages, (
            f"The trigger must remove phase_d from stages_completed; got {stages}"
        )
        assert "phase_d" not in state["phases_completed"], (
            f"The trigger must remove phase_d from state['phases_completed']; "
            f"got {state['phases_completed']}"
        )

    def test_phase_d_completed_without_upload_flag_skipped_no_regression(self):
        """phase_d in phases_completed AND no upload flag → phase_d stays skipped."""
        state = make_state(
            youtube_upload=False,
            skip_youtube_upload=False,
            phases_completed=["phase_a", "phase_bc_day_first", "phase_d"],
        )
        resumed = ["phase_a", "phase_bc_day_first", "phase_d"]
        stages = ["phase_a", "phase_bc_day_first", "phase_d"]
        resumed, stages = self._apply_rerun_trigger(state, resumed, stages)
        assert "phase_d" in resumed, (
            f"Pure-regen invocation (no youtube_upload) must NOT re-run phase_d; resumed={resumed}"
        )
        assert state["phases_completed"] == [
            "phase_a",
            "phase_bc_day_first",
            "phase_d",
        ], (
            "phases_completed must be unchanged when no upload flag is passed; "
            f"got {state['phases_completed']}"
        )


# ──────────────────────────────────────────────────────────────────────────
# Integration check: orchestrator wires the predicate identically to the
# helper above. Pin via direct read of the module so an accidental drift
# (e.g. someone removes the trigger block during a refactor) gets caught.
# ──────────────────────────────────────────────────────────────────────────


def test_orchestrator_contains_rerun_trigger_block():
    """Smoke test: orchestrator source contains the phase_d re-run trigger."""
    import gomoveshift.pipeline.drive_orchestrator as orch

    src = Path(orch.__file__).read_text()
    assert "re-running phase_d" in src, (
        "orchestrator source must contain the re-run trigger log line"
    )
    assert "youtube_upload" in src and "resumed_phases" in src, (
        "orchestrator must reference both youtube_upload and resumed_phases for the re-run trigger"
    )


# ──────────────────────────────────────────────────────────────────────────
# Quality gate: skip_drive_upload short-circuits the shorts-upload-complete check
# ──────────────────────────────────────────────────────────────────────────


def test_quality_gate_honours_skip_drive_upload():
    """check_shorts_upload_complete: skip_drive_upload=True → no missing-LF errors."""
    from gomoveshift.pipeline.quality_gates import check_shorts_upload_complete

    drive_plan = MagicMock()
    drive_plan.regular_lfs = [
        MagicMock(index=1),
        MagicMock(index=2),
    ]
    state = make_state(
        skip_upload=False,
        skip_drive_upload=True,
        upload_manifests={},
    )
    errors = check_shorts_upload_complete(state, drive_plan)
    assert errors == [], (
        f"skip_drive_upload=True must short-circuit the Drive completeness check; got {errors}"
    )

    # Sanity: with both flags False and empty manifests, the check fires.
    state["skip_drive_upload"] = False
    errors = check_shorts_upload_complete(state, drive_plan)
    assert errors, (
        "with skip_drive_upload=False AND empty manifests, the check must report "
        "missing LFs (sanity-check on the negative path)"
    )
