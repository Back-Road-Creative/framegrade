"""Phase D consolidation tests.

Verifies:
  - Super LF step removed (no super LF state writes, no super LF CSV).
  - LFs uploaded via YouTubeUploader (late-imported, mocked).
  - LFs NOT uploaded to Drive via upload_single_file.
  - One consolidated shorts CSV is written, with correct row count.
  - First Comment Text maps to parent LF's YouTube URL.
  - set_thumbnail called once per LF.
"""

from __future__ import annotations

import inspect
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.config import get_config
from gomoveshift.models import DrivePlan, RegularLFPlan
from gomoveshift.pipeline.drive_phases.phase_d import PhaseD


def _install_fake_youtube_uploader(monkeypatch, video_id_prefix="vid"):
    """Install a fake `gomoveshift.pipeline.drive_phases.youtube_uploader` module.

    The real module talks to the YouTube API. For isolation we inject a stub so
    Phase D's late-import resolves to it, and expose a MagicMock-based
    `YouTubeUploader` class.
    """
    fake_mod = ModuleType("gomoveshift.pipeline.drive_phases.youtube_uploader")

    uploader_instance = MagicMock()
    call_counter = {"n": 0}

    def _upload_video(video_path, **kwargs):  # noqa: ANN001
        call_counter["n"] += 1
        return f"{video_id_prefix}_{call_counter['n']:02d}"

    uploader_instance.upload_video.side_effect = _upload_video
    uploader_instance.set_thumbnail.return_value = True

    def _upload_if_needed(*, state, lf_idx, video_path, **kwargs):  # noqa: ANN001
        """Delegate to upload_video and build a record, mirroring the real method."""
        uploads = state.setdefault("youtube_uploads", {})
        existing = uploads.get(lf_idx)
        if existing and existing.get("video_id"):
            return existing
        video_id = _upload_video(video_path, **kwargs)
        record = {
            "video_id": video_id,
            "url": f"https://youtu.be/{video_id}",
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


@pytest.fixture
def phase_d_state(tmp_path):
    """Minimal state dict mirroring Phase C's output."""
    output = tmp_path / "output"
    output.mkdir()
    source = tmp_path / "source_drive"
    source.mkdir()

    from _state_helpers import make_state, verified_shorts_per_lf
    from gomoveshift.models import LFSegment  # local import to avoid test load order issues

    regular_lfs = []
    lf_paths = []
    for i in range(2):
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
                shorts_target=3,
                region="ireland",
                total_duration=600.0,
            )
        )
        lf_path = output / f"lf_{i + 1:02d}.mp4"
        lf_path.parent.mkdir(parents=True, exist_ok=True)
        lf_path.write_bytes(b"x" * (2 * 1024 * 1024))  # >1 MiB so validator path triggers
        lf_paths.append(lf_path)

    # A publish run posts only shorts whose own clip GPS verified their place.
    shorts_per_lf = verified_shorts_per_lf(output / "_stages", {1: 3, 2: 3})

    drive_plan = DrivePlan(
        source_files=[],
        processed_files=[],
        regular_lfs=regular_lfs,
        score_threshold=0.5,
        total_source_duration=1200.0,
        total_highlight_duration=1200.0,
    )

    return make_state(
        source_dir=source,
        output_base=output,
        region="ireland",
        publish_day="wed",
        location="Cork, Ireland",
        skip_upload=False,
        skip_csv=False,
        youtube_upload=True,
        drive_plan=drive_plan,
        regular_lf_paths=lf_paths,
        shorts_per_lf=shorts_per_lf,
        # Phase BC invariant ``_assert_per_lf_recording_date_complete``
        # populates this map in production. Test fixtures bypass Phase BC,
        # so we thread the data the contract assertion in Phase D's
        # ``_build_recording_dates_by_short_index`` expects.
        per_lf_recording_date={1: "2025-09-14", 2: "2025-09-14"},
        # Phase D's entry contract requires every rendered LF to have an
        # entry in per_lf_locations and per_lf_trim_ratio. BC populates
        # these in production; test fixtures bypass BC and must thread them.
        per_lf_locations={1: "Cork, Ireland", 2: "Cork, Ireland"},
        per_lf_trim_ratio={1: 0.7, 2: 0.7},
    )


def _phase_d_patches():
    """Context-manager list of orchestrator-level patches required by Phase D.

    Tests pass as nested `with` or via ExitStack.
    """
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


def _setup_mocks(mocks, num_lfs=2):
    (meta_cls, sched_cls, rs_cls, social_cls, upload_cls, manifest_cls, csv_cls, val_cls) = mocks

    meta = MagicMock()

    # Titles must be distinct across all LFs. Use
    # side_effect so each call returns a unique title keyed by
    # project_name.
    def _meta_side_effect(**kwargs):
        project_name = kwargs.get("project_name", "lf")
        return {
            "title": f"Title for {project_name}",
            "description": "desc",
            "tags": ["driving", "ireland"],
            "category": "Travel",
        }

    meta.generate.side_effect = _meta_side_effect
    meta_cls.return_value = meta

    # lf_slot dates must fall on days in LONG_FORM_DAYS
    # (Wed=weekday 2). 2026-04-01 is Wed; every 7 days is next Wed.
    wed_dates = [
        (datetime(2026, 4, 1) + timedelta(days=7 * i)).strftime("%Y-%m-%d") for i in range(num_lfs)
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
    shorts_slots = []
    for i in range(num_lfs):
        for j in range(3):
            shorts_slots.append(
                MagicMock(
                    short_index=i * 3 + j + 1,
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
    for idx in range(1, num_lfs * 3 + 1):
        p = MagicMock(short_index=idx, caption=f"caption_{idx}")
        posts.append(p)
    # Social generator returns 3 posts per LF call
    social.generate_all.side_effect = [posts[i * 3 : (i + 1) * 3] for i in range(num_lfs)]
    social_cls.return_value = social

    uploader = MagicMock()

    def _fake_upload_shorts(*_args, shorts_dir=None, **_kwargs):
        # Day-first contract: ``upload_shorts`` returns a LIST of Drive links
        # (one per short path); ``_upload_lf_shorts_to_drive`` zips paths→links
        # into the per-LF ``UploadManifest.files``. A bare MagicMock return
        # iterated empty → every ``Picture Url 1`` blank (the media-less-CSV
        # bug the ``METRICOOL-CSV-NO-MEDIA`` gate now rejects).
        n = len(shorts_dir) if isinstance(shorts_dir, (list, tuple)) else 2
        return [f"https://drive.google.com/s{i}" for i in range(n)]

    uploader.upload_shorts.side_effect = _fake_upload_shorts
    upload_cls.return_value = uploader

    csv = MagicMock()
    csv_cls.return_value = csv

    val = MagicMock()
    val.validate.return_value = MagicMock(passed=True, summary=lambda: "ok")
    val_cls.return_value = val

    return {
        "meta": meta,
        "sched": sched,
        "social": social,
        "uploader": uploader,
        "csv": csv,
        "drive_schedule": drive_schedule,
    }


class TestPhaseDYouTubeUpload:
    def test_lfs_uploaded_via_youtube(self, phase_d_state, monkeypatch):
        yt = _install_fake_youtube_uploader(monkeypatch)
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
            _setup_mocks((m0, m1, m2, m3, m4, m5, m6, m7))
            PhaseD(get_config()).run(phase_d_state)

        assert yt.upload_if_needed.call_count == 2
        for call in yt.upload_if_needed.call_args_list:
            kw = call.kwargs
            assert "title" in kw
            assert "description" in kw
            assert kw.get("category_id") == "2"
            assert kw.get("privacy") == "private"
            assert "publish_at" in kw

        assert "youtube_uploads" in phase_d_state
        assert set(phase_d_state["youtube_uploads"].keys()) == {1, 2}
        for lf_idx in (1, 2):
            entry = phase_d_state["youtube_uploads"][lf_idx]
            assert entry["video_id"]
            assert entry["url"].startswith("https://youtu.be/")

    def test_upload_uses_drive_schedule_dates_when_no_start_date(self, phase_d_state, monkeypatch):
        """Regression: when ``state["start_date"]`` is unset but
        ``drive_schedule.lf_slots`` carries Wednesday dates (from
        schedule_constants + --publish wed), the YouTube upload must
        use those Wednesday dates, NOT fall back to plan_weekly_mondays
        which would schedule on Mondays.

        The defect this guards: the upload path was gated on start_date
        being set, so a weekday-only publish request (--publish wed alone)
        fell into the Monday planner and scheduled LFs on Mondays instead of
        the requested Wednesday.
        """
        yt = _install_fake_youtube_uploader(monkeypatch)
        # Deliberately no start_date in state.
        assert "start_date" not in phase_d_state

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
            mocks = _setup_mocks((m0, m1, m2, m3, m4, m5, m6, m7))

            # Override lf_slots on the drive_schedule mock to carry
            # explicit Wednesday dates (what the schedule builder produces).
            wed_slots = [
                MagicMock(
                    date="2026-04-22",
                    time="11:00:00",
                    datetime_est="2026-04-22T11:00:00-04:00",
                    day_of_week="Wednesday",
                ),
                MagicMock(
                    date="2026-04-29",
                    time="11:00:00",
                    datetime_est="2026-04-29T11:00:00-04:00",
                    day_of_week="Wednesday",
                ),
            ]
            mocks["drive_schedule"].lf_slots = wed_slots

            PhaseD(get_config()).run(phase_d_state)

        publish_ats = [call.kwargs.get("publish_at") for call in yt.upload_if_needed.call_args_list]
        # publish_at must come from drive_schedule's Wed slots, not the
        # Monday fallback.
        assert len(publish_ats) == 2
        for pa, expected_date in zip(publish_ats, ["2026-04-22", "2026-04-29"]):
            assert pa is not None, "publish_at is None; upload dates lost"
            # Dates may be in UTC (different from EST); compare ISO prefix
            iso = pa.isoformat() if hasattr(pa, "isoformat") else str(pa)
            # 2026-04-22T11:00:00-04:00 = 2026-04-22T15:00:00+00:00 UTC
            # but the DATE component in UTC is still 2026-04-22
            assert iso.startswith(expected_date), (
                f"publish_at should be {expected_date} (from drive_schedule), "
                f"not Monday-planner fallback; got {iso}"
            )
            # Guard against Monday fallback explicitly
            from datetime import date

            pa_date = date.fromisoformat(iso[:10])
            assert pa_date.strftime("%A") == "Wednesday", (
                f"publish_at landed on {pa_date.strftime('%A')} {iso[:10]}; "
                f"must be Wednesday (from drive_schedule)"
            )

    def test_lfs_not_uploaded_to_drive(self, phase_d_state, monkeypatch):
        _install_fake_youtube_uploader(monkeypatch)
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
            m = _setup_mocks((m0, m1, m2, m3, m4, m5, m6, m7))
            PhaseD(get_config()).run(phase_d_state)

        # upload_single_file must NOT be called (used to handle LFs + super LFs)
        m["uploader"].upload_single_file.assert_not_called()

    def test_quota_error_saves_state_and_aborts_cleanly(self, phase_d_state, monkeypatch):
        """Phase D must treat ``YouTubeQuotaError`` distinctly from a permanent
        auth failure: save state via the save_callback, log an operator-facing
        message, and propagate so the run exits cleanly for resume after the
        quota window resets (midnight Pacific Time). Without disambiguation a multi-LF batch
        would abort mid-run on a generic HTTP 403 with no recovery hint.

        Contrast with ``_RefreshError`` (auth): requires credential rotation,
        not a time-based retry. Both escape the per-LF ``continue`` but through
        different paths so telemetry can distinguish them.
        """
        from gomoveshift.pipeline.exceptions import YouTubeQuotaError

        yt = _install_fake_youtube_uploader(monkeypatch)
        # Override upload_if_needed to raise YouTubeQuotaError on first call.
        quota_exc = YouTubeQuotaError(
            reason="quotaExceeded",
            message="Daily YouTube Data API quota exceeded; retry after midnight Pacific Time",
        )
        yt.upload_if_needed.side_effect = quota_exc

        # Provide a save_callback so we can verify Phase D persists state
        # (youtube_uploads accumulated so far) before raising.
        saved_states: list[dict] = []

        def save_cb(s):
            saved_states.append(dict(s.get("youtube_uploads") or {}))

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
            _setup_mocks((m0, m1, m2, m3, m4, m5, m6, m7))
            phase_d = PhaseD(get_config(), save_callback=save_cb)
            with pytest.raises(YouTubeQuotaError):
                phase_d.run(phase_d_state)

        # YouTubeQuotaError must propagate — the outer orchestrator exits
        # cleanly so the operator can resume after the quota window resets.
        # State save must have fired at least once (checkpoint for resume).
        assert saved_states, "Phase D must save state on YouTubeQuotaError for resume"
        # No per-LF ``continue`` silenced the error — only one upload attempt
        # was made before the batch aborted.
        assert yt.upload_if_needed.call_count == 1


class TestPhaseDConsolidatedCSV:
    def test_single_csv_written_with_correct_row_count(self, phase_d_state, monkeypatch, tmp_path):
        _install_fake_youtube_uploader(monkeypatch)
        # Use a real CSVGenerator so the consolidated CSV is actually written.
        with (
            _phase_d_patches()[0] as m0,
            _phase_d_patches()[1] as m1,
            _phase_d_patches()[2] as m2,
            _phase_d_patches()[3] as m3,
            _phase_d_patches()[4] as m4,
            _phase_d_patches()[5] as m5,
            _phase_d_patches()[7] as m7,
        ):
            _setup_mocks((m0, m1, m2, m3, m4, m5, MagicMock(), m7))
            # don't patch CSVGenerator — use real one
            PhaseD(get_config()).run(phase_d_state)

        csvs = list(phase_d_state["output_base"].glob("*_shorts_metricool.csv"))
        assert len(csvs) == 1, f"expected one consolidated CSV, got {csvs}"

        import csv as csvmod

        with open(csvs[0], encoding="utf-8-sig") as f:
            rows = list(csvmod.DictReader(f))
        # 2 LFs × 3 shorts each = 6 rows
        assert len(rows) == 6

    def test_first_comment_maps_to_parent_lf_youtube_url(self, phase_d_state, monkeypatch):
        _install_fake_youtube_uploader(monkeypatch, video_id_prefix="ytid")
        with (
            _phase_d_patches()[0] as m0,
            _phase_d_patches()[1] as m1,
            _phase_d_patches()[2] as m2,
            _phase_d_patches()[3] as m3,
            _phase_d_patches()[4] as m4,
            _phase_d_patches()[5] as m5,
            _phase_d_patches()[7] as m7,
        ):
            _setup_mocks((m0, m1, m2, m3, m4, m5, MagicMock(), m7))
            PhaseD(get_config()).run(phase_d_state)

        csv_path = next(phase_d_state["output_base"].glob("*_shorts_metricool.csv"))
        import csv as csvmod

        with open(csv_path, encoding="utf-8-sig") as f:
            rows = list(csvmod.DictReader(f))
        for row in rows:
            fc = row["First Comment Text"]
            assert fc.startswith("▶ Full drive: https://youtu.be/")
        # Rows for LF1 should reference one video id, LF2 another
        lf1_rows = rows[:3]
        lf2_rows = rows[3:]
        assert lf1_rows[0]["First Comment Text"] == lf1_rows[1]["First Comment Text"]
        assert lf1_rows[0]["First Comment Text"] != lf2_rows[0]["First Comment Text"]


class TestPhaseDThumbnails:
    def test_set_thumbnail_called_once_per_lf(self, phase_d_state, monkeypatch, tmp_path):
        yt = _install_fake_youtube_uploader(monkeypatch)

        # Create stub thumbnail so copyfile succeeds
        stub_thumb = tmp_path / "thumb.jpg"
        stub_thumb.write_bytes(b"\xff\xd8stub")

        # Mock thumbnail generator to avoid ffmpeg/PIL work
        with patch(
            "gomoveshift.pipeline.drive_phases.thumbnail_generator.generate_lf_thumbnail",
            return_value=stub_thumb,
        ):
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
                _setup_mocks((m0, m1, m2, m3, m4, m5, m6, m7))
                PhaseD(get_config()).run(phase_d_state)

        assert yt.set_thumbnail.call_count == 2
        for lf_idx in (1, 2):
            assert "thumbnail_path" in phase_d_state["youtube_uploads"][lf_idx]


class TestPhaseDRecordingDate:
    """Phase D must thread each LF's recording date into
    ``LongFormMetadataGenerator.generate(recording_date=...)`` so the
    description gets the standardized ``📅 Filmed: {date}`` line (built in
    ``LongFormMetadataGenerator._build_gms_description``).

    The source of truth is
    ``state["per_lf_recording_date"]``, populated by
    ``phase_bc_day_first._build_per_lf_recording_date_map`` from the
    day-batch partition. Indexing into ``state["episode_dates"]`` via
    ``segments[0].source_file_index`` would be wrong because
    ``_partition_lf_plans_by_date`` remaps every segment's index from
    GLOBAL to DAY-LOCAL — the lookup would return the first global
    episode's date for every LF.
    """

    def test_recording_date_threaded_per_lf(self, phase_d_state, monkeypatch):
        _install_fake_youtube_uploader(monkeypatch)

        # Fixture has 2 LFs (indices 1 and 2). The day-batch partition
        # gives each LF its own date, regardless of the (now day-local)
        # source_file_index on its segments.
        phase_d_state["per_lf_recording_date"] = {
            1: "2026-02-20",
            2: "2026-02-22",
        }

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
            mocks = _setup_mocks((m0, m1, m2, m3, m4, m5, m6, m7))
            PhaseD(get_config()).run(phase_d_state)

            generate_calls = mocks["meta"].generate.call_args_list
            dates_seen = [c.kwargs.get("recording_date") for c in generate_calls]
            assert dates_seen == ["2026-02-20", "2026-02-22"], (
                f"recording_date must be threaded per-LF from "
                f"state['per_lf_recording_date']; got {dates_seen}"
            )

    def test_recording_date_none_when_episode_dates_missing(self, phase_d_state, monkeypatch):
        """When state has no ``per_lf_recording_date`` entries AND no
        on-disk ``day_<DATE>/regular_lfs/lf_NN.mp4`` layout for the
        backstop to derive from, Phase D's BC→D boundary contract
        (``assert_phase_d_per_lf_inputs_present``) raises
        ``PhaseDEntryContractError`` at phase entry rather than silently
        emitting a CSV without recording dates.

        A ``recording_date`` of None (the legacy None-tolerance path) is
        unrepresentable for any LF with shorts, and the boundary contract
        makes it unrepresentable at the phase boundary for every LF.
        """
        _install_fake_youtube_uploader(monkeypatch)
        # Strip the fixture default so the strict assertion fires.
        phase_d_state["per_lf_recording_date"] = {}
        assert "episode_dates" not in phase_d_state

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
            _setup_mocks((m0, m1, m2, m3, m4, m5, m6, m7))
            with pytest.raises(ValueError, match="per_lf_recording_date"):
                PhaseD(get_config()).run(phase_d_state)

    def test_recording_date_none_when_lf_index_missing_from_map(self, phase_d_state, monkeypatch):
        """When ``per_lf_recording_date`` is present but missing an entry
        for an LF that has shorts, Phase D's BC→D boundary contract
        raises ``PhaseDEntryContractError`` at phase entry.
        The legacy None-tolerance behaviour
        (``recording_date=None`` for the missing LF) is unrepresentable.
        """
        _install_fake_youtube_uploader(monkeypatch)
        # Map only covers LF 1; LF 2 is intentionally absent.
        phase_d_state["per_lf_recording_date"] = {1: "2026-02-20"}

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
            _setup_mocks((m0, m1, m2, m3, m4, m5, m6, m7))
            with pytest.raises(ValueError, match="per_lf_recording_date"):
                PhaseD(get_config()).run(phase_d_state)


class TestPhaseDTitleOverridePerLF:
    """``state["per_lf_titles"]`` must flow into
    ``LongFormMetadataGenerator.generate(title_override=...)`` so the
    canonical endpoint-places + via-road title produced by
    title_builder in phase_bc wins over the SEO manager's template
    (the OST title format).
    """

    def test_title_override_flows_per_lf(self, phase_d_state, monkeypatch):
        _install_fake_youtube_uploader(monkeypatch)

        phase_d_state["per_lf_titles"] = {
            1: "Port Royal to Lady's Island via Sea Island Parkway, SC",
            2: "Macon Loop via Blue Ridge Parkway, NC",
        }

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
            mocks = _setup_mocks((m0, m1, m2, m3, m4, m5, m6, m7))
            PhaseD(get_config()).run(phase_d_state)

            # ``meta.generate`` was called once per LF (2 fixtures)
            generate_calls = mocks["meta"].generate.call_args_list
            assert len(generate_calls) == 2

            # Each call carries the per-LF title_override kwarg.
            overrides_seen = [c.kwargs.get("title_override") for c in generate_calls]
            assert overrides_seen == [
                "Port Royal to Lady's Island via Sea Island Parkway, SC",
                "Macon Loop via Blue Ridge Parkway, NC",
            ], f"title_override not threaded per-LF; got {overrides_seen}"

    def test_title_override_none_when_per_lf_titles_missing(self, phase_d_state, monkeypatch):
        """When state has no ``per_lf_titles`` dict, ``title_override``
        passed to generate() must be None (not a KeyError, not an empty
        string)."""
        _install_fake_youtube_uploader(monkeypatch)
        # Deliberately omit per_lf_titles.
        assert "per_lf_titles" not in phase_d_state

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
            mocks = _setup_mocks((m0, m1, m2, m3, m4, m5, m6, m7))
            PhaseD(get_config()).run(phase_d_state)

            generate_calls = mocks["meta"].generate.call_args_list
            for c in generate_calls:
                assert c.kwargs.get("title_override") is None, (
                    f"Missing per_lf_titles must result in title_override=None; "
                    f"got {c.kwargs.get('title_override')!r}"
                )


class TestPhaseDReadsPerLFOSTTextForTitles:
    """Integration pin: Phase D reads ``state["per_lf_ost_text"]``
    (the canonical pre-dedup OST text used for titles) and threads
    it through ``LongFormMetadataGenerator.generate`` so YouTube titles
    derive from the same source as the OST burn-in and thumbnail.

    OST text, title and
    thumbnail stay aligned, all from one state key. ``state["per_lf_titles"]``
    (dedup-suffixed) still wins for uniqueness, but the base text must
    originate from per_lf_ost_text.

    Contract: when both per_lf_ost_text and per_lf_titles are present,
    the title passed to YouTube/meta is the dedup-suffixed title
    (per_lf_titles), which is built from per_lf_ost_text upstream.
    The LF metadata and the fallback must always resolve back to the
    canonical per_lf_ost_text entry, not to an orphan template.
    """

    def test_phase_d_reads_per_lf_ost_text_for_titles(self, phase_d_state, monkeypatch):
        """When ONLY per_lf_ost_text is in state (no per_lf_titles),
        Phase D must still derive correct title_overrides — running
        ensure_unique_titles locally over per_lf_ost_text. This pins
        per_lf_ost_text as the PRIMARY source for titles; per_lf_titles
        becomes a derived/legacy fallback.

        Contract: identical per_lf_ost_text across two LFs →
        title_override for each gets "(Part N of M)" dedup suffix,
        matching what phase_bc would have produced via ensure_unique_titles.
        """
        _install_fake_youtube_uploader(monkeypatch)

        shared = "Port Royal to Lady's Island via Sea Island Parkway, SC"
        canonical_ost = {1: shared, 2: shared}
        phase_d_state["per_lf_ost_text"] = canonical_ost
        # Deliberately omit per_lf_titles — Phase D must reconstruct
        # the titles itself from per_lf_ost_text when it's available.
        assert "per_lf_titles" not in phase_d_state

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
            mocks = _setup_mocks((m0, m1, m2, m3, m4, m5, m6, m7))
            PhaseD(get_config()).run(phase_d_state)

            generate_calls = mocks["meta"].generate.call_args_list
            assert len(generate_calls) == 2, (
                f"Expected 2 generate() calls, got {len(generate_calls)}"
            )

            overrides = [c.kwargs.get("title_override") for c in generate_calls]

            # Base alignment: every title_override starts with the
            # canonical per_lf_ost_text for its LF (dedup suffix
            # appended, never rewritten).
            for lf_idx, override in zip((1, 2), overrides):
                assert override is not None, (
                    f"title_override missing for LF {lf_idx} despite "
                    f"per_lf_ost_text being present — phase_d must derive "
                    f"titles from per_lf_ost_text"
                )
                assert override.startswith(canonical_ost[lf_idx]), (
                    f"title_override for LF {lf_idx} diverges from "
                    f"per_lf_ost_text — violates OST/title alignment. "
                    f"ost_text={canonical_ost[lf_idx]!r}, "
                    f"title_override={override!r}"
                )

            # Dedup: identical ost_text across two LFs → overrides differ
            # (ensure_unique_titles ran locally over per_lf_ost_text).
            assert overrides[0] != overrides[1], (
                f"Expected Phase D to run ensure_unique_titles over "
                f"per_lf_ost_text; both overrides equal {overrides[0]!r}"
            )
            # Both overrides carry the "(Part N of M)" suffix shape.
            assert "(Part 1 of 2)" in overrides[0]
            assert "(Part 2 of 2)" in overrides[1]

    def test_phase_d_legacy_per_lf_titles_still_works(self, phase_d_state, monkeypatch):
        """Backward compat: when per_lf_ost_text is absent but
        per_lf_titles is present (legacy state from older pipeline
        runs), phase_d still uses per_lf_titles verbatim. Avoids
        breaking resumable state files written by older versions."""
        _install_fake_youtube_uploader(monkeypatch)

        phase_d_state["per_lf_titles"] = {
            1: "Port Royal to Lady's Island via Sea Island Parkway, SC",
            2: "Macon Loop via Blue Ridge Parkway, NC",
        }
        assert "per_lf_ost_text" not in phase_d_state

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
            mocks = _setup_mocks((m0, m1, m2, m3, m4, m5, m6, m7))
            PhaseD(get_config()).run(phase_d_state)

            overrides = [
                c.kwargs.get("title_override") for c in mocks["meta"].generate.call_args_list
            ]
            assert overrides == [
                "Port Royal to Lady's Island via Sea Island Parkway, SC",
                "Macon Loop via Blue Ridge Parkway, NC",
            ], f"Legacy per_lf_titles fallback broken; got {overrides}"

    def test_ost_text_stays_clean_when_title_dedup_fires(self, phase_d_state, monkeypatch):
        """After Phase D runs, ``state["per_lf_ost_text"]`` must remain
        free of ``"(Part N of M)"`` suffixes — the dedup policy is a
        TITLE-only concern. OST burn-in and thumbnail text read
        per_lf_ost_text verbatim; letting the dedup suffix leak in
        breaks the OST/title/thumbnail alignment (burned
        pixels don't match displayed title)."""
        _install_fake_youtube_uploader(monkeypatch)

        shared = "Port Royal to Lady's Island via Sea Island Parkway, SC"
        canonical_ost = {1: shared, 2: shared}
        dedup_titles = {
            1: f"{shared} (Part 1 of 2)",
            2: f"{shared} (Part 2 of 2)",
        }
        phase_d_state["per_lf_ost_text"] = canonical_ost
        phase_d_state["per_lf_titles"] = dedup_titles

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
            _setup_mocks((m0, m1, m2, m3, m4, m5, m6, m7))
            PhaseD(get_config()).run(phase_d_state)

            # After Phase D: per_lf_ost_text MUST still be pristine —
            # Phase D must never mutate it, and dedup suffix must stay
            # out of the canonical OST text.
            post_ost = phase_d_state.get("per_lf_ost_text") or {}
            assert post_ost == canonical_ost, (
                f"Phase D mutated per_lf_ost_text — canonical source "
                f"corruption. Before: {canonical_ost}. After: {post_ost}"
            )
            for idx, text in post_ost.items():
                assert "(Part " not in text, (
                    f"per_lf_ost_text[{idx}] picked up a dedup suffix after "
                    f"Phase D — OST burn would show '(Part N of M)' as if "
                    f"it were part of the route name. Got: {text!r}"
                )


class TestPhaseDSocialGeneratorPerShortLocation:
    """Each short's social surfaces (caption/title/tags/description) must be
    fed that short's OWN resolved location — the burned ``ost_text`` recorded
    rank-keyed in the day's ``shorts_ost_audit_<date>.json`` sidecar — never
    a blanket parent-LF label, and never a drive-wide string (the drive
    invariant: no drive-wide location fallback exists).

    A check on the per-LF constructor kwarg alone would let every short of an
    LF inherit the LF label: a short whose own burned OST says "Conor Pass"
    must get captions/tags for that place, not for the LF's A→B route.
    """

    @staticmethod
    def _mk_day_shorts(tmp_path, date, ost_by_rank, write_sidecar=True):
        """Create day-first short paths (+ the rank-keyed OST sidecar)."""
        stages = tmp_path / "_stages"
        shorts_dir = stages / f"day_{date}" / "shorts"
        shorts_dir.mkdir(parents=True)
        if write_sidecar:
            payload = {
                "date": date,
                "shorts": [
                    {
                        "rank": r,
                        "parent_lf_index": 1,
                        "ost_text": t,
                        "start_sec": float(r),
                        "end_sec": float(r) + 30.0,
                    }
                    for r, t in ost_by_rank.items()
                ],
            }
            (stages / f"shorts_ost_audit_{date}.json").write_text(json.dumps(payload))
        return [str(shorts_dir / f"short_{r:02d}.mp4") for r in ost_by_rank]

    def test_each_short_uses_its_own_sidecar_ost_text(self, tmp_path):
        """Two shorts of ONE LF with different sidecar ost_text must yield
        posts anchored on different locations — each short's own."""
        from gomoveshift.pipeline.drive_phases.phase_d import _build_social_posts_per_lf
        from gomoveshift.pipeline.social_generator import SocialGenerator

        short_paths = self._mk_day_shorts(
            tmp_path,
            "2026-06-01",
            {1: "Dingle Peninsula, County Kerry", 2: "Conor Pass, County Kerry"},
        )
        out = _build_social_posts_per_lf(
            social_gen_cls=SocialGenerator,
            drive_plan=SimpleNamespace(regular_lfs=[SimpleNamespace(index=1)]),
            shorts_per_lf={1: short_paths},
            per_lf_locations_override={1: "County Kerry, Ireland"},
            project_base="proj",
            output_base=tmp_path / "out",
        )
        posts = out[1]
        assert len(posts) == 2
        assert "Dingle Peninsula" in posts[0].youtube_description, (
            f"short 1 must anchor on its OWN sidecar ost_text, got: "
            f"{posts[0].youtube_description!r}"
        )
        assert "Conor Pass" in posts[1].youtube_description, (
            f"short 2 must anchor on its OWN sidecar ost_text, got: "
            f"{posts[1].youtube_description!r}"
        )
        assert "Dingle Peninsula" in posts[0].youtube_tags
        assert "Conor Pass" in posts[1].youtube_tags
        assert "Dingle Peninsula" not in posts[1].youtube_tags, (
            "short 2 inherited short 1's label — per-short resolution broken"
        )

    def test_absent_sidecar_falls_back_to_parent_lf_label_with_warn(self, tmp_path, caplog):
        """No sidecar → each short falls back to its parent-LF resolved
        label, and the fallback is LOUD (WARN), never silent."""
        from gomoveshift.pipeline.drive_phases.phase_d import _build_social_posts_per_lf
        from gomoveshift.pipeline.social_generator import SocialGenerator

        short_paths = self._mk_day_shorts(
            tmp_path, "2026-06-01", {1: "unused", 2: "unused"}, write_sidecar=False
        )
        with caplog.at_level(logging.WARNING, logger="gomoveshift.pipeline.drive_phases.phase_d"):
            out = _build_social_posts_per_lf(
                social_gen_cls=SocialGenerator,
                drive_plan=SimpleNamespace(regular_lfs=[SimpleNamespace(index=1)]),
                shorts_per_lf={1: short_paths},
                per_lf_locations_override={1: "County Kerry, Ireland"},
                project_base="proj",
                output_base=tmp_path / "out",
            )
        posts = out[1]
        assert all("County Kerry" in p.youtube_description for p in posts), (
            "fallback must be the parent-LF label"
        )
        assert any("parent-LF label" in r.message for r in caplog.records), (
            f"per-short label fallback must WARN loudly; got: {[r.message for r in caplog.records]}"
        )

    def test_drive_wide_location_fallback_is_unrepresentable(self):
        """Contract: no drive-wide location can reach a short's social
        surfaces through this builder — the parameter itself is gone
        (Phase D's entry contract already hard-fails upstream on a
        ``per_lf_locations`` gap, so the fallback was dead code)."""
        from gomoveshift.pipeline.drive_phases.phase_d import _build_social_posts_per_lf

        params = inspect.signature(_build_social_posts_per_lf).parameters
        assert "location" not in params, (
            "drive-wide `location` param resurfaced in _build_social_posts_per_lf"
        )

    def test_phase_d_raises_when_per_lf_locations_missing_entry(self, phase_d_state, monkeypatch):
        """BC→D boundary contract: if ``per_lf_locations`` lacks an
        entry for any rendered LF, Phase D raises ``PhaseDEntryContractError``
        at phase entry. The legacy drive-wide fallback is unrepresentable —
        BC populates an entry per LF in production, and a missing entry
        signals state corruption rather than intent to fall back.
        """
        _install_fake_youtube_uploader(monkeypatch)

        phase_d_state["location"] = "Drive Wide, USA"
        phase_d_state["per_lf_locations"] = {1: "Beaufort County, SC"}

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
            _setup_mocks((m0, m1, m2, m3, m4, m5, m6, m7))
            with pytest.raises(ValueError, match="per_lf_locations"):
                PhaseD(get_config()).run(phase_d_state)


# ── Phase D sidecar integrity gate ───────────────────────────────


class TestPhaseDSidecarIntegrityGate:
    """Phase D must call the sidecar integrity guard at entry.

    Background: ``youtube_uploads.json`` at ``<output_base>/`` is a local
    record of prior upload attempts, NOT a live mirror of YouTube. If the
    operator deletes drafts in YouTube Studio — or a prior aborted run
    left phantom slots — the pipeline otherwise treats those IDs as live
    and bakes dead video URLs into the Metricool CSV.

    The gate is strictly local: it checks only
    len(sidecar) == len(drive_plan.regular_lfs) + index-set agreement.
    API-level reconciliation is out of scope.
    """

    def test_mismatched_sidecar_raises_before_any_upload(self, phase_d_state, monkeypatch):
        """A sidecar with more entries than the plan has LFs must halt
        Phase D before any upload work starts — SidecarIntegrityError
        carries the phantom indices operators need."""
        from gomoveshift.pipeline.exceptions import SidecarIntegrityError
        from gomoveshift.pipeline.youtube_sidecar import save_sidecar

        output_base = phase_d_state["output_base"]
        # Plan has 2 LFs (from fixture); write a sidecar with 5 entries.
        save_sidecar(
            output_base,
            {
                i: {"video_id": f"phantom{i:02d}", "url": f"https://youtu.be/ph{i:02d}"}
                for i in range(1, 6)
            },
        )
        yt = _install_fake_youtube_uploader(monkeypatch)
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
            _setup_mocks((m0, m1, m2, m3, m4, m5, m6, m7))
            with pytest.raises(SidecarIntegrityError) as exc_info:
                PhaseD(get_config()).run(phase_d_state)
        # Guard must fire before any YouTube upload attempt.
        assert yt.upload_if_needed.call_count == 0
        err = exc_info.value
        # Plan has LFs 1..2; sidecar has 1..5 → phantoms 3, 4, 5.
        assert set(err.phantom_indices) == {3, 4, 5}

    def test_empty_sidecar_still_runs_phase_d(self, phase_d_state, monkeypatch):
        """An empty/missing sidecar is the normal fresh-run state. The
        guard must not interfere with uploads."""
        yt = _install_fake_youtube_uploader(monkeypatch)
        # No sidecar written — fixture's output_base is empty.
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
            _setup_mocks((m0, m1, m2, m3, m4, m5, m6, m7))
            PhaseD(get_config()).run(phase_d_state)
        # Should hit the upload path normally.
        assert yt.upload_if_needed.call_count == 2

    def test_force_reset_flag_wipes_and_proceeds(self, phase_d_state, monkeypatch):
        """--force-sidecar-reset must wipe the sidecar BEFORE the gate
        runs, so Phase D proceeds even when the on-disk sidecar is
        stale. Operators use this after verifying via YouTube Studio
        that the referenced videos are gone."""
        from gomoveshift.pipeline.youtube_sidecar import load_sidecar, save_sidecar

        output_base = phase_d_state["output_base"]
        # Stale sidecar with 5 entries vs 2-LF plan.
        save_sidecar(
            output_base,
            {
                i: {"video_id": f"phantom{i:02d}", "url": f"https://youtu.be/ph{i:02d}"}
                for i in range(1, 6)
            },
        )
        assert len(load_sidecar(output_base)) == 5

        # Set the reset flag on state (as the orchestrator will).
        phase_d_state["force_sidecar_reset"] = True
        # Also wipe any state.youtube_uploads so the reset behaviour is
        # observable end-to-end.
        phase_d_state.youtube_uploads = None

        yt = _install_fake_youtube_uploader(monkeypatch)
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
            _setup_mocks((m0, m1, m2, m3, m4, m5, m6, m7))
            PhaseD(get_config()).run(phase_d_state)
        # After reset + run, on-disk sidecar should reflect ONLY the
        # new uploads (2 LFs), not the 5 phantoms.
        fresh = load_sidecar(output_base)
        # fresh may be 0 if we never save mid-run, or {1, 2} after the
        # uploads. Either way, none of the phantom IDs survive.
        assert all(rec.get("video_id", "").startswith("vid") for rec in fresh.values()), (
            f"phantom IDs survived --force-sidecar-reset: {fresh}. "
            "Reset must wipe before the upload path runs."
        )
        assert yt.upload_if_needed.call_count == 2


class TestPhaseDContentGateRunsBeforeUpload:
    """The location invariant and audio-loudness HARD FAIL gates
    (``_run_final_validation_gate``) must abort the run BEFORE any Drive or
    YouTube upload — their docstrings promise "the run aborts before any
    Drive/YouTube upload". The gate must run BEFORE the uploads in
    ``PhaseD.run``, not after them, or a far-road / off-LUFS LF would already
    be published when it raised.
    """

    def _failing_gate_result(self, gate_name, message):
        gate = MagicMock(gate=gate_name, passed=False, message=message)
        return MagicMock(passed=False, gates=[gate], summary=lambda: message)

    def test_location_invariant_failure_aborts_before_any_upload(self, phase_d_state, monkeypatch):
        yt = _install_fake_youtube_uploader(monkeypatch)
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
            mocks = _setup_mocks((m0, m1, m2, m3, m4, m5, m6, m7))
            # Per-LF content validation reports the far-road class: the
            # location_invariant gate fails (dominant county in ZERO regions).
            m7.return_value.validate.return_value = self._failing_gate_result(
                "location_invariant",
                "location invariant HARD FAIL: title county "
                "'Oakland County, North Carolina' in ZERO sampled regions",
            )
            with pytest.raises(RuntimeError, match="Location invariant HARD FAIL"):
                PhaseD(get_config()).run(phase_d_state)
        # The gate must fire upstream of every publish action.
        assert mocks["uploader"].upload_shorts.call_count == 0, (
            "Drive shorts upload ran before the location-invariant gate aborted"
        )
        assert yt.upload_if_needed.call_count == 0, (
            "YouTube upload ran before the location-invariant gate aborted"
        )

    def test_audio_loudness_failure_aborts_before_any_upload(self, phase_d_state, monkeypatch):
        yt = _install_fake_youtube_uploader(monkeypatch)
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
            mocks = _setup_mocks((m0, m1, m2, m3, m4, m5, m6, m7))
            m7.return_value.validate.return_value = self._failing_gate_result(
                "audio_loudness",
                "integrated loudness -24.9 LUFS outside -20 +/- 2 window",
            )
            with pytest.raises(RuntimeError, match="Audio loudness HARD FAIL"):
                PhaseD(get_config()).run(phase_d_state)
        assert mocks["uploader"].upload_shorts.call_count == 0, (
            "Drive shorts upload ran before the audio-loudness gate aborted"
        )
        assert yt.upload_if_needed.call_count == 0, (
            "YouTube upload ran before the audio-loudness gate aborted"
        )


class TestPhaseDTitleQualityGateRunsBeforeUpload:
    """Title-text quality (distinctness / seo_degraded) must abort the run
    BEFORE any Drive or YouTube upload. These rules are
    computable from the LF metadata the moment it is built, so a violation
    has to block the bad-title video from ever being published — not fail at
    end-of-Phase-D after the LF is already public.
    """

    def test_duplicate_titles_abort_before_any_upload(self, phase_d_state, monkeypatch):
        from gomoveshift.pipeline.exceptions import PhaseQualityError

        yt = _install_fake_youtube_uploader(monkeypatch)
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
            mocks = _setup_mocks((m0, m1, m2, m3, m4, m5, m6, m7))
            # Force identical titles across every LF → the Rule-1 distinctness
            # violation the upstream title gate must catch before upload.
            mocks["meta"].generate.side_effect = lambda **kw: {
                "title": "Identical Drive Title",
                "description": "desc",
                "tags": ["driving", "ireland"],
                "category": "Travel",
            }
            with pytest.raises(PhaseQualityError, match="titles"):
                PhaseD(get_config()).run(phase_d_state)
        assert mocks["uploader"].upload_shorts.call_count == 0, (
            "Drive shorts upload ran before the title-quality gate aborted"
        )
        assert yt.upload_if_needed.call_count == 0, (
            "YouTube upload ran before the title-quality gate aborted"
        )
