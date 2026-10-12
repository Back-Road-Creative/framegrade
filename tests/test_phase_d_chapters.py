"""Phase D: Chapter generation integration tests.

Verifies:
  - ChapterGenerator is called during Phase D for each LF.
  - Generated chapters are embedded into the YouTube description via _embed_chapters.
  - Chapters are stored in metadata_per_lf and the on-disk JSON file.
  - Chapter generation failure is non-blocking (logs warning, continues).
  - When no GPX files exist, chapters step is skipped gracefully.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.config import get_config
from gomoveshift.models import DrivePlan, LFSegment, RegularLFPlan
from gomoveshift.pipeline.drive_phases.phase_d import PhaseD

from conftest import fake_generate_lf_thumbnail  # type: ignore[import-not-found]


def _install_fake_youtube_uploader(monkeypatch, video_id_prefix="vid"):
    """Install a fake youtube_uploader module for Phase D tests."""
    fake_mod = ModuleType("gomoveshift.pipeline.drive_phases.youtube_uploader")

    uploader_instance = MagicMock()
    call_counter = {"n": 0}

    def _upload_video(video_path, **kwargs):
        call_counter["n"] += 1
        return f"{video_id_prefix}_{call_counter['n']:02d}"

    uploader_instance.upload_video.side_effect = _upload_video
    uploader_instance.set_thumbnail.return_value = True

    def _upload_if_needed(*, state, lf_idx, video_path, **kwargs):
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


@pytest.fixture()
def chapter_phase_d_state(tmp_path):
    """State dict with GPX files available for chapter generation."""
    output = tmp_path / "output"
    output.mkdir()
    source = tmp_path / "source_drive"
    source.mkdir()
    stages = output / "_stages"
    stages.mkdir()

    regular_lfs = []
    lf_paths = []
    gpx_files = []

    for i in range(2):
        # Create GPX file stubs
        gpx_path = source / f"ep_{i:03d}.gpx"
        gpx_path.write_text("<gpx></gpx>")
        gpx_files.append(str(gpx_path))

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
        lf_path.write_bytes(b"x" * (2 * 1024 * 1024))
        lf_paths.append(lf_path)

    drive_plan = DrivePlan(
        source_files=[],
        processed_files=[],
        regular_lfs=regular_lfs,
        score_threshold=0.5,
        total_source_duration=1200.0,
        total_highlight_duration=1200.0,
    )

    from _state_helpers import make_state, verified_shorts_per_lf

    # A publish run posts only shorts whose own clip GPS verified their place.
    shorts_per_lf = verified_shorts_per_lf(stages, {1: 3, 2: 3})

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
        gpx_files=gpx_files,
        # Phase BC invariant ``_assert_per_lf_recording_date_complete``
        # populates this map in production. Test fixtures bypass Phase BC,
        # so we thread the data the contract assertion in Phase D's
        # ``_build_recording_dates_by_short_index`` expects.
        per_lf_recording_date={1: "2025-09-14", 2: "2025-09-14"},
        # BC→D boundary contract — every rendered LF must appear here.
        per_lf_locations={1: "Cork, Ireland", 2: "Cork, Ireland"},
        per_lf_trim_ratio={1: 0.7, 2: 0.7},
    )


def _phase_d_patches():
    """Orchestrator-level patches required by Phase D."""
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


def _setup_mocks(mocks, num_lfs=2, description="A scenic drive."):
    (meta_cls, sched_cls, rs_cls, social_cls, upload_cls, manifest_cls, csv_cls, val_cls) = mocks

    # Rule: distinct titles per LF. side_effect keyed by project_name.
    meta = MagicMock()

    def _meta_side_effect(**kwargs):
        project_name = kwargs.get("project_name", "lf")
        return {
            "title": f"Title for {project_name}",
            "description": description,
            "tags": ["driving", "ireland"],
            "category": "Travel",
        }

    meta.generate.side_effect = _meta_side_effect
    meta_cls.return_value = meta

    # Rule: lf_slot dates must fall on Wednesdays. 2026-04-01 is Wed.
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
    social.generate_all.side_effect = [posts[i * 3 : (i + 1) * 3] for i in range(num_lfs)]
    social_cls.return_value = social

    uploader = MagicMock()

    def _fake_upload_shorts(*_args, shorts_dir=None, **_kwargs):
        # Day-first contract: ``upload_shorts`` returns a LIST of Drive links
        # (one per short path); ``_upload_lf_shorts_to_drive`` zips paths→links
        # into the per-LF ``UploadManifest.files``. A bare MagicMock return
        # iterated empty → every ``Picture Url 1`` blank (the
        # media-less-CSV bug the ``METRICOOL-CSV-NO-MEDIA`` gate now rejects).
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


class TestPhaseDChapterGeneration:
    """Chapter generation wiring into Phase D."""

    def test_chapters_generated_and_embedded_in_description(
        self,
        chapter_phase_d_state,
        monkeypatch,
    ):
        """ChapterGenerator is called per-LF, chapters embedded in description."""
        _install_fake_youtube_uploader(monkeypatch)

        fake_chapters = [
            {"timestamp": "0:00", "title": "Start - Cork, Ireland"},
            {"timestamp": "5:00", "title": "Entering Cobh, County Cork"},
            {"timestamp": "9:30", "title": "Finish - Cork, Ireland"},
        ]
        fake_chapter_text = (
            "0:00 Start - Cork, Ireland\n"
            "5:00 Entering Cobh, County Cork\n"
            "9:30 Finish - Cork, Ireland"
        )

        mock_gen_instance = MagicMock()
        mock_gen_instance.generate.return_value = fake_chapters
        mock_gen_instance.format_chapters.return_value = fake_chapter_text
        mock_gen_instance.recording_date = None
        mock_gen_instance.named_routes = []

        with patch(
            "gomoveshift.pipeline.drive_phases.phase_d.ChapterGenerator",
            return_value=mock_gen_instance,
        ):
            with patch(
                "gomoveshift.pipeline.drive_phases.thumbnail_generator.generate_lf_thumbnail",
                side_effect=fake_generate_lf_thumbnail,
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
                    _setup_mocks(
                        (m0, m1, m2, m3, m4, m5, m6, m7),
                        description="A scenic drive through Cork.\n\n#DrivingVideo #Ireland",
                    )
                    PhaseD(get_config()).run(chapter_phase_d_state)

        # ChapterGenerator.generate called once per LF
        assert mock_gen_instance.generate.call_count == 2

        # Chapters stored in metadata_per_lf
        for lf_idx in (1, 2):
            meta = chapter_phase_d_state["metadata_per_lf"][lf_idx]
            assert "chapters" in meta
            assert meta["chapters"] == fake_chapters
            assert "chapter_text" in meta

        # Description updated with embedded chapters
        for lf_idx in (1, 2):
            meta = chapter_phase_d_state["metadata_per_lf"][lf_idx]
            assert "CHAPTERS" in meta["description"]
            assert "Entering Cobh" in meta["description"]

    def test_chapters_passed_to_youtube_upload_description(
        self,
        chapter_phase_d_state,
        monkeypatch,
    ):
        """The YouTube upload receives the chapter-enriched description."""
        yt = _install_fake_youtube_uploader(monkeypatch)

        # A renderable block needs >= YOUTUBE_MIN_CHAPTERS (3) timestamps,
        # else the drive path skips the embed. Use 3 so the embed path is
        # exercised (this test's contract: chapters reach the upload).
        fake_chapters = [
            {"timestamp": "0:00", "title": "Start - Cork"},
            {"timestamp": "4:45", "title": "En route - Cork"},
            {"timestamp": "9:30", "title": "Finish - Cork"},
        ]
        fake_chapter_text = "0:00 Start - Cork\n4:45 En route - Cork\n9:30 Finish - Cork"

        mock_gen_instance = MagicMock()
        mock_gen_instance.generate.return_value = fake_chapters
        mock_gen_instance.format_chapters.return_value = fake_chapter_text
        mock_gen_instance.recording_date = None
        mock_gen_instance.named_routes = []

        with patch(
            "gomoveshift.pipeline.drive_phases.phase_d.ChapterGenerator",
            return_value=mock_gen_instance,
        ):
            with patch(
                "gomoveshift.pipeline.drive_phases.thumbnail_generator.generate_lf_thumbnail",
                side_effect=fake_generate_lf_thumbnail,
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
                    _setup_mocks(
                        (m0, m1, m2, m3, m4, m5, m6, m7),
                        description="A scenic drive.\n\n#DrivingVideo",
                    )
                    PhaseD(get_config()).run(chapter_phase_d_state)

        # Check that upload_if_needed was called with chapter-enriched description
        for call in yt.upload_if_needed.call_args_list:
            desc = call.kwargs["description"]
            assert "CHAPTERS" in desc

    def test_chapter_generation_failure_is_non_blocking(
        self,
        chapter_phase_d_state,
        monkeypatch,
    ):
        """If ChapterGenerator raises, Phase D continues without chapters."""
        _install_fake_youtube_uploader(monkeypatch)

        mock_gen_instance = MagicMock()
        mock_gen_instance.generate.side_effect = RuntimeError("GPS parse failed")

        with patch(
            "gomoveshift.pipeline.drive_phases.phase_d.ChapterGenerator",
            return_value=mock_gen_instance,
        ):
            with patch(
                "gomoveshift.pipeline.drive_phases.thumbnail_generator.generate_lf_thumbnail",
                side_effect=fake_generate_lf_thumbnail,
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
                    # Should NOT raise
                    PhaseD(get_config()).run(chapter_phase_d_state)

        # Pipeline still completed — metadata_per_lf exists, uploads happened
        assert "metadata_per_lf" in chapter_phase_d_state
        assert "youtube_uploads" in chapter_phase_d_state
        assert len(chapter_phase_d_state["youtube_uploads"]) == 2

    def test_no_gpx_files_skips_chapters_gracefully(
        self,
        chapter_phase_d_state,
        monkeypatch,
    ):
        """When state has no gpx_files, chapter step is skipped entirely."""
        _install_fake_youtube_uploader(monkeypatch)
        # Remove GPX files from state
        chapter_phase_d_state["gpx_files"] = []

        with patch(
            "gomoveshift.pipeline.drive_phases.phase_d.ChapterGenerator",
        ) as mock_cls:
            with patch(
                "gomoveshift.pipeline.drive_phases.thumbnail_generator.generate_lf_thumbnail",
                side_effect=fake_generate_lf_thumbnail,
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
                    PhaseD(get_config()).run(chapter_phase_d_state)

        # ChapterGenerator should not have been instantiated
        mock_cls.assert_not_called()
        # Pipeline still completed normally
        assert "metadata_per_lf" in chapter_phase_d_state

    def test_chapter_metadata_written_to_json_file(
        self,
        chapter_phase_d_state,
        monkeypatch,
    ):
        """Chapters are persisted in the on-disk _youtube.json files."""
        _install_fake_youtube_uploader(monkeypatch)

        fake_chapters = [
            {"timestamp": "0:00", "title": "Start - Cork"},
            {"timestamp": "9:30", "title": "Finish - Cork"},
        ]
        mock_gen_instance = MagicMock()
        mock_gen_instance.generate.return_value = fake_chapters
        mock_gen_instance.format_chapters.return_value = "0:00 Start\n9:30 Finish"
        mock_gen_instance.recording_date = None
        mock_gen_instance.named_routes = []

        with patch(
            "gomoveshift.pipeline.drive_phases.phase_d.ChapterGenerator",
            return_value=mock_gen_instance,
        ):
            with patch(
                "gomoveshift.pipeline.drive_phases.thumbnail_generator.generate_lf_thumbnail",
                side_effect=fake_generate_lf_thumbnail,
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
                    PhaseD(get_config()).run(chapter_phase_d_state)

        output_base = Path(chapter_phase_d_state["output_base"])
        for lf_idx in (1, 2):
            json_path = output_base / f"source_drive_lf{lf_idx:02d}_youtube.json"
            assert json_path.exists(), f"Expected {json_path} to exist"
            data = json.loads(json_path.read_text())
            assert "chapters" in data
            assert data["chapters"] == fake_chapters

    def test_chapter_import_failure_is_non_blocking(
        self,
        chapter_phase_d_state,
        monkeypatch,
    ):
        """If ChapterGenerator import fails, Phase D continues."""
        _install_fake_youtube_uploader(monkeypatch)

        with patch(
            "gomoveshift.pipeline.drive_phases.phase_d._generate_chapters_for_lf",
            side_effect=ImportError("chapters module not available"),
        ):
            with patch(
                "gomoveshift.pipeline.drive_phases.thumbnail_generator.generate_lf_thumbnail",
                side_effect=fake_generate_lf_thumbnail,
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
                    PhaseD(get_config()).run(chapter_phase_d_state)

        # Pipeline completed
        assert "metadata_per_lf" in chapter_phase_d_state
        assert len(chapter_phase_d_state["youtube_uploads"]) == 2
