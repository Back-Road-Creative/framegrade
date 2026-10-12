"""Phase D per-LF output completeness.

Covers silent-failure bugs in Phase D:

  CSV rows: ``CSVGenerator.generate_lf_row`` had zero production callers — Phase D
      never emitted ``<project>_lfNN_metricool.csv``.

  Thumbnail generation: The thumbnail generation loop was nested inside the
      ``if not skip_upload and youtube_upload_enabled`` block and keyed on
      ``youtube_uploads.get(lf.index)``, so with the invariant
      (``skip_upload=True`` / ``youtube_upload=False``) no thumbnails ever
      landed on disk.

  Thumbnail layout: ``thumbnail_generator.generate_lf_thumbnail`` writes to a
      ``thumbnails/lf_NN/`` subdir, but
      ``tools/validate_pipeline_output.check_thumbnails`` only sees
      top-level **files**. Fix must publish a canonical
      ``thumbnails/lf_NN.jpg`` the validator can see.
"""

from __future__ import annotations

import csv as csvmod
import sys
import time
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.config import get_config
from gomoveshift.models import DrivePlan, LFSegment, RegularLFPlan
from gomoveshift.pipeline.drive_phases.phase_d import PhaseD

# Repository root — the Metricool golden header lives there.
REFERENCE_HEADER = Path(__file__).resolve().parent.parent / "metricool_reference_header.csv"


# ── Helpers copied from test_phase_d.py ──────────────────────────────────


def _install_fake_youtube_uploader(monkeypatch, video_id_prefix="vid"):
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


def _build_state(tmp_path, num_lfs=3, youtube_upload=False, skip_upload=True):
    output = tmp_path / "output"
    output.mkdir()
    source = tmp_path / "source_drive"
    source.mkdir()

    regular_lfs = []
    lf_paths = []
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
                region="ireland",
                total_duration=600.0,
            )
        )
        lf_path = output / f"lf_{i + 1:02d}.mp4"
        lf_path.write_bytes(b"x" * (2 * 1024 * 1024))
        lf_paths.append(lf_path)

    drive_plan = DrivePlan(
        source_files=[],
        processed_files=[],
        regular_lfs=regular_lfs,
        score_threshold=0.5,
        total_source_duration=600.0 * num_lfs,
        total_highlight_duration=600.0 * num_lfs,
    )

    from _state_helpers import make_state, verified_shorts_per_lf

    # A publish run posts only shorts whose own clip GPS verified their place.
    shorts_per_lf = verified_shorts_per_lf(output / "_stages", {i + 1: 2 for i in range(num_lfs)})

    return make_state(
        source_dir=source,
        output_base=output,
        region="ireland",
        publish_day="wed",
        location="Cork, Ireland",
        skip_upload=skip_upload,
        skip_csv=False,
        youtube_upload=youtube_upload,
        drive_plan=drive_plan,
        regular_lf_paths=lf_paths,
        shorts_per_lf=shorts_per_lf,
        # Phase BC invariant ``_assert_per_lf_recording_date_complete``
        # populates this map in production. Test fixtures bypass Phase BC,
        # so we thread the data the contract assertion in Phase D's
        # ``_build_recording_dates_by_short_index`` expects.
        per_lf_recording_date={i + 1: "2025-09-14" for i in range(num_lfs)},
        # BC→D boundary contract — every universal per_lf_* map must
        # cover every rendered LF. Fixtures bypass Phase BC so we populate
        # the maps directly with reasonable defaults.
        per_lf_ost_text={i + 1: f"OST {i + 1}" for i in range(num_lfs)},
        per_lf_titles={i + 1: f"Title {i + 1}" for i in range(num_lfs)},
        per_lf_locations={i + 1: "Cork, Ireland" for i in range(num_lfs)},
        per_lf_trim_ratio={i + 1: 0.75 for i in range(num_lfs)},
    )


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
    (meta_cls, sched_cls, rs_cls, social_cls, upload_cls, manifest_cls, val_cls) = mocks

    # Rule 1: distinct titles per LF. side_effect keyed by project_name.
    meta = MagicMock()

    def _meta_side_effect(**kwargs):
        project_name = kwargs.get("project_name", "lf")
        return {
            "title": f"LF Title {project_name}",
            "description": "LF description",
            "tags": ["driving", "ireland"],
            "category": "Travel",
        }

    meta.generate.side_effect = _meta_side_effect
    meta_cls.return_value = meta

    # Rule 8: lf_slot dates must land on Wednesdays. 2026-04-01 is Wed;
    # every 7 days thereafter.
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

    val = MagicMock()
    val.validate.return_value = MagicMock(passed=True, summary=lambda: "ok")
    val_cls.return_value = val


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
        _setup_mocks((m0, m1, m2, m3, m4, m5, m6), num_lfs)
        PhaseD(get_config()).run(state)


# ─────────────────────────────────────────────────────────────────────────
# Per-LF Metricool CSV emission is RETIRED.
# Per-LF ``<project>_lfNN_metricool.csv`` files were never imported by
# Metricool, never used by anyone, and contributed to operator confusion.
# The consolidated shorts CSV is the only
# Metricool deliverable. These tests now assert the inverse contract: no
# per-LF CSVs land on disk, and the consolidated CSV still ships.
# ─────────────────────────────────────────────────────────────────────────


class TestPhaseDPerLFMetricool:
    def test_phase_d_emits_no_per_lf_metricool_csv(self, tmp_path, monkeypatch):
        """Phase D must NOT write any ``<project>_lfNN_metricool.csv`` files."""
        _install_fake_youtube_uploader(monkeypatch)
        state = _build_state(tmp_path, num_lfs=3)
        _run_phase_d(state, num_lfs=3)

        output_base = state["output_base"]
        project_base = state["source_dir"].stem.replace(" ", "_").lower()

        for idx in (1, 2, 3):
            csv_path = output_base / f"{project_base}_lf{idx:02d}_metricool.csv"
            assert not csv_path.exists(), (
                f"per-LF CSV {csv_path} should not exist (per-LF CSVs are retired)"
            )

        # No glob match either — covers any stale or unexpected lf*_metricool.csv.
        per_lf = list(output_base.glob(f"{project_base}_lf*_metricool.csv"))
        assert per_lf == [], f"unexpected per-LF Metricool CSVs: {per_lf}"

    def test_consolidated_shorts_csv_still_emitted(self, tmp_path, monkeypatch):
        """Phase D must still write the consolidated ``<project>_shorts_metricool.csv``."""
        _install_fake_youtube_uploader(monkeypatch)
        # skip_upload=False so the Drive-upload runs and the shorts get media
        # URLs — the consolidated CSV is only written when each row has a
        # ``Picture Url 1`` (METRICOOL-CSV-NO-MEDIA gate).
        state = _build_state(tmp_path, num_lfs=2, skip_upload=False)
        _run_phase_d(state, num_lfs=2)

        output_base = state["output_base"]
        shorts_csvs = list(output_base.glob("*_shorts_metricool.csv"))
        assert len(shorts_csvs) == 1, (
            f"expected exactly one consolidated shorts CSV, got: {shorts_csvs}"
        )

    def test_consolidated_csv_row_count_matches_dod_expectation(self, tmp_path, monkeypatch):
        """Invariant: consolidated shorts CSV row count == Σ len(shorts_per_lf[i]).

        This pins the source-of-truth contract that ``check_csv_row_count``
        (DoD Rule 5, ``quality_gates.py``) gates on. The failure mode was
        Phase D's social posts loop using ``max(num_shorts, 1)`` which
        synthesized a phantom post for every LF that had 0 shorts in
        ``state["shorts_per_lf"]`` — those phantom posts then became phantom
        rows in ``<project>_shorts_metricool.csv``, breaking the DoD.

        Test setup: 2 LFs with non-empty shorts (target=2 each → 4 shorts
        total) PLUS the assertion that ``num_shorts`` flowing into
        ``SocialGenerator.generate_all`` is exactly ``len(shorts_per_lf[lf])``,
        never floored. The consolidated CSV's row count must equal the sum.

        The per-LF CSVs are retired, which made the consolidated CSV the only Metricool deliverable.
        """
        _install_fake_youtube_uploader(monkeypatch)
        num_lfs = 2
        # skip_upload=False: Drive-upload populates media URLs (gate below).
        state = _build_state(tmp_path, num_lfs=num_lfs, skip_upload=False)

        # Sanity: helper produces 2 shorts per LF — Σ = 4.
        expected_rows = sum(len(v) for v in state["shorts_per_lf"].values())
        assert expected_rows == 4

        captured_num_shorts: list[int] = []

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
            _setup_mocks((m0, m1, m2, m3, m4, m5, m6), num_lfs)

            # Replace social.generate_all with a callable that captures
            # num_shorts and honors it. This guarantees the invariant holds
            # under any num_shorts (including 0).
            social_inst = m3.return_value

            def _capturing_generate_all(num_shorts, *args, index_offset=0, **kwargs):
                captured_num_shorts.append(num_shorts)
                return [
                    MagicMock(
                        short_index=index_offset + i + 1,
                        caption=f"c_{index_offset + i + 1}",
                    )
                    for i in range(num_shorts)
                ]

            social_inst.generate_all.side_effect = _capturing_generate_all

            PhaseD(get_config()).run(state)

        # Invariant 1: num_shorts passed to generate_all equals
        # len(shorts_per_lf[lf]) for every LF — never floored to 1.
        for lf_idx, captured in zip(sorted(state["shorts_per_lf"]), captured_num_shorts):
            assert captured == len(state["shorts_per_lf"][lf_idx]), (
                f"LF {lf_idx}: SocialGenerator.generate_all called with "
                f"num_shorts={captured}, expected "
                f"{len(state['shorts_per_lf'][lf_idx])} "
                f"(no max(...,1) flooring allowed after per-LF CSV retirement)"
            )

        # Invariant 2: consolidated CSV row count == Σ len(shorts_per_lf[i]).
        output_base = state["output_base"]
        csvs = list(output_base.glob("*_shorts_metricool.csv"))
        assert len(csvs) == 1, csvs
        with open(csvs[0], encoding="utf-8") as f:
            reader = csvmod.reader(f)
            rows = list(reader)
        actual_rows = len(rows) - 1  # drop header
        assert actual_rows == expected_rows, (
            f"consolidated CSV row count {actual_rows} != "
            f"Σ len(shorts_per_lf) {expected_rows} (DoD Rule 5)"
        )

    def test_consolidated_csv_zero_shorts_lf_emits_zero_rows(self, tmp_path, monkeypatch):
        """Regression guard: an LF with 0 shorts contributes 0 CSV rows.

        Pre-fix, Phase D applied ``max(num_shorts, 1)`` so every LF (even
        ones with zero shorts) generated 1 phantom social post → 1 phantom
        consolidated CSV row. This test creates 2 LFs where LF 1 has 0
        shorts and LF 2 has 2 shorts; the CSV must contain exactly 2 rows,
        not 3.
        """
        _install_fake_youtube_uploader(monkeypatch)
        # skip_upload=False: Drive-upload populates media URLs (gate below).
        state = _build_state(tmp_path, num_lfs=2, skip_upload=False)
        # LF 1: zero shorts. LF 2: keep the helper's 2 shorts.
        state["shorts_per_lf"][1] = []
        expected_rows = sum(len(v) for v in state["shorts_per_lf"].values())
        assert expected_rows == 2

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
            _setup_mocks((m0, m1, m2, m3, m4, m5, m6), 2)
            # Override social mock to honor num_shorts (so 0 -> []).
            social_inst = m3.return_value

            def _honest_generate_all(num_shorts, *args, index_offset=0, **kwargs):
                return [
                    MagicMock(
                        short_index=index_offset + i + 1,
                        caption=f"c_{index_offset + i + 1}",
                    )
                    for i in range(num_shorts)
                ]

            social_inst.generate_all.side_effect = _honest_generate_all

            PhaseD(get_config()).run(state)

        output_base = state["output_base"]
        csvs = list(output_base.glob("*_shorts_metricool.csv"))
        assert len(csvs) == 1
        with open(csvs[0], encoding="utf-8") as f:
            rows = list(csvmod.reader(f))
        actual = len(rows) - 1
        assert actual == expected_rows, (
            f"0-shorts LF emitted phantom rows: got {actual}, expected {expected_rows}"
        )

    def test_empty_drive_plan_no_per_lf_csv(self, tmp_path, monkeypatch):
        """With no LFs Phase D returns successfully with no per-LF CSVs."""
        _install_fake_youtube_uploader(monkeypatch)
        state = _build_state(tmp_path, num_lfs=1)
        # Force empty drive plan, mirroring the benign short-circuit path.
        state["drive_plan"] = DrivePlan(
            source_files=[],
            processed_files=[],
            regular_lfs=[],
            score_threshold=0.5,
            total_source_duration=0.0,
            total_highlight_duration=0.0,
        )
        state["regular_lf_paths"] = []
        state["shorts_per_lf"] = {}

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
            _setup_mocks((m0, m1, m2, m3, m4, m5, m6), 1)
            PhaseD(get_config()).run(state)

        output_base = state["output_base"]
        assert list(output_base.glob("*_metricool.csv")) == []


# ─────────────────────────────────────────────────────────────────────────
# Thumbnail generation + layout
# ─────────────────────────────────────────────────────────────────────────


class TestPhaseDThumbnailsDecoupled:
    def _fake_generator_factory(self, tmp_root: Path):
        """Build a fake generator that mimics the real one's subdir layout.

        The fake writes ``<output_dir>/<lf_name>_frame1_bold.jpg`` and returns
        its path, so the Phase D canonical-file step has a real source file
        to copy.
        """

        def _fake(video_path, title, subtitle, output_dir, duration):
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            variant = output_dir / "frame1_bold.jpg"
            variant.write_bytes(b"fake-thumb-bytes" * 16)
            return variant

        return _fake

    def test_phase_d_generates_thumbnails_with_upload_disabled(self, tmp_path, monkeypatch):
        """The *primary* bug: thumbnails land on disk even when upload is off."""
        _install_fake_youtube_uploader(monkeypatch)
        state = _build_state(tmp_path, num_lfs=3, youtube_upload=False, skip_upload=True)
        fake = self._fake_generator_factory(tmp_path)
        with patch(
            "gomoveshift.pipeline.drive_phases.thumbnail_generator.generate_lf_thumbnail",
            side_effect=fake,
        ) as mock_gen:
            _run_phase_d(state, num_lfs=3)

        assert mock_gen.call_count == 3

        thumbs_dir = state["output_base"] / "thumbnails"
        for idx in (1, 2, 3):
            canonical = thumbs_dir / f"lf_{idx:02d}.jpg"
            assert canonical.exists(), f"missing canonical thumbnail {canonical}"
            assert canonical.is_file()
            assert canonical.stat().st_size > 0

    def test_phase_d_generates_thumbnails_with_upload_enabled(self, tmp_path, monkeypatch):
        """With upload on: thumbnails still written AND set_thumbnail attempted."""
        yt = _install_fake_youtube_uploader(monkeypatch)
        state = _build_state(tmp_path, num_lfs=2, youtube_upload=True, skip_upload=False)
        fake = self._fake_generator_factory(tmp_path)
        with patch(
            "gomoveshift.pipeline.drive_phases.thumbnail_generator.generate_lf_thumbnail",
            side_effect=fake,
        ):
            _run_phase_d(state, num_lfs=2)

        thumbs_dir = state["output_base"] / "thumbnails"
        for idx in (1, 2):
            assert (thumbs_dir / f"lf_{idx:02d}.jpg").exists()

        # Attach path must have been attempted for every uploaded LF.
        assert yt.set_thumbnail.call_count == 2

    def test_stale_thumbnails_replaced(self, tmp_path, monkeypatch):
        """A prior run's stale ``lf_NN/`` and top-level ``lf_NN.jpg`` are wiped."""
        _install_fake_youtube_uploader(monkeypatch)
        state = _build_state(tmp_path, num_lfs=1, youtube_upload=False, skip_upload=True)

        thumbs_dir = state["output_base"] / "thumbnails"
        stale_subdir = thumbs_dir / "lf_01"
        stale_subdir.mkdir(parents=True)
        stale_file = stale_subdir / "stale.jpg"
        stale_file.write_bytes(b"stale")
        stale_canonical = thumbs_dir / "lf_01.jpg"
        stale_canonical.write_bytes(b"old-canonical-bytes")

        old_mtime = time.time() - 3600
        import os

        os.utime(stale_file, (old_mtime, old_mtime))
        os.utime(stale_canonical, (old_mtime, old_mtime))

        fake = self._fake_generator_factory(tmp_path)
        with patch(
            "gomoveshift.pipeline.drive_phases.thumbnail_generator.generate_lf_thumbnail",
            side_effect=fake,
        ):
            _run_phase_d(state, num_lfs=1)

        fresh_canonical = thumbs_dir / "lf_01.jpg"
        assert fresh_canonical.exists()
        assert fresh_canonical.stat().st_mtime > old_mtime + 1, (
            "canonical thumbnail was not replaced"
        )
        # The stale-named file from the prior subdir must not linger unchanged.
        assert not stale_file.exists(), "prior-run stale file should be removed before regeneration"

    def test_validator_check_09_passes_after_thumbnail_fix(self, tmp_path, monkeypatch):
        """End-to-end: validate_pipeline_output's thumbnail check passes."""
        _install_fake_youtube_uploader(monkeypatch)
        state = _build_state(tmp_path, num_lfs=2, youtube_upload=False, skip_upload=True)
        fake = self._fake_generator_factory(tmp_path)
        with patch(
            "gomoveshift.pipeline.drive_phases.thumbnail_generator.generate_lf_thumbnail",
            side_effect=fake,
        ):
            _run_phase_d(state, num_lfs=2)

        # Import the validator check and run it against the output tree.
        repo_root = Path(__file__).resolve().parent.parent
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "validate_pipeline_output",
            repo_root / "tools" / "validate_pipeline_output.py",
        )
        mod = importlib.util.module_from_spec(spec)
        # Register in sys.modules BEFORE exec so the dataclass's
        # cls.__module__ lookup in the std-lib dataclasses module can find it.
        sys.modules["validate_pipeline_output"] = mod
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.modules.pop("validate_pipeline_output", None)

        project_dir = state["output_base"]
        lfs = [(1, project_dir / "lf_01.mp4"), (2, project_dir / "lf_02.mp4")]
        result = mod.check_thumbnails(project_dir, lfs)
        assert result.passed, f"validator check failed: {result.detail}"


# ─────────────────────────────────────────────────────────────────────────
# str-vs-int key drift on youtube_uploads
# ─────────────────────────────────────────────────────────────────────────


class TestYouTubeUploadsIntKeysHonored:
    def test_int_keyed_existing_uploads_skip_reupload(self, tmp_path, monkeypatch):
        """Pre-seeded int-keyed uploads survive Phase D without re-upload.

        Contract lives upstream: state load normalises
        ``youtube_uploads`` to int keys via
        ``PipelineState.from_dict``. Phase D no longer carries an
        in-phase str→int fallback because the load-time invariant
        makes the failure unrepresentable. This test pins the
        downstream behaviour — int keys flow through unchanged and
        the prior video_ids survive.
        """
        yt = _install_fake_youtube_uploader(monkeypatch)
        state = _build_state(tmp_path, num_lfs=2, youtube_upload=True, skip_upload=False)
        state["youtube_uploads"] = {
            1: {
                "video_id": "vid_A",
                "url": "https://youtu.be/vid_A",
                "title": "LF Title",
                "uploaded_at": "2026-01-01T00:00:00+00:00",
            },
            2: {
                "video_id": "vid_B",
                "url": "https://youtu.be/vid_B",
                "title": "LF Title",
                "uploaded_at": "2026-01-01T00:00:00+00:00",
            },
        }

        def fake(video_path, title, subtitle, output_dir, duration):
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            variant = Path(output_dir) / "f1_bold.jpg"
            variant.write_bytes(b"x" * 64)
            return variant

        with patch(
            "gomoveshift.pipeline.drive_phases.thumbnail_generator.generate_lf_thumbnail",
            side_effect=fake,
        ):
            _run_phase_d(state, num_lfs=2)

        assert set(state["youtube_uploads"].keys()) == {1, 2}
        assert state["youtube_uploads"][1]["video_id"] == "vid_A"
        assert state["youtube_uploads"][2]["video_id"] == "vid_B"
        assert yt.set_thumbnail.call_count == 2


# ─────────────────────────────────────────────────────────────────────────
# Thumbnail reads
# state["per_lf_ost_text"] (canonical, pre-dedup), not the dedup-suffixed
# ``meta["title"]``. Pins the title/thumbnail alignment rule: OST,
# YouTube title, and thumbnail all derive from the SAME source string.
#
# On a multi-LF day, ``ensure_unique_titles`` decorates collisions with
# ``" (Part 2 of 4)"``. That suffix must appear on the YouTube title only;
# the OST burn and thumbnail pixels stay canonical.
# ─────────────────────────────────────────────────────────────────────────


class TestThumbnailReadsPerLFOSTText:
    """Thumbnail must receive the canonical ``state["per_lf_ost_text"]`` value
    — never the dedup-suffixed ``meta["title"]``. Fallback priority:
      1. ``state["per_lf_ost_text"][lf.index]`` (canonical, pre-dedup)
      2. ``meta["title"]`` (dedup-suffixed, legacy-state backcompat)
      3. neither present → raise (no ``f"Drive LF {n}"`` placeholder is
         burned onto a public thumbnail).
    """

    def _fake_generator_factory(self, captured_titles: dict[int, str]):
        """Capture the ``title`` kwarg passed per LF call."""

        def _fake(video_path, title, subtitle, output_dir, duration):
            # Recover the LF index from the output_dir name ``lf_NN``.
            name = Path(output_dir).name
            # name is like "lf_02"; grab the numeric suffix.
            lf_idx = int(name.split("_", 1)[1])
            captured_titles[lf_idx] = title
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            variant = Path(output_dir) / "frame1_bold.jpg"
            variant.write_bytes(b"fake-thumb-bytes" * 16)
            return variant

        return _fake

    def test_thumbnail_reads_per_lf_ost_text_not_dedup_suffixed_title(self, tmp_path, monkeypatch):
        """When per_lf_ost_text differs from per_lf_titles (dedup-suffixed),
        thumbnail must receive the canonical pre-dedup text.

        Real-world trigger: a single calendar day produces 4 LFs that share
        the same base route string. ``ensure_unique_titles`` adds
        ``" (Part N of 4)"`` to per_lf_titles; state["per_lf_ost_text"]
        stays pristine. This test pins that divergence so the thumbnail
        reads the canonical text, not the suffix.
        """
        _install_fake_youtube_uploader(monkeypatch)
        state = _build_state(tmp_path, num_lfs=2, youtube_upload=False, skip_upload=True)

        # Canonical per-LF route text (what OST burn uses — single source of truth).
        canonical = "Longwood to Fort Fremont via Seaside Road, SC"
        state["per_lf_ost_text"] = {1: canonical, 2: canonical}

        captured: dict[int, str] = {}
        fake = self._fake_generator_factory(captured)
        with patch(
            "gomoveshift.pipeline.drive_phases.thumbnail_generator.generate_lf_thumbnail",
            side_effect=fake,
        ):
            _run_phase_d(state, num_lfs=2)

        # Thumbnail generator was called per LF with the canonical text —
        # NOT the dedup-suffixed meta["title"] that ``_setup_mocks`` stubs
        # to ``"LF Title source_drive_lf0N"``.
        assert captured[1] == canonical, (
            f"LF1 thumbnail should use per_lf_ost_text canonical text, got: {captured[1]!r}"
        )
        assert captured[2] == canonical, (
            f"LF2 thumbnail should use per_lf_ost_text canonical text, got: {captured[2]!r}"
        )
        # Sanity: the mocked meta title is clearly not the canonical text —
        # if the bug were still present, captured would equal the meta title.
        assert canonical not in "LF Title source_drive_lf01", (
            "precondition — canonical must differ from mocked meta title"
        )

    def test_thumbnail_falls_back_to_title_when_ost_text_absent(self, tmp_path, monkeypatch):
        """Boundary contract: a legacy state file with per_lf_ost_text entirely
        absent (or empty) is tolerated by the D-entry aggregator
        (non-strict — empty mappings allowed for legacy / degenerate
        paths), but partial coverage would raise. The defensive thumbnail
        fallback chain therefore still serves as the in-LF safety net
        when per_lf_ost_text is empty and the metadata title is the only
        signal available.
        """
        _install_fake_youtube_uploader(monkeypatch)
        state = _build_state(tmp_path, num_lfs=1, youtube_upload=False, skip_upload=True)
        # Empty per_lf_ost_text — tolerated by D-entry contract (non-strict).
        state["per_lf_ost_text"] = {}

        captured: dict[int, str] = {}
        fake = self._fake_generator_factory(captured)
        with patch(
            "gomoveshift.pipeline.drive_phases.thumbnail_generator.generate_lf_thumbnail",
            side_effect=fake,
        ):
            _run_phase_d(state, num_lfs=1)

        # Meta title from _setup_mocks — per_lf_ost_text empty so we fall
        # back to whatever metadata_per_lf[lf.index]["title"] holds.
        assert captured[1] == "LF Title source_drive_lf01", (
            f"LF1 thumbnail should fall back to meta title when per_lf_ost_text empty, "
            f"got: {captured[1]!r}"
        )

    def test_thumbnail_raises_when_both_absent(self, tmp_path, monkeypatch):
        """When a per_lf_ost_text entry is empty AND meta["title"] is absent,
        thumbnail generation RAISES rather than burning a ``f"Drive LF {n}"``
        placeholder onto a public thumbnail — no gate ever reviews thumbnail
        text. The burned-pixels twin of the upload-placeholder guard.

        The universal-field absence case is caught at D-entry; here we
        keep per_lf_ost_text in state (an empty string value) so the in-LF
        chain is exercised with a degenerate entry and must refuse.
        """
        _install_fake_youtube_uploader(monkeypatch)
        state = _build_state(tmp_path, num_lfs=1, youtube_upload=False, skip_upload=True)
        # Degrade the LF's OST text to empty — the entry is present (so the
        # D-entry contract passes) but the per-LF fallback chain must trip.
        state["per_lf_ost_text"] = {1: ""}

        # Override the meta-generator stub so ``metadata_per_lf[lf.index]``
        # carries no ``title`` key at all.
        from unittest.mock import MagicMock

        def _meta_side_effect_no_title(**kwargs):
            return {
                # no "title" key
                "description": "LF description",
                "tags": ["driving", "ireland"],
                "category": "Travel",
            }

        patches = _phase_d_patches()
        captured: dict[int, str] = {}
        fake = self._fake_generator_factory(captured)

        with (
            patches[0] as m0,
            patches[1] as m1,
            patches[2] as m2,
            patches[3] as m3,
            patches[4] as m4,
            patches[5] as m5,
            patches[6] as m6,
            patch(
                "gomoveshift.pipeline.drive_phases.thumbnail_generator.generate_lf_thumbnail",
                side_effect=fake,
            ),
        ):
            _setup_mocks((m0, m1, m2, m3, m4, m5, m6), 1)
            # Override the meta side_effect after _setup_mocks ran.
            meta = MagicMock()
            meta.generate.side_effect = _meta_side_effect_no_title
            m0.return_value = meta
            with pytest.raises(ValueError, match="no thumbnail title"):
                PhaseD(get_config()).run(state)

        assert 1 not in captured, (
            "thumbnail generation must refuse before any placeholder title "
            f"reaches the generator, got: {captured!r}"
        )


# ─────────────────────────────────────────────────────────────────────────
# Attach failure is recorded on the upload entry
# ─────────────────────────────────────────────────────────────────────────


class TestThumbnailAttachFailureMarker:
    def test_attach_failure_records_marker_on_upload_entry(self, tmp_path, monkeypatch):
        yt = _install_fake_youtube_uploader(monkeypatch)
        yt.set_thumbnail.side_effect = RuntimeError("quota exhausted")

        state = _build_state(tmp_path, num_lfs=1, youtube_upload=True, skip_upload=False)

        def fake(video_path, title, subtitle, output_dir, duration):
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            variant = Path(output_dir) / "f1_bold.jpg"
            variant.write_bytes(b"x" * 64)
            return variant

        with patch(
            "gomoveshift.pipeline.drive_phases.thumbnail_generator.generate_lf_thumbnail",
            side_effect=fake,
        ):
            _run_phase_d(state, num_lfs=1)

        entry = state["youtube_uploads"][1]
        assert entry.get("thumbnail_attach_failed"), (
            "upload entry should carry a marker recording the attach failure"
        )
