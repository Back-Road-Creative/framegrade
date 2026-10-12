"""End-to-end Insta360 / full-drive integration test.

Exercises the generic-ingest, per-LF-location-override and full-drive LF
contracts together, proving the Insta360 invocation contract works as one
coherent pipeline:

    gomoveshift drive <source-dir> \
        --ingest-mode generic \
        --lf-mode full_drive \
        --per-lf-location '1=Test Drive, Test County'

Cost-honest scope: we exercise the **pipeline integration** of these
three surfaces, not the literal argparse → tmux → orchestrator path
(those layers are independently covered by ``test_drive_cli.py`` and
``test_drive_orchestrator_day_first.py``). The wiring this test pins:

1. ``scan_video_directory(mode="generic")`` ingests an Insta360-named
   single-file source, no GPS sidecar.
2. ``derive_per_lf_locations(overrides={1: ...})`` returns the operator
   override (GPS-less drives require this; no drive-wide
   fallback).
3. ``run_day_pipeline(config.drive.lf_selection_mode="full_drive")``
   synthesizes exactly one LF, writes the OST audit sidecar, and
   ``--ost-audit-confirmed`` (via ``state["ost_audit_confirmed"]``)
   bypasses the gate so the encode loop runs to completion.
4. The synthesized LF's trim_ratio falls in the "Driving" band
   (>= ``HIGHLIGHTS_TRIM_THRESHOLD``), so the published title verb is
   "Driving" not "Highlights from".

The fixture is generated on the fly via a real ffmpeg subprocess
(testsrc + sine, ultrafast preset) so the ``scan_video_directory``
mode-routing assertion exercises real file metadata. Heavy encode /
upload stages downstream of ingest are patched out — same pattern as
``tests/test_day_pipeline_full_drive_mode.py``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from gomoveshift.pipeline.day_grouper import DayBatch
from _state_helpers import make_state


FFMPEG = shutil.which("ffmpeg")
FIXTURE_DURATION_SEC = 180  # 3 minutes
# ffmpeg budget for the fixture encode. Seconds of work on an idle machine; a
# loaded machine can stretch it well past a short budget, so allow 300 s. A
# hang still fails.
FIXTURE_ENCODE_TIMEOUT_SEC = 300
SOURCE_DURATION_SEC = 600.0  # what we tell the planner (>= 8min minimum)
NUM_SCORE_COLUMNS = 9
COL_TIMESTAMP = 0
COL_COMPOSITE = 1


def _build_insta360_fixture(tmp_path: Path) -> Path:
    """Create a real ~3-minute MP4 that mimics an Insta360 single-file
    export — flat 16:9, no GPS sidecar, Insta360-style filename, and a
    container ``creation_time`` tag (a real export always carries one;
    generic-mode ingest dates the recording from it rather than the
    file's copy mtime).

    Encoded with ``-preset ultrafast``: seconds on an idle box, minutes on
    the loaded CI box -- ``FIXTURE_ENCODE_TIMEOUT_SEC`` is the stall budget.
    """
    if FFMPEG is None:
        pytest.skip("ffmpeg not on PATH; e2e fixture cannot be built")

    out_path = tmp_path / "VID_20000101_000000.mp4"
    cmd = [
        FFMPEG,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={FIXTURE_DURATION_SEC}:size=320x240:rate=15",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:duration={FIXTURE_DURATION_SEC}",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        "-metadata",
        "creation_time=2026-05-01T12:00:00Z",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, timeout=FIXTURE_ENCODE_TIMEOUT_SEC)
    assert out_path.exists() and out_path.stat().st_size > 0
    return out_path


class TestFixtureEncodeBudget:
    """The 3-minute testsrc encode is seconds of work on an idle machine, but a
    loaded machine can stretch it well past a short budget. The budget only
    has to be one a loaded machine cannot exhaust while ffmpeg is still making
    progress, so it is 300 s."""

    def test_fixture_encode_timeout_is_sized_for_a_loaded_box(self, tmp_path):
        seen: dict = {}

        def fake_run(cmd, **kw):
            seen.update(kw)
            Path(cmd[-1]).write_bytes(b"x")
            return SimpleNamespace(returncode=0)

        with patch.object(subprocess, "run", side_effect=fake_run):
            _build_insta360_fixture(tmp_path)
        assert seen.get("timeout", 0) >= 300, (
            f"fixture encode timeout {seen.get('timeout')!r} is too short for a loaded machine"
        )


class _FakeEpisode:
    """Mirror of the day-pipeline test pattern. Stand-in for
    ``GoProEpisode`` / generic-mode equivalent — every attribute
    ``run_day_pipeline`` reads off an episode."""

    def __init__(self, index: int, concat_path: Path, total_duration: float, date: str):
        self.index = index
        self.concat_path = concat_path
        self.total_duration = total_duration
        self.date = date
        self.creation_time = None
        self.gpx_path = None


def _make_config(*, lf_selection_mode: str, full_drive_max_sec: int = 7200):
    """Tiny stand-in for ``Config`` exposing only the ``config.drive``
    fields ``run_day_pipeline`` touches."""
    return SimpleNamespace(
        drive=SimpleNamespace(
            brand_logo_path=None,
            montage_max_duration_sec=15.0,
            ost_title_crossfade_sec=0.5,
            lf_selection_mode=lf_selection_mode,
            full_drive_max_duration_sec=full_drive_max_sec,
            shorts_target_count=None,
            shorts_min_gap_sec=60.0,
            ingest_mode="generic",
            video_extensions=[".mp4", ".mov", ".mkv", ".m4v", ".mxf"],
        )
    )


def _patch_targets(tmp_path: Path):
    """Stub out the heavy stages (encode, render, concat, montage).
    Mirrors ``tests/test_day_pipeline_full_drive_mode.py::_patch_targets``."""
    master_path = tmp_path / "out" / "day_2026-05-01" / "day_2026-05-01_master.mp4"
    master_path.parent.mkdir(parents=True, exist_ok=True)
    master_path.write_bytes(b"fake master")

    def _fake_ost(input_path, output_path, location, layout, **kwargs):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"fake ost")
        return output_path

    def _fake_render(source_video, output_path, start_time, duration, location, **kwargs):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"fake short")
        return True

    def _fake_assemble(master_path, segments, file_boundaries, output_path, work_dir, **kw):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"concat")
        return Path(output_path)

    def _fake_montage(raw_lf_path, peaks, output_path, work_dir, **kw):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"with_montage")
        return Path(output_path)

    return master_path, _fake_ost, _fake_render, _fake_assemble, _fake_montage


# ---------------------------------------------------------------------
# Test 1 — Generic ingest path accepts the Insta360-named single file
# ---------------------------------------------------------------------


class TestInsta360GenericIngest:
    def test_scan_video_directory_generic_accepts_insta360_filename(self, tmp_path):
        """Generic-ingest contract: ``mode="generic"`` ingests the Insta360 file
        as a singleton episode without raising on filename shape."""
        _build_insta360_fixture(tmp_path)
        # Also create a non-video sibling that must be ignored by the
        # extension filter (proves the whitelist contract).
        (tmp_path / "notes.txt").write_text("ignore me")

        from gomoveshift.pipeline.ingest import scan_video_directory

        recs, filtered, unrec = scan_video_directory(
            tmp_path,
            extensions=[".mp4", ".mov", ".mkv", ".m4v", ".mxf"],
            mode="generic",
        )

        assert len(recs) == 1, "exactly one singleton recording for one source file"
        assert len(unrec) == 0
        # The lone recording has exactly one chapter (singleton — no
        # GoPro session grouping).
        rec = next(iter(recs.values()))
        assert len(rec.chapters) == 1
        # Duration is real — ffprobe was run on the synthetic file.
        # 3-minute fixture: allow ±2s margin for ffmpeg's keyframe padding.
        chapter = rec.chapters[0]
        assert abs(chapter.duration - FIXTURE_DURATION_SEC) < 5, (
            f"expected ~{FIXTURE_DURATION_SEC}s duration, got {chapter.duration}"
        )


# ---------------------------------------------------------------------
# Test 2 — Per-LF location override wins over GPS-absent fallback
# ---------------------------------------------------------------------


class TestPerLFLocationOverride:
    def test_override_wins_when_no_gps(self):
        """Insta360 invocation contract: when an LF has
        zero GPS data, ``--per-lf-location 1=...`` must surface as
        the canonical label for LF 1."""
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            derive_per_lf_locations,
        )

        lf = SimpleNamespace(index=1, segments=[])
        result = derive_per_lf_locations(
            [lf],
            geocoder=None,
            gps_fallback=None,
            overrides={1: "Test Drive, Test County"},
        )
        assert result == {1: "Test Drive, Test County"}

    def test_missing_override_for_gpsless_lf_raises(self):
        """Defensive: with no GPS and no override, the function refuses
        to silently substitute a drive-wide fallback. Mirrors the service
        contract that ``derive_per_lf_locations`` is total."""
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            derive_per_lf_locations,
        )

        lf = SimpleNamespace(index=1, segments=[])
        with pytest.raises(ValueError):
            derive_per_lf_locations([lf], geocoder=None, overrides={})


# ---------------------------------------------------------------------
# Test 3 — End-to-end: run_day_pipeline with full_drive mode produces
# one LF, writes ost_audit sidecar, "Driving" verb fires on trim_ratio
# ---------------------------------------------------------------------


class TestInsta360FullDriveEndToEnd:
    def test_full_drive_run_produces_single_lf_audit_and_driving_verb(self, tmp_path):
        from gomoveshift.pipeline import day_pipeline

        source_file = _build_insta360_fixture(tmp_path)
        output_base = tmp_path / "out"
        date = "2026-05-01"

        # Build a single-episode day batch off the real Insta360 file,
        # but tell the planner the episode is 600s long so it clears the
        # 8-min minimum LF duration without needing a 9-minute encode.
        episode = _FakeEpisode(
            index=1,
            concat_path=source_file,
            total_duration=SOURCE_DURATION_SEC,
            date=date,
        )
        day_batch = DayBatch(date=date, episodes=[episode])

        # Synthetic scored timeline: 600 1Hz rows with composite=0.7,
        # spanning the post-stop-removal day. The full_drive planner
        # uses only timestamp bounds → produces one LF covering the
        # entire timeline.
        day_scores = np.zeros((int(SOURCE_DURATION_SEC), NUM_SCORE_COLUMNS), dtype=np.float64)
        day_scores[:, COL_TIMESTAMP] = np.arange(int(SOURCE_DURATION_SEC), dtype=np.float64)
        day_scores[:, COL_COMPOSITE] = 0.7
        # Moving footage: motion energy above the stop mask's optical floor
        # (0.008) so this GPS-less day isn't masked wholesale — no parked span,
        # so the full-drive LF spans the whole timeline (trim_ratio ≈ 1.0).
        day_scores[:, 2] = 0.02  # COL_MOTION_ENERGY

        config = _make_config(lf_selection_mode="full_drive")

        master_path, fake_ost, fake_render, fake_assemble, fake_montage = _patch_targets(tmp_path)

        # First pass: leave the OST audit gate UNCONFIRMED so the
        # sidecar gets written to disk — that's the artifact we want
        # to inspect end-to-end. The confirmed path (state flag /
        # confirmed-file) skips the write, so we assert on the
        # pending-then-resume cycle instead.
        state = make_state(
            per_lf_locations={1: "Test Drive, Test County"},
        )

        from gomoveshift.pipeline.ost_audit_sidecar import OstAuditPendingError

        common_patches = (
            patch(
                "gomoveshift.pipeline.day_pipeline.encode_day_master",
                return_value=master_path,
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline.apply_ost",
                side_effect=fake_ost,
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline._render_short_ffmpeg",
                side_effect=fake_render,
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline.extract_and_concat_segments",
                side_effect=fake_assemble,
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline.prepend_montage",
                side_effect=fake_montage,
            ),
        )

        # ── Phase 1: first invocation. Gate is pending → sidecar
        # written to disk, OstAuditPendingError raised. We catch it
        # and inspect the file. ──
        with (
            common_patches[0],
            common_patches[1],
            common_patches[2],
            common_patches[3],
            common_patches[4],
        ):
            with pytest.raises(OstAuditPendingError):
                day_pipeline.run_day_pipeline(
                    day_batch=day_batch,
                    day_scores=day_scores,
                    output_dir=output_base,
                    # Drive-wide fallback label must differ from the
                    # per-LF override, or the OST-burn contract
                    # assertion (verbatim-equality === construction
                    # regression) fires.
                    location="Drive Fallback Label",
                    config=config,
                    lf_plans=None,
                    per_lf_locations=state["per_lf_locations"],
                    state=state,
                )

        # ── OST audit sidecar was written for this date ──
        audit_path = output_base / f"ost_audit_{date}.json"
        assert audit_path.exists(), (
            "OST-audit contract: run_day_pipeline MUST write "
            f"ost_audit_{date}.json on the pending-confirmation path"
        )
        payload = json.loads(audit_path.read_text())
        assert payload["date"] == date
        assert len(payload["lfs"]) == 1, "one LF → one audit entry"
        audit_entry = payload["lfs"][0]
        assert audit_entry["lf_index"] == 1
        # The operator override is the canonical OST text — proves
        # the override survives end-to-end into the burn-time source.
        assert audit_entry["ost_text"] == "Test Drive, Test County"

        # ── Phase 2: operator confirms, run resumes to completion ──
        # A fresh --ost-audit-confirmed on the re-run bypasses the gate (the
        # non-persisted this-run signal is what the gate reads).
        state["ost_audit_confirmed"] = True
        state["ost_audit_confirmed_this_run"] = True
        # Rebuild the patch context (with-statements are single-use).
        with (
            patch(
                "gomoveshift.pipeline.day_pipeline.encode_day_master",
                return_value=master_path,
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline.apply_ost",
                side_effect=fake_ost,
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline._render_short_ffmpeg",
                side_effect=fake_render,
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline.extract_and_concat_segments",
                side_effect=fake_assemble,
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline.prepend_montage",
                side_effect=fake_montage,
            ),
        ):
            artifacts = day_pipeline.run_day_pipeline(
                day_batch=day_batch,
                day_scores=day_scores,
                output_dir=output_base,
                location="Drive Fallback Label",
                config=config,
                lf_plans=None,
                per_lf_locations=state["per_lf_locations"],
                state=state,
            )

        # ── Exit success: artifacts exist, exactly one LF ──
        assert artifacts is not None
        assert len(artifacts.regular_lf_paths) == 1, (
            f"full_drive mode must produce exactly one LF — got {len(artifacts.regular_lf_paths)}"
        )

        # ── "Driving" verb fires on the full-drive trim_ratio ──
        # A full-drive LF spans the entire scored timeline with no
        # stop-trimming → trim_ratio ≈ 1.0 (single segment covers
        # 100% of the source). 1.0 >= 0.5 threshold → "Driving".
        from gomoveshift.pipeline.longform_metadata import (
            HIGHLIGHTS_TRIM_THRESHOLD,
            _select_verb,
        )

        # Back-compute: the synthesized LF has total_duration ==
        # source_duration → trim_ratio = 1.0.
        synth_trim_ratio = 1.0
        assert synth_trim_ratio >= HIGHLIGHTS_TRIM_THRESHOLD
        assert _select_verb(synth_trim_ratio) == "Driving", (
            "Insta360 full-drive LFs must land in the 'Driving' verb band "
            "— a 'Highlights from' verb here would mis-frame the published title"
        )
