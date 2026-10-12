"""Existing gates must pass for a single-LF full-drive shape.

The Insta360 / full-drive ingest mode (`--ingest-mode generic
--lf-mode full_drive`) produces exactly ONE regular LF per drive
(the entire source becomes the LF, no segment selection). All
per-LF gates were authored with multi-LF batch processing in
mind. This module pins that none of them short-circuit, off-by-one,
or otherwise misbehave when `len(drive_plan.regular_lfs) == 1`.

Gate paths covered (none assumes multiple LFs):

* ``ost_audit_sidecar.format_ost_audit`` / ``compute_ost_audit_hash`` /
  ``write_ost_audit`` — iterate ``lf_indices``, no count branching.
* ``youtube_sidecar.assert_sidecar_integrity`` — set comparison, no
  count branching beyond empty-sidecar fast-path.
* ``quality_gates.assert_phase_d_quality`` — accumulates rule outputs
  across LFs; rule 7 (slot count) accepts any count == plan count.
* ``phase_d.assert_audio_loudness_hard_fail`` — list-driven; raises
  iff list non-empty.
* ``validation.shorts_validator.ShortsValidator.validate_short_files``
  — file-list driven; shorts are keyed by short_index globally, not
  by per-LF index, so a single-parent-LF case still validates N shorts.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gomoveshift.models import DrivePlan, LFSegment, RegularLFPlan
from gomoveshift.pipeline.csv_generator import GOLDEN_COLUMNS
from gomoveshift.pipeline.drive_phases.phase_d import (
    assert_audio_loudness_hard_fail,
)
from gomoveshift.pipeline.exceptions import (
    PhaseQualityError,
    SidecarIntegrityError,
)
from gomoveshift.pipeline.ost_audit_sidecar import (
    OstAuditPendingError,
    compute_ost_audit_hash,
    format_ost_audit,
    write_ost_audit,
)
from gomoveshift.pipeline.quality_gates import assert_phase_d_quality
from gomoveshift.pipeline.validation.shorts_validator import ShortsValidator
from gomoveshift.pipeline.youtube_sidecar import (
    assert_sidecar_integrity,
    save_sidecar,
)

from _state_helpers import make_state


# ── Shared fixtures ───────────────────────────────────────────────────


def _make_single_lf_drive_plan() -> DrivePlan:
    """A DrivePlan shaped like a full-drive Insta360 ingest: exactly one LF."""
    seg = LFSegment(
        source_file_index=0,
        source_path=None,
        start_time=0.0,
        end_time=1800.0,
        original_start_time=0.0,
        original_end_time=1800.0,
        composite_score=0.7,
        gps_center_lat=0.0,
        gps_center_lon=0.0,
        terrain_type="countryside",
        contiguous_with_previous=False,
    )
    lf = RegularLFPlan(
        index=1,
        segments=[seg],
        montage_peaks=[seg],
        shorts_target=3,
        region="usa",
        total_duration=1800.0,
    )
    return DrivePlan(
        source_files=[],
        processed_files=[],
        regular_lfs=[lf],
        score_threshold=0.5,
        total_source_duration=1800.0,
        total_highlight_duration=1800.0,
    )


def _make_single_lf_state(
    *,
    title: str = "Driving Test Drive, Test County",
    location: str = "Test Drive, Test County",
    publish_day: str = "wed",
    lf_slot_date: str = "2026-04-29",
    shorts_count: int = 3,
    seo_degraded: bool = False,
) -> dict:
    """Phase-D-shaped state with a single LF (index 1)."""
    metadata_per_lf = {
        1: {
            "title": title,
            "description": "Test description",
            "tags": ["driving"],
            "seo_degraded": seo_degraded,
        }
    }
    shorts_per_lf = {1: [object() for _ in range(shorts_count)]}
    lf_slots = [
        SimpleNamespace(
            date=lf_slot_date,
            time="11:00:00",
            datetime_est=f"{lf_slot_date}T11:00:00-04:00",
        )
    ]
    drive_schedule = SimpleNamespace(lf_slots=lf_slots, shorts_slots=[])
    return {
        "publish_day": publish_day,
        "location": "Drive-wide label",
        "metadata_per_lf": metadata_per_lf,
        "per_lf_locations": {1: location},
        "shorts_per_lf": shorts_per_lf,
        "drive_schedule": drive_schedule,
        "allow_seo_degraded": False,
        "youtube_uploads": {},
        "skip_upload": True,
        "upload_manifests": {},
    }


def _write_consolidated_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    full_rows = []
    for r in rows:
        row = {col: "" for col in GOLDEN_COLUMNS}
        row.update(r)
        full_rows.append(row)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=GOLDEN_COLUMNS)
        writer.writeheader()
        writer.writerows(full_rows)
    return path


# ── OST audit sidecar ─────────────────────────────────────────────────


class TestOstAuditSidecarSingleLf:
    """Full-drive runs produce ONE LF/day; the audit payload still has
    one entry, the hash is stable across regenerations, and the
    write-then-read roundtrip preserves shape."""

    def test_format_ost_audit_one_lf_payload(self):
        state = make_state(
            per_lf_ost_text={1: "Killarney to Dingle via N86, Kerry"},
            per_lf_titles={1: "Driving 30min — Killarney to Dingle via N86, Kerry"},
            per_lf_recording_date={1: "2026-05-01"},
            per_lf_locations={1: "County Kerry"},
        )
        payload = format_ost_audit("2026-05-01", state, [1])
        assert payload["date"] == "2026-05-01"
        assert len(payload["lfs"]) == 1
        lf = payload["lfs"][0]
        assert lf["lf_index"] == 1
        assert lf["ost_text"] == "Killarney to Dingle via N86, Kerry"
        # thumbnail_text mirrors ost_text (single source of truth).
        assert lf["thumbnail_text"] == lf["ost_text"]
        assert lf["recording_date"] == "2026-05-01"

    def test_hash_stable_across_calls_single_lf(self):
        state = make_state(
            per_lf_ost_text={1: "Test Drive, Test County"},
            per_lf_titles={1: "Driving 30min — Test Drive, Test County"},
            per_lf_recording_date={1: "2026-05-01"},
            per_lf_locations={},
        )
        payload_a = format_ost_audit("2026-05-01", state, [1])
        payload_b = format_ost_audit("2026-05-01", state, [1])
        # generated_at differs but hash MUST not.
        assert compute_ost_audit_hash(payload_a) == compute_ost_audit_hash(payload_b)

    def test_hash_changes_when_single_lf_label_edited(self):
        """Operator edits per-LF location → hash flips → resume gate
        re-fires until operator re-confirms. Closes the contract for
        the single-LF case (multi-LF case covered elsewhere)."""
        state_a = {
            "per_lf_ost_text": {1: "Killarney to Dingle via N86, Kerry"},
            "per_lf_titles": {1: "Driving — Killarney to Dingle"},
            "per_lf_recording_date": {1: "2026-05-01"},
            "per_lf_locations": {},
        }
        state_b = dict(state_a)
        state_b["per_lf_ost_text"] = {1: "Killarney to Dingle via Conor Pass, Kerry"}
        h_a = compute_ost_audit_hash(format_ost_audit("2026-05-01", state_a, [1]))
        h_b = compute_ost_audit_hash(format_ost_audit("2026-05-01", state_b, [1]))
        assert h_a != h_b

    def test_write_ost_audit_roundtrip_single_lf(self, tmp_path):
        state = make_state(
            per_lf_ost_text={1: "Test Drive, Test County"},
            per_lf_titles={1: "Driving 30min — Test Drive, Test County"},
            per_lf_recording_date={1: "2026-05-01"},
            per_lf_locations={},
        )
        payload = format_ost_audit("2026-05-01", state, [1])
        written = write_ost_audit(tmp_path, "2026-05-01", payload)
        assert written.exists()
        loaded = json.loads(written.read_text())
        assert loaded["date"] == "2026-05-01"
        assert len(loaded["lfs"]) == 1
        assert loaded["lfs"][0]["lf_index"] == 1

    def test_ost_audit_pending_error_is_runtimeerror_subclass(self):
        # The orchestrator catches RuntimeError; OstAuditPendingError must
        # subclass it so the catch site keeps the clean-exit behaviour for
        # single-LF full-drive runs.
        assert issubclass(OstAuditPendingError, RuntimeError)


# ── Sidecar integrity (one-LF) ────────────────────────────────────────


class TestSidecarIntegritySingleLf:
    def test_empty_sidecar_with_single_lf_plan_passes(self, tmp_path):
        plan = _make_single_lf_drive_plan()
        # No sidecar on disk → fresh run → pass unconditionally.
        assert_sidecar_integrity(tmp_path, plan)

    def test_matching_one_entry_sidecar_passes(self, tmp_path):
        plan = _make_single_lf_drive_plan()
        save_sidecar(tmp_path, {1: {"video_id": "abc12345678"}})
        assert_sidecar_integrity(tmp_path, plan)

    def test_phantom_entry_against_single_lf_plan_raises(self, tmp_path):
        plan = _make_single_lf_drive_plan()
        save_sidecar(
            tmp_path,
            {
                1: {"video_id": "abc12345678"},
                2: {"video_id": "def98765432"},  # phantom — plan only has LF 1
            },
        )
        with pytest.raises(SidecarIntegrityError) as exc:
            assert_sidecar_integrity(tmp_path, plan)
        assert 2 in exc.value.phantom_indices
        assert "phantom" in str(exc.value).lower()

    def test_missing_lf1_against_single_lf_plan_raises(self, tmp_path):
        plan = _make_single_lf_drive_plan()
        # Sidecar has an unrelated phantom + lacks LF 1 → both diffs present.
        save_sidecar(tmp_path, {9: {"video_id": "zzz99999999"}})
        with pytest.raises(SidecarIntegrityError) as exc:
            assert_sidecar_integrity(tmp_path, plan)
        assert 1 in exc.value.missing_indices
        assert 9 in exc.value.phantom_indices


# ── Phase D quality gates (single LF) ─────────────────────────────────


class TestPhaseDQualitySingleLf:
    def test_full_drive_happy_path_single_lf_passes(self, tmp_path):
        state = _make_single_lf_state(shorts_count=0)
        plan = _make_single_lf_drive_plan()
        # Empty consolidated CSV → 0 expected rows, 0 actual rows.
        state["shorts_per_lf"] = {1: []}
        csv_path = _write_consolidated_csv(tmp_path / "shorts.csv", [])
        # Should not raise.
        assert_phase_d_quality(state, plan, csv_path)

    def test_full_drive_csv_count_matches_single_lf_shorts(self, tmp_path):
        state = _make_single_lf_state(shorts_count=3)
        plan = _make_single_lf_drive_plan()
        rows = [{} for _ in range(3)]
        csv_path = _write_consolidated_csv(tmp_path / "shorts.csv", rows)
        assert_phase_d_quality(state, plan, csv_path)

    def test_full_drive_title_uniqueness_vacuous_for_single_lf(self, tmp_path):
        """Rule 1 (titles distinct) is vacuous when there is only one LF —
        a single title cannot collide with itself. This pins that the rule
        does not raise a spurious "not distinct" error for ``len(titles) == 1``.
        """
        state = _make_single_lf_state(title="The Only Title On The Drive", shorts_count=0)
        state["shorts_per_lf"] = {1: []}
        plan = _make_single_lf_drive_plan()
        csv_path = _write_consolidated_csv(tmp_path / "shorts.csv", [])
        assert_phase_d_quality(state, plan, csv_path)

    def test_slot_count_mismatch_still_caught_for_single_lf(self, tmp_path):
        """Rule 7 (schedule slot count == plan LF count) — make sure a
        one-LF plan with zero scheduled slots still raises (off-by-one
        guard)."""
        state = _make_single_lf_state(shorts_count=0)
        state["shorts_per_lf"] = {1: []}
        state["drive_schedule"] = SimpleNamespace(lf_slots=[], shorts_slots=[])
        plan = _make_single_lf_drive_plan()
        csv_path = _write_consolidated_csv(tmp_path / "shorts.csv", [])
        with pytest.raises(PhaseQualityError) as exc:
            assert_phase_d_quality(state, plan, csv_path)
        msg = str(exc.value)
        assert "lf_slots count mismatch" in msg
        assert "1 regular_lfs" in msg


# ── Audio loudness hard fail (single LF) ──────────────────────────────


class TestAudioLoudnessHardFailSingleLf:
    def test_no_failures_silent_pass(self):
        # Empty list → no raise. Same contract as multi-LF.
        assert_audio_loudness_hard_fail([])

    def test_single_lf_out_of_spec_raises_naming_lf(self):
        failures = [
            "  LF 1 (project_lf01_FINAL.mp4): loudness -24.9 outside -20 +/- 2.0 LUFS window"
        ]
        with pytest.raises(RuntimeError) as exc:
            assert_audio_loudness_hard_fail(failures)
        msg = str(exc.value)
        assert "HARD FAIL" in msg
        assert "LF 1" in msg
        assert "-24.9" in msg


# ── Shorts validator (single-parent-LF case) ──────────────────────────


class TestShortsValidatorSingleParentLf:
    """Shorts ARE multi-instance even when there's only one parent LF
    (full-drive mode still uses the per-second shorts planner against
    the day master). Validator works against a flat path list keyed by
    global short_index, so the parent-LF count is invisible at this
    layer — but pin it explicitly so a future refactor that adds a per-LF
    grouping does not silently regress."""

    def test_empty_list_skipped(self):
        ffprobe = MagicMock()
        validator = ShortsValidator(ffprobe=ffprobe)
        result = validator.validate_short_files([])
        assert result.passed is True
        assert len(result.gates) == 1
        assert "skipped" in result.gates[0].message.lower()

    def test_single_parent_lf_multi_shorts_validate_per_file(self, tmp_path):
        """With a single parent LF, multiple short_*.mp4 still validate
        independently — short_index is the only key. Build a fake file
        list, mock ffprobe to return 1080x1920 in-spec data for each,
        and assert all gates pass."""
        # Create stub files (>500 KB) so the integrity check passes.
        short_files: list[Path] = []
        for i in range(1, 4):
            p = tmp_path / f"short_{i:02d}.mp4"
            p.write_bytes(b"\x00" * (501 * 1024))
            short_files.append(p)

        ffprobe = MagicMock()
        # probe_or_none returns valid 1080x1920 / 35s / 4Mbps for each call.
        ffprobe.probe_or_none.return_value = {
            "format": {"duration": "35.0", "bit_rate": "4000000"},
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "hevc",
                    "width": 1080,
                    "height": 1920,
                    # Encoding contract: in-spec 10-bit HEVC main10
                    # with SDR BT.709 limited-range tags.
                    "profile": "Main 10",
                    "pix_fmt": "yuv420p10le",
                    "color_range": "tv",
                    "color_space": "bt709",
                    "color_primaries": "bt709",
                    "color_transfer": "bt709",
                },
                {"codec_type": "audio", "codec_name": "aac"},
            ],
        }
        # probe() (used by integrity check) just needs to not raise.
        ffprobe.probe.return_value = {"format": {"duration": "35.0"}}
        # Mock the inner subprocess used by the LUFS check to report in-spec.
        ffprobe._subprocess = MagicMock()
        lufs_result = MagicMock()
        lufs_result.returncode = 0
        lufs_result.stdout = ""
        lufs_result.stderr = (
            '{\n  "input_i" : "-20.00",\n'
            '  "input_tp" : "-1.50",\n'
            '  "input_lra" : "5.00",\n'
            '  "input_thresh" : "-25.00",\n'
            '  "output_i" : "-20.00",\n'
            '  "output_tp" : "-1.50",\n'
            '  "output_lra" : "5.00",\n'
            '  "output_thresh" : "-25.00",\n'
            '  "normalization_type" : "dynamic",\n'
            '  "target_offset" : "0.00"\n}\n'
        )
        ffprobe._subprocess.run.return_value = lufs_result

        validator = ShortsValidator(ffprobe=ffprobe)
        result = validator.validate_short_files(short_files)
        assert result.passed is True, result.summary()
        # Three shorts × five sub-gates each (integrity, resolution, duration,
        # bitrate, audio_lufs) — pin >= 3 to avoid coupling to the exact gate
        # mix while still asserting per-file iteration.
        per_file_gates = [g for g in result.gates if "short_resolution:" in g.gate]
        assert len(per_file_gates) == 3, [g.gate for g in result.gates]

    def test_single_short_off_spec_resolution_fatal(self, tmp_path):
        """A single off-spec short under a single parent LF still
        produces a FATAL — no LF-count fast path bypasses the check."""
        p = tmp_path / "short_01.mp4"
        p.write_bytes(b"\x00" * (501 * 1024))

        ffprobe = MagicMock()
        ffprobe.probe_or_none.return_value = {
            "format": {"duration": "35.0", "bit_rate": "4000000"},
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,  # 16:9 — the regression class
                },
                {"codec_type": "audio", "codec_name": "aac"},
            ],
        }
        ffprobe.probe.return_value = {"format": {"duration": "35.0"}}
        ffprobe._subprocess = MagicMock()
        lufs_result = MagicMock()
        lufs_result.returncode = 0
        lufs_result.stdout = ""
        lufs_result.stderr = (
            '{\n  "input_i" : "-20.00",\n'
            '  "input_tp" : "-1.50",\n'
            '  "input_lra" : "5.00",\n'
            '  "input_thresh" : "-25.00",\n'
            '  "output_i" : "-20.00",\n'
            '  "output_tp" : "-1.50",\n'
            '  "output_lra" : "5.00",\n'
            '  "output_thresh" : "-25.00",\n'
            '  "normalization_type" : "dynamic",\n'
            '  "target_offset" : "0.00"\n}\n'
        )
        ffprobe._subprocess.run.return_value = lufs_result

        validator = ShortsValidator(ffprobe=ffprobe)
        result = validator.validate_short_files([p])
        assert result.passed is False
        resolution_gate = next(g for g in result.gates if "short_resolution:" in g.gate)
        assert resolution_gate.passed is False
        assert "1920x1080" in resolution_gate.message


# ── gate_a skeleton for balanced parts ────────────────────────────────


class TestFullDriveSkeletonIndices:
    def test_unset_parts_keep_single_lf_skeleton(self):
        from gomoveshift.pipeline.drive_orchestrator import full_drive_skeleton_indices

        drive = SimpleNamespace(full_drive_part_min_sec=None, full_drive_part_max_sec=None)
        assert full_drive_skeleton_indices(drive, None) == [1]

    def test_parts_take_skeleton_from_plans(self):
        from gomoveshift.pipeline.drive_orchestrator import full_drive_skeleton_indices

        drive = SimpleNamespace(full_drive_part_min_sec=5400, full_drive_part_max_sec=7200)
        plans = [SimpleNamespace(index=1), SimpleNamespace(index=2)]
        assert full_drive_skeleton_indices(drive, plans) == [1, 2]

    def test_parts_without_plans_yet_has_no_skeleton(self):
        from gomoveshift.pipeline.drive_orchestrator import full_drive_skeleton_indices

        drive = SimpleNamespace(full_drive_part_min_sec=5400, full_drive_part_max_sec=7200)
        assert full_drive_skeleton_indices(drive, None) is None
