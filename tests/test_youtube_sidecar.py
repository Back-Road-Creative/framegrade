"""Tests for the youtube_uploads.json sidecar integrity gate.

Background: the sidecar at
``<output_base>/youtube_uploads.json`` is a local record of prior
upload attempts, NOT a live mirror of YouTube. Operators can delete
drafts in YouTube Studio; aborted runs can leave phantom slots; the
pipeline then treats those stale entries as authoritative and bakes
dead video IDs into the Metricool CSV's First Comment Text.

These tests cover ``assert_sidecar_integrity`` — the local
state/plan agreement gate called at Phase D entry. API-level
reconciliation is intentionally out of scope; the
gate only asserts that sidecar slot count == len(drive_plan.regular_lfs)
and that the integer LF-index sets agree.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gomoveshift.models import DrivePlan, LFSegment, RegularLFPlan
from gomoveshift.pipeline.exceptions import SidecarIntegrityError
from gomoveshift.pipeline.youtube_sidecar import (
    assert_sidecar_integrity,
    assert_superlong_sidecar_integrity,
    load_sidecar,
    save_sidecar,
    save_superlong_sidecar,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_drive_plan(n_lfs: int) -> DrivePlan:
    """Build a minimal DrivePlan with *n_lfs* regular LFs, indexes 1..n."""
    regular_lfs = []
    for i in range(1, n_lfs + 1):
        seg = LFSegment(
            source_file_index=0,
            source_path=None,
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
                index=i,
                segments=[seg],
                montage_peaks=[seg],
                shorts_target=3,
                region="usa",
                total_duration=600.0,
            )
        )
    return DrivePlan(
        source_files=[],
        processed_files=[],
        regular_lfs=regular_lfs,
        score_threshold=0.5,
        total_source_duration=600.0 * n_lfs,
        total_highlight_duration=600.0 * n_lfs,
    )


def _write_sidecar(output_base, uploads):
    """Write a sidecar to disk via save_sidecar."""
    save_sidecar(output_base, uploads)


# ── Exception surface ─────────────────────────────────────────────────


class TestSidecarIntegrityError:
    """SidecarIntegrityError must be a ValueError subclass so it shows up
    in the same generic except-value-error paths, while still being
    distinguishable for operator messaging."""

    def test_is_value_error_subclass(self):
        assert issubclass(SidecarIntegrityError, ValueError)

    def test_carries_phantom_and_missing_indices(self):
        err = SidecarIntegrityError(
            sidecar_path="/tmp/youtube_uploads.json",
            phantom_indices=[8, 9, 10],
            missing_indices=[],
            message="test",
        )
        assert err.phantom_indices == [8, 9, 10]
        assert err.missing_indices == []
        assert "test" in str(err)


# ── assert_sidecar_integrity ──────────────────────────────────────────


class TestAssertSidecarIntegrity:
    """Local state/plan agreement gate at Phase D entry."""

    def test_empty_sidecar_passes(self, tmp_path):
        """Fresh runs with no sidecar on disk pass cleanly (the common
        case — nothing to validate against)."""
        drive_plan = _make_drive_plan(n_lfs=7)
        # No sidecar file written
        assert_sidecar_integrity(tmp_path, drive_plan)

    def test_empty_sidecar_dict_passes(self, tmp_path):
        """An on-disk sidecar of {} (explicit reset) also passes."""
        _write_sidecar(tmp_path, {})
        drive_plan = _make_drive_plan(n_lfs=7)
        assert_sidecar_integrity(tmp_path, drive_plan)

    def test_matching_count_passes(self, tmp_path):
        """Sidecar with exactly len(drive_plan.regular_lfs) entries AND
        matching LF indices is considered consistent."""
        uploads = {
            i: {"video_id": f"vid{i:02d}", "url": f"https://youtu.be/vid{i:02d}"}
            for i in range(1, 8)  # 1..7 inclusive
        }
        _write_sidecar(tmp_path, uploads)
        drive_plan = _make_drive_plan(n_lfs=7)
        assert_sidecar_integrity(tmp_path, drive_plan)

    def test_mismatched_extra_sidecar_entries_raises(self, tmp_path):
        """Sidecar has 10 entries but plan only has 7 — the extras are
        phantoms (often from aborted prior runs). The gate must raise
        with the offending indices in the message so an operator can
        decide to wipe or reconcile."""
        uploads = {
            i: {"video_id": f"vid{i:02d}", "url": f"https://youtu.be/vid{i:02d}"}
            for i in range(1, 11)  # 1..10 — three extras
        }
        _write_sidecar(tmp_path, uploads)
        drive_plan = _make_drive_plan(n_lfs=7)
        with pytest.raises(SidecarIntegrityError) as exc_info:
            assert_sidecar_integrity(tmp_path, drive_plan)
        err = exc_info.value
        # Phantom indices: present in sidecar, absent in plan
        assert set(err.phantom_indices) == {8, 9, 10}
        assert err.missing_indices == []
        msg = str(err)
        assert "phantom LF indices" in msg
        assert "[8, 9, 10]" in msg
        # Message must hint at the two operator paths.
        assert "--force-sidecar-reset" in msg

    def test_mismatched_missing_sidecar_entries_raises(self, tmp_path):
        """Sidecar has 3 entries but plan has 7 — four slots are missing.
        This is suspicious (did a prior run partially upload?) and
        should surface so the operator can reconcile."""
        uploads = {
            i: {"video_id": f"vid{i:02d}", "url": f"https://youtu.be/vid{i:02d}"}
            for i in range(1, 4)  # 1..3
        }
        _write_sidecar(tmp_path, uploads)
        drive_plan = _make_drive_plan(n_lfs=7)
        with pytest.raises(SidecarIntegrityError) as exc_info:
            assert_sidecar_integrity(tmp_path, drive_plan)
        err = exc_info.value
        assert err.phantom_indices == []
        assert set(err.missing_indices) == {4, 5, 6, 7}
        msg = str(err)
        assert "missing LF indices" in msg
        assert "[4, 5, 6, 7]" in msg

    def test_disjoint_indices_raises_with_both_diffs(self, tmp_path):
        """Sidecar keys that overlap neither the plan's lower nor upper
        bound produce both a phantom and a missing list."""
        # Plan has LF 1..5; sidecar has LF 10..14 — entirely disjoint.
        uploads = {
            i: {"video_id": f"vid{i:02d}", "url": f"https://youtu.be/vid{i:02d}"}
            for i in range(10, 15)
        }
        _write_sidecar(tmp_path, uploads)
        drive_plan = _make_drive_plan(n_lfs=5)
        with pytest.raises(SidecarIntegrityError) as exc_info:
            assert_sidecar_integrity(tmp_path, drive_plan)
        err = exc_info.value
        assert set(err.phantom_indices) == {10, 11, 12, 13, 14}
        assert set(err.missing_indices) == {1, 2, 3, 4, 5}

    def test_str_keys_are_normalized_to_int(self, tmp_path):
        """Sidecar JSON always has string keys. The gate must coerce to
        int LF indices to match drive_plan.regular_lfs[*].index."""
        # Write the file directly with string keys (matches save_sidecar).
        sidecar_path = tmp_path / "youtube_uploads.json"
        uploads = {str(i): {"video_id": f"vid{i:02d}"} for i in range(1, 8)}
        sidecar_path.write_text(json.dumps(uploads))
        drive_plan = _make_drive_plan(n_lfs=7)
        # Should pass: 7 str-keyed entries == 7 plan LFs with int indices 1..7
        assert_sidecar_integrity(tmp_path, drive_plan)

    def test_malformed_sidecar_keys_surface_gracefully(self, tmp_path):
        """Non-integer sidecar keys (e.g. accidentally written as LF
        names like 'lf_01') cannot be matched against plan indices —
        they should be reported as phantoms, not silently skipped."""
        sidecar_path = tmp_path / "youtube_uploads.json"
        uploads = {
            "lf_01": {"video_id": "vid01"},
            "lf_02": {"video_id": "vid02"},
        }
        sidecar_path.write_text(json.dumps(uploads))
        drive_plan = _make_drive_plan(n_lfs=7)
        with pytest.raises(SidecarIntegrityError) as exc_info:
            assert_sidecar_integrity(tmp_path, drive_plan)
        # Non-coercible keys are surfaced as the raw strings in the
        # error message so operators can spot the malformed file.
        msg = str(exc_info.value)
        assert "lf_01" in msg or "phantom" in msg.lower()

    def test_empty_plan_with_nonempty_sidecar_raises(self, tmp_path):
        """If the plan has zero LFs but the sidecar has entries, every
        sidecar entry is phantom."""
        uploads = {1: {"video_id": "vid01"}, 2: {"video_id": "vid02"}}
        _write_sidecar(tmp_path, uploads)
        drive_plan = _make_drive_plan(n_lfs=0)
        with pytest.raises(SidecarIntegrityError) as exc_info:
            assert_sidecar_integrity(tmp_path, drive_plan)
        err = exc_info.value
        assert set(err.phantom_indices) == {1, 2}
        assert err.missing_indices == []

    def test_error_message_includes_sidecar_path(self, tmp_path):
        """Operator-facing messages must point at the offending file."""
        uploads = {1: {"video_id": "vid01"}}
        _write_sidecar(tmp_path, uploads)
        drive_plan = _make_drive_plan(n_lfs=7)
        with pytest.raises(SidecarIntegrityError) as exc_info:
            assert_sidecar_integrity(tmp_path, drive_plan)
        assert str(tmp_path / "youtube_uploads.json") in str(exc_info.value)

    def test_ignores_a_populated_superlong_sidecar(self, tmp_path):
        """Regression: a populated superlong_uploads.json is never read/merged by the LF gate."""
        uploads = {i: {"video_id": f"vid{i:02d}"} for i in range(1, 4)}
        _write_sidecar(tmp_path, uploads)
        save_superlong_sidecar(tmp_path, {"2026-03-01:01": {"video_id": "PARTID"}})
        drive_plan = _make_drive_plan(n_lfs=3)
        assert_sidecar_integrity(tmp_path, drive_plan)  # still passes


def _superlong_state(parts_by_date: dict) -> SimpleNamespace:
    """``{date: [part_index, ...]}`` → minimal ``state.superlong_parts``."""
    parts = {
        d: {"part_count": len(ix), "records": [{"path": "x", "part_index": i} for i in ix]}
        for d, ix in parts_by_date.items()
    }
    return SimpleNamespace(superlong_parts=parts)


class TestAssertSuperlongSidecarIntegrity:
    def test_matching_keys_pass(self, tmp_path):
        state = _superlong_state({"2026-03-01": [1, 2]})
        uploads = {"2026-03-01:01": {"video_id": "a"}, "2026-03-01:02": {"video_id": "b"}}
        save_superlong_sidecar(tmp_path, uploads)
        assert_superlong_sidecar_integrity(tmp_path, state)

    def test_phantom_and_missing_raise(self, tmp_path):
        state = _superlong_state({"2026-03-01": [1]})
        uploads = {"2026-03-01:01": {"video_id": "a"}, "2026-02-01:09": {"video_id": "ghost"}}
        save_superlong_sidecar(tmp_path, uploads)
        with pytest.raises(SidecarIntegrityError) as exc_info:
            assert_superlong_sidecar_integrity(tmp_path, state)
        err = exc_info.value
        assert (err.phantom_indices, err.missing_indices) == (["2026-02-01:09"], [])
