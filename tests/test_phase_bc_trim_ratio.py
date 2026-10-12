"""TDD tests: phase_bc_day_first computes and persists trim_ratio.

``state["per_lf_trim_ratio"][idx]`` is the single source of truth for the
"highlights vs full drive" band selection consumed by the metadata generator
(title verb + description hook) and the shorts CSV first-comment lead.

trim_ratio = lf.total_duration / sum(unique source file durations that fed the LF)

* Fresh-encode path: computed from ``skeleton_lfs`` + ``state["file_durations"]``.
* Resume path (state empty, finals on disk): derived via ``ffprobe`` of the
  final LF + the same source-file-durations sum.
* Phase BC invariant: every ``skeleton_lfs[i].index`` must have an entry, or
  the phase raises with the missing indices listed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _seg(source_file_index: int, start: float = 0.0, end: float = 10.0):
    return SimpleNamespace(
        source_file_index=source_file_index,
        source_path=f"/tmp/fake_episode_{source_file_index}.mp4",
        start_time=start,
        end_time=end,
        original_start_time=start,
        original_end_time=end,
    )


def _lf(index: int, segments, total_duration: float):
    return SimpleNamespace(
        index=index,
        segments=segments,
        montage_peaks=[],
        total_duration=total_duration,
    )


class TestComputePerLFTrimRatio:
    """Fresh-encode path: skeleton_lfs + file_durations → per_lf_trim_ratio."""

    def test_single_lf_single_source_file(self):
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            _compute_per_lf_trim_ratio,
        )

        lfs = [_lf(1, [_seg(0)], total_duration=300.0)]
        file_durations = [1000.0]
        result = _compute_per_lf_trim_ratio(lfs, file_durations)
        assert set(result.keys()) == {1}
        assert result[1] == pytest.approx(0.3, abs=1e-4)

    def test_lf_uses_two_source_files(self):
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            _compute_per_lf_trim_ratio,
        )

        # LF kept 600s out of 1000 + 500 = 1500s of source clips
        lfs = [_lf(1, [_seg(0), _seg(1), _seg(0)], total_duration=600.0)]
        file_durations = [1000.0, 500.0, 9999.0]  # third file unused
        result = _compute_per_lf_trim_ratio(lfs, file_durations)
        assert result[1] == pytest.approx(600.0 / 1500.0, abs=1e-4)

    def test_multiple_lfs_independent(self):
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            _compute_per_lf_trim_ratio,
        )

        lfs = [
            _lf(1, [_seg(0)], total_duration=200.0),
            _lf(2, [_seg(1), _seg(2)], total_duration=900.0),
        ]
        file_durations = [1000.0, 600.0, 600.0]
        result = _compute_per_lf_trim_ratio(lfs, file_durations)
        assert result[1] == pytest.approx(0.2, abs=1e-4)
        assert result[2] == pytest.approx(900.0 / 1200.0, abs=1e-4)

    def test_zero_or_missing_source_duration_yields_none(self):
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            _compute_per_lf_trim_ratio,
        )

        lfs = [_lf(1, [_seg(7)], total_duration=300.0)]  # idx 7 not in durations
        file_durations = [1000.0]
        result = _compute_per_lf_trim_ratio(lfs, file_durations)
        # Source duration unresolved → ratio absent so the caller can fall back.
        # Per spec: surface a clear absence rather than silently degrading.
        assert 1 not in result or result[1] is None

    def test_clamps_above_one(self):
        """Edge case: planner over-counts vs file durations → clamp to 1.0."""
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            _compute_per_lf_trim_ratio,
        )

        lfs = [_lf(1, [_seg(0)], total_duration=2000.0)]
        file_durations = [1000.0]
        result = _compute_per_lf_trim_ratio(lfs, file_durations)
        assert result[1] == pytest.approx(1.0, abs=1e-4)


class TestDerivePerLFTrimRatioFromPaths:
    """Resume backstop: ffprobe the final LF on disk + same sources sum."""

    def test_uses_ffprobe_for_final_duration(self, tmp_path):
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            _derive_per_lf_trim_ratio_from_paths,
        )

        lf1_path = tmp_path / "day_2025-09-14" / "regular_lfs" / "lf_01.mp4"
        lf1_path.parent.mkdir(parents=True)
        lf1_path.write_bytes(b"")

        lfs = [_lf(1, [_seg(0)], total_duration=999.0)]  # not consulted
        file_durations = [1000.0]

        def fake_probe(p):
            return 250.0  # final duration measured from disk

        result = _derive_per_lf_trim_ratio_from_paths(
            skeleton_lfs=lfs,
            regular_lf_paths=[lf1_path],
            file_durations=file_durations,
            probe_duration_fn=fake_probe,
        )
        assert result[1] == pytest.approx(0.25, abs=1e-4)

    def test_missing_file_omits_entry(self, tmp_path):
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            _derive_per_lf_trim_ratio_from_paths,
        )

        missing = tmp_path / "nope.mp4"
        lfs = [_lf(1, [_seg(0)], total_duration=300.0)]
        result = _derive_per_lf_trim_ratio_from_paths(
            skeleton_lfs=lfs,
            regular_lf_paths=[missing],
            file_durations=[1000.0],
            probe_duration_fn=lambda p: 0.0,
        )
        assert 1 not in result


class TestTrimRatioInvariant:
    """The invariant raises when an LF index is missing from the trim_ratio map."""

    def test_invariant_passes_when_complete(self):
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            _assert_per_lf_trim_ratio_complete,
        )

        lfs = [_lf(1, [_seg(0)], 0.0), _lf(2, [_seg(0)], 0.0)]
        # Should not raise.
        _assert_per_lf_trim_ratio_complete(lfs, {1: 0.3, 2: 0.7})

    def test_invariant_raises_with_missing_indices(self):
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            _assert_per_lf_trim_ratio_complete,
        )

        lfs = [
            _lf(1, [_seg(0)], 0.0),
            _lf(2, [_seg(0)], 0.0),
            _lf(3, [_seg(0)], 0.0),
        ]
        with pytest.raises(RuntimeError) as excinfo:
            _assert_per_lf_trim_ratio_complete(lfs, {1: 0.3})
        msg = str(excinfo.value)
        assert "per_lf_trim_ratio" in msg
        assert "2" in msg
        assert "3" in msg

    def test_invariant_raises_when_value_is_none(self):
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            _assert_per_lf_trim_ratio_complete,
        )

        lfs = [_lf(1, [_seg(0)], 0.0)]
        with pytest.raises(RuntimeError):
            _assert_per_lf_trim_ratio_complete(lfs, {1: None})

    def test_invariant_skips_lfs_without_segments(self):
        """Defensive-fallback LFs (no segments) are excluded — their
        source clip metadata is unrecoverable by construction."""
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            _assert_per_lf_trim_ratio_complete,
        )

        # LF 2 has segments but no entry → must raise.
        # LF 1 has no segments and no entry → ignored.
        lfs = [_lf(1, [], 0.0), _lf(2, [_seg(0)], 0.0)]
        with pytest.raises(RuntimeError) as excinfo:
            _assert_per_lf_trim_ratio_complete(lfs, {})
        assert "2" in str(excinfo.value)
        assert "[1, 2]" not in str(excinfo.value)  # LF 1 not listed
