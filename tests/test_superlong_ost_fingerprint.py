"""Superlong part freshness must key off the burned material inputs, not
file-size alone — a re-plan that shifts a part's SEGMENT
WINDOWS (``part.ranges``) while its part INDEX stays the same must NOT
reuse the old-boundary ``sl_NN.mp4`` bytes. Mirrors the LF
fingerprint (``_lf_ost_fingerprint``) but for superlong's own inputs."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from _state_helpers import make_state


def _make_config(part_max_sec: int = 1200) -> SimpleNamespace:
    return SimpleNamespace(
        drive=SimpleNamespace(
            brand_logo_path=None,
            superlong_enabled=True,
            superlong_part_max_duration_sec=part_max_sec,
            superlong_split_grace=0.08,
            superlong_lower_third_duration_sec=8.0,
            superlong_lower_third_min_gap_sec=45.0,
        )
    )


def _fake_ost(input_path, output_path, location, layout, **kwargs):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_bytes(b"fake ost bytes")
    return Path(output_path)


def _fake_assemble(master_path, segments, file_boundaries, output_path, work_dir, **kw):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_bytes(b"concat")
    return Path(output_path)


def _produce(tmp_path, stop_mask, part_max_sec=1200):
    from gomoveshift.pipeline import day_pipeline

    master_path = tmp_path / "day_master.mp4"
    if not master_path.is_file():
        master_path.write_bytes(b"fake master")
    day_dir = tmp_path / "day_2026-06-01"
    day_dir.mkdir(exist_ok=True)
    ep = SimpleNamespace(concat_path=Path("/fake/ep_000.mp4"))
    config = _make_config(part_max_sec)

    with (
        patch("gomoveshift.pipeline.day_pipeline.apply_ost", side_effect=_fake_ost) as mock_ost,
        patch(
            "gomoveshift.pipeline.day_pipeline.extract_and_concat_segments",
            side_effect=_fake_assemble,
        ) as mock_extract,
    ):
        result = day_pipeline._produce_superlong_parts(
            date="2026-06-01",
            stop_mask=stop_mask,
            episodes=[ep],
            file_boundaries=[(0.0, 1200.0)],
            master_path=master_path,
            day_dir=day_dir,
            config=config,
            location="Test Drive, Test County",
            ost_crossfade_sec=0.5,
            state=make_state(ost_audit_confirmed_this_run=True),
        )
    return result, mock_extract, mock_ost, day_dir


class TestSuperlongOstFingerprint:
    def test_replan_with_shifted_ranges_does_not_reuse_old_boundary_bytes(self, tmp_path):
        # Nothing stopped -> single part spanning the whole 1200s.
        mask_a = np.zeros(1200, dtype=bool)
        result_a, extract_a, ost_a, day_dir = _produce(tmp_path, mask_a)
        assert result_a["part_count"] == 1
        sl_final = day_dir / "superlong" / "sl_01.mp4"
        assert sl_final.is_file() and sl_final.stat().st_size > 0
        assert extract_a.call_count == 1
        assert ost_a.call_count == 1

        # Re-plan: first 200s now stopped -> the SAME part index (1 of 1)
        # but a DIFFERENT segment window (200..1200 instead of 0..1200).
        mask_b = np.zeros(1200, dtype=bool)
        mask_b[:200] = True
        result_b, extract_b, ost_b, _ = _produce(tmp_path, mask_b)
        assert result_b["part_count"] == 1
        # The old-boundary final must NOT be silently adopted -- it must
        # be re-extracted and re-burned from the new windows.
        assert extract_b.call_count == 1
        assert ost_b.call_count == 1

    def test_unchanged_plan_skips_the_reburn(self, tmp_path):
        mask = np.zeros(1200, dtype=bool)
        _r1, extract_1, ost_1, day_dir = _produce(tmp_path, mask)
        assert extract_1.call_count == 1
        assert ost_1.call_count == 1
        fp_path = day_dir / "superlong" / "sl_01.ost.sha256"
        assert fp_path.is_file() and fp_path.read_text().strip()

        # Same stop_mask, same day_dir -> the final is fresh, so a resume
        # must not pay the ~10-15min NVENC re-encode again.
        _r2, extract_2, ost_2, _ = _produce(tmp_path, mask)
        assert extract_2.call_count == 0
        assert ost_2.call_count == 0
