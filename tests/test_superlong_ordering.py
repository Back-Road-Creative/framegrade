"""Superlong parts must encode AFTER the shorts postburn proof and
BEFORE the day's intermediate purge — not before the shorts work.

The long-forms and shorts should never queue behind a superlong part that
can run to a 4h/part cap; moving the block later lets them ship first.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from gomoveshift.pipeline.day_grouper import DayBatch
from _state_helpers import make_state


def _make_scores(duration_sec: int) -> np.ndarray:
    arr = np.zeros((duration_sec, 9), dtype=np.float64)
    arr[:, 0] = np.arange(duration_sec, dtype=np.float64)  # timestamp
    arr[:, 1], arr[:, 2] = 0.7, 0.02  # composite score, motion energy
    return arr


def _make_config() -> SimpleNamespace:
    return SimpleNamespace(
        drive=SimpleNamespace(
            brand_logo_path=None,
            montage_max_duration_sec=15.0,
            ost_title_crossfade_sec=0.5,
            lf_selection_mode="best_moments",
            shorts_target_count=None,
            shorts_min_gap_sec=60.0,
            superlong_enabled=True,
            superlong_part_max_duration_sec=1200,
            superlong_split_grace=0.08,
        )
    )


def test_superlong_dispatch_runs_after_shorts_postburn_gate_and_before_purge(tmp_path):
    """Record the call order of the shorts postburn gate, the superlong
    dispatch, and the intermediate purge. Today the superlong block fires
    before any shorts work at all; it must fire between the gate and the
    purge."""
    from gomoveshift.pipeline import day_pipeline

    date = "2026-06-01"
    config = _make_config()
    state = make_state(ost_audit_confirmed=True)

    ep = SimpleNamespace(
        index=1, concat_path=Path("/fake/ep_000.mp4"), total_duration=3600.0, date=date
    )
    day_batch = DayBatch(date=date, episodes=[ep])
    day_scores = _make_scores(3600)
    master_path = tmp_path / "out" / f"day_{date}" / f"day_{date}_master.mp4"
    master_path.parent.mkdir(parents=True, exist_ok=True)
    master_path.write_bytes(b"fake master")

    calls: list[str] = []

    def _fake_ost(input_path, output_path, location, layout, **kwargs):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"fake ost")
        return output_path

    def _fake_assemble(master_path, segments, file_boundaries, output_path, work_dir, **kw):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"concat")
        return Path(output_path)

    def _fake_montage(raw_lf_path, peaks, output_path, work_dir, **kw):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"with_montage")
        return Path(output_path)

    def _fake_render(source_video, output_path, start_time, duration, location, **kwargs):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"fake short")
        return True

    def _fake_superlong(*args, **kwargs):
        calls.append("superlong")
        return {"part_count": 0, "records": []}

    def _fake_postburn(*args, **kwargs):
        calls.append("postburn_gate")

    def _fake_purge(*args, **kwargs):
        calls.append("purge")
        return {"files_deleted": 0, "dirs_deleted": 0, "bytes_freed": 0}

    seg = SimpleNamespace(source_file_index=0, start_time=0.0, end_time=3600.0)
    plan = SimpleNamespace(index=1, segments=[seg], montage_peaks=[])

    with (
        patch("gomoveshift.pipeline.day_pipeline.encode_day_master", return_value=master_path),
        patch("gomoveshift.pipeline.day_pipeline.apply_ost", side_effect=_fake_ost),
        patch(
            "gomoveshift.pipeline.day_pipeline.extract_and_concat_segments",
            side_effect=_fake_assemble,
        ),
        patch("gomoveshift.pipeline.day_pipeline.prepend_montage", side_effect=_fake_montage),
        patch("gomoveshift.pipeline.day_pipeline._render_short_ffmpeg", side_effect=_fake_render),
        patch(
            "gomoveshift.pipeline.day_pipeline._produce_superlong_parts",
            side_effect=_fake_superlong,
        ),
        patch(
            "gomoveshift.pipeline.day_pipeline.postburn_ost_gate_or_raise",
            side_effect=_fake_postburn,
        ),
        patch(
            "gomoveshift.pipeline.day_pipeline.purge_day_intermediates",
            side_effect=_fake_purge,
        ),
    ):
        day_pipeline.run_day_pipeline(
            day_batch=day_batch,
            day_scores=day_scores,
            output_dir=tmp_path / "out",
            location="Test Drive, Test County",
            config=config,
            lf_plans=[plan],
            state=state,
        )

    assert calls == ["postburn_gate", "superlong", "purge"], (
        "superlong must dispatch after the shorts postburn gate and before "
        f"the purge, got order: {calls}"
    )
