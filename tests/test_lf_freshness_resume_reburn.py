"""Per-LF freshness check on resume.

The documented shorts-OST-gate → confirm → resume flow stops the run AFTER
every regular LF has already been burned to its final ``lf_NN.mp4`` (~10-15
min NVENC each) but BEFORE the shorts render. On resume the per-LF loop must
NOT re-burn an LF whose final output already exists and whose burned OST text
is unchanged — that is a pure waste of a long NVENC re-encode. A materially
changed LF (different burned OST text) MUST still re-burn.

These tests drive that behaviour: run the day pipeline twice against the same
output dir (a resume) and assert ``apply_ost`` is skipped for an unchanged
final but re-invoked for a changed one.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from _state_helpers import make_state

from gomoveshift.pipeline import day_pipeline
from gomoveshift.pipeline.day_grouper import DayBatch

NUM_COLUMNS = 9


def _make_scores(duration_sec: int, score: float = 0.6) -> np.ndarray:
    arr = np.zeros((duration_sec, NUM_COLUMNS), dtype=np.float64)
    arr[:, 0] = np.arange(duration_sec, dtype=np.float64)
    arr[:, 1] = score
    return arr


def _make_day_batch(date: str, n: int, per_ep: float = 1200.0) -> DayBatch:
    episodes = [
        SimpleNamespace(
            index=i + 1,
            concat_path=Path(f"/fake/ep_{i:03d}.mp4"),
            total_duration=per_ep,
            date=date,
            creation_time=None,
        )
        for i in range(n)
    ]
    return DayBatch(date=date, episodes=episodes)


def _make_seg(ep: int, start: float, end: float):
    return SimpleNamespace(
        source_file_index=ep,
        start_time=float(start),
        end_time=float(end),
        duration=float(end - start),
        gps_center_lat=0.0,
        gps_center_lon=0.0,
    )


def _make_lf_plan(idx: int, segs, peaks):
    return SimpleNamespace(index=idx, segments=segs, montage_peaks=peaks)


def _run(output_base: Path, master_path: Path, location: str):
    """Run one day pipeline invocation; return (artifacts, apply_ost mock)."""

    def _fake_asm(master_path, segments, file_boundaries, output_path, work_dir, **kw):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"concat")
        return output_path

    def _fake_prep(raw_lf_path, peaks, output_path, work_dir, **kw):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"montage")
        return output_path

    def _fake_ost(input_path, output_path, location, layout, **kw):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"ost")
        return output_path

    def _fake_render(source_video, output_path, start_time, duration, location, **kw):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"short")
        return True

    with (
        patch(
            "gomoveshift.pipeline.day_pipeline.encode_day_master",
            return_value=master_path,
        ),
        patch(
            "gomoveshift.pipeline.day_pipeline.extract_and_concat_segments",
            side_effect=_fake_asm,
        ),
        patch(
            "gomoveshift.pipeline.day_pipeline.prepend_montage",
            side_effect=_fake_prep,
        ),
        patch(
            "gomoveshift.pipeline.day_pipeline.apply_ost",
            side_effect=_fake_ost,
        ) as mock_ost,
        patch(
            "gomoveshift.pipeline.day_pipeline._render_short_ffmpeg",
            side_effect=_fake_render,
        ),
    ):
        artifacts = day_pipeline.run_day_pipeline(
            day_batch=_make_day_batch("2026-02-20", 2),
            day_scores=_make_scores(2400),
            output_dir=output_base,
            location=location,
            config=None,
            lf_plans=[_make_lf_plan(1, [_make_seg(0, 100, 200)], [])],
            state=make_state(ost_audit_confirmed=True),
        )
    return artifacts, mock_ost


def _seed_master(output_base: Path) -> Path:
    master_path = output_base / "day_2026-02-20" / "day_2026-02-20_master.mp4"
    master_path.parent.mkdir(parents=True, exist_ok=True)
    master_path.write_bytes(b"master")
    return master_path


def test_unchanged_final_is_not_reburned_on_resume(tmp_path):
    """Resume after the shorts gate: an already-final LF with unchanged OST
    must be reused, not re-NVENC'd."""
    output_base = tmp_path / "out"
    master_path = _seed_master(output_base)

    # First run: LF is burned once.
    artifacts1, ost1 = _run(output_base, master_path, "NC, USA")
    assert artifacts1 is not None
    assert ost1.call_count == 1, "first run must burn the LF once"

    final_lf = output_base / "day_2026-02-20" / "regular_lfs" / "lf_01.mp4"
    assert final_lf.exists()

    # Resume: same inputs, final already on disk → must SKIP the re-burn.
    artifacts2, ost2 = _run(output_base, master_path, "NC, USA")
    assert artifacts2 is not None
    assert ost2.call_count == 0, "resume must NOT re-burn an unchanged already-final LF"
    assert any(Path(p).name == "lf_01.mp4" for p in artifacts2.regular_lf_paths), (
        "the fresh final must still flow into the day's artifacts"
    )


def test_changed_ost_text_forces_reburn_on_resume(tmp_path):
    """A materially changed LF (different burned OST text) must still
    re-burn even though a stale final is on disk."""
    output_base = tmp_path / "out"
    master_path = _seed_master(output_base)

    artifacts1, ost1 = _run(output_base, master_path, "NC, USA")
    assert artifacts1 is not None
    assert ost1.call_count == 1

    # Resume with a DIFFERENT location → different burned OST text.
    artifacts2, ost2 = _run(output_base, master_path, "SC, USA")
    assert artifacts2 is not None
    assert ost2.call_count == 1, "a changed OST text must re-burn the LF, not reuse the stale final"
