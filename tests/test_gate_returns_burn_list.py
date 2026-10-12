"""The OST audit gate and the LF burn loop must share ONE resolved text.

``run_day_pipeline``'s per-LF OST audit gate resolves each LF's burn text
from ``state`` (state wins over the ``per_lf_ost_text``/``per_lf_locations``
call kwargs — see ``_ost_audit_state`` construction). The burn loop that
follows must burn EXACTLY that gate-resolved text, never re-derive its own
answer from the raw call kwargs alone — otherwise an operator can review one
string in the audit sidecar and have a DIFFERENT string burned onto the
irreversible LF pixels.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from gomoveshift.pipeline.day_grouper import DayBatch
from _state_helpers import make_state

NUM_COLUMNS = 9
COL_TIMESTAMP = 0
COL_COMPOSITE = 1
COL_MOTION_ENERGY = 2


def _make_scores(duration_sec: int, score: float = 0.5, motion: float = 0.05) -> np.ndarray:
    arr = np.zeros((duration_sec, NUM_COLUMNS), dtype=np.float64)
    arr[:, COL_TIMESTAMP] = np.arange(duration_sec, dtype=np.float64)
    arr[:, COL_COMPOSITE] = score
    arr[:, COL_MOTION_ENERGY] = motion
    return arr


def _make_day_batch(date: str, episode_count: int, per_ep_duration: float = 600.0):
    episodes = [
        SimpleNamespace(
            index=i + 1,
            concat_path=Path(f"/fake/ep_{i:03d}.mp4"),
            total_duration=per_ep_duration,
            date=date,
            creation_time=None,
        )
        for i in range(episode_count)
    ]
    return DayBatch(date=date, episodes=episodes)


def _make_lf_plan(index, segments, montage_peaks):
    return SimpleNamespace(index=index, segments=segments, montage_peaks=montage_peaks)


def _make_seg(ep_idx, start, end, lat=0.0, lon=0.0):
    return SimpleNamespace(
        source_file_index=ep_idx,
        start_time=float(start),
        end_time=float(end),
        duration=float(end - start),
        gps_center_lat=lat,
        gps_center_lon=lon,
    )


def test_lf_burn_loop_burns_the_gate_audited_text(tmp_path):
    """The gate audits ``state``'s per-LF text (state wins over the call
    kwarg); the burn loop must burn that SAME text, not re-resolve its own
    answer from the raw ``per_lf_ost_text`` kwarg alone."""
    from gomoveshift.pipeline import day_pipeline

    master_path = tmp_path / "out" / "day_2026-02-20" / "day_2026-02-20_master.mp4"
    master_path.parent.mkdir(parents=True, exist_ok=True)
    master_path.write_bytes(b"fake master")

    burned_locations = []

    def _fake_ost(input_path, output_path, location, layout, **kwargs):
        burned_locations.append(location)
        output_path.write_bytes(b"fake ost")
        return output_path

    def _fake_render(source_video, output_path, start_time, duration, location, **kwargs):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"fake render short")
        return True

    day_batch = _make_day_batch("2026-02-20", 3, per_ep_duration=1200.0)
    day_scores = _make_scores(3600, score=0.6)
    lf_plans = [_make_lf_plan(1, [_make_seg(0, 100, 200)], [])]

    # State supplies the AUDITED text (gate prefers state over the kwarg —
    # ``_ost_audit_state`` in day_pipeline.run_day_pipeline). The kwarg
    # below supplies a DIFFERENT string so a gate/burn split is observable.
    state = make_state(
        ost_audit_confirmed=True,
        per_lf_ost_text={1: "Audited Route, Kerry"},
    )

    with (
        patch(
            "gomoveshift.pipeline.day_pipeline.encode_day_master",
            return_value=master_path,
        ),
        patch("gomoveshift.pipeline.day_pipeline.apply_ost", side_effect=_fake_ost),
        patch(
            "gomoveshift.pipeline.day_pipeline._render_short_ffmpeg",
            side_effect=_fake_render,
        ),
        patch(
            "gomoveshift.pipeline.day_pipeline.extract_and_concat_segments",
            side_effect=lambda master_path, segments, file_boundaries, output_path, work_dir, **kw: (
                output_path.parent.mkdir(parents=True, exist_ok=True)
                or output_path.write_bytes(b"x")
                or output_path
            ),
        ),
        patch(
            "gomoveshift.pipeline.day_pipeline.prepend_montage",
            side_effect=lambda raw_lf_path, peaks, output_path, work_dir, **kw: (
                output_path.parent.mkdir(parents=True, exist_ok=True)
                or output_path.write_bytes(b"x")
                or output_path
            ),
        ),
    ):
        day_pipeline.run_day_pipeline(
            day_batch=day_batch,
            day_scores=day_scores,
            output_dir=tmp_path / "out",
            location="Drive-wide Fallback, SC",
            config=None,
            lf_plans=lf_plans,
            per_lf_ost_text={1: "Stale Kwarg Text, Not Audited"},
            state=state,
        )

    assert burned_locations, "apply_ost was never called for the LF"
    assert burned_locations[0] == "Audited Route, Kerry", (
        "burn loop must use the SAME text the OST audit gate resolved "
        f"(state-preferring), got {burned_locations[0]!r}"
    )
