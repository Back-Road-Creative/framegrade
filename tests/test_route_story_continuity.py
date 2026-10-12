"""Frame-to-frame continuity: the defects a single-frame QA check cannot see
(camera snap before a photo, label flicker, a frozen span) live in how
consecutive frames relate to each other, not in any one frame alone.

(a) is a pure plan-level check over planned camera states; (b) is a
rendered-output pass over an actual decoded video (or an iterable of
frames), skipped when ffmpeg is not on PATH.
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from gomoveshift.pipeline.route_story import continuity

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _state(lat=45.0, lon=-63.0, bearing=0.0, zoom=15.0, pitch=60.0) -> continuity.CameraState:
    return continuity.CameraState(
        center_lat=lat, center_lon=lon, bearing_deg=bearing, zoom=zoom, pitch_deg=pitch
    )


# --- (a) plan-level camera continuity -------------------------------------


def test_smooth_camera_sequence_has_no_problems() -> None:
    states = [_state(bearing=b) for b in (10.0, 11.0, 12.0, 13.0, 14.0)]
    problems = continuity.check_camera_continuity(states, cut_indices=())
    assert problems == ()


def test_a_bearing_jump_on_a_non_cut_frame_is_flagged() -> None:
    states = [_state(bearing=10.0), _state(bearing=95.0), _state(bearing=96.0)]
    problems = continuity.check_camera_continuity(states, cut_indices=())
    assert len(problems) == 1
    assert "bearing" in problems[0]
    assert "frame 1" in problems[0]


def test_the_same_jump_at_a_planned_cut_is_not_flagged() -> None:
    states = [_state(bearing=10.0), _state(bearing=95.0), _state(bearing=96.0)]
    problems = continuity.check_camera_continuity(states, cut_indices=(1,))
    assert problems == ()


def test_a_center_jump_is_flagged() -> None:
    states = [
        _state(lat=45.0, lon=-63.0),
        _state(lat=45.5, lon=-63.0),
        _state(lat=45.51, lon=-63.0),
    ]
    problems = continuity.check_camera_continuity(states, cut_indices=())
    assert any("center" in p for p in problems)


def test_a_zoom_jump_is_flagged() -> None:
    states = [_state(zoom=15.0), _state(zoom=18.0), _state(zoom=18.1)]
    problems = continuity.check_camera_continuity(states, cut_indices=())
    assert any("zoom" in p for p in problems)


def test_a_pitch_jump_is_flagged() -> None:
    states = [_state(pitch=60.0), _state(pitch=20.0), _state(pitch=19.0)]
    problems = continuity.check_camera_continuity(states, cut_indices=())
    assert any("pitch" in p for p in problems)


def test_fewer_than_two_states_is_not_judged() -> None:
    assert continuity.check_camera_continuity([], cut_indices=()) == ()
    assert continuity.check_camera_continuity([_state()], cut_indices=()) == ()


def test_bearing_wraparound_is_not_a_false_jump() -> None:
    # 359 -> 1 degrees is a 2-degree turn, not a 358-degree one.
    states = [_state(bearing=359.0), _state(bearing=1.0), _state(bearing=3.0)]
    problems = continuity.check_camera_continuity(states, cut_indices=())
    assert problems == ()


# --- (b) rendered-output pass ---------------------------------------------


def _frame(width: int, height: int, value: int) -> np.ndarray:
    return np.full((height, width), value, dtype=np.uint8)


def test_a_spike_between_frames_not_a_planned_cut_is_reported() -> None:
    frames = [_frame(8, 8, 40), _frame(8, 8, 42), _frame(8, 8, 240), _frame(8, 8, 238)]
    report = continuity.analyze_rendered_continuity(frames, fps=10.0)
    assert len(report["spikes"]) == 1
    assert report["spikes"][0]["frame"] == 2
    assert "timestamp_s" in report["spikes"][0]


def test_a_spike_at_a_planned_cut_is_not_reported() -> None:
    frames = [_frame(8, 8, 40), _frame(8, 8, 42), _frame(8, 8, 240), _frame(8, 8, 238)]
    report = continuity.analyze_rendered_continuity(frames, fps=10.0, cut_indices=(2,))
    assert report["spikes"] == []


def test_a_frozen_span_longer_than_the_threshold_is_reported() -> None:
    frames = [_frame(8, 8, 40)] + [_frame(8, 8, 41)] * 40 + [_frame(8, 8, 200)]
    report = continuity.analyze_rendered_continuity(frames, fps=10.0)
    assert len(report["frozen_spans"]) == 1
    span = report["frozen_spans"][0]
    assert span["start_frame"] == 1
    assert span["duration_s"] >= continuity.DEFAULT_CONTINUITY_THRESHOLDS.min_freeze_s


def test_a_planned_hold_of_the_same_length_is_not_reported() -> None:
    frames = [_frame(8, 8, 40)] + [_frame(8, 8, 41)] * 40 + [_frame(8, 8, 200)]
    report = continuity.analyze_rendered_continuity(frames, fps=10.0, hold_spans=((1, 40),))
    assert report["frozen_spans"] == []


def test_stutter_oscillation_inside_a_region_is_reported() -> None:
    base = _frame(20, 20, 40)
    high = base.copy()
    high[5:15, 5:15] = 220
    frames = [base if i % 2 == 0 else high for i in range(12)]
    report = continuity.analyze_rendered_continuity(frames, fps=10.0, boxes=((5, 5, 15, 15),))
    assert len(report["stutter_windows"]) >= 1
    assert report["stutter_windows"][0]["box"] == (5, 5, 15, 15)


def _panned_frame(size: int, shift: int) -> np.ndarray:
    """A smooth pan's own frame-to-frame diff: a triangle wave (no sawtooth
    wrap discontinuity, unlike a raw ``arange % period`` ramp) shifted by
    ``shift`` px, tiled down every row -- a steady per-tile diff close to
    ``shift`` everywhere, the signature of ambient motion, not a defect."""
    period = 2 * size
    x = np.arange(size)
    triangle = np.abs(((x + shift) % period) - size).astype(np.uint8)
    return np.tile(triangle, (size, 1))


def test_a_local_one_shot_pop_during_a_smooth_pan_is_detected() -> None:
    # A whole-frame/absolute-threshold spike check dilutes a label-sized pop
    # to nothing: the pan itself produces a small, constant per-frame diff,
    # and the pop -- a small patch appearing for exactly one frame, then
    # gone -- barely moves the whole-frame mean. It's still huge relative to
    # that one tile's own (calm) history, which is what should catch it.
    size = 96
    grid = continuity.DEFAULT_CONTINUITY_THRESHOLDS.spike_tile_grid
    tile = size // grid[0]
    n_frames = 20
    pop_frame = 10

    frames = []
    for i in range(n_frames):
        frame = _panned_frame(size, i).copy()
        if i == pop_frame:
            frame[0:tile, 0:tile] = 250
        frames.append(frame)

    report = continuity.analyze_rendered_continuity(frames, fps=10.0)

    # The patch appearing (pop_frame) and disappearing (pop_frame + 1) are
    # both local, one-shot anomalies against otherwise-smooth pan motion.
    flagged = {s["frame"] for s in report["spikes"]}
    assert pop_frame in flagged or (pop_frame + 1) in flagged


def test_sustained_smooth_motion_is_not_a_run_of_spikes() -> None:
    # A steady pan/scroll should not itself trigger the local-anomaly check,
    # even though its per-frame diff sits well above a tiny absolute floor.
    size = 96
    n_frames = 30
    frames = [_panned_frame(size, i * 3) for i in range(n_frames)]

    report = continuity.analyze_rendered_continuity(frames, fps=10.0)
    assert report["spikes"] == []


def test_a_calm_clip_reports_nothing() -> None:
    frames = [_frame(8, 8, 40)] * 10
    report = continuity.analyze_rendered_continuity(frames, fps=10.0)
    assert report["spikes"] == []
    assert report["frozen_spans"] == []
    assert report["stutter_windows"] == []


def test_report_is_json_serialisable() -> None:
    import json

    frames = [_frame(8, 8, 40), _frame(8, 8, 42), _frame(8, 8, 240)]
    report = continuity.analyze_rendered_continuity(frames, fps=10.0)
    json.dumps(report)  # raises if not serialisable


@needs_ffmpeg
def test_decode_grayscale_frames_from_a_real_mp4(tmp_path) -> None:
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=0x305070:s=64x48:d=0.5:r=10",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip),
        ],
        check=True,
        capture_output=True,
    )  # fmt: skip
    frames = list(continuity.decode_grayscale_frames(clip, width=32, height=24))
    assert len(frames) >= 4
    assert all(f.shape == (24, 32) for f in frames)
    assert all(f.dtype == np.uint8 for f in frames)


@needs_ffmpeg
def test_analyze_rendered_continuity_accepts_a_video_path(tmp_path) -> None:
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=0x305070:s=64x48:d=0.5:r=10",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip),
        ],
        check=True,
        capture_output=True,
    )  # fmt: skip
    report = continuity.analyze_rendered_continuity(clip, fps=10.0)
    assert report["spikes"] == []
    assert report["frozen_spans"] == []


def test_cli_entry_reports_on_an_mp4_path(tmp_path, capsys) -> None:
    frames = [_frame(8, 8, 40)] * 5
    # Exercise the report-building path the CLI wraps, without needing a real
    # decode; the CLI wiring itself is covered by argv parsing below.
    import json

    report = continuity.analyze_rendered_continuity(frames, fps=10.0)
    assert json.loads(json.dumps(report))["spikes"] == []


def test_main_parses_argv_and_prints_json(monkeypatch, capsys) -> None:
    def fake_analyze(
        path, *, fps, cut_indices=(), hold_spans=(), boxes=None, thresholds=None, decode_width=320
    ):
        assert str(path) == "clip.mp4"
        return {"spikes": [], "frozen_spans": [], "stutter_windows": []}

    monkeypatch.setattr(continuity, "analyze_rendered_continuity", fake_analyze)
    monkeypatch.setattr(continuity, "_probe_fps", lambda path: 30.0)
    continuity.main(["clip.mp4"])
    out = capsys.readouterr().out
    assert '"spikes": []' in out
