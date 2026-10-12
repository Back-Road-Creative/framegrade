"""Frame-to-frame continuity checks: the defects a single-frame QA check
cannot see at all.

``frame_qa.py`` judges one rendered frame in isolation -- a void pixel, an
unloaded tile, a marker that lags the travelled line. Every defect an
operator has actually caught by eye (a camera snap right before a photo, a
label flickering in and out, a frozen span sitting in the middle of a leg)
is a relationship BETWEEN consecutive frames, which no single-frame check
can see no matter how many of them run.

Two independent passes:

(a) ``check_camera_continuity`` -- pure, over the PLANNED per-frame camera
    states (before any pixel is rendered): flags a jump larger than a
    threshold on a frame that isn't a planned cut. Seconds to run, no I/O.
(b) ``analyze_rendered_continuity`` -- over an actual rendered video (or an
    iterable of frames): decodes a downscaled grayscale copy via ffmpeg,
    diffs consecutive frames, and reports spikes, frozen spans and
    "stutter" (short-period oscillation, e.g. flicker inside a label
    region) that aren't accounted for by the plan.

(a) is wired into ``scene_renderer.SceneMapRenderer.render_leg``, which calls
it over the leg's own planned camera states before any capture worker
starts. (b) is not wired into the render loop -- there is no rendered output
to diff until the capture is already done, so ``main`` runs it directly
against an already-rendered mp4 (no render, no scene browser) instead.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.ndimage import median_filter

# --- (a) plan-level camera continuity --------------------------------------


@dataclass(frozen=True)
class CameraState:
    """One planned frame's camera pose -- the shape ``check_camera_continuity``
    accepts, independent of whichever renderer/follow-camera produced it."""

    center_lat: float
    center_lon: float
    bearing_deg: float
    zoom: float
    pitch_deg: float


@dataclass(frozen=True)
class ContinuityThresholds:
    # Degrees of great-circle arc between consecutive frames' centers.
    # ~0.0015 deg is roughly 165m at mid-latitudes -- a follow camera's
    # own speed/accel caps (planner.DEFAULT_MOTION_LIMITS) keep real
    # per-frame travel far below this; this catches an actual snap, not a
    # fast but continuous pan.
    max_center_jump_deg: float = 0.05
    max_bearing_jump_deg: float = 25.0
    max_zoom_jump: float = 1.5
    max_pitch_jump_deg: float = 20.0

    # (b) rendered-output pass.
    # A whole-frame mean diff dilutes a label-sized pop to nothing, and an
    # absolute floor flags ordinary sustained motion as a run of "spikes".
    # Instead: split each frame into a grid, take each tile's own mean
    # abs diff, and flag a tile whose diff is a large multiple of ITS OWN
    # rolling-median baseline -- selective for a one-shot local change
    # (a map label popping in/out) even while ambient motion (a pan) keeps
    # every tile's absolute diff magnitude similar to its neighbours'.
    spike_tile_grid: tuple[int, int] = (12, 12)  # (rows, cols) on the decoded frame
    # Rolling-median window, in frames, each tile's baseline is measured
    # over -- long enough to span a planned cut/transition without being
    # dominated by the anomaly itself, short enough to track a genuine
    # change in ambient motion (e.g. a pan that speeds up).
    spike_baseline_window_frames: int = 91
    # A tile's diff must exceed this multiple of its own rolling-median
    # baseline to be a spike.
    spike_relative_factor: float = 3.0
    # ...and clear this absolute floor, so a tile that's normally almost
    # perfectly still doesn't turn a tiny diff into a huge, meaningless
    # ratio.
    spike_diff_floor: float = 4.0
    # A run of frames whose diff to the previous frame never exceeds this
    # counts as frozen.
    freeze_diff_floor: float = 2.0
    # A frozen run shorter than this many seconds is normal (a still beat
    # settling, a dissolve hold) and is not reported.
    min_freeze_s: float = 1.0
    # A "stutter" window: this many consecutive frames whose diff alternates
    # above/below this floor (oscillation), inside a checked region.
    stutter_window_frames: int = 6
    stutter_diff_floor: float = 15.0


DEFAULT_CONTINUITY_THRESHOLDS = ContinuityThresholds()


class CameraContinuityError(RuntimeError):
    """The PLANNED per-frame camera states for a map leg contain a jump
    ``check_camera_continuity`` flagged -- raised before any frame of that
    leg is captured, never after."""

    def __init__(self, issues: Sequence[str]):
        self.issues = tuple(issues)
        super().__init__("camera continuity check failed: " + "; ".join(self.issues))


def _angle_delta_deg(a: float, b: float) -> float:
    """Shortest signed distance from ``a`` to ``b`` in degrees, wrapped at 360
    -- 359 to 1 is a 2 degree turn, never a 358 degree one."""
    return (b - a + 180.0) % 360.0 - 180.0


def _great_circle_deg(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    """Angular separation in degrees -- cheap and threshold-scale only; this
    never needs metres, only "did the center jump"."""
    return math.hypot(lat_b - lat_a, lon_b - lon_a)


def check_camera_continuity(
    states: Sequence[CameraState],
    cut_indices: Iterable[int] = (),
    thresholds: ContinuityThresholds = DEFAULT_CONTINUITY_THRESHOLDS,
) -> tuple[str, ...]:
    """Flag a jump in center/bearing/zoom/pitch between consecutive planned
    frames larger than ``thresholds``, unless the LATER frame is a planned
    cut (``cut_indices``) -- a real cut is supposed to jump; anywhere else,
    that jump is the camera snapping the operator has reported by eye.

    Pure: no I/O, no rendering. Fewer than two states has nothing to compare
    and is not judged.
    """
    cuts = set(cut_indices)
    problems: list[str] = []
    for index in range(1, len(states)):
        if index in cuts:
            continue
        before, after = states[index - 1], states[index]

        center_jump = _great_circle_deg(
            before.center_lat, before.center_lon, after.center_lat, after.center_lon
        )
        if center_jump > thresholds.max_center_jump_deg:
            problems.append(
                f"frame {index}: center jumped {center_jump:.4f} deg "
                f"(max {thresholds.max_center_jump_deg:.4f} deg)"
            )

        bearing_jump = abs(_angle_delta_deg(before.bearing_deg, after.bearing_deg))
        if bearing_jump > thresholds.max_bearing_jump_deg:
            problems.append(
                f"frame {index}: bearing jumped {bearing_jump:.1f} deg "
                f"(max {thresholds.max_bearing_jump_deg:.1f} deg)"
            )

        zoom_jump = abs(after.zoom - before.zoom)
        if zoom_jump > thresholds.max_zoom_jump:
            problems.append(
                f"frame {index}: zoom jumped {zoom_jump:.2f} (max {thresholds.max_zoom_jump:.2f})"
            )

        pitch_jump = abs(after.pitch_deg - before.pitch_deg)
        if pitch_jump > thresholds.max_pitch_jump_deg:
            problems.append(
                f"frame {index}: pitch jumped {pitch_jump:.1f} deg "
                f"(max {thresholds.max_pitch_jump_deg:.1f} deg)"
            )

    return tuple(problems)


# --- (b) rendered-output pass -----------------------------------------------

Box = tuple[int, int, int, int]  # (x0, y0, x1, y1), grayscale-frame pixel coords


def decode_grayscale_frames(
    video_path: Path | str, *, width: int = 320, height: int | None = None
) -> Iterable[np.ndarray]:
    """Yield one HxW ``uint8`` grayscale frame per decoded frame of
    ``video_path``, downscaled (aspect ignored -- only relative diffs
    between frames matter here, not true geometry) via an ffmpeg subprocess.

    ``height`` defaults to ``width * 9 // 16`` (rounded to an even number,
    which ``scale`` requires) when not given.
    """
    if height is None:
        height = (width * 9 // 16) // 2 * 2
    argv = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"scale={width}:{height}:flags=fast_bilinear,format=gray",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    frame_bytes = width * height
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        if proc.stdout is None:
            raise RuntimeError("ffmpeg subprocess has no stdout pipe")
        while True:
            chunk = proc.stdout.read(frame_bytes)
            if len(chunk) < frame_bytes:
                break
            yield np.frombuffer(chunk, dtype=np.uint8).reshape(height, width)
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        stderr = proc.stderr.read() if proc.stderr is not None else b""
        returncode = proc.wait()
        if returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed decoding {video_path} for continuity analysis "
                f"(exit {returncode}): {stderr.decode('utf-8', errors='replace')}"
            )


def _probe_fps(video_path: Path | str) -> float:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    num, _, den = out.partition("/")
    return float(num) / float(den or 1)


def _diff_signal(frames: Sequence[np.ndarray]) -> list[float]:
    return [
        float(np.abs(frames[i].astype(np.int16) - frames[i - 1].astype(np.int16)).mean())
        for i in range(1, len(frames))
    ]


def _signed_mean_signal(frames: Sequence[np.ndarray]) -> list[float]:
    """Mean pixel value per frame -- used to catch a value oscillating between
    two states (a flickering label), which a magnitude-only diff signal can't
    tell apart from a steady, one-directional change of the same size."""
    return [float(frame.astype(np.float64).mean()) for frame in frames]


def _tile_diffs(frames: Sequence[np.ndarray], grid: tuple[int, int]) -> np.ndarray:
    """Per-consecutive-pair, per-tile mean abs diff: shape ``(n-1, gy, gx)``.

    The requested grid is clamped to the frame's own size (a grid finer
    than the frame is meaningless, and would floor-divide tile size to
    zero) -- real decodes are always far larger than the grid, so this
    only matters for tiny synthetic test frames.
    """
    if len(frames) < 2:
        return np.zeros((0, 1, 1))
    arr = np.stack([np.asarray(f, dtype=np.int16) for f in frames])
    diffs = np.abs(arr[1:] - arr[:-1]).astype(np.float64)
    n, h, w = diffs.shape
    gy = max(1, min(grid[0], h))
    gx = max(1, min(grid[1], w))
    ty, tx = h // gy, w // gx
    cropped = diffs[:, : ty * gy, : tx * gx]
    return cropped.reshape(n, gy, ty, gx, tx).mean(axis=(2, 4))


def _tile_spike_signal(
    tiles: np.ndarray, thresholds: ContinuityThresholds
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame (peak tile diff, peak tile diff / that tile's OWN rolling
    -median baseline) -- the ratio is what selects a one-shot local pop out
    of ambient motion that raises every tile's absolute diff together."""
    n, gy, gx = tiles.shape
    if n == 0:
        return np.zeros(0), np.zeros(0)
    flat = tiles.reshape(n, gy * gx).T  # (tiles, frames)
    window = max(1, thresholds.spike_baseline_window_frames)
    baseline = median_filter(flat, size=(1, window), mode="nearest")
    baseline = np.maximum(baseline, thresholds.spike_diff_floor)
    ratio = flat / baseline
    return flat.max(axis=0), ratio.max(axis=0)


def _spikes(
    frames: Sequence[np.ndarray], cuts: set[int], thresholds: ContinuityThresholds, fps: float
) -> list[dict]:
    tiles = _tile_diffs(frames, thresholds.spike_tile_grid)
    peak_diff, peak_ratio = _tile_spike_signal(tiles, thresholds)
    spikes = []
    for i in range(len(peak_ratio)):
        frame = i + 1
        if frame in cuts:
            continue
        if peak_ratio[i] > thresholds.spike_relative_factor:
            spikes.append(
                {
                    "frame": frame,
                    "diff": float(peak_diff[i]),
                    "ratio": float(peak_ratio[i]),
                    "timestamp_s": frame / fps,
                }
            )
    return spikes


def _frozen_spans(
    diffs: Sequence[float],
    holds: Sequence[tuple[int, int]],
    thresholds: ContinuityThresholds,
    fps: float,
) -> list[dict]:
    held: set[int] = set()
    for start, length in holds:
        held.update(range(start, start + length))

    spans = []
    run_start: int | None = None
    for i, diff in enumerate(diffs):
        frame = i + 1
        frozen = diff <= thresholds.freeze_diff_floor
        if frozen and run_start is None:
            run_start = frame
        elif not frozen and run_start is not None:
            spans.append((run_start, frame - 1))
            run_start = None
    if run_start is not None:
        spans.append((run_start, len(diffs)))

    reported = []
    for start, end in spans:
        length = end - start + 1
        duration_s = length / fps
        if duration_s < thresholds.min_freeze_s:
            continue
        if held.issuperset(range(start, end + 1)):
            continue
        reported.append(
            {
                "start_frame": start,
                "end_frame": end,
                "duration_s": duration_s,
            }
        )
    return reported


def _stutter_windows(
    box_means: dict[Box | None, list[float]],
    thresholds: ContinuityThresholds,
    fps: float,
) -> list[dict]:
    """A "stutter": the region's own mean value bounces back and forth
    between two levels for ``stutter_window_frames`` in a row -- the
    label-flicker signature. Detected on the SIGNED step-to-step delta's
    sign, not its magnitude: a flicker's magnitude-only diff signal (A->B,
    B->A) is constant-high, indistinguishable from a real, steady change of
    the same size; the sign of the delta flips every step only for genuine
    back-and-forth motion.
    """
    windows = []
    n = thresholds.stutter_window_frames
    for box, means in box_means.items():
        deltas = [means[i] - means[i - 1] for i in range(1, len(means))]
        for start in range(0, max(0, len(deltas) - n + 1)):
            window = deltas[start : start + n]
            if any(abs(d) <= thresholds.stutter_diff_floor for d in window):
                continue
            signs = [d > 0 for d in window]
            if all(signs[i] != signs[i + 1] for i in range(len(signs) - 1)):
                windows.append(
                    {
                        "start_frame": start + 1,
                        "end_frame": start + n,
                        "timestamp_s": (start + 1) / fps,
                        "box": box,
                    }
                )
    return windows


def analyze_rendered_continuity(
    frames_or_path: Path | str | Iterable[np.ndarray],
    *,
    fps: float,
    cut_indices: Iterable[int] = (),
    hold_spans: Sequence[tuple[int, int]] = (),
    boxes: Sequence[Box] | None = None,
    thresholds: ContinuityThresholds = DEFAULT_CONTINUITY_THRESHOLDS,
    decode_width: int = 320,
) -> dict:
    """Consecutive-frame diff over a rendered video or an iterable of frames.

    ``frames_or_path`` is either a path to a video (decoded via
    ``decode_grayscale_frames``) or an already-decoded iterable of HxW
    grayscale frames (as tests use, for a deterministic synthetic signal).

    ``cut_indices`` are planned-cut frame indices (spikes there are
    expected, not reported); ``hold_spans`` are planned ``(start_frame,
    length)`` holds (freezes there are expected). ``boxes`` restricts the
    stutter check to those ``(x0, y0, x1, y1)`` regions (e.g. label boxes);
    with none given, stutter runs over the whole-frame diff signal.

    Returns a JSON-serialisable dict: ``spikes``, ``frozen_spans``,
    ``stutter_windows``, each a list of records with frame indices and
    timestamps.
    """
    if isinstance(frames_or_path, (str, Path)):
        frames = list(decode_grayscale_frames(frames_or_path, width=decode_width))
    else:
        frames = list(frames_or_path)

    diffs = _diff_signal(frames)
    cuts = set(cut_indices)

    box_means: dict[Box | None, list[float]] = {}
    if boxes:
        for box in boxes:
            x0, y0, x1, y1 = box
            crops = [frame[y0:y1, x0:x1] for frame in frames]
            box_means[box] = _signed_mean_signal(crops)
    else:
        box_means[None] = _signed_mean_signal(frames)

    return {
        "spikes": _spikes(frames, cuts, thresholds, fps),
        "frozen_spans": _frozen_spans(diffs, hold_spans, thresholds, fps),
        "stutter_windows": _stutter_windows(box_means, thresholds, fps),
    }


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry: report frame-to-frame continuity defects on an already
    -rendered mp4. Never renders anything itself.

        python -m gomoveshift.pipeline.route_story.continuity path/to/film.mp4

    With no ``--cut``/``--hold`` given, every spike/freeze/stutter window in
    the file is reported (there is no planned cut list to compare against).
    """
    parser = argparse.ArgumentParser(
        description="Report frame-to-frame continuity defects in a rendered video."
    )
    parser.add_argument("video", help="path to the rendered mp4")
    parser.add_argument(
        "--cut",
        type=int,
        action="append",
        default=[],
        dest="cuts",
        help="a planned-cut frame index",
    )
    parser.add_argument(
        "--hold",
        type=str,
        action="append",
        default=[],
        dest="holds",
        help="a planned hold as START:LENGTH (frames)",
    )
    parser.add_argument("--width", type=int, default=320, help="decode width in px")
    args = parser.parse_args(argv)

    holds = tuple(tuple(int(part) for part in hold.split(":", 1)) for hold in args.holds)
    fps = _probe_fps(args.video)
    report = analyze_rendered_continuity(
        args.video, fps=fps, cut_indices=args.cuts, hold_spans=holds, decode_width=args.width
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
