"""A planned long-form is never dropped silently, and its montage concat waits on
progress rather than a fixed clock.

Production, Galley Head (1 day, 2 LF plans, 25 h of work): LF 1's
``prepend_montage`` concat of a 38 GB ``lf_01_segments.mp4`` hit the fixed
``timeout=1800``; LF 2's assembled montage came out 16.7 s for a 15.0 s plan and
failed the 10% duration check. Both logged "skipping" and ``continue``d, the day
then rendered 14 shorts for ~1 h, logged ``regular_lfs=0``, and Phase D raised
``shorts_per_lf SPLIT-BRAIN``. The split-brain guard is right; the defect is that
a planned LF vanished with a warning and the run kept spending hours.
"""

from __future__ import annotations

import threading
import time as _time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from _state_helpers import make_state
from gomoveshift.pipeline import day_pipeline, montage_builder
from gomoveshift.pipeline import ffmpeg_runner as _runner
from gomoveshift.pipeline.short_planner import ShortRange
from test_day_pipeline import _make_day_batch, _make_lf_plan, _make_scores, _make_seg

DATE = "2026-02-20"
_SHORTS = [
    ShortRange(start_sec=10.0, end_sec=30.0, integrated_score=0.9, rank=1),
    ShortRange(start_sec=100.0, end_sec=120.0, integrated_score=0.85, rank=2),
]


# --- A. a planned LF that fails is a hard failure of the day -----------------


def _write(output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_bytes(b"x")
    return output_path


def _day(tmp_path, *, concat=None, prepend=None, runner=None):
    """Run a two-LF day with every heavy stage faked. ``concat`` / ``prepend``
    override the stage fakes; returns (result_or_exception, render_calls)."""
    out = tmp_path / "out"
    master = out / f"day_{DATE}" / f"day_{DATE}_master.mp4"
    master.parent.mkdir(parents=True, exist_ok=True)
    master.write_bytes(b"m")
    renders: list = []

    def _asm(master_path, segments, file_boundaries, output_path, work_dir, **kw):
        idx = int(output_path.name[3:5])
        if concat is not None and concat(idx) is None:
            return None
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        return _write(output_path)

    def _prepend(raw_lf_path, peaks, output_path, work_dir, **kw):
        idx = int(Path(output_path).name[3:5])
        if prepend is not None:
            cause = prepend(idx)
            if cause is not None:
                if kw.get("failures") is not None:
                    kw["failures"].append(cause)
                return None
        return _write(output_path)

    def _render(source_video, output_path, *a, **kw):
        renders.append(Path(output_path).name)
        return bool(_write(output_path))

    with (
        patch("gomoveshift.pipeline.day_pipeline.encode_day_master", return_value=master),
        patch("gomoveshift.pipeline.day_pipeline.extract_and_concat_segments", side_effect=_asm),
        patch("gomoveshift.pipeline.day_pipeline.prepend_montage", side_effect=_prepend),
        patch(
            "gomoveshift.pipeline.day_pipeline.apply_ost",
            side_effect=lambda input_path, output_path, location, layout, **kw: _write(output_path),
        ),
        patch("gomoveshift.pipeline.day_pipeline._render_short_ffmpeg", side_effect=_render),
        patch("gomoveshift.pipeline.day_pipeline.check_lf_work_space"),
        patch("gomoveshift.pipeline.day_pipeline.plan_shorts", return_value=_SHORTS),
    ):
        runner_args = dict(
            day_batch=_make_day_batch(DATE, 2, per_ep_duration=1200.0),
            day_scores=_make_scores(2400),
            output_dir=out,
            location="L",
            config=None,
            lf_plans=[
                _make_lf_plan(1, [_make_seg(0, 100, 200), _make_seg(1, 50, 150)], []),
                _make_lf_plan(2, [_make_seg(1, 200, 500)], []),
            ],
            state=make_state(ost_audit_confirmed=True),
        )
        if runner is not None:
            return runner(out, renders), renders
        try:
            return day_pipeline.run_day_pipeline(**runner_args), renders
        except RuntimeError as exc:
            return exc, renders


def test_control_both_lfs_ok_renders_shorts(tmp_path):
    """Control: the fakes do reach the shorts renderer, so 'never called' below means
    something."""
    result, renders = _day(tmp_path)
    assert len(result.regular_lf_paths) == 2
    assert renders, "the control day must render at least one short"


def test_montage_failure_on_lf2_raises_before_any_short(tmp_path):
    cause = "assembled montage failed duration verification — actual=16.7s expected=15.0s"
    result, renders = _day(tmp_path, prepend=lambda idx: cause if idx == 2 else None)

    assert isinstance(result, RuntimeError), result
    assert "LF 2" in str(result)
    assert DATE in str(result)
    assert "16.7s" in str(result)
    assert renders == [], "shorts must not render once a planned LF failed"


def test_concat_failure_on_lf2_raises_before_any_short(tmp_path):
    result, renders = _day(tmp_path, concat=lambda idx: None if idx == 2 else True)

    assert isinstance(result, RuntimeError), result
    assert "LF 2" in str(result)
    assert DATE in str(result)
    assert "concat" in str(result).lower()
    assert renders == []


def test_lf_with_no_segments_raises_instead_of_vanishing(tmp_path):
    """The third silent drop: a planned LF with no segments."""

    def _run(out, renders):
        with (
            patch(
                "gomoveshift.pipeline.day_pipeline.encode_day_master",
                return_value=out / f"day_{DATE}" / f"day_{DATE}_master.mp4",
            ),
            patch("gomoveshift.pipeline.day_pipeline.check_lf_work_space"),
        ):
            try:
                return day_pipeline.run_day_pipeline(
                    day_batch=_make_day_batch(DATE, 2, per_ep_duration=1200.0),
                    day_scores=_make_scores(2400),
                    output_dir=out,
                    location="L",
                    config=None,
                    lf_plans=[_make_lf_plan(1, [], [])],
                    state=make_state(ost_audit_confirmed=True),
                )
            except RuntimeError as exc:
                return exc

    result, _ = _day(tmp_path, runner=_run)
    assert isinstance(result, RuntimeError), result
    assert "LF 1" in str(result)


def test_failed_day_is_not_recorded_complete(tmp_path):
    """run_all_days must not hand a day with a failed planned LF to
    ``on_day_complete`` (which writes ``per_day_completed``)."""
    recorded: list = []

    def _run(out, renders):
        # Real run_all_days → real run_day_pipeline; same stage fakes as _day().
        batch = _make_day_batch(DATE, 2, per_ep_duration=1200.0)
        return day_pipeline.run_all_days(
            day_batches=[batch],
            drive_scores=_make_scores(2400),
            file_boundaries=[(0.0, 1200.0), (1200.0, 2400.0)],
            output_base=out,
            location="L",
            config=None,
            lf_plans_by_date={
                DATE: [
                    _make_lf_plan(1, [_make_seg(0, 100, 200)], []),
                    _make_lf_plan(2, [_make_seg(1, 200, 500)], []),
                ]
            },
            state=make_state(ost_audit_confirmed=True),
            on_day_complete=lambda artifacts, state: recorded.append(artifacts.date),
        )

    result, renders = _day(tmp_path, prepend=lambda idx: "boom" if idx == 2 else None, runner=_run)
    assert result == []
    assert recorded == []
    assert renders == []


# --- A2. a resume re-runs a day recorded with fewer LFs than planned --------


def _partition(tmp_path, *, final_lfs, planned):
    from gomoveshift.config import Config
    from gomoveshift.pipeline.drive_phases.phase_bc_day_first import PhaseBCDayFirst

    stages = tmp_path / "out" / "_stages"
    day_dir = stages / f"day_{DATE}"
    master = day_dir / f"day_{DATE}_master.mp4"
    (day_dir / "regular_lfs").mkdir(parents=True)
    (day_dir / "shorts").mkdir(parents=True)
    master.write_bytes(b"m")
    for n in range(1, final_lfs + 1):
        (day_dir / "regular_lfs" / f"lf_{n:02d}.mp4").write_bytes(b"x")
    (day_dir / "shorts" / "short_01.mp4").write_bytes(b"x")
    state = make_state(
        per_day_completed={DATE: {"master_path": str(master)}},
        allow_short_count=True,
    )
    batch = SimpleNamespace(date=DATE, total_duration=0.0, episodes=[])
    plans = {DATE: [_make_lf_plan(i, [_make_seg(0, 0, 10)], []) for i in range(1, planned + 1)]}
    phase = PhaseBCDayFirst(Config())
    _, skipped, remaining = phase._partition_batches_by_resume_state(
        state, [batch], stages, lf_plans_by_date=plans
    )
    return skipped, remaining


def test_resume_reruns_day_recorded_with_fewer_lfs_than_planned(tmp_path):
    """Poisoned state from the Galley run: 0 of 2 LFs on disk, day recorded complete."""
    skipped, remaining = _partition(tmp_path, final_lfs=0, planned=2)
    assert skipped == []
    assert [b.date for b in remaining] == [DATE]


def test_resume_reruns_day_with_one_of_two_lfs(tmp_path):
    skipped, remaining = _partition(tmp_path, final_lfs=1, planned=2)
    assert skipped == []
    assert [b.date for b in remaining] == [DATE]


def test_resume_still_skips_a_day_with_every_planned_lf(tmp_path):
    skipped, remaining = _partition(tmp_path, final_lfs=2, planned=2)
    assert [a.date for a in skipped] == [DATE]
    assert remaining == []


# --- B. the prepend concat waits on progress, not a fixed clock ----------------


class _Stderr:
    """Yields one progress line per ``release()``, then EOF once ``finish()``."""

    def __init__(self, lines_when_done=()):
        self._sem = threading.Semaphore(0)
        self._done = False
        self.emitted = 0

    def release(self):
        self._sem.release()

    def finish(self):
        self._done = True
        self._sem.release()

    def readline(self):
        self._sem.acquire()
        if self._done:
            return ""
        self.emitted += 1
        return f"frame={self.emitted} fps=25 size=1024kB time=00:00:{self.emitted % 60:02d}.00\n"


class _Stdout:
    def read(self, n=-1):
        return ""


class _Proc:
    """Popen stand-in on a manual clock: each ``poll()`` is one loop tick of ``tick`` s.

    ``progress=True`` emits one progress line per tick; ``exit_after`` ticks later it
    exits 0 (``None`` = never)."""

    def __init__(self, clock, *, tick, progress, exit_after, out=None):
        self.clock, self.tick, self.progress, self.exit_after = clock, tick, progress, exit_after
        self.stdout, self.stderr = _Stdout(), _Stderr()
        self.polls = 0
        self.returncode = None
        self.terminated = False
        self.out = out

    def poll(self):
        self.polls += 1
        self.clock.t += self.tick
        if self.out is not None:
            Path(self.out).write_bytes(b"PARTIAL")
        if self.exit_after is not None and self.polls > self.exit_after:
            self.returncode = 0
            self.stderr.finish()
            return 0
        if self.progress:
            self.stderr.release()
        return None

    def terminate(self):
        self.terminated = True
        self.returncode = -15
        self.stderr.finish()

    def kill(self):
        self.returncode = -9
        self.stderr.finish()

    def wait(self, timeout=None):
        return self.returncode


_real_sleep = _time.sleep


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _prepend(tmp_path, proc_factory):
    raw = tmp_path / "raw_lf.mp4"
    raw.write_bytes(b"RAW")
    work = tmp_path / "work"
    work.mkdir()
    out = tmp_path / "out.mp4"
    clock = _Clock()
    proc = proc_factory(clock, out)
    real = _runner.run_ffmpeg_encode

    def _with_clock(*a, **kw):
        return real(*a, time_source=clock, **kw)

    def _build(master_path, peaks, file_boundaries, output_path, work_dir, **kw):
        Path(output_path).write_bytes(b"MONTAGE")
        return Path(output_path)

    with (
        patch.object(montage_builder, "build_montage", side_effect=_build),
        patch.object(_runner, "run_ffmpeg_encode", side_effect=_with_clock),
        patch.object(_runner.subprocess, "Popen", return_value=proc),
        patch.object(_runner.time, "sleep", side_effect=lambda _s: _real_sleep(0.01)),
    ):
        result = montage_builder.prepend_montage(
            raw_lf_path=raw,
            peaks=[SimpleNamespace(source_file_index=0, start_time=0.0, end_time=40.0)],
            output_path=out,
            work_dir=work,
            master_path=tmp_path / "master.mp4",
            file_boundaries=[(0.0, 600.0)],
        )
    return result, out, proc, clock


def test_concat_that_keeps_reporting_progress_past_1800s_is_not_killed(tmp_path):
    result, out, proc, clock = _prepend(
        tmp_path,
        lambda clock, out: _Proc(clock, tick=300.0, progress=True, exit_after=8, out=out),
    )
    assert clock.t > 1800.0, "the fake run must outlast the old fixed clock"
    assert proc.terminated is False
    assert result == out
    assert out.exists()


def test_concat_that_stalls_is_killed_and_its_partial_output_removed(tmp_path):
    result, out, proc, clock = _prepend(
        tmp_path,
        lambda clock, out: _Proc(clock, tick=300.0, progress=False, exit_after=None, out=out),
    )
    assert proc.terminated is True
    assert result is None
    assert not out.exists(), "a truncated concat must not be left on disk"


def test_concat_uses_the_stall_runner_not_a_fixed_timeout(tmp_path):
    """The command goes through run_ffmpeg_encode with its own ceiling, never a 1800 s
    wall clock."""
    raw = tmp_path / "raw_lf.mp4"
    raw.write_bytes(b"RAW")
    work = tmp_path / "work"
    work.mkdir()
    out = tmp_path / "out.mp4"
    seen: dict = {}

    def _encode(cmd, output_path, expected_duration, description, **kw):
        seen.update(kw, output_path=output_path)
        Path(output_path).write_bytes(b"ok")
        return SimpleNamespace(success=True, killed_reason=None, stderr_tail="")

    def _build(master_path, peaks, file_boundaries, output_path, work_dir, **kw):
        Path(output_path).write_bytes(b"MONTAGE")
        return Path(output_path)

    quick = MagicMock()
    with (
        patch.object(montage_builder, "build_montage", side_effect=_build),
        patch.object(_runner, "run_ffmpeg_encode", side_effect=_encode),
        patch.object(_runner, "run_ffmpeg_quick", quick),
    ):
        result = montage_builder.prepend_montage(
            raw_lf_path=raw,
            peaks=[SimpleNamespace(source_file_index=0, start_time=0.0, end_time=40.0)],
            output_path=out,
            work_dir=work,
            master_path=tmp_path / "master.mp4",
            file_boundaries=[(0.0, 600.0)],
        )
    assert result == out
    quick.assert_not_called()
    import inspect

    default = inspect.signature(_runner.run_ffmpeg_encode).parameters["max_timeout"].default
    assert seen.get("max_timeout", default) > 1800


# --- C. montage clip start: never a `-c copy` that snaps back a GOP ------------


def test_keyframes_exist_but_none_fit_reencodes_instead_of_stream_copy(tmp_path):
    """The 16.7 s montage: the centred start (122.5 s) has no keyframe in
    [122.5, 135.0], so it stayed unaligned and `-c copy` snapped back to the prior
    keyframe, replaying up to one GOP per clip (+1.7 s over 15.0 s)."""
    master = tmp_path / "master.mp4"
    master.write_bytes(b"fake")
    peak = SimpleNamespace(source_file_index=0, start_time=110.0, end_time=140.0)
    copy, reenc = MagicMock(return_value=True), MagicMock(return_value=True)
    with (
        patch.object(montage_builder, "probe_keyframes", return_value=[0.0, 100.0, 200.0]),
        patch.object(montage_builder, "stream_copy_range", copy),
        patch.object(montage_builder, "reencode_range", reenc),
    ):
        ok = montage_builder._trim_peak_to(
            master, peak, [(0.0, 600.0)], tmp_path / "clip.mp4", clip_duration=5.0
        )
    assert ok is True
    copy.assert_not_called()
    reenc.assert_called_once()
    assert reenc.call_args.args[1:3] == pytest.approx((122.5, 127.5))


def test_keyframe_aligned_start_still_stream_copies(tmp_path):
    master = tmp_path / "master.mp4"
    master.write_bytes(b"fake")
    peak = SimpleNamespace(source_file_index=0, start_time=110.0, end_time=140.0)
    copy, reenc = MagicMock(return_value=True), MagicMock(return_value=True)
    with (
        patch.object(montage_builder, "probe_keyframes", return_value=[0.0, 124.0, 200.0]),
        patch.object(montage_builder, "stream_copy_range", copy),
        patch.object(montage_builder, "reencode_range", reenc),
    ):
        assert montage_builder._trim_peak_to(
            master, peak, [(0.0, 600.0)], tmp_path / "clip.mp4", clip_duration=5.0
        )
    reenc.assert_not_called()
    assert copy.call_args.args[1] == pytest.approx(124.0)


def test_duration_validator_tolerance_is_unchanged():
    """16.7 s against a 15.0 s plan must still FAIL the 10% gate (the fix is upstream)."""
    from gomoveshift.pipeline.pipeline_validator import PipelineValidator

    with patch(
        "gomoveshift.pipeline.validation.ffprobe_client.FFprobeClient.probe_duration",
        return_value=16.7,
    ):
        result = PipelineValidator().validate_montage_output(Path("m.mp4"), target_duration=15.0)
    assert not result.passed
