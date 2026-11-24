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

    def __init__(self, lines_when_done=(), *, frozen=False):
        self._sem = threading.Semaphore(0)
        self._done = False
        self.emitted = 0
        self.released = 0
        self.calls = 0
        self.frozen = frozen

    def release(self):
        self.released += 1
        self._sem.release()

    def settled(self):
        """True once the reader has queued every released line and is blocked again."""
        return self._done or self.calls >= self.released + 1

    def finish(self):
        self._done = True
        self._sem.release()

    def readline(self):
        self.calls += 1
        self._sem.acquire()
        if self._done:
            return ""
        self.emitted += 1
        if self.frozen:
            # ffmpeg 7 while a muxer write blocks: frame, size and time frozen; only
            # fps/speed vary (rr23 ff7_stall.err).
            return f"frame=100 fps={self.emitted} size=1024kB time=00:00:04.00 speed=0.{self.emitted}x\n"
        n = self.emitted
        return (
            f"frame={n} fps=25 size={n}kB time={n // 3600:02d}:{n // 60 % 60:02d}:{n % 60:02d}.00\n"
        )


class _Stdout:
    def read(self, n=-1):
        return ""


class _Proc:
    """Popen stand-in on a manual clock: each ``poll()`` is one loop tick of ``tick`` s.

    ``progress=True`` emits one progress line per tick; ``exit_after`` ticks later it
    exits 0 (``None`` = never)."""

    def __init__(self, clock, *, tick, progress, exit_after, out=None, frozen=False):
        self.clock, self.tick, self.progress, self.exit_after = clock, tick, progress, exit_after
        self.stdout, self.stderr = _Stdout(), _Stderr(frozen=frozen)
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


def _await_consumed(proc):
    """The runner's per-tick sleep: wait until the reader has queued every released line."""
    deadline = _time.monotonic() + 5.0
    while not proc.stderr.settled() and _time.monotonic() < deadline:
        _real_sleep(0.001)


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


_BIG_INPUT = 40 * 10**9  # sparse: a 40 GB raw LF, so the concat ceiling (4000 s) sits above 1800 s


def _prepend(tmp_path, proc_factory, failures=None, raw_bytes=3):
    raw = tmp_path / "raw_lf.mp4"
    raw.write_bytes(b"RAW")
    with open(raw, "r+b") as fh:
        fh.truncate(raw_bytes)  # sparse, costs no disk
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
        patch.object(_runner.time, "sleep", side_effect=lambda _s: _await_consumed(proc)),
    ):
        result = montage_builder.prepend_montage(
            raw_lf_path=raw,
            peaks=[SimpleNamespace(source_file_index=0, start_time=0.0, end_time=40.0)],
            output_path=out,
            work_dir=work,
            master_path=tmp_path / "master.mp4",
            file_boundaries=[(0.0, 600.0)],
            failures=failures,
        )
    return result, out, proc, clock


def test_concat_that_keeps_reporting_progress_past_1800s_is_not_killed(tmp_path):
    result, out, proc, clock = _prepend(
        tmp_path,
        lambda clock, out: _Proc(clock, tick=300.0, progress=True, exit_after=8, out=out),
        raw_bytes=_BIG_INPUT,
    )
    assert clock.t > 1800.0, "the fake run must outlast the old fixed clock"
    assert proc.terminated is False
    assert result == out
    assert out.exists()


def test_concat_that_stalls_is_killed_and_its_partial_output_removed(tmp_path):
    failures: list = []
    result, out, proc, clock = _prepend(
        tmp_path,
        lambda clock, out: _Proc(clock, tick=300.0, progress=False, exit_after=None, out=out),
        failures,
    )
    assert proc.terminated is True
    assert result is None
    assert not out.exists(), "a truncated concat must not be left on disk"
    # The STALL rule, not the ceiling (which would also terminate and delete).
    assert failures == [failures[0]] and "killed (stall)" in failures[0]


def test_concat_repeating_an_identical_stats_line_is_killed_as_a_stall(tmp_path):
    """A frozen stats line (blocked muxer write) is a stall, not progress."""
    failures: list = []
    result, out, proc, clock = _prepend(
        tmp_path,
        lambda clock, out: _Proc(
            clock, tick=300.0, progress=True, exit_after=None, out=out, frozen=True
        ),
        failures,
        raw_bytes=_BIG_INPUT,
    )
    assert proc.terminated is True
    assert result is None
    assert not out.exists()
    assert "killed (stall)" in failures[0]
    assert clock.t < 86400.0, "ended by the stall rule, not the 86400 s ceiling"


@pytest.mark.parametrize(
    ("raw_bytes", "expected"),
    [
        (3, 1800),  # tiny input: the 1800 s floor
        # 40 GB: one pass to copy and one for the +faststart rewrite, each at the runner's
        # own throughput floor (MOOV_REWRITE_MIN_BYTES_PER_SEC)
        (
            _BIG_INPUT,
            2 * -(-(_BIG_INPUT + len(b"MONTAGE")) // _runner.MOOV_REWRITE_MIN_BYTES_PER_SEC),
        ),
    ],
)
def test_concat_uses_the_stall_runner_with_a_size_based_ceiling(tmp_path, raw_bytes, expected):
    """Through run_ffmpeg_encode (never run_ffmpeg_quick's fixed wall clock), `-nostdin`, and
    max_timeout = max(1800, 2 * ceil(input_bytes / MOOV_REWRITE_MIN_BYTES_PER_SEC))."""
    raw = tmp_path / "raw_lf.mp4"
    with open(raw, "wb") as fh:
        fh.truncate(raw_bytes)
    work = tmp_path / "work"
    work.mkdir()
    seen: dict = {}

    def _encode(cmd, output_path, expected_duration, description, **kw):
        seen.update(kw, cmd=cmd)
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
            output_path=tmp_path / "out.mp4",
            work_dir=work,
            master_path=tmp_path / "master.mp4",
            file_boundaries=[(0.0, 600.0)],
        )
    assert result == tmp_path / "out.mp4"
    quick.assert_not_called()
    assert seen["max_timeout"] == expected
    assert "-nostdin" in seen["cmd"]


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


# --- D. a synthesised full_drive plan carries montage peaks ---------------------


def _fd_synth_segments():
    return [
        SimpleNamespace(
            source_file_index=i,
            original_start_time=i * 1200.0,
            original_end_time=(i + 1) * 1200.0,
            composite_score=0.5,
        )
        for i in range(2)
    ]


def _fd_episodes():
    return [SimpleNamespace(concat_path=f"/nonexistent/ep{i}.mp4") for i in range(2)]


def _fd_peaks(scores):
    from gomoveshift.config import DriveConfig

    return day_pipeline._full_drive_montage_peaks(
        DriveConfig(), scores, [(0.0, 1200.0), (1200.0, 2400.0)], _fd_episodes(), _SEGS
    )


_SEGS = _fd_synth_segments()


def test_full_drive_peaks_come_from_the_score_pool_inside_the_plan():
    import numpy as np

    scores = _make_scores(2400)
    scores[:, 1] = 0.2 + 0.6 * (np.sin(np.arange(2400) / 40.0) > 0)  # alternating high/low runs
    peaks = _fd_peaks(scores)
    assert 1 <= len(peaks) <= 6  # DriveConfig.montage_peak_count
    assert {p.source_file_index for p in peaks} <= {0, 1}, "indices stay day-local"


def test_full_drive_peaks_fall_back_to_the_plans_own_segments():
    from gomoveshift.pipeline.longform_assembler import LongFormAssembler

    with patch.object(LongFormAssembler, "_build_full_drive_peak_pool", return_value=[]):
        peaks = _fd_peaks(_make_scores(2400))
    assert peaks and all(p in _SEGS for p in peaks), "an empty pool leaves the plan's segments"


def test_synthesised_full_drive_day_builds_its_montage_with_the_real_prepend(tmp_path):
    """Real ``prepend_montage`` (only ffmpeg/keycut IO faked): `montage_peaks=[]` made it return None."""
    from test_day_pipeline_full_drive_mode import _make_config as _fd_config
    from test_day_pipeline_full_drive_mode import _make_day_batch as _fd_batch

    out = tmp_path / "out"
    master = out / f"day_{DATE}" / f"day_{DATE}_master.mp4"
    master.parent.mkdir(parents=True, exist_ok=True)
    master.write_bytes(b"m")
    clips: list = []

    def _concat(files, output_path, list_file, *a, **kw):
        _write(output_path)
        return True

    def _encode(cmd, output_path, expected_duration, description, **kw):
        _write(output_path)
        return SimpleNamespace(success=True, killed_reason=None, stderr_tail="", returncode=0)

    with (
        patch("gomoveshift.pipeline.day_pipeline.encode_day_master", return_value=master),
        patch(
            "gomoveshift.pipeline.day_pipeline.extract_and_concat_segments",
            side_effect=lambda master_path, segments, file_boundaries, output_path, work_dir, **kw: (
                _write(output_path)
            ),
        ),
        patch(
            "gomoveshift.pipeline.day_pipeline.apply_ost",
            side_effect=lambda input_path, output_path, location, layout, **kw: _write(output_path),
        ),
        patch(
            "gomoveshift.pipeline.day_pipeline._render_short_ffmpeg",
            side_effect=lambda source_video, output_path, *a, **kw: bool(_write(output_path)),
        ),
        patch("gomoveshift.pipeline.day_pipeline.check_lf_work_space"),
        patch("gomoveshift.pipeline.day_pipeline.plan_shorts", return_value=_SHORTS),
        patch.object(montage_builder, "probe_keyframes", return_value=None),
        patch.object(
            montage_builder, "reencode_range", side_effect=lambda *a, **kw: clips.append(a) or True
        ),
        patch.object(montage_builder, "concat_demuxer", side_effect=_concat),
        patch(
            "gomoveshift.pipeline.validation.ffprobe_client.FFprobeClient.probe_duration",
            return_value=0.0,
        ),
        patch.object(_runner, "run_ffmpeg_encode", side_effect=_encode),
    ):
        artifacts = day_pipeline.run_day_pipeline(
            day_batch=_fd_batch(DATE, 2, per_ep_duration=1200.0),
            day_scores=_make_scores(2400),
            output_dir=out,
            location="L",
            config=_fd_config(lf_selection_mode="full_drive"),
            lf_plans=None,
            state=make_state(ost_audit_confirmed=True),
        )
    assert len(artifacts.regular_lf_paths) == 1
    assert clips, "the montage was built from at least one peak clip"


# --- E. montage clip cuts and every named failure cause -------------------------


def _trim(tmp_path, keyframes, *, peak=(110.0, 140.0)):
    master = tmp_path / "master.mp4"
    master.write_bytes(b"fake")
    pk = SimpleNamespace(source_file_index=0, start_time=peak[0], end_time=peak[1])
    copy, reenc = MagicMock(return_value=True), MagicMock(return_value=True)
    with (
        patch.object(montage_builder, "probe_keyframes", return_value=keyframes),
        patch.object(montage_builder, "stream_copy_range", copy),
        patch.object(montage_builder, "reencode_range", reenc),
    ):
        ok = montage_builder._trim_peak_to(
            master, pk, [(0.0, 600.0)], tmp_path / "clip.mp4", clip_duration=5.0
        )
    return ok, copy, reenc


def test_a_keyframe_just_before_the_start_still_reencodes(tmp_path):
    """Start 122.5 s, prior keyframe 121.0: `-c copy` would replay 1.5 s of pre-peak footage."""
    ok, copy, reenc = _trim(tmp_path, [0.0, 121.0, 200.0])
    assert ok is True
    copy.assert_not_called()
    reenc.assert_called_once()


def test_both_reencode_branches_pin_the_long_form_encoder_args(tmp_path):
    expected = montage_builder.lf_reencode_args()
    for keyframes in (None, [0.0, 100.0, 200.0]):  # probe unavailable / none fits
        ok, copy, reenc = _trim(tmp_path, keyframes)
        assert ok is True
        assert reenc.call_args.kwargs["encode_args"] == expected


def test_no_keyframe_fits_reencode_is_logged_at_warning(tmp_path, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger=montage_builder.logger.name):
        _trim(tmp_path, [0.0, 100.0, 200.0])
    assert any(
        r.levelno == logging.WARNING and "no keyframe at its start" in r.getMessage()
        for r in caplog.records
    )


def _real_prepend(tmp_path, *, clip_ok=True, concat_ok=True, probe=15.0, encode=None):
    """prepend_montage with its real build_montage; only ffmpeg/keycut/ffprobe IO is faked."""
    raw = tmp_path / "raw_lf.mp4"
    raw.write_bytes(b"RAW")
    work = tmp_path / "work"
    failures: list = []

    def _concat(files, output_path, list_file, *a, **kw):
        if concat_ok:
            _write(output_path)
        return concat_ok

    with (
        patch.object(montage_builder, "probe_keyframes", return_value=None),
        patch.object(montage_builder, "reencode_range", return_value=clip_ok),
        patch.object(montage_builder, "concat_demuxer", side_effect=_concat),
        patch(
            "gomoveshift.pipeline.validation.ffprobe_client.FFprobeClient.probe_duration",
            return_value=probe,
        ),
        patch.object(_runner, "run_ffmpeg_encode", side_effect=encode),
    ):
        result = montage_builder.prepend_montage(
            raw_lf_path=raw,
            peaks=[SimpleNamespace(source_file_index=0, start_time=0.0, end_time=40.0)],
            output_path=tmp_path / "out.mp4",
            work_dir=work,
            master_path=tmp_path / "master.mp4",
            file_boundaries=[(0.0, 600.0)],
            failures=failures,
        )
    return result, failures


def test_a_montage_that_fails_the_duration_gate_names_why(tmp_path):
    result, failures = _real_prepend(tmp_path, probe=16.7)
    assert result is None
    assert failures and "duration verification" in failures[0]
    assert "16.7" in failures[0]


def test_every_clip_extraction_failing_is_named(tmp_path):
    result, failures = _real_prepend(tmp_path, clip_ok=False)
    assert result is None
    assert failures == ["every montage peak clip extraction failed"]


def test_a_failed_montage_concat_is_named(tmp_path):
    result, failures = _real_prepend(tmp_path, concat_ok=False)
    assert result is None
    assert failures == ["montage clip concat failed"]


def test_a_nonzero_exit_prepend_concat_is_named_with_its_stderr(tmp_path):
    def _encode(cmd, output_path, expected_duration, description, **kw):
        return SimpleNamespace(
            success=False, killed_reason=None, returncode=1, stderr_tail="Invalid data found\n"
        )

    result, failures = _real_prepend(tmp_path, encode=_encode)
    assert result is None
    assert len(failures) == 1
    assert "failed (exit 1)" in failures[0] and "Invalid data found" in failures[0]
