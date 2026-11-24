"""
Tests for gomoveshift.pipeline.ffmpeg_runner module.

Tests stall detection, progress parsing, cleanup on failure, and video validation.
"""

import json
import logging
import os
import sys
import textwrap
import threading
import time
from unittest.mock import Mock, patch

import pytest

from gomoveshift.pipeline.ffmpeg_runner import (
    MOOV_REWRITE_MIN_STALL_S,
    _moov_rewrite_fallback_timeout,
    _moov_rewrite_stall_timeout,
    _read_io_counters,
    _parse_time_to_seconds,
    _stall_kill_reason,
    run_ffmpeg_encode,
    run_ffmpeg_quick,
    validate_video,
)


class _ScriptedClock:
    """Deterministic clock: returns successive values, clamping to the last.

    Lets a test drive stall/timeout decisions explicitly instead of depending
    on real elapsed time (which flakes under parallel/loaded runs).
    """

    def __init__(self, values):
        self._values = list(values)
        self._i = 0

    def __call__(self):
        v = self._values[min(self._i, len(self._values) - 1)]
        self._i += 1
        return v


class _FakeStderr:
    """A file-like stderr yielding pre-scripted lines, then "" (EOF)."""

    def __init__(self, lines):
        self._lines = list(lines)
        self._i = 0

    def readline(self):
        if self._i < len(self._lines):
            line = self._lines[self._i]
            self._i += 1
            return line
        return ""


class _FakeStdout:
    def read(self, n=-1):
        return ""  # no stdout — EOF immediately


class _FakeProc:
    """Minimal subprocess.Popen stand-in with scripted stderr + poll results.

    Combined with a _ScriptedClock this makes moov/stall behavior fully
    deterministic: no real subprocess, no real sleeps, no wall-clock timing.
    """

    def __init__(self, stderr_lines, poll_returns):
        self.stdout = _FakeStdout()
        self.stderr = _FakeStderr(stderr_lines)
        self._poll = list(poll_returns)
        self._pi = 0
        self.returncode = None
        self.terminated = False
        self.killed = False

    def poll(self):
        v = self._poll[min(self._pi, len(self._poll) - 1)]
        self._pi += 1
        if v is not None:
            self.returncode = v
        return v

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


# ── Time Parsing ──────────────────────────────────────────────────────


class TestParseTimeToSeconds:
    def test_hms_format(self):
        assert _parse_time_to_seconds("01:30:45.50") == pytest.approx(5445.5)

    def test_ms_format(self):
        assert _parse_time_to_seconds("45:30.00") == pytest.approx(2730.0)

    def test_seconds_only(self):
        assert _parse_time_to_seconds("123.45") == pytest.approx(123.45)

    def test_empty_string(self):
        assert _parse_time_to_seconds("") is None

    def test_invalid(self):
        assert _parse_time_to_seconds("not-a-time") is None


# ── run_ffmpeg_encode ─────────────────────────────────────────────────


# Real-clock children (``python3 -c`` stubs) must start, print and exit before
# the stall window closes. On a heavily loaded box a tight window can kill
# such a stub as "stall": interpreter startup alone can outlast a 5 s
# window. The kill paths are proven with ``_ScriptedClock``; a real-clock test
# only needs a window a loaded box cannot close, never a tight budget.
LOADED_BOX_STALL_S = 300
LOADED_BOX_MAX_S = 600
# The two stdout-drain tests fail THROUGH the stall kill when the drainer regresses
# (the child blocks on a full pipe and stops reporting), so their window is also the
# time a real regression costs. 60 s is still 12x the start-up latency seen above.
DRAIN_REGRESSION_STALL_S = 60
DRAIN_REGRESSION_MAX_S = 120


class TestRunFfmpegEncode:
    """Tests for the Popen-based long encode runner."""

    def test_successful_encode(self, tmp_path):
        """A successful encode returns FFmpegResult(success=True)."""
        output = tmp_path / "out.mp4"
        # Create a fake output file that ffmpeg would write
        output.write_bytes(b"fake video data")

        cmd = [
            "python3",
            "-c",
            textwrap.dedent("""\
            import sys, time
            # Simulate ffmpeg progress output on stderr
            for i in range(3):
                print(f"frame=  {i*100} fps= 60.0 size=   1024kB time=00:00:{i*10:02d}.00 speed=1.0x",
                      file=sys.stderr, flush=True)
                time.sleep(0.1)
        """),
        ]

        result = run_ffmpeg_encode(
            cmd,
            output,
            expected_duration=30.0,
            description="test encode",
            stall_timeout=LOADED_BOX_STALL_S,
            log_interval=1,
        )
        assert result.success
        assert result.returncode == 0
        assert result.killed_reason is None

    def test_stall_detection_kills_process(self, tmp_path):
        """Process killed after stall_timeout seconds of no progress output.

        Deterministic: the injected clock jumps past the stall window on the
        first loop reading, so the kill is instant and does not depend on real
        elapsed time.
        """
        output = tmp_path / "out.mp4"

        # Process that outputs nothing and would otherwise hang.
        cmd = ["python3", "-c", "import time; time.sleep(60)"]

        # start=0, then the first loop reading is 5s: no progress since start
        # (5s > 2s stall window) but within the 30s max ceiling → "stall".
        clock = _ScriptedClock([0.0, 5.0])
        result = run_ffmpeg_encode(
            cmd,
            output,
            expected_duration=30.0,
            description="test stall",
            stall_timeout=2,
            max_timeout=30,
            time_source=clock,
        )
        assert not result.success
        assert result.killed_reason == "stall"
        # Partial output should be cleaned up
        assert not output.exists()

    def test_stall_resets_on_progress(self):
        """Stall timer resets when progress output is received.

        The window is measured from the LAST observed progress, not from
        encode start — so continuous progress keeps an encode alive no matter
        how long it runs, while a genuine gap still fires. Driven by an
        explicit clock (no real elapsed time), which is what makes it
        deterministic under parallel/loaded test runs (the original real-sleep
        version flaked under xdist CPU contention).
        """
        # Progress seen recently (t=95) though the encode started at t=0 and
        # total elapsed (100s) dwarfs the 10s stall window: the reset keeps it
        # alive (it would fire if the timer were anchored at start).
        assert (
            _stall_kill_reason(
                now=100.0,
                start_time=0.0,
                last_progress_time=95.0,
                max_timeout=86400.0,
                stall_timeout=10.0,
            )
            is None
        )
        # No progress since t=80 → 20s gap > 10s window → stall fires.
        assert (
            _stall_kill_reason(
                now=100.0,
                start_time=0.0,
                last_progress_time=80.0,
                max_timeout=86400.0,
                stall_timeout=10.0,
            )
            == "stall"
        )
        # Boundary: a gap exactly equal to the window does not fire (strict >).
        assert (
            _stall_kill_reason(
                now=100.0,
                start_time=0.0,
                last_progress_time=90.0,
                max_timeout=86400.0,
                stall_timeout=10.0,
            )
            is None
        )

    def test_max_timeout_takes_precedence_over_stall(self):
        """When both ceilings are breached at once, max_timeout wins."""
        assert (
            _stall_kill_reason(
                now=200.0,
                start_time=0.0,
                last_progress_time=0.0,
                max_timeout=100.0,
                stall_timeout=10.0,
            )
            == "max_timeout"
        )

    def test_partial_file_deleted_on_failure(self, tmp_path):
        """Output file is deleted when ffmpeg returns non-zero."""
        output = tmp_path / "out.mp4"
        output.write_bytes(b"partial corrupt data")

        cmd = ["python3", "-c", "import sys; sys.exit(1)"]

        result = run_ffmpeg_encode(
            cmd,
            output,
            expected_duration=30.0,
            description="test cleanup",
            stall_timeout=LOADED_BOX_STALL_S,
        )
        assert not result.success
        assert not output.exists(), "Partial output should be deleted on failure"

    def test_max_timeout_kills_process(self, tmp_path):
        """Process killed after max_timeout even if progress is flowing.

        Deterministic: the injected clock jumps past the max ceiling on the
        first loop reading, so the kill is instant regardless of real time.
        """
        output = tmp_path / "out.mp4"

        # Process outputs progress but doesn't exit on its own.
        cmd = [
            "python3",
            "-c",
            textwrap.dedent("""\
            import sys, time
            i = 0
            while True:
                print(f"frame=  {i} fps= 60.0 size=   1024kB time=00:00:{i:02d}.00 speed=1.0x",
                      file=sys.stderr, flush=True)
                time.sleep(0.5)
                i += 1
        """),
        ]

        # First loop reading jumps to 1000s, past the 3s max ceiling. Because
        # max is checked before stall (and before draining any progress),
        # "max_timeout" wins.
        clock = _ScriptedClock([0.0, 1000.0])
        result = run_ffmpeg_encode(
            cmd,
            output,
            expected_duration=3600.0,
            description="test max timeout",
            stall_timeout=10,
            max_timeout=3,
            time_source=clock,
        )
        assert not result.success
        assert result.killed_reason == "max_timeout"
        assert not output.exists()

    def test_progress_parsing_nvenc_format(self, tmp_path):
        """Parses NVENC-style progress output correctly."""
        output = tmp_path / "out.mp4"
        output.write_bytes(b"fake")

        cmd = [
            "python3",
            "-c",
            textwrap.dedent("""\
            import sys
            print("frame=  3600 fps=142.3 q=21.0 size=  524288kB time=00:45:30.00 bitrate=1572.1kbits/s speed=1.2x",
                  file=sys.stderr, flush=True)
        """),
        ]

        result = run_ffmpeg_encode(
            cmd,
            output,
            expected_duration=4110.0,
            description="test nvenc parse",
            stall_timeout=LOADED_BOX_STALL_S,
            log_interval=1,
        )
        assert result.success
        assert result.last_speed == pytest.approx(1.2)

    def test_progress_callback_called(self, tmp_path):
        """Progress callback is called with parsed values."""
        output = tmp_path / "out.mp4"
        output.write_bytes(b"fake")
        callback_calls = []

        def cb(current, total, speed, fps):
            callback_calls.append((current, total, speed, fps))

        cmd = [
            "python3",
            "-c",
            textwrap.dedent("""\
            import sys
            print("frame=  100 fps= 30.0 size=   1024kB time=00:00:10.00 speed=2.0x",
                  file=sys.stderr, flush=True)
        """),
        ]

        result = run_ffmpeg_encode(
            cmd,
            output,
            expected_duration=60.0,
            description="test callback",
            stall_timeout=LOADED_BOX_STALL_S,
            progress_callback=cb,
        )
        assert result.success
        assert len(callback_calls) > 0
        # First callback should have time=10s
        assert callback_calls[0][0] == pytest.approx(10.0)

    def test_large_stdout_does_not_deadlock_and_is_captured(self, tmp_path):
        """Regression: stdout pipe is drained, so a child that writes
        >64 KB to stdout (the typical Linux pipe buffer size) does not
        deadlock. Captured stdout is exposed on FFmpegResult.stdout so
        callers (e.g. inline loudnorm/JSON ffprobe) can read it.

        Without the stdout drainer this test hangs until the stall
        timeout kills the child — i.e. it'd return killed_reason='stall'
        instead of succeeding.
        """
        output = tmp_path / "out.mp4"
        output.write_bytes(b"fake")

        # Write 256 KB to stdout — 4x the typical Linux pipe buffer.
        # Also emit one ffmpeg-shaped progress line on stderr so the
        # stall detector sees activity and would otherwise be ready to
        # kill — proving the assertion is about the stdout drain path,
        # not the stderr path.
        cmd = [
            "python3",
            "-c",
            textwrap.dedent("""\
            import sys
            sys.stderr.write("frame=  100 fps= 60.0 size=   1024kB time=00:00:10.00 speed=1.0x\\n")
            sys.stderr.flush()
            payload = ("A" * 1024 + "\\n") * 256  # 256 KB
            sys.stdout.write(payload)
            sys.stdout.flush()
        """),
        ]

        result = run_ffmpeg_encode(
            cmd,
            output,
            expected_duration=30.0,
            description="test stdout drain",
            stall_timeout=DRAIN_REGRESSION_STALL_S,
            max_timeout=DRAIN_REGRESSION_MAX_S,
            log_interval=1,
        )
        assert result.success, f"expected success, got killed_reason={result.killed_reason}"
        assert result.killed_reason is None
        # Captured stdout should contain the full 256 KB payload.
        assert len(result.stdout) >= 256 * 1024
        assert result.stdout.startswith("A" * 1024)

    def test_empty_stdout_default(self, tmp_path):
        """Common case: ffmpeg writes media to -o and emits nothing on stdout.
        FFmpegResult.stdout is the empty string, not None.
        """
        output = tmp_path / "out.mp4"
        output.write_bytes(b"fake")

        cmd = [
            "python3",
            "-c",
            textwrap.dedent("""\
            import sys
            sys.stderr.write("frame=  100 fps= 60.0 size=   1024kB time=00:00:10.00 speed=1.0x\\n")
            sys.stderr.flush()
        """),
        ]

        result = run_ffmpeg_encode(
            cmd,
            output,
            expected_duration=30.0,
            description="test empty stdout",
            stall_timeout=LOADED_BOX_STALL_S,
        )
        assert result.success
        assert result.stdout == ""

    def test_stdout_capture_is_bounded_to_a_tail(self, tmp_path, monkeypatch):
        """A child emitting far more stdout than the bound keeps only the tail.

        Memory held is limited to the bound; the last lines survive and the
        failure path still reports stderr's last lines.
        """
        from gomoveshift.pipeline import ffmpeg_runner

        bound = 4096
        monkeypatch.setattr(ffmpeg_runner, "_STDOUT_TAIL_MAX_CHARS", bound)
        output = tmp_path / "out.mp4"
        output.write_bytes(b"fake")

        cmd = [
            "python3",
            "-c",
            textwrap.dedent("""\
            import sys
            for i in range(20000):
                sys.stdout.write("line %06d\\n" % i)
            sys.stdout.flush()
            sys.stderr.write("fatal: boom-last-stderr-line\\n")
            sys.exit(1)
        """),
        ]

        result = run_ffmpeg_encode(
            cmd,
            output,
            expected_duration=30.0,
            description="test bounded stdout",
            stall_timeout=DRAIN_REGRESSION_STALL_S,
            max_timeout=DRAIN_REGRESSION_MAX_S,
        )
        assert not result.success
        assert len(result.stdout) <= bound
        assert result.stdout.endswith("line 019999\n")
        assert "line 000000" not in result.stdout
        assert "boom-last-stderr-line" in result.stderr_tail

    def test_moov_rewrite_extends_stall(self, tmp_path):
        """Moov atom rewrite phase extends the stall timeout.

        Deterministic via a scripted process + explicit clock. Progress then a
        'moving the moov atom' line are consumed while the clock reads 0; the
        clock then jumps to 100s (far past the 2s base window) with the process
        still alive. The encode survives ONLY because the moov line raised the
        effective stall window — asserting the extension actually took effect,
        with no real sleeps.
        """
        output = tmp_path / "out.mp4"
        output.write_bytes(b"fake")

        stderr_lines = [
            "frame=  100 fps= 30.0 size=   1024kB time=00:00:10.00 speed=1.0x\n",
            "Starting second pass: moving the moov atom to the beginning of the file\n",
        ]
        # poll: alive while the lines are consumed and the clock is still 0,
        # then exits cleanly (0) on the iteration where the clock reads 100.
        fake = _FakeProc(stderr_lines=stderr_lines, poll_returns=[None, None, None, 0])
        # start=0; small readings let the reader thread deliver both lines and
        # the extension take effect; then a 100s jump exercises the extended
        # window. Base window (2s) would kill at 100s → success proves the
        # extension.
        clock = _ScriptedClock([0.0, 0.0, 0.0, 0.0, 100.0])

        with patch("gomoveshift.pipeline.ffmpeg_runner.subprocess.Popen", return_value=fake):
            result = run_ffmpeg_encode(
                ["ffmpeg", "-i", "in.mp4", str(output)],
                output,
                expected_duration=10.0,
                description="test moov",
                stall_timeout=2,
                max_timeout=1000,
                time_source=clock,
            )

        assert result.success
        assert result.killed_reason is None
        # Never killed → the moov extension applied.
        assert fake.terminated is False
        assert fake.killed is False

    def test_moov_rewrite_window_scales_with_output_size(self):
        """The silent faststart rewrite gets time in proportion to the file size.

        Ace Basin (2026-10-26): a 118 GB day master was ~38 min into a healthy
        moov rewrite (~52 MB/s on /mnt/c) when the flat 1800s window killed it.
        """
        assert _moov_rewrite_stall_timeout(0, 600) == 1800
        assert _moov_rewrite_stall_timeout(1_000_000_000, 600) == 1800
        assert _moov_rewrite_stall_timeout(1_000_000_000, 7200) == 7200
        # 118 GB at the 20 MB/s floor.
        assert _moov_rewrite_stall_timeout(117_987_020_255, 600) == 5900

    def test_moov_rewrite_of_large_output_outlives_1800s(self, tmp_path):
        """A 118 GB output still rewriting after 1h is not killed as a stall."""
        output = tmp_path / "out.mp4"
        with open(output, "wb") as f:
            f.truncate(117_987_020_255)  # sparse: no real disk used

        stderr_lines = [
            "frame=  100 fps= 30.0 size=   1024kB time=00:00:10.00 speed=1.0x\n",
            "Starting second pass: moving the moov atom to the beginning of the file\n",
        ]
        fake = _FakeProc(stderr_lines=stderr_lines, poll_returns=[None, None, None, 0])
        clock = _ScriptedClock([0.0, 0.0, 0.0, 0.0, 3600.0])

        with patch("gomoveshift.pipeline.ffmpeg_runner.subprocess.Popen", return_value=fake):
            result = run_ffmpeg_encode(
                ["ffmpeg", "-i", "in.mp4", str(output)],
                output,
                expected_duration=10.0,
                description="test moov large",
                stall_timeout=600,
                max_timeout=86400,
                time_source=clock,
            )

        assert result.killed_reason is None
        assert result.success
        assert fake.terminated is False

    def test_moov_rewrite_of_large_output_still_stalls_eventually(self, tmp_path):
        """Past the size-scaled window a silent rewrite is still killed."""
        output = tmp_path / "out.mp4"
        with open(output, "wb") as f:
            f.truncate(117_987_020_255)

        stderr_lines = [
            "frame=  100 fps= 30.0 size=   1024kB time=00:00:10.00 speed=1.0x\n",
            "Starting second pass: moving the moov atom to the beginning of the file\n",
        ]
        fake = _FakeProc(stderr_lines=stderr_lines, poll_returns=[None])
        clock = _ScriptedClock([0.0, 0.0, 0.0, 0.0, 6000.0])

        with patch("gomoveshift.pipeline.ffmpeg_runner.subprocess.Popen", return_value=fake):
            result = run_ffmpeg_encode(
                ["ffmpeg", "-i", "in.mp4", str(output)],
                output,
                expected_duration=10.0,
                description="test moov large stall",
                stall_timeout=600,
                max_timeout=86400,
                time_source=clock,
            )

        assert result.killed_reason == "stall"
        assert fake.terminated is True


# ── +faststart rewrite timed by its own progress ─────────────────────

GALLEY_LF1_BYTES = 38_824_572_047  # Galley Head LF1, the concat that was killed as a "stall"


class _SimClock:
    """Clock the fake ffmpeg advances itself (one tick per poll), so no real time passes."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class _SimStderr:
    def __init__(self):
        self.lines = []
        self.cv = threading.Condition()
        self.done = False
        self.served = 0
        self.waiting = False

    def push(self, line):
        with self.cv:
            self.lines.append(line)
            self.cv.notify_all()

    def finish(self):
        with self.cv:
            self.done = True
            self.cv.notify_all()

    def readline(self):
        with self.cv:
            while self.served >= len(self.lines) and not self.done:
                self.waiting = True
                self.cv.wait()
            self.waiting = False
            if self.served < len(self.lines):
                self.served += 1
                return self.lines[self.served - 1]
            return ""

    def settled(self):
        with self.cv:
            return self.done or (self.served >= len(self.lines) and self.waiting)


class _SimConcat:
    """A ``-c copy -movflags +faststart`` ffmpeg: a copy pass printing advancing stats lines,
    then the mp4 muxer's silent moov rewrite, then exit 0.

    ``io`` is what ``/proc/<pid>/io`` would report (rchar+wchar): it grows through both passes
    unless ``freeze_io_in_rewrite``. The clock moves ``tick`` seconds per poll.
    """

    pid = 4242

    def __init__(self, clock, out, *, tick, copy_s, rewrite_s, size, freeze_io_in_rewrite=False):
        self.clock, self.out, self.tick = clock, out, tick
        self.copy_s, self.rewrite_s, self.size = copy_s, rewrite_s, size
        self.freeze_io_in_rewrite = freeze_io_in_rewrite
        self.stdout, self.stderr = _FakeStdout(), _SimStderr()
        self.returncode = None
        self.terminated = False
        self.n = 0
        self.moov_at = None
        self.io = 0

    def poll(self):
        self.clock.t += self.tick
        t = self.clock.t
        if self.returncode is not None:
            return self.returncode
        if t < self.copy_s:
            self.n += 1
            kib = int(self.size * t / self.copy_s) // 1024
            with open(self.out, "wb") as fh:
                fh.truncate(kib * 1024)
            self.io = 2 * kib * 1024
            self.stderr.push(
                f"frame={self.n * 25} fps=0.0 q=-1.0 size={kib}KiB "
                f"time=00:00:{self.n % 60:02d}.00 bitrate=N/A speed=1x\n"
            )
            return None
        if self.moov_at is None:
            with open(self.out, "wb") as fh:
                fh.truncate(self.size)
            self.moov_at = t
            self.stderr.push(
                "[mp4 @ 0x1] Starting second pass: moving the moov atom to the beginning "
                "of the file\n"
            )
            return None
        if not self.freeze_io_in_rewrite:
            self.io += int(2 * self.size * self.tick / max(self.rewrite_s, 1))
        if t - self.moov_at >= self.rewrite_s:
            self.stderr.push("frame=999 fps=0.0 q=-1.0 Lsize=1KiB time=00:59:59.00 bitrate=N/A\n")
            self.returncode = 0
            self.stderr.finish()
            return 0
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


def _run_concat(tmp_path, monkeypatch, *, io_readable, io_fn=None, **kw):
    """Run the prepend_montage call shape (default stall, expected_duration=0) on a sim."""
    out = tmp_path / "lf_01.mp4"
    clock = _SimClock()
    proc = _SimConcat(clock, out, **kw)
    real_sleep = time.sleep

    def _sleep(_s):  # wait for the stderr reader thread to hand over the line just pushed
        deadline = time.monotonic() + 5.0
        while not proc.stderr.settled() and time.monotonic() < deadline:
            real_sleep(0.0005)

    monkeypatch.setattr("gomoveshift.pipeline.ffmpeg_runner.subprocess.Popen", lambda *a, **k: proc)
    monkeypatch.setattr("gomoveshift.pipeline.ffmpeg_runner.time.sleep", _sleep)
    monkeypatch.setattr(
        "gomoveshift.pipeline.ffmpeg_runner._read_io_counters",
        (lambda pid: io_fn(proc))
        if io_fn
        else ((lambda pid: proc.io) if io_readable else (lambda pid: None)),
    )
    cmd = ["ffmpeg", "-nostdin", "-i", "l.txt", "-c", "copy", "-movflags", "+faststart", str(out)]
    result = run_ffmpeg_encode(
        cmd, out, expected_duration=0.0, description="prepend_montage concat", time_source=clock
    )
    return result, proc, clock, out


_REAL_REWRITE_CHILD = textwrap.dedent(
    """\
    import sys, time
    out, chunks = sys.argv[1], int(sys.argv[2])
    e = sys.stderr
    e.write("frame=   10 fps=1 q=-1.0 size=     256KiB time=00:00:01.00 bitrate=N/A speed=1x\\n")
    e.flush()
    e.write("[mp4 @ 0x1] Starting second pass: moving the moov atom to the beginning of the file\\n")
    e.flush()
    with open(out, "wb") as fh:
        for _ in range(chunks):
            fh.write(b"x" * 65536)
            fh.flush()
            time.sleep(0.02)
    e.write("frame=   10 fps=1 q=-1.0 Lsize=    9999KiB time=00:00:01.00 bitrate=N/A speed=1x\\n")
    e.flush()
    """
)


class TestMoovRewriteFollowsOwnProgress:
    """The silent +faststart rewrite is timed by its own progress, not a fixed speed floor.

    Galley Head LF1 (38.8 GB) copied at ~21.6 MB/s; the rewrite moves the same bytes, and a
    fixed 20 MB/s floor left a 1942 s window against a ~1797 s rewrite, so any volume under
    20 MB/s was killed as a "stall" on every retry.
    """

    def test_both_passes_at_18_MBps_finish_when_io_counters_move(self, tmp_path, monkeypatch):
        slow = 2157.0  # 38.8 GB at 18 MB/s, copy and rewrite alike
        result, proc, _, _ = _run_concat(
            tmp_path, monkeypatch, io_readable=True, tick=30.0, copy_s=slow, rewrite_s=slow,
            size=GALLEY_LF1_BYTES,
        )  # fmt: skip
        assert result.success and result.killed_reason is None
        assert proc.terminated is False

    def test_a_rewrite_much_slower_than_its_copy_pass_still_finishes_while_io_moves(
        self, tmp_path, monkeypatch
    ):
        result, proc, _, _ = _run_concat(
            tmp_path, monkeypatch, io_readable=True, tick=30.0, copy_s=1797.0, rewrite_s=7200.0,
            size=GALLEY_LF1_BYTES,
        )  # fmt: skip
        assert result.success and result.killed_reason is None
        assert proc.terminated is False

    def test_a_rewrite_whose_io_counters_freeze_is_killed_as_a_stall(self, tmp_path, monkeypatch):
        result, proc, clock, out = _run_concat(
            tmp_path, monkeypatch, io_readable=True, tick=30.0, copy_s=600.0, rewrite_s=10**9,
            size=GALLEY_LF1_BYTES, freeze_io_in_rewrite=True,
        )  # fmt: skip
        assert proc.terminated is True
        assert result.killed_reason == "stall"
        assert not out.exists()
        # dead within the normal rewrite floor (plus a couple of polls), not the size window,
        # and never before the floor: a plain 600 s stall window must not kill a rewrite
        assert clock.t - proc.moov_at <= MOOV_REWRITE_MIN_STALL_S + 90
        assert clock.t - proc.moov_at >= MOOV_REWRITE_MIN_STALL_S - 60

    def test_counters_unreadable_falls_back_and_18_MBps_finishes(
        self, tmp_path, monkeypatch, caplog
    ):
        slow = 2157.0
        with caplog.at_level("WARNING", logger="gomoveshift.pipeline.ffmpeg_runner"):
            result, proc, _, _ = _run_concat(
                tmp_path, monkeypatch, io_readable=False, tick=30.0, copy_s=slow,
                rewrite_s=slow, size=GALLEY_LF1_BYTES,
            )  # fmt: skip
        assert result.success and result.killed_reason is None
        assert proc.terminated is False
        fallback_logs = [r for r in caplog.records if "unreadable" in r.getMessage()]
        assert len(fallback_logs) == 1
        assert fallback_logs[0].levelname == "WARNING"

    def test_fallback_scales_with_the_run_so_a_slower_rewrite_than_copy_finishes(
        self, tmp_path, monkeypatch
    ):
        # copy pass 1797 s, rewrite 1.4x slower (2516 s): past the 1942 s size window, inside 2x elapsed
        result, proc, _, _ = _run_concat(
            tmp_path, monkeypatch, io_readable=False, tick=30.0, copy_s=1797.0, rewrite_s=2516.0,
            size=GALLEY_LF1_BYTES,
        )  # fmt: skip
        assert result.success and result.killed_reason is None
        assert proc.terminated is False

    def test_fallback_still_kills_a_stuck_rewrite_within_its_window(self, tmp_path, monkeypatch):
        result, proc, clock, out = _run_concat(
            tmp_path, monkeypatch, io_readable=False, tick=30.0, copy_s=600.0, rewrite_s=10**9,
            size=GALLEY_LF1_BYTES,
        )  # fmt: skip
        assert proc.terminated is True and result.killed_reason == "stall"
        assert not out.exists()
        assert clock.t - proc.moov_at <= _moov_rewrite_stall_timeout(GALLEY_LF1_BYTES, 600) + 90

    def test_io_counters_that_vanish_mid_rewrite_switch_to_the_fallback_window(
        self, tmp_path, monkeypatch, caplog
    ):
        out = tmp_path / "lf_01.mp4"
        clock = _SimClock()
        proc = _SimConcat(
            clock, out, tick=30.0, copy_s=1797.0, rewrite_s=2516.0, size=GALLEY_LF1_BYTES
        )
        real_sleep = time.sleep

        def _sleep(_s):
            deadline = time.monotonic() + 5.0
            while not proc.stderr.settled() and time.monotonic() < deadline:
                real_sleep(0.0005)

        def _counters(pid):  # readable at detection, unreadable after
            return proc.io if proc.moov_at is None or clock.t - proc.moov_at < 120 else None

        monkeypatch.setattr(
            "gomoveshift.pipeline.ffmpeg_runner.subprocess.Popen", lambda *a, **k: proc
        )
        monkeypatch.setattr("gomoveshift.pipeline.ffmpeg_runner.time.sleep", _sleep)
        monkeypatch.setattr("gomoveshift.pipeline.ffmpeg_runner._read_io_counters", _counters)
        with caplog.at_level("WARNING", logger="gomoveshift.pipeline.ffmpeg_runner"):
            result = run_ffmpeg_encode(
                ["ffmpeg", str(out)], out, expected_duration=0.0, time_source=clock
            )
        assert result.success and proc.terminated is False
        warned = [r for r in caplog.records if "became unreadable" in r.getMessage()]
        assert len(warned) == 1 and warned[0].levelname == "WARNING"

    def test_io_mode_logs_exactly_one_warning_naming_the_mode(self, tmp_path, monkeypatch, caplog):
        with caplog.at_level("WARNING", logger="gomoveshift.pipeline.ffmpeg_runner"):
            result, proc, _, _ = _run_concat(
                tmp_path, monkeypatch, io_readable=True, tick=30.0, copy_s=600.0, rewrite_s=3000.0,
                size=GALLEY_LF1_BYTES,
            )  # fmt: skip
        assert result.success and proc.terminated is False
        warned = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert len(warned) == 1 and "I/O counters" in warned[0], warned

    def test_fallback_window_is_exactly_twice_the_elapsed_run_when_that_is_larger(
        self, tmp_path, monkeypatch
    ):
        assert _moov_rewrite_fallback_timeout(10**9, 600, 3000.0) == 6000
        small = 10**9  # the size window is far below 2x the 3000 s copy pass
        result, proc, clock, _ = _run_concat(
            tmp_path, monkeypatch, io_readable=False, tick=30.0, copy_s=3000.0, rewrite_s=10**9,
            size=small,
        )  # fmt: skip
        assert result.killed_reason == "stall" and proc.terminated
        assert 6000 - 60 <= clock.t - proc.moov_at <= 6000 + 90

    def test_counters_that_restart_below_the_last_total_never_reset_the_stall_clock(
        self, tmp_path, monkeypatch
    ):
        state = {"k": 0}

        def io(proc):
            if proc.moov_at is not None and proc.clock.t - proc.moov_at > 60:
                state["k"] += 1000  # growing, but below the total seen at detection
                return state["k"]
            return proc.io

        result, proc, clock, _ = _run_concat(
            tmp_path, monkeypatch, io_readable=True, io_fn=io, tick=30.0, copy_s=600.0,
            rewrite_s=10**9, size=GALLEY_LF1_BYTES,
        )  # fmt: skip
        assert result.killed_reason == "stall" and proc.terminated
        assert clock.t - proc.moov_at <= MOOV_REWRITE_MIN_STALL_S + 120

    @pytest.mark.skipif(
        not os.path.exists(f"/proc/{os.getpid()}/io"), reason="needs /proc/<pid>/io (Linux)"
    )
    def test_healthy_real_rewrite_logs_one_warning_and_never_a_false_fallback(
        self, tmp_path, caplog
    ):
        """An exited, not-yet-reaped ffmpeg has unreadable /proc/<pid>/io; that is an exit,
        not a counters failure, so a healthy rewrite ends with only the io-mode WARNING."""
        child = tmp_path / "child.py"
        child.write_text(_REAL_REWRITE_CHILD)
        for i in range(5):
            out = tmp_path / f"o{i}.mp4"
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger="gomoveshift.pipeline.ffmpeg_runner"):
                result = run_ffmpeg_encode(
                    [sys.executable, "-I", str(child), str(out), "40"],
                    out,
                    expected_duration=0.0,
                    description="real rewrite",
                )
            assert result.success, result
            warned = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
            assert len(warned) == 1 and "I/O counters" in warned[0], warned


class TestReadIoCounters:
    def test_sums_rchar_and_wchar(self, tmp_path):
        (tmp_path / "77").mkdir()
        (tmp_path / "77" / "io").write_text(
            "rchar: 1000\nwchar: 250\nsyscr: 5\nsyscw: 6\nread_bytes: 0\nwrite_bytes: 0\n"
        )
        assert _read_io_counters(77, proc_root=str(tmp_path)) == 1250

    @pytest.mark.parametrize("content", ["", "garbage", "rchar: x\nwchar: 1\n", "rchar: 5\n"])
    def test_unparseable_content_is_none(self, tmp_path, content):
        (tmp_path / "77").mkdir()
        (tmp_path / "77" / "io").write_text(content)
        assert _read_io_counters(77, proc_root=str(tmp_path)) is None

    def test_missing_process_or_non_int_pid_is_none(self, tmp_path):
        assert _read_io_counters(99999, proc_root=str(tmp_path)) is None
        assert _read_io_counters(None) is None
        assert _read_io_counters(Mock()) is None

    @pytest.mark.skipif(not os.path.exists("/proc/self/io"), reason="needs Linux /proc/<pid>/io")
    def test_real_process_counters_grow_with_a_file_to_file_copy(self, tmp_path):
        src, dst = tmp_path / "a.bin", tmp_path / "b.bin"
        src.write_bytes(b"x" * 4_000_000)
        before = _read_io_counters(os.getpid())
        dst.write_bytes(src.read_bytes())
        after = _read_io_counters(os.getpid())
        assert before is not None and after is not None
        assert after - before >= 8_000_000  # 4 MB read + 4 MB written


# ── run_ffmpeg_quick ──────────────────────────────────────────────────


class TestRunFfmpegQuick:
    """Tests for the subprocess.run wrapper."""

    def test_success(self):
        """Successful command returns CompletedProcess."""
        result = run_ffmpeg_quick(
            ["python3", "-c", "print('hello')"],
            description="test success",
        )
        assert result.returncode == 0
        assert "hello" in result.stdout

    def test_timeout_deletes_output(self, tmp_path):
        """TimeoutExpired deletes the output file."""
        output = tmp_path / "out.mp4"
        output.write_bytes(b"partial data")

        with pytest.raises(RuntimeError, match="timed out"):
            run_ffmpeg_quick(
                ["python3", "-c", "import time; time.sleep(60)"],
                output_path=output,
                timeout=1,
                description="test timeout",
            )
        assert not output.exists(), "Output should be deleted on timeout"

    def test_nonzero_exit_deletes_output(self, tmp_path):
        """Non-zero exit code deletes the output file."""
        output = tmp_path / "out.mp4"
        output.write_bytes(b"partial data")

        with pytest.raises(RuntimeError, match="failed"):
            run_ffmpeg_quick(
                ["python3", "-c", "import sys; sys.exit(1)"],
                output_path=output,
                timeout=5,
                description="test exit",
            )
        assert not output.exists(), "Output should be deleted on non-zero exit"

    def test_no_output_path_no_crash(self):
        """Non-zero exit without output_path doesn't crash."""
        with pytest.raises(RuntimeError):
            run_ffmpeg_quick(
                ["python3", "-c", "import sys; sys.exit(1)"],
                description="test no output",
            )


# ── validate_video ────────────────────────────────────────────────────


class TestValidateVideo:
    """Tests for ffprobe-based video validation."""

    def test_nonexistent_file(self, tmp_path):
        """Non-existent file returns invalid."""
        result = validate_video(tmp_path / "missing.mp4")
        assert not result.valid
        assert "not exist" in result.reason

    def test_empty_file(self, tmp_path):
        """Empty file returns invalid."""
        f = tmp_path / "empty.mp4"
        f.write_bytes(b"")
        result = validate_video(f)
        assert not result.valid
        assert "empty" in result.reason

    def test_truncated_file_no_moov(self, tmp_path):
        """Truncated file (no moov atom) fails ffprobe."""
        f = tmp_path / "truncated.mp4"
        f.write_bytes(b"\x00" * 1024)  # Random bytes, not valid MP4

        result = validate_video(f)
        assert not result.valid
        # ffprobe should fail on this garbage data
        assert result.reason is not None

    def test_valid_video_with_mock(self, tmp_path):
        """Valid video with mocked ffprobe passes validation."""
        f = tmp_path / "good.mp4"
        f.write_bytes(b"fake video")

        probe_output = {
            "format": {"duration": "120.5"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264"},
                {"codec_type": "audio", "codec_name": "aac"},
            ],
        }

        with patch("gomoveshift.pipeline.ffmpeg_runner.subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout=json.dumps(probe_output),
                stderr="",
            )
            result = validate_video(f)

        assert result.valid
        assert result.duration == pytest.approx(120.5)
        assert result.has_video
        assert result.has_audio
        assert result.video_codec == "h264"

    def test_duration_mismatch_fails(self, tmp_path):
        """Duration significantly different from expected fails."""
        f = tmp_path / "short.mp4"
        f.write_bytes(b"fake")

        probe_output = {
            "format": {"duration": "60.0"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264"},
                {"codec_type": "audio", "codec_name": "aac"},
            ],
        }

        with patch("gomoveshift.pipeline.ffmpeg_runner.subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout=json.dumps(probe_output),
                stderr="",
            )
            result = validate_video(f, expected_duration=120.0, duration_tolerance=0.05)

        assert not result.valid
        assert "Duration mismatch" in result.reason

    def test_duration_within_tolerance_passes(self, tmp_path):
        """Duration within tolerance passes."""
        f = tmp_path / "ok.mp4"
        f.write_bytes(b"fake")

        probe_output = {
            "format": {"duration": "118.0"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264"},
                {"codec_type": "audio", "codec_name": "aac"},
            ],
        }

        with patch("gomoveshift.pipeline.ffmpeg_runner.subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout=json.dumps(probe_output),
                stderr="",
            )
            result = validate_video(f, expected_duration=120.0, duration_tolerance=0.05)

        assert result.valid

    def test_missing_audio_when_required_fails(self, tmp_path):
        """Missing audio stream fails when require_audio=True."""
        f = tmp_path / "noaudio.mp4"
        f.write_bytes(b"fake")

        probe_output = {
            "format": {"duration": "120.0"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264"},
            ],
        }

        with patch("gomoveshift.pipeline.ffmpeg_runner.subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout=json.dumps(probe_output),
                stderr="",
            )
            result = validate_video(f, require_audio=True)

        assert not result.valid
        assert "audio" in result.reason.lower()

    def test_missing_audio_when_not_required_passes(self, tmp_path):
        """Missing audio stream passes when require_audio=False."""
        f = tmp_path / "noaudio.mp4"
        f.write_bytes(b"fake")

        probe_output = {
            "format": {"duration": "120.0"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264"},
            ],
        }

        with patch("gomoveshift.pipeline.ffmpeg_runner.subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout=json.dumps(probe_output),
                stderr="",
            )
            result = validate_video(f, require_audio=False)

        assert result.valid

    def test_no_video_stream_fails(self, tmp_path):
        """File with no video stream fails."""
        f = tmp_path / "audioonly.mp4"
        f.write_bytes(b"fake")

        probe_output = {
            "format": {"duration": "120.0"},
            "streams": [
                {"codec_type": "audio", "codec_name": "aac"},
            ],
        }

        with patch("gomoveshift.pipeline.ffmpeg_runner.subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout=json.dumps(probe_output),
                stderr="",
            )
            result = validate_video(f)

        assert not result.valid
        assert "video" in result.reason.lower()


# ── validate_video A/V drift ────────────────────────────


class TestValidateVideoAVDrift:
    """Per-stream A/V duration drift detection.

    validate_video() previously checked only overall container duration.
    An MP4 with severe A/V stream-duration mismatch could still pass if
    the container duration happened to match expected within tolerance.
    These tests cover the per-stream comparison.
    """

    def test_av_drift_within_tolerance_passes(self, tmp_path):
        """Small A/V drift (<2%) passes validation."""
        f = tmp_path / "ok.mp4"
        f.write_bytes(b"fake")

        probe_output = {
            "format": {"duration": "30.1"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "duration": "30.0"},
                {"codec_type": "audio", "codec_name": "aac", "duration": "30.2"},
            ],
        }

        with patch("gomoveshift.pipeline.ffmpeg_runner.subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout=json.dumps(probe_output),
                stderr="",
            )
            result = validate_video(f)

        assert result.valid
        assert result.av_drift == pytest.approx(0.2, abs=0.01)
        assert result.video_duration == pytest.approx(30.0)
        assert result.audio_duration == pytest.approx(30.2)

    def test_av_drift_exceeding_threshold_fails(self, tmp_path):
        """Large A/V drift (>2%) rejects the file."""
        f = tmp_path / "drift.mp4"
        f.write_bytes(b"fake")

        # video 30s / audio 24.5s → drift 5.5s / 30s = 18.3% — well above 2%
        probe_output = {
            "format": {"duration": "30.0"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "duration": "30.0"},
                {"codec_type": "audio", "codec_name": "aac", "duration": "24.5"},
            ],
        }

        with patch("gomoveshift.pipeline.ffmpeg_runner.subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout=json.dumps(probe_output),
                stderr="",
            )
            result = validate_video(f)

        assert not result.valid
        assert "A/V drift" in result.reason
        assert result.av_drift == pytest.approx(5.5, abs=0.01)

    def test_av_drift_check_skipped_when_no_audio(self, tmp_path):
        """With require_audio=False and no audio stream, drift check is bypassed."""
        f = tmp_path / "videoonly.mp4"
        f.write_bytes(b"fake")

        probe_output = {
            "format": {"duration": "30.0"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "duration": "30.0"},
            ],
        }

        with patch("gomoveshift.pipeline.ffmpeg_runner.subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout=json.dumps(probe_output),
                stderr="",
            )
            result = validate_video(f, require_audio=False)

        assert result.valid
        assert result.av_drift == 0.0

    def test_av_drift_check_skipped_when_stream_duration_missing(self, tmp_path):
        """When per-stream duration is absent (container-only), drift check is skipped."""
        f = tmp_path / "nostreamdur.mp4"
        f.write_bytes(b"fake")

        # Streams have no "duration" field — container-only duration is common for
        # some MPEG-TS / streaming containers.
        probe_output = {
            "format": {"duration": "30.0"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264"},
                {"codec_type": "audio", "codec_name": "aac"},
            ],
        }

        with patch("gomoveshift.pipeline.ffmpeg_runner.subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout=json.dumps(probe_output),
                stderr="",
            )
            result = validate_video(f)

        assert result.valid
        assert result.av_drift == 0.0

    def test_av_drift_fields_populated_on_success(self, tmp_path):
        """On a successful validation, per-stream duration fields are populated."""
        f = tmp_path / "ok.mp4"
        f.write_bytes(b"fake")

        probe_output = {
            "format": {"duration": "60.0"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "duration": "60.0"},
                {"codec_type": "audio", "codec_name": "aac", "duration": "60.05"},
            ],
        }

        with patch("gomoveshift.pipeline.ffmpeg_runner.subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout=json.dumps(probe_output),
                stderr="",
            )
            result = validate_video(f)

        assert result.valid
        assert result.video_duration > 0.0
        assert result.audio_duration > 0.0
        assert result.av_drift == pytest.approx(0.05, abs=0.01)


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        # An identical repeat is a stall even though fps/speed vary (ffmpeg 7, blocked muxer).
        (
            [
                "frame=  100 fps=25 size=1024kB time=00:00:04.00 speed=1.0x\n",
                "frame=  100 fps=11 size=1024kB time=00:00:04.00 speed=0.4x\n",
                "frame=  100 fps=25 size=1024kB time=00:00:04.00 speed=1.0x\n",
            ],
            [True, False, False],
        ),
        # time= or size= advancing alone counts.
        (
            [
                "frame=1 size=10kB time=00:00:01.00\n",
                "frame=1 size=20kB time=00:00:01.00\n",
                "frame=1 size=20kB time=00:00:02.00\n",
                "frame=1 size=20kB time=00:00:02.00\n",
            ],
            [True, True, True, False],
        ),
        # Past an hour, and in MB.
        (
            ["frame=1 size=  900kB time=00:59:59.00\n", "frame=1 size=    2MB time=01:00:00.00\n"],
            [True, True],
        ),
        # size=N/A (piped output) falls back to time=.
        (
            ["size=N/A time=00:00:01.00 frame=1\n", "size=N/A time=00:00:02.00 frame=1\n"],
            [True, True],
        ),
        # Neither time= nor size= parses (no packet muxed yet): frame= advancing is progress,
        # a frozen frame= is not.
        (
            [
                "frame=1 fps=1 q=0.0 size=N/A time=N/A bitrate=N/A speed=N/A\n",
                "frame=2 fps=1 q=0.0 size=N/A time=N/A bitrate=N/A speed=N/A\n",
                "frame=3 fps=1 q=0.0 size=N/A time=N/A bitrate=N/A speed=N/A\n",
                "frame=3 fps=9 q=0.0 size=N/A time=N/A bitrate=N/A speed=N/A\n",
            ],
            [True, True, True, False],
        ),
        # ffmpeg pads frame= with spaces ("frame=   10"); the fallback must still read it.
        (
            [
                "frame=   10 fps= 5 q=-1.0 size=N/A time=N/A bitrate=N/A speed=N/A\n",
                "frame=   12 fps= 6 q=-1.0 size=N/A time=N/A bitrate=N/A speed=N/A\n",
                "frame=   12 fps= 7 q=-1.0 size=N/A time=N/A bitrate=N/A speed=N/A\n",
            ],
            [True, True, False],
        ),
        # size= parses while time=N/A: size decides, frame= is not consulted.
        (
            [
                "frame=    5 fps=1 q=0.0 size=     100KiB time=N/A bitrate=N/A speed=N/A\n",
                "frame=    5 fps=1 q=0.0 size=     200KiB time=N/A bitrate=N/A speed=N/A\n",
                "frame=    5 fps=2 q=0.0 size=     200KiB time=N/A bitrate=N/A speed=N/A\n",
            ],
            [True, True, False],
        ),
        # ffmpeg 7 to a file, before the first packet: size=0KiB parses (0), time=N/A, frame=0
        # repeating. Only fps/elapsed vary, so none of the repeats is progress.
        (
            [
                "frame=    0 fps=0.0 q=0.0 size=       0KiB time=N/A bitrate=N/A speed=N/A "
                "elapsed=0:00:00.20\n",
                "frame=    0 fps=0.0 q=0.0 size=       0KiB time=N/A bitrate=N/A speed=N/A "
                "elapsed=0:00:00.40\n",
            ],
            [True, False],
        ),
        # No frame=, time= or size= that parses: the raw tokens decide (identical = no progress).
        (["time=N/A bitrate=1\n", "time=N/A bitrate=2\n"], [True, False]),
        # A frame= appearing after a line that had none must not be compared to the missing one.
        (["time=N/A bitrate=1\n", "frame=1 time=N/A bitrate=1\n"], [True, False]),
        (["size=5XB speed=1\n", "size=6XB speed=1\n"], [True, True]),
    ],
)
def test_stats_progress_advanced_counts_only_a_moved_time_or_size(lines, expected):
    from gomoveshift.pipeline.ffmpeg_runner import _stats_progress_advanced

    last, got = None, []
    for line in lines:
        adv, last = _stats_progress_advanced(line, last)
        got.append(adv)
    assert got == expected
