"""
FFmpeg Runner — central module for all FFmpeg subprocess execution.

Replaces scattered subprocess.run calls with:
  - run_ffmpeg_encode(): Long encodes with Popen, progress monitoring, stall detection
  - run_ffmpeg_quick(): Short operations (ffprobe, stream copy, audio measurement)
  - validate_video(): ffprobe-based integrity check (moov atom, streams, duration)

Key improvements over raw subprocess.run:
  1. Stall detection kills hung processes (no more hard timeout guessing)
  2. All failure paths clean up partial output files
  3. Progress logging during multi-hour encodes
  4. Moov atom validation catches corrupt outputs before downstream stages
"""

import json
import logging
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Cap on captured stdout characters kept in memory per encode. Only the tail is
# retained so a multi-hour encode emitting `-progress pipe:1` records cannot
# grow the buffer without bound; the drain itself still reads everything.
_STDOUT_TAIL_MAX_CHARS = 1_048_576


@dataclass
class FFmpegResult:
    """Result of a long-running FFmpeg encode."""

    success: bool
    returncode: int
    wall_time: float
    stderr_tail: str
    killed_reason: Optional[str] = None  # "stall", "max_timeout", None
    last_progress_time: float = 0.0
    last_speed: float = 0.0
    # Captured subprocess stdout (bounded tail). Empty for the common case (ffmpeg writes
    # encoded media to its -o argument and emits nothing on stdout). Populated
    # for any caller that pipes ffmpeg/ffprobe output through stdout (e.g.
    # inline `loudnorm=print_format=json` measurement or `ffprobe -of json`
    # through run_ffmpeg_encode). The stdout pipe is drained in a background
    # thread regardless of caller use so the pipe buffer (~64 KB on Linux)
    # cannot fill and deadlock the child process.
    stdout: str = ""  # tail only, see _STDOUT_TAIL_MAX_CHARS


@dataclass
class VideoValidation:
    """Result of ffprobe-based video validation."""

    valid: bool
    reason: Optional[str] = None
    duration: float = 0.0
    has_video: bool = False
    has_audio: bool = False
    video_codec: str = ""
    file_size: int = 0
    audio_duration: float = 0.0
    video_duration: float = 0.0
    av_drift: float = 0.0


def _parse_time_to_seconds(time_str: str) -> Optional[float]:
    """Parse ffmpeg time string (HH:MM:SS.ss or SS.ss) to seconds."""
    time_str = time_str.strip()
    if not time_str:
        return None
    try:
        parts = time_str.split(":")
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        else:
            return float(parts[0])
    except (ValueError, IndexError):
        return None


def _stderr_reader(pipe, queue: Queue):
    """Thread target: read stderr lines into a queue."""
    try:
        for line in iter(pipe.readline, ""):
            queue.put(line)
    except (ValueError, OSError):
        pass
    finally:
        queue.put(None)  # Sentinel


def _stdout_drain(pipe, buffer: list[str]):
    """Thread target: drain stdout into a buffer.

    ffmpeg/ffprobe normally write encoded media to a -o file and emit
    nothing on stdout, but some flags (`-progress pipe:1`, `loudnorm
    print_format=json`, `ffprobe -of json`) do emit on stdout. With
    stdout=PIPE and no drainer, the OS-level pipe buffer (~64 KB on
    Linux) fills and blocks the child mid-write — manifesting as a
    silent hang that the existing stall-detector eventually kills with
    no useful diagnostic. Draining unconditionally is cheap (one
    background thread) and makes the failure mode unrepresentable.

    Only the last ``_STDOUT_TAIL_MAX_CHARS`` characters are retained; older
    chunks are dropped so memory stays bounded on long encodes.
    """
    limit = _STDOUT_TAIL_MAX_CHARS
    held = 0
    try:
        # Read in chunks rather than lines — stdout may be binary-ish
        # (ffmpeg -progress pipe:1 emits k=v\n records that are fine to
        # read as text, but ffprobe -of json emits one large JSON blob
        # with no newlines at the end). chunk-reads handle both.
        while True:
            chunk = pipe.read(8192)
            if not chunk:
                break
            buffer.append(chunk)
            held += len(chunk)
            while held > limit:
                excess = held - limit
                if len(buffer[0]) <= excess:
                    held -= len(buffer.pop(0))
                else:
                    buffer[0] = buffer[0][excess:]
                    held -= excess
    except (ValueError, OSError):
        pass


def _stall_kill_reason(
    now: float,
    start_time: float,
    last_progress_time: float,
    max_timeout: float,
    stall_timeout: float,
) -> Optional[str]:
    """Decide whether the encode should be killed at clock reading *now*.

    Pure function — it takes the clock reading as an argument and performs no
    I/O — so stall/timeout behavior is deterministically testable with an
    explicit clock instead of real elapsed wall time (which flakes under CPU
    contention). Returns "max_timeout" when the absolute wall-time ceiling is
    exceeded (checked first), "stall" when no progress has been observed within
    *stall_timeout* seconds, else None. The stall window is measured from the
    LAST observed progress, so continuous progress keeps an encode alive
    regardless of total elapsed time.
    """
    if now - start_time > max_timeout:
        return "max_timeout"
    if now - last_progress_time > stall_timeout:
        return "stall"
    return None


# ffmpeg prints nothing while +faststart shifts the whole file to put the moov
# atom first, so the stats-line stall clock cannot see that phase. The runner
# times it by the ffmpeg child's own I/O instead: ``/proc/<pid>/io`` ``rchar`` +
# ``wchar`` (bytes passed through read()/write()) grow for as long as the rewrite
# is moving bytes, on any volume speed, and growth resets the stall clock exactly
# like an advancing stats line. (Measured for a file-to-file ``-c copy
# -movflags +faststart``: both counters climb through the rewrite pass;
# ``read_bytes``/``write_bytes`` do not, they count page-cache misses and writeback.)
# A rewrite whose counters stop is killed after MOOV_REWRITE_MIN_STALL_S.
#
# Where the counters cannot be read (non-Linux, permission error) the window is
# the larger of the size-based window and twice the run's elapsed time so far:
# the rewrite moves the bytes the copy pass just wrote, so a slow volume scales
# with itself. The size-based term is a floor for a fast volume only; it is not a
# speed assumption (a 118 GB day master rewrote at ~52 MB/s on /mnt/c, and the
# Galley Head 38.8 GB LF copied at ~21.6 MB/s, below a fixed 20 MB/s floor's
# headroom on the rewrite).
MOOV_REWRITE_MIN_STALL_S = 1800
MOOV_REWRITE_MIN_BYTES_PER_SEC = 20_000_000
_IO_UNREADABLE_READS = 3  # consecutive loop passes (>= 0.5 s real) before the fallback


def _moov_rewrite_stall_timeout(output_size_bytes: int, stall_timeout: int) -> int:
    """Size-based stall window for the silent moov rewrite of an *output_size_bytes* file."""
    by_size = -(-output_size_bytes // MOOV_REWRITE_MIN_BYTES_PER_SEC)
    return max(stall_timeout, MOOV_REWRITE_MIN_STALL_S, by_size)


def _moov_rewrite_fallback_timeout(
    output_size_bytes: int, stall_timeout: int, elapsed_s: float
) -> int:
    """Window when ffmpeg's I/O counters are unreadable: it scales with this run's own speed."""
    return max(
        _moov_rewrite_stall_timeout(output_size_bytes, stall_timeout), int(2 * max(elapsed_s, 0.0))
    )


def _read_io_counters(pid, proc_root: str = "/proc") -> Optional[int]:
    """``rchar + wchar`` of process *pid*, or None when ``<proc_root>/<pid>/io`` is unreadable.

    None covers non-Linux, a permission error, an exited process and unparseable content:
    the caller then falls back to the elapsed-time window rather than guessing.
    """
    if not isinstance(pid, int) or isinstance(pid, bool):
        return None
    try:
        with open(f"{proc_root}/{pid}/io") as fh:
            fields = dict(line.split(":", 1) for line in fh.read().splitlines() if ":" in line)
        return int(fields["rchar"]) + int(fields["wchar"])
    except (OSError, ValueError, KeyError):
        return None


_PF_EXITING = 0x00000004  # include/linux/sched.h: the task is in do_exit()


def _child_exiting(pid, proc_root: str = "/proc") -> bool:
    """True when ``<proc_root>/<pid>/stat`` shows *pid* exiting: state Z or X, or PF_EXITING set.

    An ffmpeg that has begun to exit drops its I/O counters (``/proc/<pid>/io`` turns
    unreadable) before ``waitpid`` can report the exit, for as long as its teardown takes: a
    flush of a large output to a slow disk held that gap for 2.4-7.0 s (rr33). Fields are
    read after the LAST ``)``, since the command name may hold spaces and parentheses. An
    unreadable or unparseable stat is not proof of an exit: False.
    """
    if not isinstance(pid, int) or isinstance(pid, bool):
        return False
    try:
        with open(f"{proc_root}/{pid}/stat") as fh:
            rest = fh.read().rsplit(")", 1)[1].split()
        state, flags = rest[0], int(rest[6])
    except (OSError, ValueError, IndexError):
        return False
    return state in ("Z", "X") or bool(flags & _PF_EXITING)


_STATS_TIME_RE = re.compile(r"time=\s*(\S+)")
_STATS_SIZE_RE = re.compile(r"size=\s*(\d+)\s*([A-Za-z]*)")
_STATS_FRAME_RE = re.compile(r"frame=\s*(\d+)")
_SIZE_UNIT_BYTES = {"": 1, "B": 1, "KB": 1024, "KIB": 1024, "MB": 1024**2, "MIB": 1024**2}


def _stats_progress_advanced(line: str, last: Optional[tuple]) -> tuple[bool, Optional[tuple]]:
    """Did this stats line move past the last COUNTED one? Returns ``(advanced, new_last)``.

    ffmpeg 7 re-prints an identical stats line (frame, size and time frozen; only
    fps/speed/elapsed vary) while a muxer write blocks, so a line counts as progress only
    when its ``time=`` or ``size=`` grew. ffmpeg 7 writing to a file prints
    ``frame=    0 ... size=       0KiB time=N/A`` until the first packet is muxed: ``size=0KiB``
    parses, so that line is judged by ``size=``. Only a line with neither parseable (pipe and
    other non-seekable outputs print ``size=N/A time=N/A``) falls back to ``frame=`` growth,
    and with no ``frame=`` either, to its raw tokens differing.
    """
    t_m, s_m = _STATS_TIME_RE.search(line), _STATS_SIZE_RE.search(line)
    secs = _parse_time_to_seconds(t_m.group(1)) if t_m else None
    unit = _SIZE_UNIT_BYTES.get(s_m.group(2).upper()) if s_m else None
    size = int(s_m.group(1)) * unit if unit else None
    f_m = _STATS_FRAME_RE.search(line)
    frame = int(f_m.group(1)) if f_m else None
    cur = (secs, size, (t_m.group(1) if t_m else None, s_m.group(0) if s_m else None), frame)
    if last is None:
        return True, cur
    if secs is None and size is None:
        if frame is not None and last[3] is not None:
            advanced = frame > last[3]
        else:
            advanced = cur[2] != last[2]
    else:
        advanced = (secs is not None and (last[0] is None or secs > last[0])) or (
            size is not None and (last[1] is None or size > last[1])
        )
    return advanced, (cur if advanced else last)


def run_ffmpeg_encode(
    cmd: list[str],
    output_path: Path,
    expected_duration: float,
    description: str = "ffmpeg encode",
    stall_timeout: int = 600,
    max_timeout: int = 86400,
    progress_callback: Optional[Callable] = None,
    log_interval: int = 30,
    time_source: Callable[[], float] = time.monotonic,
) -> FFmpegResult:
    """
    Run a long FFmpeg encode with real-time progress monitoring and stall detection.

    Uses Popen with a threaded stderr reader. Kills the process if no progress
    output is seen for stall_timeout seconds. Cleans up partial output on failure.

    Args:
        cmd: FFmpeg command as list of strings
        output_path: Path to the output file (deleted on failure)
        expected_duration: Expected output duration in seconds (for progress %)
        description: Human-readable description for logging
        stall_timeout: Kill if no progress for this many seconds (default 10 min)
        max_timeout: Absolute maximum wall time (default 24h)
        progress_callback: Optional callback(current_secs, total_secs, speed, fps)
        log_interval: Seconds between progress log lines (default 30s)
        time_source: Monotonic clock used for stall/timeout decisions. Defaults
            to time.monotonic; injectable so tests can drive time deterministically.

    Returns:
        FFmpegResult with success status and details
    """
    output_path = Path(output_path)
    start_time = time_source()
    last_progress_time = start_time
    last_log_time = start_time
    last_speed = 0.0
    last_fps = 0.0
    current_secs = 0.0
    stderr_lines: list[str] = []
    killed_reason = None
    in_moov_rewrite = False
    last_counted_stats: Optional[tuple] = None
    effective_stall_timeout = stall_timeout
    io_mode = False  # in a moov rewrite: True = follow the child's I/O counters
    last_io_total: Optional[int] = None
    io_unreadable_reads = 0

    logger.info(f"  {description}: starting (stall={stall_timeout}s, max={max_timeout}s)")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as e:
        _cleanup_output(output_path)
        return FFmpegResult(
            success=False,
            returncode=-1,
            wall_time=0.0,
            stderr_tail=str(e),
            killed_reason=None,
        )

    # Start threaded stderr reader
    stderr_queue: Queue = Queue()
    reader_thread = threading.Thread(target=_stderr_reader, args=(proc.stderr, stderr_queue))
    reader_thread.daemon = True
    reader_thread.start()

    # Start threaded stdout drainer. See _stdout_drain docstring.
    stdout_chunks: list[str] = []
    stdout_thread = threading.Thread(target=_stdout_drain, args=(proc.stdout, stdout_chunks))
    stdout_thread.daemon = True
    stdout_thread.start()

    try:
        while True:
            now = time_source()

            if in_moov_rewrite and io_mode:
                io_total = _read_io_counters(getattr(proc, "pid", None))
                if io_total is None:
                    # An exiting child's /proc/<pid>/io turns unreadable (its mm is gone) before
                    # waitpid can report the exit, for as long as its teardown takes (2.4-7.0 s
                    # flushing a large output, rr33). A child /proc/<pid>/stat shows exiting is
                    # not a counters failure; otherwise only a run of unreadable reads is.
                    io_unreadable_reads += 1
                    if (
                        io_unreadable_reads >= _IO_UNREADABLE_READS
                        and proc.poll() is None
                        and not _child_exiting(getattr(proc, "pid", None))
                    ):
                        io_mode = False
                        effective_stall_timeout = _moov_rewrite_fallback_timeout(
                            _output_size(output_path), stall_timeout, now - start_time
                        )
                        logger.warning(
                            "⚠ ffmpeg_runner: %s: ffmpeg's I/O counters became unreadable during "
                            "the moov rewrite; stall window is now %ss (max of the size window "
                            "and 2x elapsed)",
                            description,
                            effective_stall_timeout,
                        )
                else:
                    io_unreadable_reads = 0
                if io_total is not None and (last_io_total is None or io_total > last_io_total):
                    last_io_total = io_total
                    last_progress_time = now

            kill_reason = _stall_kill_reason(
                now, start_time, last_progress_time, max_timeout, effective_stall_timeout
            )

            # Check max timeout
            if kill_reason == "max_timeout":
                killed_reason = "max_timeout"
                logger.error(f"  {description}: max timeout ({max_timeout}s) reached, killing")
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
                break

            # Check stall timeout
            if kill_reason == "stall":
                killed_reason = "stall"
                elapsed = now - last_progress_time
                logger.error(
                    f"  {description}: stalled for {elapsed:.0f}s "
                    f"(limit {effective_stall_timeout}s), killing"
                )
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
                break

            # Drain stderr queue
            got_output = False
            while True:
                try:
                    line = stderr_queue.get_nowait()
                except Empty:
                    break

                if line is None:
                    # Reader thread finished — process is done or closing
                    got_output = True
                    break

                got_output = True
                stderr_lines.append(line)
                # Keep only last 200 lines
                if len(stderr_lines) > 200:
                    stderr_lines = stderr_lines[-200:]

                # Detect moov rewrite phase
                if "Starting second pass" in line or "moving the moov atom" in line.lower():
                    if not in_moov_rewrite:
                        in_moov_rewrite = True
                        output_size = _output_size(output_path)
                        last_io_total = _read_io_counters(getattr(proc, "pid", None))
                        io_mode = last_io_total is not None
                        if io_mode:
                            effective_stall_timeout = max(stall_timeout, MOOV_REWRITE_MIN_STALL_S)
                            logger.warning(
                                "⚠ ffmpeg_runner: %s: moov atom rewrite phase detected "
                                "(trigger='Starting second pass' / 'moving the moov atom'); the "
                                "stall clock now follows ffmpeg's own I/O counters "
                                "(/proc/%s/io rchar+wchar): any growth counts as progress, "
                                "none for %ss kills it, for a %.1f GB output",
                                description,
                                proc.pid,
                                effective_stall_timeout,
                                output_size / 1e9,
                            )
                        else:
                            effective_stall_timeout = _moov_rewrite_fallback_timeout(
                                output_size, stall_timeout, now - start_time
                            )
                            logger.warning(
                                "⚠ ffmpeg_runner: %s: moov atom rewrite phase detected "
                                "(trigger='Starting second pass' / 'moving the moov atom') and "
                                "ffmpeg's I/O counters are unreadable: stall window falls back "
                                "to %ss = max(size window, 2x the %.0fs this run has taken so "
                                "far) for a %.1f GB output",
                                description,
                                effective_stall_timeout,
                                now - start_time,
                                output_size / 1e9,
                            )

                # Parse progress lines
                # ffmpeg outputs: frame= 1234 fps= 142 ... time=00:45:30.00 ... speed=1.2x
                if "time=" in line and ("frame=" in line or "size=" in line):
                    advanced, last_counted_stats = _stats_progress_advanced(
                        line, last_counted_stats
                    )
                    if not advanced:
                        continue
                    last_progress_time = now

                    # Reset moov rewrite stall extension once we see frame progress again
                    if in_moov_rewrite and "frame=" in line:
                        in_moov_rewrite = False
                        io_mode = False
                        effective_stall_timeout = stall_timeout

                    # Parse time
                    time_match = re.search(r"time=(\S+)", line)
                    if time_match:
                        parsed = _parse_time_to_seconds(time_match.group(1))
                        if parsed is not None:
                            current_secs = parsed

                    # Parse speed
                    speed_match = re.search(r"speed=\s*([\d.]+)x", line)
                    if speed_match:
                        last_speed = float(speed_match.group(1))

                    # Parse fps
                    fps_match = re.search(r"fps=\s*([\d.]+)", line)
                    if fps_match:
                        last_fps = float(fps_match.group(1))

                    # Callback
                    if progress_callback:
                        try:
                            progress_callback(current_secs, expected_duration, last_speed, last_fps)
                        except Exception:
                            pass

                    # Periodic progress logging
                    if now - last_log_time >= log_interval and expected_duration > 0:
                        pct = (
                            (current_secs / expected_duration * 100) if expected_duration > 0 else 0
                        )
                        remaining_secs = (
                            (expected_duration - current_secs) / last_speed if last_speed > 0 else 0
                        )
                        eta_min = remaining_secs / 60 if remaining_secs > 0 else 0

                        _cur_m, _cur_s = divmod(int(current_secs), 60)
                        _tot_m, _tot_s = divmod(int(expected_duration), 60)
                        logger.info(
                            f"  Progress: {_cur_m}:{_cur_s:02d} / {_tot_m}:{_tot_s:02d} "
                            f"({pct:.1f}%) — {last_fps:.0f}fps, {last_speed:.1f}x, "
                            f"ETA ~{eta_min:.0f}min"
                        )
                        last_log_time = now

            # Check if process exited
            retcode = proc.poll()
            if retcode is not None:
                # Drain remaining stderr
                while True:
                    try:
                        line = stderr_queue.get(timeout=1)
                    except Empty:
                        break
                    if line is None:
                        break
                    stderr_lines.append(line)
                break

            if not got_output:
                time.sleep(0.25)

    except Exception as e:
        logger.error(f"  {description}: unexpected error: {e}")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        _cleanup_output(output_path)
        return FFmpegResult(
            success=False,
            returncode=-1,
            wall_time=time_source() - start_time,
            stderr_tail=str(e),
        )

    wall_time = time_source() - start_time
    returncode = proc.returncode if proc.returncode is not None else -1
    stderr_tail = "".join(stderr_lines[-50:])

    # Determine success
    success = returncode == 0 and killed_reason is None

    if not success:
        _cleanup_output(output_path)
        if killed_reason:
            logger.error(f"  {description}: killed ({killed_reason}) after {wall_time:.0f}s")
        else:
            logger.error(f"  {description}: failed with returncode {returncode}")
            if stderr_tail:
                logger.error(f"  stderr tail:\n{stderr_tail[-1500:]}")

    reader_thread.join(timeout=5)
    stdout_thread.join(timeout=5)
    captured_stdout = "".join(stdout_chunks)

    return FFmpegResult(
        success=success,
        returncode=returncode,
        wall_time=wall_time,
        stderr_tail=stderr_tail[-1500:],
        killed_reason=killed_reason,
        last_progress_time=last_progress_time - start_time,
        last_speed=last_speed,
        stdout=captured_stdout,
    )


def _output_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def run_ffmpeg_retry_enomem(
    cmd: list[str],
    output_path: Path,
    timeout: int,
    text: bool = True,
) -> subprocess.CompletedProcess:
    """Run ffmpeg once more when the first run fails with ENOMEM.

    A large write onto a WSL drvfs mount (``/mnt/c``) can fail with
    ``Cannot allocate memory`` (``error code: -12``) from the filesystem while RAM
    is free. That is transient, so the partial output is removed and the same
    command runs again, at most twice in all. Any other failure, and a second
    ENOMEM, return at once; the caller decides what a non-zero exit means.
    """
    result = subprocess.run(cmd, capture_output=True, text=text, timeout=timeout)
    if result.returncode != 0 and _is_enomem(result.stderr):
        logger.warning(
            "ffmpeg hit ENOMEM (Cannot allocate memory) writing %s; "
            "removing the partial output and retrying once",
            output_path,
        )
        Path(output_path).unlink(missing_ok=True)
        result = subprocess.run(cmd, capture_output=True, text=text, timeout=timeout)
    return result


def _is_enomem(stderr) -> bool:
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    return bool(stderr) and ("Cannot allocate memory" in stderr or "error code: -12" in stderr)


def run_ffmpeg_quick(
    cmd: list[str],
    output_path: Optional[Path] = None,
    timeout: int = 300,
    description: str = "ffmpeg",
) -> subprocess.CompletedProcess:
    """
    Run a short FFmpeg/ffprobe command with timeout and cleanup.

    On timeout or non-zero exit: deletes output_path if it exists and raises RuntimeError.

    Args:
        cmd: Command as list of strings
        output_path: Optional output file to clean up on failure
        timeout: Timeout in seconds (default 5 min)
        description: Human-readable description for error messages

    Returns:
        subprocess.CompletedProcess on success

    Raises:
        RuntimeError: On timeout or non-zero exit code
    """
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        if output_path:
            _cleanup_output(Path(output_path))
        raise RuntimeError(
            f"{description}: timed out after {timeout}s. Command: {' '.join(cmd[:6])}..."
        )

    if result.returncode != 0:
        if output_path:
            _cleanup_output(Path(output_path))
        stderr_tail = result.stderr[-500:] if result.stderr else "No error output"
        raise RuntimeError(f"{description} failed (rc={result.returncode}):\n{stderr_tail}")

    return result


def validate_video(
    video_path: Path,
    expected_duration: Optional[float] = None,
    duration_tolerance: float = 0.05,
    require_audio: bool = True,
) -> VideoValidation:
    """
    Validate a video file using ffprobe.

    Checks:
      - File exists and is non-empty
      - ffprobe can parse it (moov atom present for MP4)
      - Has at least one video stream
      - Has audio stream (if require_audio=True)
      - Duration within tolerance of expected (if provided)

    Args:
        video_path: Path to video file
        expected_duration: Expected duration in seconds (optional)
        duration_tolerance: Allowed fractional deviation (default 5%)
        require_audio: Whether to require an audio stream

    Returns:
        VideoValidation with detailed results
    """
    video_path = Path(video_path)

    if not video_path.exists():
        return VideoValidation(valid=False, reason="File does not exist")

    file_size = video_path.stat().st_size
    if file_size == 0:
        return VideoValidation(valid=False, reason="File is empty (0 bytes)")

    # Run ffprobe
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return VideoValidation(
            valid=False,
            reason="ffprobe timed out (possible corrupt file)",
            file_size=file_size,
        )

    if result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else "unknown error"
        return VideoValidation(
            valid=False,
            reason=f"ffprobe failed: {stderr[:200]}",
            file_size=file_size,
        )

    try:
        probe = json.loads(result.stdout)
    except json.JSONDecodeError:
        return VideoValidation(
            valid=False,
            reason="ffprobe returned invalid JSON",
            file_size=file_size,
        )

    # Check for streams
    streams = probe.get("streams", [])
    has_video = any(s.get("codec_type") == "video" for s in streams)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    video_codec = ""
    for s in streams:
        if s.get("codec_type") == "video":
            video_codec = s.get("codec_name", "")
            break

    if not has_video:
        return VideoValidation(
            valid=False,
            reason="No video stream found",
            has_audio=has_audio,
            video_codec=video_codec,
            file_size=file_size,
        )

    if require_audio and not has_audio:
        return VideoValidation(
            valid=False,
            reason="No audio stream found (required)",
            has_video=has_video,
            video_codec=video_codec,
            file_size=file_size,
        )

    # Per-stream A/V duration drift check.
    # Some containers omit per-stream "duration" — in that case we fall back to the
    # overall container duration check below and skip this one.
    video_stream_dur = 0.0
    audio_stream_dur = 0.0
    for s in streams:
        codec_type = s.get("codec_type")
        try:
            stream_dur = float(s.get("duration", 0) or 0)
        except (TypeError, ValueError):
            stream_dur = 0.0
        if codec_type == "video" and stream_dur > 0:
            video_stream_dur = stream_dur
        elif codec_type == "audio" and stream_dur > 0:
            audio_stream_dur = stream_dur

    drift = 0.0
    if video_stream_dur > 0 and audio_stream_dur > 0:
        drift = abs(video_stream_dur - audio_stream_dur)
        longer = max(video_stream_dur, audio_stream_dur)
        drift_pct = drift / longer if longer > 0 else 0.0
        if drift_pct > 0.02:
            return VideoValidation(
                valid=False,
                reason=(
                    f"A/V drift: video={video_stream_dur:.2f}s "
                    f"audio={audio_stream_dur:.2f}s "
                    f"(drift={drift:.2f}s, {drift_pct:.1%} > 2%)"
                ),
                duration=float(probe.get("format", {}).get("duration", 0)),
                has_video=has_video,
                has_audio=has_audio,
                video_codec=video_codec,
                file_size=file_size,
                audio_duration=audio_stream_dur,
                video_duration=video_stream_dur,
                av_drift=drift,
            )

    # Get duration
    duration = float(probe.get("format", {}).get("duration", 0))

    # Check duration tolerance
    if expected_duration is not None and expected_duration > 0 and duration > 0:
        deviation = abs(duration - expected_duration) / expected_duration
        if deviation > duration_tolerance:
            return VideoValidation(
                valid=False,
                reason=(
                    f"Duration mismatch: {duration:.1f}s vs expected {expected_duration:.1f}s "
                    f"({deviation:.1%} deviation, tolerance {duration_tolerance:.0%})"
                ),
                duration=duration,
                has_video=has_video,
                has_audio=has_audio,
                video_codec=video_codec,
                file_size=file_size,
                audio_duration=audio_stream_dur,
                video_duration=video_stream_dur,
                av_drift=drift,
            )

    return VideoValidation(
        valid=True,
        duration=duration,
        has_video=has_video,
        has_audio=has_audio,
        video_codec=video_codec,
        file_size=file_size,
        audio_duration=audio_stream_dur,
        video_duration=video_stream_dur,
        av_drift=drift,
    )


def check_disk_space(path: Path, required_bytes: int) -> None:
    """Raise RuntimeError if *path*'s filesystem has less than *required_bytes* free.

    Intended as a preflight check before any large encode or concat so we
    fail fast with a clear message instead of producing a partial file
    that has to be restarted from scratch.
    """
    import shutil

    target = Path(path)
    # Walk up to find a directory that exists (output_dir may not yet)
    while not target.exists():
        target = target.parent
    usage = shutil.disk_usage(target)
    if usage.free < required_bytes:
        raise RuntimeError(
            f"Insufficient disk space at {path}: "
            f"{usage.free / 1_073_741_824:.1f} GB free, "
            f"need ~{required_bytes / 1_073_741_824:.1f} GB"
        )


def _cleanup_output(output_path: Path):
    """Delete a partial/corrupt output file if it exists."""
    try:
        if output_path.exists():
            size = output_path.stat().st_size
            output_path.unlink()
            logger.warning(f"  Cleaned up partial output: {output_path.name} ({size} bytes)")
    except OSError as e:
        logger.warning(f"  Failed to clean up {output_path}: {e}")
