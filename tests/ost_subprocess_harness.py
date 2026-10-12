"""Shared patch target for `ost_applier`'s ffmpeg subprocess calls.

Every OST test needs to intercept the same seams: `ost_applier`'s remaining
`subprocess.run` calls (ffprobe, audio-measure, has-audio probe, stream-copy)
AND the main burn, which now runs through
`ffmpeg_runner.run_ffmpeg_encode` (the stall-detecting Popen runner)
instead of a raw `subprocess.run`. Centralizing the redirect here
means the one fake a test already writes for `subprocess.run` keeps
controlling the burn too, adapted to `run_ffmpeg_encode`'s
`FFmpegResult` contract, instead of every test module growing a second
patch target.
"""

from pathlib import Path
from unittest.mock import patch

from gomoveshift.pipeline.ffmpeg_runner import FFmpegResult

OST_SUBPROCESS_RUN = "gomoveshift.pipeline.ost_applier.subprocess.run"
OST_FFMPEG_ENCODE = "gomoveshift.pipeline.ost_applier.run_ffmpeg_encode"


def _discard(output_path) -> None:
    """Mirror run_ffmpeg_encode's own cleanup-on-failure so a fake encode
    behaves like the real runner without every test asserting on it twice.
    Some tests monkeypatch Path.exists globally (True for every path), so
    unlink can legitimately miss a file that was never written — swallow
    that the same way ffmpeg_runner._cleanup_output does."""
    if output_path is None:
        return
    try:
        path = Path(output_path)
        if path.exists():
            path.unlink()
    except OSError:
        pass


def _encode_via(run_stub):
    """Adapt a subprocess.run-shaped stub (the fake every test already
    writes) into a stand-in for run_ffmpeg_encode. Calls the SAME stub with
    the SAME (cmd, capture_output=True, timeout=...) shape the burn used to
    call subprocess.run with, so existing cmd/timeout assertions keep
    meaning; translates its CompletedProcess-shaped return (or a raised
    exception) into the FFmpegResult run_ffmpeg_encode actually returns.
    """

    def _replacement(cmd, output_path=None, expected_duration=None, **kwargs):
        try:
            result = run_stub(cmd, capture_output=True, timeout=kwargs.get("max_timeout"))
        except Exception as exc:  # run_ffmpeg_encode never raises — it returns success=False
            _discard(output_path)
            return FFmpegResult(success=False, returncode=-1, wall_time=0.0, stderr_tail=str(exc))
        stderr = getattr(result, "stderr", b"") or b""
        stderr_text = (
            stderr.decode("utf-8", errors="replace") if isinstance(stderr, bytes) else str(stderr)
        )
        success = getattr(result, "returncode", 1) == 0
        if not success:
            _discard(output_path)
        return FFmpegResult(
            success=success,
            returncode=getattr(result, "returncode", -1),
            wall_time=0.0,
            stderr_tail=stderr_text,
        )

    return _replacement


def patch_ost_run(*args, **kwargs):
    """`unittest.mock.patch` bound to the OST subprocess.run target, with
    `run_ffmpeg_encode` patched alongside it so the same fake drives both.

    Context-manager use only (``with patch_ost_run(...) as mock_run:`` or
    inside a `with (...)` tuple) — decorator use needs the encode patch
    applied explicitly since only one mock can be auto-injected.
    """
    from contextlib import contextmanager

    @contextmanager
    def _cm():
        with patch(OST_SUBPROCESS_RUN, *args, **kwargs) as mock_run:
            with patch(OST_FFMPEG_ENCODE, side_effect=_encode_via(mock_run), create=True):
                yield mock_run

    return _cm()


def monkeypatch_ost_run(monkeypatch, replacement):
    """`monkeypatch.setattr` bound to the OST subprocess.run target, with
    `run_ffmpeg_encode` monkeypatched alongside it (see `patch_ost_run`)."""
    monkeypatch.setattr(OST_SUBPROCESS_RUN, replacement)
    monkeypatch.setattr(OST_FFMPEG_ENCODE, _encode_via(replacement), raising=False)
