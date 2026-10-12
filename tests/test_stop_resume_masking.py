"""Resume must not mask a stop-cut.

On a cache-hit resume, ``StopProcessor.process`` used to return ``stops_found=0``
with no ``keep_segments`` whenever the cut output was already fresh. The stage
then took the "No stops detected" branch: ``current_video`` never switched to the
cut file and STOP-VERIFY was skipped — so the irreversible encode consumed the
ORIGINAL stops-included video and a prior verify failure was silently masked.

The fix persists the cut result beside the output, so a cache-hit resume returns
the real ``keep_segments`` / cut path and the stage switches + verifies exactly as
on a fresh run.
"""

import os
from types import SimpleNamespace

from gomoveshift.config import Config
from gomoveshift.models import DeadSpot
from gomoveshift.pipeline.stop_processor import StopProcessor

_BIG = b"\x00" * (2 * 1024 * 1024)  # above the 1 MB freshness floor
_KEEP = [(0.0, 3.0), (7.0, 12.0)]


def _cfg() -> Config:
    return Config(stop_detection={"min_stop_duration": 2.0, "motion_threshold": 0.008})


def _stub_cut_processor(cache_dir):
    """A StopProcessor whose expensive detect/cut/probe steps are stubbed."""
    proc = StopProcessor(_cfg(), crossfade_duration=0.0, cache_dir=cache_dir)
    stops = [DeadSpot(start_time=3.0, end_time=7.0, motion_energy=0.0, suggested_trim=True)]
    proc.detect_stops = lambda _p: list(stops)
    proc._calculate_keep_segments = lambda _s, _v: list(_KEEP)
    proc._cut_and_stitch = lambda _v, out, _s: out.write_bytes(_BIG)
    proc._get_duration = lambda _p: 12.0  # bypass real ffprobe on the fake source
    return proc


def test_process_cache_hit_returns_persisted_keep_segments(tmp_path):
    """A resumed run whose cut output is fresh returns the real keep_segments,
    not the masking stops_found=0."""
    proc = _stub_cut_processor(tmp_path / "cache")
    src = tmp_path / "drive.mp4"
    src.write_bytes(_BIG)
    out = tmp_path / "drive_no_stops.mp4"

    first = proc.process(src, out)  # fresh run: cuts + persists the sidecar
    assert first["stops_found"] == 1
    assert first["keep_segments"] == _KEEP

    # Output newer than input → is_output_fresh reports a cache hit.
    os.utime(out, (src.stat().st_mtime + 10, src.stat().st_mtime + 10))

    second = proc.process(src, out)  # RESUME
    assert second.get("cached") is True
    assert second["stops_found"] == 1, "cache-hit resume masked the cut as no-stops"
    assert second["keep_segments"] == _KEEP
    assert second["output_path"] == str(out)


def test_cache_reuse_invalidated_when_detection_params_change(tmp_path):
    """A fresh cut is reused on resume only while the detection params that
    produced it match — a changed motion_threshold / min_stop_duration re-derives."""
    proc = _stub_cut_processor(tmp_path / "cache")
    src = tmp_path / "drive.mp4"
    src.write_bytes(_BIG)
    out = tmp_path / "drive_no_stops.mp4"

    proc.process(src, out)  # writes sidecar with detection_params
    os.utime(out, (src.stat().st_mtime + 10, src.stat().st_mtime + 10))

    # Same params → reuse (cache hit).
    assert proc.process(src, out).get("cached") is True

    # Change the threshold → the cached cut no longer reflects config → re-derive
    # (detect_stops runs again; result is not marked cached).
    proc.motion_threshold = 0.05
    redetected = []
    _orig_detect = proc.detect_stops
    proc.detect_stops = lambda p: redetected.append(p) or _orig_detect(p)
    result = proc.process(src, out)
    assert redetected, "changed detection params must force a re-derive, not reuse"
    assert result.get("cached") is not True


def test_stage_cache_hit_switches_current_video_and_runs_verify(tmp_path, monkeypatch):
    """End to end: on a cache-hit resume the stage switches current_video to the
    cut file and runs STOP-VERIFY, instead of the 'No stops detected' branch."""
    import gomoveshift.pipeline.orchestrator as orch
    from gomoveshift.pipeline.stages.stop_removal import StopRemovalStage
    from gomoveshift.pipeline.stop_processor import StopProcessor as RealProc

    stages_dir = tmp_path / "stages"
    stages_dir.mkdir()
    src = tmp_path / "drive.mp4"
    src.write_bytes(_BIG)
    stops_output = stages_dir / "drive_no_stops.mp4"

    class _StubProc(RealProc):
        def detect_stops(self, video_path):
            return [DeadSpot(start_time=3.0, end_time=7.0, motion_energy=0.0, suggested_trim=True)]

        def _calculate_keep_segments(self, stops, video_path):
            return list(_KEEP)

        def _cut_and_stitch(self, video_path, output_path, segments):
            output_path.write_bytes(_BIG)

        def _get_duration(self, video_path):
            return 12.0

    monkeypatch.setattr(orch, "StopProcessor", _StubProc)

    verify_calls = []

    def _spy(output_path, keep_segments, source_duration, cfg, crossfade_duration=0.0, **_):
        verify_calls.append((output_path, keep_segments))

    monkeypatch.setattr("gomoveshift.pipeline.stop_verify.verify_stops_removed", _spy)

    ctx = SimpleNamespace(
        config=_cfg(),
        crossfade_duration=0.0,
        paths=SimpleNamespace(stages_dir=stages_dir),
        gps_path=None,
        video_properties=SimpleNamespace(height=1080),
        project_name="drive",
        current_video=src,
        keep_segments=None,
    )
    stage = StopRemovalStage()

    # Fresh run: real cut + persisted sidecar; current_video switches.
    r1 = stage.run(ctx)
    assert r1.success
    assert ctx.current_video == stops_output

    # Simulate a resume ctx that still points at the stops-included original.
    ctx.current_video = src
    ctx.keep_segments = None
    verify_calls.clear()
    os.utime(stops_output, (src.stat().st_mtime + 10, src.stat().st_mtime + 10))

    r2 = stage.run(ctx)
    assert r2.success
    assert ctx.current_video == stops_output, (
        "cache-hit resume left current_video on the stops-included original"
    )
    assert verify_calls, "STOP-VERIFY was skipped on cache-hit resume"
    assert verify_calls[0][0] == stops_output
    assert list(verify_calls[0][1]) == _KEEP
