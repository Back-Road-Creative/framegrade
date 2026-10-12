"""Resume-correctness regression tests for the montage-prepend stage.

The montage-prepend stage (``montage_builder.prepend_montage``) produces
``lf_NN_with_montage.mp4`` by concatenating a 30s branded montage (built
from ``montage_peaks``) onto the assembled segments file
(``lf_NN_segments.mp4``). A crash mid-prepend can leave a partial
``with_montage`` file on disk. On resume, the pipeline must either
regenerate the file cleanly from inputs or fail cleanly — it must NOT
silently ship a truncated or stale artifact.

These tests lock in the defensive contract of ``prepend_montage``:

  1. Clean state → produces a valid output (stubbed ffmpeg).
  2. Partial/truncated ``lf_NN_with_montage.mp4`` already on disk →
     the stage MUST overwrite it (ffmpeg ``-y``), never trust it.
  3. State/checkpoint says done, but file missing →
     ``prepend_montage`` must not be a no-op; it rebuilds from inputs.
  4. Montage source (master_path / peaks / file_boundaries) missing at
     resume → stage returns None (clean fail) rather than emitting a
     raw LF without a montage, which would break the OST t=31s timing.

Authored as regression tests only; no
production changes.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from gomoveshift.pipeline import montage_builder


# ── Helpers ───────────────────────────────────────────────────────────


def _make_peak(ep_idx: int, start: float, end: float):
    """Build a peak stub matching the LFSegment-like duck-type that
    ``translate_segment_to_master`` expects."""
    return SimpleNamespace(
        source_file_index=ep_idx,
        start_time=float(start),
        end_time=float(end),
        duration=float(end - start),
    )


class _FakeCompletedProcess:
    """subprocess.CompletedProcess stand-in."""

    def __init__(self, returncode: int = 0, stderr: bytes = b""):
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = b""


def _install_fake_ffmpeg(monkeypatch, *, emit_bytes: bytes = b"FAKE_MP4_BYTES\n", rc: int = 0):
    """Replace ``subprocess.run`` inside the montage + segment modules so
    tests don't invoke real ffmpeg. Each invocation creates the intended
    output file so downstream concat steps find it."""

    def fake_run(cmd, capture_output=False, timeout=None, **kwargs):  # noqa: ARG001
        # ffprobe READS its last arg; only ffmpeg writes one. Emitting bytes
        # for an ffprobe call would overwrite the very file being probed —
        # build_montage now probes its own output, so this distinction is
        # load-bearing, not cosmetic.
        if cmd and cmd[0] == "ffmpeg":
            out = Path(cmd[-1])
            try:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(emit_bytes)
            except OSError:
                pass
        return _FakeCompletedProcess(returncode=rc)

    # montage_builder no longer calls subprocess itself: the cut primitives
    # (stream_copy_range / concat_demuxer / probe_keyframes) live in the
    # ``keycut`` package, and prepend_montage's final concat now goes through
    # ffmpeg_runner. Patching the shared ``subprocess`` module covers every
    # one of those call sites at once, as the old montage_builder.subprocess
    # patch did — they are all the same module object.
    import keycut.extract
    import keycut.keyframes

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(keycut.extract.subprocess, "run", fake_run)
    monkeypatch.setattr(keycut.keyframes.subprocess, "run", fake_run)
    keycut.clear_keyframe_cache()


@pytest.fixture
def lf_scratch(tmp_path):
    """Build an lf_NN working layout matching production."""
    lf_dir = tmp_path / "day_2026-04-01" / "lf_outputs"
    lf_dir.mkdir(parents=True)

    # Raw segments file — produced by the segment extractor in prod; we only
    # need it to exist on disk for prepend_montage's precondition check.
    raw_lf = lf_dir / "lf_01_segments.mp4"
    raw_lf.write_bytes(b"RAW_LF_BODY")

    # Master source — production refers to the day master which is
    # stream-copied from per-episode sources. Existence suffices for us.
    master = tmp_path / "day_2026-04-01" / "day_2026-04-01_master.mp4"
    master.parent.mkdir(parents=True, exist_ok=True)
    master.write_bytes(b"MASTER_BYTES")

    work_dir = lf_dir / ".lf_01_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    return SimpleNamespace(
        lf_dir=lf_dir,
        raw_lf=raw_lf,
        master=master,
        work_dir=work_dir,
        output_path=lf_dir / "lf_01_with_montage.mp4",
        peaks=[_make_peak(0, 5.0, 20.0), _make_peak(0, 40.0, 55.0)],
        # Boundaries covering both peaks with slack for the centred trim.
        file_boundaries=[(0.0, 600.0)],
    )


# ── Scenario 1: clean state ───────────────────────────────────────────


def test_clean_state_produces_with_montage_file(lf_scratch, monkeypatch):
    """No prior artifact, all inputs present → success, file exists."""
    _install_fake_ffmpeg(monkeypatch)

    assert not lf_scratch.output_path.exists()

    result = montage_builder.prepend_montage(
        raw_lf_path=lf_scratch.raw_lf,
        peaks=lf_scratch.peaks,
        output_path=lf_scratch.output_path,
        work_dir=lf_scratch.work_dir,
        master_path=lf_scratch.master,
        file_boundaries=lf_scratch.file_boundaries,
    )

    assert result == lf_scratch.output_path
    assert lf_scratch.output_path.exists()
    assert lf_scratch.output_path.stat().st_size > 0


# ── Scenario 2: truncated prior artifact (partial prepend) ────────────


def test_truncated_with_montage_is_overwritten_not_trusted(lf_scratch, monkeypatch):
    """A partial ``lf_NN_with_montage.mp4`` from a prior crashed run
    MUST NOT be silently accepted on resume. ``prepend_montage`` is
    stateless — it re-runs ffmpeg which has ``-y`` so the partial
    file is overwritten with fresh content.

    This is the key resume-correctness guarantee: no pre-existence check
    that would let partial bytes leak downstream to the OST stage.
    """
    # Simulate a crash that left a truncated output behind.
    truncated = b"PARTIAL_HEADER_NO_MOOV"
    lf_scratch.output_path.write_bytes(truncated)
    assert lf_scratch.output_path.read_bytes() == truncated

    fresh_bytes = b"FRESH_MP4_FROM_RESUME"
    _install_fake_ffmpeg(monkeypatch, emit_bytes=fresh_bytes)

    result = montage_builder.prepend_montage(
        raw_lf_path=lf_scratch.raw_lf,
        peaks=lf_scratch.peaks,
        output_path=lf_scratch.output_path,
        work_dir=lf_scratch.work_dir,
        master_path=lf_scratch.master,
        file_boundaries=lf_scratch.file_boundaries,
    )

    assert result == lf_scratch.output_path
    # The truncated bytes have been overwritten — stage is resume-safe.
    assert lf_scratch.output_path.read_bytes() != truncated
    assert lf_scratch.output_path.read_bytes() == fresh_bytes


# ── Scenario 3: state says done, file missing ─────────────────────────


def test_state_claims_done_but_file_missing_triggers_rebuild(lf_scratch, monkeypatch):
    """Even if an external checkpoint claims this LF is done, a missing
    ``lf_NN_with_montage.mp4`` means the stage must rebuild — it must
    not short-circuit on a state flag.

    ``prepend_montage`` is stateless by design: it looks only at
    inputs on disk (raw LF, master, peaks). We verify that calling
    it with a ghost output path (parent dir exists, file absent)
    actually produces a new file.
    """
    # Pre-condition: checkpoint-like file says "done" but artifact absent.
    state_file = lf_scratch.lf_dir.parent / "pipeline_state.json"
    state_file.write_text(
        '{"regular_lf_paths": ["'
        + str(lf_scratch.output_path).replace("\\", "\\\\")
        + '"], "phases_completed": ["day_first_bc"]}'
    )
    assert not lf_scratch.output_path.exists(), "output must start missing"

    _install_fake_ffmpeg(monkeypatch)

    result = montage_builder.prepend_montage(
        raw_lf_path=lf_scratch.raw_lf,
        peaks=lf_scratch.peaks,
        output_path=lf_scratch.output_path,
        work_dir=lf_scratch.work_dir,
        master_path=lf_scratch.master,
        file_boundaries=lf_scratch.file_boundaries,
    )

    assert result is not None
    assert lf_scratch.output_path.exists()
    assert lf_scratch.output_path.stat().st_size > 0


# ── Scenario 4: montage source missing at resume ──────────────────────


class TestMontageSourceMissingAtResume:
    """When the inputs the montage is built FROM are no longer present at
    resume time, ``prepend_montage`` must fail cleanly (return None).

    It must NOT:
      * Silently pass through the raw LF body (ships without montage).
      * Produce a zero-byte or partial ``with_montage`` file.

    The OST stage layouts ``regular_lf`` depend on the first 30s being
    a montage — a passthrough would overlay the drawtext on real content
    at t=31-39s, which is the exact failure mode the module docstring
    warns against.
    """

    def test_raw_lf_missing_returns_none(self, lf_scratch, monkeypatch):
        """If the raw LF (upstream segments concat) is gone, fail fast."""
        _install_fake_ffmpeg(monkeypatch)
        lf_scratch.raw_lf.unlink()

        result = montage_builder.prepend_montage(
            raw_lf_path=lf_scratch.raw_lf,
            peaks=lf_scratch.peaks,
            output_path=lf_scratch.output_path,
            work_dir=lf_scratch.work_dir,
            master_path=lf_scratch.master,
            file_boundaries=lf_scratch.file_boundaries,
        )

        assert result is None
        assert not lf_scratch.output_path.exists()

    def test_master_path_none_returns_none(self, lf_scratch, monkeypatch):
        """No master → no peak source → module contract says return None
        (never silently ship raw LF as the final output)."""
        _install_fake_ffmpeg(monkeypatch)

        result = montage_builder.prepend_montage(
            raw_lf_path=lf_scratch.raw_lf,
            peaks=lf_scratch.peaks,
            output_path=lf_scratch.output_path,
            work_dir=lf_scratch.work_dir,
            master_path=None,
            file_boundaries=lf_scratch.file_boundaries,
        )

        assert result is None
        assert not lf_scratch.output_path.exists()

    def test_empty_peaks_returns_none(self, lf_scratch, monkeypatch):
        """No peaks → no montage can be built → fail cleanly."""
        _install_fake_ffmpeg(monkeypatch)

        result = montage_builder.prepend_montage(
            raw_lf_path=lf_scratch.raw_lf,
            peaks=[],
            output_path=lf_scratch.output_path,
            work_dir=lf_scratch.work_dir,
            master_path=lf_scratch.master,
            file_boundaries=lf_scratch.file_boundaries,
        )

        assert result is None
        assert not lf_scratch.output_path.exists()

    def test_file_boundaries_none_returns_none(self, lf_scratch, monkeypatch):
        """Missing boundaries means peak translation impossible."""
        _install_fake_ffmpeg(monkeypatch)

        result = montage_builder.prepend_montage(
            raw_lf_path=lf_scratch.raw_lf,
            peaks=lf_scratch.peaks,
            output_path=lf_scratch.output_path,
            work_dir=lf_scratch.work_dir,
            master_path=lf_scratch.master,
            file_boundaries=None,
        )

        assert result is None
        assert not lf_scratch.output_path.exists()


# ── Cross-scenario invariant ──────────────────────────────────────────


def test_ffmpeg_failure_leaves_no_valid_output(lf_scratch, monkeypatch):
    """If ffmpeg returns non-zero during the prepend step, the stage
    must return None. Any stale bytes at the output path are the
    caller's problem (orchestrator should re-drive the whole LF) —
    the key invariant is: the function never reports success without
    a real ffmpeg success.
    """

    # First ffmpeg (_stream_copy_range for montage clip) OK, then final
    # concat must fail. Simplest: make all ffmpeg runs fail.
    def always_fail(cmd, capture_output=False, timeout=None, **kwargs):  # noqa: ARG001
        return _FakeCompletedProcess(returncode=1, stderr=b"fake ffmpeg failure")

    import keycut.extract
    import keycut.keyframes

    monkeypatch.setattr(subprocess, "run", always_fail)
    monkeypatch.setattr(keycut.extract.subprocess, "run", always_fail)
    monkeypatch.setattr(keycut.keyframes.subprocess, "run", always_fail)
    keycut.clear_keyframe_cache()

    result = montage_builder.prepend_montage(
        raw_lf_path=lf_scratch.raw_lf,
        peaks=lf_scratch.peaks,
        output_path=lf_scratch.output_path,
        work_dir=lf_scratch.work_dir,
        master_path=lf_scratch.master,
        file_boundaries=lf_scratch.file_boundaries,
    )

    assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
