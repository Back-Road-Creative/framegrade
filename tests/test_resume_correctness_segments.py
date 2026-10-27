"""Resume-correctness regression tests for segment extraction + concat.

Segment extraction produces `lf_NN_segments.mp4` via stream-copy slices of
the day master followed by an ffmpeg concat-demuxer concat. If a run is
interrupted mid-concat, the output file can be left present-but-partial
(playable but incomplete), and the concat list file can be left pointing
at scratch segments that have already been cleaned up by a subsequent
attempt.

These tests pin the current contract of
`gomoveshift.pipeline.day_pipeline.extract_and_concat_segments`, which wraps
the published `keycut` package:

  1. Clean state — inputs wired and ffmpeg succeeds: output is produced
     and scratch files (per-segment + list) are cleaned up.
  2. Truncated segments file from a prior crash: the extractor does NOT
     skip; it re-invokes ffmpeg, whose `-y` overwrites the partial file.
     (I.e. resume is safe even with no explicit guard in the extractor.)
  3. State claims done but `lf_NN_segments.mp4` is missing on disk: a
     follow-up call re-extracts cleanly — the caller does NOT silently
     succeed on the missing path.
  4. Per-segment scratch inputs missing before concat (e.g. scrubbed
     between extract and concat): ffmpeg concat fails and the extractor
     returns None rather than producing a phantom output.

All subprocess calls are mocked — no real ffmpeg runs.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


# ─────────────────── helpers ────────────────────────────────────────────


@contextmanager
def _patch_ffmpeg(side_effect):
    """Intercept every ffmpeg/ffprobe call the cut layer makes.

    The primitives moved into the ``keycut`` package, which runs subprocesses
    from two modules: ``keycut.extract`` (stream copy, re-encode, concat) and
    ``keycut.keyframes`` (the keyframe probe). The probe result is cached per
    (path, mtime, size) for the life of the process, so the cache is cleared
    around each patch or a later test would reuse an earlier stub's answer.
    """
    import keycut

    keycut.clear_keyframe_cache()
    try:
        with ExitStack() as stack:
            stack.enter_context(patch("keycut.extract.subprocess.run", side_effect=side_effect))
            stack.enter_context(patch("keycut.keyframes.subprocess.run", side_effect=side_effect))
            yield
    finally:
        keycut.clear_keyframe_cache()


def _seg(ep_idx: int, start: float, end: float) -> SimpleNamespace:
    return SimpleNamespace(
        source_file_index=ep_idx,
        start_time=float(start),
        end_time=float(end),
    )


def _is_concat_cmd(cmd: list[str]) -> bool:
    """ffmpeg concat-demuxer invocation has `-f concat` adjacent."""
    for i in range(len(cmd) - 1):
        if cmd[i] == "-f" and cmd[i + 1] == "concat":
            return True
    return False


def _output_path_of(cmd: list[str]) -> Path:
    """ffmpeg invocations always put the output path last."""
    return Path(cmd[-1])


def _make_run_side_effect(
    commands: list[list[str]],
    *,
    concat_inputs_must_exist: bool = True,
    fail_cmd_predicate=None,
):
    """Build a subprocess.run side_effect that records commands and
    emulates ffmpeg on success: creates the output file the command
    targets.

    - `concat_inputs_must_exist=True` mirrors real ffmpeg: if the concat
      list references missing files, the concat step returns rc=1.
    - `fail_cmd_predicate(cmd) -> bool` can force a specific cmd to fail.
    """

    def _run(cmd, *a, **kw):
        cmd_list = list(cmd)
        commands.append(cmd_list)

        if fail_cmd_predicate is not None and fail_cmd_predicate(cmd_list):
            return MagicMock(returncode=1, stdout=b"", stderr=b"forced-fail")

        if _is_concat_cmd(cmd_list):
            # Emulate concat demuxer: parse list file, verify each input exists
            list_path = Path(cmd_list[cmd_list.index("-i") + 1])
            try:
                list_text = list_path.read_text()
            except OSError:
                return MagicMock(returncode=1, stdout=b"", stderr=b"list missing")
            if concat_inputs_must_exist:
                for line in list_text.splitlines():
                    line = line.strip()
                    if not line.startswith("file "):
                        continue
                    # file 'path' — strip quotes
                    path_part = line[len("file ") :].strip().strip("'")
                    if not Path(path_part).exists():
                        return MagicMock(
                            returncode=1,
                            stdout=b"",
                            stderr=b"concat input missing: " + path_part.encode(),
                        )

        # Emulate successful ffmpeg: write the output file.
        out = _output_path_of(cmd_list)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"mp4-body-full")
        return MagicMock(returncode=0, stdout=b"", stderr=b"")

    return _run


# ─────────────────── scenario tests ─────────────────────────────────────


class TestCleanStateProducesOutput:
    """Scenario 1: Clean state — segments file does not exist; extractor
    runs stream-copy per segment + concat; output is produced atomically
    and scratch files are cleaned up."""

    def test_clean_run_produces_output_and_cleans_scratch(self, tmp_path):
        from gomoveshift.pipeline import day_pipeline

        master = tmp_path / "master.mp4"
        master.write_bytes(b"fake-master")
        boundaries = [(0.0, 1200.0), (1200.0, 2100.0)]
        segments = [
            _seg(ep_idx=0, start=100.0, end=160.0),
            _seg(ep_idx=1, start=50.0, end=90.0),
        ]
        out = tmp_path / "lf_01_segments.mp4"
        work = tmp_path / "work"

        commands: list[list[str]] = []
        side = _make_run_side_effect(commands)

        with _patch_ffmpeg(side):
            result = day_pipeline.extract_and_concat_segments(
                master_path=master,
                segments=segments,
                file_boundaries=boundaries,
                output_path=out,
                work_dir=work,
            )

        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0
        # keycut 0.2.0 records an assembly receipt, which costs extra ffprobes:
        # 1 keyframe probe + 1 stream-copy per (non-contiguous) segment
        # + 1 duration probe per piece (receipt span lengths) + 1 concat
        # + 1 duration probe of the output (receipt output.duration).
        assert len(commands) == 1 + len(segments) + len(segments) + 1 + 1
        assert [c[0] for c in commands].count("ffprobe") == 1 + len(segments) + 1
        assert commands[0][0] == "ffprobe"
        # Scratch per-segment files were cleaned up (keycut names them
        # ``keycut_NNN.mp4``; the old in-tree extractor used ``seg_NN.mp4``)
        assert not list(work.glob("keycut_*.mp4"))
        assert not list(work.glob("*.mp4"))
        # Concat list file was cleaned up
        assert not list(work.glob("*_concat.txt"))
        assert not list(work.glob("*.txt"))


class TestTruncatedSegmentsFileIsOverwritten:
    """Scenario 2: A prior crash mid-concat left `lf_NN_segments.mp4`
    truncated (50% bytes). The extractor has NO resume-skip guard in
    `extract_and_concat_segments`, so it must re-run ffmpeg and overwrite
    the partial file via `-y`. That is the safe behaviour — silently
    trusting a partial file would ship broken LFs."""

    def test_truncated_output_is_reextracted_and_overwritten(self, tmp_path):
        from gomoveshift.pipeline import day_pipeline

        master = tmp_path / "master.mp4"
        master.write_bytes(b"fake-master")
        boundaries = [(0.0, 1200.0), (1200.0, 2100.0)]
        segments = [
            _seg(ep_idx=0, start=100.0, end=160.0),
            _seg(ep_idx=1, start=50.0, end=90.0),
        ]
        out = tmp_path / "lf_01_segments.mp4"
        work = tmp_path / "work"

        # Pre-seed a truncated (partial) segments file as if a prior crash
        # left it. Bytes are not a valid mp4 moov — callers must not rely
        # on its apparent "existence" to skip re-extraction.
        out.parent.mkdir(parents=True, exist_ok=True)
        partial_bytes = b"TRUNC" * 10  # clearly partial
        out.write_bytes(partial_bytes)
        assert out.exists()

        commands: list[list[str]] = []
        side = _make_run_side_effect(commands)

        with _patch_ffmpeg(side):
            result = day_pipeline.extract_and_concat_segments(
                master_path=master,
                segments=segments,
                file_boundaries=boundaries,
                output_path=out,
                work_dir=work,
            )

        # Extractor re-ran (did not skip based on out.exists())
        concat_cmds = [c for c in commands if _is_concat_cmd(c)]
        assert len(concat_cmds) == 1, (
            "Expected exactly one concat invocation; "
            f"got {len(concat_cmds)} — extractor silently skipped or re-ran twice"
        )
        # ffmpeg is invoked with `-y` so the truncated file is overwritten
        assert "-y" in concat_cmds[0]
        # Result points at the output and the content is no longer the
        # truncated stub (our fake ffmpeg writes a fresh body)
        assert result == out
        assert out.read_bytes() != partial_bytes
        assert out.read_bytes() == b"mp4-body-full"


class TestStateSaysDoneButSegmentsMissing:
    """Scenario 3: An upstream state file may claim the segment stage
    completed, but the on-disk `lf_NN_segments.mp4` is gone (manual
    cleanup, stale staging wipe, failed copy, etc.). A fresh call to
    `extract_and_concat_segments` must not silently succeed against the
    missing path — it must re-run ffmpeg and produce a real output."""

    def test_missing_output_is_reextracted_not_silently_skipped(self, tmp_path):
        from gomoveshift.pipeline import day_pipeline

        master = tmp_path / "master.mp4"
        master.write_bytes(b"fake-master")
        boundaries = [(0.0, 1200.0)]
        segments = [_seg(ep_idx=0, start=10.0, end=20.0)]
        out = tmp_path / "regular_lfs" / "lf_02_segments.mp4"
        work = tmp_path / "regular_lfs" / ".lf_02_work"

        # Simulate "state says done but file missing": neither the output
        # nor the work dir exists.
        assert not out.exists()
        assert not work.exists()

        commands: list[list[str]] = []
        side = _make_run_side_effect(commands)

        with _patch_ffmpeg(side):
            result = day_pipeline.extract_and_concat_segments(
                master_path=master,
                segments=segments,
                file_boundaries=boundaries,
                output_path=out,
                work_dir=work,
            )

        assert result == out, "Extractor should re-produce missing output, not return None"
        assert out.exists(), "Missing segments file must be re-materialised on disk"
        assert out.stat().st_size > 0
        # At minimum: one stream-copy + one concat (plus keyframe probe)
        assert any(_is_concat_cmd(c) for c in commands)
        copies = [c for c in commands if not _is_concat_cmd(c) and c[0] != "ffprobe"]
        assert len(copies) == len(segments)


class TestConcatInputsMissingFailsGracefully:
    """Scenario 4: The per-segment scratch files referenced by the concat
    list file disappear between stream-copy and concat (e.g. wiped by a
    stale-staging cleanup). ffmpeg concat must fail loudly and the
    extractor must return None — never a phantom output path."""

    def test_missing_concat_inputs_yields_none_and_no_output(self, tmp_path):
        from gomoveshift.pipeline import day_pipeline

        master = tmp_path / "master.mp4"
        master.write_bytes(b"fake-master")
        boundaries = [(0.0, 1200.0), (1200.0, 2100.0)]
        segments = [
            _seg(ep_idx=0, start=10.0, end=20.0),
            _seg(ep_idx=1, start=5.0, end=15.0),
        ]
        out = tmp_path / "lf_03_segments.mp4"
        work = tmp_path / "work"

        commands: list[list[str]] = []

        def _run(cmd, *a, **kw):
            cmd_list = list(cmd)
            commands.append(cmd_list)
            if _is_concat_cmd(cmd_list):
                # Real ffmpeg would error; our stub also respects missing
                # inputs because `_make_run_side_effect` simulates that.
                list_path = Path(cmd_list[cmd_list.index("-i") + 1])
                list_text = list_path.read_text()
                for line in list_text.splitlines():
                    line = line.strip()
                    if line.startswith("file "):
                        path_part = line[len("file ") :].strip().strip("'")
                        if not Path(path_part).exists():
                            return MagicMock(
                                returncode=1,
                                stdout=b"",
                                stderr=b"Impossible to open concat input",
                            )
                # inputs all present — succeed
                out_path = _output_path_of(cmd_list)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(b"mp4-body-full")
                return MagicMock(returncode=0, stdout=b"", stderr=b"")

            # stream-copy extract: record the target but DO NOT write it,
            # simulating scratch files that vanished before concat.
            return MagicMock(returncode=0, stdout=b"", stderr=b"")

        with _patch_ffmpeg(_run):
            result = day_pipeline.extract_and_concat_segments(
                master_path=master,
                segments=segments,
                file_boundaries=boundaries,
                output_path=out,
                work_dir=work,
            )

        assert result is None, "Extractor must return None when concat fails"
        assert not out.exists(), "No phantom output file should be left on disk"
        # Confirm concat was actually attempted (not short-circuited)
        assert any(_is_concat_cmd(c) for c in commands)

    def test_missing_master_returns_none_before_any_ffmpeg(self, tmp_path):
        """A related graceful-error case: the day master itself is gone
        (stale staging cleanup between runs). Extractor must refuse to
        invoke ffmpeg and return None — not produce a zero-byte output."""
        from gomoveshift.pipeline import day_pipeline

        master = tmp_path / "master.mp4"  # never created
        boundaries = [(0.0, 1200.0)]
        segments = [_seg(ep_idx=0, start=10.0, end=20.0)]
        out = tmp_path / "lf_04_segments.mp4"
        work = tmp_path / "work"

        commands: list[list[str]] = []

        def _run(cmd, *a, **kw):
            commands.append(list(cmd))
            return MagicMock(returncode=0, stdout=b"", stderr=b"")

        with _patch_ffmpeg(_run):
            result = day_pipeline.extract_and_concat_segments(
                master_path=master,
                segments=segments,
                file_boundaries=boundaries,
                output_path=out,
                work_dir=work,
            )

        assert result is None
        assert not out.exists()
        assert commands == [], "No ffmpeg calls should be made when master is missing"
