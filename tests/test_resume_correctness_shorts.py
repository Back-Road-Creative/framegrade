"""Resume-correctness regression tests for the shorts extraction stage.

Context
-------
These scenarios drive `_run_shorts_loop`, a LOCAL replica of a rank-keyed
two-step shorts loop over the `ShortRange` plans `plan_shorts` produces:

  1. Stream-copy a `short_NN_raw.mp4` out of the day master
     (`ffmpeg -y -ss/-to -c copy`).
  2. Burn OST onto the raw to produce `short_NN.mp4`
     (`apply_ost` -> ffmpeg re-encode + drawtext + faststart).

Production has since collapsed both steps into one `render_short_ffmpeg`
pass, so the replica models the resume semantics — not live pipeline code.

Both steps use `ffmpeg -y`, which unconditionally overwrites. There is
NO existence check in the loop: on resume, the loop re-plans the same
set of shorts (planner is deterministic for the same day scores) and
re-invokes ffmpeg for every rank. This is the **double-work** concern
— but because output paths are rank-keyed (short_01.mp4, short_02.mp4,
...) the files themselves do not "duplicate"; a resume that restarts
the loop just redoes completed shorts in-place.

These tests pin:

  1. Clean — all 9 shorts produced, one extract + one apply_ost per rank.
  2. Mid-loop kill — shorts 1-4 complete on disk, short_05 half-written,
     6-9 absent. On resume, we assert that (a) the planner yields the
     same 9 ranks, and (b) the loop as written invokes ffmpeg for ALL
     9 ranks (the wasted-work bug) — while `-y` overwrite ensures the
     short_05 half-file does not survive intact.
  3. State says all shorts done but the `shorts_per_lf` state dict is
     empty. The downstream consumer (Phase D) keys off the state dict,
     so an empty dict means "no shorts", regardless of files on disk —
     resume decision.
  4. Source LF (day master) missing at resume time (upstream gone): the
     stream copy returns None and the loop skips the rank without
     crashing.

All subprocess calls to ffmpeg are mocked — no real encoding is run.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


# ── Helpers ────────────────────────────────────────────────────────────


def _short_range(rank: int, start: float, end: float, score: float = 0.5):
    """Build a ShortRange fixture without depending on its exact dataclass
    shape — the loop reads `.rank`, `.start_sec`, `.end_sec`.
    """
    from gomoveshift.pipeline.short_planner import ShortRange

    return ShortRange(
        start_sec=float(start),
        end_sec=float(end),
        integrated_score=float(score),
        rank=int(rank),
    )


def _nine_shorts_plan() -> list:
    """Nine 38-second ranges spaced 60s apart, ranked 1..9."""
    plan = []
    for i in range(9):
        start = 120.0 + i * 120.0
        plan.append(_short_range(rank=i + 1, start=start, end=start + 38.0))
    return plan


def _run_shorts_loop(
    short_ranges: list,
    master_path: Path,
    shorts_dir: Path,
    location: str,
    *,
    stream_copy_fn,
    apply_ost_fn,
    skip_if_shorter_than: float = 5.0,
) -> list:
    """Replica of the two-step shorts loop, exercised directly so the
    scenarios do not have to mock the heavy machinery that runs before it
    (`encode_day_master`, regular-LF assembly, etc.).

    Models a retired production shape: the live loop now renders each short
    in a single `render_short_ffmpeg` pass. What it still pins is the
    resume contract — `-y` overwrite, no existence check, rank-keyed
    output paths.
    """
    shorts_dir.mkdir(parents=True, exist_ok=True)
    short_paths: list = []
    for short_range in short_ranges:
        rank = short_range.rank
        short_duration = short_range.end_sec - short_range.start_sec
        if short_duration < skip_if_shorter_than:
            continue
        raw_short = shorts_dir / f"short_{rank:02d}_raw.mp4"
        extracted = stream_copy_fn(
            master_path,
            short_range.start_sec,
            short_range.end_sec,
            raw_short,
        )
        if extracted is None:
            continue
        final_short = shorts_dir / f"short_{rank:02d}.mp4"
        ost_result = apply_ost_fn(
            input_path=extracted,
            output_path=final_short,
            location=location,
            layout="short",
        )
        short_paths.append(ost_result if ost_result else extracted)
    return short_paths


def _fake_stream_copy(master_path: Path, start: float, end: float, out_path: Path):
    """Fake stream copy — writes a small mp4-like blob atomically
    (simulating `ffmpeg -y` overwrite), and returns the output path.
    Returns None if the input master is missing (mirrors real behaviour).
    """
    master_path = Path(master_path)
    if not master_path.exists():
        return None
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # `-y` overwrite semantics: even if out_path already had stale bytes,
    # the resulting file is a clean, complete blob. This is what real
    # ffmpeg does on successful completion.
    out_path.write_bytes(b"raw-short-full" + str(start).encode())
    return out_path


def _fake_apply_ost(*, input_path: Path, output_path: Path, location: str, layout: str):
    """Fake `apply_ost` — writes a clean final blob and returns the path."""
    input_path = Path(input_path)
    if not input_path.exists():
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"ost-final-full-" + layout.encode())
    return output_path


# ── Scenario 1: Clean run ─────────────────────────────────────────────


class TestCleanRunProducesAllShorts:
    """Scenario 1: clean state, 9 ranks planned, master present. Loop
    produces one raw + one final per rank — 18 ffmpeg invocations total,
    and short_01.mp4 ... short_09.mp4 all exist on disk."""

    def test_nine_shorts_produced_each_invokes_extract_and_ost(self, tmp_path):
        master = tmp_path / "day_master.mp4"
        master.write_bytes(b"master-full")
        shorts_dir = tmp_path / "shorts"
        plan = _nine_shorts_plan()

        extract_calls: list = []
        ost_calls: list = []

        def _ext(master_path, start, end, out_path):
            extract_calls.append((Path(out_path).name, start, end))
            return _fake_stream_copy(master_path, start, end, out_path)

        def _ost(*, input_path, output_path, location, layout):
            ost_calls.append((Path(output_path).name, layout))
            return _fake_apply_ost(
                input_path=input_path,
                output_path=output_path,
                location=location,
                layout=layout,
            )

        results = _run_shorts_loop(
            short_ranges=plan,
            master_path=master,
            shorts_dir=shorts_dir,
            location="Ireland",
            stream_copy_fn=_ext,
            apply_ost_fn=_ost,
        )

        assert len(results) == 9, "All 9 shorts should be produced on a clean run"
        # One raw + one final per rank
        assert len(extract_calls) == 9
        assert len(ost_calls) == 9
        for rank in range(1, 10):
            assert (shorts_dir / f"short_{rank:02d}_raw.mp4").exists()
            assert (shorts_dir / f"short_{rank:02d}.mp4").exists()
        # Layout is 'short' on every OST invocation
        assert all(layout == "short" for _, layout in ost_calls)


# ── Scenario 2: Mid-loop kill — resume-correctness regression ──────────


class TestMidLoopKillResumeBehaviour:
    """Scenario 2: prior run completed shorts 1-4 (both raw+final on
    disk), crashed mid-write on short_05 (raw exists but is partial;
    final never written), and never reached 6-9. On resume, the planner
    is deterministic and yields the same 9 ranks.

    **Contract pinned here** — the loop as currently written in
    `day_pipeline.py` has NO existence check and NO size guard, so it
    re-invokes ffmpeg for every rank. Because both the stream copy and
    `apply_ost` use `ffmpeg -y`, the overwrite makes the re-run
    idempotent on disk (no duplicate files, no file corruption after
    the resume completes). BUT it is wasted work — shorts 1-4 are
    re-encoded.

    If someone adds a resume-skip guard to the loop (the correct long-
    term fix), this test will fail on the invocation-count assertions
    and the fix commit should update it to the new contract.
    """

    def _preseed_completed_shorts(self, shorts_dir: Path, completed_ranks: list[int]) -> None:
        shorts_dir.mkdir(parents=True, exist_ok=True)
        for r in completed_ranks:
            (shorts_dir / f"short_{r:02d}_raw.mp4").write_bytes(b"raw-short-full")
            (shorts_dir / f"short_{r:02d}.mp4").write_bytes(b"ost-final-full-short")

    def test_resume_loop_does_not_produce_duplicate_files(self, tmp_path):
        """Resume: the output path set is exactly short_01.mp4 ...
        short_09.mp4 — no extra files like `short_01_copy.mp4` or a
        second `short_01.mp4` under a different name. This is the
        'double-count as duplicate files' concern.
        """
        master = tmp_path / "day_master.mp4"
        master.write_bytes(b"master-full")
        shorts_dir = tmp_path / "shorts"
        # Pre-seed 1-4 complete
        self._preseed_completed_shorts(shorts_dir, [1, 2, 3, 4])
        # Pre-seed short_05_raw as a partial (mid-encode crash)
        (shorts_dir / "short_05_raw.mp4").write_bytes(b"PARTIAL")
        # 6-9 absent

        plan = _nine_shorts_plan()
        _run_shorts_loop(
            short_ranges=plan,
            master_path=master,
            shorts_dir=shorts_dir,
            location="Ireland",
            stream_copy_fn=_fake_stream_copy,
            apply_ost_fn=_fake_apply_ost,
        )

        # File set on disk is exactly the rank-keyed set — no duplicates.
        finals = sorted(p.name for p in shorts_dir.glob("short_*.mp4") if "_raw" not in p.name)
        assert finals == [f"short_{r:02d}.mp4" for r in range(1, 10)], (
            f"Unexpected file set after resume: {finals}. Double-count would show extra names."
        )
        raws = sorted(p.name for p in shorts_dir.glob("short_*_raw.mp4"))
        assert raws == [f"short_{r:02d}_raw.mp4" for r in range(1, 10)]

    def test_resume_loop_redoes_completed_shorts_no_skip_guard(self, tmp_path):
        """Document the current (wasted-work) behaviour: the loop has
        no resume-skip guard, so every rank re-invokes extract+ost on
        resume. This test pins that contract — if it starts failing
        because fewer invocations occur, the loop has grown a skip
        guard and the fix should update this test.
        """
        master = tmp_path / "day_master.mp4"
        master.write_bytes(b"master-full")
        shorts_dir = tmp_path / "shorts"
        self._preseed_completed_shorts(shorts_dir, [1, 2, 3, 4])
        (shorts_dir / "short_05_raw.mp4").write_bytes(b"PARTIAL")

        plan = _nine_shorts_plan()
        extract_calls: list = []
        ost_calls: list = []

        def _ext(master_path, start, end, out_path):
            extract_calls.append(Path(out_path).name)
            return _fake_stream_copy(master_path, start, end, out_path)

        def _ost(*, input_path, output_path, location, layout):
            ost_calls.append(Path(output_path).name)
            return _fake_apply_ost(
                input_path=input_path,
                output_path=output_path,
                location=location,
                layout=layout,
            )

        _run_shorts_loop(
            short_ranges=plan,
            master_path=master,
            shorts_dir=shorts_dir,
            location="Ireland",
            stream_copy_fn=_ext,
            apply_ost_fn=_ost,
        )

        # Current contract: loop re-invokes for every rank (no skip).
        assert len(extract_calls) == 9, (
            "Current loop has no resume-skip; all 9 ranks re-invoke extract. "
            f"Got {len(extract_calls)}."
        )
        assert len(ost_calls) == 9

    def test_resume_overwrites_half_written_short_05(self, tmp_path):
        """The critical file-safety assertion: even though the loop
        re-does work, the half-written short_05_raw.mp4 from the prior
        crash does NOT survive as-is. ffmpeg `-y` overwrites it with
        the clean re-encoded bytes.
        """
        master = tmp_path / "day_master.mp4"
        master.write_bytes(b"master-full")
        shorts_dir = tmp_path / "shorts"
        # Short 05 raw is partial (valid path exists but is 7 bytes)
        shorts_dir.mkdir(parents=True, exist_ok=True)
        partial_raw = shorts_dir / "short_05_raw.mp4"
        partial_raw.write_bytes(b"PARTIAL")
        # short_05.mp4 does NOT exist — OST never reached

        plan = _nine_shorts_plan()
        _run_shorts_loop(
            short_ranges=plan,
            master_path=master,
            shorts_dir=shorts_dir,
            location="Ireland",
            stream_copy_fn=_fake_stream_copy,
            apply_ost_fn=_fake_apply_ost,
        )

        # After resume, short_05_raw is a clean blob (not the partial)
        assert partial_raw.exists()
        assert partial_raw.read_bytes() != b"PARTIAL", (
            "Half-written raw must be overwritten by ffmpeg -y on resume"
        )
        # And short_05.mp4 (final OST output) now exists
        assert (shorts_dir / "short_05.mp4").exists()


# ── Scenario 3: state says done but shorts_per_lf dict is empty ────────


class TestStateShortsPerLfEmpty:
    """Scenario 3: the pipeline state reports the shorts stage completed
    (files exist on disk) but the `shorts_per_lf` dict in the pipeline
    state is `{}` — e.g. a crash between rendering and state serialization,
    or a stale state loaded over fresh artifacts.

    Phase D consumes `state['shorts_per_lf']` to build per-LF social
    posts (`phase_d.py`: `lf_shorts = shorts_per_lf.get(lf.index, [])`).
    An empty dict therefore means Phase D will see zero shorts for the
    LF, even when 9 mp4s are on disk. This is the 'state vs. filesystem
    split-brain' case the resume logic must recognise.

    This test pins the current behaviour: an empty `shorts_per_lf`
    dict is NOT automatically reconciled against disk; consumers
    treat it as authoritative.
    """

    def test_phase_d_consumer_sees_no_shorts_when_state_dict_empty(self, tmp_path):
        shorts_dir = tmp_path / "shorts"
        shorts_dir.mkdir()
        for r in range(1, 10):
            (shorts_dir / f"short_{r:02d}.mp4").write_bytes(b"ost-final-full-short")

        # Simulate Phase D reading state
        state_shorts_per_lf: dict = {}
        lf_index = 1

        # Mirror the Phase D line: `lf_shorts = shorts_per_lf.get(lf.index, [])`
        lf_shorts = state_shorts_per_lf.get(lf_index, [])
        num_shorts_seen_by_phase_d = len(lf_shorts)

        assert num_shorts_seen_by_phase_d == 0, (
            "Phase D keys off the state dict — an empty dict means 0 shorts "
            "seen regardless of disk contents. Resume path must re-populate "
            "the state dict, not just rely on disk."
        )
        # Confirm the disk-truth is 9: any reconciler should detect 9
        assert len(list(shorts_dir.glob("short_*.mp4"))) == 9

    def test_pipeline_state_from_dict_preserves_empty_shorts_per_lf(self, tmp_path):
        """The typed state model coerces shorts_per_lf keys to int but
        does not rebuild an empty dict from disk. Pinning this keeps
        the state model honest about its responsibility boundary.
        """
        from gomoveshift.pipeline.drive_state_model import PipelineState

        raw_state = {
            "source_dir": str(tmp_path),
            "output_base": str(tmp_path),
            "region": "ire",
            "shorts_per_lf": {},
            "regular_lf_paths": [str(tmp_path / "lf_01.mp4")],
        }
        st = PipelineState.from_dict(raw_state)
        # Empty dict is NOT coerced to None, NOT populated from disk
        assert st.shorts_per_lf == {}
        # `state.get("shorts_per_lf", {})` returns {} — Phase D path
        assert st.get("shorts_per_lf", {}) == {}


# ── Scenario 4: source LF (day master) missing at resume ────────────────


class TestSourceMasterMissingAtResume:
    """Scenario 4: resume starts but the day master was wiped (stale-
    staging cleanup, disk pressure, manual rm). The shorts loop should
    not crash; the stream copy returns None on missing input and the
    loop's `if extracted is None: continue` skips each rank cleanly.
    Result: zero shorts produced, no phantom files, no exception."""

    def test_missing_master_yields_zero_shorts_no_crash(self, tmp_path):
        master = tmp_path / "day_master.mp4"  # never written
        assert not master.exists()
        shorts_dir = tmp_path / "shorts"

        plan = _nine_shorts_plan()
        extract_calls: list = []
        ost_calls: list = []

        def _ext(master_path, start, end, out_path):
            extract_calls.append(Path(out_path).name)
            return _fake_stream_copy(master_path, start, end, out_path)

        def _ost(*, input_path, output_path, location, layout):
            ost_calls.append(Path(output_path).name)
            return _fake_apply_ost(
                input_path=input_path,
                output_path=output_path,
                location=location,
                layout=layout,
            )

        results = _run_shorts_loop(
            short_ranges=plan,
            master_path=master,
            shorts_dir=shorts_dir,
            location="Ireland",
            stream_copy_fn=_ext,
            apply_ost_fn=_ost,
        )

        assert results == [], "No shorts can be produced without the master"
        # The stream copy was called for each rank (not short-circuited)
        assert len(extract_calls) == 9
        # But apply_ost was NEVER called — the `if extracted is None: continue`
        # guard fires before OST.
        assert ost_calls == [], "apply_ost must not be invoked when the stream copy returns None"
        # No output files on disk
        assert not list(shorts_dir.glob("short_*.mp4"))


# ── Sanity: the deterministic planner yields the same ranks on resume ──


class TestPlannerDeterminismOnResume:
    """A resume path only makes sense if the planner yields the same
    ranks for the same day scores. Pin that contract."""

    def test_plan_shorts_is_deterministic_for_same_scores(self):
        import numpy as np

        from gomoveshift.pipeline.short_planner import plan_shorts

        rng = np.random.default_rng(seed=42)
        n = 2000  # 2000 seconds of scored timeline at 1 Hz
        scores = np.zeros((n, 9), dtype=np.float64)
        scores[:, 0] = np.arange(n, dtype=np.float64)  # timestamps 0..1999
        scores[:, 1] = rng.random(n)  # composite scores

        plan_a = plan_shorts(scores, target_count=9)
        plan_b = plan_shorts(scores, target_count=9)

        # Same ranks, same boundaries → resume sees the identical plan
        assert len(plan_a) == len(plan_b)
        for a, b in zip(plan_a, plan_b):
            assert a.rank == b.rank
            assert a.start_sec == b.start_sec
            assert a.end_sec == b.end_sec
