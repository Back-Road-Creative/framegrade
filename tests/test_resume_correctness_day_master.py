"""Resume-correctness regression tests for the Day Master Encode stage.

Target bug class: a half-written ``day_{date}_master.mp4`` whose ffprobe
returns *some* duration (because keyframes were written before a crash)
gets silently accepted as complete by resume logic. The encoder would
then skip re-encode and downstream stages would consume a corrupt master.

These are audit-only tests — they do not modify production code. They
exercise the existing resume logic in
``gomoveshift.pipeline.day_master_encoder.encode_day_master`` and its
color-cache companion ``_load_stable_color_cache``.

Run:
    .venv/bin/python -m pytest \
      tests/test_resume_correctness_day_master.py -o "addopts=" -v
"""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _blur_off(monkeypatch):
    """Resume-correctness MECHANICS, orthogonal to the privacy blur now default-ON in
    ``encode_day_master`` (its coverage lives in ``test_day_master_blur.py``). Force
    blur off so the mocked encode / resume isn't gated by the mask build / marker."""
    monkeypatch.setattr("gomoveshift.pipeline.blur_gate.blur_required", lambda cfg: False)


# ----------------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------------


def _make_episode(index: int, concat_path: Path, duration_s: float = 1800.0):
    ep = MagicMock()
    ep.index = index
    ep.creation_time = datetime(2026, 4, 18, 10, 0, tzinfo=timezone.utc)
    ep.total_duration = duration_s
    ep.total_duration_min = duration_s / 60.0
    ep.concat_path = concat_path
    ep.gpx_path = None
    return ep


def _make_day_batch(date: str, episode_paths: list, duration_s: float = 1800.0):
    from gomoveshift.pipeline.day_grouper import DayBatch

    episodes = [_make_episode(i + 1, p, duration_s=duration_s) for i, p in enumerate(episode_paths)]
    return DayBatch(date=date, episodes=episodes)


def _install_forbidden_reencode(monkeypatch):
    """Any call into SinglePassProcessor or subprocess.run is a failure —
    resume skip path must not touch either."""

    def forbidden_process(self, video_path, output_path, location, **kwargs):
        raise AssertionError("SinglePassProcessor.process() must NOT be called on resume skip")

    def forbidden_run(cmd, *a, **kw):
        raise AssertionError(f"subprocess.run must NOT be called on resume skip: {cmd!r}")

    monkeypatch.setattr(
        "gomoveshift.pipeline.single_pass.SinglePassProcessor.process",
        forbidden_process,
    )
    monkeypatch.setattr(
        "gomoveshift.pipeline.day_master_encoder.subprocess.run",
        forbidden_run,
    )


def _install_capturing_reencode(monkeypatch):
    """Replace SinglePassProcessor.process and subprocess.run with no-op
    stubs that record calls. Returns a dict with captured calls."""

    captured = {"sp_calls": [], "run_calls": []}

    def fake_process(self, video_path, output_path, location, **kwargs):
        captured["sp_calls"].append(
            {
                "input": video_path,
                "output": output_path,
                "skip_ost": kwargs.get("skip_ost"),
                "location": location,
            }
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).touch()
        return MagicMock(success=True, output_path=output_path)

    def fake_run(cmd, *a, **kw):
        captured["run_calls"].append(list(cmd))
        # Simulate ffmpeg concat producing the raw output file so the
        # encoder can proceed to the single-pass stage.
        for tok in cmd:
            if isinstance(tok, (str, Path)) and str(tok).endswith("_raw.mp4"):
                out = Path(tok)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.touch()
        return MagicMock(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(
        "gomoveshift.pipeline.single_pass.SinglePassProcessor.process",
        fake_process,
    )
    monkeypatch.setattr(
        "gomoveshift.pipeline.day_master_encoder.subprocess.run",
        fake_run,
    )
    # Disk space check must not interfere
    monkeypatch.setattr(
        "gomoveshift.pipeline.day_master_encoder.check_disk_space",
        lambda path, required: None,
    )

    return captured


# ----------------------------------------------------------------------
# Scenario 1 — Clean state: valid master, resume skips re-encode
# ----------------------------------------------------------------------


class TestScenario1CleanState:
    """Master exists, ffprobe reports full duration → resume skips encode."""

    def test_valid_master_is_returned_without_reencoding(self, tmp_path, monkeypatch):
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_master_encoder import encode_day_master

        ep = tmp_path / "ep_001.mp4"
        ep.write_bytes(b"x" * 4096)
        batch = _make_day_batch("2026-04-18", [ep], duration_s=1800.0)

        output_dir = tmp_path / "days_out"
        output_dir.mkdir(parents=True, exist_ok=True)
        master = output_dir / "day_2026-04-18_master.mp4"
        master.write_bytes(b"M" * (16 * 1024))  # pretend complete

        # Report ffprobe full duration (matches expected 1800s)
        monkeypatch.setattr(
            "gomoveshift.pipeline.validation.ffprobe_client.FFprobeClient.probe_duration",
            lambda self, path: 1800.0,
        )
        _install_forbidden_reencode(monkeypatch)

        result = encode_day_master(
            day_batch=batch,
            output_dir=output_dir,
            config=Config(),
        )
        assert result == master
        assert master.exists()


# ----------------------------------------------------------------------
# Scenario 2 — Truncated mp4 (no moov atom): ffprobe returns 0 or partial
# ----------------------------------------------------------------------


class TestScenario2TruncatedMaster:
    """Crash-truncated mp4 must NOT silently resume-skip.

    Two sub-cases:
      2a. ffprobe returns 0.0 (missing moov atom → probe fails entirely).
      2b. ffprobe returns a small non-zero duration (keyframes landed
          before the crash, but the file is still far short of the batch
          total). This is the most dangerous variant of the bug — it's
          the one that can silently proceed if the check is weak.
    """

    def test_truncated_master_probe_zero_triggers_reencode(self, tmp_path, monkeypatch):
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_master_encoder import encode_day_master

        ep = tmp_path / "ep_001.mp4"
        ep.write_bytes(b"x" * 4096)
        batch = _make_day_batch("2026-04-18", [ep], duration_s=1800.0)

        output_dir = tmp_path / "days_out"
        output_dir.mkdir(parents=True, exist_ok=True)
        master = output_dir / "day_2026-04-18_master.mp4"
        # Write a realistic-looking truncated prefix: valid 'ftyp' box
        # header but no moov. FFprobeClient.probe_duration() catches
        # any exception and returns 0.0, so we stub that directly to
        # mirror what the real ffprobe would return on a moov-less mp4.
        master.write_bytes(
            b"\x00\x00\x00\x20ftypisom" + b"\x00" * 4096  # partial bytes
        )

        # Integrity guard probes source episodes — return full duration for
        # those so the integrity check passes; only the master probes to 0.
        def fake_probe(self, path):
            if Path(path).name == master.name:
                return 0.0
            return 1800.0

        monkeypatch.setattr(
            "gomoveshift.pipeline.validation.ffprobe_client.FFprobeClient.probe_duration",
            fake_probe,
        )
        captured = _install_capturing_reencode(monkeypatch)

        result = encode_day_master(
            day_batch=batch,
            output_dir=output_dir,
            config=Config(),
        )

        # MUST NOT silently proceed: re-encode must fire.
        assert len(captured["sp_calls"]) == 1, (
            "Truncated master with probe_duration=0 must re-encode, not skip"
        )
        assert result is not None and result == master

    def test_truncated_master_partial_duration_triggers_reencode(self, tmp_path, monkeypatch):
        """ffprobe reports ~18% duration (mid-encode
        crash with some keyframes on disk). encode_day_master must detect
        the shortfall and re-encode, not accept the partial as complete."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_master_encoder import encode_day_master

        ep = tmp_path / "ep_001.mp4"
        ep.write_bytes(b"x" * 4096)
        batch = _make_day_batch("2026-04-18", [ep], duration_s=1800.0)

        output_dir = tmp_path / "days_out"
        output_dir.mkdir(parents=True, exist_ok=True)
        master = output_dir / "day_2026-04-18_master.mp4"
        master.write_bytes(b"partial-bytes-from-mid-encode-crash" * 200)

        captured = _install_capturing_reencode(monkeypatch)

        # 18% of expected for the master only; sources probe to full duration
        # so the integrity guard passes and the 99%-master gate sees the
        # mid-encode-crash shortfall. Once the re-encode has run, the promoted
        # master is full length — that is what the promotion-time duration
        # gate in ``encode_day_master`` probes.
        def fake_probe(self, path):
            if Path(path).name == master.name:
                return 1800.0 if captured["sp_calls"] else 1800.0 * 0.18
            return 1800.0

        monkeypatch.setattr(
            "gomoveshift.pipeline.validation.ffprobe_client.FFprobeClient.probe_duration",
            fake_probe,
        )

        encode_day_master(
            day_batch=batch,
            output_dir=output_dir,
            config=Config(),
        )

        assert len(captured["sp_calls"]) == 1, (
            "Silent proceed: partial-duration master must re-encode, not be accepted as complete"
        )


# ----------------------------------------------------------------------
# Scenario 3 — State says done, mp4 missing
# ----------------------------------------------------------------------


class TestScenario3StateSaysDoneFileMissing:
    """An external pipeline_state.json may claim phase_bc_day_first is
    complete for a given day. If the encode function is invoked anyway
    (the day-first pipeline iterates every day batch unconditionally),
    the missing master file must trigger a fresh re-encode rather than
    being silently treated as already-done.
    """

    def test_state_says_done_but_file_missing_reencodes(self, tmp_path, monkeypatch):
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_master_encoder import encode_day_master

        ep = tmp_path / "ep_001.mp4"
        ep.write_bytes(b"x" * 4096)
        batch = _make_day_batch("2026-04-18", [ep], duration_s=1800.0)

        output_dir = tmp_path / "days_out"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Simulate the outer pipeline's state file claiming the day is done.
        # encode_day_master itself has no per-day bookkeeping in
        # pipeline_state; this fixture just captures the scenario for the
        # audit record — the important assertion is on encoder behavior.
        state = {
            "phases": {
                "phase_bc_day_first": {
                    "status": "completed",
                    "days": ["2026-04-18"],
                }
            }
        }
        import json as _json

        (output_dir / "pipeline_state.json").write_text(_json.dumps(state))

        # Master file is NOT on disk.
        master_path = output_dir / "day_2026-04-18_master.mp4"
        assert not master_path.exists()

        # Guard: probe_duration must never be called for a missing file
        # (encode_day_master gates on master.exists() first).
        def unexpected_probe(self, path):
            raise AssertionError(f"probe_duration called on missing master: {path}")

        monkeypatch.setattr(
            "gomoveshift.pipeline.validation.ffprobe_client.FFprobeClient.probe_duration",
            unexpected_probe,
        )
        captured = _install_capturing_reencode(monkeypatch)

        result = encode_day_master(
            day_batch=batch,
            output_dir=output_dir,
            config=Config(),
        )

        # Missing master MUST trigger full re-encode, regardless of any
        # external state claim.
        assert len(captured["sp_calls"]) == 1, (
            "Missing master with external state=done must re-encode"
        )
        assert result == master_path
        assert master_path.exists(), "Re-encode must produce the master file"


# ----------------------------------------------------------------------
# Scenario 4 — _color_cache.json stale / missing
# ----------------------------------------------------------------------


class TestScenario4ColorCacheIntegrity:
    """The stable color cache (``day_{date}_color_cache.json``) stores
    source fingerprints. If fingerprints mismatch (source files changed
    since last encode) or the cache is missing, resume must NOT silently
    reuse stale/empty color data — it must force a re-analysis.
    """

    def test_missing_color_cache_returns_none(self, tmp_path):
        from gomoveshift.pipeline.day_master_encoder import (
            _load_stable_color_cache,
        )

        output_dir = tmp_path / "days_out"
        output_dir.mkdir(parents=True, exist_ok=True)
        ep = tmp_path / "ep_001.mp4"
        ep.write_bytes(b"x" * 4096)

        result = _load_stable_color_cache(output_dir, "2026-04-18", [ep])
        assert result is None, (
            "Missing color cache must return None so re-analysis runs — "
            "silently using an empty cache would skip color correction"
        )

    def test_stale_color_cache_fingerprints_invalidates(self, tmp_path):
        import json as _json

        from gomoveshift.pipeline.day_master_encoder import (
            _load_stable_color_cache,
            _stable_cache_path,
        )

        output_dir = tmp_path / "days_out"
        output_dir.mkdir(parents=True, exist_ok=True)
        ep = tmp_path / "ep_001.mp4"
        ep.write_bytes(b"x" * 4096)

        # Write a cache whose stored fingerprints DO NOT match the
        # current source file (simulates source replaced between runs).
        cache_path = _stable_cache_path(output_dir, "2026-04-18")
        cache_path.write_text(
            _json.dumps(
                {
                    "filter_str": "eq=contrast=1.05",
                    "method": "reference-day",
                    "source_fingerprints": ["999999:111111111"],
                }
            )
        )

        result = _load_stable_color_cache(output_dir, "2026-04-18", [ep])
        assert result is None, (
            "Fingerprint-mismatched color cache must invalidate — "
            "silently reusing stale color data would corrupt output"
        )

    def test_corrupt_color_cache_missing_keys_invalidates(self, tmp_path):
        import json as _json

        from gomoveshift.pipeline.day_master_encoder import (
            _load_stable_color_cache,
            _stable_cache_path,
        )

        output_dir = tmp_path / "days_out"
        output_dir.mkdir(parents=True, exist_ok=True)
        ep = tmp_path / "ep_001.mp4"
        ep.write_bytes(b"x" * 4096)

        # Compute the current fingerprint so fingerprint check passes,
        # then omit the required keys.
        from gomoveshift.pipeline.cache import file_fingerprint

        cache_path = _stable_cache_path(output_dir, "2026-04-18")
        cache_path.write_text(
            _json.dumps(
                {
                    # NOTE: no "filter_str", no "method"
                    "source_fingerprints": [file_fingerprint(ep)],
                }
            )
        )

        result = _load_stable_color_cache(output_dir, "2026-04-18", [ep])
        assert result is None, (
            "Cache without filter_str/method must invalidate, not silently "
            "return a stub — empty filter would disable color correction"
        )

    def test_valid_color_cache_hit_returns_data(self, tmp_path):
        import json as _json

        from gomoveshift.pipeline.cache import file_fingerprint
        from gomoveshift.pipeline.color_correction import GRADE_ALGO_VERSION
        from gomoveshift.pipeline.day_master_encoder import (
            COLOR_CACHE_SCHEMA_VERSION,
            _load_stable_color_cache,
            _stable_cache_path,
        )

        output_dir = tmp_path / "days_out"
        output_dir.mkdir(parents=True, exist_ok=True)
        ep = tmp_path / "ep_001.mp4"
        ep.write_bytes(b"x" * 4096)

        # Current schema: version + grade_algo_version + corrections + method.
        # v1 (filter_str only) and a stale grade_algo_version are both misses so
        # the next run re-analyzes.
        cache_path = _stable_cache_path(output_dir, "2026-04-18")
        cache_path.write_text(
            _json.dumps(
                {
                    "version": COLOR_CACHE_SCHEMA_VERSION,
                    "grade_algo_version": GRADE_ALGO_VERSION,
                    "method": "reference-day",
                    "corrections": [],
                    "segment_duration_sec": 0,
                    "source_fingerprints": [file_fingerprint(ep)],
                }
            )
        )

        result = _load_stable_color_cache(output_dir, "2026-04-18", [ep])
        assert result is not None
        assert result["method"] == "reference-day"
        assert result["corrections"] == []


# ----------------------------------------------------------------------
# Scenario 5 — Raw concat exists, master missing: skip re-concat
# ----------------------------------------------------------------------


class TestScenario5RawConcatResume:
    """When a WSL/encoder crash lands between ``_concat_raw_episodes``
    and the single-pass encode, the large ``day_{date}_raw.mp4`` is on
    disk but the master is not. On resume, the concat step must NOT be
    re-run — on 4K driving content the raw is 100+ GB and the concat
    takes 20–30 min of I/O for bytes the encoder can use as-is.
    """

    def test_valid_raw_concat_skips_reconcat(self, tmp_path, monkeypatch):
        """Raw exists at full expected duration → concat is skipped,
        single-pass encode runs against the existing raw."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_master_encoder import encode_day_master

        ep = tmp_path / "ep_001.mp4"
        ep.write_bytes(b"x" * 4096)
        batch = _make_day_batch("2026-04-18", [ep], duration_s=1800.0)

        output_dir = tmp_path / "days_out"
        output_dir.mkdir(parents=True, exist_ok=True)
        raw = output_dir / "day_2026-04-18_raw.mp4"
        raw.write_bytes(b"R" * (16 * 1024))  # pretend full raw on disk

        # probe_duration returns full duration for the raw, nothing else exists
        def fake_probe(self, path):
            if Path(path).name.endswith("_raw.mp4"):
                return 1800.0
            return 0.0

        monkeypatch.setattr(
            "gomoveshift.pipeline.validation.ffprobe_client.FFprobeClient.probe_duration",
            fake_probe,
        )

        captured = _install_capturing_reencode(monkeypatch)

        result = encode_day_master(
            day_batch=batch,
            output_dir=output_dir,
            config=Config(),
        )

        # The bug: previously encode_day_master unconditionally ran the
        # ffmpeg concat with -y, overwriting the intact raw. Assert no
        # concat subprocess.run call was made.
        concat_calls = [
            c
            for c in captured["run_calls"]
            if any(isinstance(t, (str, Path)) and "concat" in str(t) for t in c)
        ]
        assert not concat_calls, f"Intact raw must skip re-concat on resume, got: {concat_calls}"

        # But the single-pass encode must still run to produce the master.
        assert len(captured["sp_calls"]) == 1
        assert captured["sp_calls"][0]["input"] == raw
        assert result is not None

    def test_truncated_raw_concat_triggers_reconcat(self, tmp_path, monkeypatch):
        """A raw concat whose duration is short of expected (< 99%) must
        be re-concatenated, not silently used."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_master_encoder import encode_day_master

        ep = tmp_path / "ep_001.mp4"
        ep.write_bytes(b"x" * 4096)
        batch = _make_day_batch("2026-04-18", [ep], duration_s=1800.0)

        output_dir = tmp_path / "days_out"
        output_dir.mkdir(parents=True, exist_ok=True)
        raw = output_dir / "day_2026-04-18_raw.mp4"
        raw.write_bytes(b"partial" * 128)

        # Raw is only ~50% of expected duration
        def fake_probe(self, path):
            if Path(path).name.endswith("_raw.mp4"):
                return 900.0
            return 0.0

        monkeypatch.setattr(
            "gomoveshift.pipeline.validation.ffprobe_client.FFprobeClient.probe_duration",
            fake_probe,
        )

        captured = _install_capturing_reencode(monkeypatch)

        encode_day_master(
            day_batch=batch,
            output_dir=output_dir,
            config=Config(),
        )

        concat_calls = [
            c
            for c in captured["run_calls"]
            if any(isinstance(t, (str, Path)) and "concat" in str(t) for t in c)
        ]
        assert concat_calls, "Truncated raw must trigger re-concat, not be accepted as complete"

    def test_missing_raw_runs_concat_as_normal(self, tmp_path, monkeypatch):
        """Normal first-run path: raw doesn't exist → concat runs."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_master_encoder import encode_day_master

        ep = tmp_path / "ep_001.mp4"
        ep.write_bytes(b"x" * 4096)
        batch = _make_day_batch("2026-04-18", [ep], duration_s=1800.0)

        output_dir = tmp_path / "days_out"
        output_dir.mkdir(parents=True, exist_ok=True)
        # No raw, no master.

        # probe_duration shouldn't be called for raw since it doesn't exist,
        # but tolerate the call if it happens.
        monkeypatch.setattr(
            "gomoveshift.pipeline.validation.ffprobe_client.FFprobeClient.probe_duration",
            lambda self, path: 0.0,
        )

        captured = _install_capturing_reencode(monkeypatch)

        encode_day_master(
            day_batch=batch,
            output_dir=output_dir,
            config=Config(),
        )

        # Both concat and single-pass should have fired.
        concat_calls = [
            c
            for c in captured["run_calls"]
            if any(isinstance(t, (str, Path)) and "concat" in str(t) for t in c)
        ]
        assert concat_calls, "Missing raw: concat must run as normal"
        assert len(captured["sp_calls"]) == 1


# ----------------------------------------------------------------------
# Scenario 6 — Independent ground-truth check on batch.total_duration
# ----------------------------------------------------------------------


class TestScenario6_BatchDurationIntegrity:  # noqa: N801 — sibling-class style
    """The 99% master-duration gate is only as trustworthy as
    ``batch.total_duration``. If ingest under-reports per-episode duration
    (e.g. truncated source files), the gate's expected figure is itself
    too small and a partial master can pass it.

    The fix: BEFORE the 99% compare, ffprobe each episode source
    independently, sum the durations, and assert the sum agrees with
    ``batch.total_duration`` within ``DURATION_INTEGRITY_TOLERANCE_S``.
    Any disagreement raises ``BatchDurationIntegrityError``.

    Pinned regression: ``test_resume_with_truncated_master_passes_under_reported_duration``
    demonstrates the exact bug class — a master at 89% of true duration
    paired with a batch.total_duration that's been clipped to 90% of the
    true value. The 99% gate sees actual ≥ expected*0.99 and silently
    accepts. The integrity guard catches the bad expected value first.
    """

    def test_resume_with_truncated_master_passes_under_reported_duration(
        self, tmp_path, monkeypatch
    ):
        """The bug: master is corrupt-truncated to 89% of real footage,
        batch.total_duration is itself under-reported to 90% of real.
        Master/expected = 89/90 = 0.988 > 0.99? Actually 0.989 < 0.99…
        push it slightly higher: master at 90% of real, batch at 91% of real
        → master/batch = 0.989 < 0.99. Use master 91% / batch 91.5% →
        0.9945 ≥ 0.99 → SILENTLY PASSES the 99% gate without the integrity
        guard. With the guard, source ffprobe reports the real (100%) sum,
        which disagrees with batch.total_duration → raise."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_master_encoder import (
            BatchDurationIntegrityError,
            encode_day_master,
        )

        ep1 = tmp_path / "ep_001.mp4"
        ep2 = tmp_path / "ep_002.mp4"
        ep1.write_bytes(b"x" * 4096)
        ep2.write_bytes(b"x" * 4096)
        # batch.total_duration = 2 * 1640 = 3280s — under-reported (real total
        # is 3600s, sources ffprobe to 1800 each)
        batch = _make_day_batch("2026-04-18", [ep1, ep2], duration_s=1640.0)
        assert batch.total_duration == 3280.0

        output_dir = tmp_path / "days_out"
        output_dir.mkdir(parents=True, exist_ok=True)
        master = output_dir / "day_2026-04-18_master.mp4"
        master.write_bytes(b"M" * (16 * 1024))

        # Master probes at 99.5% of (under-reported) batch — would pass the
        # 99% gate. But sources probe at the REAL 1800s each (3600s total),
        # which the integrity guard catches.
        def fake_probe(self, path):
            if Path(path).name == master.name:
                return 3280.0 * 0.995  # passes 99% of expected
            return 1800.0  # real per-episode duration

        monkeypatch.setattr(
            "gomoveshift.pipeline.validation.ffprobe_client.FFprobeClient.probe_duration",
            fake_probe,
        )
        _install_forbidden_reencode(monkeypatch)

        import pytest

        with pytest.raises(BatchDurationIntegrityError, match="under-reported"):
            encode_day_master(
                day_batch=batch,
                output_dir=output_dir,
                config=Config(),
            )

    def test_integrity_passes_when_sources_agree_with_batch(self, tmp_path, monkeypatch):
        """Honest case: per-episode probe sum matches batch.total_duration
        within tolerance → integrity passes, the 99% master gate runs as
        normal, valid master is returned without re-encoding."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_master_encoder import encode_day_master

        ep = tmp_path / "ep_001.mp4"
        ep.write_bytes(b"x" * 4096)
        batch = _make_day_batch("2026-04-18", [ep], duration_s=1800.0)

        output_dir = tmp_path / "days_out"
        output_dir.mkdir(parents=True, exist_ok=True)
        master = output_dir / "day_2026-04-18_master.mp4"
        master.write_bytes(b"M" * (16 * 1024))

        # Source agrees with batch (1800s); master is full-length.
        monkeypatch.setattr(
            "gomoveshift.pipeline.validation.ffprobe_client.FFprobeClient.probe_duration",
            lambda self, path: 1800.0,
        )
        _install_forbidden_reencode(monkeypatch)

        result = encode_day_master(
            day_batch=batch,
            output_dir=output_dir,
            config=Config(),
        )
        assert result == master

    def test_integrity_raises_when_episode_probe_fails(self, tmp_path, monkeypatch):
        """A source that ffprobes to 0.0 (corrupt/missing) is itself a
        ground-truth failure — the integrity guard cannot verify the
        batch's recorded total against an episode it cannot read.
        Raise rather than re-encode (re-encoding from a corrupt source
        will just produce another corrupt master)."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_master_encoder import (
            BatchDurationIntegrityError,
            encode_day_master,
        )

        ep1 = tmp_path / "ep_001.mp4"
        ep2 = tmp_path / "ep_002.mp4"
        ep1.write_bytes(b"x" * 4096)
        ep2.write_bytes(b"x" * 4096)
        batch = _make_day_batch("2026-04-18", [ep1, ep2], duration_s=1800.0)

        output_dir = tmp_path / "days_out"
        output_dir.mkdir(parents=True, exist_ok=True)
        master = output_dir / "day_2026-04-18_master.mp4"
        master.write_bytes(b"M" * (16 * 1024))

        # First episode probes fine; second returns 0 (corrupt source).
        def fake_probe(self, path):
            if Path(path).name == master.name:
                return 3600.0
            if Path(path).name == ep1.name:
                return 1800.0
            return 0.0  # ep2 unreadable

        monkeypatch.setattr(
            "gomoveshift.pipeline.validation.ffprobe_client.FFprobeClient.probe_duration",
            fake_probe,
        )
        _install_forbidden_reencode(monkeypatch)

        import pytest

        with pytest.raises(BatchDurationIntegrityError, match="failed ffprobe"):
            encode_day_master(
                day_batch=batch,
                output_dir=output_dir,
                config=Config(),
            )

    def test_integrity_raises_when_episode_concat_path_missing(self, tmp_path, monkeypatch):
        """An episode with concat_path=None is partial Phase A state — the
        master cannot truthfully include it, so the integrity guard refuses
        to verify."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.day_master_encoder import (
            BatchDurationIntegrityError,
            encode_day_master,
        )

        ep1 = tmp_path / "ep_001.mp4"
        ep1.write_bytes(b"x" * 4096)
        batch = _make_day_batch("2026-04-18", [ep1], duration_s=1800.0)
        # Add a second episode with concat_path=None
        from unittest.mock import MagicMock as _Mock

        ghost = _Mock()
        ghost.index = 2
        ghost.total_duration = 1800.0
        ghost.concat_path = None
        ghost.gpx_path = None
        batch.episodes.append(ghost)

        output_dir = tmp_path / "days_out"
        output_dir.mkdir(parents=True, exist_ok=True)
        master = output_dir / "day_2026-04-18_master.mp4"
        master.write_bytes(b"M" * (16 * 1024))

        monkeypatch.setattr(
            "gomoveshift.pipeline.validation.ffprobe_client.FFprobeClient.probe_duration",
            lambda self, path: 1800.0,
        )
        _install_forbidden_reencode(monkeypatch)

        import pytest

        with pytest.raises(BatchDurationIntegrityError, match="missing concat_path"):
            encode_day_master(
                day_batch=batch,
                output_dir=output_dir,
                config=Config(),
            )


# ----------------------------------------------------------------------
# Unit tests for assert_batch_duration_integrity (isolated function)
# ----------------------------------------------------------------------


class TestAssertBatchDurationIntegrity:
    """Direct unit tests of the integrity helper (no encode_day_master)."""

    def test_agreement_within_tolerance_returns_sum(self, tmp_path):
        from gomoveshift.pipeline.day_master_encoder import (
            assert_batch_duration_integrity,
        )

        ep = tmp_path / "ep_001.mp4"
        ep.write_bytes(b"x" * 1024)
        batch = _make_day_batch("2026-04-18", [ep], duration_s=1800.0)

        # Probe returns 1800.3 — within tolerance.
        probed = assert_batch_duration_integrity(batch, [ep], probe_fn=lambda p: 1800.3)
        assert probed == 1800.3

    def test_gopro_container_drift_within_2s_passes(self, tmp_path):
        """GoPro HEVC containers can report 1–2s different durations across
        probe invocations (NTSC fractional-frame rounding accumulating over
        a multi-recording day). A 1.208s drift has been seen on a long day
        (19005.693s probed vs 19004.485s stored in Phase A). Must not block
        resume — real truncation is seconds-to-minutes, not sub-2s.
        """
        from gomoveshift.pipeline.day_master_encoder import (
            assert_batch_duration_integrity,
        )

        ep = tmp_path / "ep_001.mp4"
        ep.write_bytes(b"x" * 1024)
        # Simulate a 1.208s drift on a 5.28-hour day.
        batch = _make_day_batch("2026-03-05", [ep], duration_s=19004.485)
        probed = assert_batch_duration_integrity(batch, [ep], probe_fn=lambda p: 19005.693)
        assert probed == 19005.693

    def test_at_tolerance_boundary_passes(self, tmp_path):
        """delta == tolerance must NOT raise (use strict > comparison)."""
        from gomoveshift.pipeline.day_master_encoder import (
            DURATION_INTEGRITY_TOLERANCE_S,
            assert_batch_duration_integrity,
        )

        ep = tmp_path / "ep_001.mp4"
        ep.write_bytes(b"x" * 1024)
        batch = _make_day_batch("2026-04-18", [ep], duration_s=1800.0)

        probed = assert_batch_duration_integrity(
            batch,
            [ep],
            probe_fn=lambda p: 1800.0 + DURATION_INTEGRITY_TOLERANCE_S,
        )
        assert probed == 1800.0 + DURATION_INTEGRITY_TOLERANCE_S

    def test_disagreement_beyond_tolerance_raises(self, tmp_path):
        from gomoveshift.pipeline.day_master_encoder import (
            BatchDurationIntegrityError,
            assert_batch_duration_integrity,
        )

        ep = tmp_path / "ep_001.mp4"
        ep.write_bytes(b"x" * 1024)
        batch = _make_day_batch("2026-04-18", [ep], duration_s=1800.0)

        import pytest

        with pytest.raises(BatchDurationIntegrityError, match="under-reported"):
            assert_batch_duration_integrity(batch, [ep], probe_fn=lambda p: 1900.0)

    def test_zero_probe_raises(self, tmp_path):
        from gomoveshift.pipeline.day_master_encoder import (
            BatchDurationIntegrityError,
            assert_batch_duration_integrity,
        )

        ep = tmp_path / "ep_001.mp4"
        ep.write_bytes(b"x" * 1024)
        batch = _make_day_batch("2026-04-18", [ep], duration_s=1800.0)

        import pytest

        with pytest.raises(BatchDurationIntegrityError, match="failed ffprobe"):
            assert_batch_duration_integrity(batch, [ep], probe_fn=lambda p: 0.0)

    def test_no_episodes_raises(self):
        from gomoveshift.pipeline.day_grouper import DayBatch
        from gomoveshift.pipeline.day_master_encoder import (
            BatchDurationIntegrityError,
            assert_batch_duration_integrity,
        )

        batch = DayBatch(date="2026-04-18", episodes=[])
        import pytest

        with pytest.raises(BatchDurationIntegrityError, match="no episodes"):
            assert_batch_duration_integrity(batch, [], probe_fn=lambda p: 0.0)
