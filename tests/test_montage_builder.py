"""Tests for the montage duration cap.

These tests pin the montage duration cap behavior:

* Total montage duration is capped at ``config.drive.montage_max_duration_sec``
  (default 15.0s, down from the previous hardcoded 30.0s).
* The cap is enforced by first trimming per-clip duration toward a 1.5s
  floor (the operator explicitly rejected 1.0s as too short), and then,
  only if the floor cannot satisfy the cap, reducing the peak count.
* The cap is configurable via ``config.drive.montage_max_duration_sec``
  so future tuning is a config change, not a code change.
* When the cap actually fires, an INFO log line is emitted so operators
  can grep for it in a long pipeline run.

The heavy ffmpeg extraction/concat path is NOT exercised here — it lives
behind subprocess calls already covered by resume-correctness /
silent-degradation suites. These tests target the pure planning helper
``_compute_montage_plan`` + the default wiring it receives from config.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline import montage_builder


MONTAGE_LOGGER = "gomoveshift.pipeline.montage_builder"


class TestComputeMontagePlan:
    """Pure unit tests for the montage cap planning helper.

    Isolating the planning math from the ffmpeg path lets us exercise
    every branch (cap hit vs not, floor hit vs not, count reduced vs
    not) without ever shelling out.
    """

    def test_montage_total_duration_at_cap(self):
        """A request that would produce 30s must cap at the configured
        15s limit — N*D ≤ 15 after planning."""
        # 10 peaks at 3s each = 30s, which the old hardcoded default
        # used to ship. With the new 15s cap this must shrink.
        n_used, clip_dur, capped = montage_builder._compute_montage_plan(
            peak_count=10,
            requested_total_duration=30.0,
            max_total_duration=15.0,
            min_clip_duration_sec=1.5,
        )
        assert capped is True
        assert n_used * clip_dur <= 15.0 + 1e-6, (
            f"cap violated: {n_used} * {clip_dur} = {n_used * clip_dur} > 15"
        )

    def test_montage_per_clip_floor_respected(self):
        """With many peaks relative to cap, clip duration may not drop
        below 1.5s — the operator explicitly rejected 1s as too short.
        Peak count must be reduced instead."""
        # 20 peaks at 15s total would give 0.75s/clip without a floor.
        n_used, clip_dur, capped = montage_builder._compute_montage_plan(
            peak_count=20,
            requested_total_duration=15.0,
            max_total_duration=15.0,
            min_clip_duration_sec=1.5,
        )
        assert clip_dur >= 1.5 - 1e-6, f"per-clip floor breached: {clip_dur} < 1.5"
        assert n_used <= 20, "should not invent peaks"
        assert n_used * clip_dur <= 15.0 + 1e-6

    def test_montage_below_cap_unchanged(self):
        """If the request is already under the cap, nothing changes —
        no trim, no cap log."""
        # 4 peaks at 2.5s/clip = 10s total, well under the 15s cap.
        n_used, clip_dur, capped = montage_builder._compute_montage_plan(
            peak_count=4,
            requested_total_duration=10.0,
            max_total_duration=15.0,
            min_clip_duration_sec=1.5,
        )
        assert capped is False
        assert n_used == 4
        assert clip_dur == pytest.approx(2.5)

    def test_montage_max_duration_config_overridable(self):
        """Operator-driven override: moving the cap to 10s must shrink
        the output accordingly (the 30s request must now cap at 10)."""
        n_used, clip_dur, capped = montage_builder._compute_montage_plan(
            peak_count=10,
            requested_total_duration=30.0,
            max_total_duration=10.0,
            min_clip_duration_sec=1.5,
        )
        assert capped is True
        assert n_used * clip_dur <= 10.0 + 1e-6

    def test_montage_cap_logged_at_info(self, caplog):
        """Regression pin: when the cap fires, the INFO log line with
        the ``montage capped`` sentinel is emitted so operators can
        grep it out of the 10-hour drive log."""
        with caplog.at_level(logging.INFO, logger=MONTAGE_LOGGER):
            montage_builder._compute_montage_plan(
                peak_count=10,
                requested_total_duration=30.0,
                max_total_duration=15.0,
                min_clip_duration_sec=1.5,
            )
        messages = [r.getMessage() for r in caplog.records if r.name == MONTAGE_LOGGER]
        assert any("montage capped" in m for m in messages), (
            f"expected 'montage capped' INFO line, got: {messages!r}"
        )
        # Informational payload: the original requested duration is in
        # the message so the operator can see what was shaved off.
        assert any("30" in m for m in messages), (
            f"expected original 30s in log line, got: {messages!r}"
        )

    def test_montage_no_cap_log_when_below_cap(self, caplog):
        """Negative companion: when no cap fires, no ``montage capped``
        log line appears. Keeps the drive log quiet on fast paths."""
        with caplog.at_level(logging.INFO, logger=MONTAGE_LOGGER):
            montage_builder._compute_montage_plan(
                peak_count=4,
                requested_total_duration=10.0,
                max_total_duration=15.0,
                min_clip_duration_sec=1.5,
            )
        messages = [r.getMessage() for r in caplog.records if r.name == MONTAGE_LOGGER]
        assert not any("montage capped" in m for m in messages), (
            f"unexpected cap log on below-cap input: {messages!r}"
        )


class TestTrimPeakReencodeOnProbeFailure:
    """``_trim_peak_to`` shares the segment extractor's cut-boundary slop —
    on keyframe-probe FAILURE a `-c copy` would snap back to the prior keyframe
    and replay pre-peak footage, so the centred clip is RE-ENCODED with an
    accurate seek instead."""

    def test_probe_failure_reencodes_montage_clip(self, tmp_path):
        master = tmp_path / "master.mp4"
        master.write_bytes(b"fake")
        peak = SimpleNamespace(source_file_index=0, start_time=100.0, end_time=140.0)
        boundaries = [(0.0, 600.0)]
        out = tmp_path / "clip.mp4"

        commands: list[list[str]] = []

        def _fake_run(cmd, *a, **kw):
            commands.append(list(cmd))
            if cmd[0] == "ffprobe":
                # rc=0 but no keyframe packets → probe_keyframes returns None.
                return MagicMock(returncode=0, stdout=b"", stderr=b"")
            Path(cmd[-1]).parent.mkdir(parents=True, exist_ok=True)
            Path(cmd[-1]).write_bytes(b"x")
            return MagicMock(returncode=0, stdout=b"", stderr=b"")

        # The cut primitives now live in the ``keycut`` package.
        import keycut

        keycut.clear_keyframe_cache()
        with (
            patch("keycut.extract.subprocess.run", side_effect=_fake_run),
            patch("keycut.keyframes.subprocess.run", side_effect=_fake_run),
        ):
            ok = montage_builder._trim_peak_to(master, peak, boundaries, out, clip_duration=5.0)

        assert ok is True
        extract_cmds = [c for c in commands if c[0] == "ffmpeg" and "-ss" in c]
        assert len(extract_cmds) == 1
        cmd = extract_cmds[0]
        # Re-encoded (has `-c:v` encoder), not a `-c copy` passthrough.
        assert "-c:v" in cmd, cmd
        assert not any(cmd[i] == "-c" and cmd[i + 1] == "copy" for i in range(len(cmd) - 1))


class TestConfigDefault:
    """The authoritative cap lives at ``config.drive.montage_max_duration_sec``.
    The default is 15.0s; the OST timing reads the same field to align
    the OST title crossfade window with the montage tail."""

    def test_drive_config_has_montage_max_duration_sec_default_15(self):
        from gomoveshift.config import Config

        cfg = Config()
        assert hasattr(cfg.drive, "montage_max_duration_sec"), (
            "config.drive.montage_max_duration_sec missing — OST timing depends on it"
        )
        assert cfg.drive.montage_max_duration_sec == pytest.approx(15.0)

    def test_module_default_matches_config_default(self):
        """``montage_builder.MONTAGE_DURATION_SEC`` is the module-level
        authoritative duration and must match the config default. If
        they drift, the OST location window gets wrong."""
        from gomoveshift.config import Config

        cfg = Config()
        assert montage_builder.MONTAGE_DURATION_SEC == pytest.approx(
            cfg.drive.montage_max_duration_sec
        )
