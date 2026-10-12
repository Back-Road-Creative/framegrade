"""Truthful logging around the EOF-clamp and full-drive source-file counts.

The EOF clamp guards against extracting past a file's actual EOF, with a
deliberate 0.5s guard band. The clamp firing when a segment merely lands
INSIDE that guard band (planner range end and ffprobe duration differing by
a fraction of a second) is normal and must not read as a WARNING; only a
segment that genuinely extends past the file's real duration is worth one.

Separately, the full-drive per-day/legacy log lines report "(%d source
files)" using the SEGMENT count, which over-counts whenever a day's spans
are split into more pieces than there are files (e.g. a mid-file stop).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from gomoveshift.config import DriveConfig
from gomoveshift.pipeline.longform_assembler import LongFormAssembler

LOGGER_NAME = "gomoveshift.pipeline.longform_assembler"


def _all_high_scores(duration_sec: int) -> np.ndarray:
    """Dense (N, 9) score array, uniformly above any reasonable threshold."""
    n = int(duration_sec)
    scores = np.zeros((n, 9), dtype=np.float64)
    scores[:, 0] = np.arange(n, dtype=np.float64)
    scores[:, 1] = 0.8  # composite
    scores[:, 2] = 0.05  # motion energy
    return scores


class TestEofClampLogLevel:
    def test_in_band_clamp_is_debug_not_warning(self, caplog):
        """local_end lands inside the 0.5s EOF guard band, not past the
        file's real duration -> no WARNING, a DEBUG note instead."""
        cfg = DriveConfig(min_segment_duration=90.0, max_segment_duration=0.0)
        assembler = LongFormAssembler(cfg)
        scores = _all_high_scores(600)
        # File duration matches the score array exactly: the segment's
        # computed end_time (600.0) sits just inside the guard band
        # (> 599.5) but is NOT past the real 600.0s file duration.
        file_boundaries = [(0.0, 600.0)]

        with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
            segments = assembler.find_segments_above_threshold(
                scores, threshold=0.5, file_boundaries=file_boundaries
            )

        assert segments, "expected at least one above-threshold segment"
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert not warnings, f"in-band EOF-guard clamp should not warn: {warnings}"
        debugs = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("guard band" in m for m in debugs), debugs

    def test_genuine_overrun_is_warning(self, caplog):
        """local_end genuinely exceeds file_duration -> WARNING, truthfully
        worded ('exceeds')."""
        cfg = DriveConfig(min_segment_duration=90.0, max_segment_duration=0.0)
        assembler = LongFormAssembler(cfg)
        scores = _all_high_scores(600)
        # File boundary claims only 550s, well short of the segment's
        # computed 600.0s end -> a real overrun, not just guard-band noise.
        file_boundaries = [(0.0, 550.0)]

        with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
            segments = assembler.find_segments_above_threshold(
                scores, threshold=0.5, file_boundaries=file_boundaries
            )

        assert segments
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("exceeds" in m for m in warnings), warnings


class TestFullDriveSourceFileCount:
    def _cfg(self) -> DriveConfig:
        return DriveConfig(
            lf_selection_mode="full_drive",
            full_drive_max_duration_sec=7200,
            min_lf_duration=100.0,
            min_segment_duration=30.0,
            min_motion_energy=0.0,
        )

    def _keep_all_mask(self, duration_sec: int, stop_start: int, stop_end: int) -> np.ndarray:
        mask = np.zeros(int(duration_sec), dtype=bool)
        mask[stop_start:stop_end] = True
        return mask

    def test_per_day_log_counts_distinct_files_not_segments(self, caplog):
        """A mid-file stop splits one file's span into two pieces, so a
        2-file day yields 3 LFSegments. The log must still say 2 source
        files, not 3."""
        cfg = self._cfg()
        assembler = LongFormAssembler(cfg)
        n = 1200
        scores = np.zeros((n, 11), dtype=np.float64)
        scores[:, 0] = np.arange(n, dtype=np.float64)
        scores[:, 1] = 0.5 + 0.3 * np.sin(np.linspace(0, 6.28, n))
        scores[:, 2] = 0.6
        # Stop in the middle of file 0 (200s-260s) splits it into two pieces;
        # file 1 (600s-1200s) is untouched -> 3 segments, 2 source files.
        stop_mask = self._keep_all_mask(n, 200, 260)

        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            plan = assembler.build_drive_plan_per_day(
                scores=scores,
                processed_files=[Path("/tmp/ep_00.mp4"), Path("/tmp/ep_01.mp4")],
                file_boundaries=[(0.0, 600.0), (600.0, 1200.0)],
                episode_dates=["2025-09-14", "2025-09-14"],
                stop_mask=stop_mask,
            )

        assert len(plan.regular_lfs) == 1
        lf = plan.regular_lfs[0]
        assert len(lf.segments) == 3, "fixture must actually split into 3 segments"
        distinct_files = {s.source_file_index for s in lf.segments}
        assert len(distinct_files) == 2

        msgs = [r.getMessage() for r in caplog.records if "[full_drive]" in r.getMessage()]
        assert msgs, "expected a [full_drive] summary log line"
        assert any("2 source files" in m for m in msgs), msgs
        assert not any("3 source files" in m for m in msgs), msgs

    def test_legacy_log_counts_distinct_files_not_segments(self, caplog):
        cfg = self._cfg()
        assembler = LongFormAssembler(cfg)
        n = 1200
        scores = np.zeros((n, 11), dtype=np.float64)
        scores[:, 0] = np.arange(n, dtype=np.float64)
        scores[:, 1] = 0.5 + 0.3 * np.sin(np.linspace(0, 6.28, n))
        scores[:, 2] = 0.6
        stop_mask = self._keep_all_mask(n, 200, 260)

        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            plan = assembler.build_drive_plan(
                scores=scores,
                processed_files=[Path("/tmp/ep_00.mp4"), Path("/tmp/ep_01.mp4")],
                file_boundaries=[(0.0, 600.0), (600.0, 1200.0)],
                stop_mask=stop_mask,
            )

        assert len(plan.regular_lfs) == 1
        lf = plan.regular_lfs[0]
        assert len(lf.segments) == 3, "fixture must actually split into 3 segments"

        msgs = [r.getMessage() for r in caplog.records if "Full-drive:" in r.getMessage()]
        assert msgs
        assert any("2 source files" in m for m in msgs), msgs
        assert not any("3 source files" in m for m in msgs), msgs
