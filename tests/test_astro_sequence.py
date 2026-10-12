"""Astro sequence detection is tuned against no reference card yet (see the
module docstring), so every fixture here builds a synthetic intervalometer
run by hand — steady cadence, held ISO and exposure — the same way
``test_photo_bracket.py`` rebuilds its own hand-stepped sequence.
"""

import datetime
import logging
from pathlib import Path

from gomoveshift.pipeline.photos.astro_sequence import (
    MAX_FRAME_GAP_S,
    MIN_SEQUENCE_FRAMES,
    SequenceGates,
    chunk_sequence,
    detect_sequences,
)
from test_photo_nef_exif import _raw

DAY = "2026:08:12"
START = datetime.datetime(2026, 8, 12, 22, 0, 0)


def _frame(tmp_path, name, offset_s, *, iso=3200, denom=None):
    at = (START + datetime.timedelta(seconds=offset_s)).strftime("%H:%M:%S")
    exposure = (1, denom) if denom else (15, 1)
    return _raw(tmp_path / name, when=f"{DAY} {at}", iso=iso, exposure=exposure)


def _run(tmp_path, n, *, gap_s=20, iso=3200, start_index=0):
    return [
        _frame(tmp_path, f"IMG_{start_index + i:04d}.NEF", i * gap_s, iso=iso) for i in range(n)
    ]


def _names(groups):
    return [tuple(Path(p).name for p in group) for group in groups]


class TestDetectSequences:
    def test_a_clean_constant_exposure_run_groups_whole(self, tmp_path):
        paths = _run(tmp_path, 20)
        (sequence,) = detect_sequences(paths)
        assert _names([sequence]) == [tuple(f"IMG_{i:04d}.NEF" for i in range(20))]

    def test_an_aeb_bracket_does_not_group(self, tmp_path):
        """A stepping bracket (the STEPPED_SEQUENCE shape in test_photo_bracket.py)
        never holds exposure constant frame to frame, so it never reaches
        MIN_SEQUENCE_FRAMES and produces no astro sequence at all."""
        rows = [(320, 0), (400, 30), (500, 37), (800, 48), (1600, 75)]
        paths = [
            _frame(tmp_path, f"DSC_{i:04d}.NEF", off, denom=d) for i, (d, off) in enumerate(rows)
        ]
        assert detect_sequences(paths) == []

    def test_a_gap_past_the_window_splits_the_run(self, tmp_path):
        first = _run(tmp_path, 6, start_index=0)
        gap_start = 5 * 20 + MAX_FRAME_GAP_S + 30  # last frame of ``first`` + past the window
        second = [_frame(tmp_path, f"IMG_{6 + i:04d}.NEF", gap_start + i * 20) for i in range(6)]
        assert _names(detect_sequences(first + second)) == [
            tuple(f"IMG_{i:04d}.NEF" for i in range(6)),
            tuple(f"IMG_{6 + i:04d}.NEF" for i in range(6)),
        ]

    def test_an_iso_change_ends_the_run(self, tmp_path):
        held = _run(tmp_path, 6, start_index=0)
        changed = [_frame(tmp_path, "IMG_0006.NEF", 6 * 20, iso=1600)]
        assert _names(detect_sequences(held + changed)) == [
            tuple(f"IMG_{i:04d}.NEF" for i in range(6))
        ]

    def test_fewer_than_the_minimum_is_never_a_sequence(self, tmp_path):
        assert detect_sequences([]) == []
        assert detect_sequences(_run(tmp_path, MIN_SEQUENCE_FRAMES - 1)) == []

    def test_unreadable_frames_are_excluded_without_error(self, tmp_path):
        paths = _run(tmp_path, 6)
        paths.append(_raw(tmp_path / "nodate.NEF", when=None))
        paths.append(_raw(tmp_path / "noiso.NEF", iso=None))
        (tmp_path / "junk.NEF").write_bytes(b"MM\x00\x2a" + b"\xff" * 60)
        paths.append(tmp_path / "junk.NEF")
        paths.append(tmp_path / "missing.NEF")
        assert _names(detect_sequences(paths)) == [tuple(f"IMG_{i:04d}.NEF" for i in range(6))]

    def test_the_output_does_not_depend_on_input_order(self, tmp_path):
        paths = _run(tmp_path, 20)
        expected = detect_sequences(paths)
        assert detect_sequences(list(reversed(paths))) == expected
        assert detect_sequences(paths[11:] + paths[:11]) == expected


class TestChunkSequence:
    def test_empty_run_chunks_to_nothing(self):
        assert chunk_sequence(()) == []

    def test_the_frame_cap_splits_a_chunk_and_the_short_remainder_merges(self, tmp_path):
        paths = tuple(_run(tmp_path, 12, gap_s=5))
        (run,) = detect_sequences(paths)
        chunks = chunk_sequence(run, stack_len=5, max_span_s=1_000_000.0)
        assert [len(c) for c in chunks] == [5, 7]  # trailing 2 merges into the 2nd chunk

    def test_the_span_cap_splits_a_chunk_on_its_own(self, tmp_path):
        paths = tuple(_run(tmp_path, 12, gap_s=20))
        (run,) = detect_sequences(paths)
        chunks = chunk_sequence(run, stack_len=1000, max_span_s=100.0)
        assert [len(c) for c in chunks] == [6, 6]

    def test_an_undersized_solo_chunk_is_dropped(self, tmp_path):
        paths = tuple(_run(tmp_path, MIN_SEQUENCE_FRAMES - 1, gap_s=5))
        assert chunk_sequence(paths, stack_len=1000, max_span_s=1_000_000.0) == []

    def test_a_short_trailing_chunk_is_dropped_when_the_merge_would_blow_the_span(self, tmp_path):
        paths = tuple(_run(tmp_path, 22, gap_s=20))
        (run,) = detect_sequences(paths)
        chunks = chunk_sequence(run, stack_len=20, max_span_s=380.0)
        assert [len(c) for c in chunks] == [20]  # trailing 2 dropped, not merged


class TestSparseNightGates:
    """A 960s-cadence (16-min) intervalometer night, a shape the default
    steady-cadence gates produce no astro sequence for. Only the
    frame-gap ceiling and minimum sequence length are library params here —
    ``chunk_sequence``'s ``max_span_s`` already exists and is exercised
    unchanged."""

    LOGGER_NAME = "gomoveshift.pipeline.photos.astro_sequence"

    def test_sequence_gates_is_sparse_reflects_either_override(self):
        assert not SequenceGates(MAX_FRAME_GAP_S, MIN_SEQUENCE_FRAMES).is_sparse
        assert SequenceGates(1000.0, MIN_SEQUENCE_FRAMES).is_sparse
        assert SequenceGates(MAX_FRAME_GAP_S, 3).is_sparse

    def test_default_gates_find_nothing_in_a_960s_cadence_run(self, tmp_path):
        """Pins the caveat: at today's steady-cadence defaults, a
        16-min-cadence night yields no astro sequence at all."""
        paths = _run(tmp_path, 5, gap_s=960)
        assert detect_sequences(paths) == []

    def test_sparse_gates_group_and_chunk_the_960s_cadence_run_whole(self, tmp_path):
        paths = _run(tmp_path, 5, gap_s=960)
        (sequence,) = detect_sequences(
            paths, max_frame_gap_s=1000.0, min_sequence_frames=MIN_SEQUENCE_FRAMES
        )
        assert _names([sequence]) == [tuple(f"IMG_{i:04d}.NEF" for i in range(5))]
        chunks = chunk_sequence(sequence, max_span_s=4_000.0)
        assert [len(c) for c in chunks] == [5]

    def test_sparse_gates_warn_once_default_gates_do_not(self, tmp_path, caplog):
        paths = _run(tmp_path, 5, gap_s=960)

        with caplog.at_level(logging.WARNING, logger=self.LOGGER_NAME):
            detect_sequences(paths)
        assert caplog.records == []

        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=self.LOGGER_NAME):
            detect_sequences(paths, max_frame_gap_s=1000.0, min_sequence_frames=MIN_SEQUENCE_FRAMES)
        assert len(caplog.records) == 1
        assert "blind" in caplog.records[0].message.lower()


class TestARunCrossingMidnight:
    """A steady 18 s cadence run that spans midnight is ONE sequence: the interval
    between 23:59:51 and 00:00:09 is an 18 s gap like every other one, so the date
    change must not split it."""

    @staticmethod
    def _across_midnight(tmp_path, n, *, gap_s=18):
        """A run centred on midnight: frame 0 is ``n//2`` intervals BEFORE 00:00, so the
        sequence straddles the date boundary with a steady cadence throughout."""
        start = datetime.datetime(2026, 8, 15, 23, 59, 51) - datetime.timedelta(
            seconds=(n // 2) * gap_s
        )
        return [
            _raw(
                tmp_path / f"DSC_{i:04d}.NEF",
                when=(start + datetime.timedelta(seconds=i * gap_s)).strftime("%Y:%m:%d %H:%M:%S"),
                iso=3200,
                exposure=(15, 1),
            )
            for i in range(n)
        ]

    def test_a_steady_cadence_run_is_one_sequence_across_the_date_boundary(self, tmp_path):
        """The gap check on the very next line already splits any cross-date pair further
        apart than ``max_frame_gap_s``, so a calendar-day rule can ONLY ever fire at
        midnight -- on exactly the nights this pipeline exists to process."""
        paths = self._across_midnight(tmp_path, 12)

        assert _names(detect_sequences(paths)) == [tuple(f"DSC_{i:04d}.NEF" for i in range(12))]

    def test_a_real_gap_across_midnight_still_splits(self, tmp_path):
        """Dropping the day rule must not cost the split that MATTERS: two separate
        shoots on either side of midnight are still two runs, on the gap alone."""
        first = self._across_midnight(tmp_path, 6, gap_s=18)
        last_at = datetime.datetime(2026, 8, 15, 23, 59, 51) - datetime.timedelta(seconds=3 * 18)
        last_at += datetime.timedelta(seconds=5 * 18)
        second = [
            _raw(
                tmp_path / f"DSC_{6 + i:04d}.NEF",
                when=(last_at + datetime.timedelta(seconds=MAX_FRAME_GAP_S + 30 + i * 18)).strftime(
                    "%Y:%m:%d %H:%M:%S"
                ),
                iso=3200,
                exposure=(15, 1),
            )
            for i in range(6)
        ]

        assert _names(detect_sequences(first + second)) == [
            tuple(f"DSC_{i:04d}.NEF" for i in range(6)),
            tuple(f"DSC_{6 + i:04d}.NEF" for i in range(6)),
        ]
