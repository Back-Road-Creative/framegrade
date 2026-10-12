"""Bracket detection is tuned against a hand-stepped card, not an AEB burst.

A hand-stepped sequence (``step_01..step_06``) walks 1/320 to
1/1600 at ISO 100 with 30/7/11/3/24-second gaps and one repeated exposure, so a
sub-2-second burst window finds nothing in it. Every fixture builds that shape
synthetically, so the thresholds stay pinned to realistic behaviour without a
binary in the tree.
"""

import datetime
import os
from pathlib import Path

import pytest

from gomoveshift.pipeline.photos.bracket import detect_brackets
from gomoveshift.pipeline.photos.exif_gps import (
    extract_exif_capture_metadata,
    extract_exif_capture_timestamp,
)
from test_photo_nef_exif import _raw

#: The hand-stepped sequence: (filename, clock time, shutter denominator).
STEPPED_SEQUENCE = [
    ("step_01.NEF", "10:15:00", 320),
    ("step_02.NEF", "10:15:30", 400),
    ("step_03.NEF", "10:15:37", 500),
    ("step_04.NEF", "10:15:48", 800),
    ("step_05.NEF", "10:15:51", 800),  # repeats step_04 — a duplicate, not a step
    ("step_06.NEF", "10:16:15", 1600),
]
STEPPED_GROUPING = [("step_01.NEF", "step_02.NEF", "step_03.NEF", "step_04.NEF", "step_06.NEF")]

#: That sequence's ``FNumber`` tags. It is NOT a held-aperture bracket — the
#: lens opens about 1.5 stops while the shutter shortens 2.3 — so an aperture-equality
#: rule would refuse a genuine bracket. What it does have is about 0.8 EV
#: of genuine range left over.
STEPPED_APERTURES = (10.0, 9.0, 9.0, 8.0, 8.0, 6.0)

#: Optional real-card check: a directory of real NEF files whose first six names (sorted)
#: are a hand-stepped sequence shaped like ``STEPPED_SEQUENCE``. Unset, the test skips.
STEPPED_CARD = Path(os.environ.get("GMS_TEST_RAW_CARD_DIR") or "/nonexistent-raw-card")


def _frame(tmp_path, name, at, shutter, *, iso=100, day="2026:07:22", aperture=None):
    return _raw(
        tmp_path / name,
        when=f"{day} {at}",
        iso=iso,
        exposure=(1, shutter),
        f_number=None if aperture is None else (round(aperture * 10), 10),
    )


def _card(tmp_path, rows, **kwargs):
    return [_frame(tmp_path, *row, **kwargs) for row in rows]


def _names(brackets):
    return [tuple(Path(p).name for p in group) for group in brackets]


class TestTheTimestampReader:
    def test_it_reads_seconds_off_a_raw(self, tmp_path):
        stamp = extract_exif_capture_timestamp(_frame(tmp_path, "a.NEF", "10:16:15", 1600))
        assert stamp == datetime.datetime(2026, 7, 22, 10, 16, 15)

    @pytest.mark.parametrize("when", [None, "not a date", "2026:07:22"])
    def test_a_missing_or_malformed_tag_reads_none_per_field(self, tmp_path, when):
        path = _raw(tmp_path / "x.NEF", when=when)
        assert extract_exif_capture_timestamp(path) is None
        assert extract_exif_capture_metadata(path).iso == 100  # never-raise is per field


class TestTheSteppedShape:
    def test_the_hand_stepped_sequence_is_one_bracket_without_its_duplicate(self, tmp_path):
        assert _names(detect_brackets(_card(tmp_path, STEPPED_SEQUENCE))) == STEPPED_GROUPING

    def test_the_output_does_not_depend_on_input_order(self, tmp_path):
        paths = _card(tmp_path, STEPPED_SEQUENCE)
        expected = detect_brackets(paths)
        assert _names(expected) == STEPPED_GROUPING
        assert detect_brackets(list(reversed(paths))) == expected
        assert detect_brackets(paths[3:] + paths[:3]) == expected

    def test_the_cards_own_apertures_do_not_break_it(self, tmp_path):
        """The regression the aperture rule must never cause: this sequence's
        aperture moves, and it is still a bracket — about 0.8 EV of range survives."""
        paths = [
            _frame(tmp_path, *row, aperture=f)
            for row, f in zip(STEPPED_SEQUENCE, STEPPED_APERTURES)
        ]
        assert _names(detect_brackets(paths)) == STEPPED_GROUPING

    def test_a_sweep_at_one_aperture_still_groups(self, tmp_path):
        """The textbook shape: aperture held, shutter alone."""
        paths = [_frame(tmp_path, *row, aperture=8.0) for row in STEPPED_SEQUENCE]
        assert _names(detect_brackets(paths)) == STEPPED_GROUPING

    @pytest.mark.skipif(
        len(sorted(STEPPED_CARD.glob("*.NEF"))) < len(STEPPED_SEQUENCE),
        reason="set GMS_TEST_RAW_CARD_DIR to a real NEF hand-stepped directory to run",
    )
    def test_a_real_card_groups_the_same_way(self):
        files = sorted(STEPPED_CARD.glob("*.NEF"))[: len(STEPPED_SEQUENCE)]
        expected = [tuple(files[i].name for i in (0, 1, 2, 3, 5))]
        assert _names(detect_brackets(files)) == expected


class TestWhatIsNotOneBracket:
    def test_fewer_than_three_frames_is_never_a_bracket(self, tmp_path):
        assert detect_brackets([]) == []
        assert detect_brackets(_card(tmp_path, STEPPED_SEQUENCE[:1])) == []
        assert detect_brackets(_card(tmp_path, STEPPED_SEQUENCE[:2])) == []  # a pair isn't evidence

    def test_a_gap_past_the_window_splits_the_run(self, tmp_path):
        rows = [
            ("a.NEF", "10:15:08", 320),
            ("b.NEF", "10:15:43", 400),
            ("c.NEF", "10:15:49", 500),
            ("d.NEF", "10:31:10", 320),  # 16 minutes later — a different scene
            ("e.NEF", "10:31:30", 400),
            ("f.NEF", "10:31:50", 500),
        ]
        assert _names(detect_brackets(_card(tmp_path, rows))) == [
            ("a.NEF", "b.NEF", "c.NEF"),
            ("d.NEF", "e.NEF", "f.NEF"),
        ]

    def test_an_iso_change_ends_the_run(self, tmp_path):
        rows = [("a.NEF", "10:15:08", 320), ("b.NEF", "10:15:20", 400), ("c.NEF", "10:15:30", 500)]
        paths = _card(tmp_path, rows) + [_frame(tmp_path, "d.NEF", "10:15:40", 800, iso=200)]
        assert _names(detect_brackets(paths)) == [("a.NEF", "b.NEF", "c.NEF")]

    def test_a_reversal_ends_the_run(self, tmp_path):
        rows = [
            ("a.NEF", "10:15:08", 320),
            ("b.NEF", "10:15:20", 500),
            ("c.NEF", "10:15:30", 800),
            ("d.NEF", "10:15:40", 400),  # back up — a new run, not a fourth step
        ]
        assert _names(detect_brackets(_card(tmp_path, rows))) == [("a.NEF", "b.NEF", "c.NEF")]

    def test_a_session_long_drift_does_not_chain_into_one_bracket(self, tmp_path):
        """Every neighbour is inside the gap window; only the span cap stops it."""
        rows = [
            (f"{i}.NEF", f"08:{i * 40 // 60:02d}:{i * 40 % 60:02d}", s)
            for i, s in enumerate([100, 125, 160, 200, 250, 320])
        ]
        (bracket,) = detect_brackets(_card(tmp_path, rows))
        assert _names([bracket]) == [("0.NEF", "1.NEF", "2.NEF", "3.NEF", "4.NEF")]

    def test_frames_either_side_of_midnight_never_group(self, tmp_path):
        pm = [("a.NEF", "23:59:40", 320), ("b.NEF", "23:59:50", 400), ("c.NEF", "23:59:58", 500)]
        am = [("d.NEF", "00:00:06", 640), ("e.NEF", "00:00:14", 800), ("f.NEF", "00:00:22", 1000)]
        paths = _card(tmp_path, pm) + _card(tmp_path, am, day="2026:07:23")
        assert _names(detect_brackets(paths)) == [
            ("a.NEF", "b.NEF", "c.NEF"),
            ("d.NEF", "e.NEF", "f.NEF"),
        ]

    def test_a_depth_of_field_sweep_is_not_a_bracket(self, tmp_path):
        """The one shutter ramp that isn't a bracket: stopping down in aperture
        priority lengthens the shutter by exactly what the aperture took away, so
        the run carries the whole signature — held ISO, one direction, seconds
        apart — while every frame meters the SAME. Merging it blends three depths
        of field to buy no exposure range at all.
        """
        rows = [("a.NEF", "10:15:08", 1600), ("b.NEF", "10:15:20", 800), ("c.NEF", "10:15:30", 400)]
        paths = [_frame(tmp_path, *row, aperture=f) for row, f in zip(rows, (2.8, 4.0, 5.6))]
        assert _names(detect_brackets(paths)) == []

    def test_a_run_missing_the_tag_falls_back_to_the_shutter_alone(self, tmp_path):
        """The light test needs ``FNumber`` on every frame; one file without it
        leaves the run grouped exactly as it was before the tag was read."""
        rows = [("a.NEF", "10:15:08", 320), ("b.NEF", "10:15:20", 400), ("c.NEF", "10:15:30", 500)]
        paths = [_frame(tmp_path, *row, aperture=8.0) for row in rows]
        paths.append(_frame(tmp_path, "d.NEF", "10:15:40", 800))  # no FNumber at all
        assert _names(detect_brackets(paths)) == [("a.NEF", "b.NEF", "c.NEF", "d.NEF")]

    def test_unreadable_frames_are_left_alone(self, tmp_path):
        rows = [("a.NEF", "10:15:08", 320), ("b.NEF", "10:15:20", 400), ("c.NEF", "10:15:30", 500)]
        paths = _card(tmp_path, rows)
        paths.append(_raw(tmp_path / "nodate.NEF", when=None))
        paths.append(_raw(tmp_path / "noiso.NEF", iso=None))
        (tmp_path / "junk.NEF").write_bytes(b"MM\x00\x2a" + b"\xff" * 60)
        paths.append(tmp_path / "junk.NEF")
        paths.append(tmp_path / "missing.NEF")
        assert _names(detect_brackets(paths)) == [("a.NEF", "b.NEF", "c.NEF")]
