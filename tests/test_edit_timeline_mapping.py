"""EditTimeline: output<->source mapping driven by an assembly receipt."""

import copy
import json
import random
from pathlib import Path

import pytest

from gomoveshift.analysis.shorts_time_mapping import map_from_edited_time
from gomoveshift.assembly.assembler import VideoAssembler
from gomoveshift.assembly.encoder import Chapter
from gomoveshift.config import Config
from gomoveshift.pipeline.chapters.marker_merger import MarkerMerger
from gomoveshift.pipeline.edit_timeline import (
    Crossfade,
    EditTimeline,
    Insert,
    SourcePosition,
    load_receipt,
)

FIXTURE = Path(__file__).parent / "fixtures" / "assembly_receipt_v1.json"


def _receipt(spans):
    """Build a minimal valid receipt from (source_start, source_end) tuples."""
    out = 0.0
    rows = []
    for i, (s, e) in enumerate(spans):
        rows.append(
            {
                "index": i,
                "requested_indices": [0],
                "merged_start": s,
                "merged_end": e,
                "source_start": s,
                "source_end": e,
                "method": "reencode",
                "reason": "keyframes_unknown",
                "output_start": out,
                "output_end": out + (e - s),
                "duration_source": "planned",
            }
        )
        out += e - s
    return load_receipt(
        {
            "schema": "keycut-assembly-receipt/1",
            "provenance": "planned",
            "keycut_version": None,
            "master": {"path": "m.mp4", "size_bytes": None},
            "time_base": "seconds",
            "max_gap_sec": 2.0,
            "requested": [[s, e] for s, e in spans],
            "spans": rows,
            "output": {"path": "o.mp4", "duration": out},
        }
    )


def _times(positions):
    return [p.time for p in positions]


class TestReceiptTimeline:
    def setup_method(self):
        self.tl = EditTimeline.from_receipt(load_receipt(FIXTURE))

    def test_inside_spans(self):
        assert self.tl.output_to_source(0.0) == (SourcePosition(11.5, 1.0, 0),)
        assert _times(self.tl.output_to_source(10.0)) == [21.5]
        assert _times(self.tl.output_to_source(30.0)) == [301.5]
        assert _times(self.tl.output_to_source(60.0)) == [101.5]

    def test_half_open_output_boundaries(self):
        # span 0 is [0, 28.5): 28.5 belongs to span 1, not span 0.
        assert self.tl.output_to_source(28.5) == (SourcePosition(300.0, 1.0, 1),)
        assert self.tl.output_to_source(118.5) == ()  # end of last span is excluded
        assert self.tl.output_to_source(-0.001) == ()

    def test_source_to_output_hits_and_cuts(self):
        assert self.tl.source_to_output(11.5) == 0.0  # snapped keyframe
        assert self.tl.source_to_output(300.0) == 28.5  # out of source order
        assert self.tl.source_to_output(100.0) == 58.5
        assert self.tl.source_to_output(129.0) == pytest.approx(87.5)  # merged range

    def test_source_to_output_none_when_cut(self):
        assert self.tl.source_to_output(10.0) is None  # before the snapped start
        assert self.tl.source_to_output(11.4999) is None
        assert self.tl.source_to_output(40.0) is None  # half-open end of span 0
        assert self.tl.source_to_output(200.0) is None
        assert self.tl.source_to_output(330.0) is None
        assert self.tl.source_to_output(160.0) is None

    def test_round_trip_for_every_source_second_kept(self):
        for t in (11.5, 20.0, 39.999, 300.0, 329.5, 100.0, 159.9):
            out = self.tl.source_to_output(t)
            assert out is not None
            assert _times(self.tl.output_to_source(out)) == [pytest.approx(t)]

    def test_jump_at_cut_is_not_interpolated(self):
        before = self.tl.output_to_source(28.5 - 1e-6)[0].time
        after = self.tl.output_to_source(28.5)[0].time
        assert before == pytest.approx(40.0, abs=1e-5)
        assert after == 300.0
        # No output time ever maps into the cut between them.
        rng = random.Random(7)
        for _ in range(500):
            t = rng.uniform(0, 118.49)
            (pos,) = self.tl.output_to_source(t)
            kept = 11.5 <= pos.time < 40.0 or 300.0 <= pos.time < 330.0 or 100.0 <= pos.time < 160.0
            assert kept, pos

    def test_empty_receipt(self):
        tl = EditTimeline.from_receipt(_receipt([]))
        assert tl.output_to_source(0.0) == ()
        assert tl.source_to_output(0.0) is None


class TestContinuityAndFps:
    def test_continuity_within_a_span(self):
        tl = EditTimeline.from_receipt(_receipt([(5.0, 25.0), (60.0, 80.0)]))
        rng = random.Random(1)
        ts = sorted(rng.uniform(0, 19.999) for _ in range(300))
        srcs = [tl.output_to_source(t)[0].time for t in ts]
        for (t0, s0), (t1, s1) in zip(zip(ts, srcs), zip(ts[1:], srcs[1:])):
            assert s1 - s0 == pytest.approx(t1 - t0, abs=1e-9)

    def test_fractional_fps_boundaries(self):
        frame = 1001 / 30000
        spans = [(100 * frame, 400 * frame), (900 * frame, 1000 * frame)]
        tl = EditTimeline.from_receipt(_receipt(spans))
        boundary = 300 * frame
        assert tl.output_to_source(boundary)[0].time == pytest.approx(900 * frame)
        one_before = boundary - frame
        assert tl.output_to_source(one_before)[0].time == pytest.approx(399 * frame)
        assert tl.source_to_output(400 * frame) is None
        assert tl.source_to_output(399 * frame) == pytest.approx(299 * frame)
        for k in range(0, 300):
            out = tl.source_to_output((100 + k) * frame)
            assert out == pytest.approx(k * frame, abs=1e-9)

    def test_out_of_order_spans(self):
        tl = EditTimeline.from_receipt(_receipt([(50.0, 60.0), (0.0, 10.0), (20.0, 30.0)]))
        assert tl.source_to_output(55.0) == 5.0
        assert tl.source_to_output(5.0) == 15.0
        assert tl.source_to_output(25.0) == 25.0
        assert _times(tl.output_to_source(12.0)) == [2.0]
        assert tl.source_to_output(10.0) is None


class TestInsert:
    def setup_method(self):
        self.tl = EditTimeline.from_receipt(
            _receipt([(0.0, 10.0), (50.0, 60.0)]), inserts=[Insert(at=10.0, duration=3.0)]
        )

    def test_hold_has_no_source(self):
        assert self.tl.output_to_source(10.0) == ()
        assert self.tl.output_to_source(12.999) == ()
        assert _times(self.tl.output_to_source(13.0)) == [50.0]
        assert _times(self.tl.output_to_source(9.999)) == [pytest.approx(9.999)]

    def test_later_spans_shift_by_the_hold(self):
        assert self.tl.source_to_output(50.0) == 13.0
        assert self.tl.source_to_output(9.0) == 9.0
        assert self.tl.output_to_source(23.0) == ()  # total is 23.0, end excluded

    def test_insert_inside_a_span_splits_it(self):
        tl = EditTimeline.from_receipt(_receipt([(0.0, 10.0)]), inserts=[Insert(4.0, 2.0)])
        assert _times(tl.output_to_source(3.0)) == [3.0]
        assert tl.output_to_source(5.0) == ()
        assert _times(tl.output_to_source(6.0)) == [4.0]
        assert tl.source_to_output(4.0) == 6.0  # never interpolated across the hold
        assert tl.source_to_output(3.999) == pytest.approx(3.999)

    def test_insert_at_start_and_end(self):
        tl = EditTimeline.from_receipt(
            _receipt([(0.0, 10.0)]), inserts=[Insert(0.0, 1.0), Insert(10.0, 2.0)]
        )
        assert tl.output_to_source(0.5) == ()
        assert _times(tl.output_to_source(1.0)) == [0.0]
        assert tl.output_to_source(11.5) == ()
        assert tl.source_to_output(0.0) == 1.0

    def test_two_inserts_accumulate(self):
        tl = EditTimeline.from_receipt(
            _receipt([(0.0, 10.0), (20.0, 30.0)]),
            inserts=[Insert(10.0, 1.0), Insert(0.0, 2.0)],
        )
        assert tl.source_to_output(20.0) == 13.0
        assert tl.source_to_output(0.0) == 2.0


class TestCrossfade:
    def setup_method(self):
        self.tl = EditTimeline.from_receipt(
            _receipt([(10.0, 20.0), (30.0, 50.0)]),
            crossfades=[Crossfade(after_span=0, duration=2.0)],
        )

    def test_duration_is_sum_minus_overlap(self):
        assert self.tl.output_to_source(27.999) != ()
        assert self.tl.output_to_source(28.0) == ()

    def test_single_position_outside_the_overlap(self):
        assert self.tl.output_to_source(0.0) == (SourcePosition(10.0, 1.0, 0),)
        assert self.tl.output_to_source(7.999)[0].span_index == 0
        assert len(self.tl.output_to_source(7.999)) == 1
        assert self.tl.output_to_source(10.0) == (SourcePosition(32.0, 1.0, 1),)

    def test_two_weighted_positions_inside_the_overlap(self):
        a, b = self.tl.output_to_source(9.0)  # halfway through the 8..10 overlap
        assert (a.span_index, a.time, a.weight) == (0, 19.0, pytest.approx(0.5))
        assert (b.span_index, b.time, b.weight) == (1, 31.0, pytest.approx(0.5))
        assert a.weight + b.weight == pytest.approx(1.0)

    def test_overlap_boundaries_are_half_open(self):
        start = self.tl.output_to_source(8.0)  # overlap begins: incoming weight 0
        assert [p.span_index for p in start] == [0, 1]
        assert start[1].weight == 0.0 and start[0].weight == 1.0
        assert [p.span_index for p in self.tl.output_to_source(10.0)] == [1]

    def test_source_to_output_inside_the_fade_keeps_each_spans_own_clock(self):
        assert self.tl.source_to_output(19.0) == 9.0
        assert self.tl.source_to_output(31.0) == 9.0
        assert self.tl.source_to_output(30.0) == 8.0

    def test_three_spans_two_seams(self):
        tl = EditTimeline.from_receipt(
            _receipt([(0.0, 10.0), (20.0, 30.0), (40.0, 50.0)]),
            crossfades=[Crossfade(0, 1.0), Crossfade(1, 2.0)],
        )
        assert tl.source_to_output(20.0) == 9.0
        assert tl.source_to_output(40.0) == 17.0
        assert tl.output_to_source(26.999) != () and tl.output_to_source(27.0) == ()

    def test_crossfade_then_insert(self):
        tl = EditTimeline.from_receipt(
            _receipt([(10.0, 20.0), (30.0, 50.0)]),
            crossfades=[Crossfade(0, 2.0)],
            inserts=[Insert(at=28.0, duration=1.0)],
        )
        assert tl.output_to_source(28.5) == ()
        assert tl.output_to_source(29.0) == ()

    def test_receipt_spans_are_unchanged_by_zero_overlap_inputs(self):
        plain = EditTimeline.from_receipt(_receipt([(10.0, 20.0), (30.0, 50.0)]))
        assert _times(plain.output_to_source(10.0)) == [30.0]


class TestInvalidExtras:
    def _r(self):
        return _receipt([(0.0, 10.0), (20.0, 30.0)])

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"crossfades": [Crossfade(1, 1.0)]},  # no span after the last
            {"crossfades": [Crossfade(0, 0.0)]},
            {"crossfades": [Crossfade(0, 11.0)]},  # longer than the spans
            {"crossfades": [Crossfade(0, 1.0), Crossfade(0, 1.0)]},
            {"crossfades": [Crossfade(0, float("nan"))]},
            {"inserts": [Insert(-1.0, 1.0)]},
            {"inserts": [Insert(21.0, 1.0)]},
            {"inserts": [Insert(5.0, 0.0)]},
            {"inserts": [Insert(5.0, float("inf"))]},
        ],
    )
    def test_rejected(self, kwargs):
        with pytest.raises(ValueError):
            EditTimeline.from_receipt(self._r(), **kwargs)

    def test_insert_inside_a_crossfade_is_rejected(self):
        with pytest.raises(ValueError, match="crossfade"):
            EditTimeline.from_receipt(
                self._r(), crossfades=[Crossfade(0, 4.0)], inserts=[Insert(7.0, 1.0)]
            )


class TestLegacyMappers:
    """The three legacy mappers delegate to EditTimeline and keep their answers."""

    SEGS = [(10.0, 20.0), (30.0, 50.0), (60.0, 65.0)]

    def test_chapter_and_shorts_mappers_agree_on_a_crossfaded_fixture(self):
        cf = 2.0
        tl = EditTimeline.from_segments(self.SEGS, crossfade=cf)
        # Source seconds outside every seam overlap (each seam's overlap is the
        # last `cf` seconds of the outgoing span).
        probes = [10.0, 12.5, 17.9, 30.0, 33.0, 47.9, 60.0, 62.0, 64.5]
        for src in probes:
            edited = MarkerMerger.map_to_edited_time(src, self.SEGS, crossfade=cf)
            assert edited == pytest.approx(tl.source_to_output(src))
            assert map_from_edited_time(edited, self.SEGS, cf) == pytest.approx(src)
        # Inside an overlap the shorts mapper answers with the incoming span.
        edited = MarkerMerger.map_to_edited_time(19.0, self.SEGS, crossfade=cf)
        assert map_from_edited_time(edited, self.SEGS, cf) == pytest.approx(31.0)

    def test_hard_cut_chapter_mapper_unchanged(self):
        segs = [(10.0, 20.0), (30.0, 40.0)]
        assert MarkerMerger.map_to_edited_time(5.0, segs) == 0.0  # before: snap
        assert MarkerMerger.map_to_edited_time(25.0, segs) == 10.0  # in a cut: snap
        assert MarkerMerger.map_to_edited_time(20.0, segs) == 10.0  # inclusive end
        assert MarkerMerger.map_to_edited_time(35.0, segs) == 15.0
        assert MarkerMerger.map_to_edited_time(99.0, segs) == 20.0  # past all
        assert MarkerMerger.map_to_edited_time(7.0, None) == 7.0

    def test_shorts_mapper_edges_unchanged(self):
        assert map_from_edited_time(-3.0, self.SEGS) == 7.0
        assert map_from_edited_time(1e9, self.SEGS) == 65.0
        assert map_from_edited_time(10.0, self.SEGS) == 30.0  # boundary -> next
        assert map_from_edited_time(4.0, [(5.0, 5.0), (10.0, 20.0)]) == 14.0

    def test_assembler_chapters_use_the_timeline(self):
        assembler = VideoAssembler(Config())
        keep = [(10.0, 20.0), (30.0, 40.0)]
        out = assembler._adjust_chapters_for_trims(
            [Chapter(5.0, "cut"), Chapter(15.0, "a"), Chapter(20.0, "end"), Chapter(35.0, "b")],
            keep,
        )
        assert [(c.start_time, c.title) for c in out] == [(5.0, "a"), (15.0, "b")]

    def test_json_fixture_still_untouched(self):
        raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
        assert copy.deepcopy(raw) == raw
