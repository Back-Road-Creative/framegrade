"""Chapter dedup + padding placement.

Padding: ``_pad_to_min_chapters`` spread interior pads evenly across the whole
video with no neighbor check, so a pad could land within
``YOUTUBE_MIN_CHAPTER_GAP_SEC`` of a real chapter (YouTube then refuses the
block) and could duplicate the name of an adjacent real chapter. Padding now
goes into the LARGEST gaps, is gap-checked against both neighbors, skips a
name that would duplicate an adjacent chapter, and the method validates its
own output (sorted, first at 0:00, every gap >= the minimum).

Dedup: ``merge_markers_with_chapters`` never collapsed consecutive
duplicate-NAME chapters, so two adjacent waypoints geocoding to the same town
produced "Entering X" twice in a row.

Dead-zone: the 30s dead-zone (``video_time < 30 or video_time > video_duration-30``)
silently dropped a legitimate early/late chapter. A real chapter inside the
dead-zone is now kept (still gap-deduped against Start/Finish) instead of
vanishing.
"""

from __future__ import annotations

from gomoveshift.pipeline.chapters.generator import (
    YOUTUBE_MIN_CHAPTER_GAP_SEC,
    YOUTUBE_MIN_CHAPTERS,
    ChapterGenerator,
)
from gomoveshift.pipeline.chapters.marker_merger import MarkerMerger


def _pad(chapters, video_duration, label=None):
    gen = object.__new__(ChapterGenerator)
    return gen._pad_to_min_chapters(list(chapters), video_duration, label)


def _secs(ts: str) -> int:
    s = 0
    for part in ts.split(":"):
        s = s * 60 + int(part)
    return s


def _gaps_ok(chapters) -> bool:
    times = [_secs(c["timestamp"]) for c in chapters]
    return all(b - a >= YOUTUBE_MIN_CHAPTER_GAP_SEC for a, b in zip(times, times[1:]))


# ── padding placement ────────────────────────────────────────────────


class TestPadPlacement:
    def test_pads_into_largest_gap_not_evenly(self):
        # Two real chapters bunched at the start; the whole tail is one big gap.
        chapters = [
            {"timestamp": "0:00", "title": "Start - A"},
            {"timestamp": "0:20", "title": "Finish - B"},
        ]
        out = _pad(chapters, video_duration=600.0, label="A")
        assert len(out) >= YOUTUBE_MIN_CHAPTERS
        assert out[0]["timestamp"] == "0:00"
        assert _gaps_ok(out), [c["timestamp"] for c in out]
        # The pad must land in the big 20s..600s gap, not crammed near 0:20.
        pad_times = [_secs(c["timestamp"]) for c in out if "En route" in c["title"]]
        assert pad_times and all(t > 30 for t in pad_times)

    def test_pad_never_violates_min_gap_against_neighbors(self):
        # Even-spread (old behaviour) over a video with a real chapter near the
        # midpoint would drop a pad within the min gap of it. The largest-gap
        # placement must keep every pad >= the min gap from both neighbours.
        chapters = [
            {"timestamp": "0:00", "title": "Start - A"},
            {"timestamp": "5:00", "title": "Entering Mid"},
        ]
        out = _pad(chapters, video_duration=600.0, label="A")
        assert len(out) >= YOUTUBE_MIN_CHAPTERS
        assert _gaps_ok(out), [c["timestamp"] for c in out]

    def test_pad_does_not_duplicate_adjacent_name(self):
        # An existing interior chapter already carries the exact pad name; a
        # pad landing next to it must not produce two consecutive identical
        # titles. needed=1 → the lone pad lands in the largest gap, which is
        # adjacent to the matching "En route — Townsville" chapter.
        chapters = [
            {"timestamp": "0:00", "title": "Start - A"},
            {"timestamp": "0:10", "title": "En route — Townsville"},
        ]
        out = _pad(chapters, video_duration=600.0, label="En route — Townsville")
        titles = [c["title"] for c in out]
        assert all(a != b for a, b in zip(titles, titles[1:])), titles

    def test_short_lf_left_untouched(self):
        chapters = [{"timestamp": "0:00", "title": "Start - A"}]
        out = _pad(chapters, video_duration=5.0, label="A")
        assert out == chapters

    def test_noop_when_already_renderable(self):
        chapters = [
            {"timestamp": "0:00", "title": "Start - A"},
            {"timestamp": "1:00", "title": "Entering B"},
            {"timestamp": "2:00", "title": "Finish - C"},
        ]
        out = _pad(chapters, video_duration=180.0, label="A")
        assert out == chapters


# ── consecutive duplicate-name collapse ──────────────────────────────


class TestConsecutiveDuplicateCollapse:
    def test_collapses_consecutive_same_name(self):
        class _Loc:
            def __init__(self, city):
                self.city = city
                self.name = city

        wpts = [
            (100.0, _Loc("Maple")),
            (200.0, _Loc("Maple")),  # consecutive duplicate town
            (300.0, _Loc("Birch")),
        ]
        merger = MarkerMerger()
        chapters = merger.merge_markers_with_chapters(
            wpts, None, video_duration=600.0, start_location="A", end_location="B"
        )
        titles = [c["title"] for c in chapters]
        assert all(a != b for a, b in zip(titles, titles[1:])), titles
        assert sum(1 for t in titles if "Maple" in t) == 1

    def test_collapses_same_name_across_a_terrain_band_change(self):
        """The collapse must compare the PLACE, not the decorated title.

        A band change decorates only the second of two identical towns, so a
        raw title comparison sees two different strings and keeps both — then
        the lower-third surface strips the decoration and burns "Maple" twice.
        """

        class _Loc:
            def __init__(self, city):
                self.city = city
                self.name = city

        wpts = [(100.0, _Loc("Maple")), (200.0, _Loc("Maple")), (300.0, _Loc("Birch"))]
        merger = MarkerMerger()
        chapters = merger.merge_markers_with_chapters(
            wpts,
            None,
            video_duration=600.0,
            start_location="A",
            end_location="B",
            terrain_bands=["motorway", None, None],
        )
        titles = [c["title"] for c in chapters]
        assert sum(1 for t in titles if "Maple" in t) == 1, titles


# ── dead-zone no longer drops a legitimate early/late chapter ─────────


class TestDeadZoneKeepsRealChapters:
    def test_early_chapter_not_dropped(self):
        class _Loc:
            def __init__(self, city):
                self.city = city
                self.name = city

        # A real waypoint at t=15s (inside the 30s head dead-zone) used to vanish.
        wpts = [(15.0, _Loc("Earlytown")), (300.0, _Loc("Midtown"))]
        merger = MarkerMerger()
        chapters = merger.merge_markers_with_chapters(
            wpts, None, video_duration=600.0, start_location="A", end_location="B"
        )
        titles = " ".join(c["title"] for c in chapters)
        assert "Earlytown" in titles
        times = [_secs(c["timestamp"]) for c in chapters]
        assert times == sorted(times)
        # Still renderable: monotonic, first at 0:00.
        assert chapters[0]["timestamp"] == "0:00"
