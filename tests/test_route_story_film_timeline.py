"""``FilmTimeline``: the film's own render loop plans which frames are
supposed to be a cut or a hold; this proves that plan matches what the
appending helpers actually wrote, and that the film receipt records the
rendered-pass continuity check without ever affecting delivery pass/fail.

No real film is rendered here -- synthetic frame lists / images only.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from gomoveshift.pipeline.route_story import assembly
from gomoveshift.pipeline.route_story import film


# --- FilmTimeline: recorded spans match what the helpers actually wrote ----


def test_append_hold_records_its_exact_span_on_the_timeline(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    image = Image.new("RGB", (4, 4), "red")
    timeline = assembly.FilmTimeline()
    segment = timeline.new_segment()
    fps = 30
    start_frame = 12

    next_frame = assembly._append_hold(
        frames, start_frame, image, fps, 0.6, timeline=timeline, timeline_segment=segment
    )

    assert next_frame == start_frame + 18  # 0.6s @ 30fps
    timeline.close_segment(segment, next_frame)
    cuts, holds = timeline.resolve()
    assert holds == [(start_frame, 18)]
    assert cuts == [0]  # the segment's own start is always a boundary cut


def test_append_hold_with_no_timeline_behaves_exactly_as_before(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    image = Image.new("RGB", (2, 2), "blue")

    next_frame = assembly._append_hold(frames, 0, image, 10, 0.5)

    assert next_frame == 5


def test_resolve_offsets_a_later_segments_holds_and_cuts_by_earlier_lengths() -> None:
    timeline = assembly.FilmTimeline()
    first = timeline.new_segment()
    timeline.record_cut(first, 5)
    timeline.close_segment(first, 40)

    second = timeline.new_segment()
    timeline.record_hold(second, 3, 18)
    timeline.close_segment(second, 60)

    cuts, holds = timeline.resolve()

    assert cuts == [0, 5, 40]  # 40 is the second segment's own boundary cut
    assert holds == [(43, 18)]  # 40 (segment base) + 3 (local)


def test_resolve_refuses_while_any_segment_is_still_open() -> None:
    timeline = assembly.FilmTimeline()
    timeline.new_segment()

    with pytest.raises(ValueError):
        timeline.resolve()


# --- delivery receipt wiring: record-only, never touches pass/fail --------


def test_delivery_continuity_records_the_analysis_under_delivered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "film.mp4"
    timeline = assembly.FilmTimeline()
    segment = timeline.new_segment()
    timeline.record_cut(segment, 4)
    timeline.close_segment(segment, 10)
    captured: dict = {}

    def fake_analyze(path, *, fps, cut_indices, hold_spans):
        captured["path"] = path
        captured["fps"] = fps
        captured["cut_indices"] = list(cut_indices)
        captured["hold_spans"] = list(hold_spans)
        return {"spikes": [], "frozen_spans": [], "stutter_windows": []}

    monkeypatch.setattr(film, "analyze_rendered_continuity", fake_analyze)

    delivered = film._delivery_continuity(output, 30, timeline, {"issues": []})

    assert delivered["continuity"] == {"spikes": [], "frozen_spans": [], "stutter_windows": []}
    assert delivered["issues"] == []  # existing delivery keys survive untouched
    assert captured["path"] == output
    assert captured["fps"] == 30
    assert captured["cut_indices"] == [0, 4]


def test_delivery_continuity_swallows_an_analysis_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "film.mp4"
    timeline = assembly.FilmTimeline()

    def boom(*args, **kwargs):
        raise RuntimeError("could not decode")

    monkeypatch.setattr(film, "analyze_rendered_continuity", boom)

    delivered = film._delivery_continuity(output, 30, timeline, {"issues": []})

    assert delivered["continuity"] == {"error": "could not decode"}
    assert delivered["issues"] == []


def test_delivery_continuity_leaves_a_non_dict_delivered_untouched(tmp_path: Path) -> None:
    timeline = assembly.FilmTimeline()

    assert film._delivery_continuity(tmp_path / "film.mp4", 30, timeline, None) is None


def test_delivery_continuity_leaves_a_stand_in_timeline_untouched(tmp_path: Path) -> None:
    # A patched render returns whatever it likes (an int from write_bytes, a
    # None); the receipt's delivery check must read exactly as before the
    # timeline existed.
    delivered = {"issues": [], "video": {"frames": 1}}

    assert film._delivery_continuity(tmp_path / "film.mp4", 30, 15, delivered) == delivered
    assert film._delivery_continuity(tmp_path / "film.mp4", 30, None, delivered) == delivered
