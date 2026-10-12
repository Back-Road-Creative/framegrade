"""Story beats and chapter cues from route data."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pytest

from gomoveshift.pipeline.route_story.beats import (
    Beat,
    beats_receipt,
    extract_beats,
    write_chapters_vtt,
)
from gomoveshift.pipeline.route_story.metrics import RoutePoint


@dataclass(frozen=True)
class SimpleMedia:
    route_time_s: float | None
    lat: float | None
    lon: float | None


def point(time_s: float, lon: float, elevation_m: float | None) -> RoutePoint:
    return RoutePoint(time_s=time_s, latitude=0.0, longitude=lon, elevation_m=elevation_m)


def _segments() -> list[list[RoutePoint]]:
    segment1 = [
        point(0, 0.0, 100.0),  # start, moving
        point(5, 0.0005, 100.0),  # fast approach into the dwell spot
        point(6, 0.0005, 100.0),  # dwell begins
        point(50, 0.0005, 100.0),  # dwell ends (44s stationary from the prior point)
        point(60, 0.0015, 150.0),  # climbing resumes
        point(70, 0.0025, 250.0),  # top of the steepest window
    ]
    segment2 = [
        point(200, 0.0035, 300.0),  # highest point overall, after a signal gap
        point(210, 0.004, 290.0),  # finish
    ]
    return [segment1, segment2]


def _media() -> list[SimpleMedia]:
    return [
        SimpleMedia(route_time_s=61.0, lat=0.0, lon=0.0016),
        SimpleMedia(route_time_s=64.0, lat=0.0, lon=0.0017),  # clusters with the one above
        SimpleMedia(route_time_s=140.0, lat=0.0, lon=0.003),  # separate cluster (76s away)
        SimpleMedia(route_time_s=150.0, lat=None, lon=None),  # unanchored: no coordinates
        SimpleMedia(route_time_s=None, lat=0.0, lon=0.001),  # unanchored: no route time
    ]


def test_extract_beats_orders_every_kind_by_route_time_with_deterministic_ties():
    beats = extract_beats(
        _segments(),
        media=_media(),
        min_stop_s=30.0,
        climb_window_m=250.0,
        media_cluster_s=60.0,
    )

    kinds_and_times = [(beat.kind, beat.route_time_s) for beat in beats]
    assert kinds_and_times == [
        ("start", 0.0),
        ("stop", 5.0),
        ("media", 61.0),
        ("steepest_climb", 70.0),
        ("signal_gap", 70.0),
        ("media", 140.0),
        ("high_point", 200.0),
        ("finish", 210.0),
    ]


def test_start_and_finish_always_present_with_data_only_facts():
    beats = extract_beats(_segments(), min_stop_s=30.0, climb_window_m=250.0)

    start = next(beat for beat in beats if beat.kind == "start")
    finish = next(beat for beat in beats if beat.kind == "finish")

    assert start.shot == "reveal"
    assert start.duration_s == 7.0
    assert start.facts == (("Elevation", "328 ft"),)

    assert finish.shot == "pullback"
    assert finish.lat == 0.0
    assert finish.lon == pytest.approx(0.004)
    assert finish.facts == (("Distance", "0.2 mi"), ("Duration", "1 min"))


def test_stop_beat_merges_adjacent_slow_edges_into_one_dwell():
    beats = extract_beats(_segments(), min_stop_s=30.0, climb_window_m=250.0)

    stop = next(beat for beat in beats if beat.kind == "stop")
    assert stop.route_time_s == 5.0
    assert stop.shot == "hold"
    assert stop.facts == (("Duration", "45 sec"),)


def test_short_dwell_below_min_stop_s_is_not_a_beat():
    beats = extract_beats(_segments(), min_stop_s=60.0, climb_window_m=250.0)

    assert not [beat for beat in beats if beat.kind == "stop"]


def test_high_point_is_the_max_elevation_point():
    beats = extract_beats(_segments(), min_stop_s=30.0, climb_window_m=250.0)

    high_point = next(beat for beat in beats if beat.kind == "high_point")
    assert high_point.route_time_s == 200.0
    assert high_point.elevation_m == 300.0
    assert high_point.shot == "arc"
    assert high_point.facts == (("Elevation", "984 ft"),)


def test_high_point_omitted_when_no_elevation_evidence():
    segments = [[point(0, 0.0, None), point(10, 0.001, None)]]
    beats = extract_beats(segments, min_stop_s=30.0, climb_window_m=250.0)

    assert not [beat for beat in beats if beat.kind == "high_point"]


def test_steepest_climb_prefers_the_larger_gain_over_the_wider_window():
    beats = extract_beats(_segments(), min_stop_s=30.0, climb_window_m=250.0)

    climb = next(beat for beat in beats if beat.kind == "steepest_climb")
    assert climb.route_time_s == 70.0
    assert climb.shot == "follow"
    assert climb.facts == (("Climb", "492 ft over 0.1 mi"),)


def test_steepest_climb_omitted_when_no_positive_gain_found():
    segments = [[point(0, 0.0, 200.0), point(10, 0.001, 150.0), point(20, 0.002, 100.0)]]
    beats = extract_beats(segments, min_stop_s=30.0, climb_window_m=250.0)

    assert not [beat for beat in beats if beat.kind == "steepest_climb"]


def test_media_clusters_group_within_window_and_skip_unanchored_items():
    beats = extract_beats(
        _segments(), media=_media(), min_stop_s=30.0, climb_window_m=250.0, media_cluster_s=60.0
    )

    media_beats = [beat for beat in beats if beat.kind == "media"]
    assert [beat.route_time_s for beat in media_beats] == [61.0, 140.0]
    assert media_beats[0].facts == (("Media", "2"),)
    assert media_beats[1].facts == (("Media", "1"),)
    assert media_beats[0].shot == "arrival"


def test_signal_gap_carries_the_gap_length_as_a_fact():
    beats = extract_beats(_segments(), min_stop_s=30.0, climb_window_m=250.0)

    gap = next(beat for beat in beats if beat.kind == "signal_gap")
    assert gap.route_time_s == 70.0
    assert gap.shot == "hold"
    assert gap.facts == (("Gap", "2 min"),)


def test_extract_beats_rejects_empty_input():
    with pytest.raises(ValueError):
        extract_beats([])


_CUE_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\.(\d{3}) --> (\d{2}):(\d{2}):(\d{2})\.(\d{3})$")


def _parse_vtt(text: str) -> list[tuple[float, float, str]]:
    lines = text.splitlines()
    assert lines[0] == "WEBVTT"
    cues: list[tuple[float, float, str]] = []
    i = 1
    while i < len(lines):
        if not lines[i]:
            i += 1
            continue
        match = _CUE_RE.match(lines[i])
        assert match, f"invalid cue timing line: {lines[i]!r}"
        start = (
            int(match.group(1)) * 3600
            + int(match.group(2)) * 60
            + int(match.group(3))
            + int(match.group(4)) / 1000
        )
        end = (
            int(match.group(5)) * 3600
            + int(match.group(6)) * 60
            + int(match.group(7))
            + int(match.group(8)) / 1000
        )
        title = lines[i + 1]
        cues.append((start, end, title))
        i += 2
    return cues


def test_write_chapters_vtt_writes_valid_cues_ending_at_next_beat_or_film_end(tmp_path):
    beats = [
        Beat("start", 0.0, 0.0, 0.0, 100.0, "reveal", (), 7.0),
        Beat("high_point", 10.0, 0.0, 0.0, 300.0, "arc", (("Elevation", "984 ft"),), 7.0),
        Beat("finish", 20.0, 0.0, 0.0, 290.0, "pullback", (), 5.0),
    ]
    out = tmp_path / "chapters.vtt"

    write_chapters_vtt(beats, film_time=lambda t: t * 2.0, path=out, film_duration_s=50.0)

    cues = _parse_vtt(out.read_text(encoding="utf-8"))
    assert cues == [
        (0.0, 20.0, "Start"),
        (20.0, 40.0, "High point · 984 ft"),
        (40.0, 50.0, "Finish"),
    ]


def test_write_chapters_vtt_rejects_non_monotonic_film_time(tmp_path):
    beats = [
        Beat("start", 0.0, 0.0, 0.0, None, "reveal", (), 7.0),
        Beat("finish", 10.0, 0.0, 0.0, None, "pullback", (), 5.0),
    ]

    with pytest.raises(ValueError):
        write_chapters_vtt(
            beats, film_time=lambda t: 5.0 - t, path=tmp_path / "chapters.vtt", film_duration_s=20.0
        )


def test_beats_receipt_is_json_safe(tmp_path):
    import json

    beats = extract_beats(_segments(), min_stop_s=30.0, climb_window_m=250.0)
    rows = beats_receipt(beats)

    encoded = json.dumps(rows)
    decoded = json.loads(encoded)
    assert decoded[0]["kind"] == "start"
    assert decoded[0]["facts"] == [["Elevation", "328 ft"]]
