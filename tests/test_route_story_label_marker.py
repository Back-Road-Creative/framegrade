"""A place-name pill never covers the traveller's marker.

At a summit that ends the day's route, the marker stood on the labelled
point, and the "Rocky Bald" pill was drawn straight over it: the film's subject
vanished at the film's destination.
"""

from __future__ import annotations

from pathlib import Path

from gomoveshift.pipeline.route_story import proof
from gomoveshift.pipeline.route_story.scene import capture as capture_mod

_SCENE_JS = (Path(capture_mod.__file__).parent / "scene.js").read_text(encoding="utf-8")


def _label(name: str, x: float, y: float, **extra) -> dict:
    return {"name": name, "x": x, "y": y, "visible": True, **extra}


def _marker(x: float, y: float) -> dict:
    return {"name": "", "kind": "marker", "x": x, "y": y, "visible": True}


def _overlaps(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def test_a_pill_on_the_marker_moves_off_it_and_is_still_shown():
    labels = [_label("Rocky Bald", 940, 470), _marker(960, 461)]

    (pill,) = proof.place_labels(labels, 1920, 1080)

    assert pill.text == "Rocky Bald"
    assert not _overlaps(pill.box, proof.marker_clear_box(960, 461, 1080))


def test_the_moved_pill_stays_beside_its_place():
    labels = [_label("Rocky Bald", 940, 470), _marker(960, 461)]

    (pill,) = proof.place_labels(labels, 1920, 1080)

    assert abs(pill.box[0] - 940) < 20
    assert min(abs(pill.box[1] - 470), abs(pill.box[3] - 470)) < 80


def test_the_text_moves_with_its_box():
    (before,) = proof.place_labels([_label("Rocky Bald", 940, 470)], 1920, 1080)
    (after,) = proof.place_labels([_label("Rocky Bald", 940, 470), _marker(960, 461)], 1920, 1080)

    assert after.text_xy[1] - after.box[1] == before.text_xy[1] - before.box[1]
    assert after.text_xy[0] - after.box[0] == before.text_xy[0] - before.box[0]


def test_a_pill_away_from_the_marker_is_untouched():
    alone = proof.place_labels([_label("Black Bald", 300, 300)], 1920, 1080)
    with_marker = proof.place_labels(
        [_label("Black Bald", 300, 300), _marker(960, 461)], 1920, 1080
    )

    assert with_marker == alone


def test_the_marker_is_found_wherever_it_sits_in_the_list():
    labels = [_marker(960, 461), _label("Rocky Bald", 940, 470)]

    (pill,) = proof.place_labels(labels, 1920, 1080)

    assert not _overlaps(pill.box, proof.marker_clear_box(960, 461, 1080))


def test_the_marker_never_becomes_a_pill():
    assert proof.place_labels([_marker(960, 461)], 1920, 1080) == []


def test_a_pill_with_nowhere_clear_to_go_is_left_out():
    # Earlier pills already hold the space above and below the marker.
    labels = [
        _label("Above", 940, 425),
        _label("Below", 940, 535),
        _label("Rocky Bald", 940, 470),
        _marker(960, 461),
    ]

    placed = proof.place_labels(labels, 1920, 1080)

    assert [pill.text for pill in placed] == ["Above", "Below"]


def test_the_scene_reports_where_the_marker_landed():
    assert 'kind: "marker"' in _SCENE_JS
    assert "projectLabels(pose.marker" in _SCENE_JS
