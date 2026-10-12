"""Scene label pills: readable at the delivery size, and never drawn over each other."""

from __future__ import annotations

import numpy as np

from gomoveshift.pipeline.route_story import proof
from gomoveshift.pipeline.route_story.layout import hud_layout


def _label(name: str, x: float, y: float, **extra) -> dict:
    return {"name": name, "x": x, "y": y, "visible": True, **extra}


def _overlaps(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def test_a_pill_that_would_cover_an_earlier_one_is_left_out():
    placed = proof.place_labels(
        [_label("Horse Cove Branch", 160, 90), _label("Bridge Gap", 130, 92)], 1280, 720
    )

    assert [p.text for p in placed] == ["Horse Cove Branch"]


def test_pills_with_room_between_them_are_all_kept():
    placed = proof.place_labels(
        [_label("Black Bald", 600, 230), _label("Rocky Bald", 900, 350)], 1280, 720
    )

    assert [p.text for p in placed] == ["Black Bald", "Rocky Bald"]
    assert not _overlaps(placed[0].box, placed[1].box)


def test_no_two_placed_pills_ever_overlap_in_a_crowded_frame():
    labels = [_label(f"Peak {i}", 100 + (i % 7) * 40, 100 + (i // 7) * 9) for i in range(40)]

    placed = proof.place_labels(labels, 1920, 1080)

    assert placed, "a crowded frame still shows labels"
    for i, first in enumerate(placed):
        for second in placed[i + 1 :]:
            assert not _overlaps(first.box, second.box)


def test_the_label_order_decides_who_wins_a_collision():
    first_wins = proof.place_labels(
        [_label("A Knob", 200, 200), _label("B Knob", 205, 203)], 1280, 720
    )
    second_wins = proof.place_labels(
        [_label("B Knob", 205, 203), _label("A Knob", 200, 200)], 1280, 720
    )

    assert [p.text for p in first_wins] == ["A Knob"]
    assert [p.text for p in second_wins] == ["B Knob"]


def test_label_text_grows_with_the_frame_so_1080p_is_not_tiny():
    small = proof.place_labels([_label("Tellico Bald", 300, 300)], 1280, 720)[0]
    large = proof.place_labels([_label("Tellico Bald", 300, 300)], 1920, 1080)[0]

    small_h = small.box[3] - small.box[1]
    large_h = large.box[3] - large.box[1]
    assert large_h >= 1.4 * small_h
    assert proof.label_font_px(1080) >= 20, "labels must be about 20-24 px at 1080p"


class TestAPillNeverRunsOffTheSideOfTheFrame:
    """A pill was anchored at its label's own x with no horizontal guard, so a
    label near a side edge drew its box past the frame and the name came out
    cut in half -- "Copper" lost 70 px off the right of a 1920-wide render.
    The vertical edges were already guarded; these are the other two.
    """

    def test_a_label_near_the_right_edge_keeps_its_whole_name_on_screen(self):
        (placed,) = proof.place_labels([_label("Copper", 1890, 500)], 1920, 1080)

        assert placed.box[2] <= 1920, placed.box
        assert placed.box[0] >= 0, placed.box

    def test_a_label_near_the_left_edge_keeps_its_whole_name_on_screen(self):
        (placed,) = proof.place_labels([_label("Copper", 5, 500)], 1920, 1080)

        assert placed.box[0] >= 0, placed.box
        assert placed.box[2] <= 1920, placed.box

    def test_the_text_moves_with_the_pill_it_sits_in(self):
        """Sliding the box and leaving the text behind would draw the name
        outside its own fill -- worse than the clipping it replaced."""
        (placed,) = proof.place_labels([_label("Copper", 1890, 500)], 1920, 1080)

        assert placed.box[0] <= placed.text_xy[0] <= placed.box[2]
        assert placed.box[1] <= placed.text_xy[1] <= placed.box[3]

    def test_a_name_too_wide_for_the_frame_is_left_out_rather_than_cut(self):
        wide = "Chattahoochee-Oconee National Forest Ridge Crossing"

        assert proof.place_labels([_label(wide, 100, 500)], 320, 180) == []

    def test_a_slid_pill_still_loses_to_one_already_placed(self):
        """Sliding happens before the collision checks, not after: a pill that
        slides onto an earlier one must still be dropped."""
        placed = proof.place_labels(
            [_label("Rocky Bald", 1700, 500), _label("Copper", 1890, 500)], 1920, 1080
        )

        assert [p.text for p in placed] == ["Rocky Bald"]

    def test_every_pill_in_a_crowded_edge_frame_stays_inside_the_frame(self):
        labels = [_label(f"Ridge {i}", 1870 + i, 80 + i * 60) for i in range(12)]

        placed = proof.place_labels(labels, 1920, 1080)

        assert placed, "labels at the edge are slid into view, not all dropped"
        for pill in placed:
            assert 0 <= pill.box[0] and pill.box[2] <= 1920, pill.box


class TestExtraKeepClearZonesFromTheHud:
    """The dashboard rail and elevation profile are HUD chrome, not part of
    the collision logic's own vocabulary -- a caller (``scene_renderer``)
    resolves their real ``hud_layout()`` boxes once per frame size and hands
    them in as ``extra_keep_clear`` so a label pill never lands on either.
    """

    WIDTH, HEIGHT = 1920, 1080

    def test_a_pill_that_would_sit_on_the_dashboard_rail_is_moved_off_it(self):
        layout = hud_layout(self.WIDTH, self.HEIGHT)
        dashboard_box = (
            layout.dashboard_left,
            layout.dashboard_panel_top,
            layout.dashboard_right,
            layout.dashboard_panel_top + layout.dashboard_panel_height,
        )
        label = [_label("Test Peak", 960, 980)]

        # Without the keep-clear zone the default pill lands on the rail --
        # this is what makes the assertion below a real regression guard.
        (default,) = proof.place_labels(label, self.WIDTH, self.HEIGHT)
        assert _overlaps(default.box, dashboard_box)

        placed = proof.place_labels(label, self.WIDTH, self.HEIGHT, (dashboard_box,))

        assert all(not _overlaps(pill.box, dashboard_box) for pill in placed)

    def test_a_pill_that_would_sit_on_the_elevation_profile_is_moved_off_it(self):
        layout = hud_layout(self.WIDTH, self.HEIGHT)
        profile_box = (
            layout.profile_left,
            layout.profile_top,
            layout.profile_right,
            layout.profile_top + layout.profile_height,
        )
        label = [_label("Ridge View", 960, 900)]

        (default,) = proof.place_labels(label, self.WIDTH, self.HEIGHT)
        assert _overlaps(default.box, profile_box)

        placed = proof.place_labels(label, self.WIDTH, self.HEIGHT, (profile_box,))

        assert all(not _overlaps(pill.box, profile_box) for pill in placed)

    def test_extra_keep_clear_defaults_to_empty_and_never_required(self):
        """Existing callers pass only three positional args; the new parameter
        must not become mandatory."""
        placed = proof.place_labels([_label("Black Bald", 600, 230)], self.WIDTH, self.HEIGHT)

        assert [p.text for p in placed] == ["Black Bald"]


def test_hidden_and_nameless_labels_are_never_placed():
    placed = proof.place_labels(
        [
            {"name": "Hidden", "x": 10, "y": 50, "visible": False},
            {"name": "", "x": 300, "y": 300, "visible": True},
            _label("Shown", 600, 400),
        ],
        1280,
        720,
    )

    assert [p.text for p in placed] == ["Shown"]


def test_drawing_only_touches_the_placed_pills():
    frame = np.full((720, 1280, 3), 90, dtype=np.uint8)
    labels = [_label("Horse Cove Branch", 160, 90), _label("Bridge Gap", 130, 92)]

    drawn = proof._draw_labels(frame, labels)

    (kept,) = proof.place_labels(labels, 1280, 720)
    x0, y0, x1, y1 = (int(v) for v in kept.box)
    outside = np.ones(frame.shape[:2], dtype=bool)
    outside[max(0, y0 - 1) : y1 + 2, max(0, x0 - 1) : x1 + 2] = False
    assert (drawn[outside] == 90).all(), "the losing pill left no pixels behind"
    assert (drawn[~outside] != 90).any()
