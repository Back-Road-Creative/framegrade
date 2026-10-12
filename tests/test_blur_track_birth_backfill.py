"""A track's first detected frame is not its first visible frame.

A long run can fail the cabin gate on the frame just before a vehicle's first
detection. The detection data shows why:

    94086-94089  boxes present, 0% cabin coverage  (unrelated objects)
    94090-94620  nothing at all — a 531-frame void
    94621        the oncoming cabin, covered at 76.4%

94621 is the track's BIRTH — the first frame that vehicle was ever detected.
An object is visible for a few frames before the detector first scores it, so
the frames immediately before a birth are exactly the ones that ship
unmasked.

``fill_detection_gaps`` cannot help: it brackets a gap between two detections
of the same scene, and a birth has no earlier box to bracket against. Filling
the 531-frame void would have blurred nine seconds of open road.

``blur_pre_roll_frames`` is meant to cover this but is a ``detect_and_track``
argument, so tuning it costs a full re-detection pass (hours on a long source) — and at
its default of 5 it produces no entry for the void frames at all. This backfill runs on
the already-computed sidecar instead, so it costs a mask re-render.
"""

import pytest

from gomoveshift.pipeline.blur_gate import backfill_track_births


def test_birth_is_backfilled_into_preceding_frames():
    """The long-run shape: a long void, then a box appears from nowhere."""
    frame_boxes = {94089: [[10, 10, 20, 20]], 94621: [[1829, 1325, 192, 105]]}
    filled = backfill_track_births(frame_boxes, pre_roll_frames=5)

    for f in range(94616, 94621):
        assert filled.get(f), f"frame {f} left unmasked before the birth"
        assert [1829, 1325, 192, 105] in filled[f]


def test_backfill_stops_at_pre_roll_depth():
    frame_boxes = {100: [[0, 0, 10, 10]]}
    filled = backfill_track_births(frame_boxes, pre_roll_frames=5)
    assert filled.get(95)
    assert 94 not in filled, "backfilled deeper than pre_roll_frames"


def test_continuing_track_is_not_a_birth():
    """A box that overlaps the previous populated frame is already tracked.

    Frame 100 IS a birth (nothing precedes it), so the discriminating check is
    the CONTENT of the backfill: if 101 were also treated as a birth, its
    slightly-shifted box would appear alongside the real one.
    """
    frame_boxes = {100: [[0, 0, 100, 100]], 101: [[2, 2, 100, 100]]}
    filled = backfill_track_births(frame_boxes, pre_roll_frames=5)
    assert filled[99] == [[0, 0, 100, 100]], "a continuing box was backfilled as a birth"


def test_only_the_new_box_is_backfilled():
    """One object continues, another is born — backfill just the newcomer.

    Frame 5 anchors the continuing track so frame 10 is not the first
    populated frame, which would make it a birth in its own right.
    """
    frame_boxes = {
        5: [[0, 0, 100, 100]],
        10: [[1, 1, 100, 100]],
        11: [[2, 2, 100, 100], [900, 900, 50, 50]],
    }
    filled = backfill_track_births(frame_boxes, pre_roll_frames=3)
    assert filled[8] == [[900, 900, 50, 50]], "backfilled the continuing box too"


def test_existing_entries_are_preserved_not_replaced():
    frame_boxes = {9: [[0, 0, 10, 10]], 10: [[500, 500, 40, 40]]}
    filled = backfill_track_births(frame_boxes, pre_roll_frames=3)
    assert [0, 0, 10, 10] in filled[9], "clobbered a real detection"
    assert [500, 500, 40, 40] in filled[9], "did not backfill the birth"


def test_never_backfills_before_frame_zero():
    filled = backfill_track_births({2: [[0, 0, 10, 10]]}, pre_roll_frames=5)
    assert min(filled) >= 0


def test_zero_pre_roll_is_a_no_op():
    frame_boxes = {100: [[0, 0, 10, 10]]}
    assert backfill_track_births(frame_boxes, pre_roll_frames=0) == frame_boxes


def test_string_keys_are_accepted():
    filled = backfill_track_births({"100": [[0, 0, 10, 10]]}, pre_roll_frames=2)
    assert filled[99] == [[0, 0, 10, 10]]


@pytest.mark.parametrize("boxes", [{}, {5: []}])
def test_degenerate_inputs(boxes):
    out = backfill_track_births(boxes, pre_roll_frames=5)
    assert out == {int(k): v for k, v in boxes.items()}
