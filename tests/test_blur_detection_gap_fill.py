"""Frames between detections must still carry a mask.

``detect_and_track`` runs at ``blur_sample_fps=8`` against 59.94 fps video and
emits entries only for frames it sampled or flow-tracked, leaving ~7 frames
between samples with NO entry — and the renderer treats a missing entry as
"nothing to blur". Wherever an object first appears between two samples, the
leading frames ship unmasked.

This is deliberately NOT the cabin-gate failure case. The frame in that
case sat in a 531-frame void, far past any sane ``max_gap_frames``,
and filling it would have blurred nine seconds of open road;
``backfill_track_births`` is what closes that case. The two are complementary
and the call sites chain them — gap fill first, so only births that survive
interpolation get their pre-roll extended.

``blur_pre_roll_frames`` does not close the inter-sample gap either: it
extrapolates a *confirmed, moving* track backward and shrinks the box as it
goes, so it never emits an entry for a frame the tracker skipped entirely.

Re-running with --fresh cannot help — same models, same sample_fps, same
deterministic output.
"""

import pytest

from gomoveshift.pipeline.blur_gate import fill_detection_gaps


def test_gap_frame_inherits_union_of_neighbours():
    """An inter-sample gap: frame 94620 has no entry, 94621 covers the object."""
    frame_boxes = {
        94613: [[1800, 1320, 180, 100]],
        94621: [[1829, 1325, 192, 105]],
    }
    filled = fill_detection_gaps(frame_boxes, max_gap_frames=9)

    assert 94620 in filled, "the leaking frame still has no mask entry"
    # Union of both neighbours: an appearing object and a leaving one are both
    # covered, because we cannot tell which a gap frame holds.
    assert [1800, 1320, 180, 100] in filled[94620]
    assert [1829, 1325, 192, 105] in filled[94620]


def test_every_frame_in_the_gap_is_filled():
    frame_boxes = {10: [[0, 0, 10, 10]], 18: [[5, 5, 10, 10]]}
    filled = fill_detection_gaps(frame_boxes, max_gap_frames=9)
    for f in range(10, 19):
        assert filled.get(f), f"frame {f} left unmasked"


def test_detected_frames_are_never_modified():
    frame_boxes = {10: [[0, 0, 10, 10]], 18: [[5, 5, 10, 10]]}
    filled = fill_detection_gaps(frame_boxes, max_gap_frames=9)
    assert filled[10] == [[0, 0, 10, 10]]
    assert filled[18] == [[5, 5, 10, 10]]


def test_long_gaps_are_left_alone():
    """Past the sample interval the bracketing boxes say nothing useful.

    Filling a 10-minute gap between two unrelated cars would blur bare road.
    """
    frame_boxes = {0: [[0, 0, 10, 10]], 5000: [[5, 5, 10, 10]]}
    filled = fill_detection_gaps(frame_boxes, max_gap_frames=9)
    assert set(filled) == {0, 5000}


def test_empty_entries_count_as_gaps():
    """An explicit empty list is 'nothing detected here', same as absent."""
    frame_boxes = {10: [[0, 0, 10, 10]], 14: [], 18: [[5, 5, 10, 10]]}
    filled = fill_detection_gaps(frame_boxes, max_gap_frames=9)
    assert filled[14], "empty entry was treated as a real 'no objects' verdict"


def test_string_keys_are_accepted():
    """The sidecar round-trips through JSON, which stringifies int keys."""
    filled = fill_detection_gaps({"10": [[0, 0, 10, 10]], "18": [[5, 5, 10, 10]]}, max_gap_frames=9)
    assert filled[14]


def test_no_duplicate_boxes_when_neighbours_agree():
    frame_boxes = {10: [[0, 0, 10, 10]], 18: [[0, 0, 10, 10]]}
    filled = fill_detection_gaps(frame_boxes, max_gap_frames=9)
    assert filled[14] == [[0, 0, 10, 10]]


@pytest.mark.parametrize("boxes", [{}, {5: [[0, 0, 1, 1]]}])
def test_degenerate_inputs_are_returned_unchanged(boxes):
    assert fill_detection_gaps(boxes, max_gap_frames=9) == {int(k): v for k, v in boxes.items()}


def test_gap_fill_then_birth_backfill_covers_both_shapes():
    """The call-site chain must close inter-sample gaps AND track births.

    Locks in the composition order of the two fixes.
    Frame 40 is a birth after a long void; 10->18 is an ordinary sample gap.
    """
    from gomoveshift.pipeline.blur_gate import backfill_track_births

    near, far = [0, 0, 10, 10], [500, 500, 20, 20]
    frame_boxes = {10: [near], 18: [near], 40: [far]}

    filled = backfill_track_births(
        fill_detection_gaps(frame_boxes, max_gap_frames=9),
        pre_roll_frames=5,
    )

    assert filled[14] == [near], "inter-sample gap left unmasked"
    assert far in filled[38], "track birth left unmasked"
    assert 25 not in filled, "the 22-frame void must stay empty"
