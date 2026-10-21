"""A detector miss must not unmask an object that is still in frame."""

from gomoveshift.pipeline import blur_gate
from gomoveshift.pipeline.blur_gate import (
    PLATE_BAND_FRAC,
    TRAILING_HOLD_FRAMES,
    hold_missed_detections,
    plate_band_region,
)

INTERVAL = 7


def _covers(boxes, target):
    tx, ty, tw, th = target
    return any(
        b[0] <= tx and b[1] <= ty and b[0] + b[2] >= tx + tw and b[1] + b[3] >= ty + th
        for b in boxes
    )


def test_object_missed_for_two_samples_is_interpolated():
    lo, hi = 100, 100 + 3 * INTERVAL  # two samples missed between
    out = hold_missed_detections(
        {lo: [[100, 200, 80, 60]], hi: [[121, 200, 80, 60]]}, sample_interval_frames=INTERVAL
    )
    for f in range(lo + 1, hi):
        assert out[f], f"frame {f} left unmasked"
    mid = out[lo + 10][0]
    assert 100 < mid[0] < 121


def test_unrelated_boxes_across_long_void_not_filled():
    """Ireland-style 531-frame void: unrelated boxes must not be unioned across it."""
    out = hold_missed_detections(
        {100: [[10, 10, 50, 50]], 631: [[1800, 900, 50, 50]]}, sample_interval_frames=INTERVAL
    )
    assert not out.get(300)
    assert not out.get(630)


def test_unrelated_boxes_in_short_gap_not_interpolated():
    out = hold_missed_detections(
        {100: [[10, 10, 50, 50]], 114: [[1800, 900, 50, 50]]}, sample_interval_frames=INTERVAL
    )
    assert [10, 10, 50, 50] in out[101]
    assert all(b != [1800, 900, 50, 50] for b in out[107])


def test_trailing_hold_is_bounded():
    out = hold_missed_detections({100: [[10, 10, 50, 50]]}, sample_interval_frames=INTERVAL)
    assert out[100 + TRAILING_HOLD_FRAMES]
    assert not out.get(100 + TRAILING_HOLD_FRAMES + 1)


def test_trailing_hold_before_distant_next_detection_is_bounded():
    out = hold_missed_detections(
        {100: [[10, 10, 50, 50]], 400: [[900, 500, 50, 50]]}, sample_interval_frames=INTERVAL
    )
    assert out[101] and not out.get(100 + TRAILING_HOLD_FRAMES + 1)


def test_plate_band_covers_bumper_below_cabin():
    body = (100, 100, 200, 100)
    x0, y0, x1, y1 = plate_band_region(body, 1920, 1080)
    assert y1 == 200 and x0 == 100 and x1 == 300
    assert y0 == 200 - round(100 * PLATE_BAND_FRAC)
    assert _covers([[x0, y0, x1 - x0, y1 - y0]], (140, 185, 22, 8))


def test_plate_band_clamped_to_frame():
    assert plate_band_region((-10, 1000, 100, 200), 1920, 1080)[3] == 1080
    assert blur_gate.MISSED_SAMPLE_HOLD >= 2


# Policy doc 158 section 9: plates stay in scope whatever their size. No floor.
def test_small_plate_15x6_at_1080p_is_held_and_interpolated():
    """A 15x6 px plate (1920x1080 frame) missed for two samples stays masked."""
    lo, hi = 50, 50 + 3 * INTERVAL
    out = hold_missed_detections(
        {lo: [[900, 700, 15, 6]], hi: [[903, 700, 15, 6]]}, sample_interval_frames=INTERVAL
    )
    for f in range(lo + 1, hi):
        assert out[f], f"small plate unmasked at frame {f}"
        assert out[f][0][2:] == [15, 6]


def test_moving_plate_box_translates_across_missed_samples():
    lo, hi = 50, 50 + 3 * INTERVAL
    out = hold_missed_detections(
        {lo: [[900, 700, 40, 16]], hi: [[921, 700, 40, 16]]}, sample_interval_frames=INTERVAL
    )
    xs = [out[f][0][0] for f in range(lo + 1, hi)]
    assert xs == sorted(xs) and xs[0] > 900 and xs[-1] < 921


def test_small_plate_band_region_is_not_dropped():
    x0, y0, x1, y1 = plate_band_region((1000, 600, 15, 6), 1920, 1080)
    assert x1 > x0 and y1 > y0


def test_small_fast_plate_with_no_overlap_is_still_interpolated():
    """A 15x6 plate that moves 30 px across a 3-interval gap never overlaps itself
    (IoU 0). It must still be tracked through the gap, not dropped after the
    trailing hold, or every frame past the hold leaks the plate."""
    lo, hi = 50, 50 + 3 * INTERVAL
    out = hold_missed_detections(
        {lo: [[900, 700, 15, 6]], hi: [[930, 700, 15, 6]]}, sample_interval_frames=INTERVAL
    )
    for f in range(lo + 1, hi):
        true_x = 900 + round(30 * (f - lo) / (hi - lo))
        assert _covers(out[f], (true_x, 700, 15, 6)), f"plate leaks at frame {f}"


def test_no_overlap_match_rejects_a_box_of_very_different_size():
    lo, hi = 50, 50 + 3 * INTERVAL
    out = hold_missed_detections(
        {lo: [[900, 700, 15, 6]], hi: [[930, 700, 200, 120]]}, sample_interval_frames=INTERVAL
    )
    assert all(b[2] in (15, 200) for f in range(lo + 1, hi) for b in out.get(f, []))


def test_every_mask_builder_holds_misses_before_gap_fill():
    """Wiring guard: each of the three mask builders runs the per-object hold, and
    runs it before ``fill_detection_gaps`` (which would otherwise union unrelated
    boxes first). A structural check: it fails if a call site drops the hold."""
    import inspect

    from gomoveshift.pipeline import day_master_encoder
    from gomoveshift.pipeline.photos import night_private_video
    from gomoveshift.pipeline.stages import blur_detection

    for fn in (
        blur_detection.BlurDetectionStage,
        day_master_encoder._build_day_blur_mask,
        night_private_video._render_private_video,
    ):
        src = inspect.getsource(fn)
        assert "hold_missed_detections(" in src, fn.__qualname__
        assert src.index("hold_missed_detections(") < src.index("fill_detection_gaps("), (
            fn.__qualname__
        )
