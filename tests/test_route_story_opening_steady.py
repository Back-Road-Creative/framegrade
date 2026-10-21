"""The opening's label steadiness and raster hand-off smoothness.

A viewer watching a render (~6s in) saw the town-name label set change four times in
under two seconds with one-frame pops, and the whole map flipped in a single
frame from vivid green NASA GIBS to grey-blue USDA NAIP imagery -- reading as
a different layer of the map, not the same one continuing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gomoveshift.pipeline.route_story.layers import (
    MapLabel,
    _opening_label_windows,
    _opening_visible_labels,
    _render_opening_map,
)
from gomoveshift.pipeline.route_story.metrics import RoutePoint
from gomoveshift.pipeline.route_story.satellite import SatelliteRaster

TARGET = RoutePoint(time_s=0.0, latitude=35.23, longitude=-83.56, elevation_m=900.0)
WIDTH, HEIGHT = 640, 360


def _raster(tmp_path: Path, name: str, bounds, credit: str, width: int, colour) -> SatelliteRaster:
    path = tmp_path / f"{name}.jpg"
    height = max(1, width // 2)
    Image.new("RGB", (width, height), colour).save(path, quality=95)
    west, south, east, north = bounds
    return SatelliteRaster(path, west, south, east, north, credit)


@pytest.fixture(scope="module")
def dive_rasters(tmp_path_factory: pytest.TempPathFactory) -> tuple[SatelliteRaster, ...]:
    """A coarse green band and a finer, differently toned band that fully
    covers the target's neighbourhood -- mirrors the real GIBS-to-NAIP
    hand-off (green -> grey-blue) closely enough to exercise the cross-fade
    and tone match without a multi-megapixel fixture.

    Shared read-only per module: grading each 4096-px band is ~55 s, and the
    graded/tone-matched images are cached by path, so one pair means one grade
    per worker instead of one per test."""
    tmp_path = tmp_path_factory.mktemp("dive-rasters")
    coarse = _raster(
        tmp_path, "coarse", (-101.25, 21.94, -67.5, 48.92), "Coarse band", 4096, (40, 140, 60)
    )
    fine = _raster(tmp_path, "fine", (-83.9, 35.0, -83.2, 35.5), "Fine band", 4096, (150, 160, 170))
    return coarse, fine


# --- labels: steady, no flicker ---------------------------------------------


def _labels() -> tuple[MapLabel, ...]:
    # Coordinates chosen to sit inside the opening camera's viewport only
    # over different parts of the move, so the plan actually has to decide
    # who gets a slot and when -- like the real Cherokee/Bryson City/
    # Clingmans Dome/Tellico Gap set that popped four times in ~2s.
    return (
        MapLabel("Cherokee", 35.47, -83.83, kind="town"),
        MapLabel("Bryson City", 35.43, -83.45, kind="town"),
        MapLabel("Franklin", 35.18, -83.38, kind="town"),
        MapLabel("Clingmans Dome", 35.56, -83.5, kind="poi", max_span_lon=0.6, min_span_lon=0.05),
        MapLabel("Tellico Gap", 35.3, -83.6, kind="poi", max_span_lon=0.3),
    )


def test_no_label_is_shown_for_fewer_than_the_dwell_frames(dive_rasters):
    rasters = dive_rasters
    labels = _labels()

    windows = _opening_label_windows(rasters, TARGET, WIDTH, HEIGHT, labels)

    from gomoveshift.pipeline.route_story.layers import _OPENING_LABEL_DWELL_FRACTION

    assert windows  # the fixture's labels do qualify for some part of the move
    for _label, start, end in windows:
        assert end - start >= _OPENING_LABEL_DWELL_FRACTION - 1e-9


def test_a_labels_visibility_never_flaps_on_off_on(dive_rasters):
    # The defect viewers saw: "Bryson City, Andrews" -> "+ Clingmans Dome" ->
    # a DIFFERENT set, back and forth several times in under two seconds.
    # Once a label's plan is fixed for the whole move, each label may appear
    # at most once: a single contiguous on-screen run, never on, off, then
    # on again.
    rasters = dive_rasters
    labels = _labels()

    for label in labels:
        was_visible = False
        transitions = 0
        for index in range(361):
            fraction = index / 360
            visible = _opening_visible_labels(rasters, TARGET, fraction, WIDTH, HEIGHT, labels)
            names = {candidate.name for candidate, alpha in visible if alpha > 0.0}
            is_visible = label.name in names
            if is_visible != was_visible:
                transitions += 1
            was_visible = is_visible
        assert transitions <= 2, f"{label.name} appeared/disappeared {transitions} times"


def test_a_labels_alpha_is_continuous_across_its_own_window(dive_rasters):
    rasters = dive_rasters
    labels = _labels()

    windows = _opening_label_windows(rasters, TARGET, WIDTH, HEIGHT, labels)
    assert windows
    label, start, end = windows[0]

    # Sampled at real-frame density (a 12s@30fps intro is 360 frames), finer
    # than the plan's own 241-point grid -- alpha is a continuous function of
    # the exact fraction queried, not quantised to the plan's grid, so a
    # dense-enough sample sees a smooth ramp rather than the plan's own
    # boundary snapping to the nearest grid point.
    previous_alpha = None
    steps = 359
    for i in range(steps + 1):
        fraction = start + (end - start) * i / steps
        visible = dict(
            (candidate.name, alpha)
            for candidate, alpha in _opening_visible_labels(
                rasters, TARGET, fraction, WIDTH, HEIGHT, labels
            )
        )
        alpha = visible.get(label.name, 0.0)
        if previous_alpha is not None:
            assert abs(alpha - previous_alpha) < 0.25, (
                f"{label.name} alpha jumped from {previous_alpha} to {alpha} "
                f"at fraction {fraction:.4f}"
            )
        previous_alpha = alpha


def test_a_label_already_on_screen_is_never_evicted_by_a_higher_ranked_newcomer(dive_rasters):
    rasters = dive_rasters
    # Early Town qualifies from wide up in the zoom (max_span_lon=5.0) and
    # stays qualified all the way to the end; Late City, despite outranking
    # it (kind="city" > "poi"), only starts qualifying once the span has
    # narrowed past 1.0 -- well after Early Town is already holding the
    # move's one slot.
    early = MapLabel("Early Town", 35.4, -83.6, kind="poi", max_span_lon=5.0)
    late_city = MapLabel("Late City", 35.4, -83.61, kind="city", max_span_lon=1.0)
    labels = (early, late_city)

    # One slot only, so the two labels genuinely compete for it -- with 12
    # (the real default) both fixture labels fit at once and eviction is
    # never exercised at all.
    windows = _opening_label_windows(rasters, TARGET, WIDTH, HEIGHT, labels, max_labels=1)
    windows_by_name = {label.name: (start, end) for label, start, end in windows}
    assert "Early Town" in windows_by_name
    # If the higher-priority "city" could evict Early Town mid-run, Early
    # Town's assigned window and Late City's would overlap in fraction
    # space. Sticky occupancy means they never do -- whichever holds the
    # slot keeps it until its own candidacy ends, and the other waits.
    if "Late City" in windows_by_name:
        early_start, early_end = windows_by_name["Early Town"]
        late_start, late_end = windows_by_name["Late City"]
        assert early_end <= late_start or late_end <= early_start


def test_rendering_the_opening_with_labels_does_not_raise(dive_rasters):
    rasters = dive_rasters
    labels = _labels()

    for fraction in (0.0, 0.3, 0.6, 0.9, 1.0):
        image = _render_opening_map(rasters, TARGET, fraction, WIDTH, HEIGHT, labels)
        assert image.size == (WIDTH, HEIGHT)


# --- raster hand-off: no one-frame flip, no clashing tone -------------------


def _mean_rgb(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float64).reshape(-1, 3).mean(axis=0)


def test_the_mean_colour_step_between_consecutive_opening_frames_is_capped(dive_rasters):
    rasters = dive_rasters

    previous_mean = None
    max_step = 0.0
    for index in range(121):
        fraction = index / 120
        image = _render_opening_map(rasters, TARGET, fraction, WIDTH, HEIGHT)
        mean = _mean_rgb(image)
        if previous_mean is not None:
            step = float(np.abs(mean - previous_mean).max())
            max_step = max(max_step, step)
        previous_mean = mean

    # A hard, one-frame cut between the fixture's very differently toned
    # bands would step by ~100+ on at least one channel; the cross-fade caps
    # any single adjacent-frame step well below that.
    assert max_step < 40.0, f"single-frame colour step of {max_step:.1f} looks like a hard cut"


def test_a_finer_raster_is_tone_matched_toward_its_coarser_neighbour(dive_rasters):
    from gomoveshift.pipeline.route_story.layers import _tone_matched_opening_image

    rasters = dive_rasters
    coarse, fine = rasters

    matched = _tone_matched_opening_image(rasters, fine)
    from gomoveshift.pipeline.route_story.layers import _graded_raster_image

    matched_mean = np.asarray(matched, dtype=np.float64).reshape(-1, 3).mean(axis=0)
    fine_mean = (
        np.asarray(_graded_raster_image(fine.source), dtype=np.float64).reshape(-1, 3).mean(axis=0)
    )
    coarse_mean = (
        np.asarray(_graded_raster_image(coarse.source), dtype=np.float64)
        .reshape(-1, 3)
        .mean(axis=0)
    )

    # Tone-matched should land closer to the coarser neighbour's overall
    # colour than the raw graded (un-matched) finer raster did.
    assert np.abs(matched_mean - coarse_mean).sum() < np.abs(fine_mean - coarse_mean).sum()


def test_tone_matching_keeps_the_finer_rasters_own_detail(tmp_path):
    # A preflight refused the opening: matching the finer raster's
    # CONTRAST to its blurrier coarse neighbour flattened its detail, and the
    # ladder rung on that raster dropped from 0.54 to 0.29 of the median.
    # The finer raster exists to be sharper; only its colour may move.
    from gomoveshift.pipeline.route_story.layers import (
        _graded_raster_image,
        _tone_matched_opening_image,
    )

    # Flat (no detail at all) and a colour shift small enough never to clip,
    # so only a contrast change can move the spread.
    coarse = _raster(
        tmp_path, "coarse", (-101.25, 21.94, -67.5, 48.92), "Coarse band", 1024, (120, 150, 130)
    )
    fine_path = tmp_path / "textured.jpg"
    rng = np.random.default_rng(7)
    texture = rng.integers(90, 210, size=(512, 1024, 3), dtype=np.uint8)
    Image.fromarray(texture, "RGB").save(fine_path, quality=95)
    fine = SatelliteRaster(fine_path, -83.9, 35.0, -83.2, 35.5, "Fine band")

    matched = np.asarray(_tone_matched_opening_image((coarse, fine), fine), dtype=np.float64)
    graded = np.asarray(_graded_raster_image(fine.source), dtype=np.float64)

    matched_std = matched.reshape(-1, 3).std(axis=0)
    graded_std = graded.reshape(-1, 3).std(axis=0)
    assert np.all(matched_std >= 0.9 * graded_std), (matched_std, graded_std)
