"""Synthetic-frame tests for astro_triage: no RAW fixtures, deterministic RNG.

Every frame is a hand-built float32 linear luminance array (0..1) — exactly
what ``astro_calibrate`` hands this module.
"""

from datetime import datetime, timedelta
from pathlib import Path

import cv2
import gomoveshift.pipeline.photos.astro_triage as triage_mod
import numpy as np
import pytest

from gomoveshift.pipeline.photos.astro_triage import (
    GAUSSIAN_SIGMA_TO_FWHM,
    FrameMetrics,
    StarShape,
    classify_exposure,
    classify_shape,
    classify_streaks,
    detect_stars,
    detect_streak_components,
    detect_streaks,
    measure_frame,
    triage,
    update_run_cloud_baseline,
)


def _place_stars(shape, positions, sigma_px=1.6, flux=0.6, background=0.05, noise=0.0, seed=0):
    h, w = shape
    luma = np.clip(np.random.default_rng(seed).normal(background, noise, size=shape), 0.0, None)
    yy, xx = (a.astype(np.float32) for a in np.mgrid[0:h, 0:w])
    for cx, cy in positions:
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        luma += flux * np.exp(-r2 / (2.0 * sigma_px**2))
    return np.clip(luma, 0.0, 1.0).astype(np.float32)


def _place_elongated_star(shape, center, sigma_major, sigma_minor, angle_deg, noise=0.0, seed=0):
    """A rotated anisotropic Gaussian star, unlike ``_place_stars``'s isotropic ones."""
    h, w = shape
    cx, cy = center
    theta = np.radians(angle_deg)
    luma = np.clip(np.random.default_rng(seed).normal(0.05, noise, size=shape), 0.0, None)
    yy, xx = (a.astype(np.float64) for a in np.mgrid[0:h, 0:w])
    dx, dy = xx - cx, yy - cy
    u = dx * np.cos(theta) + dy * np.sin(theta)
    v = -dx * np.sin(theta) + dy * np.cos(theta)
    luma += 0.6 * np.exp(-(u**2) / (2.0 * sigma_major**2) - (v**2) / (2.0 * sigma_minor**2))
    return np.clip(luma, 0.0, 1.0).astype(np.float32)


def _orientation_diff(a_deg, b_deg):
    """Smallest angle between two mod-180 axis orientations."""
    diff = abs(a_deg - b_deg) % 180.0
    return min(diff, 180.0 - diff)


#: Isolated 160x160 positions, spaced so no PSF wing crosses a neighbour's window.
STAR_POSITIONS = [(30.0, 40.0), (90.0, 30.0), (120.0, 100.0), (55.0, 130.0), (140.0, 140.0)]


class TestDetectStars:
    def test_it_recovers_known_stars_with_subpixel_centroids(self):
        luma = _place_stars((160, 160), STAR_POSITIONS, noise=0.004, seed=1)
        stars = detect_stars(luma)
        assert stars.dtype == np.float32
        assert stars.shape == (len(STAR_POSITIONS), 3)
        for cx, cy in STAR_POSITIONS:
            dist = np.hypot(stars[:, 0] - cx, stars[:, 1] - cy)
            assert dist.min() < 0.3, f"no sub-pixel match near ({cx}, {cy})"

    def test_a_noise_only_frame_yields_no_detections(self):
        rng = np.random.default_rng(2)
        luma = np.clip(rng.normal(0.05, 0.01, size=(160, 160)), 0.0, 1.0).astype(np.float32)
        assert len(detect_stars(luma)) == 0

    def test_a_masked_region_contributes_no_detections(self):
        luma = _place_stars((160, 160), STAR_POSITIONS, noise=0.004, seed=3)
        mask = np.ones(luma.shape, dtype=bool)
        mask[:, :80] = False  # blanks out the two leftmost stars (x=30, x=55)
        stars = detect_stars(luma, mask=mask)
        assert stars.shape[0] == 3
        assert np.all(stars[:, 0] >= 80)


class TestMeasureFrame:
    def test_it_recovers_known_star_count_and_expected_fwhm(self, tmp_path):
        sigma_px = 1.8
        luma = _place_stars((160, 160), STAR_POSITIONS, sigma_px=sigma_px, noise=0.002, seed=4)
        metrics = measure_frame(tmp_path / "star.jpg", luma)
        assert isinstance(metrics, FrameMetrics)
        assert metrics.star_count == len(STAR_POSITIONS)
        assert metrics.usable is True
        assert metrics.fwhm_px == pytest.approx(sigma_px * GAUSSIAN_SIGMA_TO_FWHM, rel=0.25)

    def test_a_starless_frame_is_unusable(self, tmp_path):
        luma = np.full((160, 160), 0.05, dtype=np.float32)
        metrics = measure_frame(tmp_path / "empty.jpg", luma)
        assert metrics.star_count == 0
        assert metrics.usable is False

    def test_mask_reaches_both_detectors_not_just_stars(self, tmp_path):
        """A mask handed to ``measure_frame`` must exclude a ground light
        from ``star_count`` AND a ground-level streak from ``streaks`` -- both
        detectors, not just ``detect_stars``."""
        h, w = 200, 200
        luma = _place_stars((h, w), STAR_POSITIONS, noise=0.002, seed=5)
        luma[180, 40] = 1.0  # a bright "ground light" well below the horizon
        cv2.line(luma, (20, 150), (170, 165), 0.9, 2)  # a ground-level streak

        unmasked = measure_frame(tmp_path / "f.jpg", luma)
        mask = np.ones((h, w), dtype=bool)
        mask[145:, :] = False  # sky only -- clears every STAR_POSITIONS star (max y=140)
        masked = measure_frame(tmp_path / "f.jpg", luma, mask=mask)

        assert masked.star_count == len(STAR_POSITIONS)
        assert unmasked.star_count > len(STAR_POSITIONS)
        assert masked.streaks == []
        assert unmasked.streaks

    def test_captured_at_threads_through_to_frame_metrics(self, tmp_path):
        luma = np.full((160, 160), 0.05, dtype=np.float32)
        when = datetime(2026, 8, 1, 22, 0, 0)
        metrics = measure_frame(tmp_path / "empty.jpg", luma, captured_at=when)
        assert metrics.captured_at == when


class TestDetectStreaks:
    def test_an_injected_line_segment_is_detected(self):
        luma = np.full((200, 200), 0.05, dtype=np.float32)
        cv2.line(luma, (20, 30), (170, 150), 0.9, 2)
        streaks = detect_streaks(luma, min_length_px=40.0)
        assert streaks
        endpoints = [(streaks[0][0], streaks[0][1]), (streaks[0][2], streaks[0][3])]
        assert any(abs(x - 20) < 15 and abs(y - 30) < 15 for x, y in endpoints)
        assert any(abs(x - 170) < 15 and abs(y - 150) < 15 for x, y in endpoints)

    def test_a_masked_region_contributes_no_streak_detections(self):
        """A car-headlight-trail-style ground streak: detected unmasked, excluded
        once its whole row range is masked out (mirrors ``detect_stars``'s contract)."""
        luma = np.full((200, 200), 0.05, dtype=np.float32)
        cv2.line(luma, (20, 150), (170, 165), 0.9, 2)  # entirely below row 100
        assert detect_streaks(luma, min_length_px=40.0)  # unmasked: detected

        mask = np.ones(luma.shape, dtype=bool)
        mask[100:, :] = False  # blanks out the whole streak's row range
        assert detect_streaks(luma, min_length_px=40.0, mask=mask) == []


class TestStreakDetectionQuality:
    """Streak quality: without morphology/dedup, raw HoughLinesP strings a single trail into
    10-25 near-duplicate segments, and non-contiguous hot pixels into fake lines; and
    the 8-bit background quantization at a dark sky can hide a real faint trail.
    """

    def test_a_single_noisy_trail_yields_exactly_one_detection(self):
        # A real (noisy, anti-aliased) trail used to come back as several
        # near-duplicate Hough segments for the one physical feature.
        rng = np.random.default_rng(42)
        luma = np.clip(rng.normal(0.05, 0.004, size=(300, 300)), 0.0, 1.0).astype(np.float32)
        cv2.line(luma, (20, 40), (260, 220), 0.9, 3, lineType=cv2.LINE_AA)
        streaks = detect_streaks(luma, min_length_px=40.0)
        assert len(streaks) == 1

    def test_scattered_noncontiguous_hot_pixels_yield_no_detection(self):
        # Hot/noisy pixels that never touch each other must never be strung
        # into a fake "line" -- a connected-component gate rejects this by
        # construction, not by a late length/shape filter alone.
        rng = np.random.default_rng(7)
        luma = np.clip(rng.normal(0.05, 0.004, size=(300, 300)), 0.0, 1.0).astype(np.float32)
        xs = list(range(20, 260, 2))
        ys = [40 + int(round((x - 20) * (220 - 40) / (260 - 20))) for x in xs]
        for x, y in zip(xs, ys):
            luma[y, x] = 0.9
        assert detect_streaks(luma, min_length_px=40.0) == []

    def test_two_distinct_parallel_trails_stay_two_detections(self):
        rng = np.random.default_rng(11)
        luma = np.clip(rng.normal(0.05, 0.004, size=(300, 300)), 0.0, 1.0).astype(np.float32)
        cv2.line(luma, (20, 50), (260, 50), 0.9, 2, lineType=cv2.LINE_AA)
        cv2.line(luma, (20, 90), (260, 90), 0.9, 2, lineType=cv2.LINE_AA)
        streaks = detect_streaks(luma, min_length_px=40.0)
        assert len(streaks) == 2

    def test_a_faint_trail_on_a_dark_sky_gradient_is_only_detected_with_the_float_fix(self):
        # The old fixed [0, 1] -> uint8 background scale gives a ~0.02 linear
        # sky one code per ~0.0039 -- comparable to this subtle sky-glow gradient
        # across the 31px median window, so the quantized background estimate
        # measurably inflates the recovered noise floor and raises the detection
        # threshold high enough to miss a real faint trail entirely.
        h, w = 300, 300
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        grad = 0.018 + (xx / w) * 0.01  # a subtle light-pollution-style gradient
        rng = np.random.default_rng(1)
        noise = rng.normal(0.0, 0.0003, size=(h, w)).astype(np.float32)
        luma = np.clip(grad + noise, 0.0, 1.0).astype(np.float32)
        cv2.line(luma, (20, 40), (260, 220), 0.023, 2, lineType=cv2.LINE_AA)  # faint trail
        assert detect_streaks(luma, min_length_px=40.0) != []

    def test_ground_mask_still_suppresses_detections_under_the_new_detector(self):
        luma = np.full((200, 200), 0.05, dtype=np.float32)
        cv2.line(luma, (20, 150), (170, 165), 0.9, 2)
        assert detect_streaks(luma, min_length_px=40.0)
        mask = np.ones(luma.shape, dtype=bool)
        mask[100:, :] = False
        assert detect_streaks(luma, min_length_px=40.0, mask=mask) == []

    def test_component_geometry_matches_the_injected_trail_within_tolerance(self):
        luma = np.full((200, 200), 0.05, dtype=np.float32)
        cv2.line(luma, (30, 100), (170, 100), 0.9, 7)  # horizontal, length=140, width=7
        components = detect_streak_components(luma, min_length_px=40.0)
        assert len(components) == 1
        component = components[0]
        assert component.length_px == pytest.approx(140.0, rel=0.15)
        assert component.width_px == pytest.approx(7.0, rel=0.4)
        assert component.x1 == pytest.approx(30.0, abs=10.0)
        assert component.x2 == pytest.approx(170.0, abs=10.0)
        assert component.mean_brightness > 0.0
        assert component.peak_brightness >= component.mean_brightness

    def test_measure_frame_populates_streak_components_alongside_the_tuple_list(self, tmp_path):
        luma = np.full((200, 200), 0.05, dtype=np.float32)
        cv2.line(luma, (20, 30), (170, 150), 0.9, 2)
        metrics = measure_frame(tmp_path / "trail.jpg", luma)
        assert len(metrics.streak_components) == len(metrics.streaks) == 1
        component = metrics.streak_components[0]
        assert (component.x1, component.y1, component.x2, component.y2) == metrics.streaks[0]

    def test_a_bright_region_wider_than_the_median_window_yields_no_detection_unmasked(self):
        # Repair: the robust rescale clips bright pixels, so a bright region wider
        # than the 31px median window used to saturate the whole window and clamp
        # the recovered background to the rescale ceiling instead of the true
        # bright level -- reading as a spurious streak on the DEFAULT unmasked
        # path (cli sets sky_mask=None unless --horizon is passed).
        h, w = 400, 400
        rng = np.random.default_rng(21)
        sky = 0.02 + rng.normal(0.0, 0.0015, size=(h, w)).astype(np.float32)
        luma = np.clip(sky, 0.0, 1.0).astype(np.float32)
        luma[250:300, 20:340] = 0.85  # bright ground band, well wider than the kernel
        assert detect_streaks(luma, min_length_px=40.0) == []

    @staticmethod
    def _blob_field(h, w, step=50, size=3):
        """Isolated ``size``x``size`` blobs -- one surviving component each.

        ``size`` must keep each blob at or above ``STREAK_MIN_COMPONENT_AREA_PX``,
        or every label is discarded by the area floor BEFORE the per-component
        extraction and this fixture measures preprocessing only. A 1x1 field
        (this test's original fixture) reaches the extraction exactly 0 times and
        cannot detect the regression below at all -- verified by counting calls
        into the geometry step, and by restoring the regression and watching the
        old assertion still pass.
        """
        luma = np.full((h, w), 0.05, dtype=np.float32)
        ys = np.arange(20, h - 20, step)
        xs = np.arange(20, w - 20, step)
        for dy in range(size):
            for dx in range(size):
                yy, xx = np.meshgrid(ys + dy, xs + dx, indexing="ij")
                luma[yy.ravel(), xx.ravel()] = 0.9
        return luma

    def test_component_extraction_scans_bounding_boxes_not_the_whole_frame(self, monkeypatch):
        """The per-label loop must read each component's BOUNDING BOX, never rescan the
        whole frame per label.

        The regression this guards -- ``np.nonzero(labels == label)`` inside the loop --
        is O(n_labels * H * W). It is invisible at this suite's 200-300 px fixtures and
        takes minutes on a 24 MP frame with thousands of components.

        Counted, not timed. A wall-clock or CPU-time threshold cannot separate the two
        implementations on a loaded machine: the larger sample is memory-bound, so a
        noisy neighbour costs it disproportionately many cache misses, and those are
        charged to this process's own CPU time as well.

        Counting the elements handed to ``np.nonzero`` asserts the SAME property directly
        and cannot flake: the bounding-box form reads 5,400 elements on the fixture below
        against the frame's 1,500,000, while the regression would read 900,000,000 --
        600x one frame, and ~166,000x what the fixed code reads. One frame's pixel count
        sits between them with room to spare, so the threshold needs no tuning and is
        indifferent to machine speed, load, and every clock.
        """
        field = self._blob_field(1000, 1500)
        frame_px = field.size
        scanned = []
        real_nonzero = np.nonzero

        def counting_nonzero(a):
            scanned.append(np.asarray(a).size)
            return real_nonzero(a)

        # Patched on numpy itself, the module object astro_triage's own ``np`` refers to.
        monkeypatch.setattr(np, "nonzero", counting_nonzero)
        detect_streaks(field, min_length_px=40.0)

        # A fixture that produced no components at all would pass the sum assertion
        # vacuously -- the whole point is that MANY components are extracted cheaply.
        assert len(scanned) > 100, f"fixture yielded only {len(scanned)} extractions"
        assert sum(scanned) < frame_px, (
            f"per-component extraction read {sum(scanned):,} elements across "
            f"{len(scanned)} components -- more than one frame's {frame_px:,} pixels. "
            "The loop is rescanning the full labels array per label instead of slicing "
            "each component's bounding box"
        )


def _fm(name, star_count, background, fwhm, shape=None, exposure_class="light"):
    return FrameMetrics(
        Path(name),
        star_count,
        background,
        fwhm,
        [],
        star_count >= 5,
        shape=shape,
        exposure_class=exposure_class,
    )


def _defocused_shape(fwhm=11.0):
    """A round, oversized PSF -- ``classify_shape``'s ``likely_defocus`` signature
    (pre-refocus frames: ellipticity 0.05-0.08, fwhm_px 10.8-11.5)."""
    return StarShape(
        fwhm_major_px=fwhm,
        fwhm_minor_px=fwhm,
        ellipticity=0.05,
        orientation_deg=0.0,
        hfr_px=fwhm / 2.0,
        saturated=False,
    )


class TestTriage:
    def test_a_star_collapse_and_background_jump_flags_cloud(self):
        rows = [(120, 0.05, 3.0), (10, 0.20, 3.0), (118, 0.05, 3.1)]  # cloud arrives mid-run
        frames = [_fm(f"f{i}.jpg", *r) for i, r in enumerate(rows)]
        report = triage(frames)
        assert report.cloud_frames == (Path("f1.jpg"),)
        assert report.dew_frames == ()

    def test_progressive_fwhm_inflation_flags_dew_but_not_the_baseline(self):
        fwhms = [3.0, 3.1, 2.9, 6.0]  # baseline = first 3; frame 3 is the dew hit
        frames = [_fm(f"f{i}.jpg", 100, 0.05, fwhm) for i, fwhm in enumerate(fwhms)]
        report = triage(frames)
        assert report.dew_frames == (Path("f3.jpg"),)
        assert report.cloud_frames == ()


# A synthetic night: pre-pause median 140 detections at background 0.03,
# post-pause median 24 detections at 0.033.
CARD_CLEAR_STARS, CARD_CLEAR_BG = 140, 0.03
CARD_CLOUD_STARS, CARD_CLOUD_BG = 24, 0.033
CHUNK_FRAMES = 20  # astro_sequence.STACK_LEN_DEFAULT: the group triage actually sees.


def _run(stars, backgrounds=None):
    """FrameMetrics for a run of ``stars`` counts (fixed FWHM: dew stays out of it)."""
    bgs = [CARD_CLEAR_BG] * len(stars) if backgrounds is None else backgrounds
    return [_fm(f"f{i}.jpg", s, b, 3.0) for i, (s, b) in enumerate(zip(stars, bgs))]


def _cloud_indices(frames, **kwargs):
    return [int(p.stem[1:]) for p in triage(frames, **kwargs).cloud_frames]


class TestTriageCloudBaseline:
    def test_a_run_that_clouds_over_and_stays_flags_every_degraded_frame(self):
        # Cloud moves in at frame 3 of a 20-frame chunk and
        # never lifts, so the group's own median degrades with it and nothing reads
        # as anomalous against it. 24/140 = 0.171 of the clear baseline.
        onset = 3
        stars = [CARD_CLEAR_STARS] * onset + [CARD_CLOUD_STARS] * (CHUNK_FRAMES - onset)
        bgs = [CARD_CLEAR_BG] * onset + [CARD_CLOUD_BG] * (CHUNK_FRAMES - onset)
        assert _cloud_indices(_run(stars, bgs)) == list(range(onset, CHUNK_FRAMES))

    @pytest.mark.parametrize("onset", [0, 1, 2, 3, 5])
    def test_the_baseline_needs_two_clear_frames_and_does_not_care_where_they_are(self, onset):
        # Envelope: the baseline is the median of the 3 clearest frames, so it survives
        # until fewer than 2 clear frames remain -- onset 0 and 1 are invisible.
        stars = [CARD_CLEAR_STARS] * onset + [CARD_CLOUD_STARS] * (CHUNK_FRAMES - onset)
        expected = list(range(onset, CHUNK_FRAMES)) if onset >= 2 else []
        assert _cloud_indices(_run(stars)) == expected

    def test_a_run_that_starts_cloudy_and_clears_late_flags_the_cloudy_opening(self):
        # Why the baseline is the clearest frames rather than the first N: here the
        # opening 15 frames ARE the cloud, so a first-N baseline would read 24
        # detections as normal and flag nothing.
        stars = [CARD_CLOUD_STARS] * 15 + [CARD_CLEAR_STARS] * 5
        assert _cloud_indices(_run(stars)) == list(range(15))

    @pytest.mark.parametrize(
        "loss_fraction,caught", [(0.50, False), (0.60, False), (0.61, False), (0.62, True)]
    )
    def test_the_slope_envelope_is_a_61_percent_star_loss_across_one_chunk(
        self, loss_fraction, caught
    ):
        # Sweep to the failure point: a linear decline over the chunk is caught only
        # once its last frame falls under 0.4 x the baseline, i.e. a loss of more than
        # ~61.3% end to end (~4.7%/frame over 20 frames). Anything shallower -- a slow
        # overcast spread across several chunks -- is NOT caught by this rule.
        stars = [
            int(round(CARD_CLEAR_STARS * (1 - loss_fraction * i / (CHUNK_FRAMES - 1))))
            for i in range(CHUNK_FRAMES)
        ]
        assert bool(_cloud_indices(_run(stars))) is caught

    def test_a_10_percent_background_rise_alone_is_not_flagged(self):
        # The background rose 0.03 -> 0.033 (1.1x), well under the 1.5x
        # jump the rule requires: the star collapse is what catches it, not
        # the background half.
        bgs = [CARD_CLEAR_BG] * 3 + [CARD_CLOUD_BG] * (CHUNK_FRAMES - 3)
        assert _cloud_indices(_run([CARD_CLEAR_STARS] * CHUNK_FRAMES, bgs)) == []

    def test_a_chunk_that_is_overcast_end_to_end_is_not_flagged(self):
        # Pinned residual limitation: triage sees one chunk, so a chunk entirely past
        # the transition has no clear frame to be judged against. Catching that needs
        # a run-level baseline threaded across chunks, which this module cannot see.
        stars = [CARD_CLOUD_STARS] * CHUNK_FRAMES
        bgs = [CARD_CLOUD_BG] * CHUNK_FRAMES
        assert _cloud_indices(_run(stars, bgs)) == []

    def test_a_clear_run_with_ordinary_scatter_is_never_flagged(self):
        rng = np.random.default_rng(0)
        stars = [int(s) for s in rng.normal(CARD_CLEAR_STARS, 10.0, CHUNK_FRAMES)]
        bgs = [float(b) for b in rng.normal(CARD_CLEAR_BG, 0.0005, CHUNK_FRAMES)]
        assert _cloud_indices(_run(stars, bgs)) == []

    def test_a_transient_cloud_crossing_a_clear_run_is_still_flagged(self):
        stars = [CARD_CLEAR_STARS] * CHUNK_FRAMES
        bgs = [CARD_CLEAR_BG] * CHUNK_FRAMES
        stars[9], bgs[9] = CARD_CLOUD_STARS, CARD_CLEAR_BG * 2.0
        assert _cloud_indices(_run(stars, bgs)) == [9]

    def test_omitting_run_baseline_reproduces_the_old_per_chunk_result_exactly(self):
        frames = _run([CARD_CLOUD_STARS] * CHUNK_FRAMES, [CARD_CLOUD_BG] * CHUNK_FRAMES)
        assert triage(frames).cloud_frames == triage(frames, run_baseline=None).cloud_frames == ()


class TestRunCloudBaseline:
    """Closes the residual pinned above: a run-level baseline, carried forward
    across chunks (never a second decode pass -- see ``update_run_cloud_baseline``),
    threaded into ``triage`` as an OPTIONAL parameter."""

    def test_a_carried_baseline_flags_a_chunk_with_no_clear_frame_of_its_own(self):
        clear_chunk = _run([CARD_CLEAR_STARS] * CHUNK_FRAMES)
        run_baseline = update_run_cloud_baseline(None, clear_chunk)
        overcast_chunk = _run([CARD_CLOUD_STARS] * CHUNK_FRAMES, [CARD_CLOUD_BG] * CHUNK_FRAMES)
        assert _cloud_indices(overcast_chunk, run_baseline=run_baseline) == list(
            range(CHUNK_FRAMES)
        )

    def test_a_run_that_never_clears_across_chunks_stays_unflagged(self):
        chunk1 = _run([CARD_CLOUD_STARS] * CHUNK_FRAMES, [CARD_CLOUD_BG] * CHUNK_FRAMES)
        chunk2 = _run([CARD_CLOUD_STARS] * CHUNK_FRAMES, [CARD_CLOUD_BG] * CHUNK_FRAMES)
        baseline = update_run_cloud_baseline(None, chunk1)
        assert _cloud_indices(chunk1, run_baseline=None) == []
        assert _cloud_indices(chunk2, run_baseline=baseline) == []

    @pytest.mark.parametrize("run_baseline", [None, (CARD_CLEAR_STARS, CARD_CLEAR_BG)])
    def test_folding_a_run_baseline_never_unflags_what_the_chunk_alone_would_flag(
        self, run_baseline
    ):
        rows = [(120, 0.05, 3.0), (10, 0.20, 3.0), (118, 0.05, 3.1)]
        frames = [_fm(f"f{i}.jpg", *r) for i, r in enumerate(rows)]
        report = triage(frames, run_baseline=run_baseline)
        assert Path("f1.jpg") in report.cloud_frames

    def test_an_all_defocused_chunk_does_seed_the_baseline_and_a_focused_chunk_reclaims_it(self):
        # CONTRACT REVERSAL, deliberate. This case previously asserted the opposite --
        # that an all-defocused chunk contributes NOTHING, so a later sharper chunk is
        # never flagged against it. That rested on the premise that a defocused frame's
        # 140 detections were inflated noise blobs.
        #
        # The premise is wrong in the case that matters: a soft-focused frame of a clear
        # sky has a high count of real stars (classify_shape reads fwhm 11.0 at
        # ellipticity 0.05 as likely_defocus), and the clouded frames are the low-count
        # ones. Contributing nothing left the run with no clear reference at all and the
        # cloud unflagged -- the feature was inert in the very case it was written for.
        #
        # The fixture below is numerically identical either way. Nothing in a star count
        # distinguishes "soft focus, clear sky" from "defocused blobs", so this is a
        # chosen trade, not a solved problem: the risk of over-flagging genuinely sharp
        # frames is accepted, and BOUNDED by the reclaim asserted at the end.
        defocused_chunk = [
            _fm(f"d{i}.jpg", 140, CARD_CLEAR_BG, 11.0, shape=_defocused_shape())
            for i in range(CHUNK_FRAMES)
        ]
        sharp_chunk = _run([CARD_CLOUD_STARS] * CHUNK_FRAMES)

        baseline = update_run_cloud_baseline(None, defocused_chunk)
        assert baseline.from_focused is False  # held provisionally, not trusted
        assert len(triage(sharp_chunk, run_baseline=baseline).cloud_frames) == CHUNK_FRAMES

        # The bound: one genuinely in-focus chunk REPLACES the provisional baseline
        # outright, so an inflated count cannot survive past real focus evidence.
        reclaimed = update_run_cloud_baseline(baseline, _run([60] * CHUNK_FRAMES))
        assert reclaimed == (60.0, CARD_CLEAR_BG, True)

    @pytest.mark.parametrize("fwhm,leaks_through", [(7.9, True), (8.0, True), (8.01, False)])
    def test_the_defocus_filter_envelope_is_classify_shapes_8px_roundness_boundary(
        self, fwhm, leaks_through
    ):
        # update_run_cloud_baseline inherits classify_shape's DEFOCUS_FWHM_MIN_PX (8.0)
        # boundary exactly rather than owning a separate threshold. A frame at or under
        # it reads "indeterminate", not "likely_defocus", so it counts as FOCUSED and its
        # star count is trusted outright -- a genuine, documented miss: nothing here can
        # tell "borderline soft focus" from "borderline in focus" any better than
        # classify_shape already does.
        #
        # Above the boundary the chunk still seeds a baseline (see the reversal above),
        # but as a PROVISIONAL one -- so what the boundary now governs is trust
        # (from_focused), not whether a contribution happens at all.
        noisy = [
            _fm(f"d{i}.jpg", 140, CARD_CLEAR_BG, fwhm, shape=_defocused_shape(fwhm))
            for i in range(CHUNK_FRAMES)
        ]
        baseline = update_run_cloud_baseline(None, noisy)
        assert baseline.star == 140.0  # seeded either way
        assert baseline.from_focused is leaks_through


class TestRunCloudBaselineDefocusFallback:
    """A night whose clear opening frames are soft-focused (high star count, read as
    likely_defocus) while the later clouded frames read `indeterminate` (low star
    count). Filtering defocus out of the run baseline therefore discarded the only
    clear sky in the run and let the cloud define it -- the fix was inert in the very
    case it was written for.
    """

    def _defocused_clear_chunk(self):
        return [
            _fm(f"d{i}.jpg", CARD_CLEAR_STARS, CARD_CLEAR_BG, 11.0, shape=_defocused_shape())
            for i in range(CHUNK_FRAMES)
        ]

    def test_an_all_defocused_opening_chunk_still_seeds_the_run_baseline(self):
        # Was: contributes nothing, baseline stays None, the cloud goes unflagged.
        baseline = update_run_cloud_baseline(None, self._defocused_clear_chunk())
        assert baseline is not None
        assert baseline[0] == float(CARD_CLEAR_STARS)

    def test_a_clouded_chunk_is_flagged_against_a_soft_clear_opening(self):
        baseline = update_run_cloud_baseline(None, self._defocused_clear_chunk())
        overcast = _run([CARD_CLOUD_STARS] * CHUNK_FRAMES, [CARD_CLOUD_BG] * CHUNK_FRAMES)
        assert _cloud_indices(overcast, run_baseline=baseline) == list(range(CHUNK_FRAMES))

    def test_a_focused_chunk_REPLACES_a_defocus_derived_baseline_rather_than_merging(self):
        # The hazard the filter existed for, contained: if the defocused numbers WERE
        # inflated noise blobs, the first genuinely in-focus chunk overwrites them
        # outright instead of max()-ing with them, so the inflation cannot persist.
        defocused = update_run_cloud_baseline(None, self._defocused_clear_chunk())
        focused = update_run_cloud_baseline(defocused, _run([60] * CHUNK_FRAMES))
        assert focused[0] == 60.0  # not max(140, 60)

    def test_a_defocused_chunk_never_raises_a_baseline_a_focused_chunk_already_set(self):
        focused = update_run_cloud_baseline(None, _run([60] * CHUNK_FRAMES))
        after = update_run_cloud_baseline(focused, self._defocused_clear_chunk())
        assert after[0] == 60.0  # the 140 defocused count is refused, not merged

    def test_a_plain_two_tuple_run_baseline_still_works_in_triage(self):
        # Callers (and the tests above) hand triage a bare (star, background) pair;
        # widening what update returns must not break that.
        overcast = _run([CARD_CLOUD_STARS] * CHUNK_FRAMES, [CARD_CLOUD_BG] * CHUNK_FRAMES)
        pair = (float(CARD_CLEAR_STARS), CARD_CLEAR_BG)
        assert _cloud_indices(overcast, run_baseline=pair) == list(range(CHUNK_FRAMES))

    def test_an_empty_chunk_leaves_the_baseline_untouched(self):
        seeded = update_run_cloud_baseline(None, _run([60] * CHUNK_FRAMES))
        assert update_run_cloud_baseline(seeded, []) == seeded
        assert update_run_cloud_baseline(None, []) is None


class TestUpdateRunCloudBaseline:
    def test_seeds_from_the_first_contribution(self):
        frames = _run([CARD_CLEAR_STARS] * CHUNK_FRAMES)
        assert update_run_cloud_baseline(None, frames) == (
            float(CARD_CLEAR_STARS),
            CARD_CLEAR_BG,
            True,  # in-focus frames, so the baseline is trusted outright
        )

    def test_merges_by_keeping_the_clearest_seen_so_far_in_either_direction(self):
        baseline = update_run_cloud_baseline(
            None, _run([100] * CHUNK_FRAMES, [0.05] * CHUNK_FRAMES)
        )
        baseline = update_run_cloud_baseline(
            baseline, _run([150] * CHUNK_FRAMES, [0.03] * CHUNK_FRAMES)
        )
        assert baseline == (150.0, 0.03, True)
        # A cloudier later chunk never erodes the clearer baseline already established.
        baseline = update_run_cloud_baseline(
            baseline, _run([90] * CHUNK_FRAMES, [0.09] * CHUNK_FRAMES)
        )
        assert baseline == (150.0, 0.03, True)


def _fm_streaks(streaks, name="f.jpg", captured_at=None):
    return FrameMetrics(Path(name), 0, 0.0, 0.0, streaks, True, captured_at=captured_at)


class TestClassifyStreaks:
    def test_a_satellite_progresses_across_three_consecutive_frames_at_plausible_cadence(self):
        t0 = datetime(2026, 8, 1, 22, 0, 0)  # Needs a real, seconds-scale gap.
        frames = [
            _fm_streaks([(10.0, 50.0, 50.0, 50.0)], "f0.jpg", captured_at=t0),
            _fm_streaks(
                [(70.0, 50.0, 110.0, 50.0)], "f1.jpg", captured_at=t0 + timedelta(seconds=5)
            ),
            _fm_streaks(
                [(130.0, 50.0, 170.0, 50.0)], "f2.jpg", captured_at=t0 + timedelta(seconds=10)
            ),
        ]
        labels = [c.label for c in classify_streaks(frames)]
        assert labels == ["satellite", "satellite", "satellite"]

    def test_a_single_frame_streak_is_unknown_without_positive_evidence(self):
        frames = [_fm_streaks([(10.0, 10.0, 60.0, 10.0)])]
        result = classify_streaks(frames)
        assert len(result) == 1
        assert result[0].label == "unknown"
        assert result[0].frame_index == 0
        assert result[0].streak == (10.0, 10.0, 60.0, 10.0)

    def test_non_collinear_streaks_in_adjacent_frames_stay_unknown(self):
        t0 = datetime(2026, 8, 1, 22, 0, 0)
        frames = [
            _fm_streaks([(10.0, 50.0, 50.0, 50.0)], "f0.jpg", captured_at=t0),
            _fm_streaks(
                [(80.0, 10.0, 80.0, 90.0)], "f1.jpg", captured_at=t0 + timedelta(seconds=5)
            ),
        ]
        labels = [c.label for c in classify_streaks(frames)]
        assert labels == ["unknown", "unknown"]

    def test_missing_capture_timestamps_decline_satellite_chaining_with_a_reason(self):
        frames = [
            _fm_streaks([(10.0, 50.0, 50.0, 50.0)], "f0.jpg"),
            _fm_streaks([(70.0, 50.0, 110.0, 50.0)], "f1.jpg"),
        ]  # collinear, close endpoints -- would chain if timestamps were present
        result = classify_streaks(frames)
        labels = [c.label for c in result]
        assert "satellite" not in labels
        assert all(c.reason for c in result)

    def test_a_960_second_real_cadence_declines_satellite_chaining_and_never_becomes_meteor(self):
        t0 = datetime(2026, 8, 1, 22, 0, 0)
        frames = [
            _fm_streaks([(10.0, 50.0, 50.0, 50.0)], "f0.jpg", captured_at=t0),
            _fm_streaks(
                [(70.0, 50.0, 110.0, 50.0)], "f1.jpg", captured_at=t0 + timedelta(seconds=960)
            ),
        ]
        result = classify_streaks(frames)
        labels = [c.label for c in result]
        assert labels == ["unknown", "unknown"]
        assert "meteor" not in labels
        assert all(c.reason and "960" in c.reason for c in result)

    def test_a_position_repeating_across_frames_is_an_artifact_while_a_one_off_stays_unknown(self):
        defect = (100.0, 100.0, 104.0, 100.0)
        one_off = (10.0, 10.0, 60.0, 10.0)
        frames = [
            _fm_streaks([defect], "f0.jpg"),
            _fm_streaks([defect, one_off], "f1.jpg"),
            _fm_streaks([defect], "f2.jpg"),
        ]
        result = classify_streaks(frames)  # no captured_at anywhere -> chaining declined
        by_frame = {(c.frame_index, c.streak): c.label for c in result}
        assert by_frame[(0, defect)] == "artifact"
        assert by_frame[(1, defect)] == "artifact"
        assert by_frame[(2, defect)] == "artifact"
        assert by_frame[(1, one_off)] == "unknown"
        assert all(c.reason for c in result)  # Run-level, stamped on every label

    def test_a_position_repeating_only_twice_is_not_yet_an_artifact(self):
        defect = (100.0, 100.0, 104.0, 100.0)
        frames = [_fm_streaks([defect], "f0.jpg"), _fm_streaks([defect], "f1.jpg")]
        labels = [c.label for c in classify_streaks(frames)]
        assert "artifact" not in labels

    def test_a_dashed_collinear_strobe_set_is_aircraft(self):
        streaks = [
            (10.0, 100.0, 18.0, 100.0),
            (32.0, 100.0, 40.0, 100.0),
            (54.0, 100.0, 62.0, 100.0),
            (76.0, 100.0, 84.0, 100.0),
        ]
        labels = [c.label for c in classify_streaks([_fm_streaks(streaks)])]
        assert labels == ["aircraft"] * 4

    def test_a_multi_segment_fixed_position_defect_is_artifact_not_aircraft(self):
        dashes = [
            (2820.0, 2912.0, 2820.0, 2920.0),
            (2820.0, 2924.0, 2820.0, 2932.0),
            (2820.0, 2936.0, 2820.0, 2944.0),
        ]
        frames = [_fm_streaks(dashes, f"f{i}.jpg") for i in range(5)]
        labels = [c.label for c in classify_streaks(frames)]
        assert labels == ["artifact"] * 15

    def test_non_increasing_capture_times_decline_satellite_chaining(self):
        t0 = datetime(2026, 8, 1, 22, 0, 0)
        big_reversal = [
            _fm_streaks(
                [(10.0, 50.0, 50.0, 50.0)], "f0.jpg", captured_at=t0 + timedelta(seconds=960)
            ),
            _fm_streaks([(70.0, 50.0, 110.0, 50.0)], "f1.jpg", captured_at=t0),
        ]
        assert "satellite" not in [c.label for c in classify_streaks(big_reversal)]

        small_reversal = [  # 5s magnitudes -- abs() gap size alone must not pass.
            _fm_streaks(
                [(130.0, 50.0, 170.0, 50.0)], "f0.jpg", captured_at=t0 + timedelta(seconds=10)
            ),
            _fm_streaks(
                [(70.0, 50.0, 110.0, 50.0)], "f1.jpg", captured_at=t0 + timedelta(seconds=5)
            ),
            _fm_streaks([(10.0, 50.0, 50.0, 50.0)], "f2.jpg", captured_at=t0),
        ]
        assert "satellite" not in [c.label for c in classify_streaks(small_reversal)]

    def test_artifact_footprint_agreement_pinned_both_sides_of_tolerance(self):
        gapped = [  # separated beyond the tolerance on both sides never clusters.
            _fm_streaks([(100.0, 100.0, 104.0, 100.0)], "g0.jpg"),
            _fm_streaks([(107.5, 100.0, 111.5, 100.0)], "g1.jpg"),  # 3.5px gap
            _fm_streaks([(92.5, 100.0, 96.5, 100.0)], "g2.jpg"),  # 3.5px gap, other side
        ]
        assert "artifact" not in [c.label for c in classify_streaks(gapped)]
        diagonal = [  # zero shared pixels, far apart -- bbox touch is not footprint
            _fm_streaks([(500.0, 500.0, 3500.0, 2500.0)], "a.jpg"),
            _fm_streaks([(600.0, 2400.0, 1200.0, 2450.0)], "b.jpg"),
            _fm_streaks([(3000.0, 600.0, 3400.0, 650.0)], "c.jpg"),
        ]
        assert "artifact" not in [c.label for c in classify_streaks(diagonal)]

        defect = (100.0, 100.0, 104.0, 100.0)
        cases = [((101.0, 100.0, 105.0, 100.0), "artifact")]  # 1px shift agrees
        cases += [((102.0, 100.0 - h, 102.0, 100.0 + h), "unknown") for h in (10.0, 40.5, 250.0)]
        for other, want in cases:  # crosser unknown at ANY length
            f0, f2 = _fm_streaks([defect], "s0"), _fm_streaks([defect], "s2")
            f1 = _fm_streaks([other, defect], "s1")
            by_frame = {(c.frame_index, c.streak): c.label for c in classify_streaks([f0, f1, f2])}
            assert by_frame[(1, defect)] == "artifact"
            assert by_frame[(1, other)] == want, other

    def test_a_position_drift_chain_never_merges_transitively(self):
        # p1~p2 and p2~p3 agree but p1~p3 doesn't -- only every-pair agreement rejects this.
        positions = [
            (100.0, 100.0, 104.0, 100.0),
            (102.9, 100.0, 106.9, 100.0),
            (105.8, 100.0, 109.8, 100.0),
        ]
        frames = [_fm_streaks([p], f"c{i}") for i, p in enumerate(positions)]
        assert "artifact" not in [c.label for c in classify_streaks(frames)]

    def test_a_far_apart_pair_is_rejected_from_the_boxes_without_measuring_it(self, monkeypatch):
        # The bbox reject keeps clustering off the ~800x path (24 vs 44852 point-segment
        # calls over 20x15 scattered streaks), so a far pair must never reach those calls.
        monkeypatch.setattr(
            triage_mod, "_point_segment_distance", lambda *a: pytest.fail("measured")
        )
        a, b = (0.0, 0.0, 60.0, 40.0), (3000.0, 3000.0, 3060.0, 3040.0)
        assert not triage_mod._footprint_agrees(a, b, triage_mod._bbox(a), triage_mod._bbox(b))

    def test_an_empty_run_returns_no_classifications(self):
        assert classify_streaks([]) == ()

    def test_a_single_frame_run_cannot_produce_a_satellite(self):
        two_streaks = [(10.0, 50.0, 50.0, 50.0), (10.0, 90.0, 50.0, 90.0)]
        labels = [c.label for c in classify_streaks([_fm_streaks(two_streaks)])]
        assert "satellite" not in labels


class TestShapeMetrics:
    """The old fixed 11px window ceilinged FWHM at its own size."""

    def test_a_star_wider_than_the_old_window_measures_above_the_old_ceiling(self, tmp_path):
        # sigma=6.0px -> true FWHM ~14.1px, past the old fixed 11px window (~7.45px ceiling).
        luma = _place_stars((200, 200), [(100.0, 100.0)], sigma_px=6.0, noise=0.001, seed=12)
        metrics = measure_frame(tmp_path / "wide.jpg", luma)
        assert metrics.fwhm_px > 9.0
        assert metrics.fwhm_px == pytest.approx(6.0 * GAUSSIAN_SIGMA_TO_FWHM, rel=0.3)
        assert metrics.shape is not None
        assert metrics.shape.saturated is False

    def test_an_elongated_star_is_high_ellipticity_oriented_and_trailing(self, tmp_path):
        angle_deg = 30.0
        luma = _place_elongated_star((200, 200), (100.0, 100.0), 4.5, 1.2, angle_deg, seed=10)
        metrics = measure_frame(tmp_path / "trail.jpg", luma)
        assert metrics.shape is not None
        assert metrics.shape.ellipticity > 0.5
        assert _orientation_diff(metrics.shape.orientation_deg, angle_deg) < 10.0
        assert classify_shape(metrics) == "likely_trailing"

    def test_a_round_broad_star_classifies_likely_defocus(self, tmp_path):
        luma = _place_stars((200, 200), [(100.0, 100.0)], sigma_px=6.0, noise=0.001, seed=11)
        metrics = measure_frame(tmp_path / "soft.jpg", luma)
        assert metrics.shape is not None
        assert metrics.shape.ellipticity < 0.2
        assert classify_shape(metrics) == "likely_defocus"

    def test_broad_corners_with_a_tight_center_classify_corner_aberration(self, tmp_path):
        luma = _place_stars((240, 240), [(120.0, 120.0)], sigma_px=1.2, noise=0.001, seed=13)
        corner_positions = [(20.0, 20.0), (220.0, 20.0), (20.0, 220.0), (220.0, 220.0)]
        yy, xx = (a.astype(np.float32) for a in np.mgrid[0:240, 0:240])
        for cx, cy in corner_positions:
            r2 = (xx - cx) ** 2 + (yy - cy) ** 2
            luma = luma + (0.6 * np.exp(-r2 / (2.0 * 5.0**2))).astype(np.float32)
        luma = np.clip(luma, 0.0, 1.0).astype(np.float32)
        metrics = measure_frame(tmp_path / "corner.jpg", luma)
        assert metrics.region_shape is not None
        assert metrics.region_shape.center is not None
        assert metrics.region_shape.corners is not None
        assert classify_shape(metrics) == "corner_aberration"

    def test_a_faint_in_focus_star_does_not_inflate_from_added_area_noise(self, tmp_path):
        # Regression: added-area noise always clears a modest star's small prev_total.
        luma = _place_stars((160, 160), [(80.0, 80.0)], sigma_px=1.6, flux=0.12, noise=0.004)
        metrics = measure_frame(tmp_path / "faint.jpg", luma)
        assert metrics.fwhm_px == pytest.approx(1.6 * GAUSSIAN_SIGMA_TO_FWHM, rel=0.3)
        assert classify_shape(metrics) != "likely_defocus"

    def test_a_weak_no_star_frame_classifies_indeterminate_without_raising(self, tmp_path):
        luma = np.full((160, 160), 0.05, dtype=np.float32)
        metrics = measure_frame(tmp_path / "empty.jpg", luma)
        assert metrics.shape is None
        assert classify_shape(metrics) == "indeterminate"


class TestMeteorDetectionIsOptIn:
    """Streak detection dominates measure_frame's cost,
    and the CLI passes stack_run no meteor policy at all, so on a normal
    run every frame pays for a search whose only consumer is a diagnostic field. It has to
    be possible to not do the work -- and 'did not look' must never be recorded as 'looked
    and found none'."""

    def _frame(self):
        # 0..1 LINEAR luma, matching what astro_calibrate hands triage -- a raw-count-scaled
        # fixture reaches the streak detector's MAD-sigma threshold differently and finds
        # nothing, which measures the fixture rather than the switch under test.
        rows, cols = np.indices((200, 200))
        frame = np.full((200, 200), 0.05, dtype=np.float32)
        for r0, c0 in ((60, 80), (120, 160), (170, 40)):
            frame += 0.6 * np.exp(-(((rows - r0) ** 2 + (cols - c0) ** 2) / (2 * 1.5**2)))
        cv2.line(frame, (30, 100), (170, 100), 0.9, 7)  # the streak the opt-in path must find
        return frame.astype(np.float32)

    def test_opting_out_skips_the_streak_scan_entirely(self, monkeypatch):
        calls = []
        real = triage_mod.detect_streak_components
        monkeypatch.setattr(
            triage_mod,
            "detect_streak_components",
            lambda *a, **k: calls.append(1) or real(*a, **k),
        )
        triage_mod.measure_frame(Path("f.NEF"), self._frame(), detect_meteors=False)
        assert calls == [], "detect_streak_components ran despite detect_meteors=False"

    def test_not_looking_is_recorded_as_unknown_not_as_empty(self):
        m = triage_mod.measure_frame(Path("f.NEF"), self._frame(), detect_meteors=False)
        assert m.streaks is None, "an empty list would claim the frame was searched"
        assert m.streak_components is None

    def test_opting_in_still_finds_streaks(self):
        m = triage_mod.measure_frame(Path("f.NEF"), self._frame(), detect_meteors=True)
        assert m.streaks, "the opt-in path must still detect the injected streak"

    def test_the_default_still_searches(self):
        m = triage_mod.measure_frame(Path("f.NEF"), self._frame())
        assert m.streaks is not None, "default must stay backward-compatible"

    def test_star_metrics_are_unaffected_by_the_switch(self):
        on = triage_mod.measure_frame(Path("f.NEF"), self._frame(), detect_meteors=True)
        off = triage_mod.measure_frame(Path("f.NEF"), self._frame(), detect_meteors=False)
        assert on.star_count == off.star_count
        assert on.fwhm_px == pytest.approx(off.fwhm_px)
        assert on.usable == off.usable

    def test_classifying_unsearched_frames_refuses_instead_of_reporting_none(self):
        skipped = triage_mod.measure_frame(
            Path("f.NEF"), self._frame(), detect_meteors=False, captured_at=datetime(2026, 8, 16)
        )
        with pytest.raises(triage_mod.StreakDetectionNotRunError, match="detect_meteors=False"):
            triage_mod.classify_streaks([skipped])

    def test_classifying_searched_frames_still_works(self):
        searched = triage_mod.measure_frame(
            Path("f.NEF"), self._frame(), detect_meteors=True, captured_at=datetime(2026, 8, 16)
        )
        assert triage_mod.classify_streaks([searched]) is not None


# ---------------------------------------------------------------------------------
# Cap-on frames (the lens cap left on at the end of a night, darks still on the card).
# The three populations, restated as the numbers ``classify_exposure`` reads --
# (median, upper half-width p84-median, p99.9) -- measured on simulated frames of the
# shape a typical night shows (14-bit Nikon Z DX, black level 1008):
# lights mean 1179.1 / std 169.1, cap-on darks 1009.0 / 17.8, mid-exposure frame 1049.3 / 55.7
# (the cap went on MID-exposure). Only ratios are read, so the unit cancels.
CAP_ON_STATS = (0.000311, 0.000896, 0.003016)  # level 0.35, structure 3.02
MID_EXPOSURE_STATS = (0.001465, 0.001644, 0.339400)  # level 0.89, structure 205.6
LIGHT_STATS = (0.012064, 0.003805, 0.355200)  # level 3.17, structure 90.2
OVERCAST_STATS = (0.020003, 0.001998, 0.026177)  # level 10.0, structure 3.09


class TestClassifyExposure:
    def test_the_cards_cap_on_darks_read_cap_on(self):
        assert classify_exposure(*CAP_ON_STATS) == "cap_on"

    def test_the_cards_light_frames_read_light(self):
        assert classify_exposure(*LIGHT_STATS) == "light"

    def test_the_mid_exposure_cap_frame_is_neither(self):
        """The cap went on DURING the exposure, so the frame holds part sky
        and part black floor. Guessing it into either population is the failure this
        test exists to prevent -- it must come back unclassified."""
        assert classify_exposure(*MID_EXPOSURE_STATS) == "indeterminate"

    def test_a_uniformly_bright_overcast_frame_is_not_cap_on(self):
        """Flat, but nowhere near the black floor -- 'it clouded over', not 'cap on'."""
        assert classify_exposure(*OVERCAST_STATS) == "indeterminate"

    def test_a_frame_with_no_spread_at_the_zero_point_is_cap_on(self):
        assert classify_exposure(0.0, 0.0, 0.0) == "cap_on"

    def test_a_flat_frame_above_the_zero_point_is_not_cap_on(self):
        assert classify_exposure(0.05, 0.0, 0.05) == "indeterminate"


class TestMeasureFrameExposureClass:
    def _capped(self, shape=(160, 160)):
        """Read noise on the calibrated black floor, negatives clipped the way
        ``astro_calibrate``'s postprocess leaves them."""
        rng = np.random.default_rng(3)
        return np.clip(rng.normal(0.0003, 0.0009, size=shape), 0.0, None).astype(np.float32)

    def test_a_cap_on_frame_is_classified_and_refused(self, tmp_path):
        metrics = measure_frame(tmp_path / "DSC_0031.NEF", self._capped(), detect_meteors=False)
        assert metrics.exposure_class == "cap_on"
        assert metrics.usable is False

    def test_a_star_field_still_reads_light_and_usable(self, tmp_path):
        luma = _place_stars((160, 160), STAR_POSITIONS, noise=0.004, seed=4)
        metrics = measure_frame(tmp_path / "DSC_0030.NEF", luma, detect_meteors=False)
        assert metrics.exposure_class == "light"
        assert metrics.usable is True

    def test_subsampling_a_large_frame_does_not_change_the_verdict(self, tmp_path, monkeypatch):
        """The percentile pass is stride-subsampled for cost; the class must not move."""
        luma = self._capped(shape=(400, 400))
        full = measure_frame(tmp_path / "full.NEF", luma, detect_meteors=False)
        monkeypatch.setattr(triage_mod, "EXPOSURE_STATS_MAX_SAMPLES", 5_000)
        sampled = measure_frame(tmp_path / "sampled.NEF", luma, detect_meteors=False)
        assert full.exposure_class == sampled.exposure_class == "cap_on"


def _cap_on_fm(name):
    """A cap-on frame as triage sees it: no stars, no FWHM, at the black floor."""
    return FrameMetrics(Path(name), 0, 0.0003, 0.0, [], False, exposure_class="cap_on")


class TestTriageCapOnFrames:
    def test_cap_on_frames_never_define_the_cloud_baseline(self):
        """The bug: sorted-lowest-background picks the darks as 'clearest sky', and
        every real light then reads as a background jump and is REJECTED downstream."""
        frames = [_cap_on_fm(f"d{i}.NEF") for i in range(3)]
        frames += [_fm(f"f{i}.NEF", 120, 0.028, 3.0) for i in range(5)]
        report = triage(frames)
        assert report.cloud_frames == ()
        assert report.cap_on_frames == tuple(Path(f"d{i}.NEF") for i in range(3))

    def test_cap_on_frames_are_not_reported_as_cloud_or_dew(self):
        frames = [_fm(f"f{i}.NEF", 120, 0.028, 3.0) for i in range(5)]
        frames += [_cap_on_fm("d0.NEF")]
        report = triage(frames)
        assert Path("d0.NEF") not in report.cloud_frames
        assert Path("d0.NEF") not in report.dew_frames

    def test_cap_on_frames_never_define_the_dew_baseline(self):
        """A cap-on frame's fwhm_px is 0 (no stars), which would zero the opening
        baseline and switch dew detection off for the whole group."""
        frames = [_cap_on_fm(f"d{i}.NEF") for i in range(3)]
        frames += [_fm(f"f{i}.NEF", 120, 0.028, 3.0) for i in range(3)]
        frames += [_fm("dew.NEF", 120, 0.028, 6.0)]
        report = triage(frames)
        assert report.dew_frames == (Path("dew.NEF"),)

    def test_an_unclassified_frame_is_reported_and_otherwise_left_alone(self):
        """A frame between the populations is NEITHER: reported, still a light
        everywhere else (the baseline's median-of-3 already absorbs one outlier)."""
        frames = [_fm(f"f{i}.NEF", 120, 0.028, 3.0) for i in range(3)]
        frames += [_fm("mid.NEF", 60, 0.020, 3.0, exposure_class="indeterminate")]
        report = triage(frames)
        assert report.indeterminate_frames == (Path("mid.NEF"),)
        assert report.cap_on_frames == ()
        assert Path("mid.NEF") not in report.cloud_frames

    def test_a_cap_on_chunk_never_moves_the_run_cloud_baseline(self):
        seeded = update_run_cloud_baseline(None, _run([120, 118, 122]))
        after = update_run_cloud_baseline(seeded, [_cap_on_fm(f"d{i}.NEF") for i in range(3)])
        assert after == seeded
