"""Every inserted film still is graded vivid, clear and clean-edged."""

from __future__ import annotations

import colorsys
import functools
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from gomoveshift.pipeline.route_story import grade as grade_module
from gomoveshift.pipeline.route_story import hud as hud_module


def _scene_image(w: int = 240, h: int = 160) -> np.ndarray:
    rng = np.random.default_rng(7)
    coarse = rng.uniform(0.05, 0.95, size=(h // 8, w // 8, 3)).astype(np.float32)
    return np.clip(cv2.resize(coarse, (w, h), interpolation=cv2.INTER_NEAREST), 0.0, 1.0)


def _hazy(scene: np.ndarray, t_min: float = 0.35, t_max: float = 1.0) -> np.ndarray:
    h, w = scene.shape[:2]
    atmosphere = np.array([0.85, 0.87, 0.9], dtype=np.float32)
    t = np.tile(np.linspace(t_max, t_min, w, dtype=np.float32), (h, 1))
    return np.clip(scene * t[..., None] + atmosphere[None, None, :] * (1 - t[..., None]), 0.0, 1.0)


def _to_image(rgb01: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(rgb01, 0.0, 1.0) * 255.0).astype(np.uint8), "RGB")


def _rich_image(w: int = 200, h: int = 200) -> Image.Image:
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[: h // 2, : w // 2] = (0.9, 0.05, 0.05)
    img[: h // 2, w // 2 :] = (0.05, 0.85, 0.1)
    img[h // 2 :, : w // 2] = (0.05, 0.1, 0.9)
    img[h // 2 :, w // 2 :] = (0.95, 0.85, 0.05)
    return _to_image(img)


def _fine_detail_energy(rgb01):
    """Std of (luma - 2px blur) over the far-ridge band, at a fixed 1600px width."""
    h = rgb01.shape[0]
    luma = grade_module._luma(rgb01[int(0.25 * h) : int(0.55 * h)]).astype(np.float32)
    small = cv2.resize(
        luma, (1600, int(luma.shape[0] * 1600 / luma.shape[1])), interpolation=cv2.INTER_AREA
    )
    return float(np.std(small - cv2.GaussianBlur(small, (0, 0), 2.0)))


def _local_contrast(rgb01):
    """Mean windowed (31px) luma std over the far-ridge band -- contrast *within* the ridge."""
    h = rgb01.shape[0]
    luma = grade_module._luma(rgb01[int(0.25 * h) : int(0.55 * h)]).astype(np.float32)
    mean = cv2.blur(luma, (31, 31))
    variance = np.maximum(cv2.blur(luma * luma, (31, 31)) - mean * mean, 0.0)
    return float(np.mean(np.sqrt(variance)))


def test_the_dither_never_moves_a_value_already_on_an_8_bit_level():
    levels = np.tile(np.linspace(0.0, 1.0, 256, dtype=np.float32)[None, :, None], (8, 1, 3))
    dithered = grade_module._dither(levels)
    assert np.array_equal(np.rint(dithered * 255.0), np.rint(levels * 255.0))
    assert abs(float(np.mean(grade_module._BAYER4))) < 1e-6


def _hazy_sky_over_a_dark_ridge(w: int = 1200, h: int = 800) -> tuple[Image.Image, np.ndarray]:
    """A flat hazy distance above a dark, textured near ridge; returns the ridge row per column."""
    rng = np.random.default_rng(11)
    rgb = np.empty((h, w, 3), dtype=np.float32)
    rgb[:] = (0.70, 0.76, 0.84)
    rgb += rng.normal(0.0, 0.01, size=rgb.shape).astype(np.float32)
    edge = (h * 0.5 + 60.0 * np.sin(np.linspace(0.0, np.pi, w))).astype(int)
    ground = rng.uniform(0.04, 0.22, size=(h, w)).astype(np.float32)
    for x in range(w):
        rgb[edge[x] :, x] = np.stack(
            [ground[edge[x] :, x] * 0.5, ground[edge[x] :, x], ground[edge[x] :, x] * 0.45], axis=-1
        )
    return _to_image(rgb), edge


def _bottom_third(image: Image.Image) -> Image.Image:
    w, h = image.size
    return image.crop((0, h - h // 3, w, h))


def _rgb01(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _mean_luma01(image: Image.Image) -> float:
    return float(np.mean(grade_module._luma(_rgb01(image))))


def _mean_hue_degrees(image: Image.Image) -> float:
    hsv = cv2.cvtColor(_rgb01(image), cv2.COLOR_RGB2HSV)
    return float(np.mean(hsv[..., 0]))


def _half_hazy_half_clear_green(w: int = 240, h: int = 160) -> Image.Image:
    top = _hazy(_scene_image(w, h // 2), t_min=0.4, t_max=0.9)
    bottom = np.tile(np.array([0.05, 0.35, 0.05], dtype=np.float32), (h - h // 2, w, 1))
    return _to_image(np.concatenate([top, bottom], axis=0))


def _vignetted_image(w: int = 240, h: int = 160, corner: int = 40) -> Image.Image:
    image = Image.new("RGB", (w, h), (200, 200, 200))
    draw = ImageDraw.Draw(image)
    for cx, cy in ((0, 0), (w, 0), (0, h), (w, h)):
        draw.ellipse((cx - corner, cy - corner, cx + corner, cy + corner), fill=(5, 5, 5))
    return image


def test_measure_returns_the_expected_keys():
    result = grade_module.measure(Image.new("RGB", (40, 30), (100, 120, 90)))

    assert set(result) == {"colourfulness", "mean_luma", "clip_fraction", "haze_estimate"}
    assert all(isinstance(value, float) for value in result.values())


def test_a_hazy_synthetic_scene_recovers_contrast_and_loses_haze():
    hazy_image = _to_image(_hazy(_scene_image()))

    before = grade_module.measure(hazy_image)
    graded = grade_module.grade_still(hazy_image)
    after = grade_module.measure(graded)

    assert after["haze_estimate"] < before["haze_estimate"]
    before_std = float(np.std(np.asarray(hazy_image.convert("L"), dtype=np.float32)))
    after_std = float(np.std(np.asarray(graded.convert("L"), dtype=np.float32)))
    assert after_std > before_std


def test_a_dark_near_ridge_never_leaves_a_bright_glow_in_the_haze_beside_it():
    """The haze just above a ridgeline was identical to the haze further up; it
    must stay so, not survive as a white band hugging the ridge."""
    image, edge = _hazy_sky_over_a_dark_ridge()

    luma = grade_module._luma(_rgb01(grade_module.grade_still(image)))

    columns = range(100, luma.shape[1] - 100, 25)
    beside = float(np.mean([luma[edge[x] - 24 : edge[x] - 4, x].mean() for x in columns]))
    further = float(np.mean([luma[edge[x] - 160 : edge[x] - 100, x].mean() for x in columns]))
    assert beside - further <= 0.02


def test_an_already_rich_image_barely_changes():
    rich_image = _rich_image()
    before = grade_module.measure(rich_image)
    assert before["colourfulness"] >= 65

    graded = grade_module.grade_still(rich_image)
    after = grade_module.measure(graded)

    assert abs(after["colourfulness"] - before["colourfulness"]) < 10


def test_an_already_saturated_haze_free_frame_never_reads_duller():
    rich_image = _rich_image()
    before = grade_module.measure(rich_image)

    graded = grade_module.grade_still(rich_image)
    after = grade_module.measure(graded)

    assert after["colourfulness"] >= before["colourfulness"] - 1.0
    assert after["colourfulness"] <= before["colourfulness"] + 12.0
    assert abs(after["mean_luma"] - before["mean_luma"]) <= 0.04


def test_a_low_colour_subject_gets_only_a_capped_lift():
    grey_sign = Image.new("RGB", (120, 80), (140, 138, 142))
    before = grade_module.measure(grey_sign)

    graded = grade_module.grade_still(grey_sign)
    after = grade_module.measure(graded)

    assert after["colourfulness"] > before["colourfulness"]
    assert after["colourfulness"] < before["colourfulness"] + 20


def test_hue_shift_stays_small_on_a_saturated_patch():
    patch = Image.new("RGB", (60, 60), (200, 90, 40))
    hue_before, _, _ = colorsys.rgb_to_hsv(200 / 255, 90 / 255, 40 / 255)

    graded = grade_module.grade_still(patch)
    r, g, b = graded.getpixel((5, 5))
    hue_after, _, _ = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)

    diff = abs(hue_after - hue_before)
    degrees = min(diff, 1 - diff) * 360
    assert degrees < 6


def test_a_clear_dark_green_near_ground_does_not_darken_or_turn_blue():
    frame = _half_hazy_half_clear_green()
    before_bottom = _bottom_third(frame)
    before_luma = _mean_luma01(before_bottom)
    before_hue = _mean_hue_degrees(before_bottom)

    graded = grade_module.grade_still(frame)
    after_bottom = _bottom_third(graded)
    after_luma = _mean_luma01(after_bottom)
    after_hue = _mean_hue_degrees(after_bottom)

    assert after_luma >= before_luma - 0.01
    assert (after_hue - before_hue) <= 12


def test_output_size_and_mode_are_preserved():
    image = Image.new("RGB", (100, 64), (30, 60, 90))

    graded = grade_module.grade_still(image)

    assert graded.size == image.size
    assert graded.mode == image.mode


def test_grading_can_be_disabled():
    image = Image.new("RGB", (60, 40), (10, 200, 30))

    assert grade_module.grade_still(image, enabled=False).tobytes() == image.tobytes()


def test_dark_rounded_corners_are_detected_and_cropped_to_the_same_aspect():
    vignetted = _vignetted_image()

    cropped = grade_module.crop_lens_corners(vignetted)

    assert cropped.size != vignetted.size
    vw, vh = vignetted.size
    cw, ch = cropped.size
    assert abs((cw / ch) - (vw / vh)) < 0.02
    assert cropped.getpixel((1, 1))[0] > 150


def _textured_dark_corner_scene(w: int = 300, h: int = 200, wedge: int = 140) -> Image.Image:
    """A frame with large, noisy dark corners (trees/riverbank framing a
    scene) and bright straight edges between them -- passes the same
    corners-dark/edges-bright test a real lens-hood vignette does, but the
    dark region is scene content extending much further in than any real
    vignette margin, not a small rounded arc."""
    rng = np.random.default_rng(3)
    img = np.full((h, w), 210, dtype=np.float32)
    noise = rng.uniform(15, 70, size=(h, w)).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    corner_wedges = (
        (yy + xx) <= wedge,
        (yy + (w - 1 - xx)) <= wedge,
        ((h - 1 - yy) + xx) <= wedge,
        ((h - 1 - yy) + (w - 1 - xx)) <= wedge,
    )
    mask = corner_wedges[0] | corner_wedges[1] | corner_wedges[2] | corner_wedges[3]
    img[mask] = noise[mask]
    rgb = np.repeat(img[..., None], 3, axis=2).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


def test_dark_textured_scene_corners_are_not_mistaken_for_vignetting():
    scene = _textured_dark_corner_scene()

    assert grade_module.crop_lens_corners(scene).size == scene.size


def test_a_clean_image_is_untouched_by_corner_crop():
    clean = Image.new("RGB", (240, 160), (200, 200, 200))

    assert grade_module.crop_lens_corners(clean).size == clean.size


def test_load_still_grades_by_default(tmp_path, monkeypatch):
    path = tmp_path / "phone.jpg"
    Image.new("RGB", (64, 48), (140, 60, 30)).save(path)
    calls = []
    monkeypatch.setattr(
        grade_module,
        "grade_still",
        lambda image, **kwargs: calls.append(kwargs) or image,
    )

    hud_module._load_still(path)

    assert calls == [{"enabled": True}]


def test_load_still_grading_can_be_disabled_module_wide(tmp_path, monkeypatch):
    path = tmp_path / "phone.jpg"
    Image.new("RGB", (64, 48), (140, 60, 30)).save(path)
    calls = []
    monkeypatch.setattr(
        grade_module,
        "grade_still",
        lambda image, **kwargs: calls.append(kwargs) or image,
    )
    monkeypatch.setattr(hud_module, "_GRADE_STILLS", False)

    hud_module._load_still(path)

    assert calls == [{"enabled": False}]


# Directory of real camera RAW fixtures plus a JSON manifest naming them. Neither
# is configured by default: set GMS_TEST_RAW_DIR and GMS_TEST_RAW_MANIFEST to run
# the real-fixture tests, which are skipped otherwise. The manifest is
# {"roles": {<role>: <file name>}, "all_stills": [<file name>, ...]}; the tests
# refer to frames only by role (hazy_ridge_a..hazy_ridge_d, rich_frame,
# dim_clear_frame, arch_frame, vignette_a, vignette_b).
_FIXTURE_DIR = Path(os.environ.get("GMS_TEST_RAW_DIR", "/nonexistent-raw-fixtures"))


def _load_raw_manifest() -> dict:
    path = os.environ.get("GMS_TEST_RAW_MANIFEST")
    if not path:
        return {"roles": {}, "all_stills": []}
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {"roles": {}, "all_stills": []}
    return {"roles": dict(data.get("roles", {})), "all_stills": list(data.get("all_stills", []))}


_RAW_MANIFEST = _load_raw_manifest()
_RAW_ROLES: dict[str, str] = _RAW_MANIFEST["roles"]
_ALL_STILLS: list[str] = _RAW_MANIFEST["all_stills"]


def _raw(role: str) -> str:
    """The fixture file name the manifest assigns to ``role``."""
    return _RAW_ROLES[role]


def _raw_fixtures_present() -> bool:
    return bool(_RAW_ROLES) and all((_FIXTURE_DIR / name).exists() for name in _RAW_ROLES.values())


# Real camera RAW decode (slow over a mounted drive) and
# ``grade_still``'s full-resolution haze/exposure/vibrance/s-curve pipeline
# (slow under load) are both pure functions of
# the fixture filename -- memoize them on DISK, keyed by fixture name +
# cropped flag + a hash of grade.py/hud.py's own source bytes (so an edit
# to the pipeline invalidates stale entries) + the raw file's size and mtime, so
# the tests below -- several of which reload or literally re-grade the SAME
# preview -- don't repeat that real work even when xdist's worksteal
# scheduler spreads them across separate worker processes that share no
# memory. The cache lives under the session's shared ``tmp_path_factory``
# basetemp, so it never outlives this pytest session. A per-key claim file
# (``O_CREAT | O_EXCL``, holding the claimant's pid) lets exactly one worker
# compute each entry; every other worker waits for the file to land, then
# reads it. A waiter that finds the claimant dead (xdist worker crash: the
# ``finally`` never ran) removes the orphaned claim and takes it over, so one
# crash costs one test, not every test sharing that key. Returns a defensive
# ``.copy()`` so no test can mutate another's cached input/output.
@functools.lru_cache(maxsize=1)
def _pipeline_source_hash() -> str:
    digest = hashlib.sha256()
    for module in (grade_module, hud_module):
        digest.update(Path(module.__file__).read_bytes())
    return digest.hexdigest()[:16]


_CLAIM_WAIT_SECONDS = 900.0


def _claim_holder_alive(claim: Path) -> bool:
    """Is the process that wrote ``claim`` still running?

    An empty claim is a holder that created the file but has not written its
    pid yet (a microsecond window) -- treat it as alive. A missing claim means
    the holder finished or failed; the caller re-checks ``final`` and re-claims.
    """
    try:
        text = claim.read_text().strip()
    except FileNotFoundError:
        return False
    if not text:
        return True
    try:
        os.kill(int(text), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _disk_memoize_image(cache_dir: Path, key: str, compute) -> Image.Image:
    """Compute-once, cross-process disk cache for a PIL image, keyed by ``key``."""
    digest = hashlib.sha256(key.encode()).hexdigest()
    final = cache_dir / f"{digest}.png"
    mode_path = cache_dir / f"{digest}.mode"
    claim = cache_dir / f"{digest}.claim"

    deadline = time.monotonic() + _CLAIM_WAIT_SECONDS
    while not final.exists():
        try:
            fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if not _claim_holder_alive(claim):
                claim.unlink(missing_ok=True)
                continue
            if time.monotonic() > deadline:
                raise TimeoutError(f"timed out waiting for cached image {key!r}")
            time.sleep(0.05)
            continue
        try:
            os.write(fd, str(os.getpid()).encode())
        finally:
            os.close(fd)
        try:
            result = compute()
            tmp = cache_dir / f"{digest}.{os.getpid()}.tmp.png"
            result.save(tmp, format="PNG", compress_level=1)
            mode_path.write_text(result.mode)
            os.replace(tmp, final)
        finally:
            claim.unlink(missing_ok=True)

    image = Image.open(final)
    image.load()
    saved_mode = mode_path.read_text()
    if image.mode != saved_mode:
        image = image.convert(saved_mode)
    return image


def _fixture_cache_key(name: str, purpose: str) -> str:
    stat = (_FIXTURE_DIR / name).stat()
    return "|".join(
        (purpose, name, _pipeline_source_hash(), str(stat.st_size), str(stat.st_mtime_ns))
    )


@pytest.fixture(scope="session")
def grade_cache_dir(tmp_path_factory) -> Path:
    base = tmp_path_factory.getbasetemp()
    if os.environ.get("PYTEST_XDIST_WORKER"):
        base = base.parent
    cache_dir = base / "route_story_grade_cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def _cached_preview(cache_dir: Path, name: str) -> Image.Image:
    def _compute() -> Image.Image:
        preview = hud_module._camera_preview(_FIXTURE_DIR / name)
        assert preview is not None, name
        return preview

    key = _fixture_cache_key(name, "preview")
    return _disk_memoize_image(cache_dir, key, _compute).copy()


def _cached_crop(cache_dir: Path, name: str) -> Image.Image:
    key = _fixture_cache_key(name, "crop")
    return _disk_memoize_image(
        cache_dir, key, lambda: grade_module.crop_lens_corners(_cached_preview(cache_dir, name))
    ).copy()


def _cached_grade(cache_dir: Path, name: str, *, cropped: bool) -> Image.Image:
    key = _fixture_cache_key(name, f"grade-cropped={cropped}")

    def _compute() -> Image.Image:
        source = _cached_crop(cache_dir, name) if cropped else _cached_preview(cache_dir, name)
        return grade_module.grade_still(source)

    return _disk_memoize_image(cache_dir, key, _compute).copy()


def test_disk_memoize_image_caches_by_key_and_returns_independent_copies(tmp_path):
    calls = []

    def _compute() -> Image.Image:
        calls.append(len(calls))
        return Image.new("RGB", (4, 3), (10, 20, 30))

    first = _disk_memoize_image(tmp_path, "key-a", _compute)
    second = _disk_memoize_image(tmp_path, "key-a", _compute)
    assert len(calls) == 1
    assert first.mode == second.mode == "RGB"
    assert first.size == second.size == (4, 3)
    assert np.array_equal(np.asarray(first), np.asarray(second))

    first.putpixel((0, 0), (255, 255, 255))
    third = _disk_memoize_image(tmp_path, "key-a", _compute)
    assert np.array_equal(np.asarray(third), np.asarray(second))
    assert len(calls) == 1

    # A key change (standing in for a changed pipeline-source hash or a
    # changed raw-file mtime, both of which flow into the real fixture key)
    # recomputes rather than reusing the stale entry.
    fourth = _disk_memoize_image(tmp_path, "key-b", _compute)
    assert len(calls) == 2
    assert fourth.mode == "RGB"
    assert np.array_equal(np.asarray(fourth), np.asarray(second))


@pytest.mark.skipif(not _raw_fixtures_present(), reason="real camera fixtures not present")
class TestRealCameraFixtures:
    @pytest.mark.parametrize("role", ["hazy_ridge_a", "hazy_ridge_c"])
    def test_hazy_ridgeline_frames_gain_contrast_and_colour(self, grade_cache_dir, role):
        preview = _cached_preview(grade_cache_dir, _raw(role))
        before = grade_module.measure(preview)

        graded = _cached_grade(grade_cache_dir, _raw(role), cropped=False)
        after = grade_module.measure(graded)

        assert after["haze_estimate"] < before["haze_estimate"], role
        assert after["colourfulness"] > before["colourfulness"], role

    def test_an_already_rich_real_frame_changes_only_a_little(self, grade_cache_dir):
        preview = _cached_preview(grade_cache_dir, _raw("rich_frame"))
        before = grade_module.measure(preview)

        graded = _cached_grade(grade_cache_dir, _raw("rich_frame"), cropped=False)
        after = grade_module.measure(graded)

        assert abs(after["colourfulness"] - before["colourfulness"]) < 18
        assert after["clip_fraction"] <= max(before["clip_fraction"], 0.01) + 0.005

    def test_a_real_lens_hood_corner_is_removed_and_aspect_kept(self, grade_cache_dir):
        preview = _cached_preview(grade_cache_dir, _raw("rich_frame"))
        cropped = _cached_crop(grade_cache_dir, _raw("rich_frame"))

        pw, ph = preview.size
        cw, ch = cropped.size
        assert (cw, ch) != (pw, ph)
        assert abs((cw / ch) - (pw / ph)) < 0.02

    def test_the_near_ground_does_not_darken_or_turn_blue(self, grade_cache_dir):
        preview = _cached_preview(grade_cache_dir, _raw("hazy_ridge_c"))
        before_bottom = _bottom_third(preview)
        before_luma = _mean_luma01(before_bottom)
        before_hue = _mean_hue_degrees(before_bottom)

        graded = _cached_grade(grade_cache_dir, _raw("hazy_ridge_c"), cropped=False)
        after_bottom = _bottom_third(graded)
        after_luma = _mean_luma01(after_bottom)
        after_hue = _mean_hue_degrees(after_bottom)

        assert after_luma >= 0.95 * before_luma
        assert (after_hue - before_hue) <= 12

    @pytest.mark.parametrize(
        "role", ["hazy_ridge_a", "hazy_ridge_b", "hazy_ridge_c", "hazy_ridge_d"]
    )
    def test_the_near_ground_retains_brightness_and_hue_across_ridgeline_frames(
        self, grade_cache_dir, role
    ):
        cropped = _cached_crop(grade_cache_dir, _raw(role))
        before_bottom = _bottom_third(cropped)
        before_luma = _mean_luma01(before_bottom)
        before_hue = _mean_hue_degrees(before_bottom)

        graded = _cached_grade(grade_cache_dir, _raw(role), cropped=True)
        after_bottom = _bottom_third(graded)
        after_luma = _mean_luma01(after_bottom)
        after_hue = _mean_hue_degrees(after_bottom)

        assert after_luma >= 0.93 * before_luma, role
        assert (after_hue - before_hue) <= 12, role

    @pytest.mark.parametrize("role", ["hazy_ridge_b", "hazy_ridge_c"])
    def test_the_far_ridge_gains_local_contrast_and_loses_haze(self, grade_cache_dir, role):
        cropped = _cached_crop(grade_cache_dir, _raw(role))
        contrast_before = _local_contrast(_rgb01(cropped))

        before_haze = grade_module.measure(cropped)["haze_estimate"]
        graded = _cached_grade(grade_cache_dir, _raw(role), cropped=True)
        contrast_after = _local_contrast(_rgb01(graded))
        after_haze = grade_module.measure(graded)["haze_estimate"]

        # Clearly crisper within the ridge, clearly less veiled, clearly
        # more colourful -- but NOT maximal: an earlier 1.5x / -35% haze
        # target was met only by a grade that haloed every ridgeline (see
        # the guard below). Contrast is windowed on purpose: the spread of
        # the whole band also counts a bright glow painted beside a dark
        # ridge as "contrast", which is the defect, not the goal.
        assert contrast_after >= 1.25 * contrast_before, role
        assert after_haze <= before_haze * 0.82, role
        before_colour = grade_module.measure(cropped)["colourfulness"]
        after_colour = grade_module.measure(graded)["colourfulness"]
        assert after_colour >= before_colour + 12.0, role

    @pytest.mark.parametrize(
        "role", ["hazy_ridge_a", "hazy_ridge_b", "hazy_ridge_c", "hazy_ridge_d"]
    )
    def test_a_hazy_ridge_is_never_oversharpened_into_halos_and_blocks(self, grade_cache_dir, role):
        # Fine-detail energy (luma minus a 2px blur, at a fixed 1600px width)
        # in the far-ridge band. A natural grade roughly doubles it at most;
        # a haloed, posterised grade far exceeds this bound.
        cropped = _cached_crop(grade_cache_dir, _raw(role))
        graded = _cached_grade(grade_cache_dir, _raw(role), cropped=True)
        before = _fine_detail_energy(_rgb01(cropped))
        after = _fine_detail_energy(_rgb01(graded))
        assert after <= 2.5 * before, role

    @pytest.mark.parametrize("n", range(len(_ALL_STILLS)))
    def test_no_real_still_ever_reads_duller_than_the_source(self, grade_cache_dir, n):
        # "the source" is what grading itself sees -- the frame after
        # crop_lens_corners, matching production order (hud._load_still crops
        # before grading). A structural crop can legitimately change a
        # frame's own colourfulness by discarding real content; that is not
        # a grading regression and is outside what grading can guarantee.
        name = _ALL_STILLS[n]
        if not (_FIXTURE_DIR / name).exists():
            pytest.skip(f"still {n} not present")
        cropped = _cached_crop(grade_cache_dir, name)
        before = grade_module.measure(cropped)["colourfulness"]
        graded = _cached_grade(grade_cache_dir, name, cropped=True)
        after = grade_module.measure(graded)["colourfulness"]
        assert after >= before - 1.0, n

    def test_a_clear_dim_frame_is_lifted_only_a_little_not_chased_to_mid_grey(
        self, grade_cache_dir
    ):
        preview = _cached_preview(grade_cache_dir, _raw("dim_clear_frame"))
        before_luma = grade_module.measure(preview)["mean_luma"]

        graded = _cached_grade(grade_cache_dir, _raw("dim_clear_frame"), cropped=True)
        after_luma = grade_module.measure(graded)["mean_luma"]

        assert after_luma <= 0.40
        assert after_luma - before_luma <= grade_module._EXPOSURE_UNDEREXPOSED_MAX_DELTA + 0.02

    def test_an_arch_with_dark_straight_corners_is_not_cropped(self, grade_cache_dir):
        preview = _cached_preview(grade_cache_dir, _raw("arch_frame"))
        cropped = _cached_crop(grade_cache_dir, _raw("arch_frame"))

        assert cropped.size == preview.size

    def test_a_forest_with_dark_scene_corners_is_not_mistaken_for_vignetting(self, grade_cache_dir):
        # The dim, clear frame: dark, textured tree/riverbank corners read as a large
        # extent by _corner_vignette_extent -- far past any real lens-hood
        # margin -- and must NOT be cropped.
        preview = _cached_preview(grade_cache_dir, _raw("dim_clear_frame"))
        cropped = _cached_crop(grade_cache_dir, _raw("dim_clear_frame"))

        assert cropped.size == preview.size

    @pytest.mark.parametrize("role", ["vignette_a", "vignette_b"])
    def test_real_lens_vignetting_corners_are_still_cropped(self, grade_cache_dir, role):
        # Both vignette frames carry real lens-hood vignetting (small,
        # rounded, dark corner arcs) with extents comfortably inside the
        # real-vignette range, and must still crop.
        preview = _cached_preview(grade_cache_dir, _raw(role))
        cropped = _cached_crop(grade_cache_dir, _raw(role))

        assert cropped.size != preview.size, role


class TestDiskMemoizeClaims:
    """``_disk_memoize_image``'s cross-worker claim must survive a holder that
    dies mid-compute. When several xdist workers crash while holding claims, the
    ``finally`` that unlinks the claim never runs, and every worker waiting on
    those keys would otherwise sit out the full deadline and raise
    TimeoutError -- a few crashes become many failures."""

    @staticmethod
    def _tiny_image() -> Image.Image:
        return Image.new("RGB", (4, 4), (10, 20, 30))

    @staticmethod
    def _dead_pid() -> int:
        proc = subprocess.Popen(["true"])
        proc.wait()
        return proc.pid

    def test_claim_held_by_a_dead_process_is_reclaimed_and_recomputed(self, tmp_path):
        digest = hashlib.sha256(b"k").hexdigest()
        claim = tmp_path / f"{digest}.claim"
        claim.write_text(str(self._dead_pid()))
        calls: list[int] = []

        def compute() -> Image.Image:
            calls.append(1)
            return self._tiny_image()

        image = _disk_memoize_image(tmp_path, "k", compute)

        assert calls == [1]
        assert image.getpixel((0, 0)) == (10, 20, 30)
        assert not claim.exists()
        assert (tmp_path / f"{digest}.png").exists()

    def test_claim_held_by_a_live_process_is_waited_on(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys.modules[__name__], "_CLAIM_WAIT_SECONDS", 0.2)
        digest = hashlib.sha256(b"k").hexdigest()
        (tmp_path / f"{digest}.claim").write_text(str(os.getpid()))

        with pytest.raises(TimeoutError, match="'k'"):
            _disk_memoize_image(tmp_path, "k", self._tiny_image)

    def test_claim_with_no_pid_yet_is_treated_as_live(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys.modules[__name__], "_CLAIM_WAIT_SECONDS", 0.2)
        digest = hashlib.sha256(b"k").hexdigest()
        (tmp_path / f"{digest}.claim").write_text("")

        with pytest.raises(TimeoutError, match="'k'"):
            _disk_memoize_image(tmp_path, "k", self._tiny_image)

    def test_claim_records_the_holder_pid(self, tmp_path):
        digest = hashlib.sha256(b"k").hexdigest()
        seen: list[str] = []

        def compute() -> Image.Image:
            seen.append((tmp_path / f"{digest}.claim").read_text())
            return self._tiny_image()

        _disk_memoize_image(tmp_path, "k", compute)

        assert seen == [str(os.getpid())]
