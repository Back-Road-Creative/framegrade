"""``still_card_frame`` is the single recipe for what an inserted photo looks
like at a given point of its hold: full-bleed for a landscape-ish still, a
full-height foreground on a blurred shaded backdrop for a portrait/narrow
one, and a smooth, deterministic push-in across the hold either way."""

from __future__ import annotations

from PIL import Image

from gomoveshift.pipeline.route_story.hud import still_card_frame


def _solid(width: int, height: int, color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (width, height), color)


class TestFullBleedLandscape:
    def test_a_3x2_still_in_a_16x9_frame_is_full_bleed(self):
        still = _solid(300, 200, (200, 60, 10))

        frame = still_card_frame(still, 320, 180, 0.0)

        assert frame.size == (320, 180)
        for corner in [(0, 0), (319, 0), (0, 179), (319, 179)]:
            # No blurred backdrop and no (2, 9, 12, 105) shade under it -- a
            # corner is the still's own cover-fit pixel, not a darkened one.
            assert frame.getpixel(corner) == (200, 60, 10)

    def test_a_4x3_still_in_a_16x9_frame_is_full_bleed(self):
        still = _solid(400, 300, (30, 180, 90))

        frame = still_card_frame(still, 320, 180, 0.0)

        for corner in [(0, 0), (319, 0), (0, 179), (319, 179)]:
            assert frame.getpixel(corner) == (30, 180, 90)


class TestPortraitKeepsTheBackdrop:
    def test_a_narrow_still_fills_the_full_frame_height(self):
        still = _solid(200, 600, (240, 240, 10))

        frame = still_card_frame(still, 320, 180, 0.0)

        assert frame.size == (320, 180)
        center_x = 320 // 2
        # No 0.91 inset: the foreground reaches the very top and bottom rows.
        assert frame.getpixel((center_x, 0)) == (240, 240, 10)
        assert frame.getpixel((center_x, 179)) == (240, 240, 10)

    def test_a_narrow_still_still_shows_the_blurred_shaded_backdrop_at_the_sides(self):
        still = _solid(200, 600, (240, 240, 10))

        frame = still_card_frame(still, 320, 180, 0.0)

        # Off to the side of the centered foreground, the pixel is the
        # backdrop (blurred + shaded), not the foreground's own colour.
        assert frame.getpixel((2, 90)) != (240, 240, 10)


class TestPushIn:
    def test_scale_grows_monotonically_smoothly_and_deterministically(self, monkeypatch):
        still = _solid(400, 300, (10, 10, 10))
        boxes: list[tuple[float, float, float, float]] = []
        original_resize = Image.Image.resize

        def spy_resize(self, size, resample=None, box=None, **kwargs):
            if box is not None:
                boxes.append(box)
            return original_resize(self, size, resample, box=box, **kwargs)

        monkeypatch.setattr(Image.Image, "resize", spy_resize)

        fps = 30
        seconds = 2.0
        count = round(fps * seconds)
        for index in range(count):
            progress = index / (count - 1)
            still_card_frame(still, 320, 180, progress)

        assert len(boxes) == count
        widths = [box[2] - box[0] for box in boxes]

        # The crop box shrinks (never grows) as progress advances -- the
        # visible source region narrows, i.e. the delivered scale grows.
        assert all(widths[i] >= widths[i + 1] - 1e-6 for i in range(len(widths) - 1))
        # It actually moves: the last frame is a materially tighter crop
        # than the first.
        assert widths[-1] < widths[0]
        # No two consecutive frames are identical mid-hold.
        mid = len(widths) // 2
        assert widths[mid] != widths[mid + 1]
        # No jump between consecutive frames larger than a small bound.
        deltas = [abs(widths[i] - widths[i + 1]) / widths[0] for i in range(len(widths) - 1)]
        assert max(deltas) < 0.01

    def test_same_inputs_produce_byte_identical_frames(self):
        still = _solid(400, 300, (5, 90, 200))

        first = still_card_frame(still, 320, 180, 0.37)
        second = still_card_frame(still, 320, 180, 0.37)

        assert first.tobytes() == second.tobytes()


class TestPreparedStillIsNeverUpsampled:
    """The still is bounded once per photo; that bound must leave enough real
    pixels that even the fully pushed-in frame is a downscale -- a 4:3 photo
    contained in a 1.25x box would otherwise be stretched ~13% and look soft."""

    def _max_push_crop(self, prepared: Image.Image, width: int, height: int) -> float:
        from gomoveshift.pipeline.route_story.hud import STILL_PUSH_IN_MAX_SCALE

        ratio = width / height
        base_width = min(prepared.width, prepared.height * ratio)
        return base_width / STILL_PUSH_IN_MAX_SCALE

    def test_full_bleed_stills_keep_a_full_frame_of_pixels_at_full_push_in(self):
        from gomoveshift.pipeline.route_story.hud import prepare_still

        for size in [(6000, 4000), (4000, 3000), (4032, 3024)]:
            prepared = prepare_still(_solid(*size, (90, 90, 90)), 1920, 1080)
            assert self._max_push_crop(prepared, 1920, 1080) >= 1920, size

    def test_a_portrait_still_keeps_a_full_frame_height_at_full_push_in(self):
        from gomoveshift.pipeline.route_story.hud import STILL_PUSH_IN_MAX_SCALE, prepare_still

        prepared = prepare_still(_solid(4000, 6000, (90, 90, 90)), 1920, 1080)
        assert prepared.height / STILL_PUSH_IN_MAX_SCALE >= 1080

    def test_a_small_still_is_never_blown_up_here(self):
        from gomoveshift.pipeline.route_story.hud import prepare_still

        assert prepare_still(_solid(1000, 700, (90, 90, 90)), 1920, 1080).size == (1000, 700)

    def test_the_film_prepares_every_still_this_way(self, tmp_path, monkeypatch):
        from gomoveshift.pipeline.route_story import assembly as assembly_module

        seen = []
        real = assembly_module.prepare_still

        def spy(still, width, height):
            seen.append((still.size, width, height))
            return real(still, width, height)

        monkeypatch.setattr(assembly_module, "prepare_still", spy)
        monkeypatch.setattr(
            assembly_module, "_load_still", lambda path, source: _solid(4000, 3000, (9, 9, 9))
        )
        monkeypatch.setattr(assembly_module, "_save_frame", lambda image, frames, frame, **_: None)

        class _Photo:
            source = tmp_path / "x.jpg"

        assembly_module._append_photo(tmp_path, 0, _Photo(), 2, 1.0, 320, 180, mapped=True)
        assert seen == [((4000, 3000), 320, 180)]
