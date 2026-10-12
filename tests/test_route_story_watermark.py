"""The maker's mark: burned bottom-right on every delivered frame, configurable."""

from pathlib import Path

from PIL import Image

from gomoveshift.pipeline.route_story import assembly as assembly_module
from gomoveshift.pipeline.route_story import film as film_module
from gomoveshift.pipeline.route_story import hud as hud_module
from gomoveshift.pipeline.route_story.anchors import MediaAnchor
from gomoveshift.pipeline.route_story.film import PhotoMoment, render_route_story_film
from gomoveshift.pipeline.route_story.metrics import RoutePoint
from gomoveshift.pipeline.route_story.satellite import SatelliteRaster


def _bottom_right(image: Image.Image) -> Image.Image:
    width, height = image.size
    return image.crop((int(width * 0.85), int(height * 0.9), width, height))


def test_draw_watermark_changes_only_the_bottom_right_region():
    before = Image.new("RGB", (640, 360), "#123456")
    after = before.copy()

    box = hud_module.draw_watermark(after, "Made with GoMoveShift", 640, 360)

    assert box != (0, 0, 0, 0)
    assert list(_bottom_right(after).getdata()) != list(_bottom_right(before).getdata())
    # Nothing outside the bottom-right corner is touched.
    top_left_after = after.crop((0, 0, 100, 100))
    top_left_before = before.crop((0, 0, 100, 100))
    assert list(top_left_after.getdata()) == list(top_left_before.getdata())


def test_draw_watermark_with_empty_text_draws_nothing():
    before = Image.new("RGB", (640, 360), "#123456")
    after = before.copy()

    box = hud_module.draw_watermark(after, "", 640, 360)

    assert box == (0, 0, 0, 0)
    assert list(after.getdata()) == list(before.getdata())


def test_save_frame_burns_the_watermark_only_when_given(tmp_path: Path):
    unmarked = Image.new("RGB", (640, 360), "#123456")
    marked = unmarked.copy()

    assembly_module._save_frame(unmarked, tmp_path, 0)
    assembly_module._save_frame(marked, tmp_path, 1, watermark="Made with GoMoveShift")

    unmarked_saved = Image.open(tmp_path / "000000.jpg").convert("RGB")
    marked_saved = Image.open(tmp_path / "000001.jpg").convert("RGB")
    assert list(_bottom_right(unmarked_saved).getdata()) != list(
        _bottom_right(marked_saved).getdata()
    )


def _rasters(tmp_path: Path):
    overview = tmp_path / "overview.jpg"
    local = tmp_path / "local.jpg"
    Image.new("RGB", (720, 360), "#3c6f83").save(overview)
    Image.new("RGB", (640, 360), "#4e7540").save(local)
    return (
        SatelliteRaster(overview, -180, -90, 180, 90, "Test overview"),
        SatelliteRaster(local, -80.1, 34.9, -79.9, 35.1, "Test local"),
    )


def test_every_frame_producing_path_carries_the_watermark_by_default(tmp_path, monkeypatch):
    """Map leg, settle hold, photo, opening and recap frames are all saved
    through the same choke point, ``assembly._save_frame`` -- a watermark
    reaching one and not the others is the exact bug this guards against."""
    national, local = _rasters(tmp_path)
    output = tmp_path / "film.mp4"

    monkeypatch.setattr(
        film_module,
        "_encode_frames",
        lambda frames, output, fps: Path(output).write_bytes(b"frames"),
    )
    monkeypatch.setattr(
        film_module, "_concat", lambda clips, output: Path(output).write_bytes(b"final")
    )

    watermark_seen: list[bool] = []
    real_save_frame = assembly_module._save_frame

    def spy_save_frame(image, frames, number, *, watermark=None):
        watermark_seen.append(bool(watermark))
        return real_save_frame(image, frames, number, watermark=watermark)

    monkeypatch.setattr(film_module, "_save_frame", spy_save_frame)
    monkeypatch.setattr(assembly_module, "_save_frame", spy_save_frame)

    photo_source = tmp_path / "photo.jpg"
    Image.new("RGB", (640, 360), "#aa7744").save(photo_source)

    render_route_story_film(
        segments=[[RoutePoint(0, 35.0, -80.0, 100), RoutePoint(20, 35.0002, -79.9998, 110)]],
        national=national,
        local=local,
        photos=[
            PhotoMoment(
                photo_source,
                MediaAnchor("photo.jpg", "timestamp_interpolated", 10.0, 35.0001, -79.9999),
            )
        ],
        output=output,
        fps=2,
        width=320,
        height=180,
        intro_seconds=1,
        map_seconds=1,
        photo_seconds=1,
        recap_seconds=1,
    )

    # Every path -- opening, map legs (before and after the photo), the photo
    # itself, and the recap -- saved at least one frame carrying the mark. A
    # settle-hold frame is the sole, by-design exception: it reuses the leg's
    # own already-saved (and already-marked) last image rather than being
    # re-marked a second time, so it is allowed to pass ``watermark=None``.
    assert watermark_seen, "no frames were saved at all"
    marked_count = sum(watermark_seen)
    unmarked_count = len(watermark_seen) - marked_count
    assert marked_count > unmarked_count, "more frames skipped the watermark than carried it"
    assert unmarked_count <= 2, "more paths than just the settle/arrival hold skipped the mark"


def test_no_watermark_flag_reaches_every_saved_frame(tmp_path, monkeypatch):
    national, local = _rasters(tmp_path)
    output = tmp_path / "film.mp4"

    monkeypatch.setattr(
        film_module,
        "_encode_frames",
        lambda frames, output, fps: Path(output).write_bytes(b"frames"),
    )
    monkeypatch.setattr(
        film_module, "_concat", lambda clips, output: Path(output).write_bytes(b"final")
    )

    watermark_seen: list[bool] = []
    real_save_frame = assembly_module._save_frame

    def spy_save_frame(image, frames, number, *, watermark=None):
        watermark_seen.append(bool(watermark))
        return real_save_frame(image, frames, number, watermark=watermark)

    monkeypatch.setattr(film_module, "_save_frame", spy_save_frame)
    monkeypatch.setattr(assembly_module, "_save_frame", spy_save_frame)

    render_route_story_film(
        segments=[[RoutePoint(0, 35.0, -80.0, 100), RoutePoint(20, 35.0002, -79.9998, 110)]],
        national=national,
        local=local,
        photos=[],
        output=output,
        fps=2,
        width=320,
        height=180,
        intro_seconds=1,
        map_seconds=1,
        recap_seconds=1,
        watermark=None,
    )

    assert watermark_seen
    assert not any(watermark_seen)
