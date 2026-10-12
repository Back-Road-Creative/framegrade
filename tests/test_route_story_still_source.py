"""A film's photo holds can show the camera's own rendering instead of the RAW developer's."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from gomoveshift.pipeline.route_story import assembly as assembly_module
from gomoveshift.pipeline.route_story import film as film_module
from gomoveshift.pipeline.route_story import hud as hud_module


def _jpeg(size, colour, *, orientation: int | None = None) -> bytes:
    image = Image.new("RGB", size, colour)
    buffer = io.BytesIO()
    if orientation is None:
        image.save(buffer, "JPEG", quality=95)
    else:
        exif = Image.Exif()
        exif[274] = orientation
        image.save(buffer, "JPEG", quality=95, exif=exif)
    return buffer.getvalue()


def _raw(tmp_path: Path) -> Path:
    path = tmp_path / "frame.nef"
    path.write_bytes(b"not an image pillow can open")
    return path


def _developed_red(monkeypatch) -> None:
    decoded = np.zeros((40, 60, 3), dtype=np.uint8)
    decoded[..., 0] = 255
    monkeypatch.setattr(hud_module, "develop", lambda path: SimpleNamespace(rgb=decoded))


def test_the_default_still_is_the_developed_raw(tmp_path, monkeypatch):
    _developed_red(monkeypatch)
    monkeypatch.setattr(hud_module, "_camera_preview", lambda path: pytest.fail("not asked for"))

    still = hud_module._load_still(_raw(tmp_path))

    assert still.getpixel((5, 5)) == (255, 0, 0)


def test_camera_source_shows_the_embedded_camera_rendering(tmp_path, monkeypatch):
    _developed_red(monkeypatch)
    green = Image.open(io.BytesIO(_jpeg((3000, 2000), (20, 200, 40))))
    monkeypatch.setattr(hud_module, "_camera_preview", lambda path: green)

    still = hud_module._load_still(_raw(tmp_path), source="camera")

    assert still.size == (3000, 2000)
    red, green_channel, _ = still.getpixel((5, 5))
    assert green_channel > 150 and red < 80


def test_a_raw_with_no_usable_preview_falls_back_to_the_developer(tmp_path, monkeypatch, caplog):
    _developed_red(monkeypatch)
    monkeypatch.setattr(hud_module, "_camera_preview", lambda path: None)

    with caplog.at_level("WARNING"):
        still = hud_module._load_still(_raw(tmp_path), source="camera")

    assert still.getpixel((5, 5)) == (255, 0, 0)
    assert "no usable camera preview" in caplog.text


def test_a_thumbnail_sized_preview_is_not_usable(monkeypatch, tmp_path):
    thumb = SimpleNamespace(format="jpeg", data=_jpeg((160, 120), (20, 200, 40)))
    monkeypatch.setattr(hud_module, "_embedded_thumb", lambda path: thumb)

    assert hud_module._camera_preview(_raw(tmp_path)) is None


def test_a_full_size_preview_is_turned_upright(monkeypatch, tmp_path):
    thumb = SimpleNamespace(format="jpeg", data=_jpeg((3000, 2000), (20, 200, 40), orientation=6))
    monkeypatch.setattr(hud_module, "_embedded_thumb", lambda path: thumb)

    preview = hud_module._camera_preview(_raw(tmp_path))

    assert preview is not None and preview.size == (2000, 3000)


def test_a_preview_with_no_exif_orientation_follows_the_raw_flip(monkeypatch, tmp_path):
    thumb = SimpleNamespace(format="jpeg", data=_jpeg((3000, 2000), (20, 200, 40)), flip=6)
    monkeypatch.setattr(hud_module, "_embedded_thumb", lambda path: thumb)

    preview = hud_module._camera_preview(_raw(tmp_path))

    assert preview is not None and preview.size == (2000, 3000)


def test_an_ordinary_jpeg_is_unaffected_by_the_source(tmp_path):
    path = tmp_path / "phone.jpg"
    Image.new("RGB", (64, 48), (10, 20, 200)).save(path)

    assert (
        hud_module._load_still(path, source="camera").tobytes()
        == hud_module._load_still(path).tobytes()
    )


def test_an_unknown_still_source_is_refused(tmp_path):
    with pytest.raises(ValueError, match="still source"):
        hud_module._load_still(_raw(tmp_path), source="vivid")


def test_the_photo_hold_passes_the_source_to_the_loader(tmp_path, monkeypatch):
    seen = {}

    def fake_load(path, *, source="developed"):
        seen["source"] = source
        return Image.new("RGB", (64, 48), (10, 20, 200))

    monkeypatch.setattr(assembly_module, "_load_still", fake_load)
    photo = SimpleNamespace(source=tmp_path / "frame.nef")

    assembly_module._append_photo(
        tmp_path, 0, photo, 2, 1.0, 64, 36, mapped=True, still_source="camera"
    )

    assert seen["source"] == "camera"


def _run_main(tmp_path: Path, monkeypatch, *extra: str) -> tuple[dict, dict]:
    gpx = tmp_path / "day.gpx"
    gpx.write_text(
        """<gpx><trk><trkseg>
        <trkpt lat="35" lon="-80"><ele>100</ele><time>2026-09-16T12:00:00Z</time></trkpt>
        <trkpt lat="35.01" lon="-79.99"><ele>110</ele><time>2026-09-16T12:00:10Z</time></trkpt>
        </trkseg></trk></gpx>""",
        encoding="utf-8",
    )
    photos = tmp_path / "photos"
    photos.mkdir()
    national, local = tmp_path / "national.jpg", tmp_path / "local.jpg"
    Image.new("RGB", (64, 32), "#3c6f83").save(national)
    Image.new("RGB", (64, 32), "#4e7540").save(local)
    receipt = tmp_path / "receipt.json"
    rendered: dict = {}

    def fake_render(**kwargs):
        rendered.update(kwargs)
        Path(kwargs["output"]).write_bytes(b"fake film bytes")

    def fake_encode_final(source, destination):
        Path(destination).write_bytes(Path(source).read_bytes())

    monkeypatch.setattr(film_module, "render_route_story_film", fake_render)
    monkeypatch.setattr(film_module, "encode_final", fake_encode_final)
    argv = [
        "--gpx", str(gpx), "--recorded-on", "2026-09-16", "--photos-dir", str(photos),
        "--national-raster", str(national), "--national-credit", "Test overview",
        "--local-raster", str(local), "--local-credit", "Test local",
        "--local-bounds", "-80.1", "34.9", "-79.9", "35.1",
        "--output", str(tmp_path / "film.mp4"), "--receipt", str(receipt),
        *extra,
    ]  # fmt: skip
    assert film_module.main(argv) == 0
    return rendered, json.loads(receipt.read_text(encoding="utf-8"))


def test_the_film_develops_its_stills_unless_told_otherwise(tmp_path, monkeypatch):
    rendered, receipt = _run_main(tmp_path, monkeypatch)

    assert rendered["still_source"] == "developed"
    assert receipt["export"]["still_source"] == "developed"


def test_the_film_can_be_told_to_use_the_camera_rendering(tmp_path, monkeypatch):
    rendered, receipt = _run_main(tmp_path, monkeypatch, "--still-source", "camera")

    assert rendered["still_source"] == "camera"
    assert receipt["export"]["still_source"] == "camera"


def test_the_command_line_refuses_an_unknown_still_source(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        _run_main(tmp_path, monkeypatch, "--still-source", "vivid")
