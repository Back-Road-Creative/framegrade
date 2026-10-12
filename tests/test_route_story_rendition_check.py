"""Receipt ``preset.rendition_check``: the social cut's rendition of record.

The route-story social cut is delivered by ``encode_final`` -- there is no
second, rendition-specific encode. What this adds is a record-only check,
read from the SAME ``check_delivery`` measurement the receipt already makes,
that the delivered file actually matches the ``route_story_short`` rendition
of record (audio present, sized right, under the ceiling). A social cut
missing its audio stream is a delivery issue, not a quiet receipt footnote --
that is the contract test for the social-cut delivery decision.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from gomoveshift.pipeline.route_story import film as film_module


def _main_argv(tmp_path: Path, *extra: str) -> list[str]:
    gpx = tmp_path / "track.gpx"
    gpx.write_text(
        """<gpx xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg>
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
    return [
        "--gpx", str(gpx), "--recorded-on", "2026-09-16", "--photos-dir", str(photos),
        "--national-raster", str(national), "--national-credit", "Test overview",
        "--local-raster", str(local), "--local-credit", "Test local",
        "--local-bounds", "-80.1", "34.9", "-79.9", "35.1",
        "--output", str(tmp_path / "film.mp4"), "--receipt", str(tmp_path / "receipt.json"),
        "--fps", "24",
        *extra,
    ]  # fmt: skip


def _patch_render_pipeline(monkeypatch, *, delivery: dict):
    monkeypatch.setattr(
        film_module,
        "render_route_story_film",
        lambda **kwargs: Path(kwargs["output"]).write_bytes(b"fake film bytes"),
    )
    monkeypatch.setattr(
        film_module, "encode_final", lambda s, d: Path(d).write_bytes(Path(s).read_bytes())
    )
    monkeypatch.setattr(
        film_module,
        "check_delivery",
        lambda path, *, width, height, fps: dict(delivery),
    )


def _receipt(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))


def test_a_healthy_social_delivery_gets_an_all_true_rendition_check(tmp_path, monkeypatch):
    _patch_render_pipeline(
        monkeypatch,
        delivery={
            "issues": [],
            "video": {"frames": 1, "width": 1080, "height": 1920},
            "audio": {"codec": "aac", "sample_rate": 48000, "channels": 2},
            "duration_s": 45.0,
        },
    )

    code = film_module.main(_main_argv(tmp_path, "--preset", "social"))

    assert code == 0
    receipt = _receipt(tmp_path)
    assert receipt["preset"]["rendition"] == "route_story_short"
    assert receipt["preset"]["rendition_check"] == {
        "audio_stream": True,
        "size_ok": True,
        "within_ceiling": True,
    }


def test_a_social_delivery_missing_audio_fails_the_run_and_records_it(tmp_path, monkeypatch):
    _patch_render_pipeline(
        monkeypatch,
        delivery={
            "issues": ["expected one audio stream, found 0"],
            "video": {"frames": 1, "width": 1080, "height": 1920},
            "duration_s": 45.0,
        },
    )

    code = film_module.main(_main_argv(tmp_path, "--preset", "social"))

    assert code == 2
    receipt = _receipt(tmp_path)
    assert receipt["preset"]["rendition_check"]["audio_stream"] is False


def test_the_full_preset_carries_no_rendition_check(tmp_path, monkeypatch):
    _patch_render_pipeline(
        monkeypatch,
        delivery={
            "issues": [],
            "video": {"frames": 1, "width": 1920, "height": 1080},
            "audio": {"codec": "aac", "sample_rate": 48000, "channels": 2},
            "duration_s": 120.0,
        },
    )

    code = film_module.main(_main_argv(tmp_path))

    assert code == 0
    receipt = _receipt(tmp_path)
    assert receipt["preset"]["rendition"] is None
    assert "rendition_check" not in receipt["preset"]
