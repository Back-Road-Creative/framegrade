"""Route Story thumbnail and metadata outputs beside the receipt.

A full-preset export produces a thumbnail (or records why it couldn't) and,
given --metadata, a JSON metadata sidecar next to the receipt. A social
(9:16) cut never gets a thumbnail -- there's no long-form upload for it to
sit beside.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
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
        "--width", "640", "--height", "360", "--fps", "24",
        *extra,
    ]  # fmt: skip


def _patch_render_pipeline(monkeypatch, *, delivery_issues=None):
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
        lambda path, *, width, height, fps: {
            "issues": delivery_issues or [],
            "video": {"frames": 1},
            "duration_s": 12.5,
        },
    )


class TestThumbnailOnAFullPresetExport:
    def _thumbnail_calls(self, monkeypatch, *, returns: Path | None):
        calls: list[dict] = []

        def fake_generate(video_path, title, subtitle, output_dir, duration, **kwargs):
            calls.append(
                dict(
                    video_path=Path(video_path),
                    title=title,
                    subtitle=subtitle,
                    output_dir=Path(output_dir),
                    duration=duration,
                )
            )
            if returns is None:
                return None
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            made = output_dir / "candidate.jpg"
            made.write_bytes(b"thumbnail bytes")
            return made

        monkeypatch.setattr(film_module, "generate_lf_thumbnail", fake_generate)
        return calls

    def test_a_generated_thumbnail_is_moved_into_place_and_recorded(self, tmp_path, monkeypatch):
        _patch_render_pipeline(monkeypatch)
        calls = self._thumbnail_calls(monkeypatch, returns=True)
        thumbnail_path = tmp_path / "thumb.jpg"

        code = film_module.main(_main_argv(tmp_path, "--thumbnail", str(thumbnail_path)))

        assert code == 0
        assert len(calls) == 1
        assert calls[0]["duration"] == 12.5
        assert thumbnail_path.is_file()
        assert thumbnail_path.read_bytes() == b"thumbnail bytes"
        receipt = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
        assert receipt["thumbnail"]["path"] == "thumb.jpg"
        import hashlib

        assert receipt["thumbnail"]["sha256"] == hashlib.sha256(b"thumbnail bytes").hexdigest()

    def test_no_thumbnail_from_the_generator_records_a_failed_status_without_failing_the_run(
        self, tmp_path, monkeypatch
    ):
        _patch_render_pipeline(monkeypatch)
        self._thumbnail_calls(monkeypatch, returns=None)
        thumbnail_path = tmp_path / "thumb.jpg"

        code = film_module.main(_main_argv(tmp_path, "--thumbnail", str(thumbnail_path)))

        assert code == 0
        assert not thumbnail_path.exists()
        receipt = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
        assert receipt["thumbnail"] == {"status": "failed"}

    def test_a_social_preset_export_never_calls_the_thumbnail_generator(
        self, tmp_path, monkeypatch
    ):
        _patch_render_pipeline(monkeypatch)
        calls = self._thumbnail_calls(monkeypatch, returns=True)
        thumbnail_path = tmp_path / "thumb.jpg"

        code = film_module.main(
            _main_argv(tmp_path, "--preset", "social", "--thumbnail", str(thumbnail_path))
        )

        assert code == 0
        assert calls == []
        assert not thumbnail_path.exists()
        receipt = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
        assert "thumbnail" not in receipt

    def test_no_thumbnail_flag_means_no_generator_call_and_no_receipt_key(
        self, tmp_path, monkeypatch
    ):
        _patch_render_pipeline(monkeypatch)
        calls = self._thumbnail_calls(monkeypatch, returns=True)

        code = film_module.main(_main_argv(tmp_path))

        assert code == 0
        assert calls == []
        receipt = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
        assert "thumbnail" not in receipt


@dataclass(frozen=True)
class _StubMetadata:
    """Same shape as route_story.metadata.RouteStoryMetadata -- a real
    dataclass, since film.py serializes it with dataclasses.asdict."""

    title: str
    description: str = "A route"
    tags: tuple = ("route story",)
    recording_date: str = "2026-09-16"
    format: str = "route_story"
    playlist_hint: str = "route_story"


def _install_metadata_stub(monkeypatch, *, build=None, check=None):
    import types

    stub = types.ModuleType("gomoveshift.pipeline.route_story.metadata")

    class DuplicateTitleError(Exception):
        pass

    class LedgerUnreadableError(Exception):
        pass

    stub.DuplicateTitleError = DuplicateTitleError
    stub.LedgerUnreadableError = LedgerUnreadableError
    stub.build_route_story_metadata = build or (
        lambda receipt, *, recorded_on, title=None, credits=(), activity=None: _StubMetadata(
            title=title or "Driving Test Route"
        )
    )
    stub.check_title_against_ledger = check or (lambda meta: None)
    monkeypatch.setitem(sys.modules, "gomoveshift.pipeline.route_story.metadata", stub)
    return stub


class TestMetadataSidecarOnAFilmExport:
    def test_metadata_is_built_written_and_recorded_on_the_receipt(self, tmp_path, monkeypatch):
        _patch_render_pipeline(monkeypatch)
        seen_credits = []

        def build(receipt, *, recorded_on, title=None, credits=(), activity=None):
            seen_credits.append(list(credits))
            return _StubMetadata(title=title or "Driving Test Route")

        _install_metadata_stub(monkeypatch, build=build)
        metadata_path = tmp_path / "metadata.json"

        code = film_module.main(
            _main_argv(
                tmp_path,
                "--metadata", str(metadata_path),
                "--credit", "Aerial imagery Co",
                "--credit", "Local historian",
            )
        )  # fmt: skip

        assert code == 0
        assert seen_credits == [["Aerial imagery Co", "Local historian"]]
        written = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert written["title"] == "Driving Test Route"
        assert written["format"] == "route_story"
        receipt = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
        assert receipt["metadata"] == {
            "path": "metadata.json",
            "title": "Driving Test Route",
            "format": "route_story",
        }

    def test_no_metadata_flag_means_no_receipt_key_and_no_file(self, tmp_path, monkeypatch):
        _patch_render_pipeline(monkeypatch)
        _install_metadata_stub(monkeypatch)

        code = film_module.main(_main_argv(tmp_path))

        assert code == 0
        receipt = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
        assert "metadata" not in receipt

    def test_a_duplicate_title_fails_the_run_before_anything_publishes(
        self, tmp_path, monkeypatch, capsys
    ):
        _patch_render_pipeline(monkeypatch)

        def check(meta):
            stub = sys.modules["gomoveshift.pipeline.route_story.metadata"]
            raise stub.DuplicateTitleError(f"title already published: {meta.title!r}")

        _install_metadata_stub(monkeypatch, check=check)
        metadata_path = tmp_path / "metadata.json"

        with pytest.raises(SystemExit):
            film_module.main(_main_argv(tmp_path, "--metadata", str(metadata_path)))

        assert "already published" in capsys.readouterr().err
        assert not metadata_path.exists()

    def test_the_builder_being_unavailable_is_recorded_not_fatal(self, tmp_path, monkeypatch):
        # A None entry in sys.modules makes the import raise ImportError, the
        # same shape as a build without the metadata module -- film.py must degrade
        # gracefully rather than crash the whole export.
        monkeypatch.setitem(sys.modules, "gomoveshift.pipeline.route_story.metadata", None)
        _patch_render_pipeline(monkeypatch)
        metadata_path = tmp_path / "metadata.json"

        code = film_module.main(_main_argv(tmp_path, "--metadata", str(metadata_path)))

        assert code == 0
        assert not metadata_path.exists()
        receipt = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
        assert receipt["metadata"] == {"status": "builder-unavailable"}


class TestReceiptRecordsTheResolvedHudCaps:
    def test_display_filter_hud_matches_the_resolved_activity_profile(self, tmp_path, monkeypatch):
        _patch_render_pipeline(monkeypatch)

        code = film_module.main(_main_argv(tmp_path, "--activity", "cycling"))

        assert code == 0
        receipt = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
        assert receipt["display_filter"]["activity"] == "cycling"
        assert receipt["display_filter"]["hud"] == {
            "speed_cap_mps": 25.0,
            "grade_cap_percent": 20.0,
        }
