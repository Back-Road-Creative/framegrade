"""``photos ingest --style`` bakes a creative look into the gallery JPEGs.

Ingest used to ship only the corrective base grade, so no style ever reached
the published site. Now an explicit ``--style`` (or the ``ingest_style`` config
default) opts in: the chosen style is baked over each cluster's base correction
from a SINGLE decode — scene profile and grade share it — exactly as
``photos develop`` does. With no style, ingest grades the cluster base verbatim,
byte-identical to before this feature. Blur and every other gate are untouched.
"""

import textwrap

import pytest
from PIL import Image

from gomoveshift import config as config_mod
from gomoveshift.cli.handlers import photos as photos_handler
from gomoveshift.cli.parser import _build_parser
from gomoveshift.pipeline.photos import develop as develop_mod
from gomoveshift.pipeline.photos.cull import CullFloors
from gomoveshift.pipeline.photos import ingest as ingest_mod
from gomoveshift.pipeline.photos.develop import DecodedPhoto
from gomoveshift.pipeline.photos.ingest import IngestReport, ingest
from gomoveshift.pipeline.photos.scene import compute_scene_profile
from gomoveshift.pipeline.photos.styles import apply_style

JOURNEY_MD = textwrap.dedent("""\
    ---
    title: Journey A
    photo_count: 1
    days:
    - day_number: 1
      date: '2025-01-01'
      photos:
      - location: Place A
        photos:
        - photo-001.jpg
    ---
    Body text
""")
DAY = dict(day_date="2025-01-01", location="Place A")
WARM = (200, 150, 100)


def _site(tmp_path):
    journey = tmp_path / "content" / "journeys" / "journey-a.md"
    journey.parent.mkdir(parents=True)
    journey.write_text(JOURNEY_MD)
    (tmp_path / "static" / "images" / "journeys" / "journey-a").mkdir(parents=True)
    return journey


def _card(tmp_path, colors):
    card = tmp_path / "card"
    card.mkdir()
    for name, color in colors.items():
        Image.new("RGB", (32, 24), color).save(card / name)
    return card


def _spy_develop(monkeypatch):
    calls = []
    real = develop_mod.develop

    def spy(source, params=None, cluster_index=0):
        calls.append((source, params, cluster_index))
        return real(source, params, cluster_index)

    monkeypatch.setattr(ingest_mod, "develop", spy)
    return calls


class TestLibraryStyleBake:
    @pytest.fixture(autouse=True)
    def _flat_frames_are_the_subject(self, monkeypatch):
        """These tests grade flat solid-colour JPEGs on purpose; the cull floor
        is not what they exercise."""
        monkeypatch.setattr(ingest_mod, "CULL_FLOORS", CullFloors(0.0, 1.0, 1.0))

    def test_no_style_grades_the_cluster_base_verbatim(self, tmp_path, monkeypatch):
        """Default path: develop gets the unmodified cluster params on a Path
        source (decoded inside develop, as before) — byte-identical to today."""
        journey = _site(tmp_path)
        card = _card(tmp_path, {"a.jpg": WARM})
        calls = _spy_develop(monkeypatch)
        base = develop_mod.compute_batch_corrections([card / "a.jpg"])[0][0]

        ingest(card, journey, **DAY)

        assert len(calls) == 1
        source, params, _ = calls[0]
        assert not isinstance(source, DecodedPhoto)  # a Path — develop decodes it
        assert params == base  # cluster base, unmodified

    def test_style_bakes_apply_style_over_cluster_base_from_one_decode(self, tmp_path, monkeypatch):
        journey = _site(tmp_path)
        card = _card(tmp_path, {"a.jpg": WARM})
        calls = _spy_develop(monkeypatch)
        base, cluster_index = develop_mod.compute_batch_corrections([card / "a.jpg"])[0]

        ingest(card, journey, style="vivid", style_strength=1.5, **DAY)

        assert len(calls) == 1
        source, params, ci = calls[0]
        # The grade reused the single decode — no second demosaic for the scene.
        assert isinstance(source, DecodedPhoto)
        scene = compute_scene_profile(source.rgb)
        assert params == apply_style("vivid", base, scene, 1.5)
        assert ci == cluster_index


def _parse(argv):
    return _build_parser().parse_args(argv)


def _argv(card, journey, *extra):
    return [
        "photos", "ingest", str(card),
        "--journey", str(journey), "--day", "2025-01-01", "--location", "Place A",
        *extra,
    ]  # fmt: skip


def _spy_ingest(monkeypatch):
    seen = {}

    def fake(*args, **kwargs):
        seen.update(kwargs)
        return IngestReport(ingested=["a.jpg"], location="Place A")

    monkeypatch.setattr(ingest_mod, "ingest", fake)
    return seen


class TestCliDoor:
    def test_threads_style_and_strength_into_ingest(self, tmp_path, monkeypatch):
        journey = _site(tmp_path)
        card = _card(tmp_path, {"a.jpg": WARM})
        seen = _spy_ingest(monkeypatch)

        argv = _argv(card, journey, "--style", "moody", "--style-strength", "2")
        assert photos_handler._cmd_photos_ingest(_parse(argv)) == 0
        assert seen["style"] == "moody"
        assert seen["style_strength"] == 2.0

    def test_rejects_unknown_style_before_ingest(self, tmp_path, monkeypatch):
        journey = _site(tmp_path)
        card = _card(tmp_path, {"a.jpg": WARM})
        monkeypatch.setattr(ingest_mod, "ingest", lambda *a, **k: pytest.fail("must not ingest"))

        assert (
            photos_handler._cmd_photos_ingest(_parse(_argv(card, journey, "--style", "bogus"))) == 1
        )

    def test_no_style_flag_passes_empty_style_through(self, tmp_path, monkeypatch):
        journey = _site(tmp_path)
        card = _card(tmp_path, {"a.jpg": WARM})
        seen = _spy_ingest(monkeypatch)

        assert photos_handler._cmd_photos_ingest(_parse(_argv(card, journey))) == 0
        assert seen["style"] == ""

    def test_config_ingest_style_default_is_used_when_flag_omitted(self, tmp_path, monkeypatch):
        journey = _site(tmp_path)
        card = _card(tmp_path, {"a.jpg": WARM})
        seen = _spy_ingest(monkeypatch)
        cfg = config_mod.get_config()
        cfg.photos.ingest_style = "golden"
        monkeypatch.setattr(config_mod, "get_config", lambda *a, **k: cfg)

        assert photos_handler._cmd_photos_ingest(_parse(_argv(card, journey))) == 0
        assert seen["style"] == "golden"
