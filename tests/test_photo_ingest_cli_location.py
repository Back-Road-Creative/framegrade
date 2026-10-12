"""``photos ingest --location`` is optional so EXIF GPS can elect the label.

The library already resolves operator-label > EXIF > fail-closed; these cover
the CLI door in front of it: argparse must let a location-less run through, and
must not relax the requirement anywhere else. The geocoder is always stubbed —
no test here touches the network.
"""

import textwrap

import numpy as np
import pytest

from gomoveshift.cli.handlers import photos as photos_handler
from gomoveshift.cli.parser import _build_parser
from gomoveshift.pipeline.photos import ingest as ingest_mod
from gomoveshift.pipeline.photos.develop import DevelopedPhoto
from gomoveshift.pipeline.photos.exif_gps import NO_EXIF_GPS, RESOLVED, PhotoLocation

JOURNEY_MD = textwrap.dedent("""\
    ---
    title: Journey A
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


def _parse(argv):
    return _build_parser().parse_args(argv)


def _site(tmp_path):
    journey = tmp_path / "content" / "journeys" / "journey-a.md"
    journey.parent.mkdir(parents=True)
    journey.write_text(JOURNEY_MD)
    (tmp_path / "static" / "images" / "journeys" / "journey-a").mkdir(parents=True)
    card = tmp_path / "card"
    card.mkdir()
    (card / "a.DNG").write_bytes(b"raw bytes")
    return journey, card


def _stub_develop(monkeypatch):
    """Decode/grade is develop.py's contract; bind a stand-in for the CLI path."""
    real_ingest = ingest_mod.ingest

    def _developed(path):
        return DevelopedPhoto(
            rgb=np.random.default_rng(7).integers(0, 256, (4, 4, 3), dtype=np.uint8),
            path=path,
            params=None,
            cluster_index=0,
        )

    def _ingest(*args, **kwargs):
        return real_ingest(*args, develop_fn=_developed, **kwargs)

    monkeypatch.setattr(ingest_mod, "ingest", _ingest)


def _stub_geocode(monkeypatch, result):
    monkeypatch.setattr(ingest_mod, "derive_location_from_photos", lambda *a, **k: result)


def _ingest_argv(card, journey, *extra):
    return ["photos", "ingest", str(card), "--journey", str(journey), "--day", "2025-01-01", *extra]


class TestExifMaySupplyTheLocation:
    def test_run_without_location_resolves_from_exif_and_persists_it(
        self, tmp_path, monkeypatch, capsys
    ):
        journey, card = _site(tmp_path)
        _stub_develop(monkeypatch)
        _stub_geocode(monkeypatch, PhotoLocation("County Clare, Ireland", RESOLVED))

        args = _parse(_ingest_argv(card, journey))
        rc = photos_handler._cmd_photos_ingest(args)

        assert rc == 0
        # Persisted once into the journey — the operator can see which label won.
        assert "location: County Clare, Ireland" in journey.read_text()
        assert "County Clare, Ireland" in capsys.readouterr().out

    def test_explicit_location_beats_exif_and_is_never_geocoded_over(self, tmp_path, monkeypatch):
        journey, card = _site(tmp_path)
        _stub_develop(monkeypatch)
        monkeypatch.setattr(
            ingest_mod,
            "derive_location_from_photos",
            lambda *a, **k: pytest.fail("an operator label must never be geocoded over"),
        )

        args = _parse(_ingest_argv(card, journey, "--location", "Place B"))

        assert photos_handler._cmd_photos_ingest(args) == 0
        assert "location: Place B" in journey.read_text()

    def test_no_exif_and_no_flag_fails_loudly_and_never_invents_a_label(
        self, tmp_path, monkeypatch, capsys
    ):
        journey, card = _site(tmp_path)
        _stub_develop(monkeypatch)
        _stub_geocode(monkeypatch, PhotoLocation(reason=NO_EXIF_GPS))

        args = _parse(_ingest_argv(card, journey))
        rc = photos_handler._cmd_photos_ingest(args)

        assert rc == 1
        err = capsys.readouterr().err
        assert NO_EXIF_GPS in err and "--location" in err
        # Fail-closed leaves the journey byte-identical — no fabricated label.
        assert journey.read_text() == JOURNEY_MD


class TestNoCollateralDamage:
    """``--location`` is declared per-subparser, not shared — keep it that way."""

    def test_ingest_still_requires_journey_and_day(self, tmp_path):
        _, card = _site(tmp_path)
        for missing in (
            ["photos", "ingest", str(card), "--day", "2025-01-01"],
            ["photos", "ingest", str(card), "--journey", "journey-a"],
        ):
            with pytest.raises(SystemExit):
                _parse(missing)

    def test_other_subcommands_keep_their_own_location_surface(self, tmp_path):
        video = tmp_path / "v.mp4"
        video.touch()
        # pipeline: optional, GPS-derivable. drive: optional, empty default.
        assert _parse(["pipeline", str(video), "--publish", "mon"]).location is None
        assert _parse(["drive", str(tmp_path)]).location == ""
