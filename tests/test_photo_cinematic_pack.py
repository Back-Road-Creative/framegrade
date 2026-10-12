"""The cinematic ``--look`` pack: five film/blockbuster grades in the box.

Presets are DATA built out of the knobs ``photos develop`` already has — no new
engine, no new flag — so what a hermetic test can prove is exactly four things:
every preset is legal on the REAL config validator and the real CLI range gates,
every one renders visibly apart from the natural grade AND from its four siblings
(the same ``MIN_RENDERED_DISTINCTNESS`` bar two auto-selected variants must clear
to both ship), each renders the same bytes twice, and each moves the frame in the
direction its name promises. Whether ``teal_orange`` is at the right STRENGTH on a
real NEF is a manual eyeball check on the reference card; no synthesized frame stands in for
that, which is why these assertions are directional and never magic pixel values.
"""

from itertools import combinations

import numpy as np
import pytest
from PIL import Image

import gomoveshift.config as config_module
from gomoveshift.cli.handlers import photos as photos_handler
from gomoveshift.cli.parser import _build_parser
from gomoveshift.config import Config, PhotosConfig
from gomoveshift.pipeline.photos.develop import MIN_RENDERED_DISTINCTNESS
from gomoveshift.pipeline.photos.looks import _CINEMATIC, BUILTIN_LOOKS

CINEMATIC = ("teal_orange", "bleach_bypass", "cross_process", "cinestill", "polaroid")


@pytest.fixture(autouse=True)
def _bare_config(monkeypatch):
    """Nothing configured: the presets have to come from the built-in table."""
    monkeypatch.setattr(photos_handler, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: Config(photos=PhotosConfig()))


def _frame(path):
    """A synthesized photo: a left-to-right luma ramp under three colour bands, so
    a shadow/highlight direction assertion has both ends of the tone range and the
    per-hue controls have something to grab."""
    ramp = np.tile(np.linspace(0.02, 0.98, 128), (96, 1))
    rgb = np.repeat(ramp[:, :, None], 3, axis=2)
    for band, channel in enumerate((0, 1, 2)):
        rgb[band * 32 : band * 32 + 32, :, channel] *= 1.2
    Image.fromarray(np.clip(rgb * 255, 0, 255).astype(np.uint8), "RGB").save(path, "JPEG")
    return path


def _develop(tmp_path, photo, tag, extra=""):
    """Develop ``photo`` through the REAL CLI path — every range gate, the curve
    validator and the look layer included — and return the exported master."""
    out = tmp_path / tag
    argv = f"photos develop {photo} --out {out} --styles natural {extra}".split()
    assert photos_handler.cmd_photos(_build_parser().parse_args(argv), None) == 0
    (exported,) = sorted(out.glob("*.jpg"))
    return exported


def _pixels(*args, **kwargs):
    return np.asarray(Image.open(_develop(*args, **kwargs)), dtype=np.int16)


def _distance(a, b):
    """The project's rendered-distinctness measure: mean absolute 8-bit levels."""
    return float(np.abs(a - b).mean())


def _tone_masks(natural):
    luma = natural.mean(axis=2)
    return luma <= np.percentile(luma, 25), luma >= np.percentile(luma, 75)


def test_every_cinematic_preset_is_legal_on_the_real_config_validator():
    """A preset that config would reject is a preset the operator cannot override."""
    assert tuple(_CINEMATIC) == CINEMATIC, "the pack ships all five, in order"
    assert PhotosConfig(looks=_CINEMATIC).looks == _CINEMATIC
    assert set(CINEMATIC) <= set(BUILTIN_LOOKS), "the pack must reach the public table"


def test_each_look_renders_apart_from_natural_and_from_every_sibling(tmp_path):
    """Five names that render as one look are four wasted names. ``--border none``
    holds the frame size steady so polaroid's matte cannot stand in for a grade."""
    photo = _frame(tmp_path / "IMG_0100.jpg")
    rendered = {"natural": _pixels(tmp_path, photo, "natural")}
    for name in CINEMATIC:
        rendered[name] = _pixels(tmp_path, photo, name, f"--look {name} --border none")
    for name in CINEMATIC:
        apart = _distance(rendered[name], rendered["natural"])
        assert apart >= MIN_RENDERED_DISTINCTNESS, f"{name} renders {apart:.2f} from natural"
    for a, b in combinations(CINEMATIC, 2):
        apart = _distance(rendered[a], rendered[b])
        assert apart >= MIN_RENDERED_DISTINCTNESS, f"{a} renders {apart:.2f} from {b}"


def test_a_look_renders_the_same_bytes_twice(tmp_path):
    """cinestill carries the grain, the one stage that could go non-deterministic."""
    photo = _frame(tmp_path / "IMG_0101.jpg")
    runs = [_develop(tmp_path, photo, f"run{i}", "--look cinestill") for i in (1, 2)]
    assert runs[0].read_bytes() == runs[1].read_bytes()


def test_teal_orange_cools_the_shadows_and_warms_the_highlights(tmp_path):
    photo = _frame(tmp_path / "IMG_0102.jpg")
    natural = _pixels(tmp_path, photo, "natural")
    look = _pixels(tmp_path, photo, "teal", "--look teal_orange")
    shadows, highlights = _tone_masks(natural)
    warmth = lambda img, m: float((img[:, :, 0] - img[:, :, 2])[m].mean())  # noqa: E731
    assert warmth(look, shadows) < warmth(natural, shadows), "shadows must go teal"
    assert warmth(look, highlights) > warmth(natural, highlights), "highlights must go orange"


def test_bleach_bypass_drains_the_colour_and_raises_the_contrast(tmp_path):
    photo = _frame(tmp_path / "IMG_0103.jpg")
    natural = _pixels(tmp_path, photo, "natural")
    look = _pixels(tmp_path, photo, "bleach", "--look bleach_bypass")
    chroma = lambda img: float((img.max(axis=2) - img.min(axis=2)).mean())  # noqa: E731
    assert chroma(look) < 0.7 * chroma(natural), "the silver-retention look is desaturated"
    assert look.mean(axis=2).std() > natural.mean(axis=2).std(), "and harder, not just greyer"


def test_cross_process_greens_the_shadows_and_yellows_the_highlights(tmp_path):
    """Built from three INDEPENDENT channel curves, so the two ends move apart —
    a split-tone pair could not push shadows and highlights on different axes."""
    photo = _frame(tmp_path / "IMG_0104.jpg")
    natural = _pixels(tmp_path, photo, "natural")
    look = _pixels(tmp_path, photo, "xpro", "--look cross_process")
    shadows, highlights = _tone_masks(natural)
    green = lambda img, m: float((img[:, :, 1] - img[:, :, 0])[m].mean())  # noqa: E731
    yellow = lambda img, m: float((img[:, :, :2].mean(axis=2) - img[:, :, 2])[m].mean())  # noqa: E731
    assert green(look, shadows) > green(natural, shadows), "shadows must go green-cyan"
    assert yellow(look, highlights) > yellow(natural, highlights), "highlights must go yellow"


def test_polaroid_wears_the_white_matte_the_look_asked_for(tmp_path):
    """The one preset carrying a presentational knob: ``border`` has to resolve
    through the look layer, and resolve to WHITE rather than any old matte."""
    photo = _frame(tmp_path / "IMG_0105.jpg")
    matted = _pixels(tmp_path, photo, "matted", "--look polaroid")
    bare = _pixels(tmp_path, photo, "bare", "--look polaroid --border none")
    assert matted.shape[0] > bare.shape[0] and matted.shape[1] > bare.shape[1]
    assert matted[0, 0].min() >= 250, "a white matte, not black or a filmstrip"
