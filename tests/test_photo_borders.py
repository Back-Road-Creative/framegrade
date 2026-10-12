"""Presentational borders/frames for ``photos develop`` exports (``--border``).

Borders are a pure post-develop image op applied at the CLI EXPORT step only:
``_add_border`` pads the finished frame and ``export_jpeg`` writes it. ``none``
(the default) is an EXACT no-op, so the default export stays byte-identical; the
contact-sheet panels are never bordered (a review artifact). An invalid style is
refused at the argparse/config gate before any decode.
"""

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

import gomoveshift.config as config_module
from gomoveshift.cli.handlers import photos as photos_handler
from gomoveshift.cli.parser import _build_parser
from gomoveshift.config import Config, PhotosConfig
from gomoveshift.pipeline.photos import develop as develop_mod


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr(photos_handler, "configure_logging", lambda *a, **k: None)


def _use_photos_config(monkeypatch, **kwargs):
    photos = PhotosConfig(**kwargs)
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: Config(photos=photos))


def _write_photo(path, size=(48, 64)):
    rng = np.random.default_rng(11)
    Image.fromarray(rng.integers(0, 255, (*size, 3), dtype=np.uint8), "RGB").save(path, "JPEG")
    return path


def _run(argv):
    return photos_handler.cmd_photos(_build_parser().parse_args(argv.split()), None)


def _pad(long_edge):
    return max(1, round(develop_mod.BORDER_FRACTION * long_edge))


# ── _add_border: the pure pad-and-paste primitive ────────────────────


def test_add_border_none_is_exact_noop():
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, (40, 60, 3), dtype=np.uint8)
    out = develop_mod._add_border(rgb, "none")
    assert out.shape == rgb.shape
    assert np.array_equal(out, rgb), "none must leave the exported bytes untouched"


@pytest.mark.parametrize("style, fill", [("white", 255), ("black", 0)])
def test_add_border_matte_pads_and_preserves_interior(style, fill):
    rgb = np.full((50, 80, 3), 120, dtype=np.uint8)
    out = develop_mod._add_border(rgb, style)
    pad = _pad(80)
    assert out.shape == (50 + 2 * pad, 80 + 2 * pad, 3)
    # all four border bands are the matte colour
    assert np.all(out[:pad] == fill) and np.all(out[-pad:] == fill)
    assert np.all(out[:, :pad] == fill) and np.all(out[:, -pad:] == fill)
    # the interior is the original frame, unchanged
    assert np.array_equal(out[pad : pad + 50, pad : pad + 80], rgb)


def test_add_border_filmstrip_has_wider_bars_and_sprocket_holes():
    rgb = np.full((120, 200, 3), 130, dtype=np.uint8)
    out = develop_mod._add_border(rgb, "filmstrip")
    pad = _pad(200)
    bar = max(pad + 1, round(develop_mod.FILMSTRIP_BAR_FRACTION * 200))
    assert out.shape == (120 + 2 * bar, 200 + 2 * pad, 3)
    assert bar > pad, "top/bottom bars are wider than the side matte"
    assert np.array_equal(out[bar : bar + 120, pad : pad + 200], rgb), "interior preserved"
    assert np.all(out[bar : bar + 120, :pad] == 0), "side matte is pure black"
    top_bar = out[:bar]
    assert top_bar.max() == 255, "sprocket holes are white"
    assert top_bar.min() == 0, "the bar background is black"


def test_add_border_is_deterministic():
    rgb = np.full((60, 90, 3), 77, dtype=np.uint8)
    assert np.array_equal(
        develop_mod._add_border(rgb, "filmstrip"), develop_mod._add_border(rgb, "filmstrip")
    )


def test_add_border_rejects_unknown_style():
    with pytest.raises(ValueError, match="unknown border style"):
        develop_mod._add_border(np.zeros((8, 8, 3), dtype=np.uint8), "rainbow")


# ── the three homes agree on the style set ───────────────────────────


def test_border_styles_list_is_the_expected_four():
    assert develop_mod.BORDER_STYLES == ("none", "white", "black", "filmstrip")


def test_config_border_default_off_and_validates_against_the_engine():
    assert PhotosConfig().border == "none"
    for style in develop_mod.BORDER_STYLES:
        assert PhotosConfig(border=style).border == style
    with pytest.raises(ValidationError):
        PhotosConfig(border="rainbow")


def test_border_argparse_rejects_unknown_style(tmp_path):
    photo = _write_photo(tmp_path / "IMG_0001.jpg")
    with pytest.raises(SystemExit):
        _build_parser().parse_args(f"photos develop {photo} --border rainbow".split())


# ── the CLI export loop applies the border ───────────────────────────


def test_cli_border_grows_export_but_leaves_contact_panels_unbordered(tmp_path, monkeypatch):
    _use_photos_config(monkeypatch)
    photo = _write_photo(tmp_path / "IMG_0002.jpg")
    captured = {}
    real_sheet = develop_mod.compose_contact_sheet

    def spy(variants, cols=None):
        captured["panels"] = [rgb.shape for _, rgb in variants]
        return real_sheet(variants, cols=cols)

    monkeypatch.setattr(develop_mod, "compose_contact_sheet", spy)
    out = tmp_path / "d"
    argv = f"photos develop {photo} --out {out} --styles natural --border white --contact-sheet"
    assert _run(argv) == 0
    with Image.open(out / "IMG_0002_natural.jpg") as im:
        assert im.width > 64 and im.height > 48, "the matte grows the exported master"
        exported = np.asarray(im.convert("RGB"))
    assert exported[0].mean() > 240, "the top matte band is white in the export"
    assert captured["panels"] == [(48, 64, 3)], "the contact-sheet panel stays un-bordered"


def test_cli_border_none_export_is_byte_identical_to_no_flag(tmp_path, monkeypatch):
    _use_photos_config(monkeypatch)
    photo = _write_photo(tmp_path / "IMG_0003.jpg")
    plain, noned = tmp_path / "plain", tmp_path / "noned"
    assert _run(f"photos develop {photo} --out {plain} --styles natural") == 0
    assert _run(f"photos develop {photo} --out {noned} --styles natural --border none") == 0
    assert (plain / "IMG_0003_natural.jpg").read_bytes() == (
        noned / "IMG_0003_natural.jpg"
    ).read_bytes(), "the default border path stays byte-identical"


# ── a look variant exports with its OWN border ───────────────────────
# ``polaroid`` ships ``border: white`` — the matte IS the look, so the panel that
# renders it must carry it; a look naming none rides the run's border as before.

LOOKS = {"matte": {"border": "white"}, "grainy": {"grain": 0.5}}


def _panel(out, name):
    """One exported panel as ``((width, height), rgb)``."""
    with Image.open(out / name) as im:
        return im.size, np.asarray(im.convert("RGB"))


def _matted(long_edge=64):
    return (64 + 2 * _pad(long_edge), 48 + 2 * _pad(long_edge))


def test_a_look_variant_renders_with_its_own_border(tmp_path, monkeypatch):
    """The run names no border at all; the look carries one, so its panel gets it."""
    _use_photos_config(monkeypatch, looks=LOOKS)
    photo = _write_photo(tmp_path / "IMG_0005.jpg")
    out = tmp_path / "d"
    assert _run(f"photos develop {photo} --out {out} --styles natural --look-variant matte") == 0
    size, rgb = _panel(out, "IMG_0005_look_matte.jpg")
    assert size == _matted(), "the look's own matte grows its panel"
    assert rgb[0].mean() > 240 and rgb[:, 0].mean() > 240, "and the matte is the look's white"
    assert _panel(out, "IMG_0005_natural.jpg")[0] == (64, 48), "the run's own border stays off"


def test_each_panel_takes_its_own_border_and_the_rest_take_the_runs(tmp_path, monkeypatch):
    """``--border black`` is the run's: only the look carrying its own overrides it,
    and a look naming none still gets the run-level border."""
    _use_photos_config(monkeypatch, looks=LOOKS)
    photo = _write_photo(tmp_path / "IMG_0006.jpg")
    out = tmp_path / "d"
    argv = f"photos develop {photo} --out {out} --styles natural --border black"
    assert _run(f"{argv} --look-variant matte --look-variant grainy") == 0
    matte_size, matte_rgb = _panel(out, "IMG_0006_look_matte.jpg")
    assert matte_size == _matted() and matte_rgb[0].mean() > 240, "the look's own white wins"
    for name in ("IMG_0006_natural.jpg", "IMG_0006_look_grainy.jpg"):
        size, rgb = _panel(out, name)
        assert size == _matted() and rgb[0].mean() < 20, f"{name} keeps the run's black"


def test_no_border_anywhere_leaves_every_panel_untouched(tmp_path, monkeypatch):
    """``none`` is an exact no-op, so the shipped default path must not have moved."""
    _use_photos_config(monkeypatch, looks=LOOKS)
    photo = _write_photo(tmp_path / "IMG_0007.jpg")
    out = tmp_path / "d"
    seen = []
    real = develop_mod._add_border
    monkeypatch.setattr(
        develop_mod, "_add_border", lambda rgb, style: (seen.append(style), real(rgb, style))[1]
    )
    argv = f"photos develop {photo} --out {out} --styles natural --look-variant grainy"
    assert _run(argv) == 0
    assert seen == ["none", "none"], "every panel still takes the run's none"
    assert _panel(out, "IMG_0007_look_grainy.jpg")[0] == (64, 48), "and nothing was padded"


def test_a_look_border_outside_the_style_set_is_refused_at_the_door(tmp_path, monkeypatch, capsys):
    """A look sets its knobs past both the argparse ``choices`` and the config
    validator, so the style is re-gated at the door — before any decode, never once
    per photo inside the export loop."""
    _use_photos_config(monkeypatch, looks={"bad": {"border": "rainbow"}})
    photo = _write_photo(tmp_path / "IMG_0008.jpg")
    out = tmp_path / "d"
    assert _run(f"photos develop {photo} --out {out} --look-variant bad") == 1
    assert _run(f"photos develop {photo} --out {out} --look bad") == 1
    err = capsys.readouterr().err
    assert err.count("rainbow") == 2 and "failed to develop" not in err, err
    assert not (out.exists() and list(out.glob("*.jpg"))), "nothing decoded or written"


@pytest.mark.parametrize(
    "config_border, flag, expected",
    [
        ("none", "", "none"),  # neither set -> off (byte-identical default)
        ("white", " --border filmstrip", "filmstrip"),  # flag beats config
        ("black", "", "black"),  # config used when the flag is absent
    ],
)
def test_border_flag_and_config_resolution(tmp_path, monkeypatch, config_border, flag, expected):
    photo = _write_photo(tmp_path / "IMG_0004.jpg")
    _use_photos_config(
        monkeypatch, developed_output_dir=str(tmp_path / "out"), border=config_border
    )
    seen = []
    real = develop_mod._add_border
    monkeypatch.setattr(
        develop_mod, "_add_border", lambda rgb, style: (seen.append(style), real(rgb, style))[1]
    )
    assert _run(f"photos develop {photo} --styles natural{flag}") == 0
    assert seen == [expected]
