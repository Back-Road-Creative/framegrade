"""``photos develop --look-variant NAME``: one named look, one contact-sheet panel.

``--look`` is a DEFAULTS layer standing in for unset flags, so exactly one look
applies per run and two candidates can never be compared side by side — the sheet's
whole reason for existing. ``--look-variant`` is repeatable like ``--lut`` and
panels each named look over the base grade THAT LOOK NAMES (its own ``styles``),
falling back to the natural correction for a look naming none. The two compose."""

import json

import numpy as np
import pytest
from PIL import Image

import gomoveshift.config as config_module
from gomoveshift.cli.handlers import photos as photos_handler
from gomoveshift.cli.parser import _build_parser
from gomoveshift.config import Config, PhotosConfig

# ``grainy`` is spatial-only: a panel ignoring the look's OWN knobs renders natural.
LOOKS = {
    "grainy": {"grain": 0.6},
    "teal": {"split_tone_shadow": [0.0, 0.05, 0.12]},
    "deep": {"vignette": 0.6},
    "noop": {"grain": 0.0},
    # The seven built-ins' shape: one colour knob riding a base style of its own.
    "tealmoody": {"styles": "moody", "split_tone_shadow": [0.0, 0.05, 0.12]},
    "bogusbase": {"styles": "nosuchstyle", "grain": 0.2},
}


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr(photos_handler, "configure_logging", lambda *a, **k: None)


def _use_looks(monkeypatch, looks=LOOKS):
    photos = PhotosConfig(looks=looks)
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: Config(photos=photos))


def _write_photo(path):
    rng = np.random.default_rng(11)
    Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8), "RGB").save(path, "JPEG")
    return path


def _run(argv):
    return photos_handler.cmd_photos(_build_parser().parse_args(argv.split()), None)


def _develop(monkeypatch, photo, out, argv_extra=""):
    """Develop ``photo`` into ``out`` with no forced ``--styles``; return every
    export as ``{filename: bytes}`` so a comparison covers NAMES as well as pixels."""
    _use_looks(monkeypatch)
    assert _run(f"photos develop {photo} --out {out} {argv_extra}") == 0
    return {p.name: p.read_bytes() for p in sorted(out.glob("*.jpg"))}


def _render(monkeypatch, photo, out, argv_extra=""):
    """``_develop`` with the run pinned to the single ``natural`` style variant."""
    return _develop(monkeypatch, photo, out, f"--styles natural {argv_extra}")


def _write_cube(path):
    """A 2-point ``.cube`` swapping R and B, in the Adobe R-fastest row order — a
    LUT panel that is visibly not the grade underneath it."""
    rows = [f"{b:.1f} {g:.1f} {r:.1f}" for b in (0, 1) for g in (0, 1) for r in (0, 1)]
    path.write_text("LUT_3D_SIZE 2\n" + "\n".join(rows) + "\n")
    return path


def test_no_look_variant_is_inert_and_byte_identical(tmp_path, monkeypatch):
    """An empty list means no new code path runs: the shipped render, exactly."""
    from gomoveshift.pipeline.photos.develop import plan_variants

    photo = _write_photo(tmp_path / "IMG_1.jpg")
    bare = _render(monkeypatch, photo, tmp_path / "bare")
    _use_looks(monkeypatch, {})
    assert _render(monkeypatch, photo, tmp_path / "cfg") == bare, "a look nobody rendered is inert"
    assert plan_variants(photo, ["natural"]).look_variants == []


def test_each_look_variant_is_its_own_panel_in_flag_order(tmp_path, monkeypatch, capsys):
    photo = _write_photo(tmp_path / "IMG_2.jpg")
    out = tmp_path / "d"
    panels = _render(monkeypatch, photo, out, "--look-variant teal --look-variant grainy")
    written = [t.rsplit("/", 1)[-1] for t in capsys.readouterr().out.split() if t.endswith(".jpg")]
    assert written == ["IMG_2_natural.jpg", "IMG_2_look_teal.jpg", "IMG_2_look_grainy.jpg"]
    assert panels["IMG_2_look_teal.jpg"] != panels["IMG_2_natural.jpg"], "its colour chain renders"
    assert panels["IMG_2_look_grainy.jpg"] != panels["IMG_2_natural.jpg"], "so do its spatial knobs"


def test_a_look_panel_renders_over_the_base_style_the_look_names(tmp_path, monkeypatch):
    """``--look NAME`` and ``--look-variant NAME`` are ONE picture. A look names its
    own base ``styles`` and the colour knobs ride THAT base; a panel graded from the
    natural correction instead was a different picture under the same name."""
    photo = _write_photo(tmp_path / "IMG_9.jpg")
    panel = _render(monkeypatch, photo, tmp_path / "v", "--look-variant tealmoody")
    whole = _develop(monkeypatch, photo, tmp_path / "l", "--look tealmoody")
    natural = _render(monkeypatch, photo, tmp_path / "n", "--split-tone-shadow 0.0 0.05 0.12")
    assert panel["IMG_9_look_tealmoody.jpg"] == whole["IMG_9_moody.jpg"], "the panel IS the look"
    assert panel["IMG_9_look_tealmoody.jpg"] != natural["IMG_9_natural.jpg"], "not the bare base"


def test_a_look_naming_no_base_style_still_grades_from_the_natural_base(tmp_path, monkeypatch):
    """The bit-identity half: a look with no ``styles`` renders exactly as before —
    its colour chain over the measured natural correction, byte for byte."""
    photo = _write_photo(tmp_path / "IMG_10.jpg")
    panel = _render(monkeypatch, photo, tmp_path / "v", "--look-variant teal")
    natural = _render(monkeypatch, photo, tmp_path / "n", "--split-tone-shadow 0.0 0.05 0.12")
    assert panel["IMG_10_look_teal.jpg"] == natural["IMG_10_natural.jpg"]


def test_a_styled_look_panel_moves_no_surface_but_its_own(tmp_path, monkeypatch):
    """The boundary a look's base style may not cross: the run's styled variants and
    the LUT looks (which ride the run's own base grade) stay byte-identical when a
    styled look panel joins the sheet — only the new panel appears."""
    photo = _write_photo(tmp_path / "IMG_11.jpg")
    cube = _write_cube(tmp_path / "swap.cube")
    argv = f"--styles natural,moody --lut {cube}"
    without = _develop(monkeypatch, photo, tmp_path / "a", argv)
    with_look = _develop(monkeypatch, photo, tmp_path / "b", f"{argv} --look-variant tealmoody")
    assert set(with_look) - set(without) == {"IMG_11_look_tealmoody.jpg"}
    assert {n: b for n, b in with_look.items() if n in without} == without


def test_a_look_variant_naming_an_unknown_base_style_is_refused_at_the_door(
    tmp_path, monkeypatch, capsys
):
    """Upstream gate, matching the run's own ``--styles`` refusal: an unknown base
    style would otherwise raise inside ``apply_style`` once per photo, mid-batch."""
    photo = _write_photo(tmp_path / "IMG_12.jpg")
    out = tmp_path / "d"
    _use_looks(monkeypatch)
    assert _run(f"photos develop {photo} --out {out} --look-variant bogusbase") == 1
    assert "nosuchstyle" in capsys.readouterr().err
    assert not (out.exists() and list(out.glob("*.jpg"))), "nothing decoded or written"


def test_an_unknown_look_variant_is_refused_at_the_door(tmp_path, monkeypatch, capsys):
    """Tagging both groups, exactly as the ``--look`` refusal does."""
    from gomoveshift.pipeline.photos import looks as builtin

    photo = _write_photo(tmp_path / "IMG_4.jpg")
    out = tmp_path / "d"
    monkeypatch.setattr(builtin, "BUILTIN_LOOKS", {"cyberpunk": {"vignette": 0.2}})
    _use_looks(monkeypatch, {"airy": {"clarity": 0.2}})
    assert _run(f"photos develop {photo} --out {out} --look-variant bogus") == 1
    err = capsys.readouterr().err
    assert "bogus" in err and "built-in: cyberpunk" in err and "configured: airy" in err, err
    assert not (out.exists() and list(out.glob("*.jpg"))), "nothing decoded or written"


def test_look_fills_the_run_while_look_variant_panels_it(tmp_path, monkeypatch):
    """Both flags in one run: ``--look`` still fills every panel's unset flags, and
    the variant panel is exactly those resolved knobs plus its own, as flags."""
    photo = _write_photo(tmp_path / "IMG_5.jpg")
    both = _render(monkeypatch, photo, tmp_path / "b", "--look deep --look-variant teal")
    filled = _render(monkeypatch, photo, tmp_path / "f", "--vignette 0.6")
    knobs = "--vignette 0.6 --split-tone-shadow 0.0 0.05 0.12"
    composed = _render(monkeypatch, photo, tmp_path / "c", knobs)
    assert set(both) == {"IMG_5_natural.jpg", "IMG_5_look_teal.jpg"}
    assert both["IMG_5_natural.jpg"] == filled["IMG_5_natural.jpg"], "--look still fills flags"
    assert both["IMG_5_look_teal.jpg"] == composed["IMG_5_natural.jpg"], "the panel rides them too"


def test_the_sidecar_records_the_look_name(tmp_path, monkeypatch):
    """A contact sheet is auditable back to the recipe that drew each panel."""
    photo = _write_photo(tmp_path / "IMG_6.jpg")
    out = tmp_path / "d"
    _render(monkeypatch, photo, out, "--look-variant teal --grade-sidecar")
    assert json.loads((out / "IMG_6_look_teal.grade.json").read_text())["look"] == {"name": "teal"}
    assert "look" not in json.loads((out / "IMG_6_natural.grade.json").read_text())


def test_an_indistinct_look_variant_still_renders_as_lut_looks_do(tmp_path, monkeypatch):
    """``lut_looks`` are NOT run through the rendered-distinctness gate — that
    guards AUTO-SELECTED styles — and an explicitly named look is the same
    operator authority an explicit ``--styles`` list is: rendered verbatim."""
    photo = _write_photo(tmp_path / "IMG_7.jpg")
    flat = _render(monkeypatch, photo, tmp_path / "flat", "--look-variant noop")
    assert set(flat) == {"IMG_7_natural.jpg", "IMG_7_look_noop.jpg"}
    assert flat["IMG_7_look_noop.jpg"] == flat["IMG_7_natural.jpg"], "identical, and kept"


def test_the_dry_run_prints_the_look_variants(tmp_path, monkeypatch, capsys):
    photo = _write_photo(tmp_path / "IMG_8.jpg")
    _use_looks(monkeypatch)
    argv = "--dry-run --styles natural --look-variant teal --look-variant grainy"
    assert _run(f"photos develop {photo} {argv}") == 0
    assert "look variants: look_teal, look_grainy" in capsys.readouterr().out
