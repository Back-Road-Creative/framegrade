"""``--lut-over STYLE``: compose a film ``.cube`` LUT over a STYLED grade.

A LUT look used to ride the neutral base correction only, so "Portra over
moody's matte" was unbuildable. ``lut_over`` grades with that style's params
first and applies the cube on top; unset (the default) is the shipped
base-graded surface, byte for byte.
"""

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from pycube_lut import apply_cube, load_cube

import gomoveshift.config as config_module
from gomoveshift.cli.handlers import photos as photos_handler
from gomoveshift.cli.parser import _build_parser
from gomoveshift.config import Config, PhotosConfig
from gomoveshift.pipeline.photos import develop as dev
from gomoveshift.pipeline.photos.styles import STYLES, apply_style


def _write_photo(path):
    """A landscape-ish frame — warm ground under a blue sky, lightly textured —
    so the styles have real chroma and tonal range to move."""
    h, w = 96, 128
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    rgb = np.empty((h, w, 3), np.float32)
    rgb[..., 0] = 0.22 + 0.50 * y
    rgb[..., 1] = 0.34 + 0.28 * y
    rgb[..., 2] = 0.78 - 0.42 * y
    rgb += np.random.default_rng(4).normal(0.0, 0.03, rgb.shape).astype(np.float32)
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), "RGB").save(
        path, "JPEG", quality=95
    )
    return path


def _film(r, g, b):
    """A film-emulation-ish look: lifted, slightly faded blacks with a warm bias."""
    lift = lambda v: v * 0.90 + 0.07  # noqa: E731
    return (min(1.0, lift(r) * 1.06), lift(g), min(1.0, lift(b) * 0.92))


def _write_cube(path, fn=_film, size=17):
    axis = np.linspace(0.0, 1.0, size)
    lines = [f"LUT_3D_SIZE {size}", "DOMAIN_MIN 0 0 0", "DOMAIN_MAX 1 1 1"]
    for b in axis:
        for g in axis:
            for r in axis:
                o = fn(r, g, b)
                lines.append(f"{o[0]:.6f} {o[1]:.6f} {o[2]:.6f}")
    path.write_text("\n".join(lines) + "\n")
    return load_cube(path)


def _lut_surface_by_hand(photo, cube, params_of, strength=dev.STYLE_STRENGTH_DEFAULT):
    """Render the LUT look the long way — the pre-change algorithm spelled out:
    denoise once under the base budget, grade, apply the cube, clarity, sharpen,
    vignette, grain. ``params_of(plan)`` picks the grade the cube rides on."""
    plan = dev.plan_variants(photo, ["natural"], strength, [("film", cube)])
    denoised, _ = dev._denoise(plan.decoded.rgb, plan.base)
    graded = dev._grade(denoised, params_of(plan))
    look = dev._clarity(apply_cube(graded, cube, plan.lut_strength), plan.clarity)
    sharpened, _ = dev._sharpen(look, plan.sharpen)
    return dev._grain(dev._vignette(sharpened, plan.vignette), plan.grain)


def _looks(photo, cube, strength=dev.STYLE_STRENGTH_DEFAULT, **kwargs):
    """``{name: rgb}`` for one develop run over a single ``film`` LUT look."""
    options = dev.DevelopOptions(**kwargs)
    variants = dev.develop_variants(photo, ["natural"], strength, [("film", cube)], options)
    return {name: developed.rgb for name, developed in variants}


def test_unset_lut_over_renders_the_shipped_base_graded_surface(tmp_path):
    """The default must stay byte-identical to the pre-change LUT-over-base path."""
    photo = _write_photo(tmp_path / "IMG_0090.jpg")
    cube = _write_cube(tmp_path / "portra.cube")
    rendered = _looks(photo, cube, lut_over="")["film"]
    assert np.array_equal(rendered, _lut_surface_by_hand(photo, cube, lambda p: p.base))


def test_lut_over_grades_with_that_styles_params_before_the_cube(tmp_path):
    """The composed surface matches a by-hand moody grade + cube, exactly."""
    photo = _write_photo(tmp_path / "IMG_0091.jpg")
    cube = _write_cube(tmp_path / "portra.cube")
    composed = _looks(photo, cube, lut_over="moody")["film_over_moody"]
    expected = _lut_surface_by_hand(
        photo, cube, lambda p: apply_style("moody", p.base, p.scene, dev.STYLE_STRENGTH_DEFAULT)
    )
    assert np.array_equal(composed, expected)


def test_lut_over_is_a_visibly_different_lut_surface(tmp_path):
    """Not a one-level nudge: the composed look must clear the same rendered
    distinctness bar two style variants must clear to both ship."""
    photo = _write_photo(tmp_path / "IMG_0092.jpg")
    cube = _write_cube(tmp_path / "portra.cube")
    plain = _looks(photo, cube)["film"].astype(int)
    composed = _looks(photo, cube, lut_over="moody")["film_over_moody"].astype(int)
    assert float(np.abs(plain - composed).mean()) >= dev.MIN_RENDERED_DISTINCTNESS


def test_style_strength_scales_the_composed_grade(tmp_path):
    """``--lut-over`` names a STYLE, so the style lever applies to it."""
    photo = _write_photo(tmp_path / "IMG_0093.jpg")
    cube = _write_cube(tmp_path / "portra.cube")
    at_1 = _looks(photo, cube, lut_over="moody")["film_over_moody"].astype(int)
    at_2 = _looks(photo, cube, strength=2.0, lut_over="moody")["film_over_moody"].astype(int)
    assert float(np.abs(at_1 - at_2).mean()) >= 2.0


def test_styled_variants_are_untouched_by_lut_over(tmp_path):
    """``--lut-over`` composes UNDER the LUT only — the style variants are the
    same pixels whether or not it is set."""
    photo = _write_photo(tmp_path / "IMG_0094.jpg")
    cube = _write_cube(tmp_path / "portra.cube")
    plain = _looks(photo, cube)
    composed = _looks(photo, cube, lut_over="moody")
    assert np.array_equal(plain["natural"], composed["natural"])


def test_operator_colour_fields_still_never_ride_the_lut_surface(tmp_path):
    """The per-hue/split-tone/curve/vibrance/mono move rides styled variants, not
    the LUT's grade — ``--lut-over`` composes a STYLE, nothing else."""
    photo = _write_photo(tmp_path / "IMG_0095.jpg")
    cube = _write_cube(tmp_path / "portra.cube")
    options = dev.DevelopOptions(vibrance=1.5, lut_over="moody")
    plan = dev.plan_variants(photo, ["natural"], luts=[("film", cube)], options=options)
    assert plan.lut_params is not None
    assert plan.lut_params.vibrance == 0.0 and plan.lut_params.hue_sat == ()
    assert all(p.vibrance == 1.5 for _, p in plan.styled)


# ── CLI / config exposure ────────────────────────────────────────────


def _run(cmd, extra=()):
    """``photos develop`` through the real parser + handler (``extra`` carries
    an argument ``split()`` can't, e.g. the empty string)."""
    return photos_handler.cmd_photos(_build_parser().parse_args([*cmd.split(), *extra]), None)


def _use_config(monkeypatch, **kwargs):
    photos = PhotosConfig(**kwargs)
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: Config(photos=photos))


def test_unknown_lut_over_style_is_rejected_before_any_decode(tmp_path, monkeypatch, capsys):
    photo = _write_photo(tmp_path / "IMG_0096.jpg")
    _use_config(monkeypatch, developed_output_dir=str(tmp_path / "out"))
    monkeypatch.setattr(dev, "decode", lambda p: pytest.fail("decoded despite a bad --lut-over"))
    assert _run(f"photos develop {photo} --lut-over kodachrome") == 1
    err = capsys.readouterr().err
    assert "--lut-over" in err and "kodachrome" in err
    assert all(name in err for name in STYLES), "the message must list the available styles"


def test_empty_lut_over_is_accepted_as_off(tmp_path, monkeypatch):
    photo = _write_photo(tmp_path / "IMG_0097.jpg")
    out = tmp_path / "out"
    _use_config(monkeypatch, developed_output_dir=str(out))
    assert _run(f"photos develop {photo} --styles natural", ["--lut-over", ""]) == 0
    assert (out / "IMG_0097_natural.jpg").exists()


def test_flag_beats_config_and_names_the_composed_variant(tmp_path, monkeypatch, capsys):
    cube_path = tmp_path / "portra.cube"
    _write_cube(cube_path)
    _use_config(monkeypatch, luts={"portra": str(cube_path)}, lut_over="golden")
    photo = _write_photo(tmp_path / "IMG_0098.jpg")
    out = tmp_path / "out"
    cmd = f"photos develop {photo} --out {out} --styles natural --lut portra --lut-over moody"
    assert _run(cmd) == 0
    assert (out / "IMG_0098_portra_over_moody.jpg").exists(), "the flag beats photos.lut_over"
    assert not (out / "IMG_0098_portra_over_golden.jpg").exists()
    assert "portra_over_moody" in capsys.readouterr().out


def test_dry_run_names_the_composed_surface(tmp_path, monkeypatch, capsys):
    cube = tmp_path / "portra.cube"
    _write_cube(cube)
    _use_config(monkeypatch, developed_output_dir=str(tmp_path / "out"))
    photo = _write_photo(tmp_path / "IMG_0099.jpg")
    assert (
        _run(f"photos develop {photo} --dry-run --styles natural --lut {cube} --lut-over moody")
        == 0
    )
    out = capsys.readouterr().out
    assert "portra_over_moody" in out and "style-strength does not apply" not in out


def test_config_rejects_an_unknown_lut_over_style():
    with pytest.raises(ValidationError):
        PhotosConfig(lut_over="kodachrome")
    assert PhotosConfig(lut_over="moody").lut_over == "moody"
    assert PhotosConfig().lut_over == ""


# ── the LUT surface is the operator colour knobs' blind spot ─────────
#
# A film LUT is a fixed table: its whole value is that it renders the film's
# look, not the operator's. Every colour knob is therefore ``replace``d onto the
# STYLED variants and deliberately never onto ``base``/``lut_params`` — stated in
# ``plan_variants``, in ``develop_variants`` and in the develop-CLI docs, and held
# by nothing but each feature author remembering it. Eleven knobs have landed that
# way one PR at a time; the twelfth is a leak waiting to happen, and it would be
# invisible (a slightly operator-tinted film emulation renders fine). These pin it.

DUOTONE = (0.05, 0.02, 0.0, 1.0, 0.9, 0.65)

#: knob -> a value that visibly moves a styled variant on the fixture frame.
LUT_BLIND_COLOUR_KNOBS = {
    "hue_shift": (180.0,) * 6,
    "hue_sat": (5.0,) * 6,
    "hue_lum": (0.35,) * 6,
    "splittone_shadow": (0.5, -0.5, 0.5),
    "splittone_midtone": (0.5, -0.5, 0.5),
    "splittone_highlight": (-0.5, 0.5, -0.5),
    "curve_l": (0.0, 0.6, 1.0, 0.4),
    "curve_r": (0.0, 1.0, 1.0, 0.0),
    "curve_g": (0.0, 0.8, 1.0, 0.1),
    "curve_b": (0.0, 0.9, 1.0, 0.2),
    # White balance is a colour move like every knob above it: the operator's
    # temperature must ride the styled variants and leave the film table alone.
    "white_balance_r": 0.1,
    "white_balance_b": -0.1,
    "white_balance_g": 0.5,
    "white_balance_b_high": -0.1,
    # Kelvin/tint fold into the r/b/g axes above; exposure and the recovery pair
    # are tone moves that ride the same styled-only path (plan_variants).
    "white_balance_kelvin": 1500.0,
    "white_balance_tint": 0.3,
    "exposure_ev": 1.0,
    "shadow_recovery": 10.0,
    "highlight_recovery": 10.0,
    "vibrance": 2.0,
    "mono_mix": (1.4, 0.2, -0.6),
    "duotone": DUOTONE,
    "reference_lab": (72.0, 28.0, 44.0, 14.0, 7.0, 9.0),
    # A masked colour layer is still a COLOUR move: it rides the styled variants
    # and must stay off the film table like every knob above it.
    "layers": ({"mask": "midtones", "curve_r": (0.0, 1.0, 1.0, 0.0)},),
}

#: knob -> the whole recipe it needs. These only bite ALONGSIDE another knob, so
#: each is proven against that partner alone rather than a plain render: two dial
#: or bend a ``duotone`` ramp, and the highlight-only GREEN white balance reaches
#: nothing on a landscape whose green channel never crosses the 0.75 highlight
#: threshold — so a ``curve_g`` lifts it into the range that knob exists for. RED
#: crosses it only through the fixture's noise, which the measured base denoise
#: now smooths away, so ``white_balance_r_high`` gets the same ``curve_r`` lift.
LUT_BLIND_PAIRED_KNOBS = {
    "duotone_strength": {"duotone": DUOTONE, "duotone_strength": 0.0},
    "duotone_mid": {"duotone": DUOTONE, "duotone_mid": (0.9, 0.1, 0.1)},
    "white_balance_g_high": {"curve_g": (0.0, 0.9, 1.0, 1.0), "white_balance_g_high": 0.5},
    "white_balance_r_high": {"curve_r": (0.0, 0.9, 1.0, 1.0), "white_balance_r_high": 0.1},
}

#: The rest of ``DevelopOptions``: the spatial stages, which ride EVERY rendered
#: surface including the LUT looks by design, and the LUT surface's own levers.
LUT_FACING_OPTIONS = frozenset(
    {
        "vignette", "dehaze", "sky_darken", "sky_saturate", "clarity", "grain", "grain_size",
        "halation", "halation_tint", "graduated_nd", "graduated_nd_angle",
        "light_leak", "chromatic_ab", "holo_sheen", "scanline", "glitch", "glow", "sharpen",
        "lut_strength", "lut_over", "lut_auto",
        "straighten_angle", "crop_rect", "lens_correct",
    }
)  # fmt: skip

#: The third class: knobs that reach NEITHER surface. ``look_strength`` dials a
#: ``--look-variant`` panel toward the run's base render, so no styled variant and
#: no LUT look can see it — see tests/test_photo_look_strength.py.
LOOK_PANEL_OPTIONS = frozenset({"look_strength"})


@pytest.mark.parametrize("knob", sorted(LUT_BLIND_COLOUR_KNOBS))
def test_an_operator_colour_knob_never_reaches_the_lut_surface(tmp_path, knob):
    """Setting one colour knob must move the styled variant and leave the LUT look
    byte-identical. The styled assertion is what keeps this from passing vacuously
    on a knob whose value happens to be a no-op on the fixture frame."""
    photo = _write_photo(tmp_path / f"IMG_{abs(hash(knob)) % 9000:04d}.jpg")
    cube = _write_cube(tmp_path / "film.cube")
    plain = _looks(photo, cube)
    moved = _looks(photo, cube, **{knob: LUT_BLIND_COLOUR_KNOBS[knob]})
    assert not np.array_equal(moved["natural"], plain["natural"]), (
        f"{knob} must move the styled variant, else this test proves nothing"
    )
    assert np.array_equal(moved["film"], plain["film"]), (
        f"{knob} leaked onto the LUT surface — a film emulation must stay the film's "
        "look, not the operator's"
    )


@pytest.mark.parametrize("knob", sorted(LUT_BLIND_PAIRED_KNOBS))
def test_a_paired_colour_knob_never_reaches_the_lut_surface(tmp_path, knob):
    """Same contract for the two ramp knobs, against the ramp as the baseline."""
    photo = _write_photo(tmp_path / f"IMG_{abs(hash(knob)) % 9000:04d}.jpg")
    cube = _write_cube(tmp_path / "film.cube")
    recipe = LUT_BLIND_PAIRED_KNOBS[knob]
    ramp = _looks(photo, cube, **{k: v for k, v in recipe.items() if k != knob})
    moved = _looks(photo, cube, **recipe)
    assert not np.array_equal(moved["natural"], ramp["natural"]), (
        f"{knob} must move the styled variant, else this test proves nothing"
    )
    assert np.array_equal(moved["film"], ramp["film"]), f"{knob} leaked onto the LUT surface"


def test_every_develop_option_is_classified_lut_blind_or_lut_facing():
    """The sync half: a NEW ``DevelopOptions`` field must be declared in one of the
    sets above, so its author has to decide — and prove — whether it rides the
    LUT surface. Without this a knob added a year from now simply goes unguarded,
    which is exactly how the eleven above accumulated."""
    fields = set(dev.DevelopOptions.__dataclass_fields__)
    classified = (
        set(LUT_BLIND_COLOUR_KNOBS)
        | set(LUT_BLIND_PAIRED_KNOBS)
        | LUT_FACING_OPTIONS
        | LOOK_PANEL_OPTIONS
    )
    assert not fields - classified, (
        f"unclassified DevelopOptions fields: {sorted(fields - classified)} — add each to "
        "LUT_BLIND_COLOUR_KNOBS (with a value that moves a styled variant), to "
        "LUT_FACING_OPTIONS, or to LOOK_PANEL_OPTIONS"
    )
    assert not classified - fields, (
        f"classified names that are no longer DevelopOptions fields: {sorted(classified - fields)}"
    )
