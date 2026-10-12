"""``photos develop``: one decode fans out to N styled grades exported
as local ``<stem>_<style>.jpg`` working files, plus the PhotosConfig keys."""

from types import SimpleNamespace

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


def _write_photo(path):
    rng = np.random.default_rng(11)
    Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8), "RGB").save(path, "JPEG")
    return path


def _run(argv):
    return photos_handler.cmd_photos(_build_parser().parse_args(argv.split()), None)


def test_develop_variants_single_decode_and_natural_anchor(tmp_path, monkeypatch):
    photo = _write_photo(tmp_path / "IMG_0001.jpg")
    calls = []
    real_decode = develop_mod.decode
    monkeypatch.setattr(develop_mod, "decode", lambda p: (calls.append(p), real_decode(p))[1])
    results = list(develop_mod.develop_variants(photo))
    assert len(calls) == 1, "every variant must render from ONE decode"
    assert len(results) == 3
    assert results[0][0] == "natural"


def test_cli_exports_stem_style_names_and_respects_explicit_styles(tmp_path, monkeypatch):
    _use_photos_config(monkeypatch)
    photo = _write_photo(tmp_path / "IMG_0001.jpg")
    out = tmp_path / "developed"
    assert _run(f"photos develop {photo} --out {out}") == 0
    names = sorted(p.name for p in out.glob("*.jpg"))
    assert len(names) == 3
    assert "IMG_0001_natural.jpg" in names
    out2 = tmp_path / "explicit"
    assert _run(f"photos develop {photo} --out {out2} --styles mono,soft") == 0
    assert sorted(p.name for p in out2.glob("*.jpg")) == ["IMG_0001_mono.jpg", "IMG_0001_soft.jpg"]


def test_cli_rejects_unknown_style_and_unsupported_extension(tmp_path, monkeypatch, capsys):
    _use_photos_config(monkeypatch)
    photo = _write_photo(tmp_path / "IMG_0003.jpg")
    out = tmp_path / "d"
    assert _run(f"photos develop {photo} --out {out} --styles natural,bogus") == 1
    assert "bogus" in capsys.readouterr().err
    txt = tmp_path / "notes.txt"
    txt.write_text("x")
    assert _run(f"photos develop {txt} --out {out}") == 1
    assert "unsupported" in capsys.readouterr().err.lower()
    assert not out.exists(), "a rejected invocation must write nothing"


def test_cli_dry_run_writes_nothing_but_prints_plan(tmp_path, monkeypatch, capsys):
    _use_photos_config(monkeypatch)
    photo = _write_photo(tmp_path / "IMG_0004.jpg")
    exports = []
    monkeypatch.setattr(develop_mod, "export_jpeg", lambda *a, **k: exports.append(a))
    assert _run(f"photos develop {photo} --dry-run") == 0
    out = capsys.readouterr().out
    assert exports == []
    assert "exposure_class=" in out and "styles:" in out and "natural" in out


def test_cli_continue_on_error_multi_file(tmp_path, monkeypatch, capsys):
    _use_photos_config(monkeypatch)
    bad = tmp_path / "BAD.jpg"
    bad.write_bytes(b"not a jpeg")
    good = _write_photo(tmp_path / "GOOD.jpg")
    out = tmp_path / "d"
    assert _run(f"photos develop {bad} {good} --out {out} --styles natural") == 1
    assert (out / "GOOD_natural.jpg").is_file(), "a bad file must not sink the rest"
    assert "BAD.jpg" in capsys.readouterr().err


def test_cli_out_defaults_from_config_and_required_when_unset(tmp_path, monkeypatch, capsys):
    out = tmp_path / "from-config"
    _use_photos_config(monkeypatch, developed_output_dir=str(out))
    photo = _write_photo(tmp_path / "IMG_0005.jpg")
    assert _run(f"photos develop {photo} --styles natural") == 0
    assert (out / "IMG_0005_natural.jpg").is_file()
    _use_photos_config(monkeypatch)  # key unset and no --out -> hard fail
    assert _run(f"photos develop {photo}") == 1
    assert "--out" in capsys.readouterr().err


def _fake_variants(monkeypatch, variants):
    """Bypass decode/render: yield canned frames as ``develop_variants`` would."""

    def fake(
        source,
        styles=None,
        strength=develop_mod.STYLE_STRENGTH_DEFAULT,
        luts=None,
        lut_strength=1.0,
        vignette=0.0,
        clarity=0.0,
        grain=0.0,
        hue_shift=(),
        hue_sat=(),
        hue_lum=(),
        splittone_shadow=(),
        splittone_highlight=(),
        curve_r=(),
        curve_g=(),
        curve_b=(),
        vibrance=0.0,
        sharpen=1.0,
        mono_mix=(),
        duotone=(),
        lut_over="",
        subject=None,
    ):
        for name, rgb in variants:
            yield name, SimpleNamespace(rgb=rgb)

    monkeypatch.setattr(develop_mod, "develop_variants", fake)


def _two_wide_variants():
    """Two same-shape frames wider than the panel bound, with distinct pixels."""
    return [
        ("natural", np.full((240, 2400, 3), 40, dtype=np.uint8)),
        ("vivid", np.full((240, 2400, 3), 220, dtype=np.uint8)),
    ]


def test_cli_contact_sheet_is_one_bounded_labeled_row(tmp_path, monkeypatch):
    _use_photos_config(monkeypatch, contact_sheet_root=str(tmp_path / "review"))
    photo = _write_photo(tmp_path / "IMG_0007.jpg")
    _fake_variants(monkeypatch, _two_wide_variants())
    out = tmp_path / "d"
    argv = f"photos develop {photo} --out {out} --styles natural,vivid --contact-sheet"
    assert _run(argv) == 0
    sheet = tmp_path / "review" / "d" / "IMG_0007_contact.jpg"
    assert sheet.is_file()
    with Image.open(sheet) as img:
        assert img.format == "JPEG"
        assert img.width == 2 * develop_mod.CONTACT_PANEL_WIDTH
        assert img.height > 120, "sheet must carry a label strip above the 120px panels"


def test_cli_contact_sheet_composites_distinct_panels_and_labels(tmp_path, monkeypatch):
    _use_photos_config(monkeypatch, contact_sheet_root=str(tmp_path / "review"))
    photo = _write_photo(tmp_path / "IMG_0008.jpg")
    _fake_variants(monkeypatch, _two_wide_variants())
    out = tmp_path / "d"
    assert _run(f"photos develop {photo} --out {out} --styles natural,vivid --contact-sheet") == 0
    with Image.open(tmp_path / "review" / "d" / "IMG_0008_contact.jpg") as img:
        arr = np.asarray(img.convert("RGB"))
    width = develop_mod.CONTACT_PANEL_WIDTH
    left, right = arr[-100:, :width], arr[-100:, width:]
    assert abs(float(left.mean()) - float(right.mean())) > 100, "panels must differ per variant"
    strip = arr[: develop_mod.CONTACT_LABEL_HEIGHT]
    assert strip.max() > 200, "label text must be drawn on the strip"
    assert strip.mean() < 80, "label strip background must stay dark for contrast"


def _four_variants():
    """Four same-shape frames wider than the panel bound, distinct per style."""
    return [
        ("natural", np.full((240, 2400, 3), 40, dtype=np.uint8)),
        ("vivid", np.full((240, 2400, 3), 100, dtype=np.uint8)),
        ("moody", np.full((240, 2400, 3), 160, dtype=np.uint8)),
        ("golden", np.full((240, 2400, 3), 220, dtype=np.uint8)),
    ]


def test_compose_contact_sheet_default_is_unbroken_single_row():
    variants = _four_variants()
    default = develop_mod.compose_contact_sheet(variants)
    # A 240x2400 frame bounds to 1200 wide, 120 tall. The default lays every
    # panel in ONE row: total width = sum of panel widths, one label strip over
    # the tallest panel. cols>=len collapses to that same unbroken row.
    panel_h = 120
    assert default.shape == (
        develop_mod.CONTACT_LABEL_HEIGHT + panel_h,
        len(variants) * develop_mod.CONTACT_PANEL_WIDTH,
        3,
    )
    assert np.array_equal(default, develop_mod.compose_contact_sheet(variants, cols=len(variants)))


def test_compose_contact_sheet_wraps_into_grid_with_cols():
    variants = _four_variants()
    grid = develop_mod.compose_contact_sheet(variants, cols=2)
    # Four 1200x120 panels at 2 cols -> a 2x2 grid: two panels wide, two rows
    # each carrying its own label strip above a 120px panel.
    panel_h = 120
    assert grid.shape == (
        2 * (develop_mod.CONTACT_LABEL_HEIGHT + panel_h),
        2 * develop_mod.CONTACT_PANEL_WIDTH,
        3,
    )


def test_cli_contact_cols_wraps_the_sheet(tmp_path, monkeypatch):
    _use_photos_config(monkeypatch, contact_sheet_root=str(tmp_path / "review"))
    photo = _write_photo(tmp_path / "IMG_0012.jpg")
    _fake_variants(monkeypatch, _four_variants())
    out = tmp_path / "d"
    argv = (
        f"photos develop {photo} --out {out} "
        "--styles natural,vivid,moody,golden --contact-sheet --contact-cols 2"
    )
    assert _run(argv) == 0
    with Image.open(tmp_path / "review" / "d" / "IMG_0012_contact.jpg") as img:
        assert img.width == 2 * develop_mod.CONTACT_PANEL_WIDTH
        assert img.height == 2 * (develop_mod.CONTACT_LABEL_HEIGHT + 120)


def test_cli_no_contact_sheet_without_flag(tmp_path, monkeypatch):
    _use_photos_config(monkeypatch)
    photo = _write_photo(tmp_path / "IMG_0009.jpg")
    _fake_variants(monkeypatch, _two_wide_variants())
    out = tmp_path / "d"
    assert _run(f"photos develop {photo} --out {out} --styles natural,vivid") == 0
    names = sorted(p.name for p in out.glob("*.jpg"))
    assert names == ["IMG_0009_natural.jpg", "IMG_0009_vivid.jpg"]


def test_cli_dry_run_with_contact_sheet_writes_nothing(tmp_path, monkeypatch):
    _use_photos_config(monkeypatch)
    photo = _write_photo(tmp_path / "IMG_0010.jpg")
    exports = []
    monkeypatch.setattr(develop_mod, "export_jpeg", lambda *a, **k: exports.append(a))
    assert _run(f"photos develop {photo} --dry-run --contact-sheet --styles natural") == 0
    assert exports == []
    assert list(tmp_path.rglob("*_contact.jpg")) == []


def test_cli_contact_sheet_skipped_when_a_variant_fails(tmp_path, monkeypatch, capsys):
    _use_photos_config(monkeypatch)
    photo = _write_photo(tmp_path / "IMG_0011.jpg")

    def fake(
        source,
        styles=None,
        strength=develop_mod.STYLE_STRENGTH_DEFAULT,
        luts=None,
        lut_strength=1.0,
        vignette=0.0,
        clarity=0.0,
        grain=0.0,
        hue_shift=(),
        hue_sat=(),
        hue_lum=(),
        splittone_shadow=(),
        splittone_highlight=(),
        curve_r=(),
        curve_g=(),
        curve_b=(),
        vibrance=0.0,
        sharpen=1.0,
        mono_mix=(),
        duotone=(),
        lut_over="",
        subject=None,
    ):
        yield "natural", SimpleNamespace(rgb=np.full((60, 80, 3), 40, dtype=np.uint8))
        raise develop_mod.PhotoDecodeError("boom")

    monkeypatch.setattr(develop_mod, "develop_variants", fake)
    out = tmp_path / "d"
    assert _run(f"photos develop {photo} --out {out} --styles natural,vivid --contact-sheet") == 1
    assert not (out / "IMG_0011_contact.jpg").exists(), "a sheet needs ALL variants rendered"
    assert "boom" in capsys.readouterr().err


def test_cluster_variance_override_reaches_batch_default(monkeypatch):
    seen = []
    monkeypatch.setattr(
        develop_mod, "cluster_files_by_lighting", lambda fs, t: (seen.append(t), [[0]])[1]
    )
    monkeypatch.setattr(develop_mod, "_measure_rgb", lambda p: np.zeros((8, 8, 3), dtype=np.uint8))
    _use_photos_config(monkeypatch, cluster_variance_threshold=0.7)
    develop_mod.compute_batch_corrections(["a.jpg"])
    _use_photos_config(monkeypatch)  # key unset -> the span constant, one knob
    develop_mod.compute_batch_corrections(["a.jpg"])
    assert seen == [0.7, develop_mod.SPAN_SIMILARITY_THRESHOLD]


# ── --style-strength (the style lever) ───────────────────────────────


@pytest.mark.parametrize(
    "config_strength, flag, expected",
    [
        (1.0, "", develop_mod.STYLE_STRENGTH_DEFAULT),  # neither set -> shipped bake
        (1.5, " --style-strength 2.5", 2.5),  # flag beats config
        (2.0, "", 2.0),  # config used when the flag is absent
    ],
)
def test_style_strength_flag_and_config_resolution(
    tmp_path, monkeypatch, config_strength, flag, expected
):
    photo = _write_photo(tmp_path / "IMG_0007.jpg")
    _use_photos_config(
        monkeypatch, developed_output_dir=str(tmp_path / "out"), style_strength=config_strength
    )
    seen = []
    real = develop_mod.develop_variants
    monkeypatch.setattr(
        develop_mod,
        "develop_variants",
        lambda *a, **k: (seen.append(a[2]), real(*a, **k))[1],  # a[2] = strength
    )
    assert _run(f"photos develop {photo}{flag}") == 0
    assert seen == [expected]


def test_non_positive_style_strength_is_rejected_before_any_decode(tmp_path, monkeypatch, capsys):
    photo = _write_photo(tmp_path / "IMG_0010.jpg")
    _use_photos_config(monkeypatch, developed_output_dir=str(tmp_path / "out"))
    monkeypatch.setattr(
        develop_mod, "decode", lambda p: pytest.fail("decoded despite a bad strength")
    )
    assert _run(f"photos develop {photo} --style-strength 0") == 1
    assert "--style-strength must be a finite value > 0" in capsys.readouterr().err


@pytest.mark.parametrize("bad", ["nan", "inf"])
def test_non_finite_style_strength_is_rejected_before_any_decode(
    tmp_path, monkeypatch, capsys, bad
):
    """NaN loses every comparison, so a gate spelled ``strength <= 0`` waves it
    through and the NaN reaches ``CorrectionParams`` — the render then silently
    yields a DIFFERENT grade than either the flag or the config asked for, with
    no error anywhere. Every sibling knob gate is spelled ``not lo <= v <= hi``,
    which refuses NaN for free; this one must refuse it too."""
    photo = _write_photo(tmp_path / "IMG_0012.jpg")
    _use_photos_config(monkeypatch, developed_output_dir=str(tmp_path / "out"))
    monkeypatch.setattr(
        develop_mod, "decode", lambda p: pytest.fail("decoded despite a bad strength")
    )
    assert _run(f"photos develop {photo} --style-strength {bad}") == 1
    assert "--style-strength must be a finite value > 0" in capsys.readouterr().err


def test_non_finite_style_strength_from_config_is_rejected(tmp_path, monkeypatch, capsys):
    """The gate reads the RESOLVED value, so a ``photos.style_strength: .nan`` in
    config.yaml — YAML spells NaN natively — is refused on the same line."""
    photo = _write_photo(tmp_path / "IMG_0013.jpg")
    _use_photos_config(
        monkeypatch, developed_output_dir=str(tmp_path / "out"), style_strength=float("nan")
    )
    monkeypatch.setattr(
        develop_mod, "decode", lambda p: pytest.fail("decoded despite a bad strength")
    )
    assert _run(f"photos develop {photo}") == 1
    assert "--style-strength must be a finite value > 0" in capsys.readouterr().err


def test_stronger_lever_renders_a_more_extreme_look(tmp_path, monkeypatch):
    """End to end: the same photo at x2 must land further from natural than x1."""
    photo = _write_photo(tmp_path / "IMG_0011.jpg")
    _use_photos_config(monkeypatch, developed_output_dir=str(tmp_path / "out"))

    def spread(strength):
        plan = develop_mod.plan_variants(photo, ["natural", "moody"], strength)
        natural, moody = (p for _, p in plan.styled)
        return abs(moody.saturation_mult - natural.saturation_mult) + abs(
            moody.white_balance_b - natural.white_balance_b
        )

    assert spread(2.0) > spread(1.0)


def test_develop_module_default_matches_the_style_library():
    from gomoveshift.pipeline.photos import styles as styles_mod

    assert develop_mod.STYLE_STRENGTH_DEFAULT == styles_mod.DEFAULT_STRENGTH


# ── --lut (film-emulation .cube looks) ───────────────────────────────


def _write_cube(path, fn, size=2):
    """A size^3 .cube whose (r,g,b) grid sample is ``fn(r,g,b)``, R-fastest order."""
    axis = np.linspace(0.0, 1.0, size)
    lines = [f"LUT_3D_SIZE {size}", "DOMAIN_MIN 0 0 0", "DOMAIN_MAX 1 1 1"]
    for b in axis:
        for g in axis:
            for r in axis:
                o = fn(r, g, b)
                lines.append(f"{o[0]:.6f} {o[1]:.6f} {o[2]:.6f}")
    path.write_text("\n".join(lines) + "\n")


def test_lut_by_path_adds_a_variant_named_for_the_file_stem(tmp_path, monkeypatch):
    _use_photos_config(monkeypatch)
    photo = _write_photo(tmp_path / "IMG_0020.jpg")
    cube = tmp_path / "kodachrome.cube"
    _write_cube(cube, lambda r, g, b: (r, g, b))
    out = tmp_path / "d"
    assert _run(f"photos develop {photo} --out {out} --lut {cube}") == 0
    names = sorted(p.name for p in out.glob("*.jpg"))
    assert "IMG_0020_kodachrome.jpg" in names, "LUT variant named for the .cube stem"
    assert "IMG_0020_natural.jpg" in names, "LUT looks ADD to the auto trio, never replace it"
    assert len(names) == 4, "auto trio (3) + one LUT look"


def test_lut_by_config_name_uses_the_key_as_the_variant_name(tmp_path, monkeypatch):
    cube = tmp_path / "K64.cube"
    _write_cube(cube, lambda r, g, b: (r, g, b))
    _use_photos_config(monkeypatch, luts={"kodachrome": str(cube)})
    photo = _write_photo(tmp_path / "IMG_0021.jpg")
    out = tmp_path / "d"
    assert _run(f"photos develop {photo} --out {out} --styles natural --lut kodachrome") == 0
    assert sorted(p.name for p in out.glob("*.jpg")) == [
        "IMG_0021_kodachrome.jpg",  # the config KEY, not the K64.cube file stem
        "IMG_0021_natural.jpg",
    ]


def test_lut_look_actually_transforms_pixels(tmp_path, monkeypatch):
    _use_photos_config(monkeypatch)
    photo = _write_photo(tmp_path / "IMG_0022.jpg")
    swap = tmp_path / "swap.cube"
    _write_cube(swap, lambda r, g, b: (b, r, g))  # rotate channels (affine → size 2 exact)
    out = tmp_path / "d"
    assert _run(f"photos develop {photo} --out {out} --styles natural --lut {swap}") == 0
    with Image.open(out / "IMG_0022_natural.jpg") as im:
        natural = np.asarray(im.convert("RGB")).astype(int)
    with Image.open(out / "IMG_0022_swap.jpg") as im:
        looked = np.asarray(im.convert("RGB")).astype(int)
    assert float(np.abs(natural - looked).mean()) > 20, "the LUT must visibly re-grade the frame"


def test_bad_lut_fails_the_run_before_writing_anything(tmp_path, monkeypatch, capsys):
    _use_photos_config(monkeypatch)
    photo = _write_photo(tmp_path / "IMG_0023.jpg")
    bad = tmp_path / "bad.cube"
    bad.write_text("LUT_3D_SIZE 2\n0 0 0\n1 1 1\n")  # 2 data rows, an 8-row grid needs 8
    out = tmp_path / "d"
    assert _run(f"photos develop {photo} --out {out} --lut {bad}") == 1
    assert "LUT" in capsys.readouterr().err
    assert not out.exists(), "a bad LUT is caught upfront — nothing is written"


def test_dry_run_lists_lut_looks_and_writes_nothing(tmp_path, monkeypatch, capsys):
    _use_photos_config(monkeypatch)
    photo = _write_photo(tmp_path / "IMG_0024.jpg")
    cube = tmp_path / "portra.cube"
    _write_cube(cube, lambda r, g, b: (r, g, b))
    exports = []
    monkeypatch.setattr(develop_mod, "export_jpeg", lambda *a, **k: exports.append(a))
    assert _run(f"photos develop {photo} --dry-run --styles natural --lut {cube}") == 0
    out = capsys.readouterr().out
    assert exports == []
    assert "lut looks:" in out and "portra" in out


# ── --lut-strength (the LUT opacity lever) ───────────────────────────


@pytest.mark.parametrize(
    "config_strength, flag, expected",
    [
        (1.0, "", 1.0),  # neither set -> full-strength LUT
        (0.8, " --lut-strength 0.3", 0.3),  # flag beats config
        (0.5, "", 0.5),  # config used when the flag is absent
    ],
)
def test_lut_strength_flag_and_config_resolution(
    tmp_path, monkeypatch, config_strength, flag, expected
):
    photo = _write_photo(tmp_path / "IMG_0030.jpg")
    _use_photos_config(
        monkeypatch, developed_output_dir=str(tmp_path / "out"), lut_strength=config_strength
    )
    seen = []
    real = develop_mod.develop_variants
    monkeypatch.setattr(
        develop_mod,
        "develop_variants",
        lambda *a, **k: (seen.append(a[4]), real(*a, **k))[1],  # a[4] = the options record
    )
    assert _run(f"photos develop {photo} --styles natural{flag}") == 0
    assert [o.lut_strength for o in seen] == [expected]


def test_out_of_range_lut_strength_is_rejected_before_any_decode(tmp_path, monkeypatch, capsys):
    photo = _write_photo(tmp_path / "IMG_0031.jpg")
    _use_photos_config(monkeypatch, developed_output_dir=str(tmp_path / "out"))
    monkeypatch.setattr(
        develop_mod, "decode", lambda p: pytest.fail("decoded despite a bad lut strength")
    )
    assert _run(f"photos develop {photo} --lut-strength 1.5") == 1
    assert "--lut-strength must be in [0, 1]" in capsys.readouterr().err


# ── --vignette (the radial-vignette spatial stage) ───────────────────


@pytest.mark.parametrize(
    "config_vignette, flag, expected",
    [
        (0.0, "", 0.0),  # neither set -> off (byte-identical default)
        (0.2, " --vignette 0.5", 0.5),  # flag beats config
        (0.3, "", 0.3),  # config used when the flag is absent
    ],
)
def test_vignette_flag_and_config_resolution(
    tmp_path, monkeypatch, config_vignette, flag, expected
):
    photo = _write_photo(tmp_path / "IMG_0040.jpg")
    _use_photos_config(
        monkeypatch, developed_output_dir=str(tmp_path / "out"), vignette=config_vignette
    )
    seen = []
    real = develop_mod.develop_variants
    monkeypatch.setattr(
        develop_mod,
        "develop_variants",
        lambda *a, **k: (seen.append(a[4]), real(*a, **k))[1],  # a[4] = the options record
    )
    assert _run(f"photos develop {photo} --styles natural{flag}") == 0
    assert [o.vignette for o in seen] == [expected]


def test_out_of_range_vignette_is_rejected_before_any_decode(tmp_path, monkeypatch, capsys):
    photo = _write_photo(tmp_path / "IMG_0041.jpg")
    _use_photos_config(monkeypatch, developed_output_dir=str(tmp_path / "out"))
    monkeypatch.setattr(
        develop_mod, "decode", lambda p: pytest.fail("decoded despite a bad vignette")
    )
    assert _run(f"photos develop {photo} --vignette 1.5") == 1
    assert "--vignette must be in [0, 1]" in capsys.readouterr().err


# ── --clarity (the midtone local-contrast spatial stage) ─────────────


@pytest.mark.parametrize(
    "config_clarity, flag, expected",
    [
        (0.0, "", 0.0),  # neither set -> off (byte-identical default)
        (0.2, " --clarity 0.5", 0.5),  # flag beats config
        (0.3, "", 0.3),  # config used when the flag is absent
    ],
)
def test_clarity_flag_and_config_resolution(tmp_path, monkeypatch, config_clarity, flag, expected):
    photo = _write_photo(tmp_path / "IMG_0050.jpg")
    _use_photos_config(
        monkeypatch, developed_output_dir=str(tmp_path / "out"), clarity=config_clarity
    )
    seen = []
    real = develop_mod.develop_variants
    monkeypatch.setattr(
        develop_mod,
        "develop_variants",
        lambda *a, **k: (seen.append(a[4]), real(*a, **k))[1],  # a[4] = the options record
    )
    assert _run(f"photos develop {photo} --styles natural{flag}") == 0
    assert [o.clarity for o in seen] == [expected]


def test_out_of_range_clarity_is_rejected_before_any_decode(tmp_path, monkeypatch, capsys):
    photo = _write_photo(tmp_path / "IMG_0051.jpg")
    _use_photos_config(monkeypatch, developed_output_dir=str(tmp_path / "out"))
    monkeypatch.setattr(
        develop_mod, "decode", lambda p: pytest.fail("decoded despite a bad clarity")
    )
    assert _run(f"photos develop {photo} --clarity 1.5") == 1
    assert "--clarity must be in [-1, 1]" in capsys.readouterr().err
    assert _run(f"photos develop {photo} --clarity -1.5") == 1
    assert "--clarity must be in [-1, 1]" in capsys.readouterr().err


def test_negative_clarity_is_accepted_and_reaches_the_develop_options(tmp_path, monkeypatch):
    """Softening is a real setting, not a silently-clamped one."""
    photo = _write_photo(tmp_path / "IMG_0052.jpg")
    _use_photos_config(monkeypatch, developed_output_dir=str(tmp_path / "out"))
    seen = []
    real = develop_mod.develop_variants
    monkeypatch.setattr(
        develop_mod,
        "develop_variants",
        lambda *a, **k: (seen.append(a[4]), real(*a, **k))[1],  # a[4] = the options record
    )
    assert _run(f"photos develop {photo} --styles natural --clarity -0.5") == 0
    assert [o.clarity for o in seen] == [-0.5]


# ── --grain (the luminance film-grain spatial stage) ─────────────────


@pytest.mark.parametrize(
    "config_grain, flag, expected",
    [
        (0.0, "", 0.0),  # neither set -> off (byte-identical default)
        (0.2, " --grain 0.5", 0.5),  # flag beats config
        (0.3, "", 0.3),  # config used when the flag is absent
    ],
)
def test_grain_flag_and_config_resolution(tmp_path, monkeypatch, config_grain, flag, expected):
    photo = _write_photo(tmp_path / "IMG_0060.jpg")
    _use_photos_config(monkeypatch, developed_output_dir=str(tmp_path / "out"), grain=config_grain)
    seen = []
    real = develop_mod.develop_variants
    monkeypatch.setattr(
        develop_mod,
        "develop_variants",
        lambda *a, **k: (seen.append(a[4]), real(*a, **k))[1],  # a[4] = the options record
    )
    assert _run(f"photos develop {photo} --styles natural{flag}") == 0
    assert [o.grain for o in seen] == [expected]


def test_out_of_range_grain_is_rejected_before_any_decode(tmp_path, monkeypatch, capsys):
    photo = _write_photo(tmp_path / "IMG_0061.jpg")
    _use_photos_config(monkeypatch, developed_output_dir=str(tmp_path / "out"))
    monkeypatch.setattr(develop_mod, "decode", lambda p: pytest.fail("decoded despite a bad grain"))
    assert _run(f"photos develop {photo} --grain 1.5") == 1
    assert "--grain must be in [0, 1]" in capsys.readouterr().err


# ── --hue-shift / --hue-sat / --hue-lum (per-hue HSL, six anchor bands) ──


def test_hue_off_by_default_leaves_every_variant_neutral(tmp_path):
    """No flag, no config -> every styled variant carries empty hue tuples, so
    the grade is unaffected by the hue stage (empty = _apply_hsl no-op)."""
    photo = _write_photo(tmp_path / "IMG_0070.jpg")
    plan = develop_mod.plan_variants(photo, ["natural", "moody"])
    for _, params in plan.styled:
        assert params.hue_shift == () and params.hue_sat == () and params.hue_lum == ()
    assert plan.base.hue_sat == ()  # the LUT-look base stays clean too


def test_hue_bands_thread_onto_every_styled_variant_not_the_base(tmp_path):
    """A non-empty band lands on EVERY styled variant's params (natural too),
    but never the base -- LUT looks ride a clean film-emulation base."""
    photo = _write_photo(tmp_path / "IMG_0071.jpg")
    options = develop_mod.DevelopOptions(hue_sat=(1, 1, 1, 1, 1.4, 1))
    plan = develop_mod.plan_variants(photo, ["natural", "moody"], options=options)
    for _, params in plan.styled:
        assert params.hue_sat == (1, 1, 1, 1, 1.4, 1)
        assert params.hue_shift == () and params.hue_lum == ()
    assert plan.base.hue_sat == ()


def test_hue_flags_thread_through_the_handler(tmp_path, monkeypatch):
    photo = _write_photo(tmp_path / "IMG_0072.jpg")
    _use_photos_config(monkeypatch, developed_output_dir=str(tmp_path / "out"))
    seen = []
    real = develop_mod.develop_variants
    monkeypatch.setattr(
        develop_mod,
        "develop_variants",
        lambda *a, **k: (seen.append(a[4]), real(*a, **k))[1],  # a[4] = the options record
    )
    assert _run(f"photos develop {photo} --styles natural --hue-sat 1 1 1 1 1.4 1") == 0
    assert [(o.hue_shift, o.hue_sat, o.hue_lum) for o in seen] == [((), (1, 1, 1, 1, 1.4, 1), ())]


def test_hue_config_default_applies_when_flag_absent(tmp_path, monkeypatch):
    photo = _write_photo(tmp_path / "IMG_0073.jpg")
    _use_photos_config(
        monkeypatch, developed_output_dir=str(tmp_path / "out"), hue_lum=[1, 1, 1, 1, 1, 0.8]
    )
    seen = []
    real = develop_mod.develop_variants
    monkeypatch.setattr(
        develop_mod,
        "develop_variants",
        lambda *a, **k: (seen.append(a[4]), real(*a, **k))[1],  # a[4] = the options record
    )
    assert _run(f"photos develop {photo} --styles natural") == 0
    assert [o.hue_lum for o in seen] == [(1, 1, 1, 1, 1, 0.8)]


def test_hue_seven_values_thread_through_the_handler(tmp_path, monkeypatch):
    """The orange slot rides the same flag: 7 values reach the options record."""
    photo = _write_photo(tmp_path / "IMG_0076.jpg")
    _use_photos_config(monkeypatch, developed_output_dir=str(tmp_path / "out"))
    seen = []
    real = develop_mod.develop_variants
    monkeypatch.setattr(
        develop_mod,
        "develop_variants",
        lambda *a, **k: (seen.append(a[4]), real(*a, **k))[1],  # a[4] = the options record
    )
    assert _run(f"photos develop {photo} --styles natural --hue-sat 1 1.4 1 1 1 1 1") == 0
    assert [o.hue_sat for o in seen] == [(1, 1.4, 1, 1, 1, 1, 1)]


def test_hue_wrong_value_count_refused_before_any_decode(tmp_path, monkeypatch, capsys):
    """Six and seven are the legal arities; anything else fails at the door, and
    the refusal names both."""
    photo = _write_photo(tmp_path / "IMG_0074.jpg")
    _use_photos_config(monkeypatch, developed_output_dir=str(tmp_path / "out"))
    monkeypatch.setattr(
        develop_mod, "decode", lambda p: pytest.fail("decoded despite a bad hue-sat")
    )
    assert _run(f"photos develop {photo} --styles natural --hue-sat 1 1 1") == 1
    err = capsys.readouterr().err
    assert "--hue-sat" in err and "6" in err and "7" in err


def test_hue_config_accepts_six_or_seven_and_rejects_wrong_length():
    ok = PhotosConfig(hue_sat=[1, 1, 1, 1, 1.4, 1])
    assert ok.hue_sat == [1, 1, 1, 1, 1.4, 1]
    assert PhotosConfig(hue_sat=[1, 1.2, 1, 1, 1, 1.4, 1]).hue_sat == [1, 1.2, 1, 1, 1, 1.4, 1]
    with pytest.raises(ValidationError):
        PhotosConfig(hue_sat=[1, 1, 1, 1])


def test_out_of_range_hue_sat_rejected_before_any_decode(tmp_path, monkeypatch, capsys):
    photo = _write_photo(tmp_path / "IMG_0075.jpg")
    _use_photos_config(monkeypatch, developed_output_dir=str(tmp_path / "out"))
    monkeypatch.setattr(
        develop_mod, "decode", lambda p: pytest.fail("decoded despite a bad hue-sat")
    )
    assert _run(f"photos develop {photo} --styles natural --hue-sat 1 1 1 1 1 9") == 1
    assert "--hue-sat values must each be in [0, 5]" in capsys.readouterr().err


# ── --vibrance (low-sat-weighted chroma boost) ───────────────────────


def test_vibrance_flag_threads_through_the_handler(tmp_path, monkeypatch):
    photo = _write_photo(tmp_path / "IMG_0080.jpg")
    _use_photos_config(monkeypatch, developed_output_dir=str(tmp_path / "out"))
    seen = []
    real = develop_mod.develop_variants
    monkeypatch.setattr(
        develop_mod,
        "develop_variants",
        lambda *a, **k: (seen.append(a[4]), real(*a, **k))[1],  # a[4] = the options record
    )
    assert _run(f"photos develop {photo} --styles natural --vibrance 0.8") == 0
    assert [o.vibrance for o in seen] == [0.8]


def test_vibrance_config_default_applies_when_flag_absent(tmp_path, monkeypatch):
    photo = _write_photo(tmp_path / "IMG_0081.jpg")
    _use_photos_config(monkeypatch, developed_output_dir=str(tmp_path / "out"), vibrance=0.5)
    seen = []
    real = develop_mod.develop_variants
    monkeypatch.setattr(
        develop_mod,
        "develop_variants",
        lambda *a, **k: (seen.append(a[4]), real(*a, **k))[1],  # a[4] = the options record
    )
    assert _run(f"photos develop {photo} --styles natural") == 0
    assert [o.vibrance for o in seen] == [0.5]


def test_out_of_range_vibrance_is_rejected_before_any_decode(tmp_path, monkeypatch, capsys):
    photo = _write_photo(tmp_path / "IMG_0082.jpg")
    _use_photos_config(monkeypatch, developed_output_dir=str(tmp_path / "out"))
    monkeypatch.setattr(
        develop_mod, "decode", lambda p: pytest.fail("decoded despite a bad vibrance")
    )
    assert _run(f"photos develop {photo} --vibrance 3.0") == 1
    assert "--vibrance must be in [0, 2]" in capsys.readouterr().err


def test_contact_sheet_lands_under_the_review_root_not_the_out_dir(tmp_path, monkeypatch):
    """A sheet is a REVIEW artifact and belongs on the operator's review drive.

    Written into ``--out`` it lands wherever the run happened to develop —
    a scratch dir, a tmp path, a worktree — and the operator never sees it. The root is
    config, and the run's own ``--out`` basename is the subfolder, so two packs
    developed side by side keep their sheets apart without a clock.
    """
    root = tmp_path / "review"
    _use_photos_config(monkeypatch, contact_sheet_root=str(root))
    photo = _write_photo(tmp_path / "IMG_0101.jpg")
    _fake_variants(monkeypatch, _two_wide_variants())
    out = tmp_path / "electronic"
    argv = f"photos develop {photo} --out {out} --styles natural,vivid --contact-sheet"
    assert _run(argv) == 0
    assert (root / "electronic" / "IMG_0101_contact.jpg").is_file()
    assert not (out / "IMG_0101_contact.jpg").exists(), "the sheet must not stay in --out"
    assert (out / "IMG_0101_natural.jpg").is_file(), "masters still land in --out"


def test_a_missing_review_drive_is_refused_at_the_door(tmp_path, monkeypatch):
    """Refused BEFORE any decode: an unmounted drive must not cost a develop.

    Failing at write time would burn the whole render and then drop the sheet,
    which is the late-gate this rule exists to avoid.
    """
    _use_photos_config(monkeypatch, contact_sheet_root=str(tmp_path / "nodrive" / "sheets"))
    photo = _write_photo(tmp_path / "IMG_0102.jpg")
    argv = f"photos develop {photo} --out {tmp_path / 'd'} --styles natural --contact-sheet"
    assert _run(argv) == 1
    assert not (tmp_path / "d").exists(), "refused before anything was developed"


def test_no_contact_sheet_flag_never_touches_the_review_root(tmp_path, monkeypatch):
    """The gate is scoped to the flag: a plain develop runs with the drive absent."""
    _use_photos_config(monkeypatch, contact_sheet_root=str(tmp_path / "nodrive" / "sheets"))
    photo = _write_photo(tmp_path / "IMG_0103.jpg")
    assert _run(f"photos develop {photo} --out {tmp_path / 'd'} --styles natural") == 0
    assert not (tmp_path / "nodrive").exists()
