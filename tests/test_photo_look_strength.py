"""``--look-strength``: one dial turns every look into a family.

A look variant is all-or-nothing without it — the recipe renders whole or not at
all, and the only lever the operator had was hand-editing the amplitudes inside
the recipe (which is how ``neon_edge`` ended up measuring 3.03 rendered
distinctness against a 5.0 floor: no longer a look at all). The dial lerps the
look's RENDERED frame toward the run's own base render,
``out = base + s * (look - base)``, so every look carries a continuous family.
1.0 (the default) is the shipped path, byte for byte, and pays no extra render.
"""

import numpy as np
import pytest
from PIL import Image

import gomoveshift.config as config_module
from gomoveshift.cli.handlers import photos as photos_handler
from gomoveshift.cli.parser import _build_parser
from gomoveshift.config import Config, PhotosConfig
from gomoveshift.pipeline.photos import develop as dev

# Colour AND spatial, so the dial is proved over the whole render, not one stage.
NEON = {"split_tone_shadow": [0.0, 0.05, 0.12], "vignette": 0.6, "grain": 0.4}


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr(photos_handler, "configure_logging", lambda *a, **k: None)


def _write_photo(path):
    rng = np.random.default_rng(11)
    Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8), "RGB").save(path, "JPEG")
    return path


def _look(**knobs):
    return dev.look_variant_options({**NEON, **knobs}, dev.DevelopOptions())


def _panel(photo, **knobs):
    """The ``look_neon`` panel's pixels for one recipe."""
    rendered = dict(
        dev.develop_variants(photo, ["natural"], look_variants=[("neon", _look(**knobs))])
    )
    return rendered[f"{dev.LOOK_VARIANT_PREFIX}neon"].rgb


def _base_render(photo):
    """The run's own base render — what a look dialled to 0.0 must land on."""
    plan = dev.plan_variants(photo, ["natural"])
    denoised, _ = dev._denoise(plan.decoded.rgb, plan.base)
    return dev._spatial_chain(dev._layered(dev._grade(denoised, plan.base)), plan)[0]


def _distance(a, b):
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def _run(argv):
    return photos_handler.cmd_photos(_build_parser().parse_args(argv.split()), None)


def test_full_strength_is_the_shipped_look_render_byte_for_byte(tmp_path):
    """The default may not re-render, re-quantize, or otherwise move one pixel."""
    photo = _write_photo(tmp_path / "IMG_S1.jpg")
    look = _look()
    plan = dev.plan_variants(photo, ["natural"], look_variants=[("neon", look)])
    denoised, _ = dev._denoise(plan.decoded.rgb, plan.base)
    graded = dev._grade(denoised, dev._look_params(plan.base, look))
    shipped = dev._spatial_chain(dev._layered(graded, look.layers), look)[0]
    assert np.array_equal(_panel(photo), shipped), "the un-dialled render, unchanged"
    assert np.array_equal(_panel(photo, look_strength=1.0), shipped)


def test_zero_lands_on_the_run_s_own_base_render(tmp_path):
    photo = _write_photo(tmp_path / "IMG_S2.jpg")
    assert np.array_equal(_panel(photo, look_strength=0.0), _base_render(photo))


def test_half_strength_lands_strictly_between_the_two_ends(tmp_path):
    photo = _write_photo(tmp_path / "IMG_S3.jpg")
    base, full = _base_render(photo), _panel(photo)
    half = _panel(photo, look_strength=0.5)
    assert not np.array_equal(half, base) and not np.array_equal(half, full), "neither end"
    assert _distance(half, base) < _distance(full, base), "half a look is nearer the base"
    assert _distance(half, full) < _distance(base, full), "and nearer the look than the base is"
    lo, hi = np.minimum(base, full), np.maximum(base, full)
    assert ((half >= lo) & (half <= hi)).all(), "every pixel sits on the segment between them"


def test_the_base_render_is_paid_once_and_only_when_a_look_asks(tmp_path, monkeypatch):
    """Lazy AND shared: the base render costs one extra pass for a whole photo, and
    a full-strength run never reaches it — which is what makes the default free."""
    photo = _write_photo(tmp_path / "IMG_S4.jpg")
    seen: list[int] = []
    real = dev._spatial_chain

    def counted(rgb, o):
        seen.append(1)
        return real(rgb, o)

    monkeypatch.setattr(dev, "_spatial_chain", counted)
    dialled = [(n, _look(look_strength=s)) for n, s in (("a", 0.4), ("b", 0.7))]
    list(dev.develop_variants(photo, ["natural"], look_variants=dialled))
    assert len(seen) == 4, "one styled panel + two looks + ONE shared base render"
    seen.clear()
    list(dev.develop_variants(photo, ["natural"], look_variants=[(n, _look()) for n in "ab"]))
    assert len(seen) == 3, "at full strength the shipped path runs, with no extra render"


def test_the_dial_is_recallable_from_a_recipe_and_gated_at_the_cli_door(
    tmp_path, monkeypatch, capsys
):
    photo = _write_photo(tmp_path / "IMG_S5.jpg")
    looks = {"neon": NEON, "half": {**NEON, "look_strength": 0.5}}
    monkeypatch.setattr(
        config_module, "get_config", lambda *a, **k: Config(photos=PhotosConfig(looks=looks))
    )

    def develop(out, extra):
        assert _run(f"photos develop {photo} --out {out} --styles natural {extra}") == 0
        return {p.name: p.read_bytes() for p in out.glob("*.jpg")}

    recipe = develop(tmp_path / "r", "--look-variant half")
    flag = develop(tmp_path / "f", "--look-variant neon --look-strength 0.5")
    assert recipe["IMG_S5_look_half.jpg"] == flag["IMG_S5_look_neon.jpg"], "one dial, two doors"
    assert recipe["IMG_S5_look_half.jpg"] != flag["IMG_S5_natural.jpg"], "and it still renders"
    bad = tmp_path / "bad"
    assert _run(f"photos develop {photo} --out {bad} --look-strength 1.5") == 1
    assert "--look-strength must be in [0, 1]" in capsys.readouterr().err
    assert not bad.exists(), "refused before any decode"


def test_a_dial_no_panel_can_apply_is_refused_at_the_door(tmp_path, monkeypatch, capsys):
    """``--look`` FOLDS a look into the run's own knobs (``_apply_look``), leaving
    no look-free base render to blend toward and no record of which knob came from
    the look — so the dial cannot mean there what it means on a panel. It used to
    be accepted, range-gated, recorded, and then silently ignored: full-strength
    exports, exit 0, no warning. The flag and a recipe's own ``look_strength:``
    resolve to ONE value, so both doors get one answer."""
    photo = _write_photo(tmp_path / "IMG_S6.jpg")
    looks = {"neon": NEON, "half": {**NEON, "look_strength": 0.5}}
    monkeypatch.setattr(
        config_module, "get_config", lambda *a, **k: Config(photos=PhotosConfig(looks=looks))
    )
    for i, extra in enumerate(
        ("--look neon --look-strength 0.5", "--look half", "--look-strength 0")
    ):
        out = tmp_path / f"nodial{i}"
        assert _run(f"photos develop {photo} --out {out} --styles natural {extra}") == 1
        err = capsys.readouterr().err
        assert "--look-strength" in err and "--look-variant" in err, err
        assert not out.exists(), "refused before any decode"
    ok = tmp_path / "ok"
    assert _run(f"photos develop {photo} --out {ok} --styles natural --look neon") == 0
    assert list(ok.glob("*.jpg")), "the whole look still bakes"
