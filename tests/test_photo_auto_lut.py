"""``--lut-auto N``: scene-adaptive AUTOMATIC LUT selection. ``characterize_cube``
measures what a cube DOES to the graded frame, ``score_lut`` scores that against the
scene AND against how much of the frame it kills, ``select_luts`` keeps the best N —
deterministically. ``0`` is the manual ``--lut`` path, byte for byte."""

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

import pycube_lut as lut_mod

import gomoveshift.config as config_module
from gomoveshift.cli.handlers import photos as photos_handler
from gomoveshift.cli.parser import _build_parser
from gomoveshift.config import Config, PhotosConfig
from gomoveshift.pipeline.photos import develop as dev
from gomoveshift.pipeline.photos.scene import SceneProfile

#: name -> cube transform, including the two traps: ``bw``, which a signed "a rich frame
#: wants less chroma" demand hands every low-headroom scene, and ``crush``, whose rolloff
#: reads mild over 0-1 and eats a real frame's whole shadow range.
LOOKS = {
    "neutral": lambda r, g, b: (r, g, b),
    "warm": lambda r, g, b: (min(1.0, r * 1.18), g, b * 0.82),
    "punch": lambda r, g, b: tuple(min(1.0, max(0.0, 0.4 + 1.3 * (v - 0.4))) for v in (r, g, b)),
    "bw": lambda r, g, b: ((r + g + b) / 3.0,) * 3,
    "crush": lambda r, g, b: tuple(max(0.0, (v - 0.3) / 0.7) for v in (r, g, b)),
}
BASE_SCENE = SceneProfile(
    exposure_class="normal",
    warmth="neutral",
    sat_profile="normal",
    haze="clear",
    dynamic_range="normal",
    is_near_monochrome=False,
    chroma_headroom=0.5,
)
#: A frame in the range a graded photo here ACTUALLY occupies — not a 0-1 lattice.
_L = np.linspace(36.0, 168.0, 24)
FRAME = np.stack(np.meshgrid(_L, _L, _L, indexing="ij"), -1).reshape(1, -1, 3).astype(np.uint8)


def _write_photo(path):  # warm ground under a blue sky: real chroma and range
    y = np.linspace(0.0, 1.0, 96, dtype=np.float32)[:, None] * np.ones((1, 128), np.float32)
    rgb = np.stack([0.22 + 0.50 * y, 0.34 + 0.28 * y, 0.78 - 0.42 * y], -1)
    rgb += np.random.default_rng(4).normal(0.0, 0.03, rgb.shape).astype(np.float32)
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), "RGB").save(path, "JPEG")
    return path


def _cube(path, look="neutral", size=9):
    fn, axis = LOOKS[look], np.linspace(0.0, 1.0, size)
    rows = ["%.6f %.6f %.6f" % fn(r, g, b) for b in axis for g in axis for r in axis]
    path.write_text("\n".join([f"LUT_3D_SIZE {size}", *rows]) + "\n")
    return lut_mod.load_cube(path)


def _library(tmp_path, names=tuple(LOOKS)):
    return {n: _cube(tmp_path / f"{n}.cube", n) for n in names}


def _scene(**kwargs):
    return replace(BASE_SCENE, **kwargs)


def _ranked(cubes, scene):
    return [n for n, _, _ in lut_mod.select_luts(list(cubes.items()), scene, FRAME, len(cubes))]


def test_manual_lut_path_is_byte_identical_when_auto_is_zero(tmp_path):
    """``lut_auto=0`` renders the pre-change manual surface, byte for byte."""
    photo = _write_photo(tmp_path / "IMG_1000.jpg")
    cube = _cube(tmp_path / "film.cube", "warm")
    plan = dev.plan_variants(photo, ["natural"], 1.0, [("film", cube)])
    denoised, _ = dev._denoise(plan.decoded.rgb, plan.base)
    graded = lut_mod.apply_cube(dev._grade(denoised, plan.base), cube, 1.0)
    sharpened, _ = dev._sharpen(dev._clarity(graded, plan.clarity), plan.sharpen)
    expected = dev._grain(dev._vignette(sharpened, plan.vignette), plan.grain)

    options = dev.DevelopOptions(lut_auto=0)
    rendered = dict(dev.develop_variants(photo, ["natural"], 1.0, [("film", cube)], options))
    assert "film" in rendered, "the manual look keeps its unsuffixed name"
    assert np.array_equal(rendered["film"].rgb, expected)


def test_cubes_characterise_on_the_axis_they_move(tmp_path):
    t = {n: lut_mod.characterize_cube(c, FRAME) for n, c in _library(tmp_path).items()}
    assert t["neutral"] == pytest.approx((0.0, 1.0, 1.0, 0.0), abs=1e-3)  # warm/contrast/sat/crush
    assert t["warm"].warm_shift > 0.03 > abs(t["punch"].warm_shift)
    assert t["punch"].contrast_gain > 1.15 > t["warm"].contrast_gain
    assert t["bw"].saturation_gain == pytest.approx(0.0, abs=1e-3)
    assert t["crush"].crushed > lut_mod.LUT_MAX_CRUSH  # measured where the photo LIVES


def test_scoring_matches_the_cube_to_the_scene_deterministically(tmp_path):
    cubes = _library(tmp_path)
    cubes["alpha"] = _cube(tmp_path / "alpha.cube", "warm")  # scores tie with "warm"
    # An already-warm frame must not cook — _golden's restraint, mirrored. A rich frame
    # wants a chroma LEAN, not zero chroma: scoring every axis as distance from a TARGET
    # (never "less is better") is what stops the one B&W cube in a library winning every
    # low-headroom scene, as it did on the real 35-cube one.
    warm_rich = _scene(warmth="warm", sat_profile="rich", chroma_headroom=0.15)
    assert _ranked(cubes, warm_rich)[0] == "neutral"
    # A flat muted frame wants spread and chroma — but never bought by killing shadows.
    muted_flat = _scene(sat_profile="muted", dynamic_range="flat", chroma_headroom=0.95)
    assert _ranked(cubes, muted_flat)[0] == "punch"
    assert _ranked(cubes, muted_flat)[-1] == "crush"
    cool_hazy = _scene(warmth="cool", haze="hazy", dynamic_range="flat", chroma_headroom=0.8)
    winners = {_ranked(cubes, s)[0] for s in (warm_rich, muted_flat, cool_hazy)}
    assert len(winners) >= 2, f"collapsed onto one LUT: {winners}"
    ranking = _ranked(cubes, cool_hazy)
    assert ranking == _ranked(cubes, cool_hazy) == _ranked(cubes, cool_hazy)
    assert ranking.index("alpha") < ranking.index("warm"), "identical scores break by name"


def _run(cmd):  # ── CLI / config exposure ──
    return photos_handler.cmd_photos(_build_parser().parse_args(cmd.split()), None)


def _use_config(monkeypatch, tmp_path, names=tuple(LOOKS), **kwargs):
    _library(tmp_path, names)
    photos = PhotosConfig(luts={n: str(tmp_path / f"{n}.cube") for n in names}, **kwargs)
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: Config(photos=photos))


def test_lut_auto_renders_n_auto_named_looks_and_shows_why(tmp_path, monkeypatch, capsys):
    """Composition rule: explicit ``--lut`` specs BECOME the candidate pool."""
    photo = _write_photo(tmp_path / "IMG_1001.jpg")
    out = tmp_path / "out"
    _use_config(monkeypatch, tmp_path, developed_output_dir=str(out))
    specs = f"--lut {tmp_path / 'neutral.cube'} --lut {tmp_path / 'warm.cube'}"
    assert _run(f"photos develop {photo} --styles natural --lut-auto 1 {specs}") == 0
    assert len(list(out.glob("IMG_1001_*_auto.jpg"))) == 1, sorted(p.name for p in out.iterdir())
    assert _run(f"photos develop {photo} --dry-run --styles natural --lut-auto 2 {specs}") == 0
    printed = capsys.readouterr().out
    assert printed.count("score=") == 2, "the dry run names each pick and WHY it won"


def test_lut_auto_over_the_candidate_count_is_rejected(tmp_path, monkeypatch, capsys):
    photo = _write_photo(tmp_path / "IMG_1002.jpg")
    _use_config(monkeypatch, tmp_path, ("neutral",), developed_output_dir=str(tmp_path / "out"))
    monkeypatch.setattr(dev, "decode", lambda p: pytest.fail("decoded despite a bad --lut-auto"))
    assert _run(f"photos develop {photo} --lut-auto 4") == 1
    err = capsys.readouterr().err
    assert "--lut-auto" in err and "4" in err and "[0, 1]" in err
    assert PhotosConfig().lut_auto == 0 and dev.DevelopOptions().lut_auto == 0  # OFF default
    with pytest.raises(ValidationError):
        PhotosConfig(lut_auto=-1)  # the config default is gated the same way
