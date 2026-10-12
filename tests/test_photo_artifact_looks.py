"""The ARTIFACT half of the ELECTRONIC pack: vhs / glitch_pop / leaked_film.

Six spatial stages (glow, light-leak, chromatic-ab, holo-sheen, scanline, glitch)
shipped with nothing in the box using any of them; these three presets are the
recipes that do. Each is NAMED for one stage, so the assertion that carries this
file renders the SAME recipe with that ONE knob stripped -- as an operator-config
look, through the real ``photos develop --look`` door -- and measures the structure
only that stage can make. Anchoring on the natural grade instead passes a recipe
whose artifact knob was deleted outright: the base style and the colour knobs
already land it far from natural, which is exactly how a look that merely
validates gets mistaken for a look.

Floors are set where the stripped twin measures (each is commented) -- no magic
pixel values. Strength calibration stays a manual eyeball gate; a hermetic test can
only pin that the presets are legal, distinct, deterministic and that each one
actually engages the stage it is named for.
"""

import numpy as np
import pytest
from PIL import Image

import gomoveshift.config as config_module
from gomoveshift.cli.handlers import photos as photos_handler
from gomoveshift.cli.parser import _build_parser
from gomoveshift.config import Config, PhotosConfig
from gomoveshift.pipeline.photos.develop import MIN_RENDERED_DISTINCTNESS
from gomoveshift.pipeline.photos.looks import _ELECTRONIC, BUILTIN_LOOKS

PACK = ("vhs", "glitch_pop", "leaked_film")
# The stage each preset is named for — and the one knob its stripped twin drops.
STAGE = {"vhs": "scanline", "glitch_pop": "glitch", "leaked_film": "light_leak"}
BARE = "-bare"  # suffix of the stripped twin's configured-look name

# The synthesized frame: luma ramps down the ROWS (so the top of the frame is the
# dark end), the left half is neutral and the right half sweeps every anchor hue.
HEIGHT, WIDTH = 96, 128
# Geometry, not tone: ``LIGHT_LEAK_CORNER`` is top-right, and the corner diagonally
# opposite it sits past ``LIGHT_LEAK_REACH`` of the diagonal by construction.
TOP, BOTTOM = slice(0, HEIGHT // 3), slice(2 * HEIGHT // 3, HEIGHT)
LEFT, RIGHT = slice(0, WIDTH // 3), slice(2 * WIDTH // 3, WIDTH)

# Spectral bins below this carry the source's own smooth top-to-bottom ramp and the
# grade's response to it; a tube's line structure lands far above them.
RIPPLE_MIN_BIN = 6
# A scanline-free twin's residual ripple is broadband noise off grain and the JPEG
# encode (0.10 levels measured); the shipped preset's line measures 2.74 — 28x.
MIN_RIPPLE_RATIO = 8.0
# Rows count as torn when they move by more than JPEG round-off between two renders.
TORN_ROW_LEVELS = 1.0
# ``GLITCH_TORN_FRACTION`` tears half the bands at amount 1.0, so a conservative
# amount moves well under half the rows; a whole-frame move is not a band tear.
MAX_TORN_FRACTION = 0.5


def _write_source(path):
    """Synthesized only — no NEF, no /mnt/d, no ffmpeg."""
    hue = np.tile(np.linspace(0, 255, WIDTH, dtype=np.uint8), (HEIGHT, 1))
    value = np.repeat(np.linspace(12, 250, HEIGHT, dtype=np.uint8)[:, None], WIDTH, axis=1)
    sat = np.zeros_like(hue)
    sat[:, WIDTH // 2 :] = 190
    Image.fromarray(np.dstack([hue, sat, value]), "HSV").convert("RGB").save(
        path, "JPEG", quality=95
    )
    return path


def _develop(photo, out, argv_extra):
    """Run the real CLI develop and return its ONE exported variant as float RGB."""
    argv = f"photos develop {photo} --out {out} {argv_extra}".split()
    assert photos_handler.cmd_photos(_build_parser().parse_args(argv), None) == 0
    exports = sorted(out.glob("*.jpg"))
    assert len(exports) == 1, f"expected one variant, got {[p.name for p in exports]}"
    return np.asarray(Image.open(exports[0]).convert("RGB")).astype(np.float64)


def _stripped():
    """Each preset MINUS the one stage it is named for, keyed ``<name>-bare`` —
    derived from the shipped recipe, so a preset that lost its artifact knob makes
    the twin identical to it and every assertion below goes red."""
    return {
        f"{name}{BARE}": {k: v for k, v in _ELECTRONIC[name].items() if k != knob}
        for name, knob in STAGE.items()
    }


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    """Every ``_ELECTRONIC`` preset, the natural grade and each new preset's
    stripped twin, rendered ONCE against one frame: {name: float RGB}."""
    root = tmp_path_factory.mktemp("artifact")
    photo = _write_source(root / "IMG_0001.jpg")
    # The twins ride ``photos.looks``, the operator half of the same registry — so
    # they reach the identical door, resolution and chain the shipped preset does.
    config = Config(photos=PhotosConfig(looks=_stripped()))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(photos_handler, "configure_logging", lambda *a, **k: None)
        mp.setattr(config_module, "get_config", lambda *a, **k: config)
        frames = {"natural": _develop(photo, root / "natural", "--styles natural")}
        for name in (*sorted(_ELECTRONIC), *sorted(config.photos.looks)):
            frames[name] = _develop(photo, root / name.replace("-", "_"), f"--look {name}")
    return frames


def _luma(rgb):
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _distance(a, b):
    return float(np.mean(np.abs(a - b)))


def _row_ripple(frame):
    """Amplitude of the finest row-periodic structure in the frame's row-mean luma:
    the strongest spectral line above ``RIPPLE_MIN_BIN``. Averaging the columns
    first is what makes this a LINE measure — per-pixel texture (grain) averages
    away, while a gain that is one value per row survives whole. The Hann window is
    not optional: the source's top-to-bottom ramp is a step at the wrap, and its
    leakage alone measures 4+ levels at every bin, which is larger than the line
    being looked for."""
    rows = _luma(frame).mean(axis=1)
    spectrum = np.abs(np.fft.rfft((rows - rows.mean()) * np.hanning(rows.size)))
    return float(spectrum[RIPPLE_MIN_BIN:].max() / rows.size)


def _torn_rows(look, bare):
    """Fraction of the frame's rows that moved between the two renders."""
    return float(np.mean(np.abs(look - bare).mean(axis=(1, 2)) > TORN_ROW_LEVELS))


def _lift(look, bare, rows, cols, channel=slice(None)):
    """How much the region brightened between the twin and the shipped preset."""
    return float(np.mean(look[rows, cols, channel]) - np.mean(bare[rows, cols, channel]))


def test_the_pack_ships_three_artifact_presets_the_real_config_validator_accepts():
    assert set(PACK) <= set(_ELECTRONIC)
    # The validator an operator's own config.yaml hits — a preset that failed it
    # would ship a built-in table the CLI refuses to accept as a config override.
    recipes = {name: _ELECTRONIC[name] for name in PACK}
    assert PhotosConfig(looks=recipes).looks == recipes
    assert all(BUILTIN_LOOKS[name] == _ELECTRONIC[name] for name in PACK)
    # Each is named for a stage it actually carries — the recipe IS the claim.
    assert all(_ELECTRONIC[name][knob] > 0.0 for name, knob in STAGE.items())


@pytest.mark.parametrize("name", PACK)
def test_each_look_renders_visibly_apart_from_the_natural_grade(name, rendered):
    measured = _distance(rendered[name], rendered["natural"])
    assert measured >= MIN_RENDERED_DISTINCTNESS, (
        f"{name} renders {measured:.2f} levels off natural"
    )


@pytest.mark.parametrize(
    "first,second",
    [(a, b) for a in sorted(_ELECTRONIC) for b in sorted(_ELECTRONIC) if a < b and b in PACK],
)
def test_every_new_preset_renders_apart_from_every_pack_mate(first, second, rendered):
    """Seven names in the pack, seven looks — not four looks wearing seven names."""
    measured = _distance(rendered[first], rendered[second])
    assert measured >= MIN_RENDERED_DISTINCTNESS, f"{first} vs {second}: {measured:.2f} levels"


def test_a_look_renders_byte_identically_twice(tmp_path, monkeypatch):
    """glitch_pop is the strictest case: ``_glitch`` draws its tear pattern from an
    rng, and a look that did not reproduce would make every eyeball calibration —
    and the develop cache — unrepeatable."""
    monkeypatch.setattr(photos_handler, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: Config(photos=PhotosConfig()))
    photo = _write_source(tmp_path / "IMG_0002.jpg")
    first = _develop(photo, tmp_path / "one", "--look glitch_pop")
    second = _develop(photo, tmp_path / "two", "--look glitch_pop")
    assert np.array_equal(first, second)


def test_vhs_lays_a_line_structure_its_scanline_free_twin_has_none_of(rendered):
    look, bare = rendered["vhs"], rendered[f"vhs{BARE}"]
    ripple, residual = _row_ripple(look), _row_ripple(bare)
    assert ripple > MIN_RIPPLE_RATIO * residual, (
        f"vhs must read as a tube: {ripple:.3f} vs the scanline-free {residual:.3f}"
    )


def test_glitch_pop_tears_bands_of_rows_sideways_not_the_whole_frame(rendered):
    look, bare = rendered["glitch_pop"], rendered[f"glitch_pop{BARE}"]
    torn = _torn_rows(look, bare)
    assert torn > 0.0, "the signal fault must actually displace rows"
    assert torn < MAX_TORN_FRACTION, f"{torn:.2f} of rows moved — that is a shift, not a tear"


def test_leaked_film_fogs_the_leak_corner_warm_and_leaves_the_far_one_alone(rendered):
    look, bare = rendered["leaked_film"], rendered[f"leaked_film{BARE}"]
    corner = _lift(look, bare, TOP, RIGHT)
    assert corner > 0.0, "the leak corner must gain light"
    assert corner > _lift(look, bare, BOTTOM, LEFT), (
        "a leak is a CORNER fog, not an exposure push on the whole frame"
    )
    assert _lift(look, bare, TOP, RIGHT, 0) > _lift(look, bare, TOP, RIGHT, 2), (
        "LIGHT_LEAK_TINT is warm: red must gain more than blue"
    )
