"""The ELECTRONIC built-in look pack: cyberpunk / synthwave / chrome / neon_noir.

A preset that fails the config validator, renders on top of a pack-mate, or
drifts a knob's SIGN is worthless in the box, and none of that shows in a table
of numbers. So every preset is rendered through the real ``photos develop
--look`` door on a synthesized frame and judged on pixels: it must reach
``MIN_RENDERED_DISTINCTNESS`` from the natural grade AND from each pack-mate,
reproduce byte for byte, and move colour in the DIRECTION its name promises.

Direction is measured against the look's OWN base style, never against natural:
these presets pick contrasty bases, so a look whose colour knobs were deleted
outright still lands far from natural and a natural-anchored assertion passes a
gutted recipe. The ratio floors below are set where a gutted recipe measures
(each is commented) — no magic pixel values; taste is set on the proof sheet.
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

PACK = ("cyberpunk", "synthwave", "chrome", "neon_noir")

# The synthesized frame's geometry, so a test names a region instead of hunting
# percentiles: luma ramps down the ROWS, the left half is neutral (a split-tone
# or curve move is the only thing that can colour it) and the right half sweeps
# every anchor hue (where a per-hue saturation move has something to bite on).
HEIGHT, WIDTH = 96, 128
FLAT, SWEEP = slice(0, WIDTH // 2), slice(WIDTH // 2, WIDTH)
TOE, TOP = slice(0, HEIGHT // 4), slice(3 * HEIGHT // 4, HEIGHT)

# Ratio floors, set where a GUTTED recipe measures. Neutralise the per-hue move
# and halation alone still drains the warm end to ~0.85 of what the neon end
# keeps, and the base style alone still darkens to ~0.88; the shipped presets
# reach 0.43-0.55 and 0.57. These thresholds sit between the two.
MAX_DRAIN_RATIO = 0.70
MAX_CRUSH_RATIO = 0.80
# A grey duotone separates its channels by <1 level; the shipped ramp by ~70.
MIN_TONE_SEPARATION = 20.0


def _write_source(path):
    """Synthesized only — no NEF, no /mnt/d, no ffmpeg."""
    hue = np.tile(np.linspace(0, 255, WIDTH, dtype=np.uint8), (HEIGHT, 1))
    value = np.repeat(np.linspace(12, 250, HEIGHT, dtype=np.uint8)[:, None], WIDTH, axis=1)
    sat = np.zeros_like(hue)
    sat[:, SWEEP] = 190
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


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    """Each look, the natural grade and every look's bare base style, rendered
    ONCE against one frame: {name | "base:<style>": float RGB}."""
    root = tmp_path_factory.mktemp("electronic")
    photo = _write_source(root / "IMG_0001.jpg")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(photos_handler, "configure_logging", lambda *a, **k: None)
        mp.setattr(config_module, "get_config", lambda *a, **k: Config(photos=PhotosConfig()))
        frames = {"natural": _develop(photo, root / "natural", "--styles natural")}
        frames["base:natural"] = frames["natural"]
        for name in PACK:
            frames[name] = _develop(photo, root / name, f"--look {name}")
            style = _ELECTRONIC[name]["styles"]
            if f"base:{style}" not in frames:
                frames[f"base:{style}"] = _develop(photo, root / f"b_{style}", f"--styles {style}")
    return frames


def _base(rendered, name):
    """The look's own base style rendered BARE — its colour knobs' reference."""
    return rendered[f"base:{_ELECTRONIC[name]['styles']}"]


def _luma(rgb):
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _chroma(rgb):
    return rgb.max(axis=2) - rgb.min(axis=2)


def _teal(frame):
    """How far green+blue lead red."""
    return float(np.mean((frame[..., 1] + frame[..., 2]) / 2 - frame[..., 0]))


def _magenta(frame):
    """How far red+blue lead green."""
    return float(np.mean((frame[..., 0] + frame[..., 2]) / 2 - frame[..., 1]))


def _chroma_ratio(look, base, warm):
    """Look-over-base chroma on the sweep half's warm (or cool) pixels."""
    mask = (base[:, SWEEP, 0] > base[:, SWEEP, 2]) == warm
    return float(np.mean(_chroma(look[:, SWEEP])[mask]) / np.mean(_chroma(base[:, SWEEP])[mask]))


def _drain(look, base):
    """How much of its chroma the warm end keeps RELATIVE to the neon end.

    Not the bare warm ratio: a heavy halation tint drags every hue's chroma
    around on its own, so the bare number passes a look whose per-hue move was
    deleted. Measured against the same frame's cyan/blue end, only a real
    selective desaturation moves it."""
    return _chroma_ratio(look, base, warm=True) / _chroma_ratio(look, base, warm=False)


def _distance(a, b):
    return float(np.mean(np.abs(a - b)))


def test_the_pack_ships_four_presets_the_real_config_validator_accepts():
    # A subset, not the whole dict: the artifact three (vhs / glitch_pop /
    # leaked_film) joined this pack later and carry tests/test_photo_artifact_looks.py.
    # That this PACK plus that one still cover _ELECTRONIC whole is not proved here —
    # tests/test_photo_look_packs_cover_the_table.py holds that gate for all four packs.
    assert set(PACK) <= set(_ELECTRONIC)
    # The validator an operator's own config.yaml hits — a preset that failed it
    # would ship a built-in table the CLI refuses to accept as a config override.
    assert PhotosConfig(looks=_ELECTRONIC).looks == _ELECTRONIC
    assert all(BUILTIN_LOOKS[name] == _ELECTRONIC[name] for name in PACK)


@pytest.mark.parametrize("name", PACK)
def test_each_look_renders_visibly_apart_from_the_natural_grade(name, rendered):
    measured = _distance(rendered[name], rendered["natural"])
    assert measured >= MIN_RENDERED_DISTINCTNESS, (
        f"{name} renders {measured:.2f} levels off natural"
    )


@pytest.mark.parametrize("first,second", [(a, b) for a in PACK for b in PACK if a < b])
def test_every_pair_in_the_pack_renders_apart(first, second, rendered):
    """Four names, four looks — not one look wearing four names."""
    measured = _distance(rendered[first], rendered[second])
    assert measured >= MIN_RENDERED_DISTINCTNESS, f"{first} vs {second}: {measured:.2f} levels"


def test_a_look_renders_byte_identically_twice(tmp_path, monkeypatch):
    """Grain and halation are the two stages that could carry run-to-run noise;
    both draw from fixed seeds, and a look that did not reproduce would make
    every eyeball calibration unrepeatable."""
    monkeypatch.setattr(photos_handler, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: Config(photos=PhotosConfig()))
    photo = _write_source(tmp_path / "IMG_0002.jpg")
    first = _develop(photo, tmp_path / "one", "--look synthwave")
    second = _develop(photo, tmp_path / "two", "--look synthwave")
    assert np.array_equal(first, second)


def test_cyberpunk_cools_the_shadows_and_pushes_the_highlights_magenta(rendered):
    look, base = rendered["cyberpunk"], _base(rendered, "cyberpunk")
    assert _teal(look[TOE, FLAT]) > _teal(base[TOE, FLAT]), "shadows must move toward teal"
    assert _magenta(look[TOP, FLAT]) > _magenta(base[TOP, FLAT]), "highlights toward magenta"
    assert _chroma_ratio(look, base, warm=False) > 1.0, "the neon end gains chroma, not loses it"


def test_synthwave_tones_the_whole_frame_purple_into_pink(rendered):
    look = rendered["synthwave"]
    green = np.mean(look[..., 1])
    assert min(np.mean(look[..., 0]), np.mean(look[..., 2])) - green > MIN_TONE_SEPARATION, (
        "green is the floor of the violet-to-pink tone: a colour, not a grey"
    )
    assert np.mean(look[TOE, FLAT, 2]) > np.mean(look[TOE, FLAT, 0]), "the shadow end is purple"
    assert np.mean(look[TOP, FLAT, 0]) > np.mean(look[TOP, FLAT, 2]), "the highlight end is pink"


def test_chrome_cools_the_frame_and_drains_every_hue_but_blue_and_cyan(rendered):
    look, base = rendered["chrome"], _base(rendered, "chrome")

    def cool(frame):
        return float(np.mean(frame[:, FLAT, 2] - frame[:, FLAT, 0]))

    assert cool(look) > cool(base), "chrome is cool at BOTH ends, not split"
    assert _drain(look, base) < MAX_DRAIN_RATIO, "the warm end must drain"
    assert _chroma_ratio(look, base, warm=False) >= 0.9, "blue and cyan are what survives"


def test_neon_noir_crushes_the_tone_and_keeps_only_the_neon_hues(rendered):
    look, base = rendered["neon_noir"], _base(rendered, "neon_noir")
    assert np.mean(_luma(look)) / np.mean(_luma(base)) < MAX_CRUSH_RATIO, "low-key means darker"
    assert _drain(look, base) < MAX_DRAIN_RATIO, "the warm end must drain"
    assert _chroma_ratio(look, base, warm=False) >= 1.0, "the cyan/magenta signage is HELD"
