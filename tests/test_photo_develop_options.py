"""``DevelopOptions``: the operator look/spatial knobs as ONE frozen record.

The knobs used to travel as 17 positionals repeated across two signatures, so
every new creative feature appended at the same shared line. They now travel as
fields. This file pins the two properties that make that swap safe: every
default is still the OFF/neutral value, and a default record renders exactly
what the explicitly-neutral one does.
"""

import dataclasses

import numpy as np
import pytest
from PIL import Image

from gomoveshift.pipeline.photos import develop as dev

# The pre-record defaults, written as literals so an accidental change to any
# one of them fails here loudly instead of silently re-grading every photo.
NEUTRAL = {
    "vignette": 0.0,
    "dehaze": 0.0,
    "sky_darken": 0.0,
    "sky_saturate": 0.0,
    "graduated_nd": 0.0,
    "graduated_nd_angle": 0.0,
    "glow": 0.0,
    "clarity": 0.0,
    "grain": 0.0,
    "grain_size": 1.0,
    "halation": 0.0,
    "halation_tint": (),
    "light_leak": 0.0,
    "chromatic_ab": 0.0,
    "holo_sheen": 0.0,
    "scanline": 0.0,
    "glitch": 0.0,
    "sharpen": 1.0,
    "hue_shift": (),
    "hue_sat": (),
    "hue_lum": (),
    "splittone_shadow": (),
    "splittone_midtone": (),
    "splittone_highlight": (),
    "curve_l": (),
    "curve_r": (),
    "curve_g": (),
    "curve_b": (),
    "white_balance_r": 0.0,
    "white_balance_b": 0.0,
    "white_balance_g": 0.0,
    "white_balance_r_high": 0.0,
    "white_balance_b_high": 0.0,
    "white_balance_g_high": 0.0,
    "white_balance_kelvin": 0.0,
    "white_balance_tint": 0.0,
    "exposure_ev": 0.0,
    "shadow_recovery": 0.0,
    "highlight_recovery": 0.0,
    "vibrance": 0.0,
    "mono_mix": (),
    "duotone": (),
    "duotone_strength": 1.0,
    "duotone_mid": (),
    "reference_lab": (),
    "layers": (),
    "look_strength": 1.0,
    "lut_strength": 1.0,
    "lut_over": "",
    "lut_auto": 0,
    "straighten_angle": 0.0,
    "crop_rect": (),
    "lens_correct": False,
}


def _write_photo(path):
    rng = np.random.default_rng(11)
    Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8), "RGB").save(path, "JPEG")
    return path


def test_every_default_is_the_off_neutral_value():
    """Field set AND defaults, both directions: a knob added without an
    OFF default (or renamed) fails here before it can move any pixels."""
    options = dev.DevelopOptions()
    assert {f.name: getattr(options, f.name) for f in dataclasses.fields(options)} == NEUTRAL


def test_the_record_is_frozen_so_one_shared_default_instance_is_safe():
    """``options: DevelopOptions = DevelopOptions()`` shares ONE instance across
    every call — safe only because no caller can mutate it."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        dev.DevelopOptions().vignette = 0.5


def test_default_record_renders_byte_identically_to_explicit_neutral_values(tmp_path):
    photo = _write_photo(tmp_path / "IMG_OPT.jpg")
    default = dict(dev.develop_variants(photo, ["natural"], options=dev.DevelopOptions()))
    explicit = dict(dev.develop_variants(photo, ["natural"], options=dev.DevelopOptions(**NEUTRAL)))
    assert np.array_equal(default["natural"].rgb, explicit["natural"].rgb)
