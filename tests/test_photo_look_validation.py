"""A look's LIST knobs are refused at the door, not per photo, mid-batch.

``LOOK_RANGES`` gates scalars only, so every list knob a look or a masked layer
can carry — the three ``curve_*``, the three hue bands, the two split tones,
``bw_mix``, ``duotone`` — reached the render unchecked and raised inside
``CorrectionParams``, once per photo, after a batch of decodes had already been
paid for. ``--look-variant`` had no range gate at all, so the same configured
look that ``--look`` refused rendered happily through the variant flag.
"""

import argparse

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

import gomoveshift.config as config_module
from gomoveshift.cli.handlers import photos as photos_handler
from gomoveshift.cli.handlers.photos import _apply_look
from gomoveshift.cli.parser import _build_parser
from gomoveshift.config import Config, PhotosConfig

# Legal by every rule below: x strictly ascending in [0, 1], y in [0, 1], six
# band slots in [0, 5], a three-value tint inside [-0.5, 0.5].
GOOD = {
    "curve_r": [0.0, 0.05, 0.5, 0.55, 1.0, 1.0],
    "hue_sat": [1.1, 1.0, 0.9, 1.2, 1.3, 1.0],
    "split_tone_shadow": [-0.04, 0.01, 0.12],
}


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr(photos_handler, "configure_logging", lambda *a, **k: None)


def _write_photo(path):
    rng = np.random.default_rng(7)
    Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8), "RGB").save(path, "JPEG")
    return path


def _run(argv, monkeypatch, looks=None):
    photos = PhotosConfig(looks=looks or {})
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: Config(photos=photos))
    return photos_handler.cmd_photos(_build_parser().parse_args(argv.split()), None)


def _use_builtin_looks(monkeypatch, table):
    from gomoveshift.pipeline.photos import looks as builtin

    monkeypatch.setattr(builtin, "BUILTIN_LOOKS", table)


@pytest.mark.parametrize(
    "knobs,named",
    [
        ({"curve_r": [1.0, 0.0, 0.0, 1.0]}, "ascending"),
        ({"curve_g": [0.0, 0.0, 0.5]}, "curve_g"),
        ({"curve_b": [0.0, 0.0, 1.0, 1.4]}, "curve_b"),
        ({"hue_sat": [1.0] * 5}, "hue_sat"),
        ({"hue_sat": [9.0] * 6}, "hue_sat"),
        ({"hue_shift": [400.0] * 6}, "hue_shift"),
        ({"split_tone_shadow": [0.0, 0.9, 0.0]}, "split_tone_shadow"),
        ({"split_tone_highlight": [0.1, 0.1]}, "split_tone_highlight"),
        ({"bw_mix": [0.0, 0.0, 0.0]}, "bw_mix"),
        ({"duotone": [0.0, 0.0, 0.0, 1.0, 1.0, 1.4]}, "duotone"),
        ({"hue_lum": "bright"}, "hue_lum"),
    ],
)
def test_config_load_refuses_a_malformed_list_knob_naming_the_look(knobs, named):
    """Upstream gate: a bad list costs a batch of decodes to find at render time."""
    with pytest.raises(ValidationError) as err:
        PhotosConfig(looks={"split_world": knobs})
    assert "split_world" in str(err.value) and named in str(err.value)


@pytest.mark.parametrize(
    "layer,named",
    [
        ({"mask": "sky", "curve_r": [1.0, 0.0, 0.0, 1.0]}, "ascending"),
        ({"mask": "sky", "hue_sat": [1.0] * 5}, "hue_sat"),
        ({"mask": "ground", "split_tone_shadow": [0.0, 0.9, 0.0]}, "split_tone_shadow"),
    ],
)
def test_a_malformed_layer_list_never_reaches_correction_params(layer, named, monkeypatch):
    """The old home of this refusal: ``CorrectionParams.__post_init__``, reached
    once per photo from ``_layered``, mid-batch — and blind to a range besides.
    Exploding it proves the refusal below is the config door's, not the render's."""
    import gomoveshift.pipeline.color_correction as cc

    def _boom(self):
        raise AssertionError("CorrectionParams reached")

    monkeypatch.setattr(cc.CorrectionParams, "__post_init__", _boom)
    with pytest.raises(ValidationError) as err:
        PhotosConfig(looks={"split_world": {"layers": [layer]}})
    assert "split_world" in str(err.value) and named in str(err.value)


def test_both_look_doors_refuse_the_same_out_of_range_band(tmp_path, monkeypatch, capsys):
    """``--look`` refused it through the flag gates and ``--look-variant`` did not,
    so one table of knobs meant two different answers. A built-in carries the bad
    band here because a *configured* one can no longer be constructed at all."""
    photo = _write_photo(tmp_path / "IMG_1.jpg")
    _use_builtin_looks(monkeypatch, {"hot": {"styles": "natural", "hue_sat": [9.0] * 6}})
    assert _run(f"photos develop {photo} --out {tmp_path / 'a'} --look hot", monkeypatch) == 1
    assert "hue-sat" in capsys.readouterr().err

    out = tmp_path / "b"
    assert _run(f"photos develop {photo} --out {out} --look-variant hot", monkeypatch) == 1
    assert "hot" in capsys.readouterr().err
    assert not (out.exists() and list(out.glob("*.jpg"))), "nothing decoded or written"


def test_a_seven_anchor_band_reaches_a_look_through_every_door(tmp_path, monkeypatch):
    """Orange at 30°, second slot: legal on ``--hue-sat``, legal in ``photos.hue_sat``,
    legal in ``CorrectionParams`` — and unreachable from a look. Every look door routes
    through ``check_look_lists``, whose registry row could only say ONE exact length, so
    no configured look, masked layer or ``--look-variant`` could carry the anchor the
    engine already renders."""
    photo = _write_photo(tmp_path / "IMG_3.jpg")
    seven = [1.0, 1.6, 1.0, 1.0, 1.0, 1.0, 1.0]  # orange second, the rest neutral
    looks = {"citrus": {"hue_sat": seven, "layers": [{"mask": "sky", "hue_lum": seven}]}}
    assert PhotosConfig(looks=looks).looks == looks, "config load refused seven"
    config_module.check_layers("--layers", [{"mask": "sky", "hue_shift": [0.0] * 7}])
    out = tmp_path / "e"
    argv = f"photos develop {photo} --out {out} --styles natural --look-variant citrus"
    assert _run(argv, monkeypatch, looks=looks) == 0
    assert (out / "IMG_3_look_citrus.jpg").exists(), "the seven-slot look rendered no panel"


def test_the_band_refusal_names_both_legal_lengths():
    """Six or seven, and nothing else — the door still refuses every other length, and
    says which two it takes rather than naming one and hiding the other."""
    with pytest.raises(ValidationError) as err:
        PhotosConfig(looks={"citrus": {"hue_sat": [1.0] * 8}})
    assert "6 or 7" in str(err.value), "the refusal named only one of the two lengths"


def _all_actions():
    """Every argparse action in the CLI, subparsers walked."""
    stack, seen = [_build_parser()], []
    while stack:
        parser = stack.pop()
        for action in parser._actions:
            if isinstance(getattr(action, "choices", None), dict):
                stack.extend(action.choices.values())
            seen.append(action)
    return seen


def test_a_look_naming_a_lut_is_refused_at_the_door():
    """``--lut`` APPENDS, so ``args.lut`` is ``None`` or a LIST of specs. ``lut`` was
    a registered look knob all the same, so ``photos.looks.vintage.lut: portra``
    loaded clean and ``_apply_look``'s scalar fill wrote the bare STRING into it —
    ``_resolve_luts`` then iterated it and the operator got a resolution failure
    naming the single letter ``p``. The LUT is the RUN's choice: refused at LOAD,
    naming the look, so no such string can exist to be spelled out."""
    with pytest.raises(ValidationError) as err:
        PhotosConfig(looks={"vintage": {"lut": "portra"}})
    message = str(err.value)
    assert "vintage" in message, "the refusal must name the look"
    assert "--lut" in message, "the refusal must point at the run-level flag"


def test_no_look_knob_may_fill_an_append_flag():
    """The bug class, not the one instance: ``_apply_look`` fills a flag dest with
    the look's raw value, which is only sound where that dest holds a scalar. A look
    knob whose flag APPENDS holds a list, so every scalar it is given is read
    element-wise — a string, character by character."""
    from gomoveshift.config import LOOK_KNOBS

    appends = {a.dest for a in _all_actions() if isinstance(a, argparse._AppendAction)}
    assert not appends & LOOK_KNOBS, (
        f"look knobs whose flag appends: {sorted(appends & LOOK_KNOBS)} — a look fills "
        "each with a bare value the resolver then iterates"
    )


def test_the_lut_flag_and_a_look_carrying_no_lut_are_untouched():
    """The blast radius: the refusal is the look door's alone. ``--lut`` still
    collects its specs as a list, and a look that names no LUT still fills."""
    args = _build_parser().parse_args("photos develop IMG_1.jpg --lut portra --lut kodak".split())
    _apply_look(args, PhotosConfig(looks={"warm": GOOD}).looks["warm"])
    assert args.lut == ["portra", "kodak"], "the flag still appends its own specs"
    assert args.curve_r == GOOD["curve_r"], "a look with no lut key still fills"


def test_a_valid_look_still_loads_and_renders_through_both_doors(tmp_path, monkeypatch):
    """The gate moves WHERE a value is refused, never WHICH values are legal."""
    photo = _write_photo(tmp_path / "IMG_2.jpg")
    looks = {"warm": {**GOOD, "layers": [{"mask": "sky", "hue_sat": [1.2] * 6}]}}
    assert PhotosConfig(looks=looks).looks == looks
    out = tmp_path / "d"
    argv = f"photos develop {photo} --out {out} --styles natural --look-variant warm"
    assert _run(argv, monkeypatch, looks=looks) == 0
    assert (out / "IMG_2_look_warm.jpg").exists(), "the valid look rendered its own panel"
