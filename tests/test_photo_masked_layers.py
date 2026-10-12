"""``photos.looks`` ``layers:`` — a colour recipe that rides ONE region of a frame.

Every colour knob on ``photos develop`` is whole-frame and ``masks.py``'s named
soft masks were a registry nothing could reach, so teal sky over warm ground,
background desaturation and a highlight-only tint were all inexpressible. A look
may now carry ``layers``: each is a mask name plus colour knobs, graded onto the
already-graded frame and lerped through its mask, composing over the global grade
in list order. No ``layers`` ⇒ not one line of the seam runs.

A layer may also name TWO masks plus a ``mask_op`` (``intersect``/``union``/
``subtract``), because a scene's real region is often narrower than any single
entry — ``highlights`` is every pixel above 2/3 luma, which on a landscape is a
third of the sky. The combinators run on the RAW pair, before feather and invert.
"""

import hashlib
import json
from dataclasses import fields

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

import gomoveshift.config as config_module
from gomoveshift.cli.handlers import photos as photos_handler
from gomoveshift.cli.parser import _build_parser
from gomoveshift.config import LAYER_MASK_PARAMS, LAYER_MASKS, Config, PhotosConfig
from gomoveshift.pipeline import color_correction as cc
from gomoveshift.pipeline.photos import develop as dev
from gomoveshift.pipeline.photos import masks as masks_mod
from gomoveshift.pipeline.photos.looks import _MASKED

# ``linear`` at its defaults ramps top (exactly 0) to bottom (exactly 1), so one
# mask hands a test a region AND a clean complement to assert on. The two recipes
# do not commute — a red flip after a chroma drain is not a drain after a flip.
GROUND = {"mask": "linear", "curve_r": [0.0, 1.0, 1.0, 0.0]}
DRAIN = {"mask": "radial", "hue_sat": [0.0] * 6}
# The TONE half: the Lightroom local move — lift the shadows, steepen the sky.
SHADOW_LIFT = {"mask": "shadows", "gamma": 1.25}
SKY_CONTRAST = {"mask": "sky", "contrast_curve": 0.15}

# A layer may instead name TWO masks and a ``mask_op``: the bottom gradient AND the
# centre, a region neither entry expresses on its own.
COMBO = {"mask": ["linear", "radial"], "mask_op": "intersect", "curve_r": [0.0, 1.0, 1.0, 0.0]}
PAIR = ["radial", "highlights"]


def _frame(h=64, w=96):
    """Warm ground under a blue sky, in this chain's real tonal range."""
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    rgb = np.empty((h, w, 3), np.float32)
    rgb[..., 0] = 0.20 + 0.44 * y
    rgb[..., 1] = 0.30 + 0.24 * y
    rgb[..., 2] = 0.66 - 0.36 * y
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def _horizon(h=96, w=128, sky=36):
    """A graded sky over dark ground, LANDSCAPE — ``_sky_mask`` returns all zeros on
    a portrait frame, and a FLAT sky holds no contrast for a curve to move."""
    rgb = np.empty((h, w, 3), np.float32)
    ramp = np.linspace(0.40, 0.80, sky, dtype=np.float32)[:, None, None]
    rgb[:sky] = np.array((0.62, 0.76, 1.0), np.float32) * ramp
    rgb[sky:] = (0.30, 0.24, 0.15)
    rgb[sky:] *= np.linspace(0.62, 1.30, w, dtype=np.float32)[None, :, None]
    return (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def _luma(frame):
    return 0.2126 * frame[..., 0] + 0.7152 * frame[..., 1] + 0.0722 * frame[..., 2]


def _write_photo(path):
    Image.fromarray(_frame(), "RGB").save(path, "JPEG", quality=95)
    return path


def _render(photo, **knobs):
    options = dev.DevelopOptions(**knobs)
    variants = dev.develop_variants(photo, ["natural"], options=options)
    return dict(variants)["natural"].rgb


def test_no_layers_renders_byte_identically_and_never_reaches_a_mask(tmp_path, monkeypatch):
    """The shipped path, and not merely a no-op that runs: with nothing configured
    the seam must not so much as meter a mask. The second half is what stops that
    from passing vacuously — with a layer, the mask IS reached."""
    photo = _write_photo(tmp_path / "IMG_0001.jpg")
    plain = _render(photo)

    def _boom(*_args, **_kwargs):
        raise AssertionError("mask_for reached")

    monkeypatch.setattr(masks_mod, "mask_for", _boom)
    assert np.array_equal(_render(photo), plain), "no layers must be the shipped render"
    with pytest.raises(AssertionError, match="mask_for reached"):
        _render(photo, layers=(GROUND,))


def test_a_layer_moves_only_where_its_mask_is_non_zero(tmp_path):
    """``sharpen=0`` on both renders because the unsharp mask is a NEIGHBOURHOOD
    op: it would smear the layered region across the mask boundary and the
    complement assertion would be measuring the wrong thing."""
    photo = _write_photo(tmp_path / "IMG_0002.jpg")
    base = _render(photo, sharpen=0.0)
    out = _render(photo, sharpen=0.0, layers=(GROUND,))
    region = masks_mod.mask_for("linear", base) > 0.0
    assert region.any() and not region.all(), "the fixture needs a region AND a complement"
    assert np.array_equal(out[~region], base[~region]), "a layer touched the mask's zero region"
    assert not np.array_equal(out[region], base[region]), "a layer moved nothing it named"


def test_invert_moves_the_complement_instead():
    frame = _frame()
    mask = masks_mod.mask_for("linear", frame)
    out = dev._layered(frame, ({**GROUND, "invert": True},))
    assert np.array_equal(out[mask >= 1.0], frame[mask >= 1.0]), "invert kept the mask's region"
    assert not np.array_equal(out[mask <= 0.0], frame[mask <= 0.0]), "invert moved no complement"


def test_layers_compose_over_the_graded_frame_in_list_order():
    frame = _frame()
    stepwise = dev._layered(dev._layered(frame, (GROUND,)), (DRAIN,))
    assert np.array_equal(dev._layered(frame, (GROUND, DRAIN)), stepwise), "not applied in order"
    assert not np.array_equal(dev._layered(frame, (DRAIN, GROUND)), stepwise), "order is inert"


def test_a_masked_gamma_lifts_the_shadows_and_leaves_the_highlights_byte_identical():
    """The move a colour knob cannot make: brightness where the mask is, and
    nowhere else. ``shadows`` at its defaults reaches exactly 1 and exactly 0 on
    this frame, so both halves of the claim are exact rather than directional."""
    frame = _frame()
    mask = masks_mod.mask_for("shadows", frame)
    lifted, kept = mask >= 1.0, mask <= 0.0
    assert lifted.any() and kept.any(), "the fixture needs a full-shadow region AND a clean one"
    out = dev._layered(frame, (SHADOW_LIFT,))
    assert _luma(out)[lifted].mean() > _luma(frame)[lifted].mean() + 1.0, "the shadows never lifted"
    assert np.array_equal(out[kept], frame[kept]), "a shadow lift reached the highlights"


def test_a_masked_contrast_curve_steepens_the_sky_and_nothing_below_it():
    """Contrast is a SPREAD, so the assertion is the region's own luma deviation —
    the sky's grows, and the ground the sky mask zeroes stays byte for byte."""
    frame = _horizon()
    mask = masks_mod.mask_for("sky", frame)
    sky, ground = mask >= 0.6, mask <= 0.0
    assert sky.any() and ground.any(), "the fixture needs a sky AND a clean ground"
    out = dev._layered(frame, (SKY_CONTRAST,))
    assert _luma(out)[sky].std() > _luma(frame)[sky].std(), "the sky's contrast never moved"
    assert np.array_equal(out[ground], frame[ground]), "the sky's curve reached the ground"


def test_a_tone_knob_cannot_ride_the_colour_round_trip_that_carries_the_rest():
    """The silent drop this seam closes, pinned as the REASON the tone knobs are set
    on the params record directly: a layer's colour chain IS the field names
    ``DevelopOptions`` and ``CorrectionParams`` share, and no tone knob is one — so
    routed through that trip a ``gamma`` would vanish without an error and render
    exactly nothing."""
    dropped = dev._look_params(
        cc.CorrectionParams(), dev.look_variant_options(SHADOW_LIFT, dev.DevelopOptions())
    )
    assert dropped.gamma == cc.CorrectionParams().gamma, "the round trip grew a tone field"
    frame = _frame()
    assert not np.array_equal(dev._layered(frame, (SHADOW_LIFT,)), frame), "the tone knob is inert"


def test_every_tone_knob_a_layer_may_carry_can_actually_render():
    """A layer knob that is no ``CorrectionParams`` field renders NOTHING, which is
    the bug above — ``_layered`` hands these to ``dataclasses.replace``, which
    refuses an unknown name, and each carries a range so a bad value is refused at
    the door instead of being clamped mid-render."""
    tone = config_module.LAYER_TONE_KNOBS
    assert tone <= config_module.LAYER_KNOBS, "a tone knob the door refuses is unreachable"
    assert tone <= {f.name for f in fields(cc.CorrectionParams)}
    assert tone <= set(config_module.LOOK_RANGES), "every tone knob needs its range"
    assert config_module.LOOK_RANGES["black_point"] == (cc.BLACK_POINT_MIN, cc.BLACK_POINT_MAX)
    assert config_module.LOOK_RANGES["gamma"][1] == cc.GAMMA_MAX


def test_a_layer_shapes_its_own_mask_and_the_defaults_render_as_today():
    """``linear``'s ``angle``/``span`` are per-MASK knobs the layer could not reach, so
    every recipe rode the registry defaults. Spelling those defaults out must be
    byte-identical — the regression guard on every layer that names no shape param."""
    frame = _frame()
    plain = dev._layered(frame, (GROUND,))
    tilted = dev._layered(frame, ({**GROUND, "angle": 0.0, "span": 0.25},))
    assert not np.array_equal(tilted, plain), "a layer's shape params never reached the mask"
    spelled = dev._layered(frame, ({**GROUND, "angle": 90.0, "span": 0.5},))
    assert np.array_equal(spelled, plain), "the registry defaults must render as today"


def test_a_seven_anchor_band_rides_a_masked_layer(tmp_path):
    """``check_layers`` routes a layer's lists through the same registry row the look
    door reads, and that row could say one exact length only — so a masked recipe could
    never reach the orange anchor (30°, second slot) ``_apply_hsl`` has rendered since
    it shipped. The drain here is orange-ONLY, the move six slots cannot make."""
    layer = {"mask": "radial", "hue_sat": [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]}
    assert PhotosConfig(looks={"citrus": {"layers": [layer]}}).looks["citrus"]["layers"] == [layer]
    frame = _frame()
    out = dev._layered(frame, (layer,))
    region = masks_mod.mask_for("radial", frame) > 0.0
    assert region.any() and not region.all(), "the fixture needs a region AND a complement"
    assert not np.array_equal(out[region], frame[region]), "the orange band moved nothing"


#: The two shipped looks written on the geometry masks — ``vignette_wash`` on two
#: ``radial`` layers, ``golden_ground`` on one ``linear`` — digested off the render as
#: it stood BEFORE either mask grew a position/size knob. A geometry default that no
#: longer reproduces the shipped falloff fails here rather than in a reviewer's eye.
GEOMETRY_LOOKS = {
    "vignette_wash": "09bbfd1ef3e82a69",  # pragma: allowlist secret
    "golden_ground": "a98909e24ea22e86",  # pragma: allowlist secret
}


@pytest.mark.parametrize("look,digest", sorted(GEOMETRY_LOOKS.items()))
def test_a_geometry_look_renders_byte_for_byte_what_it_shipped(look, digest):
    out = dev._layered(_frame(), tuple(_MASKED[look]["layers"]))
    assert hashlib.sha256(out.tobytes()).hexdigest()[:16] == digest, look


def test_a_layer_places_its_mask_where_the_subject_is_not_only_in_the_middle():
    """The gap this closes end to end: a recipe could brighten the centre or ramp in
    from an edge, and nothing else. A layer carries the geometry through ``_layered``,
    and naming none of it renders exactly the centred falloff every look already rode."""
    frame = _frame()
    centred = dev._layered(frame, (DRAIN,))
    placed = dev._layered(frame, ({**DRAIN, "cx": 0.2, "cy": 0.8, "size": 0.35},))
    assert not np.array_equal(placed, centred), "a layer's geometry never reached the mask"
    spelled = dev._layered(frame, ({**DRAIN, "cx": 0.5, "cy": 0.5, "size": 1.0, "aspect": 1.0},))
    assert np.array_equal(spelled, centred), "the registry defaults must render as today"


def test_the_mask_literal_mirrors_the_live_registry():
    """``config.py`` may not import the pipeline, so the legal mask names are a
    local literal there — as ``LOOK_KNOBS`` and ``BORDER_STYLES`` are. This is
    what keeps the two from drifting when the registry grows a name or a knob."""
    assert LAYER_MASKS == frozenset(masks_mod.MASKS)
    assert {k: frozenset(v) for k, v in LAYER_MASK_PARAMS.items()} == masks_mod.MASK_PARAMS
    assert config_module.LAYER_MASK_OPS == frozenset(masks_mod.MASK_OPS)


def test_the_layers_help_text_names_every_mask_an_operator_may_use(capsys):
    """The list in ``--help`` was hand-kept and had gone stale: ``glints`` and
    ``texture`` shipped weeks apart and neither reached the text, so a mask an operator
    could not discover was a mask that did not exist to them. It is derived from the
    same mirror ``check_layers`` validates against, asserted on the RENDERED help
    because that is the surface — argparse's own wrapping folded in."""
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["photos", "develop", "--help"])
    assert ", ".join(LAYER_MASK_PARAMS) in " ".join(capsys.readouterr().out.split())


def _operands(frame):
    return masks_mod.mask_for(PAIR[0], frame), masks_mod.mask_for(PAIR[1], frame)


def _combined(frame, op, **kw):
    return masks_mod.mask_for(PAIR, frame, op=op, **kw)


def test_intersect_is_at_most_each_operand_and_still_names_a_region():
    """The highlights INSIDE the centre — narrower than either operand, and a
    real region: an intersection that came out empty would be a look that renders
    nothing, which no assertion on the bound alone would catch."""
    frame = _frame()
    a, b = _operands(frame)
    both = _combined(frame, "intersect")
    assert (both <= a + 1e-6).all() and (both <= b + 1e-6).all(), "an intersection widened"
    assert (both > 0.0).any(), "the two masks never overlap — the fixture proves nothing"


def test_union_is_at_least_each_operand_everywhere():
    frame = _frame()
    a, b = _operands(frame)
    either = _combined(frame, "union")
    assert (either >= a - 1e-6).all() and (either >= b - 1e-6).all(), "a union narrowed"
    assert (either < 1.0).any(), "a union covering the whole frame proves nothing"


def test_subtract_takes_the_second_mask_out_of_the_first():
    frame = _frame()
    a, b = _operands(frame)
    assert np.allclose(_combined(frame, "subtract"), a * (1.0 - b), atol=1e-6)


def test_a_single_named_mask_still_renders_exactly_what_it_shipped():
    """Regression guard on the path every shipped look rides: the combinator seam
    must not touch a string ``mask`` — no op, no pair, the registry entry clipped."""
    frame = _frame()
    for name in masks_mod.MASKS:
        raw = np.clip(np.asarray(masks_mod.MASKS[name](frame), np.float32), 0.0, 1.0)
        assert np.array_equal(masks_mod.mask_for(name, frame), raw), name


def test_the_pair_combines_before_feather_and_before_invert():
    """The order the docstring pins, pinned here too: the RAW masks combine, THEN
    ``feather`` reshapes the one transition band that came out of them, THEN
    ``invert``. Feathering each operand first bends both bands before they meet —
    a measurably different region, so the two orders cannot drift silently."""
    frame = _frame()
    a, b = _operands(frame)
    soft = _combined(frame, "intersect", feather=2.0)
    assert np.allclose(soft, masks_mod._feathered(a * b, 2.0), atol=1e-6), "not combine-first"
    each_first = masks_mod._feathered(a, 2.0) * masks_mod._feathered(b, 2.0)
    assert float(np.abs(soft - each_first).max()) > 0.01, "the two orders must differ"
    assert np.allclose(_combined(frame, "intersect", feather=2.0, invert=True), 1.0 - soft, 1e-6)


def test_a_combined_layer_rides_the_intersection_through_the_render():
    """``_layered`` must thread the pair AND the operator, not just the first name."""
    frame = _frame()
    both = masks_mod.mask_for(COMBO["mask"], frame, op=COMBO["mask_op"])
    out = dev._layered(frame, (COMBO,))
    assert (both > 0.0).any() and (both <= 0.0).any(), "the fixture needs a region AND its outside"
    assert np.array_equal(out[both <= 0.0], frame[both <= 0.0]), "a combined layer leaked outside"
    assert not np.array_equal(out[both > 0.0], frame[both > 0.0]), "a combined layer moved nothing"


@pytest.mark.parametrize(
    "layer,named",
    [
        ({"mask": "depth", "vibrance": 0.4}, "depth"),
        ({"mask": "sky", "vignette": 0.4}, "vignette"),
        ({"mask": "sky", "vibrance": 0.4, "feather": 0.0}, "feather"),
        ({"mask": "sky", "vibrance": 0.4, "invert": "yes"}, "invert"),
        ({"mask": "sky"}, "knob"),
        ({"vibrance": 0.4}, "mask"),
        ("sky", "list"),
        ({"mask": "sky", "vibrance": 0.4, "angle": 30.0}, "angle"),
        ({"mask": "linear", "vibrance": 0.4, "span": "wide"}, "span"),
        ({"mask": ["sky", "radial"], "vibrance": 0.4}, "mask_op"),
        ({"mask": ["sky", "radial"], "mask_op": "blend", "vibrance": 0.4}, "blend"),
        ({"mask": ["sky"], "mask_op": "union", "vibrance": 0.4}, "two"),
        ({"mask": ["sky", "radial", "ground"], "mask_op": "union", "vibrance": 0.4}, "two"),
        ({"mask": ["sky", "depth"], "mask_op": "union", "vibrance": 0.4}, "depth"),
        ({"mask": "sky", "mask_op": "union", "vibrance": 0.4}, "two mask names"),
    ],
)
def test_config_load_refuses_a_malformed_layer_naming_the_look(layer, named):
    """Upstream gate: a bad layer costs a batch of decodes to find at render time.
    ``vignette`` is a real develop knob — spatial stages stay whole-frame; ``angle``
    is a real SHAPE param, but of ``linear``, so on ``sky`` it is an inert typo."""
    layers = layer if isinstance(layer, str) else [layer]
    with pytest.raises(ValidationError) as err:
        PhotosConfig(looks={"split_world": {"layers": layers}})
    assert "split_world" in str(err.value) and named in str(err.value)


@pytest.mark.parametrize(
    "knob,value",
    [("gamma", 3.0), ("black_point", 40), ("contrast_curve", -0.2), ("saturation_mult", 5.0)],
)
def test_config_load_refuses_an_out_of_range_tone_value_naming_the_look(knob, value):
    """Every bound is read off the applier that renders the knob, so past it the
    move either clamps (``black_point``) or renders exactly nothing (a negative
    ``contrast_curve``, below the applier's gate) — refused at LOAD either way."""
    with pytest.raises(ValidationError) as err:
        PhotosConfig(looks={"lifted": {"layers": [{"mask": "shadows", knob: value}]}})
    assert "lifted" in str(err.value) and f"{knob}={value!r} must be in" in str(err.value)


def test_a_look_carries_its_layers_through_the_cli_into_the_sidecar(tmp_path, monkeypatch):
    """End to end, the operator's path: a configured look renders its layers and
    the ``.grade.json`` records them, so a sheet audits back to the recipe. Twice,
    byte for byte — a masked layer must not cost determinism."""
    photo = _write_photo(tmp_path / "IMG_0003.jpg")
    look = {"styles": "natural", "layers": [GROUND]}
    monkeypatch.setattr(photos_handler, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(
        config_module,
        "get_config",
        lambda *a, **k: Config(photos=PhotosConfig(looks={"split_world": look})),
    )

    def _run(out, extra):
        argv = f"photos develop {photo} --out {out} --grade-sidecar {extra}".split()
        assert photos_handler.cmd_photos(_build_parser().parse_args(argv), None) == 0
        return sorted(out.glob("*.jpg"))[0]

    plain = _run(tmp_path / "plain", "--styles natural")
    looked = _run(tmp_path / "look", "--look split_world")
    again = _run(tmp_path / "again", "--look split_world")
    assert looked.read_bytes() != plain.read_bytes(), "the look's layers never reached the render"
    assert looked.read_bytes() == again.read_bytes(), "two renders must be byte-identical"
    sidecar = json.loads(looked.with_suffix(".grade.json").read_text())
    assert sidecar["options"]["layers"] == [GROUND], "the sidecar must record the resolved layers"
