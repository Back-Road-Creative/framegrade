"""One photo knob, one place to declare it — ``config.PHOTO_KNOBS`` and its tables.

Many knobs once all appended to the SAME three or four lines of ``config.py``
(``LOOK_KNOBS``, ``LOOK_RANGES``, ``LAYER_COLOUR_KNOBS``, ``LAYER_TONE_KNOBS``), so
every merge invalidated every other open branch — git cannot merge two edits to one
line — and each landing cost a rebase. Each knob now
declares itself on its OWN row and every surface is derived from it, the same
derive-don't-list move ``masks.MASK_PARAMS`` makes (it reads each mask's accepted
params off the signature, so adding a mask never conflicts with another mask change).

The snapshot below is the refactor's proof: the tables EXACTLY as they were
hand-listed before a registry existed, so "no knob gained, lost or changed a range"
is checkable rather than asserted. It passes before and after. Its four later
additions — ``duotone_mid``, ``halation_tint``, ``split_tone_midtone``, ``curve_l``
— are the list knobs the registry showed had NO shape at all, each given the bounds
its own flag was already enforcing at the CLI door.
"""

import ast
import inspect

import pytest
from pydantic import ValidationError

from gomoveshift import config as c

# The list knobs that shipped AFTER ``check_look_lists`` existed and slipped it:
# each carries a list its own FLAG gates (``_resolve_tint`` / ``_resolve_curve`` /
# ``_resolve_mono``) while its registry row declared no shape at all, so the look
# door skipped it. (knob, a wrong length, a value outside the flag's own bounds.)
UNGATED = [
    ("halation_tint", [1.0, 0.3], [1.0, 0.3, 1.8]),
    ("split_tone_midtone", [0.1, 0.2], [0.1, 0.2, 0.9]),
    ("curve_l", [0.0, 0.0, 0.5], [0.0, 0.0, 1.4, 0.9]),
    ("duotone_mid", [0.5, 0.5], [0.5, 0.5, 1.7]),
]

LOOK = {
    "border", "bw_mix", "chromatic_ab", "clarity", "curve_b", "curve_g", "curve_l", "curve_r",
    "dehaze", "duotone", "duotone_mid", "duotone_strength", "exposure_ev", "format", "glitch",
    "glow", "grain",
    "graduated_nd", "graduated_nd_angle",
    "grain_size", "halation", "halation_tint", "highlight_recovery", "holo_sheen", "hue_lum",
    "hue_sat", "hue_shift",
    "layers", "light_leak", "look_strength", "lut_auto", "lut_over", "lut_strength",
    "reference", "scanline", "shadow_recovery", "sharpen", "sky_darken", "sky_saturate",
    "split_tone_highlight",
    "split_tone_midtone",
    "split_tone_shadow", "style_strength", "styles", "vibrance", "vignette", "white_balance_g",
    "white_balance_kelvin", "white_balance_r", "white_balance_b", "white_balance_r_high",
    "white_balance_b_high",
    "white_balance_g_high", "white_balance_tint",
}  # fmt: skip
# ``lut`` left this table: it is a real develop flag, but an APPEND one the run owns
# (``run=True``), so a look naming it is refused by name — see LOOK_RUN_KNOBS.
# ``straighten``/``crop`` joined it: frame geometry is a run-level property every
# variant of one photo shares, never a look's. ``lens_correct`` joined for the same
# reason: it runs before geometry, so a look naming it would vary the EXIF-matched
# correction per variant of one photo.
RUN = {"crop", "lens_correct", "lut", "straighten"}
ALIASES = {
    "crop_rect": "crop",
    "mono_mix": "bw_mix",
    "reference_lab": "reference",
    "splittone_highlight": "split_tone_highlight",
    "splittone_midtone": "split_tone_midtone",
    "splittone_shadow": "split_tone_shadow",
    "straighten_angle": "straighten",
}
RANGES = {
    "black_point": (-10.0, 20.0), "chromatic_ab": (0.0, 1.0), "clarity": (-1.0, 1.0),
    "contrast_curve": (0.0, 0.15), "dehaze": (0.0, 1.0), "duotone_strength": (0.0, 1.0),
    "exposure_ev": (-3.0, 3.0),
    "gamma": (0.77, 1.30), "glitch": (0.0, 1.0), "glow": (0.0, 1.0), "grain": (0.0, 1.0),
    "graduated_nd": (0.0, 1.0), "graduated_nd_angle": (-90.0, 90.0),
    "grain_size": (0.25, 4.0), "halation": (0.0, 1.0), "highlight_recovery": (0.0, 20.0),
    "holo_sheen": (0.0, 1.0),
    "light_leak": (0.0, 1.0), "look_strength": (0.0, 1.0), "lut_strength": (0.0, 1.0),
    "saturation_mult": (0.0, 2.0), "scanline": (0.0, 1.0), "shadow_recovery": (0.0, 20.0),
    "sharpen": (0.0, 2.0),
    "sky_darken": (0.0, 1.0), "sky_saturate": (0.0, 1.0), "straighten": (-45.0, 45.0),
    "vibrance": (0.0, 2.0), "vignette": (0.0, 1.0),
    "white_balance_g": (-0.5, 0.5), "white_balance_g_high": (-0.5, 0.5),
    "white_balance_kelvin": (-2000.0, 2000.0),
    "white_balance_r": (-0.1, 0.1), "white_balance_b": (-0.1, 0.1),
    "white_balance_r_high": (-0.1, 0.1), "white_balance_b_high": (-0.1, 0.1),
    "white_balance_tint": (-0.5, 0.5),
}  # fmt: skip
LIST_RANGES = {
    "bw_mix": ((3,), -2.0, 3.0), "duotone": ((6,), 0.0, 1.0), "duotone_mid": ((3,), 0.0, 1.0),
    "halation_tint": ((3,), 0.0, 1.0), "hue_lum": ((6, 7), 0.0, 5.0),
    "hue_sat": ((6, 7), 0.0, 5.0), "hue_shift": ((6, 7), -180.0, 180.0),
    "split_tone_highlight": ((3,), -0.5, 0.5), "split_tone_midtone": ((3,), -0.5, 0.5),
    "split_tone_shadow": ((3,), -0.5, 0.5),
}  # fmt: skip
CURVES = {"curve_b", "curve_g", "curve_l", "curve_r"}
# The six ``white_balance_*`` axes joined this set deliberately — a masked layer may
# carry its own white balance. They are the ONE addition since the snapshot was taken:
# the renderer already rendered them (they are fields both knob records share, so
# ``_look_params`` derives them), and only their absence here kept ``check_layers`` from
# accepting them. See tests/test_photo_layer_white_balance.py.
LAYER_COLOUR = {
    "bw_mix", "curve_b", "curve_g", "curve_l", "curve_r", "duotone", "duotone_mid",
    "duotone_strength", "hue_lum", "hue_sat", "hue_shift", "split_tone_highlight",
    "split_tone_midtone", "split_tone_shadow", "vibrance",
    "white_balance_b", "white_balance_b_high", "white_balance_g", "white_balance_g_high",
    "white_balance_r", "white_balance_r_high",
}  # fmt: skip
LAYER_TONE = {"black_point", "contrast_curve", "gamma", "saturation_mult"}


def test_the_derived_tables_are_exactly_the_hand_listed_ones():
    """The behaviour-preserving proof. Every one of these was a hand-kept literal;
    each is now a comprehension over ``PHOTO_KNOBS``, and must still hold the same
    names, the same bounds and the same container types down to the last knob."""
    assert set(c.LOOK_KNOBS) == LOOK
    assert set(c.LOOK_RUN_KNOBS) == RUN
    assert c.LOOK_ALIASES == ALIASES
    assert c.LOOK_RANGES == RANGES
    assert c.LOOK_LIST_RANGES == LIST_RANGES
    assert set(c.LOOK_CURVE_KNOBS) == CURVES
    assert set(c.LAYER_COLOUR_KNOBS) == LAYER_COLOUR
    assert set(c.LAYER_TONE_KNOBS) == LAYER_TONE
    assert set(c.LAYER_KNOBS) == LAYER_COLOUR | LAYER_TONE
    for table in (c.LOOK_KNOBS, c.LOOK_CURVE_KNOBS, c.LAYER_COLOUR_KNOBS, c.LAYER_TONE_KNOBS):
        assert isinstance(table, frozenset), "a validator does membership tests on these"
    assert isinstance(c.LAYER_KNOBS, frozenset)


def test_one_declared_knob_reaches_every_surface_it_claims():
    """The half-wiring guard: the next knob is a row, and a row lands in every table
    it claims by construction. A knob that claimed ``layer="tone"`` and never reached
    ``LAYER_TONE_KNOBS`` — the shape of a past bug, where four appends had to agree
    — cannot be expressed."""
    assert hasattr(c, "PHOTO_KNOBS"), "there is no one place to declare a knob"
    declared: set[str] = set()
    for knob in c.PHOTO_KNOBS:
        assert knob.name not in declared, f"{knob.name} is declared twice"
        declared.add(knob.name)
        assert not (knob.span and knob.items), f"{knob.name}: a scalar or a list, not both"
        assert knob.layer in ("", "colour", "tone"), f"{knob.name}: unknown layer surface"
        assert (knob.name in c.LOOK_KNOBS) is knob.look
        assert c.LOOK_RANGES.get(knob.name) == knob.span
        assert (knob.name in c.LOOK_LIST_RANGES) is (knob.items is not None)
        if knob.items is not None:
            assert c.LOOK_LIST_RANGES[knob.name] == (knob.lengths, *knob.items[1:])
        assert (knob.name in c.LOOK_CURVE_KNOBS) is knob.curve
        assert (knob.name in c.LOOK_RUN_KNOBS) is knob.run
        assert not (knob.run and knob.look), f"{knob.name}: the run owns it, no look may"
        assert (knob.name in c.LAYER_COLOUR_KNOBS) is (knob.layer == "colour")
        assert (knob.name in c.LAYER_TONE_KNOBS) is (knob.layer == "tone")
        assert (knob.name in c.LAYER_KNOBS) is bool(knob.layer)
        assert c.LOOK_ALIASES.get(knob.alt) == (knob.name if knob.alt else None)
    for table in (
        c.LOOK_KNOBS,
        c.LOOK_RANGES,
        c.LOOK_LIST_RANGES,
        c.LOOK_CURVE_KNOBS,
        c.LAYER_KNOBS,
        set(c.LOOK_ALIASES.values()),
    ):
        assert set(table) <= declared, "a surface carries a knob the registry never declared"


@pytest.mark.parametrize("knob, wrong_length, out_of_range", UNGATED)
def test_a_looks_malformed_list_is_refused_at_config_load(knob, wrong_length, out_of_range):
    """Both malformed shapes are refused where the operator can still fix them —
    at LOAD, naming the look — instead of once per photo, mid-batch. Neither value
    is newly illegal: each is exactly what that knob's own flag already refuses."""
    for bad in (wrong_length, out_of_range):
        with pytest.raises(ValidationError) as excinfo:
            c.PhotosConfig(looks={"neon": {knob: bad}})
        message = str(excinfo.value)
        assert "look 'neon'" in message, f"{knob}: the refusal must name the look"
        assert knob in message, f"{knob}: the refusal must name the knob"


def test_the_look_door_refuses_before_the_per_photo_one_ever_runs():
    """The bug class the gate closes: ``CorrectionParams`` refuses these too, but
    that raise lands inside the render — once per photo, after the decodes are
    paid. Refused at load, a malformed value never reaches the record at all."""
    from gomoveshift.pipeline.color_correction import CorrectionParams  # test-time: no cycle

    for field, bad in (("splittone_midtone", (0.1, 0.2)), ("curve_l", (0.0, 0.0, 0.5))):
        with pytest.raises(ValueError):
            CorrectionParams(**{field: bad})
        with pytest.raises(ValidationError):
            c.PhotosConfig(looks={"neon": {c.LOOK_ALIASES.get(field, field): list(bad)}})


def _cli_list_gates():
    """Every list knob's shape as its ``photos develop`` FLAG enforces it, read off the
    CLI door itself — nothing hand-listed here. The call site names the resolver and
    hands over its literal bounds, so {flag → resolver + args} is derived rather than
    transcribed; the caller then PROBES the resolver for the lengths it takes."""
    from gomoveshift.cli.handlers import photos as door

    gates = {}
    for call in ast.walk(ast.parse(inspect.getsource(door))):
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)):
            continue
        if not call.func.id.startswith("_resolve_"):
            continue
        lits = [
            ast.literal_eval(a) for a in call.args if isinstance(a, (ast.Constant, ast.UnaryOp))
        ]
        flag = next((v for v in lits if isinstance(v, str) and v.startswith("--")), None)
        if flag is None:  # ``_resolve_luts`` — a file path, not a list of numbers
            continue
        nums = [v for v in lits if isinstance(v, (int, float))]
        gates[flag.lstrip("-").replace("-", "_")] = (getattr(door, call.func.id), flag, nums)
    return gates


def _accepted_lengths(resolve, flag, nums):
    """The lengths that flag's own resolver ACCEPTS, measured: hand it a list of
    in-range values at each length and see which come back. Empty (0) is OFF on every
    door and is not a declared shape."""
    mid = (float(nums[-2]) + float(nums[-1])) / 2
    ok = []
    for n in range(1, 10):
        try:
            resolve([mid] * n, [], flag, *nums)
        except ValueError:
            continue
        ok.append(n)
    return tuple(ok)


def test_each_list_knobs_shape_is_derived_from_the_flag_that_gates_it():
    """The guard, and the reason it exists: ``items`` could name ONE exact length
    while ``_resolve_hue_band`` had accepted six OR seven since orange shipped, so the
    registry silently refused through every look door a band the flag took happily.
    Both numbers are read off the door here — the flag's resolver is probed, its bounds
    come from the call site — so no length in this file is a copy anyone can forget to
    update. Same derive-don't-list move as ``masks.MASK_PARAMS`` (each mask's params
    read off its signature) and the ``_HASHED_SOURCES`` guard."""
    gates = _cli_list_gates()
    rows = {k.name: k for k in c.PHOTO_KNOBS}
    assert set(gates) == {k.name for k in c.PHOTO_KNOBS if k.items or k.curve}, (
        "a list knob has no CLI gate, or a gated flag is no declared list knob"
    )
    for name, (resolve, flag, nums) in sorted(gates.items()):
        knob = rows[name]
        if not nums:  # ``_resolve_curve``: shaped, not sized — any even length >= 4
            assert knob.curve and knob.items is None, f"{name}: the flag gates a free curve"
            continue
        assert knob.lengths == _accepted_lengths(resolve, flag, nums), (
            f"{name}: the look door and {flag} disagree on how many values it takes"
        )
        assert knob.items[1:] == tuple(nums[-2:]), f"{name}: {flag}'s bounds are not the row's"


def test_every_list_valued_knob_declares_its_shape():
    """The guard, and the reason it exists: a row that declares no
    ``items`` and no ``curve`` is a knob ``check_look_lists`` silently SKIPS, which
    is how each of them shipped past a gate that already existed. A knob's
    ``PhotosConfig`` default already says it is a list of numbers, so the omission
    is detectable here — the next list knob cannot slip the door the same way."""
    fields = c.PhotosConfig.model_fields
    for knob in c.PHOTO_KNOBS:
        field = fields.get(knob.alt) or fields.get(knob.name)
        if field is None or field.annotation != list[float]:
            continue
        assert knob.items is not None or knob.curve, (
            f"{knob.name} is a list of numbers with no shape declared: the look door "
            "skips it and a malformed value survives to the render"
        )
