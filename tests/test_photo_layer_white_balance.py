"""White balance is a LOCAL move too — a masked layer may carry its own.

Every other colour knob in the grade chain could already ride a mask: a layer can
lift shadows on the ground, pull saturation out of the sky, swing hue in one band.
White balance was the exception, and it is the one an operator reaches for first —
"warm the ground, leave the sky alone" is the single most common local correction
in a landscape frame, and the only way to express it was two renders and a manual
composite outside the pipeline.

Nothing in the renderer had to change for this, which is the point. ``_layered``
derives a layer's colour chain from the field names ``DevelopOptions`` and
``CorrectionParams`` SHARE, and all six white-balance fields were already on both;
``_grade`` already routes them through ``_apply_colorbalance``. The knobs were
absent from ``LAYER_COLOUR_KNOBS`` alone, and ``check_layers`` refuses any key that
set does not carry — so the capability was fully built and fenced off at the door
by one missing flag per row.

The additive/replace question that makes ``_wb_shifted`` subtle whole-frame does not
arise here: ``_layered`` lays each layer over a FRESH ``CorrectionParams()``, so the
base is zero and ``+ v`` and ``= v`` are the same number. A layer's warmth is its
own, not the measured cast plus its own.
"""

import numpy as np
import pytest

from gomoveshift.config import LAYER_COLOUR_KNOBS, PhotosConfig, check_layers
from gomoveshift.pipeline.color_correction import CorrectionParams
from gomoveshift.pipeline.photos.develop import _grade, _layered
from gomoveshift.pipeline.photos.masks import mask_for

WB_KNOBS = (
    "white_balance_r",
    "white_balance_b",
    "white_balance_g",
    "white_balance_r_high",
    "white_balance_b_high",
    "white_balance_g_high",
)


def _frame():
    """A flat mid-grey frame: any channel move is the layer's alone, and a mask
    that selects part of it leaves a measurable inside/outside split."""
    return np.full((32, 32, 3), 128, dtype=np.uint8)


@pytest.mark.parametrize("knob", WB_KNOBS)
def test_every_white_balance_axis_is_a_layer_colour_knob(knob):
    """All six axes or none: a partial flip is the half-wired state the knob
    registry exists to make unrepresentable. ``white_balance_g`` has no whole-frame
    producer, so leaving it out would have been easy and silently wrong."""
    assert knob in LAYER_COLOUR_KNOBS, f"{knob} cannot ride a mask"


@pytest.mark.parametrize("knob", WB_KNOBS)
def test_a_layer_carrying_one_axis_is_accepted_at_the_door(knob):
    """``check_layers`` refuses any key outside the layer knob set, so this is the
    gate that actually blocked the feature — assert it opened, per axis."""
    check_layers("--layers", [{"mask": "sky", knob: 0.02}])
    looks = {"warm_ground": {"layers": [{"mask": "ground", knob: 0.02}]}}
    assert PhotosConfig(looks=looks).looks == looks


def test_a_layer_white_balance_moves_pixels_and_only_under_its_mask():
    """The capability, not the registration: a masked warm shift must actually
    change red where the mask selects and leave the rest of the frame untouched.
    Without this the six flags could all be present and render nothing.

    ``linear`` is the mask here because it is GEOMETRIC — it splits any frame into
    selected and unselected by position. The content masks cannot serve: on a flat
    test frame ``sky``, ``shadows`` and ``highlights`` all meter to zero everywhere
    (there is no sky in flat grey), so a layer through one of them renders nothing
    and the test would pass against a no-op.
    """
    rgb = _frame()
    mask = mask_for("linear", rgb)
    inside, outside = mask >= mask.max() * 0.99, mask == 0.0
    assert inside.any() and outside.any(), "the mask did not split the frame"

    warmed = _layered(rgb, [{"mask": "linear", "white_balance_r": 0.05}])
    assert warmed.shape == rgb.shape

    moved = warmed.astype(int) - rgb.astype(int)
    assert moved[:, :, 0][inside].max() > 0, "a positive red shift did not warm the mask"
    assert not moved[outside].any(), "a masked white balance leaked outside its mask"


def test_an_empty_layer_list_is_still_byte_for_byte_the_shipped_path():
    """The blast radius. Adding six names to a frozenset must not disturb the
    default render, which reaches no mask at all."""
    rgb = _frame()
    assert np.array_equal(_layered(rgb, []), rgb)


def test_the_layer_white_balance_is_its_own_not_the_measured_cast_plus_its_own():
    """``_wb_shifted`` ADDS whole-frame, because a look grades over the MEASURED
    base and replacing there would discard the cast correction. Inside a layer the
    base record is fresh, so the operator's number is the number applied.

    Both paths run through ``_look_params``, so the difference is invisible in the
    source and only an equivalence pins it: through ``ground`` — which meters to
    1.0 everywhere on a flat frame, so the lerp selects the layer entirely — the
    layer must equal a DIRECT grade carrying that one field and nothing else. If a
    layer ever started from the run's measured params instead, this diverges by
    exactly the cast.
    """
    rgb = _frame()
    assert mask_for("ground", rgb).min() == 1.0, "ground no longer selects the whole flat frame"

    layered = _layered(rgb, [{"mask": "ground", "white_balance_r": 0.04}])
    direct = _grade(rgb, CorrectionParams(white_balance_r=0.04), 8)
    assert np.array_equal(layered, direct), "a layer's white balance is not the value written"
